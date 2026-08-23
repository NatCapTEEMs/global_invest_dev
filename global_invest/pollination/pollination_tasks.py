"""Dynamic pollination ES-shock task (V_F, OSD).

Runs the pollination sufficiency and value calculation on our SEALS 300 m maps at EACH SEALS
anchor year (seals_years), then piecewise-linearly interpolates the shock to annual values. Writes
pollination_interpolated.csv -- the file build_combined_afeall_cc_es reads -- into the shared ES-shock
directory (p.es_shock_dir). Grafted by consumers via add_pollination_tasks (dispatch on p.dynamic_es).
"""
from __future__ import annotations
import os
import gc
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import rasterio
from joblib import Parallel, delayed
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window
from scipy.ndimage import convolve
from tqdm import tqdm
import glob
import numpy as np
import pandas as pd
import hazelbean as hb
from global_invest import utilities
from global_invest.pollination import pollination_functions as pf



# ---------------------------------------------------------------------------------------------
# Vendored from crop_benefits: the sufficiency and value raster steps. They stream global
# rasters window by window, so they open and write files and belong here rather than beside
# the arithmetic.
# ---------------------------------------------------------------------------------------------

def read_raster(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read a single-band GeoTIFF.

    Returns
    -------
    data : np.ndarray
        2-D float32 array.
    meta : dict
        Rasterio profile (used to write matching outputs).
    """
    path = Path(path)
    logger.debug("Reading raster: %s", path)

    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        meta = src.meta.copy()

    return data, meta


def save_csv(df: pd.DataFrame, path: Path | str) -> Path:
    """Save DataFrame as CSV and log the result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %d rows -> %s", len(df), path)
    return path


def _process_tile(
    lulc_path: Path | str,
    window: Window,
    height: int,
    width: int,
    bounds_top: float,
    transform_e: float,
    pixel_lat_deg: float,
    pixel_lon_deg: float,
    ag_classes: set[int],
    nat_classes: set[int],
    fill_non_ag: float | None = None,
    stable_ag_lulc_path: str | None = None,
) -> Tuple[Window, np.ndarray, int] | None:
    """Worker function to process a single tile.

    Parameters
    ----------
    fill_non_ag:
        When ``None`` (default) non-agricultural pixels are written as NaN
        (excluded from the 5 km average).  When set to a float (typically
        ``0.0``) every valid (non-nodata) non-agricultural pixel receives that
        value, so the 5 km average is weighted by the fraction of agricultural
        area in the cell — matching the methodology used in the original
        InVEST/GLOBIO pollination repos.
    stable_ag_lulc_path:
        When provided, the agricultural mask is restricted to pixels that are
        agricultural in *both* ``lulc_path`` and this path (intersection).
        Nature counts still come from ``lulc_path`` only.  Non-stable-ag pixels
        receive NaN regardless of ``fill_non_ag``.  Used by
        :func:`run_pollination_sufficiency_300m_stable_ag`.
    """
    h = window.height
    w = window.width

    # Open LULC locally in the worker process
    with rasterio.open(lulc_path) as src:
        lulc_nodata = src.nodata
        # Read core LULC
        lulc_core = src.read(1, window=window)

        # Identify AG pixels — optionally intersected with other-period LULC
        ag_mask = np.isin(lulc_core, list(ag_classes))
        if stable_ag_lulc_path is not None:
            with rasterio.open(stable_ag_lulc_path) as src2:
                lulc_other = src2.read(1, window=window)
            ag_mask = ag_mask & np.isin(lulc_other, list(ag_classes))

        if not ag_mask.any():
            if fill_non_ag is None:
                return None
            # fill_non_ag mode: return zeros for all valid (non-nodata) pixels
            # so they are included in the 5 km average and pull it toward zero.
            out_core = np.full((h, w), np.nan, dtype=np.float32)
            if lulc_nodata is not None:
                valid = lulc_core != int(lulc_nodata)
            else:
                valid = np.ones((h, w), dtype=bool)
            out_core[valid] = fill_non_ag
            return window, out_core, 0

        # Calculate kernel for this tile latitude
        # Window off is row_off, col_off
        r0 = int(window.row_off)
        
        lat_mid = _tile_mid_lat(bounds_top, transform_e, r0, h)
        ry, rx = _compute_radii_pixels(lat_mid, pixel_lat_deg, pixel_lon_deg)
        kernel = _make_elliptical_kernel(ry, rx)
        kernel_sum = int(kernel.sum())
        
        # Expand window for convolution context
        nat_win = _window_expand_aniso(window, pad_y=ry, pad_x=rx, height=height, width=width)
        lulc_nat = src.read(1, window=nat_win)
        nat_mask = np.isin(lulc_nat, list(nat_classes)).astype(np.uint8)
        
        # Convolve
        counts = convolve(nat_mask, weights=kernel, mode="constant", cval=0.0).astype(np.int32)
        
        # Extract core
        nat_row0 = int(nat_win.row_off)
        nat_col0 = int(nat_win.col_off)
        r_off = int(r0 - nat_row0)
        c0 = int(window.col_off)
        c_off = int(c0 - nat_col0)
        
        counts_core = counts[r_off:r_off + h, c_off:c_off + w]
        
        # Calc sufficiency
        denom = counts_core[ag_mask]
        valid = denom > 0
        
        frac = np.zeros_like(denom, dtype=np.float32)
        if valid.any():
            frac[valid] = denom[valid].astype(np.float32) / float(kernel_sum)
            
        suff_vals = np.minimum(frac / np.float32(_THRESHOLD), 1.0)
        
        # Write back
        out_core = np.full((h, w), np.nan, dtype=np.float32)

        if fill_non_ag is not None:
            # Fill valid non-ag pixels so they are included in the 5 km average.
            if lulc_nodata is not None:
                valid_pixels = lulc_core != int(lulc_nodata)
            else:
                valid_pixels = np.ones((h, w), dtype=bool)
            out_core[valid_pixels & ~ag_mask] = fill_non_ag

        out_core_ag = out_core[ag_mask]
        out_core_ag[:] = suff_vals
        out_core[ag_mask] = out_core_ag

        ag_count = int(ag_mask.sum())

        return window, out_core, ag_count


def run_pollination_sufficiency_300m(
    cfg: pf.SufficiencySettings,
    lulc_path: Path | None = None,
    scenario: str = "2020",
    fill_non_ag: float | None = None,
    stable_ag_lulc_path: Path | None = None,
) -> Path:
    """Compute 300 m pollination sufficiency from ESA-CCI LULC or custom LULC map.

    For every agricultural pixel, counts natural-habitat neighbours within
    a latitude-adjusted elliptical kernel and normalises against _THRESHOLD.
    Returns the path to the output GeoTIFF.

    Parameters
    ----------
    fill_non_ag:
        Passed through to :func:`_process_tile`.  ``None`` (default) leaves
        non-agricultural pixels as NaN so the 5 km average is computed only
        over agricultural sub-pixels.  Pass ``0.0`` to use the original-repo
        approach where non-agricultural pixels contribute zero to the average,
        scaling the 5 km sufficiency by the agricultural area fraction.
    """
    lulc_path = lulc_path or cfg.lulc_path
    if not lulc_path or str(lulc_path) == ".":
        raise ValueError("lulc_esa path not defined in config (and no override provided)")
        
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pollination_sufficiency_{scenario}_300m.tif"
    
    logger.info("Computing 300m Pollination Sufficiency...")
    logger.info("Input LULC: %s", lulc_path)
    logger.info("Output: %s", out_path)
    
    # 0. Interpret LULC mapping
    if scenario == "2020":
        # ESA-CCI default classes
        ag_classes = _AG_CLASSES
        nat_classes = _NAT_CLASSES
        logger.info("Using default ESA-CCI LULC classes.")
    else:
        # SEALS future scenario classes
        # 1 = urban, 2 = cropland, 3 = grassland, 4 = forest, 5 = non-forest natural
        # 6 = water, 7 = barren land
        ag_classes = _SEALS_AG_CLASSES
        nat_classes = _SEALS_NAT_CLASSES
        logger.info("Using SEALS LULC classes.")
    
    if out_path.exists():
        logger.info("Output already exists, skipping computation.")
        return out_path

    # 1. Open / Profile
    with rasterio.open(lulc_path) as src:
        profile = src.profile.copy()
        transform = src.transform
        bounds_top = src.bounds.top
        height, width = src.shape
        pixel_lat_deg = abs(transform.e)
        pixel_lon_deg = abs(transform.a)
        
        # Prepare output profile
        out_profile = profile.copy()
        out_profile.update(
            dtype="float32",
            nodata=np.nan,
            **pf.get_compression_profile(cfg, "continuous")
        )
        
    # 2. Init output with NaNs
    logger.info("Initializing output with NaNs...")
    with rasterio.open(out_path, "w", **out_profile) as dst:
        pass

    # 3. Compute Parallel
    tile_size = cfg.tile_size
    n_tiles_row = math.ceil(height / tile_size)
    n_tiles_col = math.ceil(width / tile_size)
    
    # Prepare Windows
    windows = []
    for tr in range(n_tiles_row):
        for tc in range(n_tiles_col):
            r0 = tr * tile_size
            c0 = tc * tile_size
            h = min(tile_size, height - r0)
            w = min(tile_size, width - c0)
            windows.append(Window(c0, r0, w, h))
            
    ag_pixels = 0
    
    # Generator for parallel results
    # n_jobs=-1 uses all cores. 
    # use return_as="generator" to yield results as they complete
    
    logger.info("Processing %d tiles with parallel execution...", len(windows))
    
    # Scan all agricultural pixels to measure the proportion of natural habitat within foraging range, creating a 300m sufficiency index grid
    _stable_ag_str = str(stable_ag_lulc_path) if stable_ag_lulc_path is not None else None
    results = Parallel(n_jobs=cfg.n_workers, return_as="generator")(
        delayed(_process_tile)(
            str(lulc_path),
            w,
            height,
            width,
            bounds_top,
            transform.e,
            pixel_lat_deg,
            pixel_lon_deg,
            ag_classes,
            nat_classes,
            fill_non_ag,
            _stable_ag_str,
        ) for w in windows
    )
    
    with rasterio.open(out_path, "w", **out_profile) as dst:
        with tqdm(total=len(windows), desc="Processing Tiles") as pbar:
            for res in results:
                if res is not None:
                    win, data, count = res
                    dst.write(data, 1, window=win)
                    ag_pixels += count

                
                pbar.update(1)
                    
    logger.info("Completed 300m sufficiency. Ag pixels processed: %d", ag_pixels)
    gc.collect()
    return out_path


def run_pollination_sufficiency_5km(cfg: pf.SufficiencySettings, input_300m: Path | None = None, scenario: str = "2020") -> Path:
    """Resample 300 m sufficiency to 5 km using average resampling.

    Matches the grid of "country_raster" (5 km template).
    Returns the path to the 5 km output GeoTIFF.
    """
    if input_300m is None:
        input_300m = cfg.output_dir / f"pollination_sufficiency_{scenario}_300m.tif"
        
    out_dir = cfg.output_dir
    out_path = out_dir / f"pollination_sufficiency_{scenario}_5km.tif"
    
    # Template target: We need a 5km grid.
    # The config has 'country_raster' which should be 5km.
    template_path = cfg.country_raster_path
    
    logger.info("Resampling sufficiency to 5km...")
    logger.info("Input: %s", input_300m)
    logger.info("Template: %s", template_path)
    logger.info("Output: %s", out_path)
    
    if out_path.exists():
        logger.info("Output already exists, skipping resampling.")
        return out_path

    if not input_300m.exists():
        raise FileNotFoundError(f"300m sufficiency raster not found at {input_300m}")
        
    with rasterio.open(template_path) as tgt:
        tgt_profile = tgt.profile.copy()
        tgt_transform = tgt.transform
        tgt_crs = tgt.crs
        tgt_shape = (tgt.height, tgt.width)
        
    tgt_profile.update(
        dtype="float32",
        nodata=np.nan,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        predictor=2,
        bigtiff="if_safer"
    )

    with rasterio.open(input_300m) as src:
        # Downsample the highly detailed 300m sufficiency map to a generalized 5km grid utilizing averaging to match crop production scales
        # Reproject using Average resampling (ignoring nodata)
        # This gives average sufficiency of ARABLE land in the cell
        data_5km = np.empty(tgt_shape, dtype=np.float32)
        
        reproject(
            source=rasterio.band(src, 1),
            destination=data_5km,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=tgt_transform,
            dst_crs=tgt_crs,
            dst_nodata=np.nan,
            resampling=Resampling.average
        )
        
    with rasterio.open(out_path, "w", **tgt_profile) as dst:
        dst.write(data_5km, 1)
        
    logger.info("Created 5km sufficiency raster.")
    return out_path


def run_pollination_sufficiency_300m_stable_ag(
    cfg: pf.SufficiencySettings,
    lulc_path: Path,
    other_lulc_path: Path,
    scenario: str,
) -> Path:
    """Compute 300 m sufficiency restricted to pixels agricultural in both periods.

    A thin wrapper around :func:`run_pollination_sufficiency_300m` that
    restricts the agricultural mask to the *intersection* of the two LULC
    rasters: a pixel is treated as agricultural only if it carries an
    agricultural code in **both** ``lulc_path`` (the primary LULC used for
    nature counting) and ``other_lulc_path`` (the other-period LULC).

    - Nature neighbours are counted from ``lulc_path`` only (correct: each
      period's sufficiency reflects that period's surrounding landscape).
    - Non-stable-ag pixels receive NaN and are excluded from the 5 km average.

    This closes the sub-pixel gap in the PNAS replication: by restricting
    both the baseline and scenario sufficiency rasters to the same stable-ag
    mask before averaging to 5 km, the 5 km diff captures only changes in
    sufficiency on continuously farmed land — exactly as the original
    ``pollination_shock()`` function did at 300 m.

    Parameters
    ----------
    lulc_path:
        Primary LULC raster.  Used for both ag classification and nature
        counting.
    other_lulc_path:
        The other period's LULC raster.  Only its ag mask is used (nature
        is NOT counted from here).
    scenario:
        Output label; determines the output filename.

    Returns
    -------
    Path
        Path to ``pollination_sufficiency_{scenario}_300m.tif``.
    """
    return run_pollination_sufficiency_300m(
        cfg,
        lulc_path=lulc_path,
        scenario=scenario,
        fill_non_ag=None,           # NaN for non-stable-ag (exclude from average)
        stable_ag_lulc_path=other_lulc_path,
    )


def run_pollination_valuation_5km(cfg: pf.SufficiencySettings, scenario: str = "2020", target_year: int = 2024) -> Path:
    """Compute 5 km pollination-value rasters (Value × Sufficiency).

    Uses the pre-calculated global pollination value raster (from build_pollination_value.py)
    and multiplies it by the 5 km sufficiency index.
    """
    suff_path = cfg.output_dir / f"pollination_sufficiency_{scenario}_5km.tif"

    # Input: Pre-calculated global pollination value
    poll_value_path = cfg.value_raster_dir / f"poll_value_global_{target_year}usd.tif"

    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_total = out_dir / f"value_pollination_sufficiency_{scenario}_5km.tif"

    logger.info("Computing 5km Pollination Value (weighted by sufficiency, %d USD)...", target_year)
    logger.info("  Pollination Value Input: %s", poll_value_path)
    logger.info("  Sufficiency Input      : %s", suff_path)
    
    if out_total.exists():
        logger.info("Output already exists, skipping valuation.")
        return out_total

    if not poll_value_path.exists():
        raise FileNotFoundError(f"Global pollination value raster missing: {poll_value_path}")
    if not suff_path.exists():
        raise FileNotFoundError(f"Sufficiency raster missing: {suff_path}")

    # Load both rasters
    with rasterio.open(poll_value_path) as src_val, rasterio.open(suff_path) as src_suff:
        # Safety checks
        if src_val.shape != src_suff.shape:
             raise RuntimeError(f"Shape mismatch: Val {src_val.shape} vs Suff {src_suff.shape}")
        
        # Read and mask
        val_data = src_val.read(1).astype(np.float32)
        suff_data = src_suff.read(1).astype(np.float32)
        
        # Handle nodata
        val_nodata = src_val.nodata
        suff_nodata = src_suff.nodata
        
        if val_nodata is not None:
             val_data[val_data == val_nodata] = np.nan
        if suff_nodata is not None:
             suff_data[suff_data == suff_nodata] = np.nan
             
        # Calculate: Intersect baseline pollination values with the corresponding sufficiency index (Value * Sufficiency)
        
        result = val_data * suff_data
        
        # Output profile from source
        profile = src_val.profile.copy()
        profile.update(dtype="float32", nodata=-9999.0, compress="lzw")
        
        # Fill NaN with nodata for output
        out_data = np.nan_to_num(result, nan=-9999.0)
        
        with rasterio.open(out_total, "w", **profile) as dst:
             dst.write(out_data, 1)
             
    logger.info("Saved sufficiency-weighted value: %s", out_total)
    return out_total


def redistribution_value_to_300m(cfg: pf.SufficiencySettings, scenario: str = "2020") -> Path:
    """Redistribute 5 km pollination value back to 300 m pixels.

    Each 300 m pixel receives a share of its parent 5 km cell's value
    proportional to its sufficiency weight (suff_i / Σ suff).
    """
    large_path = cfg.output_dir / f"value_pollination_sufficiency_{scenario}_5km.tif"
    fine_path = cfg.output_dir / f"pollination_sufficiency_{scenario}_300m.tif"
    out_path = cfg.output_dir / f"value_pollination_sufficiency_{scenario}_300m.tif"

    logger.info("Redistributing value to 300m...")

    if out_path.exists():
        logger.info("Output already exists, skipping redistribution.")
        return out_path

    if not large_path.exists():
        raise FileNotFoundError(f"5km value raster missing: {large_path}")
    
    with rasterio.open(large_path) as large, rasterio.open(fine_path) as fine:
        large_vals = large.read(1).astype(np.float64)
        if large.nodata is not None:
             large_vals = np.where(large_vals == large.nodata, 0, large_vals)
        else:
             large_vals = np.nan_to_num(large_vals, nan=0.0)

        # Area ratio: large pixel area / fine pixel area (in degree² units; cos(lat) cancels).
        # Multiplying by this converts the redistributed fraction back to USD/km² so that
        # summing val_300m * area_300m recovers the same total as summing val_5km * area_5km.
        area_ratio = (abs(large.transform.a) * abs(large.transform.e)) / (
            abs(fine.transform.a) * abs(fine.transform.e)
        )
        logger.info("  Area ratio (5km/300m pixel): %.2f", area_ratio)

        profile = fine.profile.copy()
        profile.update(dtype="float32", nodata=0.0, **pf.get_compression_profile(cfg, "continuous"))

        block_size = cfg.tile_size

        with rasterio.open(out_path, "w", **profile) as dst:
             n_blocks = math.ceil(fine.height / block_size) * math.ceil(fine.width / block_size)

             # Downscale the 5km aggregated pollination value back into smaller 300m sections proportionally based on local weights
             with tqdm(total=n_blocks, desc="300m Redistribution") as pbar:
                for br in range(0, fine.height, block_size):
                    bh = min(block_size, fine.height - br)
                    for bc in range(0, fine.width, block_size):
                        bw = min(block_size, fine.width - bc)
                        window = Window(bc, br, bw, bh)

                        suff = fine.read(1, window=window).astype(np.float64)
                        mask = suff > 0
                        weights = suff[mask]

                        if weights.size == 0:
                            dst.write(np.zeros((bh, bw), dtype=np.float32), 1, window=window)
                            pbar.update(1)
                            continue

                        # Centroids
                        fine_r, fine_c = np.where(mask)
                        # Transform to coords
                        xs, ys = rasterio.transform.xy(fine.transform, br + fine_r, bc + fine_c, offset='center')
                        xs = np.array(xs)
                        ys = np.array(ys)

                        # Parent indices
                        # row = (y - top) / dy, col = (x - left) / dx
                        parent_c = np.floor((xs - large.transform.c) / large.transform.a).astype(int)
                        # large.transform.e is usually negative
                        parent_r = np.floor((ys - large.transform.f) / large.transform.e).astype(int)

                        parent_r = np.clip(parent_r, 0, large.height - 1)
                        parent_c = np.clip(parent_c, 0, large.width - 1)

                        # Vectorized attribution
                        # We need to group by parent pixel to calculate sum_weights per parent
                        flat_idx = parent_r * large.width + parent_c

                        unique_idx, input_idx = np.unique(flat_idx, return_inverse=True)
                        parent_vals = large_vals.flatten()[unique_idx]

                        sum_weights = np.zeros(unique_idx.size, dtype=np.float64)
                        np.add.at(sum_weights, input_idx, weights)

                        # Redistributions
                        # val_i = (weight_i / sum_weights_parent) * val_parent * area_ratio
                        # area_ratio converts the fractional density back to USD/km² so that
                        # Σ(val_300m * area_300m) == Σ(val_5km * area_5km).
                        # Avoid div by zero
                        divisor = sum_weights[input_idx]
                        valid_div = divisor > 0

                        res = np.zeros_like(weights)
                        if valid_div.any():
                             res[valid_div] = (weights[valid_div] / divisor[valid_div]) * parent_vals[input_idx][valid_div] * area_ratio

                        out_block = np.zeros((bh, bw), dtype=np.float32)
                        out_block[mask] = res.astype(np.float32)

                        dst.write(out_block, 1, window=window)
                        pbar.update(1)
                        
    return out_path


def mask_protected_areas_300m(cfg: pf.SufficiencySettings, scenario: str = "2020") -> Path:
    """
    Mask pollination value inside protected areas (PA == 1 -> value = 0).

    Reads the 300m pollination-value raster and a binary PA raster,
    zeroes all pixels that fall inside a protected area, and writes
    a masked raster together with a summary CSV.

    Returns
    -------
    Path
        Path to the masked output raster.
    """
    value_path = (
        cfg.output_dir
        / f"value_pollination_sufficiency_{scenario}_300m.tif"
    )
    pa_path = cfg.pa_raster_300m_path
    out_raster = (
        cfg.output_dir
        / f"value_pollination_sufficiency_{scenario}_300m_no_agri_in_PA.tif"
    )
    out_csv = (
        cfg.output_dir
        / f"summary_300m_no_agri_in_PA_{scenario}.csv"
    )

    logger.info("Masking PA pixels from 300m pollination value...")
    logger.info("Value input : %s", value_path)
    logger.info("PA raster   : %s", pa_path)

    if out_raster.exists():
        logger.info("Output already exists, skipping PA masking.")
        return out_raster

    if not value_path.exists():
        raise FileNotFoundError(f"300m value raster missing: {value_path}")
    if not pa_path.exists():
        raise FileNotFoundError(f"PA raster missing: {pa_path}")

    with rasterio.open(value_path) as val_src, rasterio.open(pa_path) as pa_src:
        # Safety checks
        if val_src.transform != pa_src.transform:
            raise RuntimeError("Transforms do not match")
        if val_src.crs != pa_src.crs:
            raise RuntimeError("CRS does not match")
        if val_src.width != pa_src.width or val_src.height != pa_src.height:
            raise RuntimeError("Raster shapes do not match")

        profile = val_src.profile.copy()
        profile.update(dtype="float32", nodata=0, **pf.get_compression_profile(cfg, "continuous"))

        transform = val_src.transform
        pixel_lat_deg = abs(transform.e)
        pixel_lon_deg = abs(transform.a)
        bounds_top = val_src.bounds.top

        total_before = 0.0
        total_after = 0.0

        with rasterio.open(out_raster, "w", **profile) as dst:
            # We process block by block to avoid loading full raster
            # For each block, exclude or zero-out values located strictly inside designated protected Ecological Boundaries
            # Force block processing based on output structure
            for _, window in tqdm(list(dst.block_windows(1)), desc="Masking PA"):
                # Read
                value = val_src.read(1, window=window).astype(np.float64)
                pa = pa_src.read(1, window=window)

                # Check if we have data
                if value.size == 0:
                    continue

                # Normalize nodata -> NaN
                if val_src.nodata is not None:
                    value[value == val_src.nodata] = np.nan

                # Area-weighted pixel area for this block (USD/km² * km² = USD)
                row_off = window.row_off
                tile_h = window.height
                mid_lat = bounds_top - (row_off + tile_h / 2.0 + 0.5) * pixel_lat_deg
                cos_lat = max(abs(math.cos(math.radians(mid_lat))), 0.001)
                pixel_area_km2 = (pixel_lat_deg * 111.32) * (pixel_lon_deg * 111.32 * cos_lat)

                # Sum before (USD/km² * km² = USD, ignoring NaNs)
                total_before += float(np.nansum(value * pixel_area_km2))

                # Mask: PA == 1 -> 0
                # Assuming PA=1 is protected
                if pa_src.nodata is not None:
                    pass

                value[pa == 1] = 0.0

                # Sum after
                total_after += float(np.nansum(value * pixel_area_km2))

                # Write masked raster (NaN -> 0, nodata = 0)
                out_data = np.nan_to_num(value, nan=0.0).astype(np.float32)
                dst.write(out_data, 1, window=window)

    removed = total_before - total_after
    pct_removed = (removed / total_before * 100) if total_before > 0 else 0.0

    # Summary CSV
    summary = pd.DataFrame([{
        "total_value_before_mask": round(total_before, 2),
        "total_value_after_mask": round(total_after, 2),
        "value_removed_by_PA": round(removed, 2),
        "percent_removed": round(pct_removed, 4),
    }])
    save_csv(summary, out_csv)

    logger.info("PA masking complete.")
    logger.info("  Before: %.2f  After: %.2f  Removed: %.4f%%",
                total_before, total_after, pct_removed)
    logger.info("Output:  %s", out_raster)
    logger.info("Summary: %s", out_csv)

    return out_raster


def summarize_run(cfg: pf.SufficiencySettings, scenario: str = "2020") -> None:
    """Summarize total values of all generated valuation rasters."""
    logger = logging.getLogger("poll_suff_pipeline")
    out_dir = cfg.output_dir
    summary_path = out_dir / f"pollination_sufficiency_valuation_totals_{scenario}.csv"
    
    # Rasters to check (Name -> Description)
    targets = {
        f"value_pollination_sufficiency_{scenario}_5km.tif": "5km Valuation (ag_value * sufficiency)",
        f"value_pollination_sufficiency_{scenario}_300m.tif": "300m Valuation (redistributed)",
        f"value_pollination_sufficiency_{scenario}_300m_no_agri_in_PA.tif": "300m Valuation (Masked: No Ag in PA)",
        f"nature_value_pollination_{scenario}_300m.tif": "Nature Attribution (Value provided by Nature pixels)",
        f"pa_value_pollination_{scenario}_300m.tif": "PA Analysis (Value provided by PA pixels)"
    }
    
    rows = []
    
    for fname, desc in targets.items():
        path = out_dir / fname
        if not path.exists():
            continue
            
        try:
            arr, meta = read_raster(path)
            
            # Mask nodata
            nodata = meta.get("nodata")
            if nodata is not None:
                arr[arr == nodata] = np.nan
            
            arr[arr < 0] = np.nan

            arr = arr.astype(np.float64) # Ensure precision for sum
          
            area_km2 = pf.build_area_km2_raster(meta)
            mass = pf.convert_density_to_mass(arr, area_km2)
            total_usd = float(np.nansum(mass))
            
            rows.append({
                "raster_filename": fname,
                "description": desc,
                "total_value_usd2024": total_usd,
                "path": str(path)
            })
            logger.info(f"Summary: {fname} = {total_usd/1e9:.3f} B USD")
            
        except Exception as e:
            logger.error(f"Failed to summarize {fname}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(summary_path, index=False)
        logger.info(f"Saved valuation totals to: {summary_path}")


def run_pollination_diff_5km_pnas(
    cfg: pf.SufficiencySettings,
    scenario: str,
    baseline_scenario: str = "2023_pnas",
    scenario_value_path: Path | None = None,
    baseline_value_path: Path | None = None,
) -> Path:
    """Compute a 5 km difference raster restricted to stable cropland.

    Replicates the behaviour of ``pollination_shock()`` in the original
    InVEST/GLOBIO repos: only changes in pollination sufficiency on
    continuously farmed land are captured.

    Two modes:

    **5 km mask mode** (default when ``scenario_value_path`` is ``None``):
        Uses the pre-built stable-ag mask at 5 km to zero out cells where
        land converted.  Fast but approximate — partial conversions within a
        5 km cell are not handled at sub-pixel resolution.

    **Exact sub-pixel mode** (when explicit paths are provided):
        ``scenario_value_path`` and ``baseline_value_path`` point to value
        rasters already computed from stable-ag-restricted sufficiency (via
        :func:`run_pollination_sufficiency_300m_stable_ag`).
        The 5 km averages in those rasters already exclude converted sub-pixels,
        so no additional masking is needed — a plain subtraction is sufficient.
        This exactly matches the 300 m pixel-level masking of the original repos.

    Parameters
    ----------
    scenario:
        Label of the future LULC scenario.  Also determines the output filename.
    baseline_scenario:
        Label of the baseline scenario.  Also determines the output filename.
    scenario_value_path:
        Override the scenario value raster path (exact sub-pixel mode).
    baseline_value_path:
        Override the baseline value raster path (exact sub-pixel mode).

    Returns
    -------
    Path
        ``diff_value_pollination_sufficiency_{scenario}_vs_{baseline_scenario}_5km.tif``
    """
    out_dir  = cfg.output_dir
    out_path = out_dir / f"diff_value_pollination_sufficiency_{scenario}_vs_{baseline_scenario}_5km.tif"

    logger.info(
        "Computing PNAS-style difference raster (stable ag only): %s − %s",
        scenario, baseline_scenario,
    )

    if out_path.exists():
        logger.info("Output already exists, skipping: %s", out_path)
        return out_path

    # Resolve input paths
    s_path = scenario_value_path  or (out_dir / f"value_pollination_sufficiency_{scenario}_5km.tif")
    b_path = baseline_value_path  or (out_dir / f"value_pollination_sufficiency_{baseline_scenario}_5km.tif")
    exact_mode = scenario_value_path is not None

    required = [s_path, b_path]
    if not exact_mode:
        mask_path = out_dir / f"stable_ag_mask_{scenario}_vs_{baseline_scenario}_5km.tif"
        required.append(mask_path)

    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Required input missing: {p}")

    with rasterio.open(s_path) as src_s, rasterio.open(b_path) as src_b:
        if src_s.shape != src_b.shape:
            raise RuntimeError(
                f"Shape mismatch: scenario {src_s.shape} vs baseline {src_b.shape}"
            )
        scen_data = src_s.read(1).astype(np.float32)
        base_data = src_b.read(1).astype(np.float32)
        if src_s.nodata is not None:
            scen_data[scen_data == src_s.nodata] = np.nan
        if src_b.nodata is not None:
            base_data[base_data == src_b.nodata] = np.nan

        diff = scen_data - base_data
        profile = src_s.profile.copy()

    if not exact_mode:
        # 5 km mask mode: zero out cells not agricultural in both periods
        with rasterio.open(mask_path) as src_m:
            stable = src_m.read(1).astype(np.float32)
            if src_m.nodata is not None:
                stable[stable == src_m.nodata] = np.nan
        diff = np.where(stable == 1.0, diff, 0.0).astype(np.float32)
    # exact mode: no masking needed — converted sub-pixels already excluded
    # from the stable-ag sufficiency averages that produced s_path / b_path.

    profile.update(dtype="float32", nodata=-9999.0, compress="lzw")
    out_data = np.nan_to_num(diff, nan=-9999.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out_data, 1)

    logger.info("Saved PNAS-style difference raster: %s", out_path)
    return out_path


def publish_inputs(p):
    """Every GEP task's first line: the pollination es_config row (defaults layer -- a caller-set
    value wins) plus the shared country references and the results registry. gep_base_year and
    the value raster (gep_quantity_input_path) are a PAIR defaulting to 2023 -- the source
    rasters on hand; the GEP manuscript's base year is 2019, so regenerate
    poll_value_global_2019usd.tif with the recipe in pollination_sufficiency and update BOTH cells (the row
    names the year twice) before quoting a manuscript-aligned number."""
    utilities.hydrate_es_config(p, 'pollination', log=hb.log)
    utilities.hydrate_es_parameters(p, 'pollination', log=hb.log)
    utilities.initialize_country_paths(p)
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def _read_masked(raster_path):
    """A single-band raster as float64 with its nodata read as NaN, and its metadata beside it."""
    import rasterio
    with rasterio.open(raster_path) as src:
        arr = src.read(1).astype(np.float64)
        arr[arr == src.nodata] = np.nan
        return arr, src.meta.copy()


def _zonal_context(denominator_path, correspondence_gpkg):
    """The fixed side of every zonal percent change, read once.

    The baseline value raster defines the grid, so the pixel-area raster and the burned zone ids
    are built on it. All three plus the zone labels are handed to pf.zonal_pct_change with each
    scenario's difference array.
    """
    import geopandas as gpd
    from rasterio.features import rasterize
    from global_invest.pollination.pollination_sufficiency import build_area_km2_raster

    gdf = gpd.read_file(correspondence_gpkg, engine='pyogrio')
    if gdf.crs is None or gdf.crs.to_epsg() != pf.LATLON_EPSG:
        gdf = gdf.to_crs(pf.LATLON_EPSG)
    gdf[pf.REGION_ID_FIELD] = gdf[pf.REGION_ID_FIELD].astype(int)

    baseline, meta = _read_masked(denominator_path)
    zones = rasterize(
        ((g, int(r)) for g, r in zip(gdf.geometry, gdf[pf.REGION_ID_FIELD]) if g is not None and not g.is_empty),
        out_shape=(meta['height'], meta['width']), transform=meta['transform'],
        fill=pf.NO_ZONE_ID, dtype=np.int32)
    return baseline, pf.build_area_km2_raster(meta), zones, pf.zone_labels_from_boundary(gdf)


def pollination_shock(p):
    """Per-scenario 300 m LULC at each SEALS anchor year -> V_F/OSD shock, piecewise-interp to annual.

    Caller sets on p: es_shock_years (SEALS anchor years, from seals_years),
    es_shock_base_year, es_shock_scenarios, es_lulc_path_template
    ({scenario}/{year}) or scenario_lulc_paths, pollination_base_year_lulc_path (or the shared es_base_year_lulc_path),
    pollination_shock_output_path. Optional: es_shock_base_scenario, pollination_shock_acts,
    region_boundary_path.
    """
    # Default into the es_shocks parent dir. Runtime, not build time: p.es_shock_dir is
    # published by that task, which ProjectFlow runs before this one.
    if not getattr(p, 'pollination_shock_output_path', None):
        p.pollination_shock_output_path = os.path.join(getattr(p, 'es_shock_dir', None) or p.project_dir, 'pollination_interpolated.csv')
    if not p.run_this:
        return
    # Export keys and boundary are es_parameters shipped defaults; a consumer's own tables win.
    utilities.hydrate_es_parameters(p, 'pollination', log=hb.log)
    base_scenario      = utilities.required_base_scenario(p, 'pollination')
    es_shock_base_year = int(p.es_shock_base_year)
    es_shock_scenarios = list(p.es_shock_scenarios)      # was read unbound: this task raised NameError
    anchor_years = sorted(y for y in map(int, p.es_shock_years) if y > es_shock_base_year)

    cfg = pf.configure_crop_benefits(p, es_shock_base_year)            # crop_benefits Config -> our base_data + task dir

    if not getattr(p, 'scenario_lulc_paths', None):
        tmpl = p.es_lulc_path_template
        p.scenario_lulc_paths = {s: {y: glob.glob(tmpl.format(scenario=s, year=y))[0]
                                     for y in anchor_years if glob.glob(tmpl.format(scenario=s, year=y))}
                                 for s in [base_scenario] + es_shock_scenarios}

    # base-year SEALS7 map: per-ES override, else the ES-shared attr. NEVER p.base_year_lulc_path, which
    # SEALS OWNS and overwrites at runtime with its raw-ESA source (a raw-ESA base map would make the
    # SEALS7-keyed sufficiency lookup produce garbage). This base-year value IS pollination's primary
    # denominator (it feeds both fixedbase and contemp), so a missing one is fatal.
    base_map = getattr(p, 'pollination_base_year_lulc_path', None) or getattr(p, 'es_base_year_lulc_path', None)
    if not base_map:
        raise ValueError('pollination base-year LULC not set: point p.es_base_year_lulc_path (or '
                         'p.pollination_base_year_lulc_path) at the SEALS7 base-year map.')
    # The denominator (unpaired 2023 value) is year- and scenario-independent, so the fixed side of
    # the zonal step is built once.
    denominator_path = pf.baseline_denominator(cfg, base_map, es_shock_base_year)
    baseline_arr, area_arr, zones_arr, zone_labels = _zonal_context(denominator_path, p.region_boundary_path)

    # value[scenario][year] = per-zone % change of that scenario's year-map vs the 2023 baseline (stable
    # ag). level_usd = the denominator of that % change, the per-zone absolute baseline value in base-year
    # USD, emitted so the GEP chain can consume this task instead of rerunning the same rasters.
    value, level_usd = {}, None
    for year in anchor_years:
        for scen in [base_scenario] + es_shock_scenarios:
            diff_arr, _ = _read_masked(pf.scenario_diff_raster(
                cfg, scenario=f'{scen}_{year}', lulc_path=p.scenario_lulc_paths[scen][year],
                baseline_lulc_path=base_map, target_year=es_shock_base_year))
            pct, level = pf.zonal_pct_change(diff_arr, baseline_arr, area_arr, zones_arr, zone_labels)
            value.setdefault(scen, {})[year] = pct
            if level_usd is None:
                level_usd = level

    # The shock numerator is scenario minus nature-off baseline at each anchor. anchor_shock_tables
    # puts it over the two denominators and dynamic_shock_rows expands those to annual rows.
    rows = []
    for scen in es_shock_scenarios:
        anchor_shock, anchor_contemp = pf.anchor_shock_tables(
            {y: value[scen][y] for y in anchor_years},
            {y: value[base_scenario][y] for y in anchor_years})
        rows += pf.dynamic_shock_rows(anchor_shock, anchor_contemp, level_usd, scen,
                                      p.pollination_shock_acts, es_shock_base_year)

    out = pd.DataFrame(rows)
    utilities.assert_shock_table_sound(out, es_shock_scenarios, 'pollination')
    out.to_csv(p.pollination_shock_output_path, index=False)
    print('  pollination shock: %d rows, %d scenarios (shock_pct=shock_pct_contemp=/baseline-year value, shock_pct_fixedbase=/2023 value) -> %s'
          % (len(out), out['scenario'].nunique() if rows else 0, p.pollination_shock_output_path))
    # value_usd_base is the GEP hand-off, not read by GTAP (build_combined_afeall takes shock_pct only).
    if level_usd is not None and len(level_usd):
        print('  pollination value (GEP): %d zones, total %.4g base-year USD -> column value_usd_base'
              % (len(level_usd), float(level_usd.sum())))
    return True


def pollination_shock_static(p):
    """Static per-scenario pollination shock -> V_F/OSD, linear ramp 0->es_shock_end_year, from the frozen table.

    The fallback add_pollination_tasks selects when <2 SEALS map years exist. READS
    input_dir/raw_dependencies/pollination_dependency.csv (override p.pollination_dependency_path) and
    subtracts the baseline_ignore_damages row (the frozen table's nature-off baseline; == ignore
    dependencies, just the old label kept in this CSV), ramping that difference linearly from 0 at
    es_shock_base_year. NEVER writes back to raw_dependencies -- output goes to p.pollination_shock_output_path
    (pollination_interpolated.csv), the same file the dynamic task writes. Caller sets:
    es_shock_base_year, es_shock_end_year, es_shock_scenarios,
    pollination_shock_output_path; scenario->raw via p.pollination_scenario_map (default: identity --
    each scenario maps to its own name; a scenario the table labels differently is warned about loudly
    and skipped rather than silently zeroed, so set the map for those);
    sectors via p.pollination_shock_acts (default ('V_F', 'OSD'), matching the dynamic task).
    """
    # Default into the es_shocks parent dir. Runtime, not build time: p.es_shock_dir is
    # published by that task, which ProjectFlow runs before this one.
    if not getattr(p, 'pollination_shock_output_path', None):
        p.pollination_shock_output_path = os.path.join(getattr(p, 'es_shock_dir', None) or p.project_dir, 'pollination_interpolated.csv')
    if not p.run_this:
        return

    utilities.hydrate_es_parameters(p, 'pollination', log=hb.log)   # shipped defaults; caller wins
    es_shock_base_year = int(p.es_shock_base_year)
    es_shock_end_year = int(p.es_shock_end_year)
    pollination_scenario_map = getattr(p, 'pollination_scenario_map', {})
    es_shock_scenarios = list(p.es_shock_scenarios)

    poll_path = getattr(p, 'pollination_dependency_path', None) or os.path.join(
        p.input_dir, 'raw_dependencies', 'pollination_dependency.csv')
    if not os.path.exists(poll_path):
        print('  pollination shock: dependency csv not found (%s) -- skipping' % poll_path)
        return

    df = hb.df_read(poll_path)
    # Normalise the one year-suffixed label so pollination's table presents the same scenario names as
    # the other services (carbon is plain everywhere, erosion strips '_2050' on read). net_zero_2050 is
    # the only alias here, so target it explicitly rather than a blunt '_2050' strip that would silently
    # mangle any future label carrying that suffix. Keeps the consumer's pollination_scenario_map at identity.
    df['scenario'] = df['scenario'].replace({'net_zero_2050': 'net_zero'})
    df = df[df['ENDW'] != 'AEZ0']  # AEZ0 not valid in GTAP
    # The base resolves through the candidate mechanism (fatal if absent): the table's spelling
    # may differ from the configured name (nature-off aliasing in utilities).
    base_scenario = utilities.required_base_scenario(p, 'pollination')
    raw_base = utilities.resolve_base_scenario(df['scenario'].values, pollination_scenario_map, base_scenario, 'pollination')
    base = df[df['scenario'] == raw_base].set_index(['ENDW', 'REG'])['value'].astype(float)

    rows = []
    for our_scn in es_shock_scenarios:
        raw_scn = utilities.resolve_raw_scenario(df['scenario'].values, pollination_scenario_map, our_scn, 'pollination')
        if raw_scn is None:
            continue
        scn_vals = df[df['scenario'] == raw_scn].set_index(['ENDW', 'REG'])['value'].astype(float)
        rows += pf.static_shock_rows(base, scn_vals, our_scn, p.pollination_shock_acts,
                                     es_shock_base_year, es_shock_end_year)

    out = pd.DataFrame(rows)
    utilities.assert_shock_table_sound(out, es_shock_scenarios, 'pollination')
    out.to_csv(p.pollination_shock_output_path, index=False)
    nz = out[(out['year'] == es_shock_end_year) & (out['shock_pct'] != 0)] if len(out) else out
    print('  pollination shock: %d rows, %d scenarios, %d nonzero @%d (static, uncapped) -> %s'
          % (len(out), out['scenario'].nunique() if len(out) else 0, len(nz), es_shock_end_year,
             p.pollination_shock_output_path))
    return True


# =============================================================================
# GEP valuation tasks. The ES shock above and the valuation below are separate
# consumers of the same crop_benefits science; neither depends on the other.
# The value raster (poll_value_global_<year>usd.tif, a crop_benefits output) is
# USD per cell with crop prices and pollination dependence already embedded
# upstream, so lambda = 1 and no price join happens here: quantity IS value.
# =============================================================================

def pollination_value_by_region(p):
    """GEP quantity stage: per-region sum of the pollination value raster (USD per cell) on the
    service's aggregation surface (the gep_regions cells)."""
    publish_inputs(p)
    p.pollination_value_by_region_path = os.path.join(p.cur_dir, "pollination_value_by_region.csv")
    if not p.run_this:
        return
    utilities.summarize_raster_by_region(
        value_raster_path=p.gep_quantity_input_path,
        region_boundary_path=p.gep_regions_input_path,
        out_path=p.pollination_value_by_region_path,
        year=p.gep_base_year, id_column=p.gep_regions_id_col)
    # Full-scale conservation: the region sums must add up to the raster's own total
    # (verified at 100.0000% on the real raster when this check was added).
    df_regions = hb.df_read(p.pollination_value_by_region_path)
    utilities.assert_zonal_conservation(df_regions['total'].sum(),
                                        p.gep_quantity_input_path, 'pollination')
    return True


def gep_calculation(p):
    """GEP valuation for pollination: r264 region values -> ONE row per country (r250)."""
    publish_inputs(p)
    service_results = {}
    p.results['pollination'] = service_results
    p.results['pollination']['gep_by_country_base_year'] = os.path.join(p.cur_dir, "gep_by_country_base_year.csv")
    # Only register results this task actually writes (per-year results belong to a multi-year run).

    if hb.path_all_exist(list(service_results.values())):
        hb.log("All results already exist. Skipping GEP calculation for pollination.")
        return
    hb.log("Starting GEP calculation for pollination.")

    # 1. Per-region (r264) USD value -> one row per COUNTRY (r250), written as the per-country CSV
    #    that is the source of truth for every sum. Summing the r264-expanded table instead would
    #    double-count split countries (see utilities docstring).
    df_q264 = hb.df_read(p.pollination_value_by_region_path)
    df_gep = pf.collapse_regions_to_countries(df_q264)
    hb.df_write(df_gep, service_results['gep_by_country_base_year'])

    # 2. Map only: r264-expanded, each sub-region carries its country's value, never summed
    #    (carbon template; the repo-wide map convention is the open flag-3 decision).
    df_regions = pf.expand_country_values_to_regions(df_q264, df_gep)
    gdf = hb.df_merge(p.gdf_countries_simplified, df_regions, how='outer',
                      left_on='ee_r264_id', right_on='ee_r264_id')
    gdf.to_file(service_results['gep_by_country_base_year'].replace('.csv', '.gpkg'), driver='GPKG')

    # 3. National total = sum over the one-row-per-country table.
    value_gep_base_year = df_gep['pollination_gep'].sum()
    hb.log(f"Total pollination GEP for base year {p.gep_base_year}: {value_gep_base_year}")
    return value_gep_base_year


def gep_load_results(p):
    """Load GEP results from a PRIOR calculation run so the report renders without recomputing.
    Fails loudly if absent (run run_pollination.py first). The results-only entry point."""
    publish_inputs(p)
    result_path = os.path.join(p.intermediate_dir, 'gep_calculation', 'gep_by_country_base_year.csv')
    if not hb.path_exists(result_path):
        raise FileNotFoundError(
            f"pollination GEP results not found at {result_path}. "
            f"Run the calculation first (run_pollination.py), then re-run results.")
    p.results.setdefault('pollination', {})
    p.results['pollination']['gep_by_country_base_year'] = result_path


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)

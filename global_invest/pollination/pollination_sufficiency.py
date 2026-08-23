"""The pollination sufficiency and value rasters, vendored from crop_benefits.

The science comes from the crop_benefits repository: a sufficiency raster from the share of pollinator habitat
within foraging range of each cell, and a value raster from that sufficiency against the
precomputed baseline pollination value. It arrived here as an installed package, `crop_benefits`,
which was declared in no pyproject and configured through a gitignored local.yaml, so the module
failed to import at all on a machine that lacked both. Only these functions were ever used, about
1,100 lines of its 7,572, and they are now part of this repo like every other service's science.

What changed in the move, and nothing else did: the crop_benefits `Config` object is gone. It was
read for seven attributes, three of which our own driver already overrode, so the four that
remained are now plain arguments. Paths come off the ProjectFlow object, which is where every
other service gets them.
"""
from __future__ import annotations

import gc
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import rasterio
from joblib import Parallel, delayed
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window
from scipy.ndimage import convolve
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class SufficiencySettings:
    """What the raster steps below need, in place of the crop_benefits Config they used to read.

    That Config was loaded from a gitignored local.yaml with `validate=False`, so a missing or
    wrong file did not fail up front, it just proceeded. These seven fields are everything the
    four vendored modules ever read off it, and the pollination task fills them from the
    ProjectFlow object.

    Attributes:
        output_dir (Path): where the sufficiency and value rasters are written, the task's own dir.
        value_raster_dir (Path): where the precomputed baseline pollination-value raster lives.
        country_raster_path (Path): the raster defining the 5 km target grid. The valuation needs
            sufficiency and value on one grid, so this points at the value raster itself.
        lulc_path (Path): the land-cover map the sufficiency is computed from.
        pa_raster_300m_path (Path): the protected-area raster, for the protected-area summary.
        tile_size (int): rows per block when streaming a raster.
        n_workers (int): parallel workers for the tiled sufficiency pass.
    """
    output_dir: Path
    value_raster_dir: Path
    country_raster_path: Path
    lulc_path: Path = None
    pa_raster_300m_path: Path = None
    tile_size: int = 2048
    n_workers: int = 4


# The compression profiles the vendored writers ask for, which used to come off the same Config.
COMPRESSION_PROFILES = {
    'continuous': {'compress': 'DEFLATE', 'predictor': 3, 'zlevel': 6, 'tiled': True,
                   'blockxsize': 256, 'blockysize': 256, 'BIGTIFF': 'IF_SAFER'},
    'categorical': {'compress': 'DEFLATE', 'predictor': 2, 'zlevel': 6, 'tiled': True,
                    'blockxsize': 256, 'blockysize': 256, 'BIGTIFF': 'IF_SAFER'},
    'defaults': {'compress': 'DEFLATE', 'tiled': True, 'BIGTIFF': 'IF_SAFER'},
}



# ---------------------------------------------------------------------------------------------
# Helpers that came with the four modules, from crop_benefits.raster.spatial and crop_benefits.io.
# ---------------------------------------------------------------------------------------------


def build_area_km2_raster(meta: dict) -> np.ndarray:
    """
    Build a 2-D pixel-area raster (km²) from a rasterio profile.
    
    The resulting raster has the same shape as defined in 'meta'.
    """
    transform = meta["transform"]
    nrows = meta["height"]
    ncols = meta["width"]
    
    # Latitude of pixel centers: y = f + e * (row + 0.5)
    # Note: 'e' is usually negative (pixel height) in north-up images
    latitudes = transform.f + transform.e * (np.arange(nrows) + 0.5)
    
    area_per_row = pixel_area_km2(latitudes)
    # Broadcast row areas across all columns
    return np.repeat(area_per_row[:, None], ncols, axis=1).astype(np.float32)


def convert_density_to_mass(density_raster: np.ndarray, area_km2_raster: np.ndarray) -> np.ndarray:
    """
    Convert density (e.g. tonnes/km²) to mass (e.g. tonnes).
    
    mass = density * area
    """
    mass = np.full_like(density_raster, np.nan, dtype=np.float32)
    valid = np.isfinite(density_raster) & (area_km2_raster > 0)
    mass[valid] = density_raster[valid] * area_km2_raster[valid]
    return mass


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


def get_compression_profile(
    cfg: SufficiencySettings,
    profile_name: str = "continuous"
) -> Dict[str, Any]:
    """
    Retrieve compression settings for a given profile name.

    Parameters
    ----------
    cfg : Config
        Pipeline configuration.
    profile_name : str
        Name of the profile (e.g., 'continuous', 'categorical', 'defaults').

    Returns
    -------
    dict
        Dictionary of rasterio creation options (compress, predictor, etc.).
    """
    if profile_name == "continuous":
        return COMPRESSION_PROFILES['continuous'].copy()
    elif profile_name == "categorical":
        return COMPRESSION_PROFILES['categorical'].copy()
    else:
        return COMPRESSION_PROFILES['defaults'].copy()


def save_csv(df: pd.DataFrame, path: Path | str) -> Path:
    """Save DataFrame as CSV and log the result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %d rows -> %s", len(df), path)
    return path


# ---------------------------------------------------------------------------------------------
# The sufficiency raster.
# ---------------------------------------------------------------------------------------------


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
    cfg: SufficiencySettings,
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
            **get_compression_profile(cfg, "continuous")
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

def run_pollination_sufficiency_5km(cfg: SufficiencySettings, input_300m: Path | None = None, scenario: str = "2020") -> Path:
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
    cfg: SufficiencySettings,
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


# ---------------------------------------------------------------------------------------------
# The value raster.
# ---------------------------------------------------------------------------------------------


logger = logging.getLogger(__name__)


def run_pollination_valuation_5km(cfg: SufficiencySettings, scenario: str = "2020", target_year: int = 2024) -> Path:
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


def redistribution_value_to_300m(cfg: SufficiencySettings, scenario: str = "2020") -> Path:
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
        profile.update(dtype="float32", nodata=0.0, **get_compression_profile(cfg, "continuous"))

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


def mask_protected_areas_300m(cfg: SufficiencySettings, scenario: str = "2020") -> Path:
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
        profile.update(dtype="float32", nodata=0, **get_compression_profile(cfg, "continuous"))

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


def summarize_run(cfg: SufficiencySettings, scenario: str = "2020") -> None:
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
          
            area_km2 = build_area_km2_raster(meta)
            mass = convert_density_to_mass(arr, area_km2)
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


# ---------------------------------------------------------------------------------------------
# The scenario difference.
# ---------------------------------------------------------------------------------------------


logger = logging.getLogger(__name__)


def run_pollination_diff_5km_pnas(
    cfg: SufficiencySettings,
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

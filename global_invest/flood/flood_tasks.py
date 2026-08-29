# -*- coding: utf-8 -*-
"""File handling and task definitions for the flood-control account.

Everything that touches disk lives here: fetching and warping the hazard rasters, delineating the
service-demanding areas, routing the service flow, reading the damage tables, windowing every
country out of the global grid, and writing the results. The equations these steps apply are in
`flood_functions`, which knows nothing about where anything is stored.

That direction is one-way and worth keeping so. A calculation that opens its own inputs can only be
checked by replacing the file reader, and a test that replaces a file reader is testing the wiring
rather than the arithmetic.

Configuration arrives through the configure_* functions, each reading the ProjectFlow object once
and setting the module constants the drivers below read. The task wrappers at the end are the seam
`flood_initialize` grafts onto a task tree.
"""
# =============================================================================
# flood_tasks.py
#
# Key science functions for the GEP flood-control / flood-regulation service.
# Follows the same run_x.py / x_tasks.py / x_functions.py / x_initialization.py
# structure Justin described (2026-07-07 email) for the terrestrial_carbon
# module, and mirrors global_invest/erosion/erosion_functions.py.
#
# The original Flood GEP scripts and notebooks map onto five sections:
#
#   A) Input preparation            (download_and_prep_jrc_flood_depth.ipynb,
#      + hazard/exposure layers      sda_step_2A_make_lulc_to_sda_mapping_esa300.py,
#                                    build_sda_from_esa300m.py, road_sda.ipynb,
#                                    qa_spa_global_step1.py)
#   B) SDA delineation              (sda_step2_build_sda_global.py)
#      per ISO3 x return period
#   C) Service flow SPA -> SDA      (serviceflow_step3_spa_to_sda_ratio_global.py)
#   D) Monetary valuation           (build_damage_table_USD2019.py = 4A,
#      (4A -> 4B -> 4C -> 4D)        flood_gep_step4b_pixel_damage_USD2019.py,
#                                    flood_gep_step4c_ead_USD2019_global.py,
#                                    flood_gep_step4d_export_global_USD2019.py)
#   E) Maps & figures               (analyze_step4d_global_results.py)
#
# Generic raster/table/plotting helpers reused across sections live in
# flood_utils.py and are imported by name below.
#
# -----------------------------------------------------------------------------
# DESIGN NOTE ON CONFIG (same rationale as erosion_functions.py)
# -----------------------------------------------------------------------------
# The original scripts were argparse CLIs and notebooks that hardcoded paths
# and knobs as module-level constants. To keep the working science logic
# intact rather than risk introducing bugs while threading a config object
# through every nested reference, each section keeps its constants as
# module-level globals *with the same defaults the originals used*, and
# exposes a configure_*(p) function that a task calls first to override them
# from the ProjectFlow object `p`. run_flood.py sets the p.flood_* attributes.
#
# ONE DELIBERATE DEVIATION FROM erosion_functions.py: that module reuses the
# name ROOT in all three sections, so the last definition silently wins for
# every section (a latent bug). Here each section's constants have distinct
# names, and the genuinely shared ones (ROOT / INPUTS / OUTPUTS) are defined
# exactly once at the top.
#
# -----------------------------------------------------------------------------
# DESIGN NOTE ON THE THREE "INVOKED" SCRIPTS
# -----------------------------------------------------------------------------
# Three of the original scripts are long, heavily-validated argparse CLIs whose
# bodies encode a lot of hard-won edge-case handling (smart-skip signatures,
# depth-raster auto-detection, robust column detection, GPKG enrichment):
#
#   sda_step2_build_sda_global.py             (Section B)
#   build_damage_table_USD2019.py             (Section D, step 4A)
#   flood_gep_step4d_export_global_USD2019.py (Section D, step 4D)
#
# Rather than re-type them (and risk silently changing behaviour), this module
# keeps them byte-for-byte as files under flood/scripts/ and drives them via
# _invoke_script(), which builds an argv from the configured constants and runs
# the script's own main(). Inlining these three the way Sections A/C/D-4B/D-4C
# are inlined is the main outstanding refactor -- see README "Known issues".
# =============================================================================
from __future__ import annotations

import importlib.util
import json
import os
import sys
import warnings
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

DAMAGE_DEPTH_MODE = "interpolated"   # or "banded"; see _band_depth_inca
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features as rio_features
from rasterio.mask import mask as rio_mask
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform as window_transform
from shapely.ops import unary_union

import matplotlib
matplotlib.use("Agg")  # HPC batch nodes have no display
import matplotlib.pyplot as plt

from global_invest.flood.flood_utils import (
    assert_exists,
    assert_same_grid,
    pixel_area_m2,
    pixel_area_km2,
    mercator_area_scale,
    raster_profile_string,
    warn_if_geographic,
    random_windows,
    atomic_write_raster,
    raster_ok,
    sha256_file,
    file_fingerprint,
    pick_iso3_column,
    pick_name_column,
    load_admin0,
    norm_label,
    find_col,
    to_float,
    to_num,
    write_csv,
    safe_mean,
    integrate_trapezoid,
    rp_to_p,
    fmt_usd,
    fmt_usd_millions,
    fmt_percent,
    build_interval_labels,
    compute_classification,
    savefig,
    top_n,
    plot_raster_global,
    plot_publication_choropleth_categorical,
)
from global_invest.flood import flood_utils


# =============================================================================
# SHARED PROJECT PATHS (defined ONCE -- see design note above)
# =============================================================================
# Where the flood inputs and intermediates live. The source repo read this from a
# GEP_FLOOD_ROOT environment variable, defaulting to one account's cluster path. In this library
# machine-specific configuration lives in es_parameters.csv, never in code and never in an
# environment variable, so `flood_root_dir` is hydrated onto p and this module-level default only
# names where a fresh machine would look. Nothing here reads the environment.
ROOT = Path(os.environ.get('GEP_FLOOD_ROOT') or
            os.path.join(os.path.expanduser('~'), 'flood_gep'))
INPUTS = None
OUTPUTS = None

ADMIN0_PATH = None

# Return periods actually on disk for this project: RP10, RP20, RP50, RP500.
#
# The JRC GlobalMaps collection also publishes RP100 and RP200 at the same URL
# pattern, but they were not downloaded. The cost of the gap is small and
# measurable. The probability band between RP50 (p=0.02) and RP500 (p=0.002)
# contributes only ~8-10% of total EAD, and linear interpolation across it
# overstates damage at the interior points by roughly 6-10%. Net effect on EAD:
#
#     concave (saturating) damage curve   +0.5%
#     near-linear in log RP               +0.5%
#     convex (tail-heavy)                 +1.3%
#
# So four return periods biases EAD HIGH by about 1%. Defensible, and worth
# stating rather than leaving implicit -- but adding RP100/RP200 is still the
# cheapest available accuracy improvement, since the amplification factor is
# larger for frequent events and EAD weights them most heavily.
RETURN_PERIODS = [10, 20, 50, 500]



import hazelbean as hb
from global_invest import utilities
from global_invest.flood import flood_functions as ff


# Raster-window helpers: they take a rasterio dataset or a Window rather than
# arrays, so they belong on this side of the split.
def _normalize_spa_ratio(arr: np.ndarray, nodata) -> np.ndarray:
    out = arr.astype(np.float32, copy=False)
    if nodata is not None:
        out = np.where(out == nodata, np.nan, out)
    return np.clip(out, 0.0, 1.0)


def _iter_tiles(win: Window, size: int):
    """Yield sub-windows of `win`, at most size x size, in absolute raster coords."""
    r0, c0 = int(win.row_off), int(win.col_off)
    h, w = int(win.height), int(win.width)
    for r in range(r0, r0 + h, size):
        for c in range(c0, c0 + w, size):
            yield Window(c, r, min(size, c0 + w - c), min(size, r0 + h - r))


def _country_window(ds, geom, pad: int = 1) -> Optional[Window]:
    """Bounding-box window of a geometry, clamped to the raster extent."""
    minx, miny, maxx, maxy = geom.bounds
    win = from_bounds(minx, miny, maxx, maxy, transform=ds.transform)
    win = win.round_offsets(op="floor").round_lengths(op="ceil")
    win = Window(max(0, win.col_off - pad), max(0, win.row_off - pad),
                 win.width + 2 * pad, win.height + 2 * pad)
    win = win.intersection(Window(0, 0, ds.width, ds.height))
    if win.width <= 0 or win.height <= 0:
        return None
    return win


def _amp_tile(amp_src, shape, transform, crs) -> Optional[np.ndarray]:
    """Reproject the amplification field onto one tile. Cheap; no full-grid read."""
    if amp_src is None:
        return None
    field = np.full(shape, np.nan, dtype="float32")
    reproject(
        rasterio.band(amp_src, 1), field,
        src_transform=amp_src.transform, src_crs=amp_src.crs,
        dst_transform=transform, dst_crs=crs,
        resampling=Resampling.average,
        src_nodata=amp_src.nodata, dst_nodata=np.nan,
    )
    return np.maximum(np.where(np.isfinite(field), field, 1.0), 1.0).astype("float32")

def configure_shared(p):
    """
    Override the project-wide paths. Every configure_*(p) below calls this
    first, so a task only ever needs to call its own configure function.
    """
    global ROOT, INPUTS, OUTPUTS, ADMIN0_PATH, RETURN_PERIODS

    ROOT = Path(getattr(p, 'flood_root', ROOT))
    INPUTS = Path(getattr(p, 'flood_input_dir', ROOT / "inputs"))
    OUTPUTS = Path(getattr(p, 'flood_output_dir', ROOT / "outputs"))
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    ADMIN0_PATH = Path(getattr(
        p, 'flood_country_boundary_path',
        INPUTS / "country_vector" / "country_boundary_r250_with_iso3.gpkg"))

    RETURN_PERIODS = list(getattr(p, 'flood_return_periods', RETURN_PERIODS))


def _invoke_script(script_path: Path, argv: List[str], label: str = ""):
    """
    Load one of the preserved original CLI scripts as a module and run its
    main() with a constructed argv. See the design note at the top of this file
    for why these three scripts are driven rather than inlined.
    """
    script_path = Path(script_path)
    assert_exists(
        script_path,
        f"Original {label or script_path.name} script not found. Set the "
        f"corresponding p.flood_*_script_path, or place the script under "
        f"global_invest/flood/scripts/.")

    spec = importlib.util.spec_from_file_location(f"_flood_script_{script_path.stem}", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "main"):
        raise AttributeError(f"{script_path} has no main() to invoke.")

    old_argv = sys.argv
    sys.argv = [str(script_path)] + [str(a) for a in argv]
    try:
        print(f"[INVOKE] {script_path.name} {' '.join(str(a) for a in argv)}")
        return mod.main()
    finally:
        sys.argv = old_argv


LULC_PATH = None


JRC_ZIP_URL_TEMPLATE = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/FLOODS/GlobalMaps/floodMapGL_rp{rp}y.zip"
)


DEPTH_RAW_DIR = None


DEPTH_EXTRACT_DIR = None


DEPTH_ALIGNED_DIR = None


DEPTH_MASK_DIR = None


SDA_MAPPING_JSON = None


ESA_CODEBOOK = {
    0: "No Data",
    10: "Cropland, rainfed",
    11: "Cropland, rainfed, herbaceous cover",
    12: "Cropland, rainfed, tree or shrub cover",
    20: "Cropland, irrigated or post-flooding",
    30: "Mosaic cropland (>50%) / natural vegetation (<50%)",
    40: "Mosaic natural vegetation (>50%) / cropland (<50%)",
    50: "Tree cover, broadleaved, evergreen",
    60: "Tree cover, broadleaved, deciduous",
    61: "Tree cover, broadleaved, deciduous, closed",
    62: "Tree cover, broadleaved, deciduous, open",
    70: "Tree cover, needleleaved, evergreen",
    71: "Tree cover, needleleaved, evergreen, closed",
    72: "Tree cover, needleleaved, evergreen, open",
    80: "Tree cover, needleleaved, deciduous",
    81: "Tree cover, needleleaved, deciduous, closed",
    82: "Tree cover, needleleaved, deciduous, open",
    90: "Tree cover, mixed leaf type",
    100: "Mosaic tree and shrub (>50%) / herbaceous cover (<50%)",
    110: "Mosaic herbaceous cover (>50%) / tree and shrub (<50%)",
    120: "Shrubland",
    121: "Evergreen shrubland",
    122: "Deciduous shrubland",
    130: "Grassland",
    140: "Lichens and mosses",
    150: "Sparse vegetation",
    151: "Sparse tree",
    152: "Sparse shrub",
    153: "Sparse herbaceous",
    160: "Tree cover, flooded, fresh/brackish water",
    170: "Tree cover, flooded, saline water",
    180: "Shrub/herbaceous cover, flooded",
    190: "Urban areas",
    200: "Bare areas",
    201: "Consolidated bare areas",
    202: "Unconsolidated bare areas",
    210: "Water bodies",
    220: "Permanent snow and ice",
}


ESA_TO_SDA = {
    "artif": [190],
    "crop": [10, 11, 12, 20, 30, 40],
    "pasture": [130],
}


SDA_CODE = {"none": 0, "artif": 1, "crop": 2, "pasture": 3, "roads": 4}


INCLUDE_PASTURE = True


GLOBAL_SDA_TIF = None


GLOBAL_SDA_LEGEND_CSV = None


SPA_PATH = None


SPA_RATIO_PATH = None


QA_DIR = None


DEPTH_THRESHOLD_M = 0.1  # depths at/below this are treated as no damage


def configure_inputs(p):
    """
    Override the Section-A (input preparation) constants from the ProjectFlow
    object. Called by flood_tasks.task_prepare_flood_inputs().
    """
    configure_shared(p)

    global LULC_PATH, JRC_ZIP_URL_TEMPLATE, DEPTH_RAW_DIR, DEPTH_EXTRACT_DIR
    global DEPTH_ALIGNED_DIR, DEPTH_MASK_DIR, SDA_MAPPING_JSON
    global ESA_TO_SDA, INCLUDE_PASTURE, GLOBAL_SDA_TIF, GLOBAL_SDA_LEGEND_CSV
    global SPA_PATH, SPA_RATIO_PATH, QA_DIR, DEPTH_THRESHOLD_M

    LULC_PATH = Path(getattr(p, 'flood_lulc_path', INPUTS / "lulc" / "lulc_esa_2019_int_reproj.tif"))

    JRC_ZIP_URL_TEMPLATE = getattr(p, 'flood_jrc_zip_url_template', JRC_ZIP_URL_TEMPLATE)
    DEPTH_RAW_DIR = Path(getattr(p, 'flood_depth_raw_dir', INPUTS / "floodplain_depth_raw"))
    DEPTH_EXTRACT_DIR = Path(getattr(p, 'flood_depth_extract_dir', DEPTH_RAW_DIR / "extracted_tifs"))
    DEPTH_ALIGNED_DIR = Path(getattr(p, 'flood_depth_aligned_dir',
                                     INPUTS / "floodplain_depth" / "aligned_to_lulc"))
    DEPTH_MASK_DIR = Path(getattr(p, 'flood_depth_mask_dir',
                                  INPUTS / "floodplain_depth" / "masks_aligned_to_lulc"))

    SDA_MAPPING_JSON = Path(getattr(p, 'flood_sda_mapping_json',
                                    INPUTS / "lulc_to_sda_mapping" / "lulc_to_sda_mapping.json"))
    ESA_TO_SDA = getattr(p, 'flood_esa_to_sda', ESA_TO_SDA)
    INCLUDE_PASTURE = getattr(p, 'flood_include_pasture', True)

    GLOBAL_SDA_TIF = Path(getattr(p, 'flood_global_sda_raster_path',
                                  INPUTS / "sda" / "sda_esa300m_artif_crop_pasture.tif"))
    GLOBAL_SDA_LEGEND_CSV = Path(getattr(p, 'flood_global_sda_legend_csv',
                                         GLOBAL_SDA_TIF.parent / "sda_esa300m_legend.csv"))

    SPA_PATH = Path(getattr(p, 'flood_spa_path', INPUTS / "global_spa_ben" / "global_prr_spa.tif"))
    SPA_RATIO_PATH = Path(getattr(p, 'flood_spa_ratio_path',
                                  INPUTS / "global_spa_ben" / "global_upstream_spa_ratio.tif"))

    QA_DIR = Path(getattr(p, 'flood_qa_dir', OUTPUTS / "qa_maps"))
    QA_DIR.mkdir(parents=True, exist_ok=True)

    DEPTH_THRESHOLD_M = getattr(p, 'flood_depth_threshold_m', 0.1)


def _download(url: str, dst: Path) -> None:
    from urllib.request import urlretrieve

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[SKIP] already downloaded: {dst.name}")
        return
    print(f"[DL] {url}")
    urlretrieve(url, dst.as_posix())
    if not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError(f"Download failed or empty file: {dst}")
    print(f"[OK]  saved: {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


def _extract_zip(zip_path: Path, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: List[Path] = []
    print(f"[UNZIP] {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as z:
        tifs = [m for m in z.namelist() if m.lower().endswith((".tif", ".tiff"))]
        if not tifs:
            raise RuntimeError(f"No GeoTIFF found in {zip_path}")
        for m in tifs:
            out_path = out_dir / Path(m).name
            if out_path.exists() and out_path.stat().st_size > 0:
                extracted.append(out_path)
                continue
            z.extract(m, out_dir)
            nested = out_dir / m
            if nested.exists() and nested != out_path:
                nested.rename(out_path)
            extracted.append(out_path)
    print(f"[OK]  extracted {len(extracted)} tif(s)")
    return extracted


def _warp_to_lulc(src_tif: Path, dst_tif: Path, ref_profile: dict,
                  resampling=Resampling.bilinear) -> None:
    """
    Warp a depth raster onto the LULC grid, streaming to disk. Depth is a
    continuous variable, so bilinear is appropriate (never nearest).
    """
    with rasterio.open(src_tif) as src:
        src_nodata = src.nodata if src.nodata is not None else -9999.0

        prof = ref_profile.copy()
        prof.update(driver="GTiff", dtype="float32", count=1, nodata=-9999.0,
                    compress="DEFLATE", tiled=True, BIGTIFF="YES")

        dst_tif.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_tif, "w", **prof) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform, src_crs=src.crs, src_nodata=src_nodata,
                dst_transform=prof["transform"], dst_crs=prof["crs"],
                dst_nodata=prof["nodata"], resampling=resampling,
            )


def _write_mask_from_depth(depth_tif: Path, mask_tif: Path, threshold: float = 0.0) -> None:
    with rasterio.open(depth_tif) as src:
        arr = src.read(1).astype("float32")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        mask = (np.nan_to_num(arr, nan=0.0) > threshold).astype("uint8")
        prof = src.profile.copy()
        prof.update(dtype="uint8", count=1, nodata=0, compress="DEFLATE",
                    tiled=True, BIGTIFF="IF_SAFER")
    atomic_write_raster(mask_tif, prof, mask)


def download_and_align_jrc_depth(return_periods: Optional[List[int]] = None) -> Dict[int, Path]:
    """
    Download the JRC global flood hazard maps (water depth, metres) for each
    return period and warp them onto the LULC reference grid, so every later
    step can do pixelwise depth-damage without re-warping.
    """
    rps = return_periods if return_periods is not None else RETURN_PERIODS
    assert_exists(LULC_PATH, "LULC reference grid is required before aligning depth.")

    with rasterio.open(LULC_PATH) as ref:
        ref_profile = ref.profile.copy()
        ref_profile.update(width=ref.width, height=ref.height,
                           transform=ref.transform, crs=ref.crs)

    print("[INFO] LULC reference grid:")
    print("       CRS:", ref_profile["crs"])
    print("       shape:", (ref_profile["height"], ref_profile["width"]))
    print("       transform:", ref_profile["transform"])

    out: Dict[int, Path] = {}
    for rp in rps:
        zip_path = DEPTH_RAW_DIR / f"floodMapGL_rp{rp}y.zip"
        _download(JRC_ZIP_URL_TEMPLATE.format(rp=rp), zip_path)
        tifs = _extract_zip(zip_path, DEPTH_EXTRACT_DIR)

        for tif in tifs:
            out_depth = DEPTH_ALIGNED_DIR / f"JRC_flood_depth_rp{rp}y__matchLULC.tif"
            if raster_ok(out_depth):
                print(f"[SKIP] aligned depth exists: {out_depth.name}")
            else:
                print(f"[WARP] RP{rp}: {tif.name} -> {out_depth.name}")
                _warp_to_lulc(tif, out_depth, ref_profile)
                print(f"[OK]   wrote: {out_depth}")

            out_mask = DEPTH_MASK_DIR / f"JRC_flood_mask_rp{rp}y__matchLULC.tif"
            if raster_ok(out_mask):
                print(f"[SKIP] mask exists: {out_mask.name}")
            else:
                _write_mask_from_depth(out_depth, out_mask, threshold=0.0)
                print(f"[OK]   wrote: {out_mask}")

            out[rp] = out_depth
    return out


def write_lulc_to_sda_mapping() -> Path:
    """
    Write the JRC-INCA style LULC -> SDA mapping JSON, then QA it against the
    codes actually present in the LULC raster.

    Everything in ESA_CODEBOOK that is not artif/crop/pasture is written to
    'ignore' explicitly, so a reviewer can confirm every code was classified
    on purpose rather than by falling through to 0.
    """
    built_up = list(ESA_TO_SDA.get("artif", []))
    cropland = list(ESA_TO_SDA.get("crop", []))
    pasture = list(ESA_TO_SDA.get("pasture", [])) if INCLUDE_PASTURE else []

    assigned = set(built_up) | set(cropland) | set(pasture)
    ignore = sorted(c for c in ESA_CODEBOOK if c not in assigned and c != 0)

    mapping = {
        # REQUIRED by the Section B SDA builder
        "artif": built_up,
        "crop": cropland,
        "pasture": pasture,
        "ignore": ignore,
        # Aliases (clarity / backward compatibility with the original scripts)
        "built_up": built_up,
        "cropland": cropland,
    }

    SDA_MAPPING_JSON.parent.mkdir(parents=True, exist_ok=True)
    SDA_MAPPING_JSON.write_text(json.dumps(mapping, indent=2))
    print(f"[OK] Wrote SDA mapping JSON -> {SDA_MAPPING_JSON}")

    codes = _sample_unique_lulc_codes(LULC_PATH)
    print(f"[INFO] Unique LULC codes sampled: {len(codes)} -> {codes[:20]}")

    all_codes = set(codes)
    for key in ("artif", "crop", "pasture", "ignore"):
        missing = [c for c in mapping[key] if c not in all_codes]
        if missing:
            print(f"[WARN] Mapping key '{key}' has codes not seen in raster: {missing}")

    categorized = set().union(*(set(v) for v in mapping.values()))
    uncategorized = sorted(c for c in all_codes if c not in categorized)
    if uncategorized:
        print("[WARN] Raster contains codes not explicitly categorized:")
        print(f"       {uncategorized}")
        print("       These will be treated as NON-SDA (sda_class = 0).")

    return SDA_MAPPING_JSON


def _sample_unique_lulc_codes(lulc_path: Path, n_windows: int = 60, win_size: int = 1024,
                              full_scan: bool = False, rng_seed: int = 42) -> List[int]:
    codes = set()
    with rasterio.open(lulc_path) as src:
        nodata = src.nodata
        if full_scan:
            print("[INFO] Performing FULL raster scan for unique codes...")
            for _, window in src.block_windows(1):
                arr = src.read(1, window=window)
                if nodata is not None:
                    arr = arr[arr != nodata]
                if arr.size:
                    codes.update(np.unique(arr).tolist())
        else:
            print("[INFO] Sampling windows for unique codes...")
            for window in random_windows(src.width, src.height, n_windows, win_size, seed=rng_seed):
                arr = src.read(1, window=window)
                if nodata is not None:
                    arr = arr[arr != nodata]
                if arr.size:
                    codes.update(np.unique(arr).tolist())
    return sorted(int(c) for c in codes)


def build_global_sda_raster() -> Path:
    """
    Build the global categorical SDA raster from ESA CCI land cover,
    block-wise so the full-resolution global grid never lands in RAM.

    Assignment is priority-ordered (pasture < crop < artif) so that a code
    appearing in more than one list resolves deterministically.
    """
    assert_exists(LULC_PATH, "LULC raster is required to build the SDA raster.")

    artif_set = set(ESA_TO_SDA.get("artif", []))
    crop_set = set(ESA_TO_SDA.get("crop", []))
    pasture_set = set(ESA_TO_SDA.get("pasture", [])) if INCLUDE_PASTURE else set()

    GLOBAL_SDA_TIF.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(LULC_PATH) as src:
        nodata = src.nodata
        profile = src.profile.copy()
        profile.update(dtype=rasterio.uint8, count=1, nodata=SDA_CODE["none"],
                       compress="deflate", tiled=True, BIGTIFF="IF_SAFER")

        tmp = GLOBAL_SDA_TIF.with_suffix(".tif.tmp")
        with rasterio.open(tmp, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                lulc = src.read(1, window=window)
                sda = np.zeros(lulc.shape, dtype=np.uint8)

                valid = np.ones(lulc.shape, dtype=bool)
                if nodata is not None:
                    valid &= (lulc != nodata)

                if pasture_set:  # lowest priority
                    sda[valid & np.isin(lulc, list(pasture_set))] = SDA_CODE["pasture"]
                if crop_set:
                    sda[valid & np.isin(lulc, list(crop_set))] = SDA_CODE["crop"]
                if artif_set:  # highest priority
                    sda[valid & np.isin(lulc, list(artif_set))] = SDA_CODE["artif"]

                dst.write(sda, 1, window=window)
        tmp.replace(GLOBAL_SDA_TIF)

    print("[OK] SDA raster written:", GLOBAL_SDA_TIF)
    write_sda_legend_csv()
    _report_sda_histogram(GLOBAL_SDA_TIF)
    return GLOBAL_SDA_TIF


def write_sda_legend_csv() -> Path:
    """Complete ESA code legend: lucode, label, sda_type, sda_code, rule used."""
    esa_to_type = {k: "none" for k in ESA_CODEBOOK}
    for k in ESA_TO_SDA.get("artif", []):
        esa_to_type[k] = "artif"
    for k in ESA_TO_SDA.get("crop", []):
        esa_to_type[k] = "crop"
    if INCLUDE_PASTURE:
        for k in ESA_TO_SDA.get("pasture", []):
            if esa_to_type.get(k, "none") == "none":
                esa_to_type[k] = "pasture"

    rule = (f"artif={ESA_TO_SDA.get('artif')}; crop={ESA_TO_SDA.get('crop')}; "
            f"pasture={ESA_TO_SDA.get('pasture')} (include={INCLUDE_PASTURE})")

    rows = []
    for lucode in sorted(ESA_CODEBOOK):
        sda_type = esa_to_type.get(lucode, "none")
        rows.append({
            "lucode": lucode,
            "label": ESA_CODEBOOK[lucode],
            "sda_type": sda_type,
            "sda_code": SDA_CODE[sda_type],
            "notes": rule if sda_type != "none" else "ignored (non-SDA)",
        })
    write_csv(pd.DataFrame(rows), GLOBAL_SDA_LEGEND_CSV)
    print("[OK] Legend written:", GLOBAL_SDA_LEGEND_CSV)
    return GLOBAL_SDA_LEGEND_CSV


def _report_sda_histogram(path: Path):
    counts: Dict[int, int] = {}
    with rasterio.open(path) as ds:
        for _, window in ds.block_windows(1):
            arr = ds.read(1, window=window)
            vals, cts = np.unique(arr, return_counts=True)
            for v, c in zip(vals.tolist(), cts.tolist()):
                counts[v] = counts.get(v, 0) + c
        total = sum(counts.values())
        print("\n[QA] SDA pixel distribution:")
        for k in sorted(counts):
            print(f"  SDA={k}: {counts[k]:,}  ({counts[k] / total if total else 0:.3%})")
        print("\n[QA] Raster metadata:")
        print(raster_profile_string(ds))
    return counts


def qa_spa_raster(sample_windows: int = 80, window_size: int = 1024) -> Path:
    """
    Validate the global SPA raster (produced upstream of this pipeline from
    runoff-retention potential) and write reproducible QA outputs:
    a quicklook PNG, a per-country SPA area CSV, and an alignment report.

    This does NOT recompute SPA -- it checks the raster is usable by Section C.
    """
    from rasterio.features import rasterize

    assert_exists(SPA_PATH, "SPA raster is required.")
    QA_DIR.mkdir(parents=True, exist_ok=True)

    report = ["=== SPA QA REPORT (Section A) ===\n", f"SPA: {SPA_PATH}\n\n"]

    with rasterio.open(SPA_PATH) as ds:
        report += ["=== SPA METADATA ===\n", raster_profile_string(ds) + "\n"]
        nodata = ds.nodata
        if nodata is None:
            report.append("[WARN] SPA raster has no nodata defined.\n")

        vals, finite_px, one_px = [], 0, 0
        for win in random_windows(ds.width, ds.height, sample_windows, window_size):
            a = ds.read(1, window=win)
            m = (a != nodata) if nodata is not None else np.isfinite(a)
            finite_px += int(m.sum())
            if m.any():
                v = a[m]
                vals.append(v)
                one_px += int((v == 1).sum())

        if vals:
            unique = np.unique(np.concatenate(vals))
            report.append(f"Sampled unique (excluding nodata): {unique[:50].tolist()}\n")
            if len(unique) <= 3 and set(unique.tolist()).issubset({0, 1}):
                report.append("[OK] Sample suggests binary SPA (0/1).\n")
            else:
                report.append("[WARN] SPA is not strictly binary (0/1). Investigate.\n")
            report.append(f"Sample SPA fraction (proxy) = {one_px / max(1, finite_px):.6f}\n\n")
        else:
            report.append("[WARN] No valid pixels sampled.\n\n")

        # Quicklook (decimated read -- never load the global grid at full res)
        step = max(1, int(max(ds.width, ds.height) / 2500))
        a = ds.read(1, out_shape=(int(ds.height / step), int(ds.width / step)))
        if nodata is not None:
            a = np.where(a == nodata, np.nan, a)
        plt.figure()
        plt.imshow(a, interpolation="nearest")
        plt.title("global_prr_spa (quicklook)")
        plt.axis("off")
        out_png = QA_DIR / "global_prr_spa_quicklook.png"
        savefig(out_png, dpi=200)
        report.append(f"[OK] Wrote quicklook PNG: {out_png}\n\n")

        # Per-country SPA area, computed window-by-window off the country bbox
        admin0 = load_admin0(ADMIN0_PATH)[["iso3", "geometry"]].to_crs(ds.crs)
        pix_m2 = pixel_area_m2(ds.transform)

        rows = []
        for _, r in admin0.iterrows():
            geom = r["geometry"]
            if geom is None or geom.is_empty:
                continue
            minx, miny, maxx, maxy = geom.bounds
            row_min, col_min = ds.index(minx, maxy)
            row_max, col_max = ds.index(maxx, miny)
            row0, row1 = max(0, min(row_min, row_max)), min(ds.height, max(row_min, row_max))
            col0, col1 = max(0, min(col_min, col_max)), min(ds.width, max(col_min, col_max))
            if row1 <= row0 or col1 <= col0:
                continue

            win = rasterio.windows.Window(col0, row0, col1 - col0, row1 - row0)
            spa = ds.read(1, window=win)
            valid = (spa != nodata) if nodata is not None else np.isfinite(spa)

            mask = rasterize(
                [(geom, 1)], out_shape=(int(win.height), int(win.width)),
                transform=ds.window_transform(win), fill=0, dtype="uint8",
                all_touched=False,
            ).astype(bool)

            inside = valid & mask
            if inside.sum() == 0:
                rows.append({"iso3": r["iso3"], "spa_area_km2": 0.0, "spa_frac_in_country": 0.0})
                continue
            spa1 = (spa == 1) & inside
            rows.append({
                "iso3": r["iso3"],
                "spa_area_km2": float(spa1.sum() * pix_m2 / 1e6),
                "spa_frac_in_country": float(spa1.sum() / inside.sum()),
            })

        out_csv = QA_DIR / "global_spa_country_summary.csv"
        write_csv(pd.DataFrame(rows).sort_values("iso3"), out_csv)
        report.append(f"[OK] Wrote country SPA summary: {out_csv}\n\n")

    # Alignment checks against the grids SPA has to line up with
    for label, path in (("LULC", LULC_PATH),
                        ("DEPTH", DEPTH_ALIGNED_DIR / f"JRC_flood_depth_rp{RETURN_PERIODS[-1]}y__matchLULC.tif")):
        if Path(path).exists():
            with rasterio.open(path) as x:
                report += [f"=== ALIGNMENT CHECK: {label} ===\n", raster_profile_string(x) + "\n"]
        else:
            report.append(f"[WARN] Alignment check missing {label}: {path}\n")

    out_txt = QA_DIR / "global_spa_alignment_report.txt"
    out_txt.write_text("".join(report))
    print(f"[OK] Wrote report: {out_txt}")
    return out_txt


def prepare_all_inputs(skip_download: bool = False) -> dict:
    """Section A driver: everything the accounting steps need on disk."""
    results = {}
    if not skip_download:
        results["depth_rasters"] = download_and_align_jrc_depth()
    results["sda_mapping_json"] = write_lulc_to_sda_mapping()
    if not raster_ok(GLOBAL_SDA_TIF):
        results["global_sda_raster"] = build_global_sda_raster()
    else:
        print(f"[SKIP] Global SDA raster already exists: {GLOBAL_SDA_TIF}")
        results["global_sda_raster"] = GLOBAL_SDA_TIF
    if Path(SPA_PATH).exists():
        results["spa_qa_report"] = qa_spa_raster()
    else:
        warnings.warn(f"[WARN] SPA raster not found, skipping QA: {SPA_PATH}")
    return results


SDA_SCRIPT_PATH = Path(__file__).parent / "scripts" / "sda_step2_build_sda_global.py"


SDA_ISO3_LIST = ""          # blank = all countries in Admin0


SDA_START = 0


SDA_N = 0                   # 0 = all remaining


SDA_SKIP_DONE = True


SDA_ALL_TOUCHED = False


SDA_USE_ROADS = False


SDA_ROADS_PATH = None


SDA_WITH_POP = False


SDA_POP_PATH = None


SDA_WRITE_DEPTHBIN = False


SDA_DEPTHBIN_MAX = 6.0


def configure_sda(p):
    """
    Override the Section-B constants from the ProjectFlow object.
    Called by flood_tasks.task_build_sda().
    """
    configure_inputs(p)  # Section B consumes Section A's paths

    global SDA_SCRIPT_PATH, SDA_ISO3_LIST, SDA_START, SDA_N, SDA_SKIP_DONE
    global SDA_ALL_TOUCHED, SDA_USE_ROADS, SDA_ROADS_PATH, SDA_WITH_POP
    global SDA_POP_PATH, SDA_WRITE_DEPTHBIN, SDA_DEPTHBIN_MAX

    SDA_SCRIPT_PATH = Path(getattr(p, 'flood_sda_script_path', SDA_SCRIPT_PATH))
    SDA_ISO3_LIST = getattr(p, 'flood_iso3_list', "")
    SDA_START = getattr(p, 'flood_iso3_start', 0)
    SDA_N = getattr(p, 'flood_iso3_n', 0)
    SDA_SKIP_DONE = getattr(p, 'flood_skip_done', True)
    SDA_ALL_TOUCHED = getattr(p, 'flood_all_touched', False)

    SDA_USE_ROADS = getattr(p, 'flood_use_roads', False)
    SDA_ROADS_PATH = Path(getattr(p, 'flood_roads_path', INPUTS / "roads" / "roads_mask_match_depth.tif"))
    SDA_WITH_POP = getattr(p, 'flood_with_pop', False)
    SDA_POP_PATH = Path(getattr(p, 'flood_pop_path', INPUTS / "pop" / "GlobPOP_Count_30arc_2020_I32.tif"))
    SDA_WRITE_DEPTHBIN = getattr(p, 'flood_write_depthbin', False)
    SDA_DEPTHBIN_MAX = getattr(p, 'flood_depthbin_max', 6.0)


def build_sda_global():
    """Section B driver: build per-country, per-RP SDA rasters."""
    argv = [
        "--mapping-json", SDA_MAPPING_JSON,
        "--lulc-path", LULC_PATH,
        "--depth-dir", DEPTH_ALIGNED_DIR,
        "--rps", ",".join(str(r) for r in RETURN_PERIODS),
        "--depth-threshold", DEPTH_THRESHOLD_M,
        "--start", SDA_START,
        "--n", SDA_N,
    ]
    if SDA_ISO3_LIST:
        argv += ["--iso3", SDA_ISO3_LIST]
    if SDA_SKIP_DONE:
        argv += ["--skip-done"]
    if SDA_ALL_TOUCHED:
        argv += ["--all-touched"]
    if INCLUDE_PASTURE:
        argv += ["--include-pasture"]
    if SDA_USE_ROADS:
        argv += ["--use-roads", "--roads-path", SDA_ROADS_PATH]
    if SDA_WITH_POP:
        argv += ["--with-pop", "--pop-path", SDA_POP_PATH]
    if SDA_WRITE_DEPTHBIN:
        argv += ["--write-depthbin", "--depthbin-max", SDA_DEPTHBIN_MAX]

    return _invoke_script(SDA_SCRIPT_PATH, argv, label="Step 2 SDA builder")


FLOW_SKIP_DONE = True


FLOW_ALL_TOUCHED = False


FLOW_WRITE_RASTERS = True


FLOW_INCLUDE_EXISTING = True   # include already-computed outputs in the summary


FLOW_SUMMARY_CSV = None


def configure_service_flow(p):
    """
    Override the Section-C constants from the ProjectFlow object.
    Called by flood_tasks.task_compute_service_flow().
    """
    configure_inputs(p)

    global FLOW_SKIP_DONE, FLOW_ALL_TOUCHED, FLOW_WRITE_RASTERS
    global FLOW_INCLUDE_EXISTING, FLOW_SUMMARY_CSV

    FLOW_SKIP_DONE = getattr(p, 'flood_skip_done', True)
    FLOW_ALL_TOUCHED = getattr(p, 'flood_all_touched', False)
    FLOW_WRITE_RASTERS = getattr(p, 'flood_write_service_flow_rasters', True)
    FLOW_INCLUDE_EXISTING = getattr(p, 'flood_include_existing_in_summary', True)
    FLOW_SUMMARY_CSV = Path(getattr(p, 'flood_service_flow_summary_csv',
                                    OUTPUTS / "global_service_flow_spa_to_sda.csv"))


def _service_flow_stats(iso3: str, rp: int, sda_class_file: Path,
                        sda_mask_file: Path, flow_file: Path) -> dict:
    """
    Read stats back off the written rasters. Doing it this way (rather than
    from the in-memory arrays) means --skip-done runs still populate the global
    summary CSV, instead of reporting "no outputs produced" as older versions did.
    """
    with rasterio.open(sda_class_file) as csrc, \
         rasterio.open(sda_mask_file) as msrc, \
         rasterio.open(flow_file) as fsrc:

        sda_class = csrc.read(1)
        sda_mask = (msrc.read(1) == 1)

        flow = fsrc.read(1)
        if fsrc.nodata is not None:
            flow = np.where(flow == fsrc.nodata, np.nan, flow)

        # Same Web Mercator area inflation as Step 4B -- these are clipped
        # windows, so the row offset relative to the full grid is unknown here.
        # Approximate with the window's own latitude band, which is exact enough
        # for reported areas (they are diagnostics, not money).
        pix_km2 = pixel_area_km2(fsrc.transform)
        if VAL_LATITUDE_CORRECT_AREA:
            scale = mercator_area_scale(fsrc.transform, 0, fsrc.height)
            area_km2 = pix_km2 * np.broadcast_to(scale, sda_mask.shape)
        else:
            area_km2 = np.full(sda_mask.shape, pix_km2, dtype="float32")
        served = sda_mask & (flow > 0)

        def class_stats(cls: int) -> Tuple[float, float]:
            m = (sda_class == cls) & sda_mask
            return float(area_km2[m].sum()), safe_mean(flow[m])

        artif_area, artif_mean = class_stats(SDA_CODE["artif"])
        crop_area, crop_mean = class_stats(SDA_CODE["crop"])
        past_area, past_mean = class_stats(SDA_CODE["pasture"])
        road_area, road_mean = class_stats(SDA_CODE["roads"])

        return {
            "iso3": iso3,
            "rp": rp,
            "sda_area_km2_total": float(area_km2[sda_mask].sum()),
            "sda_area_km2_served": float(area_km2[served].sum()),
            "mean_spa_ratio_on_sda": safe_mean(flow[sda_mask]),
            "sda_area_km2_artif": artif_area, "mean_spa_ratio_artif": artif_mean,
            "sda_area_km2_crop": crop_area, "mean_spa_ratio_crop": crop_mean,
            "sda_area_km2_pasture": past_area, "mean_spa_ratio_pasture": past_mean,
            "sda_area_km2_roads": road_area, "mean_spa_ratio_roads": road_mean,
        }


def _service_flow_one_iso3(iso3: str, admin0: gpd.GeoDataFrame, spa_src) -> Tuple[list, int, int]:
    rows: list = []
    iso_dir = OUTPUTS / iso3
    if not iso_dir.exists():
        return rows, 0, 0

    sda_files = sorted(iso_dir.glob(f"sda_class_{iso3}_rp*.tif"))
    if not sda_files:
        return rows, 0, 0

    processed = skipped = 0
    geom = admin0.loc[admin0.iso3 == iso3].geometry.values

    for cls_file in sda_files:
        rp = int(cls_file.stem.split("rp")[-1])
        mask_file = iso_dir / f"sda_mask_{iso3}_rp{rp}.tif"
        flow_file = iso_dir / f"service_flow_frac_{iso3}_rp{rp}.tif"

        if not mask_file.exists():
            warnings.warn(f"[WARN] {iso3} RP{rp}: missing SDA mask, skipping.")
            continue

        if FLOW_SKIP_DONE and raster_ok(flow_file):
            skipped += 1
            if FLOW_INCLUDE_EXISTING:
                rows.append(_service_flow_stats(iso3, rp, cls_file, mask_file, flow_file))
            continue

        with rasterio.open(cls_file) as sda_src:
            arr, tr = rio_mask(sda_src, geom, crop=True, filled=True,
                               nodata=0, all_touched=FLOW_ALL_TOUCHED)
            sda_class = arr[0].astype(np.uint8)

        with rasterio.open(mask_file) as msrc:
            arr, _ = rio_mask(msrc, geom, crop=True, filled=True,
                              nodata=0, all_touched=FLOW_ALL_TOUCHED)
            sda_mask = (arr[0] == 1)

        # Reproject SPA ratio ONTO the SDA grid (never the reverse).
        spa_on_sda = np.full(sda_class.shape, np.nan, dtype=np.float32)
        reproject(
            rasterio.band(spa_src, 1), spa_on_sda,
            src_transform=spa_src.transform, src_crs=spa_src.crs,
            dst_transform=tr, dst_crs=spa_src.crs,
            resampling=Resampling.average,
            src_nodata=spa_src.nodata, dst_nodata=np.nan,
        )

        spa_ratio = _normalize_spa_ratio(spa_on_sda, spa_src.nodata)
        flow = np.where(sda_mask, spa_ratio, 0.0)

        if FLOW_WRITE_RASTERS:
            prof = spa_src.profile.copy()
            prof.update(height=flow.shape[0], width=flow.shape[1], transform=tr,
                        nodata=-9999, dtype="float32", count=1)
            atomic_write_raster(flow_file, prof,
                                np.where(np.isfinite(flow), flow, -9999).astype("float32"))

        rows.append(_service_flow_stats(iso3, rp, cls_file, mask_file, flow_file))
        processed += 1

    return rows, processed, skipped


def compute_service_flow_global(iso3_list: Optional[List[str]] = None) -> Path:
    """Section C driver: SPA -> SDA service flow fraction for every ISO3 x RP."""
    assert_exists(SPA_RATIO_PATH, "Upstream SPA ratio raster is required for Section C.")
    admin0 = load_admin0(ADMIN0_PATH)

    all_rows, total_p, total_s = [], 0, 0
    run_list = iso3_list if iso3_list else sorted(admin0.iso3.unique())

    with rasterio.open(SPA_RATIO_PATH) as spa_src:
        for iso in run_list:
            rows, processed, skipped = _service_flow_one_iso3(iso, admin0, spa_src)
            all_rows.extend(rows)
            total_p += processed
            total_s += skipped

    if all_rows:
        write_csv(pd.DataFrame(all_rows), FLOW_SUMMARY_CSV)
        print(f"[DONE] Global service-flow summary -> {FLOW_SUMMARY_CSV}")
        print(f"[INFO] processed={total_p}, skipped_existing={total_s}")
    else:
        print("[INFO] Nothing to do (no SDA outputs found under "
              f"{OUTPUTS}; run Section B first).")
    return FLOW_SUMMARY_CSV


VAL_DAMAGE_TABLE_SCRIPT = Path(__file__).parent / "scripts" / "build_damage_table_USD2019.py"


VAL_STEP4D_SCRIPT = Path(__file__).parent / "scripts" / "flood_gep_step4d_export_global_USD2019.py"


VAL_DAMAGE_DIR = None


VAL_CANONICAL_EUR_CSV = None


VAL_FACTORS_CSV = None


VAL_DAMAGE_LONG_CSV = None


VAL_DAMAGE_WIDE_CSV = None


VAL_SDA_DAMAGE_LONG_CSV = None


VAL_SDA_DAMAGE_WIDE_CSV = None


VAL_SET_PASTURE_EQUAL_CROP = True


SDA_CODE_TO_TYPE = {1: "artif", 2: "crop", 3: "pasture"}


DEPTH_BINS_M = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6]


VAL_LATITUDE_CORRECT_AREA = True


VAL_WRITE_DAMAGE_RASTERS = True


VAL_TAIL_MODE = "flat"          # "flat" | "zero" behaviour for p -> 0


VAL_ADD_P1_ZERO = False          # anchor (p=1, D=0)


VAL_ENFORCE_MONOTONE = False


VAL_WRITE_INTEGRATION_POINTS = True


VAL_EXPORT_DIR = None


VAL_REGION_COL = "region_wb"


VAL_FILL_MISSING_ZERO = True


VAL_COMPUTE_EAD_RASTERS = False


VAL_MOSAIC_GLOBAL_RASTER = False


VAL_APPLY_SERVICE_FLOW = False


VAL_SERVICE_FLOW_CSV = None


SCENARIOS = ("current", "degraded_insitu", "degraded_bare")


SCENARIO_SUFFIX = {"current": "",
                   "degraded_insitu": "__degraded_insitu",
                   "degraded_bare": "__degraded_bare"}


VAL_AMPLIFICATION_DIR = None


VAL_AMPLIFICATION_PATTERN = "amplification_{scenario}_rp{rp}.tif"


VAL_CN_TABLE = None


VAL_GEP_CSV = None


VAL_PROTECTION_CSV = None          # CSV with iso3, protection_rp (from FLOPROS)


VAL_REPORT_PROTECTION_SPLIT = False


FLOPROS_DOCUMENTED_ISO3 = {
    "ARG", "AUS", "AUT", "BEL", "BGD", "BLZ", "BRA", "CAN", "CHE", "CHN",
    "CZE", "DEU", "DNK", "ESP", "GBR", "GHA", "HRV", "HUN", "IDN", "IND",
    "IRL", "ITA", "JPN", "MDG", "MOZ", "NLD", "NZL", "POL", "ROU", "RUS",
    "SGP", "SVK", "THA", "TWN", "USA", "VNM", "ZAF",
}


VAL_PROTECTION_DOCUMENTED_ONLY = True


VAL_PROTECTION_EVIDENCE_CSV = None   # overrides the embedded set if supplied


def configure_valuation(p):
    """
    Override the Section-D constants from the ProjectFlow object.
    Called by flood_tasks.task_compute_flood_damages().
    """
    configure_inputs(p)

    global VAL_DAMAGE_TABLE_SCRIPT, VAL_STEP4D_SCRIPT, VAL_DAMAGE_DIR
    global VAL_CANONICAL_EUR_CSV, VAL_FACTORS_CSV
    global VAL_DAMAGE_LONG_CSV, VAL_DAMAGE_WIDE_CSV
    global VAL_SDA_DAMAGE_LONG_CSV, VAL_SDA_DAMAGE_WIDE_CSV
    global VAL_SET_PASTURE_EQUAL_CROP, SDA_CODE_TO_TYPE, DEPTH_BINS_M
    global VAL_LATITUDE_CORRECT_AREA
    global VAL_WRITE_DAMAGE_RASTERS, VAL_TAIL_MODE, VAL_ADD_P1_ZERO
    global VAL_ENFORCE_MONOTONE, VAL_WRITE_INTEGRATION_POINTS
    global VAL_EXPORT_DIR, VAL_REGION_COL, VAL_FILL_MISSING_ZERO
    global VAL_COMPUTE_EAD_RASTERS, VAL_MOSAIC_GLOBAL_RASTER
    global VAL_APPLY_SERVICE_FLOW, VAL_SERVICE_FLOW_CSV
    global VAL_AMPLIFICATION_DIR, VAL_AMPLIFICATION_PATTERN, VAL_CN_TABLE
    global VAL_GEP_CSV, VAL_PROTECTION_CSV, VAL_REPORT_PROTECTION_SPLIT
    global VAL_PROTECTION_DOCUMENTED_ONLY, VAL_PROTECTION_EVIDENCE_CSV
    global FLOPROS_DOCUMENTED_ISO3

    VAL_DAMAGE_TABLE_SCRIPT = Path(getattr(p, 'flood_damage_table_script_path', VAL_DAMAGE_TABLE_SCRIPT))
    VAL_STEP4D_SCRIPT = Path(getattr(p, 'flood_step4d_script_path', VAL_STEP4D_SCRIPT))

    VAL_DAMAGE_DIR = Path(getattr(p, 'flood_damage_dir', INPUTS / "flood_damage"))
    VAL_CANONICAL_EUR_CSV = Path(getattr(p, 'flood_canonical_eur_csv',
                                         VAL_DAMAGE_DIR / "country_landtype_flood_damage_JRC_EUR_m2.csv"))
    VAL_FACTORS_CSV = Path(getattr(p, 'flood_currency_factors_csv',
                                   VAL_DAMAGE_DIR / "_currency_audit" /
                                   "currency_conversion_factors_EUR2010_to_USD2019.csv"))
    VAL_DAMAGE_LONG_CSV = Path(getattr(p, 'flood_damage_long_csv',
                                       VAL_DAMAGE_DIR / "damage_functions_depth_USD2019_long.csv"))
    VAL_DAMAGE_WIDE_CSV = Path(getattr(p, 'flood_damage_wide_csv',
                                       VAL_DAMAGE_DIR / "damage_functions_depth_USD2019_wide.csv"))
    VAL_SDA_DAMAGE_LONG_CSV = Path(getattr(p, 'flood_sda_damage_long_csv',
                                           VAL_DAMAGE_DIR / "damage_functions_sda_depth_USD2019_long.csv"))
    VAL_SDA_DAMAGE_WIDE_CSV = Path(getattr(p, 'flood_sda_damage_wide_csv',
                                           VAL_DAMAGE_DIR / "damage_functions_sda_depth_USD2019_wide.csv"))
    VAL_SET_PASTURE_EQUAL_CROP = getattr(p, 'flood_set_pasture_equal_crop', True)

    SDA_CODE_TO_TYPE = dict(getattr(p, 'flood_sda_code_to_type', SDA_CODE_TO_TYPE))
    if getattr(p, 'flood_use_roads', False) and 4 not in SDA_CODE_TO_TYPE:
        SDA_CODE_TO_TYPE[4] = "roads"
    DEPTH_BINS_M = list(getattr(p, 'flood_depth_bins_m', DEPTH_BINS_M))

    VAL_LATITUDE_CORRECT_AREA = getattr(p, 'flood_latitude_correct_area', True)
    VAL_WRITE_DAMAGE_RASTERS = getattr(p, 'flood_write_damage_rasters', True)
    VAL_TAIL_MODE = getattr(p, 'flood_ead_tail_mode', "flat")
    globals()['DAMAGE_DEPTH_MODE'] = getattr(p, 'flood_damage_depth_mode', "interpolated")
    VAL_ADD_P1_ZERO = getattr(p, 'flood_ead_add_p1_zero', False)
    VAL_ENFORCE_MONOTONE = getattr(p, 'flood_ead_enforce_monotone', False)
    VAL_WRITE_INTEGRATION_POINTS = getattr(p, 'flood_ead_write_points', True)

    VAL_EXPORT_DIR = Path(getattr(p, 'flood_global_export_dir', OUTPUTS / "_global"))
    VAL_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    VAL_REGION_COL = getattr(p, 'flood_region_col', "region_wb")
    VAL_FILL_MISSING_ZERO = getattr(p, 'flood_fill_missing_zero', True)
    VAL_COMPUTE_EAD_RASTERS = getattr(p, 'flood_compute_ead_rasters', False)
    VAL_MOSAIC_GLOBAL_RASTER = getattr(p, 'flood_mosaic_global_raster', False)

    VAL_APPLY_SERVICE_FLOW = getattr(p, 'flood_apply_service_flow', False)
    VAL_SERVICE_FLOW_CSV = Path(getattr(p, 'flood_service_flow_summary_csv',
                                        OUTPUTS / "global_service_flow_spa_to_sda.csv"))

    VAL_AMPLIFICATION_DIR = Path(getattr(p, 'flood_amplification_dir',
                                         INPUTS / "counterfactual"))
    VAL_AMPLIFICATION_PATTERN = getattr(p, 'flood_amplification_pattern',
                                        "amplification_{scenario}_rp{rp}.tif")
    VAL_CN_TABLE = Path(getattr(p, 'flood_cn_table',
                                VAL_AMPLIFICATION_DIR / "esa_cci_CN_three_scenarios.csv"))
    VAL_GEP_CSV = Path(getattr(p, 'flood_gep_csv',
                               VAL_EXPORT_DIR / "step4e_flood_gep_USD2019.csv"))
    _prot = getattr(p, 'flood_protection_csv', None)
    VAL_PROTECTION_CSV = Path(_prot) if _prot else None
    VAL_REPORT_PROTECTION_SPLIT = getattr(p, 'flood_report_protection_split', False)
    VAL_PROTECTION_DOCUMENTED_ONLY = getattr(
        p, 'flood_protection_documented_only', True)
    _ev = getattr(p, 'flood_protection_evidence_csv', None)
    VAL_PROTECTION_EVIDENCE_CSV = Path(_ev) if _ev else None
    if VAL_PROTECTION_EVIDENCE_CSV and VAL_PROTECTION_EVIDENCE_CSV.exists():
        # Which countries count as having documented protection decides how much damage is
        # truncated, so a file that cannot be read must stop the run rather than fall back.
        _e = pd.read_csv(VAL_PROTECTION_EVIDENCE_CSV)
        _c = find_col(_e, ("iso3", "iso_a3", "adm0_a3"))
        if not _c:
            raise NameError(
                'No ISO3 column in %s. Looked for iso3, iso_a3, adm0_a3; found %s. The documented '
                'protection set decides which countries have their damage truncated, so falling '
                'back to the embedded list would silently change the answer.'
                % (VAL_PROTECTION_EVIDENCE_CSV, list(_e.columns)))
        FLOPROS_DOCUMENTED_ISO3 = set(_e[_c].astype(str).str.upper().str.strip())
        print(f"[INFO] documented-protection set loaded from "
              f"{VAL_PROTECTION_EVIDENCE_CSV.name}: "
              f"{len(FLOPROS_DOCUMENTED_ISO3)} ISO3")


def build_damage_tables():
    """Convert the canonical JRC EUR2010 damage table into USD2019 sector and SDA tables."""
    argv = [
        "--canonical-eur", VAL_CANONICAL_EUR_CSV,
        "--factors-csv", VAL_FACTORS_CSV,
        "--out-long", VAL_DAMAGE_LONG_CSV,
        "--out-wide", VAL_DAMAGE_WIDE_CSV,
        "--out-sda-long", VAL_SDA_DAMAGE_LONG_CSV,
        "--out-sda-wide", VAL_SDA_DAMAGE_WIDE_CSV,
        "--audit-dir", VAL_DAMAGE_DIR / "_currency_audit",
    ]
    if VAL_SET_PASTURE_EQUAL_CROP:
        argv += ["--set-pasture-equal-crop"]
    return _invoke_script(VAL_DAMAGE_TABLE_SCRIPT, argv, label="Step 4A damage table builder")


def load_damage_table_wide(path: Path) -> Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]:
    """Load the wide SDA damage curves into {(iso3, sda_type): (depths, damages)}."""
    df = pd.read_csv(path)
    df["iso3"] = df["iso3"].astype(str).str.strip().str.upper()
    df["sda_type"] = df["sda_type"].astype(str).str.strip().str.lower()

    needed = ["iso3", "sda_type"] + [ff._fmt_depth_col(d) for d in DEPTH_BINS_M]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Damage table missing required columns: {missing}\n"
                         f"Columns={list(df.columns)}")

    xs = np.array(DEPTH_BINS_M, dtype="float32")
    curves = {}
    for _, r in df.iterrows():
        ys = np.array([float(r[ff._fmt_depth_col(d)]) for d in DEPTH_BINS_M], dtype="float32")
        curves[(r["iso3"], r["sda_type"])] = (xs, ys)
    return curves


INCA_DEPTH_BANDS = np.array([0.25, 0.5, 1.0, 1.5, 2.5, 3.5, 4.5, 5.5], dtype="float32")


def _find_depth_raster(rp: int) -> Optional[Path]:
    for pattern in (f"*rp{rp}y*matchLULC*.tif", f"*rp{rp}*matchLULC*.tif", f"*rp{rp}*.tif"):
        hits = sorted(Path(DEPTH_ALIGNED_DIR).glob(pattern))
        if hits:
            return hits[0]
    return None


def _open_amp(scenario: str, rp: int):
    """Open the amplification raster for one scenario x RP, or None."""
    if scenario == "current":
        return None
    path = Path(VAL_AMPLIFICATION_DIR) / VAL_AMPLIFICATION_PATTERN.format(
        scenario=scenario, rp=rp)
    if not path.exists():
        warnings.warn(
            f"[WARN] no amplification raster for {scenario} RP{rp} at {path}. "
            f"Build it with counterfactual/build_amplification_rasters.py. "
            f"Falling back to current depths (GEP for this RP will be zero).")
        return None
    return rasterio.open(path)


def compute_pixel_damages(iso3_list: Optional[List[str]] = None,
                          scenario: str = "current",
                          tile: int = 2048) -> List[Path]:
    """
    Step 4B: for each ISO3 x RP, interpolate the depth-damage curve at every
    flooded SDA pixel and sum to country totals in USD2019.

        damage_per_m2 = interp(curve[iso3, sda_type], depth_m)
        damage_usd    = damage_per_m2 * pixel_area_m2

    scenario "current"                            depths as delivered by JRC
    scenario "degraded_insitu" / "degraded_bare"  depths amplified by A(i); see
                                                  the SCENARIOS note above

    TILED READING -- WHY THIS FUNCTION LOOKS LIKE THIS
    -------------------------------------------------
    The original implementation clipped the whole country at once with
    rio_mask(crop=True). That allocates several arrays the size of the country's
    BOUNDING BOX, which is fine for compact countries and fatal for others.

    Norway's bounding box is 43,389 x 8,861 = 384 million pixels, because
    Svalbard stretches it to ~81 N -- almost all of that window is empty Arctic.
    Depth (float32) + SDA (int32) + damage (float32) + masks is 6-10 GB per
    return period. At an 8 GB allocation it was OOM-killed, no step4b CSV was
    written, Step 4C recorded status="missing_step4b" with ead=0, and Step 4D
    reported the country as zero. Silent, plausible, and wrong: France,
    Australia and Norway all appeared as legitimate zeros for this reason.

    Reading in tiles makes peak memory a function of `tile`, not of country
    extent -- roughly 80 MB at the 2048 default. Russia, Canada, the USA and
    every antimeridian country now run in the same footprint as Germany.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}")
    suffix = SCENARIO_SUFFIX[scenario]

    assert_exists(VAL_SDA_DAMAGE_WIDE_CSV, "Run step 4A (build_damage_tables) first.")
    assert_exists(GLOBAL_SDA_TIF, "Global SDA raster is required (Section A).")

    curves = load_damage_table_wide(VAL_SDA_DAMAGE_WIDE_CSV)
    admin0 = load_admin0(ADMIN0_PATH)

    with rasterio.open(GLOBAL_SDA_TIF) as sda_ds:
        warn_if_geographic(sda_ds, "SDA raster")
        pix_area_m2 = pixel_area_m2(sda_ds.transform)
        sda_meta = sda_ds.meta.copy()

    print(f"[INFO] SDA raster: {GLOBAL_SDA_TIF}")
    if VAL_LATITUDE_CORRECT_AREA:
        print(f"[INFO] Pixel area (m^2): {pix_area_m2:,.1f} at the equator, "
              f"scaled by cos^2(lat) per row")
    else:
        print(f"[INFO] Pixel area (m^2): {pix_area_m2:,.1f} UNCORRECTED "
              f"-- overstates area 4x at 60N")
    print(f"[INFO] RPs: {RETURN_PERIODS} | scenario: {scenario} | tile: {tile}")

    run_list = iso3_list if iso3_list else sorted(admin0.iso3.unique())
    written: List[Path] = []

    for iso3 in run_list:
        rows = admin0[admin0.iso3 == iso3]
        if rows.empty:
            continue
        geom = unary_union(rows.geometry.values)
        if len(rows) > 1:
            print(f"[INFO] {iso3}: unioned {len(rows)} Admin0 features.")

        out_dir = OUTPUTS / iso3
        out_dir.mkdir(parents=True, exist_ok=True)
        ras_dir = out_dir / "rasters"
        if VAL_WRITE_DAMAGE_RASTERS:
            ras_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for rp in RETURN_PERIODS:
            depth_path = _find_depth_raster(rp)
            if depth_path is None:
                print(f"[WARN] {iso3}: no depth raster for RP={rp} in {DEPTH_ALIGNED_DIR}")
                continue

            amp_src = _open_amp(scenario, rp)
            try:
                with rasterio.open(depth_path) as dds, rasterio.open(GLOBAL_SDA_TIF) as sds:
                    assert_same_grid(dds, sds, f"depth_rp{rp}", "sda")

                    win = _country_window(dds, geom)
                    if win is None:
                        print(f"[WARN] {iso3} RP{rp}: empty window in raster space")
                        continue
                    win_tr = window_transform(win, dds.transform)

                    dst = None
                    if VAL_WRITE_DAMAGE_RASTERS:
                        meta = sda_meta.copy()
                        meta.update(driver="GTiff", height=int(win.height),
                                    width=int(win.width), transform=win_tr,
                                    count=1, dtype="float32", nodata=0.0,
                                    compress="DEFLATE", tiled=True,
                                    BIGTIFF="IF_SAFER")
                        out_tif = ras_dir / f"damage_USD2019_rp{rp}{suffix}.tif"
                        tmp_tif = out_tif.with_suffix(".tif.tmp")
                        dst = rasterio.open(tmp_tif, "w", **meta)

                    totals_by_sda = {t: 0.0 for t in set(SDA_CODE_TO_TYPE.values())}
                    total_all = 0.0
                    n_tiles = n_active = 0

                    for sub in _iter_tiles(win, tile):
                        n_tiles += 1
                        sub_tr = window_transform(sub, dds.transform)

                        depth = dds.read(1, window=sub).astype("float32")
                        valid = np.isfinite(depth)
                        if dds.nodata is not None:
                            valid &= (depth != dds.nodata)
                        if not valid.any():
                            continue

                        # Restrict to the country. geometry_mask on the tile is
                        # cheap and replaces the whole-country clip.
                        inside = rio_features.geometry_mask(
                            [geom], out_shape=depth.shape, transform=sub_tr,
                            invert=True, all_touched=False)
                        valid &= inside
                        if not valid.any():
                            continue

                        if scenario != "current":
                            amp = _amp_tile(amp_src, depth.shape, sub_tr, dds.crs)
                            if amp is not None:
                                # Applied BEFORE thresholding, so pixels currently
                                # below the damage threshold can cross it. That
                                # newly-inundated margin is real avoided damage.
                                depth = np.where(valid, depth * amp, depth).astype("float32")

                        valid &= (depth > DEPTH_THRESHOLD_M)
                        if not valid.any():
                            continue

                        sda = sds.read(1, window=sub).astype("int16")
                        dmg_tile = np.zeros(depth.shape, dtype="float32") if dst else None
                        touched = False

                        # True ground area per pixel. Depends only on latitude,
                        # so one value per row broadcast across the tile.
                        if VAL_LATITUDE_CORRECT_AREA:
                            area = pix_area_m2 * np.broadcast_to(
                                mercator_area_scale(dds.transform,
                                                    int(sub.row_off),
                                                    int(sub.height)),
                                depth.shape)
                        else:
                            area = np.full(depth.shape, pix_area_m2, dtype="float32")

                        for code, sda_type in SDA_CODE_TO_TYPE.items():
                            m = valid & (sda == code)
                            if not m.any():
                                continue
                            key = (iso3, sda_type)
                            if key not in curves:
                                continue
                            xs, ys = curves[key]
                            dmg_usd = ff.interp_damage_per_m2(depth[m], xs, ys, mode=DAMAGE_DEPTH_MODE) * area[m]
                            totals_by_sda[sda_type] += float(dmg_usd.sum())
                            total_all += float(dmg_usd.sum())
                            if dmg_tile is not None:
                                dmg_tile[m] = dmg_usd
                            touched = True

                        if touched:
                            n_active += 1
                        if dst is not None and dmg_tile is not None:
                            rel = Window(sub.col_off - win.col_off,
                                         sub.row_off - win.row_off,
                                         sub.width, sub.height)
                            dst.write(dmg_tile, 1, window=rel)

                    if dst is not None:
                        dst.close()
                        tmp_tif.replace(out_tif)

                    missing = [t for (i3, t) in
                               [(iso3, t) for t in SDA_CODE_TO_TYPE.values()]
                               if (iso3, t) not in curves]
                    if missing:
                        print(f"[WARN] {iso3}: no damage curve for {sorted(set(missing))}; "
                              f"those classes contribute zero.")

                    rec = {"iso3": iso3, "rp": rp, "damage_total_usd2019": total_all}
                    rec.update({f"damage_{t}_usd2019": v
                                for t, v in totals_by_sda.items()})
                    records.append(rec)

                    print(f"[OK] {iso3} RP{rp} [{scenario}]: "
                          f"{total_all:,.2f} USD2019 "
                          f"| window {int(win.height)}x{int(win.width)} "
                          f"| tiles {n_active}/{n_tiles} active")
            finally:
                if amp_src is not None:
                    amp_src.close()

        if records:
            out_csv = out_dir / f"step4b_damage_by_rp_USD2019{suffix}.csv"
            rec_df = pd.DataFrame(records).sort_values(["iso3", "rp"])
            if VAL_APPLY_SERVICE_FLOW and scenario == "current":
                rec_df = ff._attach_service_flow(rec_df, iso3, _load_service_flow_table())
            write_csv(rec_df, out_csv)
            written.append(out_csv)
            print(f"[OK] Wrote summary -> {out_csv} (rows={len(rec_df)})")
        else:
            print(f"[WARN] {iso3}: no RP records written (missing depth rasters?)")

    return written


def _load_service_flow_table() -> Optional[pd.DataFrame]:
    """Read Section C's summary once, keyed by (iso3, rp)."""
    if not Path(VAL_SERVICE_FLOW_CSV).exists():
        warnings.warn(
            f"[WARN] flood_apply_service_flow is on but no service-flow summary "
            f"at {VAL_SERVICE_FLOW_CSV}. Run Section C first. Attributed damages "
            f"will be omitted.")
        return None
    flow = pd.read_csv(VAL_SERVICE_FLOW_CSV)
    needed = {"iso3", "rp", "mean_spa_ratio_on_sda"}
    if not needed.issubset(flow.columns):
        warnings.warn(f"[WARN] {VAL_SERVICE_FLOW_CSV} lacks {needed - set(flow.columns)}; "
                      "attributed damages will be omitted.")
        return None
    flow["iso3"] = flow["iso3"].astype(str).str.upper().str.strip()
    flow["rp"] = pd.to_numeric(flow["rp"], errors="coerce")
    return flow


def load_protection_table() -> Optional[pd.DataFrame]:
    """
    Read the FLOPROS-derived flood protection standards, one return period per
    ISO3. Expected columns: iso3, protection_rp.
    """
    if not VAL_PROTECTION_CSV or not Path(VAL_PROTECTION_CSV).exists():
        return None
    df = pd.read_csv(VAL_PROTECTION_CSV)
    c_iso = find_col(df, ("iso3", "iso_a3", "adm0_a3"))
    c_rp = find_col(df, ("protection_rp", "protection", "flopros", "merged_rp",
                         "protection_standard", "rp"))
    if c_iso is None or c_rp is None:
        warnings.warn(f"[WARN] {VAL_PROTECTION_CSV} needs iso3 + protection_rp; "
                      f"have {list(df.columns)}. Protection split skipped.")
        return None
    out = df[[c_iso, c_rp]].rename(columns={c_iso: "iso3", c_rp: "protection_rp"})
    out["iso3"] = out["iso3"].astype(str).str.upper().str.strip()
    out["protection_rp"] = pd.to_numeric(out["protection_rp"], errors="coerce")
    out = out.dropna(subset=["protection_rp"]).drop_duplicates("iso3")

    # Prefer an evidence column from prep_flopros.py; else use the embedded set.
    c_ev = find_col(df, ("protection_evidence", "evidence"))
    if c_ev:
        ev = df[[c_iso, c_ev]].rename(columns={c_iso: "iso3", c_ev: "protection_evidence"})
        ev["iso3"] = ev["iso3"].astype(str).str.upper().str.strip()
        out = out.merge(ev.drop_duplicates("iso3"), on="iso3", how="left")
    else:
        out["protection_evidence"] = np.where(
            out.iso3.isin(FLOPROS_DOCUMENTED_ISO3), "documented", "gdp_inferred")

    n_doc = int((out.protection_evidence == "documented").sum())
    print(f"[INFO] protection standards: {len(out)} countries "
          f"({n_doc} documented, {len(out)-n_doc} GDP-inferred)")

    if VAL_PROTECTION_DOCUMENTED_ONLY:
        # NaN means "no truncation applied" downstream -- the country is still
        # reported, just untruncated, with the reason recorded.
        mask = out.protection_evidence != "documented"
        out.loc[mask, "protection_rp"] = np.nan
        print(f"[INFO] truncation restricted to documented protection; "
              f"{int(mask.sum())} countries reported untruncated")
    return out


def _write_step4c(iso3_dir: Path, iso3: str, ead: float, *, rps_used: List[int],
                  status: str, detail: str, ead_attributed: float = np.nan,
                  suffix: str = "", ead_nc: float = np.nan,
                  protection_rp: Optional[float] = None,
                  protection_evidence: str = "") -> Path:
    out_csv = iso3_dir / f"step4c_ead_USD2019{suffix}.csv"
    write_csv(pd.DataFrame([{
        "iso3": iso3,
        "ead_usd2019": float(ead),
        # Attribution of residual damage to naturally-served floodplains.
        # NOT avoided damage -- see VAL_APPLY_SERVICE_FLOW note.
        "ead_attributed_to_spa_usd2019": float(ead_attributed),
        # Vallecillo Eq.7 split -- reported as sensitivity, not the default.
        # NC  = value from natural capital alone (events rarer than defences)
        # NC+ = natural capital supporting existing defences
        "ead_nc_usd2019": float(ead_nc),
        "ead_ncplus_usd2019": float(ead - ead_nc) if np.isfinite(ead_nc) else np.nan,
        "protection_rp": protection_rp if protection_rp is not None else np.nan,
        # "documented" -> NC/NC+ computed;  "gdp_inferred" -> reported untruncated
        "protection_evidence": protection_evidence,
        "rps_used": ",".join(map(str, rps_used)),
        "tail_mode": VAL_TAIL_MODE,
        "add_p1_zero": bool(VAL_ADD_P1_ZERO),
        "enforce_monotone": bool(VAL_ENFORCE_MONOTONE),
        "applied_service_flow": bool(VAL_APPLY_SERVICE_FLOW),
        "status": status,
        "detail": detail,
    }]), out_csv)
    return out_csv


def compute_ead_by_country(scenario: str = "current") -> pd.DataFrame:
    """
    Step 4C: convert Step 4B's damage-by-RP into an EAD for every ISO3 folder.

    A step4c CSV is written for EVERY ISO3 folder, even when Step 4B is
    missing or empty, because Step 4D aggregates across folders and a silently
    absent file is indistinguishable from a genuine zero.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}")
    suffix = SCENARIO_SUFFIX[scenario]

    prot_tbl = load_protection_table() if VAL_REPORT_PROTECTION_SPLIT else None
    if VAL_REPORT_PROTECTION_SPLIT and prot_tbl is None:
        warnings.warn("[WARN] protection split requested but no usable table; "
                      "NC/NC+ columns will be NaN.")

    iso3_dirs = sorted(d for d in OUTPUTS.iterdir()
                       if d.is_dir() and len(d.name) == 3 and d.name.isalpha())
    print(f"[INFO] Step 4C [{scenario}]: {len(iso3_dirs)} ISO3 folders under {OUTPUTS}")

    results = []
    for iso3_dir in iso3_dirs:
        iso3 = iso3_dir.name.upper()
        step4b_csv = iso3_dir / f"step4b_damage_by_rp_USD2019{suffix}.csv"

        if not step4b_csv.exists():
            _write_step4c(iso3_dir, iso3, 0.0, rps_used=[], status="missing_step4b",
                          detail="no step4b file found", suffix=suffix)
            results.append({"iso3": iso3, "status": "missing_step4b", "ead_usd2019": 0.0})
            continue

        try:
            df = pd.read_csv(step4b_csv)
        except Exception as e:
            _write_step4c(iso3_dir, iso3, 0.0, rps_used=[], status="bad_step4b_csv",
                          detail=f"failed_read:{e}", suffix=suffix)
            results.append({"iso3": iso3, "status": "bad_step4b_csv", "ead_usd2019": 0.0})
            continue

        c_rp = find_col(df, ("rp", "return period", "return_period", "rp_years"))
        c_dmg = find_col(df, ("damage_total_usd2019", "total_damage_usd2019",
                              "damage_usd2019", "total_damage", "damage"))
        if c_rp is None or c_dmg is None:
            _write_step4c(iso3_dir, iso3, 0.0, rps_used=[], status="missing_columns",
                          detail=f"need rp+damage cols; have={list(df.columns)}", suffix=suffix)
            results.append({"iso3": iso3, "status": "missing_columns", "ead_usd2019": 0.0})
            continue

        rp = pd.to_numeric(df[c_rp], errors="coerce").to_numpy()
        dmg = pd.to_numeric(df[c_dmg], errors="coerce").to_numpy()

        prot = None
        if prot_tbl is not None:
            hit = prot_tbl.loc[prot_tbl.iso3 == iso3, "protection_rp"]
            prot = float(hit.iloc[0]) if len(hit) else None

        # Protection standard and its evidence class for this country.
        # 0 is meaningful (unprotected); NaN means "documented-only restriction
        # applied, do not truncate". Both must be defined before any call to
        # _write_step4c below.
        prot, prot_ev = None, ""
        if prot_tbl is not None:
            hit = prot_tbl.loc[prot_tbl.iso3 == iso3]
            if len(hit):
                v = hit["protection_rp"].iloc[0]
                prot = float(v) if pd.notna(v) else None
                prot_ev = str(hit["protection_evidence"].iloc[0])

        ead, pts, msgs, ead_nc = ff.compute_ead_from_points(
            rp, dmg, add_p1_zero=VAL_ADD_P1_ZERO, tail_mode=VAL_TAIL_MODE,
            enforce_monotone=VAL_ENFORCE_MONOTONE, protection_rp=prot)

        # Second, attributed series -- integrated with identical boundary
        # assumptions so the two EADs are directly comparable.
        ead_attr = np.nan
        c_attr = find_col(df, ("damage_attributed_to_spa_usd2019",))
        if VAL_APPLY_SERVICE_FLOW and c_attr is not None:
            attr = pd.to_numeric(df[c_attr], errors="coerce").to_numpy()
            if np.isfinite(attr).any():
                ead_attr, _, _, _ = ff.compute_ead_from_points(
                    rp, attr, add_p1_zero=VAL_ADD_P1_ZERO, tail_mode=VAL_TAIL_MODE,
                    enforce_monotone=VAL_ENFORCE_MONOTONE)

        rps_used = sorted(int(x) for x in pd.Series(rp).dropna().unique() if x > 0)
        _write_step4c(iso3_dir, iso3, ead, rps_used=rps_used, status="ok",
                      detail=";".join(msgs), ead_attributed=ead_attr, suffix=suffix,
                      ead_nc=ead_nc, protection_rp=prot,
                      protection_evidence=prot_ev)

        if VAL_WRITE_INTEGRATION_POINTS:
            pts2 = pts.copy()
            pts2.insert(0, "iso3", iso3)
            write_csv(pts2, iso3_dir / f"step4c_ead_USD2019{suffix}__integration_points.csv")

        if len(rps_used) < 4:
            warnings.warn(f"[WARN] {iso3}: only {len(rps_used)} unique RP points "
                          f"({rps_used}). Fewer than four makes the EAD integral "
                          f"unreliable -- check the depth rasters for this country.")

        results.append({"iso3": iso3, "status": "ok", "ead_usd2019": float(ead),
                        "ead_attributed_to_spa_usd2019": float(ead_attr)})
        print(f"[OK] {iso3}: EAD = {fmt_usd(ead)}")

    status_df = pd.DataFrame(results)
    write_csv(status_df, VAL_EXPORT_DIR / f"step4c_global_status{suffix}.csv")
    n_ok = int((status_df["status"] == "ok").sum()) if not status_df.empty else 0
    total = float(pd.to_numeric(status_df.get("ead_usd2019"), errors="coerce").sum()) \
        if not status_df.empty else 0.0
    print(f"[DONE] Step 4C [{scenario}]: ok={n_ok} / {len(iso3_dirs)}, "
          f"total EAD ${total:,.0f}")

    # ⚠ "ok" counts countries that completed, not countries that produced a number. On 2026-08-29
    # this printed ok=250/250 while every country was $0, because the run had hydrated no config
    # and there was nothing to value -- and the zero total flowed all the way into a published
    # step4e table without anything raising. A whole-world zero is never a real result, so it stops
    # the run here rather than being discovered by comparing against someone else's figure.
    if not status_df.empty and total <= 0.0:
        raise ValueError(
            'Step 4C [%s] finished with %d of %d countries ok but a total EAD of $0. That is not a '
            'result: it means the damage inputs were absent or unconfigured. Check that '
            'es_parameters and es_config hydrated -- ProjectFlow seeds them from input_template '
            'beside the calling script, so a runner in a subdirectory finds none.'
            % (scenario, n_ok, len(iso3_dirs)))
    return status_df


def export_global_results():
    """Consolidate per-country EAD into global CSV / GPKG (and optional rasters)."""
    argv = [
        "--admin0-gpkg", ADMIN0_PATH,
        "--outputs-root", OUTPUTS,
        "--out-dir", VAL_EXPORT_DIR,
        "--ead-filename", "step4c_ead_USD2019.csv",
        "--region-col", VAL_REGION_COL,
        "--rps", ",".join(str(r) for r in RETURN_PERIODS),
        "--tail-mode", VAL_TAIL_MODE,
        "--write-csvs", "--write-gpkg", "--status-report",
    ]
    if VAL_FILL_MISSING_ZERO:
        argv += ["--fill-missing-zero"]
    if not VAL_ADD_P1_ZERO:
        argv += ["--no-add-p1-zero"]
    if VAL_COMPUTE_EAD_RASTERS:
        argv += ["--compute-ead-rasters"]
    if VAL_MOSAIC_GLOBAL_RASTER:
        argv += ["--mosaic-global-raster"]

    return _invoke_script(VAL_STEP4D_SCRIPT, argv, label="Step 4D global export")


def export_attributed_summary() -> Optional[Path]:
    """
    Companion to Step 4D.

    The preserved 4D script only knows about `ead_usd2019`, so it cannot carry
    the attributed series. Rather than fork that script, this walks the Step 4C
    files directly and writes a parallel country table with both columns plus
    the Admin0 name/region join, so the two numbers can be compared per country
    and per region.

    Skipped entirely when VAL_APPLY_SERVICE_FLOW is off.
    """
    if not VAL_APPLY_SERVICE_FLOW:
        return None

    rows = []
    for iso3_dir in sorted(d for d in OUTPUTS.iterdir()
                           if d.is_dir() and len(d.name) == 3 and d.name.isalpha()):
        f = iso3_dir / "step4c_ead_USD2019.csv"
        if not f.exists():
            continue
        # A country that cannot be read must stop the run: it would otherwise leave the global
        # total silently, and the sum would still look plausible.
        rows.append(pd.read_csv(f))

    if not rows:
        warnings.warn("[WARN] No Step 4C files found; attributed summary not written.")
        return None

    df = pd.concat(rows, ignore_index=True)
    df["iso3"] = df["iso3"].astype(str).str.upper().str.strip()

    admin0 = load_admin0(ADMIN0_PATH)
    keep = ["iso3"]
    name_col = pick_name_column(admin0)
    if name_col:
        keep.append(name_col)
    if VAL_REGION_COL in admin0.columns:
        keep.append(VAL_REGION_COL)
    enrich = admin0[keep].drop_duplicates(subset=["iso3"])
    if name_col:
        enrich = enrich.rename(columns={name_col: "country_name"})

    df = enrich.merge(df, on="iso3", how="left")

    gross = pd.to_numeric(df["ead_usd2019"], errors="coerce")
    attr = pd.to_numeric(df.get("ead_attributed_to_spa_usd2019"), errors="coerce")
    df["attributed_share"] = np.where(gross > 0, attr / gross, np.nan)

    out_csv = VAL_EXPORT_DIR / "step4d_country_ead_with_service_flow_USD2019.csv"
    write_csv(df, out_csv)

    print(f"[OK] Attributed summary -> {out_csv}")
    print(f"     gross global EAD      = {fmt_usd(np.nansum(gross))}")
    print(f"     attributed to SPA     = {fmt_usd(np.nansum(attr))}")
    if np.nansum(gross) > 0:
        print(f"     attributed share      = {np.nansum(attr) / np.nansum(gross):.1%}")
    print("     NOTE: attribution of residual damage, NOT avoided damage.")
    return out_csv


def compute_flood_gep() -> Optional[Path]:
    """
    Difference each degraded scenario against current to get the service value.

        GEP_flood_k = EAD_degraded_k - EAD_current_k
        prevention_share_k = GEP_k / EAD_degraded_k

    Both degraded scenarios are reported side by side:

      *_insitu  ecosystems present but degraded (UMRB scenario-2 CN values).
                Conservative; use for the flood manuscript.
      *_bare    cover removed (TR-55 fallow bare soil). Same baseline as InVEST
                SDR's RKLS, so THIS is the column that belongs in a combined GEP
                table alongside the erosion module.

    Requires compute_ead_by_country() to have run for all three scenarios.
    """
    def collect(suffix: str, label: str) -> Optional[pd.DataFrame]:
        rows = []
        for d in sorted(x for x in OUTPUTS.iterdir()
                        if x.is_dir() and len(x.name) == 3 and x.name.isalpha()):
            f = d / f"step4c_ead_USD2019{suffix}.csv"
            if f.exists():
                # A country that cannot be read must stop the run rather than vanish from the
                # scenario total.
                rows.append(pd.read_csv(f))
        if not rows:
            return None
        df = pd.concat(rows, ignore_index=True)
        cols = {"ead_usd2019": f"ead_{label}_usd2019"}
        for c in ("ead_nc_usd2019", "ead_ncplus_usd2019"):
            if c in df.columns:
                cols[c] = c.replace("usd2019", f"{label}_usd2019")
        keep = ["iso3"] + [c for c in cols if c in df.columns]
        return df[keep].rename(columns=cols)

    cur = collect("", "current")
    if cur is None:
        warnings.warn("[WARN] no current-scenario Step 4C outputs found.")
        return None

    df = cur.copy()
    df["iso3"] = df["iso3"].astype(str).str.upper().str.strip()
    found = []
    for sc in ("degraded_insitu", "degraded_bare"):
        lab = sc.replace("degraded_", "")
        deg = collect(SCENARIO_SUFFIX[sc], lab)
        if deg is None:
            warnings.warn(f"[WARN] no Step 4C outputs for {sc}; skipping.")
            continue
        deg["iso3"] = deg["iso3"].astype(str).str.upper().str.strip()
        df = df.merge(deg, on="iso3", how="outer")
        found.append(lab)

    if not found:
        warnings.warn("[WARN] no degraded scenarios available. Run run_gep_chain().")
        return None

    c = pd.to_numeric(df["ead_current_usd2019"], errors="coerce")
    for lab in found:
        d = pd.to_numeric(df[f"ead_{lab}_usd2019"], errors="coerce")
        df[f"gep_flood_{lab}_usd2019"] = d - c
        df[f"prevention_share_{lab}"] = np.where(d > 0, (d - c) / d, np.nan)
        neg = int((df[f"gep_flood_{lab}_usd2019"] < 0).sum())
        if neg:
            warnings.warn(f"[WARN] {lab}: {neg} countries have negative GEP "
                          f"(degraded < current). Check both scenarios used the "
                          f"same RPs and damage curves.")

    admin0 = load_admin0(ADMIN0_PATH)
    keep = ["iso3"]
    name_col = pick_name_column(admin0)
    if name_col:
        keep.append(name_col)
    if VAL_REGION_COL in admin0.columns:
        keep.append(VAL_REGION_COL)
    enrich = admin0[keep].drop_duplicates(subset=["iso3"])
    if name_col:
        enrich = enrich.rename(columns={name_col: "country_name"})
    df = enrich.merge(df, on="iso3", how="right")

    write_csv(df, VAL_GEP_CSV)

    print(f"[OK] Flood GEP table -> {VAL_GEP_CSV}")
    print(f"     EAD current           = {fmt_usd(np.nansum(c))}")
    for lab in found:
        g = np.nansum(df[f"gep_flood_{lab}_usd2019"])
        print(f"     GEP ({lab:6s})         = {fmt_usd(g)}")
    if len(found) == 2:
        gi = np.nansum(df["gep_flood_insitu_usd2019"])
        gb = np.nansum(df["gep_flood_bare_usd2019"])
        print(f"     -> bracketed range    = {fmt_usd(gi)} to {fmt_usd(gb)}")
        print("     Use 'bare' for any combined table with erosion (same baseline).")
    if VAL_REPORT_PROTECTION_SPLIT:
        nc_cols = [c for c in df.columns if c.startswith("ead_nc_")]
        print("     NC / NC+ columns present (Vallecillo Eq.7 sensitivity).")
        if nc_cols:
            covered = pd.to_numeric(df[nc_cols[0]], errors="coerce").notna()
            share = (pd.to_numeric(df.ead_current_usd2019, errors="coerce")[covered].sum()
                     / max(np.nansum(c), 1e-9))
            print(f"     truncation applied to {int(covered.sum())} countries "
                  f"= {share:.1%} of current EAD (documented protection only);")
            print("     the remainder are reported untruncated.")
    return VAL_GEP_CSV


def run_gep_chain(skip_damage_tables: bool = True,
                  scenarios: Optional[List[str]] = None) -> dict:
    """
    Paired-scenario driver: run 4B/4C for current plus each degraded scenario,
    then difference. Roughly triples Step 4B cost with both degraded scenarios,
    so it is a deliberate separate task rather than part of run_valuation_chain().
    """
    out = {}
    if not skip_damage_tables:
        out["damage_tables"] = build_damage_tables()

    run = scenarios if scenarios else list(SCENARIOS)
    for n, sc in enumerate(run, 1):
        print(f"\n=== scenario {n} of {len(run)}: {sc} ===")
        out[f"step4b_{sc}"] = compute_pixel_damages(scenario=sc)
        out[f"step4c_{sc}"] = compute_ead_by_country(scenario=sc)

    out["gep"] = compute_flood_gep()
    return out


def run_valuation_chain(skip_damage_tables: bool = False) -> dict:
    """Section D driver: 4A -> 4B -> 4C -> 4D (+ attributed companion export)."""
    out = {}
    if not skip_damage_tables:
        out["damage_tables"] = build_damage_tables()
    out["step4b"] = compute_pixel_damages()
    out["step4c"] = compute_ead_by_country()
    out["step4d"] = export_global_results()
    out["attributed"] = export_attributed_summary()
    return out


MAP_FIG_DIR = None


MAP_COUNTRY_EAD_CSV = None


MAP_GLOBAL_TOTALS_CSV = None


MAP_SERVICE_FLOW_CSV = None



def publish_inputs(p):
    """Every task's first line: the flood es_config row, its es_parameters data references, the
    shared country references, and the project paths resolved from configuration.

    The house shape, so a task gets its settings whether it runs alone or inside a tree. It is also
    where `configure_paths` runs, which is what turns the module's path constants from None into
    real locations under `flood_root_dir`.
    """
    utilities.hydrate_es_config(p, 'flood', log=hb.log)
    utilities.hydrate_es_parameters(p, 'flood', log=hb.log)
    utilities.initialize_country_paths(p)
    configure_paths(p)
    if not hasattr(p, 'results'):
        p.results = {}
    return p

def configure_paths(p):
    """Resolve every project path from configuration, at task time rather than at import.

    The source repo built these as module-level constants under an absolute root taken from a
    GEP_FLOOD_ROOT environment variable, so importing the module on any other machine bound them to
    a directory that did not exist. They are declared None above and set here from
    `p.flood_root_dir`, which es_parameters supplies -- machine configuration lives in that CSV, not
    in code. Every task calls this through publish_inputs, so a task gets real paths whether it runs
    alone or in a tree.
    """
    global ROOT, INPUTS, OUTPUTS, ADMIN0_PATH, LULC_PATH, DEPTH_RAW_DIR, DEPTH_EXTRACT_DIR, DEPTH_ALIGNED_DIR, DEPTH_MASK_DIR, SDA_MAPPING_JSON, GLOBAL_SDA_TIF, GLOBAL_SDA_LEGEND_CSV, SPA_PATH, SPA_RATIO_PATH, QA_DIR, SDA_ROADS_PATH, SDA_POP_PATH, FLOW_SUMMARY_CSV, VAL_DAMAGE_DIR, VAL_CANONICAL_EUR_CSV, VAL_FACTORS_CSV, VAL_DAMAGE_LONG_CSV, VAL_DAMAGE_WIDE_CSV, VAL_SDA_DAMAGE_LONG_CSV, VAL_SDA_DAMAGE_WIDE_CSV, VAL_EXPORT_DIR, VAL_SERVICE_FLOW_CSV, VAL_AMPLIFICATION_DIR, VAL_CN_TABLE, VAL_GEP_CSV, MAP_FIG_DIR, MAP_COUNTRY_EAD_CSV, MAP_GLOBAL_TOTALS_CSV, MAP_SERVICE_FLOW_CSV
    root = getattr(p, 'flood_root_dir', None)
    if root:
        ROOT = Path(str(root))
    elif ROOT is None:
        raise ValueError(
            'flood_root_dir is not set. It is a machine-specific path, so it belongs in '
            'es_parameters.csv rather than in code or an environment variable.')
    INPUTS = ROOT / "inputs"
    OUTPUTS = ROOT / "outputs"
    ADMIN0_PATH = INPUTS / "country_vector" / "country_boundary_r250_with_iso3.gpkg"
    LULC_PATH = INPUTS / "lulc" / "lulc_esa_2019_int_reproj.tif"
    DEPTH_RAW_DIR = INPUTS / "floodplain_depth_raw"
    DEPTH_EXTRACT_DIR = DEPTH_RAW_DIR / "extracted_tifs"
    DEPTH_ALIGNED_DIR = INPUTS / "floodplain_depth" / "aligned_to_lulc"
    DEPTH_MASK_DIR = INPUTS / "floodplain_depth" / "masks_aligned_to_lulc"
    SDA_MAPPING_JSON = INPUTS / "lulc_to_sda_mapping" / "lulc_to_sda_mapping.json"
    GLOBAL_SDA_TIF = INPUTS / "sda" / "sda_esa300m_artif_crop_pasture.tif"
    GLOBAL_SDA_LEGEND_CSV = INPUTS / "sda" / "sda_esa300m_legend.csv"
    SPA_PATH = INPUTS / "global_spa_ben" / "global_prr_spa.tif"
    SPA_RATIO_PATH = INPUTS / "global_spa_ben" / "global_upstream_spa_ratio.tif"
    QA_DIR = OUTPUTS / "qa_maps"
    SDA_ROADS_PATH = INPUTS / "roads" / "roads_mask_match_depth.tif"
    SDA_POP_PATH = INPUTS / "pop" / "GlobPOP_Count_30arc_2020_I32.tif"
    FLOW_SUMMARY_CSV = OUTPUTS / "global_service_flow_spa_to_sda.csv"
    VAL_DAMAGE_DIR = INPUTS / "flood_damage"
    VAL_CANONICAL_EUR_CSV = VAL_DAMAGE_DIR / "country_landtype_flood_damage_JRC_EUR_m2.csv"
    VAL_FACTORS_CSV = VAL_DAMAGE_DIR / "_currency_audit" / "currency_conversion_factors_EUR2010_to_USD2019.csv"
    VAL_DAMAGE_LONG_CSV = VAL_DAMAGE_DIR / "damage_functions_depth_USD2019_long.csv"
    VAL_DAMAGE_WIDE_CSV = VAL_DAMAGE_DIR / "damage_functions_depth_USD2019_wide.csv"
    VAL_SDA_DAMAGE_LONG_CSV = VAL_DAMAGE_DIR / "damage_functions_sda_depth_USD2019_long.csv"
    VAL_SDA_DAMAGE_WIDE_CSV = VAL_DAMAGE_DIR / "damage_functions_sda_depth_USD2019_wide.csv"
    VAL_EXPORT_DIR = OUTPUTS / "_global"
    VAL_SERVICE_FLOW_CSV = OUTPUTS / "global_service_flow_spa_to_sda.csv"
    VAL_AMPLIFICATION_DIR = INPUTS / "counterfactual"
    VAL_CN_TABLE = INPUTS / "counterfactual" / "esa_cci_CN_three_scenarios.csv"
    VAL_GEP_CSV = OUTPUTS / "_global" / "step4e_flood_gep_USD2019.csv"
    MAP_FIG_DIR = OUTPUTS / "_global" / "figures"
    MAP_COUNTRY_EAD_CSV = OUTPUTS / "_global" / "step4d_country_ead_USD2019.csv"
    MAP_GLOBAL_TOTALS_CSV = OUTPUTS / "_global" / "step4d_global_totals_USD2019.csv"
    MAP_SERVICE_FLOW_CSV = OUTPUTS / "global_service_flow_spa_to_sda.csv"
    return p



MAP_K_CLASSES = 5


MAP_TOP_N = 20


MAP_MONEY_UNIT_LABEL = "2019 USD million"


MAP_RASTER_DOWNSAMPLE_FACTOR = 6


def configure_maps(p):
    """
    Override the Section-E constants from the ProjectFlow object. Also pushes
    EXCLUDE_ISO3 / ROBINSON_CRS / USD_TO_MILLIONS / TOP_N onto flood_utils,
    since plot_publication_choropleth_categorical() (which lives there) reads
    those as module globals. Called by
    flood_tasks.task_generate_maps_and_figures().
    """
    configure_shared(p)

    global MAP_FIG_DIR, MAP_COUNTRY_EAD_CSV, MAP_GLOBAL_TOTALS_CSV
    global MAP_SERVICE_FLOW_CSV, MAP_K_CLASSES, MAP_TOP_N
    global MAP_MONEY_UNIT_LABEL, MAP_RASTER_DOWNSAMPLE_FACTOR

    export_dir = Path(getattr(p, 'flood_global_export_dir', OUTPUTS / "_global"))
    MAP_FIG_DIR = Path(getattr(p, 'flood_figures_dir', export_dir / "figures"))
    MAP_FIG_DIR.mkdir(parents=True, exist_ok=True)

    MAP_COUNTRY_EAD_CSV = Path(getattr(p, 'flood_country_ead_csv',
                                       export_dir / "step4d_country_ead_USD2019.csv"))
    MAP_GLOBAL_TOTALS_CSV = Path(getattr(p, 'flood_global_totals_csv',
                                         export_dir / "step4d_global_totals_USD2019.csv"))
    MAP_SERVICE_FLOW_CSV = Path(getattr(p, 'flood_service_flow_summary_csv',
                                        OUTPUTS / "global_service_flow_spa_to_sda.csv"))

    MAP_K_CLASSES = getattr(p, 'flood_map_k_classes', 5)
    MAP_TOP_N = getattr(p, 'flood_top_n', 20)
    MAP_MONEY_UNIT_LABEL = getattr(p, 'flood_money_unit_label', "2019 USD million")
    MAP_RASTER_DOWNSAMPLE_FACTOR = getattr(p, 'flood_raster_downsample_factor', 6)

    # Push shared plotting constants onto flood_utils (see docstring above).
    flood_utils.EXCLUDE_ISO3 = getattr(p, 'flood_exclude_iso3', {"ATA"})
    flood_utils.ROBINSON_CRS = getattr(p, 'flood_robinson_crs', "+proj=robin")
    flood_utils.USD_TO_MILLIONS = getattr(p, 'flood_usd_to_millions', 1e6)
    flood_utils.TOP_N = MAP_TOP_N


def _load_country_ead() -> pd.DataFrame:
    assert_exists(MAP_COUNTRY_EAD_CSV, "Run Section D (step 4D) before Section E.")
    df = pd.read_csv(MAP_COUNTRY_EAD_CSV)
    df["iso3"] = df["iso3"].astype(str).str.upper().str.strip()
    ead_col = find_col(df, ("ead_usd2019", "ead usd2019", "ead"))
    if ead_col is None:
        raise ValueError(f"No EAD column found in {MAP_COUNTRY_EAD_CSV}: {list(df.columns)}")
    df = df.rename(columns={ead_col: "ead_usd2019"})
    df["ead_usd2019"] = pd.to_numeric(df["ead_usd2019"], errors="coerce")
    return df


def generate_all_maps_and_figures() -> dict:
    """
    Section E driver: publication figures from Section D's country table --
    a global EAD choropleth (Fisher-Jenks), a top-N country bar chart, a
    regional breakdown, and the mean SPA->SDA service-flow map if Section C ran.
    """
    MAP_FIG_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {}

    country = _load_country_ead()
    admin0 = load_admin0(ADMIN0_PATH)

    joined = admin0.merge(country, on="iso3", how="left")

    # 1) Global choropleth of Expected Annual Damage
    png = MAP_FIG_DIR / "map_country_ead_USD2019.png"
    plot_publication_choropleth_categorical(
        joined, value_col="ead_usd2019",
        title="Expected Annual Flood Damage to Service Demanding Areas",
        out_png=png, legend_title=MAP_MONEY_UNIT_LABEL,
        scheme="fisher_jenks", k=MAP_K_CLASSES,
        value_unit="usd_millions", label_format="usd_millions",
    )
    outputs["map_country_ead"] = png
    print(f"[OK] Wrote {png}")

    # 2) Top-N countries by EAD
    top = top_n(country, "ead_usd2019", MAP_TOP_N).copy()
    if not top.empty:
        top["_m"] = top["ead_usd2019"] / flood_utils.USD_TO_MILLIONS
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(top["iso3"][::-1], top["_m"][::-1])
        ax.set_xlabel(MAP_MONEY_UNIT_LABEL)
        ax.set_title(f"Top {MAP_TOP_N} countries by Expected Annual Flood Damage")
        png = MAP_FIG_DIR / "bar_top_countries_ead.png"
        savefig(png)
        outputs["bar_top_countries"] = png
        print(f"[OK] Wrote {png}")

    # 3) Regional breakdown, if Step 4D enriched with a region column
    if "region_wb" in country.columns:
        reg = (country.groupby("region_wb", dropna=True)["ead_usd2019"]
               .sum().sort_values(ascending=False) / flood_utils.USD_TO_MILLIONS)
        if not reg.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(reg.index[::-1], reg.values[::-1])
            ax.set_xlabel(MAP_MONEY_UNIT_LABEL)
            ax.set_title("Expected Annual Flood Damage by World Bank region")
            png = MAP_FIG_DIR / "bar_region_ead.png"
            savefig(png)
            outputs["bar_region"] = png
            print(f"[OK] Wrote {png}")

    # 4) Mean SPA -> SDA service flow by country (Section C output)
    if Path(MAP_SERVICE_FLOW_CSV).exists():
        flow = pd.read_csv(MAP_SERVICE_FLOW_CSV)
        if {"iso3", "mean_spa_ratio_on_sda"}.issubset(flow.columns):
            agg = (flow.groupby("iso3", as_index=False)["mean_spa_ratio_on_sda"].mean())
            agg["iso3"] = agg["iso3"].astype(str).str.upper()
            j2 = admin0.merge(agg, on="iso3", how="left")
            png = MAP_FIG_DIR / "map_mean_service_flow_frac.png"
            plot_publication_choropleth_categorical(
                j2, value_col="mean_spa_ratio_on_sda",
                title="Mean upstream SPA share serving flood-exposed SDA",
                out_png=png, legend_title="Service flow fraction (0-1)",
                scheme="quantiles", k=MAP_K_CLASSES,
                value_unit="raw", label_format="percent",
            )
            outputs["map_service_flow"] = png
            print(f"[OK] Wrote {png}")
    else:
        warnings.warn(f"[WARN] No service-flow summary at {MAP_SERVICE_FLOW_CSV}; "
                      "skipping the service-flow map.")

    return outputs


# ---------------------------------------------------------------------------------------------
# Task wrappers: the seam flood_initialize grafts onto a ProjectFlow tree.
# ---------------------------------------------------------------------------------------------
def task_prepare_flood_inputs(p):
    """
    Task wrapper for Section A: download and align the JRC flood depth
    rasters to the LULC grid, write the LULC -> SDA mapping JSON, build the
    global SDA raster + audit legend, and QA the SPA raster. Originally
    download_and_prep_jrc_flood_depth.ipynb, sda_step_2A_make_lulc_to_sda_
    mapping_esa300.py, build_sda_from_esa300m.py and qa_spa_global_step1.py.
    """
    publish_inputs(p)
    configure_inputs(p)

    service_results = p.results.setdefault('flood', {})
    service_results['global_sda_raster'] = str(GLOBAL_SDA_TIF)
    service_results['sda_mapping_json'] = str(SDA_MAPPING_JSON)

    skip_download = getattr(p, 'flood_skip_depth_download', False)
    prepare_all_inputs(skip_download=skip_download)
    return True


def task_build_sda(p):
    """
    Task wrapper for Section B: intersect economic assets (from LULC) with the
    floodplain (from depth rasters) to delineate Service Demanding Areas per
    ISO3 x return period. Originally sda_step2_build_sda_global.py.

    The underlying script does its own smart-skip (signature hashing of inputs
    + options), so re-running is cheap; the guard here only short-circuits when
    the caller has explicitly asked to skip.
    """
    publish_inputs(p)
    configure_sda(p)
    build_sda_global()
    return True


def task_compute_service_flow(p):
    """
    Task wrapper for Section C: reproject the upstream SPA ratio onto the SDA
    grid and record the flood-regulation service flow fraction for each
    ISO3 x RP. Originally serviceflow_step3_spa_to_sda_ratio_global.py.
    Writes global_service_flow_spa_to_sda.csv, which Section E maps.
    """
    publish_inputs(p)
    configure_service_flow(p)

    service_results = p.results.setdefault('flood', {})
    service_results['service_flow_summary'] = str(FLOW_SUMMARY_CSV)

    compute_service_flow_global()
    return True


def task_compute_flood_damages(p):
    """
    Task wrapper for Section D: the monetary chain 4A -> 4B -> 4C -> 4D.
    Builds USD2019 depth-damage tables, computes pixel-wise damages per return
    period, integrates them over exceedance probability into Expected Annual
    Damage, and packages the global CSV/GPKG outputs. Originally
    build_damage_table_USD2019.py, flood_gep_step4b_pixel_damage_USD2019.py,
    flood_gep_step4c_ead_USD2019_global.py and
    flood_gep_step4d_export_global_USD2019.py.
    """
    publish_inputs(p)
    configure_valuation(p)

    service_results = p.results.setdefault('flood', {})
    service_results['country_ead_csv'] = os.path.join(
        str(VAL_EXPORT_DIR), "step4d_country_ead_USD2019.csv")
    service_results['global_totals_csv'] = os.path.join(
        str(VAL_EXPORT_DIR), "step4d_global_totals_USD2019.csv")

    if hb.path_all_exist([service_results['country_ead_csv']]):
        hb.log("step4d_country_ead_USD2019.csv already exists. "
               "Skipping the flood valuation chain.")
    else:
        skip_tables = getattr(p, 'flood_skip_damage_tables', False)
        run_valuation_chain(skip_damage_tables=skip_tables)

    return True


def task_compute_flood_gep(p):
    """
    Task wrapper for the paired-scenario counterfactual: run Step 4B/4C for both
    the current and the degraded world, then difference them into

        GEP_flood = EAD_degraded - EAD_current

    This is the construct comparable to the erosion module's GEP. It roughly
    doubles Section D's cost, so it is a separate task rather than part of
    task_compute_flood_damages().
    """
    publish_inputs(p)
    configure_valuation(p)

    service_results = p.results.setdefault('flood', {})
    service_results['flood_gep_csv'] = str(VAL_GEP_CSV)

    if hb.path_all_exist([service_results['flood_gep_csv']]):
        hb.log("step4e_flood_gep_USD2019.csv already exists. Skipping GEP chain.")
    else:
        run_gep_chain(
            skip_damage_tables=getattr(p, 'flood_skip_damage_tables', True),
            scenarios=getattr(p, 'flood_gep_scenarios', None))

    return True


def task_generate_maps_and_figures(p):
    """
    Task wrapper for Section E: publication-ready choropleths and charts built
    from task_compute_flood_damages()'s and task_compute_service_flow()'s
    outputs. Originally analyze_step4d_global_results.py.
    """
    publish_inputs(p)
    configure_maps(p)
    generate_all_maps_and_figures()
    return True


def gep_calculation(p):
    """GEP valuation for flood: the per-country avoided damage the pipeline produces.

    The quantity is the flood damage ecosystems prevent, which the author's own export writes as
    `flood_gep_for_merge_v2_2024hazard.csv` -- iso3_r250_label and gep_const2019_usd, $11.40bn over
    162 non-zero countries. That file is the account's input rather than something recomputed here,
    for the same reason the other read-a-value services work that way: the full pipeline needs the
    cluster inputs and about three hours, and the number it produces is this one.

    ⚠ The distinction that matters: this is the GEP, NOT the $887.8bn of undefended expected annual
    damage the same pipeline reports. Reproducing the damage side settles the port and not the
    account.
    """
    publish_inputs(p)
    service_results, already_done = utilities.begin_gep_calculation(p, 'flood')
    if already_done:
        return

    df_gep = hb.df_read(p.flood_gep_for_merge_path)
    df_gep = df_gep.rename(columns={'gep_const2019_usd': 'flood_gep'})
    df_gep['year'] = int(p.gep_base_year)
    # The country attributes every other service's table carries, so this output can be read,
    # grouped and reported the same way.
    attribute_columns = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name', 'continent',
                         'region_un', 'region_wb', 'income_grp', 'subregion']
    countries = utilities.collapse_countries_to_r250(p.df_countries)
    countries = countries[[c for c in attribute_columns if c in countries.columns]]
    df_gep = countries.merge(df_gep, on='iso3_r250_label', how='left')
    hb.df_write(df_gep, service_results['gep_by_country_base_year'])
    hb.log('Flood GEP: %.4g USD over %d countries with a positive value.'
           % (df_gep['flood_gep'].sum(), (df_gep['flood_gep'] > 0).sum()))
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)

# -*- coding: utf-8 -*-
"""File handling and task definitions for the flood-control account.

Everything that touches disk lives here: fetching and warping the hazard rasters, delineating the
service-demanding areas, routing the service flow, reading the damage tables, windowing every
country out of the global grid, and writing the results. The equations these steps apply are in
`flood_functions`, which knows nothing about where anything is stored.

That direction is one-way and worth keeping so. A calculation that opens its own inputs can only be
checked by replacing the file reader, and a test that replaces a file reader is testing the wiring
rather than the arithmetic.

Settings come from es_parameters and are read off `p` where they are used. `publish_inputs` is the
one place that decides where a file lives, so every task sees the same layout however it is run. The
task wrappers at the end are the seam `flood_initialize` grafts onto a task tree.
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
# Generic raster/table/plotting helpers live in flood_functions.py and the
# shared utilities.
# =============================================================================
from __future__ import annotations

import glob
import json
import os
import warnings
import zipfile
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

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

from global_invest.flood import flood_functions as ff
from global_invest import utilities


# =============================================================================
# SHARED PROJECT PATHS (defined ONCE -- see design note above)
# =============================================================================
# Machine-specific, so es_parameters owns it. Every path below is derived from it.


# The return periods the damage integral runs over. es_parameters owns it: dropping one
# changes EAD without changing anything a run reports.



import hazelbean as hb


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







JRC_ZIP_URL_TEMPLATE = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/FLOODS/GlobalMaps/floodMapGL_rp{rp}y.zip"
)












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

# The countries FLOPROS documents a protection standard for; the rest are inferred from GDP.
FLOPROS_DOCUMENTED_ISO3 = {
    "ARG", "AUS", "AUT", "BEL", "BGD", "BLZ", "BRA", "CAN", "CHE", "CHN",
    "CZE", "DEU", "DNK", "ESP", "GBR", "GHA", "HRV", "HUN", "IDN", "IND",
    "IRL", "ITA", "JPN", "MDG", "MOZ", "NLD", "NZL", "POL", "ROU", "RUS",
    "SGP", "SVK", "THA", "TWN", "USA", "VNM", "ZAF",
}


def sda_code_to_type(p):
    """SDA raster codes to the land types the damage tables are keyed on.

    Args:
        p: the ProjectFlow object, for `flood_use_roads`.

    Returns:
        dict: raster code to land type, including roads only when the run uses them.
    """
    types = {code: label for label, code in SDA_CODE.items() if label not in ("none", "roads")}
    if p.flood_use_roads:
        types[SDA_CODE["roads"]] = "roads"
    return types


def documented_protection_iso3(p):
    """The ISO3s counted as having a documented protection standard.

    Args:
        p: the ProjectFlow object, for `flood_protection_evidence_path`.

    Returns:
        set: ISO3 codes, from the evidence table when one is supplied, else FLOPROS_DOCUMENTED_ISO3.

    Raises:
        NameError: if a supplied evidence table has no ISO3 column. Which countries these are
            decides how much damage is truncated, so an unreadable table stops the run.
    """
    if not p.flood_protection_evidence_path:
        return FLOPROS_DOCUMENTED_ISO3
    evidence = hb.df_read(str(p.flood_protection_evidence_path))
    column = utilities.find_col(evidence, ("iso3", "iso_a3", "adm0_a3"))
    if not column:
        raise NameError(
            'No ISO3 column in %s. Looked for iso3, iso_a3, adm0_a3; found %s.'
            % (p.flood_protection_evidence_path, list(evidence.columns)))
    return set(evidence[column].astype(str).str.upper().str.strip())





def _download(url: str, dst: str) -> None:
    from urllib.request import urlretrieve

    hb.create_directories(os.path.dirname(str(dst)))
    if hb.path_exists(dst) and os.path.getsize(str(dst)) > 0:
        print(f"[SKIP] already downloaded: {os.path.basename(str(dst))}")
        return
    print(f"[DL] {url}")
    urlretrieve(url, str(dst))
    if not hb.path_exists(dst) or os.path.getsize(str(dst)) == 0:
        raise RuntimeError(f"Download failed or empty file: {dst}")
    print(f"[OK]  saved: {dst} ({os.path.getsize(str(dst)) / 1e6:.1f} MB)")


def _extract_zip(zip_path: str, out_dir: str) -> List[str]:
    hb.create_directories(str(out_dir))
    extracted: List[str] = []
    print(f"[UNZIP] {os.path.basename(str(zip_path))}")
    with zipfile.ZipFile(zip_path, "r") as z:
        tifs = [m for m in z.namelist() if m.lower().endswith((".tif", ".tiff"))]
        if not tifs:
            raise RuntimeError(f"No GeoTIFF found in {zip_path}")
        for m in tifs:
            out_path = os.path.join(str(out_dir), os.path.basename(m))
            if hb.path_exists(out_path) and os.path.getsize(str(out_path)) > 0:
                extracted.append(out_path)
                continue
            z.extract(m, out_dir)
            nested = os.path.join(str(out_dir), m)
            if hb.path_exists(nested) and nested != out_path:
                nested.rename(out_path)
            extracted.append(out_path)
    print(f"[OK]  extracted {len(extracted)} tif(s)")
    return extracted


def _warp_to_lulc(src_tif: str, dst_tif: str, ref_profile: dict,
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

        hb.create_directories(os.path.dirname(str(dst_tif)))
        with rasterio.open(dst_tif, "w", **prof) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform, src_crs=src.crs, src_nodata=src_nodata,
                dst_transform=prof["transform"], dst_crs=prof["crs"],
                dst_nodata=prof["nodata"], resampling=resampling,
            )


def _write_mask_from_depth(depth_tif: str, mask_tif: str, threshold: float = 0.0) -> None:
    with rasterio.open(depth_tif) as src:
        arr = src.read(1).astype("float32")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        mask = (np.nan_to_num(arr, nan=0.0) > threshold).astype("uint8")
        prof = src.profile.copy()
        prof.update(dtype="uint8", count=1, nodata=0, compress="DEFLATE",
                    tiled=True, BIGTIFF="IF_SAFER")
    utilities.save_raster_completely(mask_tif, prof, mask)


def download_and_align_jrc_depth(p, return_periods: Optional[List[int]] = None) -> Dict[int, str]:
    """
    Download the JRC global flood hazard maps (water depth, metres) for each
    return period and warp them onto the LULC reference grid, so every later
    step can do pixelwise depth-damage without re-warping.
    """
    rps = return_periods if return_periods is not None else p.flood_return_periods
    utilities.assert_exists(p.flood_lulc_path, "LULC reference grid is required before aligning depth.")

    with rasterio.open(p.flood_lulc_path) as ref:
        ref_profile = ref.profile.copy()
        ref_profile.update(width=ref.width, height=ref.height,
                           transform=ref.transform, crs=ref.crs)

    print("[INFO] LULC reference grid:")
    print("       CRS:", ref_profile["crs"])
    print("       shape:", (ref_profile["height"], ref_profile["width"]))
    print("       transform:", ref_profile["transform"])

    out: Dict[int, str] = {}
    for rp in rps:
        zip_path = os.path.join(p.flood_depth_raw_path, f"floodMapGL_rp{rp}y.zip")
        _download(JRC_ZIP_URL_TEMPLATE.format(rp=rp), zip_path)
        tifs = _extract_zip(zip_path, p.flood_depth_extract_path)

        for tif in tifs:
            out_depth = os.path.join(p.flood_depth_aligned_path, f"JRC_flood_depth_rp{rp}y__matchLULC.tif")
            if utilities.raster_ok(out_depth):
                print(f"[SKIP] aligned depth exists: {os.path.basename(str(out_depth))}")
            else:
                print(f"[WARP] RP{rp}: {os.path.basename(str(tif))} -> {os.path.basename(str(out_depth))}")
                _warp_to_lulc(tif, out_depth, ref_profile)
                print(f"[OK]   wrote: {out_depth}")

            out_mask = os.path.join(p.flood_depth_mask_path, f"JRC_flood_mask_rp{rp}y__matchLULC.tif")
            if utilities.raster_ok(out_mask):
                print(f"[SKIP] mask exists: {os.path.basename(str(out_mask))}")
            else:
                _write_mask_from_depth(out_depth, out_mask, threshold=0.0)
                print(f"[OK]   wrote: {out_mask}")

            out[rp] = out_depth
    return out


def write_lulc_to_sda_mapping(p) -> str:
    """
    Write the JRC-INCA style LULC -> SDA mapping JSON, then QA it against the
    codes actually present in the LULC raster.

    Everything in ESA_CODEBOOK that is not artif/crop/pasture is written to
    'ignore' explicitly, so a reviewer can confirm every code was classified
    on purpose rather than by falling through to 0.
    """
    built_up = list(ESA_TO_SDA.get("artif", []))
    cropland = list(ESA_TO_SDA.get("crop", []))
    pasture = list(ESA_TO_SDA.get("pasture", [])) if p.flood_include_pasture else []

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

    hb.create_directories(os.path.dirname(p.flood_sda_mapping_path))
    hb.write_to_file(json.dumps(mapping, indent=2), p.flood_sda_mapping_path)
    print(f"[OK] Wrote SDA mapping JSON -> {p.flood_sda_mapping_path}")

    codes = _sample_unique_lulc_codes(p.flood_lulc_path)
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

    return p.flood_sda_mapping_path


def _sample_unique_lulc_codes(lulc_path: str, n_windows: int = 60, win_size: int = 1024,
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
            for window in utilities.random_windows(src.width, src.height, n_windows, win_size, seed=rng_seed):
                arr = src.read(1, window=window)
                if nodata is not None:
                    arr = arr[arr != nodata]
                if arr.size:
                    codes.update(np.unique(arr).tolist())
    return sorted(int(c) for c in codes)


def build_global_sda_raster(p) -> str:
    """
    Build the global categorical SDA raster from ESA CCI land cover,
    block-wise so the full-resolution global grid never lands in RAM.

    Assignment is priority-ordered (pasture < crop < artif) so that a code
    appearing in more than one list resolves deterministically.
    """
    utilities.assert_exists(p.flood_lulc_path, "LULC raster is required to build the SDA raster.")

    artif_set = set(ESA_TO_SDA.get("artif", []))
    crop_set = set(ESA_TO_SDA.get("crop", []))
    pasture_set = set(ESA_TO_SDA.get("pasture", [])) if p.flood_include_pasture else set()

    hb.create_directories(os.path.dirname(p.flood_sda_raster_path))

    with rasterio.open(p.flood_lulc_path) as src:
        nodata = src.nodata
        profile = src.profile.copy()
        profile.update(dtype=rasterio.uint8, count=1, nodata=SDA_CODE["none"],
                       compress="deflate", tiled=True, BIGTIFF="IF_SAFER")

        tmp = p.flood_sda_raster_path + ".tmp"
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
        os.replace(tmp, p.flood_sda_raster_path)

    print("[OK] SDA raster written:", p.flood_sda_raster_path)
    write_sda_legend_csv(p)
    _report_sda_histogram(p.flood_sda_raster_path)
    return p.flood_sda_raster_path


def write_sda_legend_csv(p) -> str:
    """Complete ESA code legend: lucode, label, sda_type, sda_code, rule used."""
    esa_to_type = {k: "none" for k in ESA_CODEBOOK}
    for k in ESA_TO_SDA.get("artif", []):
        esa_to_type[k] = "artif"
    for k in ESA_TO_SDA.get("crop", []):
        esa_to_type[k] = "crop"
    if p.flood_include_pasture:
        for k in ESA_TO_SDA.get("pasture", []):
            if esa_to_type.get(k, "none") == "none":
                esa_to_type[k] = "pasture"

    rule = (f"artif={ESA_TO_SDA.get('artif')}; crop={ESA_TO_SDA.get('crop')}; "
            f"pasture={ESA_TO_SDA.get('pasture')} (include={p.flood_include_pasture})")

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
    utilities.write_csv(pd.DataFrame(rows), p.flood_sda_legend_path)
    print("[OK] Legend written:", p.flood_sda_legend_path)
    return p.flood_sda_legend_path


def _report_sda_histogram(path: str):
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
        print(utilities.raster_profile_string(ds))
    return counts


def qa_spa_raster(p, sample_windows: int = 80, window_size: int = 1024) -> str:
    """
    Validate the global SPA raster (produced upstream of this pipeline from
    runoff-retention potential) and write reproducible QA outputs:
    a quicklook PNG, a per-country SPA area CSV, and an alignment report.

    This does NOT recompute SPA -- it checks the raster is usable by Section C.
    """
    from rasterio.features import rasterize

    utilities.assert_exists(p.flood_spa_path, "SPA raster is required.")
    hb.create_directories(p.flood_qa_dir)

    report = ["=== SPA QA REPORT (Section A) ===\n", f"SPA: {p.flood_spa_path}\n\n"]

    with rasterio.open(p.flood_spa_path) as ds:
        report += ["=== SPA METADATA ===\n", utilities.raster_profile_string(ds) + "\n"]
        nodata = ds.nodata
        if nodata is None:
            report.append("[WARN] SPA raster has no nodata defined.\n")

        vals, finite_px, one_px = [], 0, 0
        for win in utilities.random_windows(ds.width, ds.height, sample_windows, window_size):
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
        out_png = os.path.join(p.flood_qa_dir, "global_prr_spa_quicklook.png")
        utilities.savefig(out_png, dpi=200)
        report.append(f"[OK] Wrote quicklook PNG: {out_png}\n\n")

        # Per-country SPA area, computed window-by-window off the country bbox
        admin0 = load_admin0(p.flood_country_vector_path)[["iso3", "geometry"]].to_crs(ds.crs)
        pix_m2 = utilities.pixel_area_m2(ds.transform)

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

        out_csv = p.flood_spa_country_summary_path
        utilities.write_csv(pd.DataFrame(rows).sort_values("iso3"), out_csv)
        report.append(f"[OK] Wrote country SPA summary: {out_csv}\n\n")

    # Alignment checks against the grids SPA has to line up with
    for label, path in (("LULC", p.flood_lulc_path),
                        ("DEPTH", os.path.join(p.flood_depth_aligned_path, f"JRC_flood_depth_rp{p.flood_return_periods[-1]}y__matchLULC.tif"))):
        if hb.path_exists(path):
            with rasterio.open(path) as x:
                report += [f"=== ALIGNMENT CHECK: {label} ===\n", utilities.raster_profile_string(x) + "\n"]
        else:
            report.append(f"[WARN] Alignment check missing {label}: {path}\n")

    out_txt = os.path.join(p.flood_qa_dir, "global_spa_alignment_report.txt")
    hb.write_to_file("".join(report), str(out_txt))
    print(f"[OK] Wrote report: {out_txt}")
    return out_txt


def prepare_all_inputs(p, skip_download: bool = False) -> dict:
    """Section A driver: everything the accounting steps need on disk."""
    results = {}
    if not skip_download:
        results["depth_rasters"] = download_and_align_jrc_depth(p)
    results["sda_mapping_json"] = write_lulc_to_sda_mapping(p)
    if not utilities.raster_ok(p.flood_sda_raster_path):
        results["global_sda_raster"] = build_global_sda_raster(p)
    else:
        print(f"[SKIP] Global SDA raster already exists: {p.flood_sda_raster_path}")
        results["global_sda_raster"] = p.flood_sda_raster_path
    if hb.path_exists(p.flood_spa_path):
        results["spa_qa_report"] = qa_spa_raster(p)
    else:
        warnings.warn(f"[WARN] SPA raster not found, skipping QA: {p.flood_spa_path}")
    return results






def _required(p, attribute, _unused_default=None):
    """A setting es_parameters owns. No code default: a missing row must fail, not be guessed."""
    if not hasattr(p, attribute):
        raise AttributeError(
            '%s is not set. It belongs in es_parameters.csv under the flood service; a default here '
            'would decide the run silently.' % attribute)
    return getattr(p, attribute)




SDA_CODE_VERSION = "2025-12-15_sda_step2_smartskip_v2_depth_inputs"


DEPTH_RASTER_PATTERN = "JRC_flood_depth_rp{rp}y__matchLULC.tif"

# Step 4C writes one of these under each ISO3 directory.
EAD_FILE_NAME = "step4c_ead_USD2019.csv"


# -----------------------------------------------------------------------------#
# Process one ISO3
# -----------------------------------------------------------------------------#
def process_country(
    iso3: str,
    admin0: gpd.GeoDataFrame,
    out_root: str,
    rp_map: dict[int, str],
    lulc_path: str,
    mapping: dict,
    depth_threshold_m: float,
    all_touched: bool,
    include_pasture: bool,
    use_roads: bool,
    roads_path: str,
    with_pop: bool,
    pop_path: str,
    write_depthbin: bool,
    depthbin_max: float,
) -> pd.DataFrame:

    iso3 = iso3.upper().strip()
    aoi = admin0[admin0["iso3"] == iso3]
    if aoi.empty:
        raise ValueError(f"ISO3 {iso3} not found in Admin0.")

    out_dir = os.path.join(str(out_root), iso3)
    hb.create_directories(str(out_dir))
    out_csv = os.path.join(str(out_dir), f"sda_summary_{iso3}.csv")

    artif_ids   = set(mapping.get("artif", []))
    crop_ids    = set(mapping.get("crop", []))
    pasture_ids = set(mapping.get("pasture", [])) if include_pasture else set()
    ignore_ids  = set(mapping.get("ignore", []))

    if (len(artif_ids) + len(crop_ids) + len(pasture_ids) == 0) and (not use_roads):
        raise ValueError(
            "Your mapping JSON provides no SDA codes (artif/crop/pasture all empty) and roads are disabled.\n"
            "Fix mapping-json or run with --use-roads."
        )

    if not hb.path_exists(lulc_path):
        raise FileNotFoundError(f"LULC raster not found:\n  {lulc_path}")
    if use_roads and not hb.path_exists(roads_path):
        raise FileNotFoundError(f"Roads raster not found:\n  {roads_path}")
    if with_pop and not hb.path_exists(pop_path):
        raise FileNotFoundError(f"Population raster not found:\n  {pop_path}")

    country_geom = aoi.geometry.values[0]

    print(f"\n=== SDA: Processing {iso3} ===")
    print(f"[INFO] depth_threshold_m: {depth_threshold_m:.3f} | all_touched={all_touched}")
    print(f"[INFO] mapping sizes: artif={len(artif_ids)} crop={len(crop_ids)} pasture={len(pasture_ids)} ignore={len(ignore_ids)}")
    print(f"[INFO] options: include_pasture={include_pasture} use_roads={use_roads} with_pop={with_pop} write_depthbin={write_depthbin}")

    metrics: list[dict] = []

    with rasterio.open(lulc_path) as lulc_src:
        roads_cm = rasterio.open(roads_path) if use_roads else nullcontext()
        pop_cm   = rasterio.open(pop_path)   if with_pop  else nullcontext()

        with roads_cm as roads_src, pop_cm as pop_src:
            for rp, depth_path in rp_map.items():
                if not hb.path_exists(depth_path):
                    print(f"[WARN] Missing depth raster for RP{rp}; skipping:\n  {depth_path}")
                    continue

                with rasterio.open(depth_path) as depth_src:
                    # Reproject AOI geometry to raster CRS (depth is the driver)
                    if admin0.crs != depth_src.crs:
                        geom_r = gpd.GeoDataFrame(geometry=[country_geom], crs=admin0.crs).to_crs(depth_src.crs).geometry.values[0]
                    else:
                        geom_r = country_geom

                    # HARD LOCK: LULC and depth must be aligned globally
                    utilities.assert_same_grid(depth_src, lulc_src, label_a="DEPTH", label_b="LULC")

                    # Clip depth
                    depth_img, depth_tr = rio_mask(
                        depth_src, [geom_r],
                        crop=True, filled=True,
                        nodata=depth_src.nodata if depth_src.nodata is not None else -9999.0,
                        all_touched=all_touched,
                    )
                    depth = depth_img[0].astype("float32")

                    nd = depth_src.nodata
                    valid_depth = np.isfinite(depth) if nd is None else (depth != float(nd))

                    depth_q = depth.copy()
                    depth_q[~valid_depth] = np.nan
                    depth_q[(depth_q < 0) & np.isfinite(depth_q)] = 0.0

                    floodplain = valid_depth & (depth_q > depth_threshold_m)

                    # Clip LULC (must match by construction)
                    lulc_img, lulc_tr = rio_mask(
                        lulc_src, [geom_r],
                        crop=True, filled=True,
                        nodata=lulc_src.nodata if lulc_src.nodata is not None else 255,
                        all_touched=all_touched,
                    )
                    lulc = lulc_img[0]

                    if depth_tr != lulc_tr or depth.shape != lulc.shape:
                        raise ValueError(
                            f"Post-clip mismatch for {iso3} RP{rp}.\n"
                            f"  depth shape={depth.shape}, lulc shape={lulc.shape}\n"
                            "Inputs must be pre-aligned; check your depth alignment or mask settings."
                        )

                    lulc_valid = np.ones_like(lulc, dtype=bool)
                    if lulc_src.nodata is not None:
                        lulc_valid &= (lulc != lulc_src.nodata)
                    if ignore_ids:
                        lulc_valid &= ~np.isin(lulc, list(ignore_ids))

                    base_mask = floodplain & lulc_valid

                    sda_class = np.zeros(lulc.shape, dtype=np.uint8)
                    if pasture_ids:
                        sda_class[base_mask & np.isin(lulc, list(pasture_ids))] = 3
                    if crop_ids:
                        sda_class[base_mask & np.isin(lulc, list(crop_ids))] = 2
                    if artif_ids:
                        sda_class[base_mask & np.isin(lulc, list(artif_ids))] = 1

                    if use_roads:
                        roads_img, _ = rio_mask(
                            roads_src, [geom_r],
                            crop=True, filled=True,
                            nodata=roads_src.nodata if roads_src.nodata is not None else 0,
                            all_touched=all_touched,
                        )
                        roads = roads_img[0]
                        roads_mask = (roads == 1) & floodplain
                        sda_class[roads_mask] = 4

                    sda_mask = (sda_class > 0)

                    pix_km2 = utilities.pixel_area_km2(depth_tr)
                    area_total_km2 = float(sda_mask.sum() * pix_km2)
                    area_artif_km2 = float((sda_class == 1).sum() * pix_km2)
                    area_crop_km2  = float((sda_class == 2).sum() * pix_km2)
                    area_past_km2  = float((sda_class == 3).sum() * pix_km2)
                    area_roads_km2 = float((sda_class == 4).sum() * pix_km2)

                    pop_in_sda = np.nan
                    if with_pop:
                        target_profile = depth_src.profile.copy()
                        target_profile.update(height=depth.shape[0], width=depth.shape[1], transform=depth_tr, crs=depth_src.crs)
                        pop_reproj = ff.reproject_pop_to_target(pop_src, target_profile)
                        pop_in_sda = float(pop_reproj[sda_mask].sum())

                    print(
                        f"[INFO] {iso3} RP{rp}: SDA_area={area_total_km2:,.2f} km² "
                        f"(artif={area_artif_km2:,.2f}, crop={area_crop_km2:,.2f}, pasture={area_past_km2:,.2f}, roads={area_roads_km2:,.2f}) "
                        + (f"| pop_in_SDA={pop_in_sda:,.0f}" if with_pop else "")
                    )

                    prof_base = depth_src.profile.copy()
                    prof_base.update(
                        height=depth.shape[0],
                        width=depth.shape[1],
                        transform=depth_tr,
                        crs=depth_src.crs,
                        compress="DEFLATE",
                        tiled=True,
                        BIGTIFF="IF_SAFER",
                    )

                    class_tif = os.path.join(str(out_dir), f"sda_class_{iso3}_rp{rp}.tif")
                    prof_class = prof_base.copy()
                    prof_class.update(dtype="uint8", count=1, nodata=0)
                    utilities.save_raster_completely(class_tif, prof_class, sda_class.astype("uint8"), band=1)

                    mask_tif = os.path.join(str(out_dir), f"sda_mask_{iso3}_rp{rp}.tif")
                    utilities.save_raster_completely(mask_tif, prof_class, sda_mask.astype("uint8"), band=1)

                    if write_depthbin:
                        nodata_mask = ~valid_depth
                        depthbin = ff.build_depthbin_index(depth_q, nodata_mask=nodata_mask, max_depth=depthbin_max)
                        db_tif = os.path.join(str(out_dir), f"sda_depthbin_idx_{iso3}_rp{rp}.tif")
                        prof_db = prof_base.copy()
                        prof_db.update(dtype="int16", count=1, nodata=-1)
                        utilities.save_raster_completely(db_tif, prof_db, depthbin.astype("int16"), band=1)

                    metrics.append({
                        "iso3": iso3,
                        "rp": int(rp),
                        "depth_threshold_m": float(depth_threshold_m),
                        "sda_area_km2_total": area_total_km2,
                        "sda_area_km2_artif": area_artif_km2,
                        "sda_area_km2_crop":  area_crop_km2,
                        "sda_area_km2_pasture": area_past_km2,
                        "sda_area_km2_roads": area_roads_km2,
                        "pop_in_sda": float(pop_in_sda) if np.isfinite(pop_in_sda) else np.nan,
                        "depth_raster": str(depth_path),
                        "lulc_raster": str(lulc_path),
                        "roads_raster": str(roads_path) if use_roads else "",
                        "pop_raster": str(pop_path) if with_pop else "",
                    })

    df = pd.DataFrame(metrics)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote → {out_csv}")
    return df




def _service_flow_stats(p, iso3: str, rp: int, sda_class_file: str,
                        sda_mask_file: str, flow_file: str) -> dict:
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
        pix_km2 = utilities.pixel_area_km2(fsrc.transform)
        if p.flood_latitude_correct_area:
            scale = ff.mercator_area_scale(fsrc.transform, 0, fsrc.height)
            area_km2 = pix_km2 * np.broadcast_to(scale, sda_mask.shape)
        else:
            area_km2 = np.full(sda_mask.shape, pix_km2, dtype="float32")
        served = sda_mask & (flow > 0)

        def class_stats(cls: int) -> Tuple[float, float]:
            m = (sda_class == cls) & sda_mask
            return float(area_km2[m].sum()), utilities.safe_mean(flow[m])

        artif_area, artif_mean = class_stats(SDA_CODE["artif"])
        crop_area, crop_mean = class_stats(SDA_CODE["crop"])
        past_area, past_mean = class_stats(SDA_CODE["pasture"])
        road_area, road_mean = class_stats(SDA_CODE["roads"])

        return {
            "iso3": iso3,
            "rp": rp,
            "sda_area_km2_total": float(area_km2[sda_mask].sum()),
            "sda_area_km2_served": float(area_km2[served].sum()),
            "mean_spa_ratio_on_sda": utilities.safe_mean(flow[sda_mask]),
            "sda_area_km2_artif": artif_area, "mean_spa_ratio_artif": artif_mean,
            "sda_area_km2_crop": crop_area, "mean_spa_ratio_crop": crop_mean,
            "sda_area_km2_pasture": past_area, "mean_spa_ratio_pasture": past_mean,
            "sda_area_km2_roads": road_area, "mean_spa_ratio_roads": road_mean,
        }


def _service_flow_one_iso3(p, iso3: str, admin0: gpd.GeoDataFrame, spa_src) -> Tuple[list, int, int]:
    rows: list = []
    iso_dir = os.path.join(p.flood_sda_country_dir, iso3)
    if not hb.path_exists(iso_dir):
        return rows, 0, 0

    sda_files = sorted(glob.glob(os.path.join(str(iso_dir), f"sda_class_{iso3}_rp*.tif")))
    if not sda_files:
        return rows, 0, 0

    processed = skipped = 0
    geom = admin0.loc[admin0.iso3 == iso3].geometry.values

    for cls_file in sda_files:
        rp = int(os.path.splitext(os.path.basename(cls_file))[0].split("rp")[-1])
        mask_file = os.path.join(str(iso_dir), f"sda_mask_{iso3}_rp{rp}.tif")
        flow_file = os.path.join(str(iso_dir), f"service_flow_frac_{iso3}_rp{rp}.tif")

        if not hb.path_exists(mask_file):
            warnings.warn(f"[WARN] {iso3} RP{rp}: missing SDA mask, skipping.")
            continue

        if p.flood_skip_done and utilities.raster_ok(flow_file):
            skipped += 1
            if p.flood_include_existing_in_summary:
                rows.append(_service_flow_stats(p, iso3, rp, cls_file, mask_file, flow_file))
            continue

        with rasterio.open(cls_file) as sda_src:
            arr, tr = rio_mask(sda_src, geom, crop=True, filled=True,
                               nodata=0, all_touched=p.flood_all_touched)
            sda_class = arr[0].astype(np.uint8)

        with rasterio.open(mask_file) as msrc:
            arr, _ = rio_mask(msrc, geom, crop=True, filled=True,
                              nodata=0, all_touched=p.flood_all_touched)
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

        if p.flood_write_service_flow_rasters:
            prof = spa_src.profile.copy()
            prof.update(height=flow.shape[0], width=flow.shape[1], transform=tr,
                        nodata=-9999, dtype="float32", count=1)
            utilities.save_raster_completely(flow_file, prof,
                                np.where(np.isfinite(flow), flow, -9999).astype("float32"))

        rows.append(_service_flow_stats(p, iso3, rp, cls_file, mask_file, flow_file))
        processed += 1

    return rows, processed, skipped


def compute_service_flow_global(p, iso3_list: Optional[List[str]] = None) -> str:
    """Section C driver: SPA -> SDA service flow fraction for every ISO3 x RP."""
    utilities.assert_exists(p.flood_spa_ratio_path, "Upstream SPA ratio raster is required for Section C.")
    admin0 = load_admin0(p.flood_country_vector_path)

    all_rows, total_p, total_s = [], 0, 0
    run_list = iso3_list if iso3_list else sorted(admin0.iso3.unique())

    with rasterio.open(p.flood_spa_ratio_path) as spa_src:
        for iso in run_list:
            rows, processed, skipped = _service_flow_one_iso3(p, iso, admin0, spa_src)
            all_rows.extend(rows)
            total_p += processed
            total_s += skipped

    if all_rows:
        utilities.write_csv(pd.DataFrame(all_rows), p.flood_service_flow_path)
        print(f"[DONE] Global service-flow summary -> {p.flood_service_flow_path}")
        print(f"[INFO] processed={total_p}, skipped_existing={total_s}")
    else:
        print("[INFO] Nothing to do (no SDA outputs found under "
              f"{p.flood_sda_country_dir}; run Section B first).")
    return p.flood_service_flow_path




DEPTH_BINS_M = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6]





SCENARIOS = ("current", "degraded_insitu", "degraded_bare")


SCENARIO_SUFFIX = {"current": "",
                   "degraded_insitu": "__degraded_insitu",
                   "degraded_bare": "__degraded_bare"}



def load_combined_multiplier(
    factors_csv: Optional[str] = None,
    factors_json: Optional[str] = None
) -> float:
    """
    Load EUR2010 -> USD2019 combined_multiplier.

    Accepted sources:

    1) factors_csv:
       a) tidy table with columns like [name,value] or [key,value]
          including 'combined_multiplier'
       b) or columns present anywhere:
          - fx_usd_per_eur_2010_avg
          - inflator_us_2010_to_2019
          => combined = fx * inflator

    2) factors_json:
       recursively find:
          - combined_multiplier
          OR (fx_usd_per_eur_2010_avg AND inflator_us_2010_to_2019)
    """
    if factors_csv:
        df = pd.read_csv(factors_csv)
        df = df.applymap(ff.clean_missing)

        cols = ff.make_colmap(df)

        # --- Case (a): tidy name/value or key/value
        name_col = cols.get("name") or cols.get("key") or cols.get("parameter")
        value_col = cols.get("value") or cols.get("val")

        if name_col and value_col:
            tmp = df[[name_col, value_col]].dropna()
            tmp[name_col] = tmp[name_col].astype(str).str.strip().str.lower()
            hit = tmp[tmp[name_col] == "combined_multiplier"]
            if not hit.empty:
                return float(hit[value_col].iloc[0])

            # fallback: fx + inflator in tidy format
            fx = tmp[tmp[name_col].isin({"fx_usd_per_eur_2010_avg", "fx"})]
            infl = tmp[tmp[name_col].isin({"inflator_us_2010_to_2019", "inflator"})]
            if (not fx.empty) and (not infl.empty):
                return float(fx[value_col].iloc[0]) * float(infl[value_col].iloc[0])

        # --- Case (b): wide single-row columns
        # Search any columns that match expected factor column names
        col_fx = cols.get("fx usd per eur 2010 avg") or cols.get("fx_usd_per_eur_2010_avg")
        col_infl = cols.get("inflator us 2010 to 2019") or cols.get("inflator_us_2010_to_2019")
        col_comb = cols.get("combined multiplier") or cols.get("combined_multiplier")

        if col_comb and df[col_comb].dropna().shape[0] > 0:
            return float(df[col_comb].dropna().iloc[0])

        if col_fx and col_infl:
            fxv = df[col_fx].dropna()
            inv = df[col_infl].dropna()
            if (len(fxv) > 0) and (len(inv) > 0):
                return float(fxv.iloc[0]) * float(inv.iloc[0])

        raise ValueError(f"Could not find combined_multiplier (or fx+inflator) in factors CSV: {factors_csv}")

    if factors_json:
        data = json.loads(open(str(factors_json), encoding="utf-8").read())

        def find_key(obj: Any, keynames: Iterable[str]) -> Optional[float]:
            """DFS search for numeric leaf with a matching key name."""
            if isinstance(obj, dict):
                for k, v in obj.items():
                    k0 = ff.normalize_label(k)
                    if k0 in {ff.normalize_label(kn) for kn in keynames}:
                        try:
                            return float(v)
                        except Exception:
                            pass
                    got = find_key(v, keynames)
                    if got is not None:
                        return got
            elif isinstance(obj, list):
                for it in obj:
                    got = find_key(it, keynames)
                    if got is not None:
                        return got
            return None

        comb = find_key(data, ["combined_multiplier", "combined multiplier"])
        if comb is not None:
            return float(comb)

        fx = find_key(data, ["fx_usd_per_eur_2010_avg", "fx usd per eur 2010 avg", "fx"])
        infl = find_key(data, ["inflator_us_2010_to_2019", "inflator us 2010 to 2019", "inflator"])
        if fx is not None and infl is not None:
            return float(fx) * float(infl)

        raise ValueError(f"Could not find combined_multiplier in factors JSON: {factors_json}")

    raise ValueError("Provide either --factors-csv or --factors-json")


def load_canonical_eur_table(canonical_eur: str, audit_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load a canonical EUR2010 table.

    Supported canonical input forms:
    1) WIDE: ISO3, JRC_Region, LandType, Max_Damage_Euro_per_m2, 0m, 0.5m, ...
       -> converted to LONG: iso3, landtype, depth_m, damage_per_m2 (EUR2010/m²)

    2) LONG: iso3, landtype, depth_m, damage_per_m2
       -> returned as-is (ensuring required columns exist)

    NOTE:
    Your canonical file often is WIDE with depth columns; it will NOT have
    'depth_m' and 'damage_per_m2' as columns. This function handles that.
    """
    df = pd.read_csv(canonical_eur)
    df = df.applymap(ff.clean_missing)

    cols = ff.make_colmap(df)

    # If already long:
    has_long = ("depth_m" in cols or "depth m" in cols) and ("damage_per_m2" in cols or "damage per m2" in cols)
    if has_long:
        iso_col = cols.get("iso3") or cols.get("iso 3") or cols.get("iso")
        land_col = cols.get("landtype") or cols.get("land type") or cols.get("sector")
        depth_col = cols.get("depth_m") or cols.get("depth m") or cols.get("depth")
        dmg_col = cols.get("damage_per_m2") or cols.get("damage per m2") or cols.get("damage")
        out = df[[iso_col, land_col, depth_col, dmg_col]].copy()
        out.columns = ["iso3", "landtype", "depth_m", "damage_per_m2"]
        out["depth_m"] = out["depth_m"].apply(utilities.to_float)
        out["damage_per_m2"] = out["damage_per_m2"].apply(utilities.to_float)
        return out

    # Otherwise, try wide:
    iso_col = cols.get("iso3") or cols.get("iso 3") or cols.get("iso")
    land_col = cols.get("landtype") or cols.get("land type") or cols.get("sector")
    if not iso_col:
        # sometimes ISO3 is uppercase
        iso_col = next((c for c in df.columns if str(c).strip().upper() == "ISO3"), None)
    if not land_col:
        land_col = next((c for c in df.columns if str(c).strip().lower() in {"landtype", "land_type"}), None)

    if iso_col is None or land_col is None:
        msg = f"Canonical EUR table missing ISO3 or LandType. Columns found: {list(df.columns)}"
        if audit_dir:
            hb.create_directories(str(audit_dir))
            hb.write_to_file(msg, os.path.join(str(audit_dir), "diagnostics_canonical_missing_iso_or_landtype.txt"))
        raise ValueError(msg)

    # depth columns are like '0m','0.5m'... OR numeric strings etc
    depth_cols = []
    depth_vals = []
    for c in df.columns:
        d = ff.parse_depth_colname(c)
        if d is not None:
            depth_cols.append(c)
            depth_vals.append(d)

    if len(depth_cols) == 0:
        msg = (
            "Canonical EUR table appears wide but has no depth columns like '0m','0.5m',... "
            f"Columns found: {list(df.columns)}"
        )
        if audit_dir:
            hb.create_directories(str(audit_dir))
            hb.write_to_file(msg, os.path.join(str(audit_dir), "diagnostics_canonical_no_depth_cols.txt"))
        raise ValueError(msg)

    # Melt wide -> long
    tmp = df[[iso_col, land_col] + depth_cols].copy()
    tmp = tmp.rename(columns={iso_col: "iso3", land_col: "landtype"})
    long = tmp.melt(id_vars=["iso3", "landtype"], var_name="depth_col", value_name="damage_per_m2")
    long["depth_m"] = long["depth_col"].apply(ff.parse_depth_colname).astype(float)
    long = long.drop(columns=["depth_col"])

    long["iso3"] = long["iso3"].astype(str).str.strip()
    long["landtype"] = long["landtype"].astype(str).str.strip()
    long["damage_per_m2"] = long["damage_per_m2"].apply(utilities.to_float)

    return long


def build_canonical_from_components(
    fractional_long_csv: str,
    maxdamage_long_csv: str,
    iso3_region_csv: str,
    audit_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Build canonical EUR2010 LONG table from components.

    FRACTIONAL curves LONG must contain:
      - JRC_Region
      - LandType
      - depth_m
      - fraction

    MAX DAMAGE LONG must contain:
      - iso3
      - LandType
      - max_damage_eur_m2   (or similar)
    (Your fixed file is: jrc_max_damage_values_long_with_iso3.csv)

    ISO3->REGION must contain:
      - iso3
      - JRC_Region

    OUTPUT:
      iso3, landtype, depth_m, damage_per_m2   (EUR2010 per m²)
    """
    frac = pd.read_csv(fractional_long_csv).applymap(ff.clean_missing)
    maxd = pd.read_csv(maxdamage_long_csv).applymap(ff.clean_missing)
    iso3reg = pd.read_csv(iso3_region_csv).applymap(ff.clean_missing)

    fcols = ff.make_colmap(frac)
    mcols = ff.make_colmap(maxd)
    rcols = ff.make_colmap(iso3reg)

    # ---- Fraction required columns (underscore/space tolerant!)
    f_reg  = fcols.get("jrc_region") or fcols.get("jrc region")
    f_land = fcols.get("landtype")   or fcols.get("land type")
    f_depth = fcols.get("depth_m") or fcols.get("depth m") or fcols.get("depth")
    f_frac = fcols.get("fraction") or fcols.get("frac")
    if None in (f_reg, f_land, f_depth, f_frac):
        raise ValueError(f"fractional_long_csv missing required columns. Columns={list(frac.columns)}")

    # ---- Maxdamage required columns
    m_iso = mcols.get("iso3") or mcols.get("country iso3") or mcols.get("iso 3")
    m_land = mcols.get("landtype") or mcols.get("land type")
    m_max = (
        mcols.get("max_damage_eur_m2")
        or mcols.get("max damage eur m2")
        or mcols.get("max_damage_euro_per_m2")
        or mcols.get("max damage euro per m2")
        or mcols.get("max_damage_euro_per_m2".replace("_", " "))
    )
    if m_iso is None:
        raise ValueError(f"maxdamage_long_csv missing ISO3 column. Columns={list(maxd.columns)}")
    if m_land is None or m_max is None:
        raise ValueError(f"maxdamage_long_csv missing LandType or max-damage column. Columns={list(maxd.columns)}")

    # ---- Region required columns
    r_iso = rcols.get("iso3") or rcols.get("iso 3")
    r_reg = rcols.get("jrc_region") or rcols.get("jrc region")
    if r_iso is None or r_reg is None:
        raise ValueError(f"iso3_region_csv missing required columns. Columns={list(iso3reg.columns)}")

    # ---- Clean & normalize
    frac2 = frac[[f_reg, f_land, f_depth, f_frac]].copy()
    frac2.columns = ["JRC_Region", "LandType", "depth_m", "fraction"]
    frac2["JRC_Region"] = frac2["JRC_Region"].astype(str).str.strip()
    frac2["LandType"] = frac2["LandType"].astype(str).apply(ff.normalize_landtype_label).str.strip()
    frac2["depth_m"] = frac2["depth_m"].apply(utilities.to_float)
    frac2["fraction"] = frac2["fraction"].apply(utilities.to_float)

    max2 = maxd[[m_iso, m_land, m_max]].copy()
    max2.columns = ["iso3", "LandType", "max_damage_eur_m2"]
    max2["iso3"] = max2["iso3"].astype(str).str.strip()
    max2["LandType"] = max2["LandType"].astype(str).apply(ff.normalize_landtype_label).str.strip()
    max2["max_damage_eur_m2"] = max2["max_damage_eur_m2"].apply(utilities.to_float)

    reg2 = iso3reg[[r_iso, r_reg]].copy()
    reg2.columns = ["iso3", "JRC_Region"]
    reg2["iso3"] = reg2["iso3"].astype(str).str.strip()
    reg2["JRC_Region"] = reg2["JRC_Region"].astype(str).str.strip()

    # ---- Join max damages with regions
    mx = max2.merge(reg2, on="iso3", how="left")

    # Diagnostics for missing regions (should be none if your mapping is complete)
    missing_reg = mx[mx["JRC_Region"].isna()][["iso3", "LandType"]].drop_duplicates()
    if len(missing_reg) > 0:
        warnings.warn(f"[WARN] Missing JRC_Region for {len(missing_reg)} iso3×landtype rows; these will be dropped.")
        if audit_dir:
            hb.create_directories(str(audit_dir))
            missing_reg.to_csv(os.path.join(str(audit_dir), "diagnostics_missing_jrc_region_for_iso3.csv"), index=False)

    mx = mx.dropna(subset=["JRC_Region"])

    # ---- Join with fractions on (region, landtype, depth)
    out = mx.merge(
        frac2,
        left_on=["JRC_Region", "LandType"],
        right_on=["JRC_Region", "LandType"],
        how="left"
    )

    # After join, fraction varies by depth; ensure depth present
    # NOTE: if landtype labels mismatch, fraction will be NaN.
    missing_frac = out[out["fraction"].isna()][["LandType", "JRC_Region"]].drop_duplicates()
    if len(missing_frac) > 0:
        warnings.warn(
            f"[WARN] Missing fractions for {len(missing_frac)} LandType/JRC_Region combos. "
            f"Examples:\n{missing_frac.head(10).to_string(index=False)}"
        )
        if audit_dir:
            hb.create_directories(str(audit_dir))
            missing_frac.to_csv(os.path.join(str(audit_dir), "diagnostics_missing_fraction_landtype_region.csv"), index=False)

    # Compute depth damages
    out["damage_per_m2"] = out["max_damage_eur_m2"] * out["fraction"]

    eur_long = out[["iso3", "LandType", "depth_m", "damage_per_m2"]].copy()
    eur_long = eur_long.rename(columns={"LandType": "landtype"})
    eur_long["landtype"] = eur_long["landtype"].apply(ff.normalize_landtype_label)
    eur_long["depth_m"] = eur_long["depth_m"].apply(utilities.to_float)
    eur_long["damage_per_m2"] = eur_long["damage_per_m2"].apply(utilities.to_float)

    # Drop bad rows
    eur_long = eur_long.dropna(subset=["iso3", "landtype", "depth_m", "damage_per_m2"])

    return eur_long


def build_damage_tables(p):
    """Step 4A: the canonical JRC EUR2010 table as USD2019 sector and SDA tables.

    Every value is multiplied by one combined factor, FX at 2010 times the US inflator to 2019.
    Cropland's curve is copied to pasture when `flood_set_pasture_equal_crop` is set, which is the
    JRC default: pasture has no curve of its own.
    """
    multiplier = load_combined_multiplier(factors_csv=p.flood_currency_factors_path, factors_json=None)
    eur_long = load_canonical_eur_table(p.flood_canonical_eur_path,
                                        audit_dir=p.flood_currency_audit_dir)
    eur_long["landtype"] = eur_long["landtype"].apply(ff.normalize_landtype_label)

    usd = eur_long.copy()
    usd["damage_per_m2"] = usd["damage_per_m2"] * float(multiplier)
    usd["currency"] = "USD2019"
    usd["iso3"] = usd["iso3"].astype(str).str.strip()
    usd["depth_m"] = usd["depth_m"].apply(utilities.to_float)
    usd["damage_per_m2"] = usd["damage_per_m2"].apply(utilities.to_float)
    usd = usd.dropna(subset=["iso3", "landtype", "depth_m", "damage_per_m2"])

    sector_long = usd[["iso3", "landtype", "depth_m", "damage_per_m2", "currency"]].copy()
    sector_long["landtype"] = sector_long["landtype"].apply(ff.sector_label_from_landtype)
    utilities.write_csv(sector_long, p.flood_damage_long_path)
    utilities.write_csv(ff.pivot_wide(sector_long, group_cols=["iso3", "landtype"],
                         value_col="damage_per_m2"), p.flood_damage_wide_table_path)

    sda = usd.copy()
    sda["sda_type"] = sda["landtype"].apply(ff.landtype_to_sda)
    sda = sda.dropna(subset=["sda_type"])
    sda_long = sda[["iso3", "sda_type", "depth_m", "damage_per_m2", "currency"]].copy()
    if p.flood_set_pasture_equal_crop:
        pasture = sda_long[sda_long["sda_type"] == "crop"].copy()
        pasture["sda_type"] = "pasture"
        sda_long = pd.concat([sda_long, pasture], ignore_index=True)
    utilities.write_csv(sda_long, p.flood_sda_damage_long_path)
    utilities.write_csv(ff.pivot_wide(sda_long, group_cols=["iso3", "sda_type"],
                         value_col="damage_per_m2"), p.flood_sda_damage_wide_path)

    hb.log('Step 4A: %d sector rows, %d SDA rows, multiplier %.7f'
           % (len(sector_long), len(sda_long), float(multiplier)))
    return p.flood_sda_damage_wide_path

def load_damage_table_wide(path: str) -> Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]:
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


def _find_depth_raster(p, rp: int) -> Optional[str]:
    for pattern in (f"*rp{rp}y*matchLULC*.tif", f"*rp{rp}*matchLULC*.tif", f"*rp{rp}*.tif"):
        hits = sorted(glob.glob(os.path.join(p.flood_depth_aligned_path, pattern)))
        if hits:
            return hits[0]
    return None


def _open_amp(p, scenario: str, rp: int):
    """Open the amplification raster for one scenario x RP, or None."""
    if scenario == "current":
        return None
    path = os.path.join(p.flood_amplification_path,
                        p.flood_amplification_pattern.format(scenario=scenario, rp=rp))
    if not hb.path_exists(path):
        warnings.warn(
            f"[WARN] no amplification raster for {scenario} RP{rp} at {path}. "
            f"Build it with counterfactual/build_amplification_rasters.py. "
            f"Falling back to current depths (GEP for this RP will be zero).")
        return None
    return rasterio.open(path)


def compute_pixel_damages(p, iso3_list: Optional[List[str]] = None,
                          scenario: str = "current",
                          tile: int = 2048) -> List[str]:
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

    utilities.assert_exists(p.flood_sda_damage_wide_path, "Run step 4A (build_damage_tables) first.")
    utilities.assert_exists(p.flood_sda_raster_path, "Global SDA raster is required (Section A).")

    curves = load_damage_table_wide(p.flood_sda_damage_wide_path)
    admin0 = load_admin0(p.flood_country_vector_path)

    with rasterio.open(p.flood_sda_raster_path) as sda_ds:
        utilities.warn_if_geographic(sda_ds, "SDA raster")
        pix_area_m2 = utilities.pixel_area_m2(sda_ds.transform)
        sda_meta = sda_ds.meta.copy()

    print(f"[INFO] SDA raster: {p.flood_sda_raster_path}")
    if p.flood_latitude_correct_area:
        print(f"[INFO] Pixel area (m^2): {pix_area_m2:,.1f} at the equator, "
              f"scaled by cos^2(lat) per row")
    else:
        print(f"[INFO] Pixel area (m^2): {pix_area_m2:,.1f} UNCORRECTED "
              f"-- overstates area 4x at 60N")
    print(f"[INFO] RPs: {p.flood_return_periods} | scenario: {scenario} | tile: {tile}")

    run_list = iso3_list if iso3_list else sorted(admin0.iso3.unique())
    written: List[str] = []

    for iso3 in run_list:
        rows = admin0[admin0.iso3 == iso3]
        if rows.empty:
            continue
        geom = unary_union(rows.geometry.values)
        if len(rows) > 1:
            print(f"[INFO] {iso3}: unioned {len(rows)} Admin0 features.")

        out_dir = os.path.join(p.flood_valuation_country_dir, iso3)
        hb.create_directories(str(out_dir))
        ras_dir = os.path.join(str(out_dir), "rasters")
        if p.flood_write_damage_rasters:
            hb.create_directories(str(ras_dir))

        records = []
        for rp in p.flood_return_periods:
            depth_path = _find_depth_raster(p, rp)
            if depth_path is None:
                print(f"[WARN] {iso3}: no depth raster for RP={rp} in {p.flood_depth_aligned_path}")
                continue

            amp_src = _open_amp(p, scenario, rp)
            try:
                with rasterio.open(depth_path) as dds, rasterio.open(p.flood_sda_raster_path) as sds:
                    utilities.assert_same_grid(dds, sds, f"depth_rp{rp}", "sda")

                    win = _country_window(dds, geom)
                    if win is None:
                        print(f"[WARN] {iso3} RP{rp}: empty window in raster space")
                        continue
                    win_tr = window_transform(win, dds.transform)

                    dst = None
                    if p.flood_write_damage_rasters:
                        meta = sda_meta.copy()
                        meta.update(driver="GTiff", height=int(win.height),
                                    width=int(win.width), transform=win_tr,
                                    count=1, dtype="float32", nodata=0.0,
                                    compress="DEFLATE", tiled=True,
                                    BIGTIFF="IF_SAFER")
                        out_tif = os.path.join(str(ras_dir), f"damage_USD2019_rp{rp}{suffix}.tif")
                        tmp_tif = str(out_tif) + ".tmp"
                        dst = rasterio.open(tmp_tif, "w", **meta)

                    totals_by_sda = {t: 0.0 for t in set(sda_code_to_type(p).values())}
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

                        valid &= (depth > p.flood_depth_threshold_m)
                        if not valid.any():
                            continue

                        sda = sds.read(1, window=sub).astype("int16")
                        dmg_tile = np.zeros(depth.shape, dtype="float32") if dst else None
                        touched = False

                        # True ground area per pixel. Depends only on latitude,
                        # so one value per row broadcast across the tile.
                        if p.flood_latitude_correct_area:
                            area = pix_area_m2 * np.broadcast_to(
                                ff.mercator_area_scale(dds.transform,
                                                    int(sub.row_off),
                                                    int(sub.height)),
                                depth.shape)
                        else:
                            area = np.full(depth.shape, pix_area_m2, dtype="float32")

                        for code, sda_type in sda_code_to_type(p).items():
                            m = valid & (sda == code)
                            if not m.any():
                                continue
                            key = (iso3, sda_type)
                            if key not in curves:
                                continue
                            xs, ys = curves[key]
                            dmg_usd = ff.interp_damage_per_m2(depth[m], xs, ys, mode=p.flood_damage_depth_mode) * area[m]
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
                        os.replace(tmp_tif, str(out_tif))

                    missing = [t for (i3, t) in
                               [(iso3, t) for t in sda_code_to_type(p).values()]
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
            out_csv = os.path.join(str(out_dir), f"step4b_damage_by_rp_USD2019{suffix}.csv")
            rec_df = pd.DataFrame(records).sort_values(["iso3", "rp"])
            if p.flood_apply_service_flow and scenario == "current":
                rec_df = ff._attach_service_flow(rec_df, iso3, _load_service_flow_table(p))
            utilities.write_csv(rec_df, out_csv)
            written.append(out_csv)
            print(f"[OK] Wrote summary -> {out_csv} (rows={len(rec_df)})")
        else:
            print(f"[WARN] {iso3}: no RP records written (missing depth rasters?)")

    return written


def _load_service_flow_table(p) -> Optional[pd.DataFrame]:
    """Read Section C's summary once, keyed by (iso3, rp)."""
    if not hb.path_exists(p.flood_service_flow_path):
        warnings.warn(
            f"[WARN] flood_apply_service_flow is on but no service-flow summary "
            f"at {p.flood_service_flow_path}. Run Section C first. Attributed damages "
            f"will be omitted.")
        return None
    flow = pd.read_csv(p.flood_service_flow_path)
    needed = {"iso3", "rp", "mean_spa_ratio_on_sda"}
    if not needed.issubset(flow.columns):
        warnings.warn(f"[WARN] {p.flood_service_flow_path} lacks {needed - set(flow.columns)}; "
                      "attributed damages will be omitted.")
        return None
    flow["iso3"] = flow["iso3"].astype(str).str.upper().str.strip()
    flow["rp"] = pd.to_numeric(flow["rp"], errors="coerce")
    return flow


def load_protection_table(p) -> Optional[pd.DataFrame]:
    """
    Read the FLOPROS-derived flood protection standards, one return period per
    ISO3. Expected columns: iso3, protection_rp.
    """
    if not p.flood_protection_path or not hb.path_exists(p.flood_protection_path):
        return None
    df = pd.read_csv(p.flood_protection_path)
    c_iso = utilities.find_col(df, ("iso3", "iso_a3", "adm0_a3"))
    c_rp = utilities.find_col(df, ("protection_rp", "protection", "flopros", "merged_rp",
                         "protection_standard", "rp"))
    if c_iso is None or c_rp is None:
        warnings.warn(f"[WARN] {p.flood_protection_path} needs iso3 + protection_rp; "
                      f"have {list(df.columns)}. Protection split skipped.")
        return None
    out = df[[c_iso, c_rp]].rename(columns={c_iso: "iso3", c_rp: "protection_rp"})
    out["iso3"] = out["iso3"].astype(str).str.upper().str.strip()
    out["protection_rp"] = pd.to_numeric(out["protection_rp"], errors="coerce")
    out = out.dropna(subset=["protection_rp"]).drop_duplicates("iso3")

    # Prefer an evidence column from prep_flopros.py; else use the embedded set.
    c_ev = utilities.find_col(df, ("protection_evidence", "evidence"))
    if c_ev:
        ev = df[[c_iso, c_ev]].rename(columns={c_iso: "iso3", c_ev: "protection_evidence"})
        ev["iso3"] = ev["iso3"].astype(str).str.upper().str.strip()
        out = out.merge(ev.drop_duplicates("iso3"), on="iso3", how="left")
    else:
        out["protection_evidence"] = np.where(
            out.iso3.isin(documented_protection_iso3(p)), "documented", "gdp_inferred")

    n_doc = int((out.protection_evidence == "documented").sum())
    print(f"[INFO] protection standards: {len(out)} countries "
          f"({n_doc} documented, {len(out)-n_doc} GDP-inferred)")

    if p.flood_protection_documented_only:
        # NaN means "no truncation applied" downstream -- the country is still
        # reported, just untruncated, with the reason recorded.
        mask = out.protection_evidence != "documented"
        out.loc[mask, "protection_rp"] = np.nan
        print(f"[INFO] truncation restricted to documented protection; "
              f"{int(mask.sum())} countries reported untruncated")
    return out


def _write_step4c(p, iso3_dir: str, iso3: str, ead: float, *, rps_used: List[int],
                  status: str, detail: str, ead_attributed: float = np.nan,
                  suffix: str = "", ead_nc: float = np.nan,
                  protection_rp: Optional[float] = None,
                  protection_evidence: str = "") -> str:
    out_csv = os.path.join(str(iso3_dir), f"step4c_ead_USD2019{suffix}.csv")
    utilities.write_csv(pd.DataFrame([{
        "iso3": iso3,
        "ead_usd2019": float(ead),
        # Attribution of residual damage to naturally-served floodplains.
        # NOT avoided damage.
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
        "tail_mode": p.flood_ead_tail_mode,
        "add_p1_zero": bool(p.flood_ead_add_p1_zero),
        "enforce_monotone": bool(p.flood_ead_enforce_monotone),
        "applied_service_flow": bool(p.flood_apply_service_flow),
        "status": status,
        "detail": detail,
    }]), out_csv)
    return out_csv


def compute_ead_by_country(p, scenario: str = "current") -> pd.DataFrame:
    """
    Step 4C: convert Step 4B's damage-by-RP into an EAD for every ISO3 folder.

    A step4c CSV is written for EVERY ISO3 folder, even when Step 4B is
    missing or empty, because Step 4D aggregates across folders and a silently
    absent file is indistinguishable from a genuine zero.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}")
    suffix = SCENARIO_SUFFIX[scenario]

    prot_tbl = load_protection_table(p) if p.flood_report_protection_split else None
    if p.flood_report_protection_split and prot_tbl is None:
        warnings.warn("[WARN] protection split requested but no usable table; "
                      "NC/NC+ columns will be NaN.")

    iso3_dirs = list_iso3_dirs(p.flood_valuation_country_dir)
    print(f"[INFO] Step 4C [{scenario}]: {len(iso3_dirs)} ISO3 folders under {p.flood_valuation_country_dir}")

    results = []
    for iso3_dir in iso3_dirs:
        iso3 = os.path.basename(str(iso3_dir)).upper()
        step4b_csv = os.path.join(str(iso3_dir), f"step4b_damage_by_rp_USD2019{suffix}.csv")

        if not hb.path_exists(step4b_csv):
            _write_step4c(p, iso3_dir, iso3, np.nan, rps_used=[], status="missing_step4b",
                          detail="no step4b file found", suffix=suffix)
            results.append({"iso3": iso3, "status": "missing_step4b", "ead_usd2019": np.nan})
            continue

        try:
            df = pd.read_csv(step4b_csv)
        except Exception as e:
            _write_step4c(p, iso3_dir, iso3, np.nan, rps_used=[], status="bad_step4b_csv",
                          detail=f"failed_read:{e}", suffix=suffix)
            results.append({"iso3": iso3, "status": "bad_step4b_csv", "ead_usd2019": np.nan})
            continue

        c_rp = utilities.find_col(df, ("rp", "return period", "return_period", "rp_years"))
        c_dmg = utilities.find_col(df, ("damage_total_usd2019", "total_damage_usd2019",
                              "damage_usd2019", "total_damage", "damage"))
        if c_rp is None or c_dmg is None:
            _write_step4c(p, iso3_dir, iso3, np.nan, rps_used=[], status="missing_columns",
                          detail=f"need rp+damage cols; have={list(df.columns)}", suffix=suffix)
            results.append({"iso3": iso3, "status": "missing_columns", "ead_usd2019": np.nan})
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
            rp, dmg, add_p1_zero=p.flood_ead_add_p1_zero, tail_mode=p.flood_ead_tail_mode,
            enforce_monotone=p.flood_ead_enforce_monotone, protection_rp=prot)

        # Second, attributed series -- integrated with identical boundary
        # assumptions so the two EADs are directly comparable.
        ead_attr = np.nan
        c_attr = utilities.find_col(df, ("damage_attributed_to_spa_usd2019",))
        if p.flood_apply_service_flow and c_attr is not None:
            attr = pd.to_numeric(df[c_attr], errors="coerce").to_numpy()
            if np.isfinite(attr).any():
                ead_attr, _, _, _ = ff.compute_ead_from_points(
                    rp, attr, add_p1_zero=p.flood_ead_add_p1_zero, tail_mode=p.flood_ead_tail_mode,
                    enforce_monotone=p.flood_ead_enforce_monotone)

        rps_used = sorted(int(x) for x in pd.Series(rp).dropna().unique() if x > 0)
        _write_step4c(p, iso3_dir, iso3, ead, rps_used=rps_used, status="ok",
                      detail=";".join(msgs), ead_attributed=ead_attr, suffix=suffix,
                      ead_nc=ead_nc, protection_rp=prot,
                      protection_evidence=prot_ev)

        if p.flood_ead_write_points:
            pts2 = pts.copy()
            pts2.insert(0, "iso3", iso3)
            utilities.write_csv(pts2, os.path.join(str(iso3_dir), f"step4c_ead_USD2019{suffix}__integration_points.csv"))

        if len(rps_used) < 4:
            warnings.warn(f"[WARN] {iso3}: only {len(rps_used)} unique RP points "
                          f"({rps_used}). Fewer than four makes the EAD integral "
                          f"unreliable -- check the depth rasters for this country.")

        results.append({"iso3": iso3, "status": "ok", "ead_usd2019": float(ead),
                        "ead_attributed_to_spa_usd2019": float(ead_attr)})
        print(f"[OK] {iso3}: EAD = {utilities.fmt_usd(ead)}")

    status_df = pd.DataFrame(results)
    utilities.write_csv(status_df, os.path.join(p.flood_global_export_dir, f"step4c_global_status{suffix}.csv"))
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

    # A country whose step 4B did not finish has no EAD, which is not the same as an EAD of zero.
    # Recording it as zero is what made France, Australia and Norway look like legitimate zeros
    # after they were OOM-killed, so the failures stop the run and name themselves.
    failed = status_df[status_df["status"] != "ok"] if not status_df.empty else status_df
    if not failed.empty:
        by_status = failed.groupby("status")["iso3"].apply(lambda s: ", ".join(sorted(s)))
        raise ValueError(
            'Step 4C [%s]: %d of %d countries produced no EAD, so the global total would be short '
            'by however much they are worth:\n%s'
            % (scenario, len(failed), len(iso3_dirs),
               "\n".join('  %-16s %s' % (k, v) for k, v in by_status.items())))
    return status_df


def list_iso3_dirs(outputs_root: str) -> List[str]:
    """
    ISO3 dirs are direct children of outputs_root with 3-letter names.
    Example: outputs/GMB, outputs/USA, outputs/ZAF
    """
    if not hb.path_exists(str(outputs_root)):
        raise FileNotFoundError(f"outputs_root does not exist: {outputs_root}")
    return sorted(os.path.join(str(outputs_root), d) for d in os.listdir(str(outputs_root))
                  if os.path.isdir(os.path.join(str(outputs_root), d)) and len(d) == 3)


def find_step4c_file(iso3_dir: str, ead_filename: str) -> Optional[str]:
    """
    Find Step 4C EAD CSV within an ISO3 directory.

    Priority:
      1) exact filename at ISO3 root
      2) exact filename anywhere under ISO3 (recursive)
      3) heuristic glob patterns (recursive)

    Returns the newest match by modified time if multiple are found.
    """
    p0 = os.path.join(str(iso3_dir), ead_filename)
    if hb.path_exists(p0):
        return p0

    hits = glob.glob(os.path.join(str(iso3_dir), "**", ead_filename), recursive=True)
    if hits:
        return max(hits, key=os.path.getmtime)

    patterns = [
        "*step4c*ead*USD2019*.csv",
        "*step4c*EAD*USD2019*.csv",
        "*ead*USD2019*.csv",
        "*step4c*ead*.csv",
    ]
    cand: List[str] = []
    for pat in patterns:
        cand.extend(glob.glob(os.path.join(str(iso3_dir), "**", pat), recursive=True))
    if cand:
        return max(cand, key=os.path.getmtime)

    return None


def read_step4c_ead_robust(step4c_csv: str, iso3_hint: str) -> pd.DataFrame:
    """
    Read per-country Step 4C EAD CSV and return standardized:
      iso3, ead_usd2019

    Supports two common formats:
      A) wide table with an EAD column (and optionally iso3 column)
      B) metric/value table (e.g., metric='ead_usd2019', value=...)

    If no iso3 column exists, iso3_hint is used.
    """
    df = pd.read_csv(step4c_csv)

    # Case B: metric/value format
    c_metric = utilities.find_col(df, ("metric",))
    c_value = utilities.find_col(df, ("value", "val"))
    if c_metric is not None and c_value is not None:
        m = df[c_metric].astype(str).str.strip().str.lower()
        # Accept any row whose metric contains 'ead' and 'usd2019'
        mask = m.str.contains("ead") & (m.str.contains("usd2019") | m.str.contains("2019"))
        if mask.any():
            v = utilities.to_float(df.loc[mask, c_value].iloc[0])
            return pd.DataFrame([{"iso3": iso3_hint, "ead_usd2019": v}])

    # Case A: table with EAD column
    c_iso = utilities.find_col(df, ("iso3", "country", "country_iso3"))
    c_ead = utilities.find_col(df, (
        "ead_usd2019", "ead", "expected annual damage", "expected_annual_damage",
        "ead_total_usd2019", "ead_total", "ead_usd_2019"
    ))
    if c_ead is None:
        raise ValueError(f"Could not find an EAD column in: {step4c_csv} (cols={list(df.columns)})")

    if c_iso is None:
        df["iso3"] = iso3_hint
        c_iso = "iso3"

    out = df[[c_iso, c_ead]].copy()
    out.columns = ["iso3", "ead_usd2019"]
    out["iso3"] = out["iso3"].astype(str).str.strip().str.upper()
    out["ead_usd2019"] = out["ead_usd2019"].apply(utilities.to_float)
    out = out.dropna(subset=["iso3", "ead_usd2019"])
    return out


def compute_global_totals(country_df: pd.DataFrame) -> pd.DataFrame:
    """
    Global total EAD is the sum across ISO3 entries.
    Units: USD2019/year (absolute, not scaled).
    """
    total = float(country_df["ead_usd2019"].sum(skipna=True))
    return pd.DataFrame([{"metric": "global_total_ead_usd2019", "value": total}])


def export_global_results(p, df_countries):
    """Step 4D: consolidate the per-country Step 4C files into the global table.

    Country attributes come from `df_countries`, which every service shares, rather than from
    columns on the boundary geometry. That is what lets the pipeline run against any Admin0 layer
    carrying an ISO3 column.

    Args:
        df_countries (pd.DataFrame): the shared country table, from initialize_country_paths.

    Returns:
        str: the per-country table written.
    """
    admin0 = load_admin0(p.flood_country_vector_path)
    iso_col = 'iso3'
    name_col = utilities.find_col(admin0, ("name", "country", "country_name", "admin", "name_long",
                                 "name_en", "sovereignt"))
    keep = [iso_col] + ([name_col] if name_col else [])
    enrich = admin0[keep].drop_duplicates(subset=[iso_col]).copy()
    if name_col:
        enrich = enrich.rename(columns={name_col: "country_name"})

    # region_wb from the shared country table, not from the geometry. Reading it off the boundary
    # is what tied this step to one particular Admin0 file.
    countries = utilities.collapse_countries_to_r250(df_countries)
    region = countries[['iso3_r250_label', p.flood_region_col]].rename(
        columns={'iso3_r250_label': 'iso3'}) if p.flood_region_col in countries.columns else None
    if region is not None:
        enrich = enrich.merge(region, on='iso3', how='left')

    rows = []
    iso3_dirs = list_iso3_dirs(p.flood_valuation_country_dir)
    for iso3_dir in iso3_dirs:
        iso3 = os.path.basename(str(iso3_dir)).upper()
        step4c = find_step4c_file(iso3_dir, EAD_FILE_NAME)
        if step4c is None:
            continue
        part = read_step4c_ead_robust(step4c, iso3_hint=iso3)
        if not part.empty:
            rows.append(part)

    if rows:
        country = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["iso3"], keep="last")
    else:
        country = pd.DataFrame(columns=["iso3", "ead_usd2019"])
    country["iso3"] = country["iso3"].astype(str).str.upper().str.strip()

    if p.flood_fill_missing_zero:
        country = enrich[['iso3']].merge(country, on='iso3', how='left')
        country['ead_usd2019'] = pd.to_numeric(country['ead_usd2019'], errors='coerce').fillna(0.0)
    country = country.merge(enrich, on='iso3', how='left')

    total = float(pd.to_numeric(country['ead_usd2019'], errors='coerce').sum())
    if len(country) and total <= 0.0:
        raise ValueError(
            'Step 4D assembled %d countries with a total EAD of $0. The Step 4C files exist but '
            'carry nothing, which means the valuation ran unconfigured.' % len(country))

    utilities.write_csv(country, p.flood_country_ead_path)
    utilities.write_csv(compute_global_totals(country), p.flood_global_totals_path)
    hb.log('Step 4D: %d countries, total EAD $%s' % (len(country), format(total, ',.0f')))
    return p.flood_country_ead_path

def export_attributed_summary(p) -> Optional[str]:
    """
    Companion to Step 4D.

    The preserved 4D script only knows about `ead_usd2019`, so it cannot carry
    the attributed series. Rather than fork that script, this walks the Step 4C
    files directly and writes a parallel country table with both columns plus
    the Admin0 name/region join, so the two numbers can be compared per country
    and per region.

    Skipped entirely when flood_apply_service_flow is off.
    """
    if not p.flood_apply_service_flow:
        return None

    rows = []
    for iso3_dir in list_iso3_dirs(p.flood_valuation_country_dir):
        f = os.path.join(str(iso3_dir), EAD_FILE_NAME)
        if not hb.path_exists(f):
            continue
        # A country that cannot be read must stop the run: it would otherwise leave the global
        # total silently, and the sum would still look plausible.
        rows.append(pd.read_csv(f))

    if not rows:
        warnings.warn("[WARN] No Step 4C files found; attributed summary not written.")
        return None

    df = pd.concat(rows, ignore_index=True)
    df["iso3"] = df["iso3"].astype(str).str.upper().str.strip()

    admin0 = load_admin0(p.flood_country_vector_path)
    keep = ["iso3"]
    name_col = utilities.pick_name_column(admin0)
    if name_col:
        keep.append(name_col)
    if p.flood_region_col in admin0.columns:
        keep.append(p.flood_region_col)
    enrich = admin0[keep].drop_duplicates(subset=["iso3"])
    if name_col:
        enrich = enrich.rename(columns={name_col: "country_name"})

    df = enrich.merge(df, on="iso3", how="left")

    gross = pd.to_numeric(df["ead_usd2019"], errors="coerce")
    attr = pd.to_numeric(df.get("ead_attributed_to_spa_usd2019"), errors="coerce")
    df["attributed_share"] = np.where(gross > 0, attr / gross, np.nan)

    out_csv = p.flood_country_ead_with_service_flow_path
    utilities.write_csv(df, out_csv)

    print(f"[OK] Attributed summary -> {out_csv}")
    print(f"     gross global EAD      = {utilities.fmt_usd(np.nansum(gross))}")
    print(f"     attributed to SPA     = {utilities.fmt_usd(np.nansum(attr))}")
    if np.nansum(gross) > 0:
        print(f"     attributed share      = {np.nansum(attr) / np.nansum(gross):.1%}")
    print("     NOTE: attribution of residual damage, NOT avoided damage.")
    return out_csv


def compute_flood_gep(p) -> Optional[str]:
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
        for d in list_iso3_dirs(p.flood_valuation_country_dir):
            f = os.path.join(str(d), f"step4c_ead_USD2019{suffix}.csv")
            if hb.path_exists(f):
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

    admin0 = load_admin0(p.flood_country_vector_path)
    keep = ["iso3"]
    name_col = utilities.pick_name_column(admin0)
    if name_col:
        keep.append(name_col)
    if p.flood_region_col in admin0.columns:
        keep.append(p.flood_region_col)
    enrich = admin0[keep].drop_duplicates(subset=["iso3"])
    if name_col:
        enrich = enrich.rename(columns={name_col: "country_name"})
    df = enrich.merge(df, on="iso3", how="right")

    utilities.write_csv(df, p.flood_gep_path)

    print(f"[OK] Flood GEP table -> {p.flood_gep_path}")
    print(f"     EAD current           = {utilities.fmt_usd(np.nansum(c))}")
    for lab in found:
        g = np.nansum(df[f"gep_flood_{lab}_usd2019"])
        print(f"     GEP ({lab:6s})         = {utilities.fmt_usd(g)}")
    if len(found) == 2:
        gi = np.nansum(df["gep_flood_insitu_usd2019"])
        gb = np.nansum(df["gep_flood_bare_usd2019"])
        print(f"     -> bracketed range    = {utilities.fmt_usd(gi)} to {utilities.fmt_usd(gb)}")
        print("     Use 'bare' for any combined table with erosion (same baseline).")
    if p.flood_report_protection_split:
        nc_cols = [c for c in df.columns if c.startswith("ead_nc_")]
        print("     NC / NC+ columns present (Vallecillo Eq.7 sensitivity).")
        if nc_cols:
            covered = pd.to_numeric(df[nc_cols[0]], errors="coerce").notna()
            share = (pd.to_numeric(df.ead_current_usd2019, errors="coerce")[covered].sum()
                     / max(np.nansum(c), 1e-9))
            print(f"     truncation applied to {int(covered.sum())} countries "
                  f"= {share:.1%} of current EAD (documented protection only);")
            print("     the remainder are reported untruncated.")
    return p.flood_gep_path


def run_gep_chain(p, skip_damage_tables: bool = True,
                  scenarios: Optional[List[str]] = None) -> dict:
    """
    Paired-scenario driver: run 4B/4C for current plus each degraded scenario,
    then difference. Roughly triples Step 4B cost with both degraded scenarios,
    so it is a deliberate separate task rather than part of run_valuation_chain().
    """
    out = {}
    if not skip_damage_tables:
        out["damage_tables"] = build_damage_tables(p)

    run = scenarios if scenarios else list(SCENARIOS)
    for n, sc in enumerate(run, 1):
        print(f"\n=== scenario {n} of {len(run)}: {sc} ===")
        out[f"step4b_{sc}"] = compute_pixel_damages(p, scenario=sc)
        out[f"step4c_{sc}"] = compute_ead_by_country(p, scenario=sc)

    out["gep"] = compute_flood_gep(p)
    return out


def run_valuation_chain(p, df_countries, skip_damage_tables: bool = False) -> dict:
    """Section D driver: 4A -> 4B -> 4C -> 4D (+ attributed companion export)."""
    out = {}
    if not skip_damage_tables:
        out["damage_tables"] = build_damage_tables(p)
    out["step4b"] = compute_pixel_damages(p)
    out["step4c"] = compute_ead_by_country(p)
    out["step4d"] = export_global_results(p, df_countries)
    out["attributed"] = export_attributed_summary(p)
    return out




def load_admin0(path: str, layer: Optional[str] = None) -> gpd.GeoDataFrame:
    """
    Load Admin0 polygons, normalize the ISO3 column to lowercase 'iso3',
    repair invalid geometries with buffer(0), drop empties.
    """
    path = str(path)
    utilities.assert_exists(path, "Admin0 boundary file is required.")
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"Admin0 has no CRS: {path}")
    iso_col = utilities.pick_iso3_column(gdf)
    if iso_col is None:
        raise ValueError(f"No ISO3-like column found. Columns: {list(gdf.columns)}")
    gdf["iso3"] = gdf[iso_col].astype(str).str.upper().str.strip()
    gdf["geometry"] = gdf["geometry"].buffer(0)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return gdf



# ---------------------------------------------------------------------------------------------
# Section B smart-skip: the run signature and what it compares against. These open files, so
# they belong on this side of the split rather than in flood_functions.
# ---------------------------------------------------------------------------------------------
def signature_path(out_dir: str, iso3: str) -> str:
    return os.path.join(out_dir, f"sda_run_signature_{iso3}.json")


def build_run_signature(*, depth_threshold: float, all_touched: bool, include_pasture: bool,
                        use_roads: bool, with_pop: bool, write_depthbin: bool, depthbin_max: float,
                        lulc_path: str, mapping_path: str, roads_path: str, pop_path: str,
                        depth_dir: str, depth_json: str, rp_map: dict[int, str]) -> dict:
    """Fingerprint of the settings and inputs one country's SDA outputs were built from.

    Args:
        depth_threshold, all_touched, include_pasture, use_roads, with_pop, write_depthbin,
            depthbin_max: the settings that change what the outputs contain.
        lulc_path, mapping_path, roads_path, pop_path: inputs, fingerprinted by size and mtime.
        depth_dir, depth_json: how the depth rasters were selected.
        rp_map: return period to depth raster.

    Returns:
        dict: the signature, including a `signature_sha256` over every field but the timestamp.
    """
    sig = {
        "code_version": SDA_CODE_VERSION,
        "created_utc": datetime.utcnow().isoformat() + "Z",

        "depth_threshold": float(depth_threshold),
        "all_touched": bool(all_touched),
        "include_pasture": bool(include_pasture),
        "use_roads": bool(use_roads),
        "with_pop": bool(with_pop),
        "write_depthbin": bool(write_depthbin),
        "depthbin_max": float(depthbin_max),

        # depth input controls (so signature changes when you change rp selection or sources)
        "depth_dir": str(depth_dir) if depth_dir else "",
        "depth_json": str(depth_json) if depth_json else "",
        "rps": sorted([int(rp) for rp in rp_map.keys()]),

        "lulc": utilities.file_fingerprint(str(lulc_path)),
        "mapping_json": {
            **utilities.file_fingerprint(str(mapping_path)),
            "sha256": utilities.sha256_file(str(mapping_path)) if hb.path_exists(mapping_path) else None,
        },
        "roads": utilities.file_fingerprint(str(roads_path)) if use_roads else {"path": str(roads_path), "exists": False},
        "pop": utilities.file_fingerprint(str(pop_path)) if with_pop else {"path": str(pop_path), "exists": False},

        "depth_rasters": {int(rp): utilities.file_fingerprint(path) for rp, path in rp_map.items()},
    }

    tmp = dict(sig)
    tmp.pop("created_utc", None)
    sig["signature_sha256"] = hashlib.sha256(
        json.dumps(tmp, sort_keys=True).encode("utf-8")).hexdigest()
    return sig


def read_old_signature(out_dir: str, iso3: str) -> dict | None:
    path = signature_path(out_dir, iso3)
    if not hb.path_exists(path):
        return None
    try:
        return json.loads(open(path, encoding="utf-8").read())
    except Exception:
        return None


def outputs_complete_for_iso3(out_dir: str, iso3: str, rp_map: dict[int, str], write_depthbin: bool) -> bool:
    summary = os.path.join(out_dir, f"sda_summary_{iso3}.csv")
    if not hb.path_exists(summary):
        return False

    for rp in rp_map.keys():
        class_tif = os.path.join(out_dir, f"sda_class_{iso3}_rp{int(rp)}.tif")
        mask_tif  = os.path.join(out_dir, f"sda_mask_{iso3}_rp{int(rp)}.tif")
        if not hb.path_exists(class_tif) or not hb.path_exists(mask_tif):
            return False
        if not utilities.raster_ok(class_tif) or not raster_ok(mask_tif):
            return False

        if write_depthbin:
            db_tif = os.path.join(out_dir, f"sda_depthbin_idx_{iso3}_rp{int(rp)}.tif")
            if not hb.path_exists(db_tif) or (not utilities.raster_ok(db_tif)):
                return False

    return True


def should_skip_iso3(out_dir: str, iso3: str, new_sig: dict, rp_map: dict[int, str], write_depthbin: bool) -> bool:
    old = read_old_signature(out_dir, iso3)
    if old is None:
        return False
    if old.get("signature_sha256") != new_sig.get("signature_sha256"):
        return False
    return outputs_complete_for_iso3(out_dir, iso3, rp_map=rp_map, write_depthbin=write_depthbin)


def write_signature(out_dir: str, iso3: str, sig: dict):
    hb.write_to_file(json.dumps(sig, indent=2, sort_keys=True), signature_path(out_dir, iso3))


def load_mapping(mapping_path: str) -> dict:
    if not hb.path_exists(mapping_path):
        raise FileNotFoundError(f"mapping JSON not found:\n  {mapping_path}")

    mapping = json.loads(open(mapping_path, encoding="utf-8").read())

    if "artif" not in mapping and "built_up" in mapping:
        mapping["artif"] = mapping["built_up"]
    if "crop" not in mapping and "cropland" in mapping:
        mapping["crop"] = mapping["cropland"]

    for k in ["artif", "crop", "pasture", "ignore"]:
        if k not in mapping or mapping[k] is None:
            mapping[k] = []

    def _to_int_list(x):
        out = []
        for v in x:
            try:
                out.append(int(v))
            except Exception:
                pass
        return out

    for k in ["artif", "crop", "pasture", "ignore"]:
        mapping[k] = _to_int_list(mapping.get(k, []))

    return mapping

def publish_inputs(p):
    """Every task's first line: the flood es_config row, its es_parameters settings and data
    references, and the shared country references.

    Inputs are es_parameters `*_path` rows resolved by `get_path`; outputs are named by the task
    that writes them, under its own `cur_dir`. Cheap and idempotent, so a task stays a working
    piece on its own.
    """
    utilities.hydrate_es_config(p, 'flood', log=hb.log)
    utilities.hydrate_es_parameters(p, 'flood', log=hb.log)
    utilities.initialize_country_paths(p)

    # Optional companions: a blank es_parameters cell means it is not supplied, and the readers
    # branch on None rather than on a path that happens not to exist.
    for optional in ('flood_protection_path', 'flood_protection_evidence_path'):
        value = getattr(p, optional, None)
        setattr(p, optional, str(value) if value else None)
    # Blank means every scenario in SCENARIOS rather than a chosen subset.
    p.flood_gep_scenarios = getattr(p, 'flood_gep_scenarios', None) or None

    if not hasattr(p, 'results'):
        p.results = {}
    return p




MAP_MONEY_UNIT_LABEL = "2019 USD million"



def _load_country_ead(p) -> pd.DataFrame:
    utilities.assert_exists(p.flood_country_ead_path, "Run Section D (step 4D) before Section E.")
    df = pd.read_csv(p.flood_country_ead_path)
    df["iso3"] = df["iso3"].astype(str).str.upper().str.strip()
    ead_col = utilities.find_col(df, ("ead_usd2019", "ead usd2019", "ead"))
    if ead_col is None:
        raise ValueError(f"No EAD column found in {p.flood_country_ead_path}: {list(df.columns)}")
    df = df.rename(columns={ead_col: "ead_usd2019"})
    df["ead_usd2019"] = pd.to_numeric(df["ead_usd2019"], errors="coerce")
    return df


def generate_all_maps_and_figures(p) -> dict:
    """
    Section E driver: publication figures from Section D's country table --
    a global EAD choropleth (Fisher-Jenks), a top-N country bar chart, a
    regional breakdown, and the mean SPA->SDA service-flow map if Section C ran.
    """
    hb.create_directories(p.flood_figures_dir)
    outputs = {}

    country = _load_country_ead(p)
    admin0 = load_admin0(p.flood_country_vector_path)

    joined = admin0.merge(country, on="iso3", how="left")

    # 1) Global choropleth of Expected Annual Damage
    png = os.path.join(p.flood_figures_dir, "map_country_ead_USD2019.png")
    utilities.plot_publication_choropleth_categorical(
        joined, value_col="ead_usd2019",
        title="Expected Annual Flood Damage to Service Demanding Areas",
        out_png=png, legend_title=MAP_MONEY_UNIT_LABEL,
        scheme="fisher_jenks", k=p.flood_map_k_classes,
        value_unit="usd_millions", label_format="usd_millions",
    )
    outputs["map_country_ead"] = png
    print(f"[OK] Wrote {png}")

    # 2) Top-N countries by EAD
    top = utilities.top_n(country, "ead_usd2019", p.flood_top_n).copy()
    if not top.empty:
        top["_m"] = top["ead_usd2019"] / utilities.USD_TO_MILLIONS
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(top["iso3"][::-1], top["_m"][::-1])
        ax.set_xlabel(MAP_MONEY_UNIT_LABEL)
        ax.set_title(f"Top {p.flood_top_n} countries by Expected Annual Flood Damage")
        png = os.path.join(p.flood_figures_dir, "bar_top_countries_ead.png")
        utilities.savefig(png)
        outputs["bar_top_countries"] = png
        print(f"[OK] Wrote {png}")

    # 3) Regional breakdown, if Step 4D enriched with a region column
    if "region_wb" in country.columns:
        reg = (country.groupby("region_wb", dropna=True)["ead_usd2019"]
               .sum().sort_values(ascending=False) / utilities.USD_TO_MILLIONS)
        if not reg.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(reg.index[::-1], reg.values[::-1])
            ax.set_xlabel(MAP_MONEY_UNIT_LABEL)
            ax.set_title("Expected Annual Flood Damage by World Bank region")
            png = os.path.join(p.flood_figures_dir, "bar_region_ead.png")
            utilities.savefig(png)
            outputs["bar_region"] = png
            print(f"[OK] Wrote {png}")

    # 4) Mean SPA -> SDA service flow by country (Section C output)
    if hb.path_exists(p.flood_service_flow_path):
        flow = pd.read_csv(p.flood_service_flow_path)
        if {"iso3", "mean_spa_ratio_on_sda"}.issubset(flow.columns):
            agg = (flow.groupby("iso3", as_index=False)["mean_spa_ratio_on_sda"].mean())
            agg["iso3"] = agg["iso3"].astype(str).str.upper()
            j2 = admin0.merge(agg, on="iso3", how="left")
            png = os.path.join(p.flood_figures_dir, "map_mean_service_flow_frac.png")
            utilities.plot_publication_choropleth_categorical(
                j2, value_col="mean_spa_ratio_on_sda",
                title="Mean upstream SPA share serving flood-exposed SDA",
                out_png=png, legend_title="Service flow fraction (0-1)",
                scheme="quantiles", k=p.flood_map_k_classes,
                value_unit="raw", label_format="percent",
            )
            outputs["map_service_flow"] = png
            print(f"[OK] Wrote {png}")
    else:
        warnings.warn(f"[WARN] No service-flow summary at {p.flood_service_flow_path}; "
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
    p.flood_qa_dir = os.path.join(p.cur_dir, 'qa_maps')
    p.flood_spa_country_summary_path = os.path.join(p.flood_qa_dir, 'global_spa_country_summary.csv')
    if not p.run_this:
        return True

    service_results = p.results.setdefault('flood', {})
    service_results['global_sda_raster'] = p.flood_sda_raster_path
    service_results['sda_mapping_json'] = p.flood_sda_mapping_path

    prepare_all_inputs(p, skip_download=_required(p, 'flood_skip_depth_download'))
    return True


def build_sda_global(p):
    """Section B: delineate the Service Demanding Areas for every country and return period.

    An SDA is the exposed asset in the floodplain -- built surface, cropland, pasture -- so it is
    where flood damage lands and therefore where the regulation service has value. One raster and
    one summary row per country per return period.

    A country whose signature matches and whose outputs are complete is skipped, which is what makes
    a rerun cheap. A country that raises stops the run: it would otherwise leave the global summary
    while the summary still looked complete.
    """
    admin0 = load_admin0(p.flood_country_vector_path)
    iso_all = sorted(set(admin0["iso3"].astype(str).str.upper().tolist()))
    mapping = load_mapping(p.flood_sda_mapping_path)
    rp_map = {rp: os.path.join(p.flood_depth_aligned_path, DEPTH_RASTER_PATTERN.format(rp=rp))
              for rp in p.flood_return_periods}

    if str(p.flood_iso3_list).strip():
        run_list = [x.strip().upper() for x in str(p.flood_iso3_list).split(",") if x.strip()]
    else:
        start = max(int(p.flood_iso3_start), 0)
        run_list = iso_all[start:(start + int(p.flood_iso3_n)) if int(p.flood_iso3_n) > 0 else None]

    signature = build_run_signature(
        depth_threshold=p.flood_depth_threshold_m, all_touched=p.flood_all_touched,
        include_pasture=p.flood_include_pasture, use_roads=p.flood_use_roads,
        with_pop=p.flood_with_pop, write_depthbin=p.flood_write_depthbin,
        depthbin_max=p.flood_depthbin_max, lulc_path=p.flood_lulc_path,
        mapping_path=p.flood_sda_mapping_path, roads_path=p.flood_roads_path,
        pop_path=p.flood_pop_path, depth_dir=p.flood_depth_aligned_path, depth_json="",
        rp_map=rp_map)
    rows = []
    for iso3 in run_list:
        out_dir = os.path.join(p.flood_sda_country_dir, iso3)
        hb.create_directories(str(out_dir))
        if p.flood_skip_done and should_skip_iso3(out_dir, iso3, signature, rp_map=rp_map,
                                              write_depthbin=p.flood_write_depthbin):
            summary = os.path.join(str(out_dir), f"sda_summary_{iso3}.csv")
            if hb.path_exists(summary):
                rows.append(hb.df_read(str(summary)))
            continue
        df = process_country(
            iso3=iso3, admin0=admin0, out_root=p.flood_sda_country_dir, rp_map=rp_map,
            lulc_path=p.flood_lulc_path, mapping=mapping,
            depth_threshold_m=p.flood_depth_threshold_m, all_touched=p.flood_all_touched,
            include_pasture=p.flood_include_pasture, use_roads=p.flood_use_roads,
            roads_path=p.flood_roads_path, with_pop=p.flood_with_pop, pop_path=p.flood_pop_path,
            write_depthbin=p.flood_write_depthbin, depthbin_max=p.flood_depthbin_max)
        write_signature(out_dir, iso3, signature)
        if not df.empty:
            rows.append(df)

    if not rows:
        raise ValueError('Section B produced no SDA summaries for any of %d countries.' % len(run_list))
    combined = pd.concat(rows, ignore_index=True)
    utilities.write_csv(combined, p.flood_sda_summary_path)
    hb.log('Section B: %d countries, %d SDA rows' % (len(run_list), len(combined)))
    return p.flood_sda_summary_path


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
    # Section C reads the per-country SDA rasters written here, so the directory is published
    # before the run_this guard: a skipped task still tells the later ones where its work is.
    p.flood_sda_country_dir = p.cur_dir
    p.flood_sda_summary_path = os.path.join(p.cur_dir, 'global_sda_summary_countries.csv')
    if not p.run_this:
        return True

    build_sda_global(p)
    return True


def task_compute_service_flow(p):
    """
    Task wrapper for Section C: reproject the upstream SPA ratio onto the SDA
    grid and record the flood-regulation service flow fraction for each
    ISO3 x RP. Originally serviceflow_step3_spa_to_sda_ratio_global.py.
    Writes global_service_flow_spa_to_sda.csv, which Section E maps.
    """
    publish_inputs(p)
    p.flood_service_flow_path = os.path.join(p.cur_dir, 'global_service_flow_spa_to_sda.csv')
    if not p.run_this:
        return True

    service_results = p.results.setdefault('flood', {})
    service_results['service_flow_summary'] = p.flood_service_flow_path

    compute_service_flow_global(p)
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

    # The per-country valuation directory and the global tables this section writes. Published
    # before the run_this guard so the GEP task, which writes its degraded scenarios alongside
    # them, finds the same place whether or not this task ran.
    p.flood_valuation_country_dir = p.cur_dir
    p.flood_global_export_dir = os.path.join(p.cur_dir, '_global')
    p.flood_currency_audit_dir = os.path.join(p.flood_global_export_dir, '_currency_audit')
    p.flood_country_ead_path = os.path.join(p.flood_global_export_dir, 'step4d_country_ead_USD2019.csv')
    p.flood_global_totals_path = os.path.join(p.flood_global_export_dir, 'step4d_global_totals_USD2019.csv')
    p.flood_country_ead_with_service_flow_path = os.path.join(
        p.flood_global_export_dir, 'step4d_country_ead_with_service_flow_USD2019.csv')
    p.flood_gep_path = os.path.join(p.flood_global_export_dir, 'step4e_flood_gep_USD2019.csv')
    if not p.run_this:
        return True
    hb.create_directories([p.flood_global_export_dir, p.flood_currency_audit_dir])

    service_results = p.results.setdefault('flood', {})
    service_results['country_ead_csv'] = p.flood_country_ead_path
    service_results['global_totals_csv'] = p.flood_global_totals_path

    if hb.path_all_exist([service_results['country_ead_csv']]):
        hb.log("%s already exists. Skipping the flood valuation chain."
               % os.path.basename(service_results['country_ead_csv']))
    else:
        skip_tables = _required(p, 'flood_skip_damage_tables')
        run_valuation_chain(p, p.df_countries, skip_damage_tables=skip_tables)

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
    if not p.run_this:
        return True

    service_results = p.results.setdefault('flood', {})
    service_results['flood_gep_csv'] = p.flood_gep_path

    if hb.path_all_exist([service_results['flood_gep_csv']]):
        hb.log("step4e_flood_gep_USD2019.csv already exists. Skipping GEP chain.")
    else:
        run_gep_chain(
            p, skip_damage_tables=_required(p, 'flood_skip_damage_tables'),
            scenarios=p.flood_gep_scenarios)

    return True


def task_generate_maps_and_figures(p):
    """
    Task wrapper for Section E: publication-ready choropleths and charts built
    from task_compute_flood_damages()'s and task_compute_service_flow()'s
    outputs. Originally analyze_step4d_global_results.py.
    """
    publish_inputs(p)
    p.flood_figures_dir = p.cur_dir
    if not p.run_this:
        return True

    generate_all_maps_and_figures(p)
    return True


def gep_calculation(p):
    """GEP valuation for flood: the per-country avoided damage the pipeline produces.

    The quantity is the flood damage ecosystems prevent: `ead_bare - ead_current`, the difference
    between expected annual damage in a degraded world and in today's. `compute_flood_gep` computes
    it from the three scenario runs and writes `step4e_flood_gep_USD2019.csv`.

    The author's export is the fallback when that chain has not run. Which was used is recorded in
    the log and in service_results.

    ⚠ This is the GEP, not the expected annual damage the same pipeline reports.
    """
    publish_inputs(p)
    service_results, already_done = utilities.begin_gep_calculation(p, 'flood')
    if already_done:
        return

    computed = p.flood_gep_path if p.flood_gep_path else None
    if computed is not None and hb.path_exists(computed):
        df_gep = hb.df_read(str(computed))
        column = next((c for c in ('gep_flood_bare_usd2019', 'gep_flood_insitu_usd2019')
                       if c in df_gep.columns), None)
        if column is None:
            raise NameError(
                'No gep_flood column in %s; found %s. The counterfactual chain writes '
                'gep_flood_bare_usd2019 and gep_flood_insitu_usd2019, and bare is the one the '
                'combined table with erosion uses, so a file without either has not run it.'
                % (computed, list(df_gep.columns)))
        df_gep = df_gep.rename(columns={column: 'flood_gep'})
        source = 'computed here (%s, %s)' % (os.path.basename(str(computed)), column)
    else:
        df_gep = hb.df_read(p.flood_gep_for_merge_path)
        df_gep = df_gep.rename(columns={'gep_const2019_usd': 'flood_gep'})
        source = "the author's export (%s)" % os.path.basename(str(p.flood_gep_for_merge_path))
    service_results['flood_gep_source'] = source
    df_gep['year'] = int(p.gep_base_year)
    # The country attributes every other service's table carries, so this output can be read,
    # grouped and reported the same way.
    attribute_columns = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name', 'continent',
                         'region_un', 'region_wb', 'income_grp', 'subregion']
    countries = utilities.collapse_countries_to_r250(p.df_countries)
    countries = countries[[c for c in attribute_columns if c in countries.columns]]
    df_gep = countries.merge(df_gep, on='iso3_r250_label', how='left')
    hb.df_write(df_gep, service_results['gep_by_country_base_year'])
    hb.log('Flood GEP: %.4g USD over %d countries with a positive value, from %s.'
           % (df_gep['flood_gep'].sum(), (df_gep['flood_gep'] > 0).sum(), source))
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JRC-style Flood Control — Step 2: Build SDA (Service Demanding Areas)
GLOBAL by Country × Return Period (RP), using flood depth rasters + ESA CCI 300m LULC.

Dated: December 15, 2025 | NatCap TEEMs / GEP Flood | Nfamara K Dampha | FINAL
UPDATED: Dec 2025 — SMART SKIP + AUTO-UPDATE-IF-DIFFERENT
UPDATED: Dec 2025 — DEPTH INPUTS FIXED (auto-detect RP rasters + CLI depth-dir/depth-json + RP selection)

=====================================================================================
CONCEPT (JRC/INCA / SEEA-EA aligned)
=====================================================================================
Build SDA maps:
  SDA = (economic assets from LULC) ∩ (floodplain from depth rasters)

Outputs consumed by:
  - Step 3 (service flow SPA → SDA)
  - Step 4 (monetary valuation)

=====================================================================================
IMPORTANT ALIGNMENT PRINCIPLE
=====================================================================================
This SDA builder assumes:
  - Depth rasters are already aligned to the LULC grid (CRS + transform + pixel size)
  - We hard-lock that condition using assert_same_grid()

If you ever change depth alignment, re-align depth to LULC FIRST (do not warp SDA silently).

=====================================================================================
DEPTH INPUTS (robust handling)
=====================================================================================
This script supports three depth input modes:

A) --depth-json: explicit mapping {rp: "/path/to/depth.tif"} (most explicit)
B) --depth-dir: auto-detect depth rasters in a folder (default is your aligned_to_lulc folder)
C) fallback RP_MAP_DEFAULT (still provided, but auto-detect is preferred)

Auto-detect recognizes common filename patterns:
  - rp10y, rp20y, rp50y, rp100y, rp200y, rp500y
  - rp10, rp100, etc. (less strict fallback)

You can select which RPs to process with:
  --rps "10,20,50,100,200,500"
If not provided, script processes all detected RPs.

=====================================================================================
EXAMPLE RUN (Jupyter)
=====================================================================================
%run "$GEP_FLOOD_ROOT/code_final/global_invest/flood/scripts/sda_step2_build_sda_global.py" \
  --start 0 --n 10 --skip-done \
  --mapping-json str(FP.ROOT) + "/inputs/lulc_to_sda_mapping/lulc_to_sda_mapping.json" \
  --lulc-path str(FP.ROOT) + "/inputs/lulc/lulc_esa_2019_int_reproj.tif" \
  --depth-dir str(FP.ROOT) + "/inputs/floodplain_depth/aligned_to_lulc" \
  --rps "10,20,50,100,200,500" \
  --depth-threshold 0.1 --with-pop
"""

from __future__ import annotations
import flood_paths as FP

import argparse
import json
import os
import re
import hashlib
from datetime import datetime
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import reproject, Resampling


# =============================================================================
# 0) VERSIONING (BUMP WHEN YOU CHANGE CORE LOGIC)
# =============================================================================
CODE_VERSION = "2025-12-15_sda_step2_smartskip_v2_depth_inputs"


# -----------------------------------------------------------------------------#
# 0) Paths (edit only if directory structure changes)
# -----------------------------------------------------------------------------#
ROOT   = Path(os.environ.get("GEP_FLOOD_ROOT", str(FP.ROOT) + ""))
INPUTS = ROOT / "inputs"
OUT    = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

ADMIN0_PATH = INPUTS / "country_vector" / "country_boundary_r250_with_iso3.gpkg"

# Default depth folder (your confirmed location)
DEPTH_DIR_DEFAULT = INPUTS / "floodplain_depth" / "aligned_to_lulc"

# Default *example* file names (kept for documentation / fallback only)
RP_MAP_DEFAULT = {
    10:  DEPTH_DIR_DEFAULT / "JRC_flood_depth_rp10y__matchLULC.tif",
    20:  DEPTH_DIR_DEFAULT / "JRC_flood_depth_rp20y__matchLULC.tif",
    50:  DEPTH_DIR_DEFAULT / "JRC_flood_depth_rp50y__matchLULC.tif",
    100: DEPTH_DIR_DEFAULT / "JRC_flood_depth_rp100y__matchLULC.tif",
    200: DEPTH_DIR_DEFAULT / "JRC_flood_depth_rp200y__matchLULC.tif",
    500: DEPTH_DIR_DEFAULT / "JRC_flood_depth_rp500y__matchLULC.tif",
}

LANDCOVER_DEFAULT = INPUTS / "lulc" / "lulc_esa_2019_int_reproj.tif"
ROADS_DEFAULT     = INPUTS / "roads" / "roads_mask_match_depth.tif"
POP_DEFAULT       = INPUTS / "pop" / "GlobPOP_Count_30arc_2020_I32.tif"


# =============================================================================
# SMART-SKIP HELPERS
# =============================================================================
def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def file_fingerprint(path: Path) -> dict:
    p = Path(path) if path is not None else Path("")
    if (not p) or (not p.exists()):
        return {"path": str(path), "exists": False}
    st = p.stat()
    return {"path": str(p), "exists": True, "bytes": int(st.st_size), "mtime": float(st.st_mtime)}


def signature_path(out_dir: Path, iso3: str) -> Path:
    return out_dir / f"sda_run_signature_{iso3}.json"


def build_run_signature(args, rp_map: dict[int, Path]) -> dict:
    sig = {
        "code_version": CODE_VERSION,
        "created_utc": datetime.utcnow().isoformat() + "Z",

        "depth_threshold": float(args.depth_threshold),
        "all_touched": bool(args.all_touched),
        "include_pasture": bool(args.include_pasture),
        "use_roads": bool(args.use_roads),
        "with_pop": bool(args.with_pop),
        "write_depthbin": bool(args.write_depthbin),
        "depthbin_max": float(args.depthbin_max),

        # depth input controls (so signature changes when you change rp selection or sources)
        "depth_dir": str(args.depth_dir) if args.depth_dir else "",
        "depth_json": str(args.depth_json) if args.depth_json else "",
        "rps": sorted([int(rp) for rp in rp_map.keys()]),

        "lulc": file_fingerprint(Path(args.lulc_path)),
        "mapping_json": {
            **file_fingerprint(Path(args.mapping_json)),
            "sha256": sha256_file(Path(args.mapping_json)) if Path(args.mapping_json).exists() else None,
        },
        "roads": file_fingerprint(Path(args.roads_path)) if args.use_roads else {"path": str(args.roads_path), "exists": False},
        "pop": file_fingerprint(Path(args.pop_path)) if args.with_pop else {"path": str(args.pop_path), "exists": False},

        "depth_rasters": {int(rp): file_fingerprint(p) for rp, p in rp_map.items()},
    }

    tmp = dict(sig)
    tmp.pop("created_utc", None)
    sig["signature_sha256"] = _sha256_bytes(json.dumps(tmp, sort_keys=True).encode("utf-8"))
    return sig


def read_old_signature(out_dir: Path, iso3: str) -> dict | None:
    p = signature_path(out_dir, iso3)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def atomic_write_raster(final_path: Path, profile: dict, array: np.ndarray, band: int = 1):
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except Exception:
        pass

    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(array, band)

    os.replace(tmp_path, final_path)


def raster_ok(path: Path) -> bool:
    try:
        with rasterio.open(path) as src:
            _ = src.read(1, window=((0, 1), (0, 1)))
        return True
    except Exception:
        return False


def outputs_complete_for_iso3(out_dir: Path, iso3: str, rp_map: dict[int, Path], write_depthbin: bool) -> bool:
    summary = out_dir / f"sda_summary_{iso3}.csv"
    if not summary.exists():
        return False

    for rp in rp_map.keys():
        class_tif = out_dir / f"sda_class_{iso3}_rp{int(rp)}.tif"
        mask_tif  = out_dir / f"sda_mask_{iso3}_rp{int(rp)}.tif"
        if not class_tif.exists() or not mask_tif.exists():
            return False
        if not raster_ok(class_tif) or not raster_ok(mask_tif):
            return False

        if write_depthbin:
            db_tif = out_dir / f"sda_depthbin_idx_{iso3}_rp{int(rp)}.tif"
            if not db_tif.exists() or (not raster_ok(db_tif)):
                return False

    return True


def should_skip_iso3(out_dir: Path, iso3: str, new_sig: dict, rp_map: dict[int, Path], write_depthbin: bool) -> bool:
    old = read_old_signature(out_dir, iso3)
    if old is None:
        return False
    if old.get("signature_sha256") != new_sig.get("signature_sha256"):
        return False
    return outputs_complete_for_iso3(out_dir, iso3, rp_map=rp_map, write_depthbin=write_depthbin)


def write_signature(out_dir: Path, iso3: str, sig: dict):
    signature_path(out_dir, iso3).write_text(json.dumps(sig, indent=2, sort_keys=True))


# -----------------------------------------------------------------------------#
# CLI
# -----------------------------------------------------------------------------#
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--iso3", default="", help="Comma-separated ISO3 list (e.g., KHM,BOL). If blank: run all.")
    p.add_argument("--start", type=int, default=0, help="Start index for batch runs (when --iso3 blank).")
    p.add_argument("--n", type=int, default=0, help="Number of countries to run (0 = all remaining).")
    p.add_argument("--skip-done", action="store_true",
                   help=("SMART SKIP: skip ISO3 ONLY IF stored signature matches current signature AND outputs complete.\n"
                         "If signature differs, ISO3 is re-run and outputs overwritten for that ISO3."))
    p.add_argument("--all-touched", action="store_true", help="rio_mask(all_touched=True) for more inclusive clipping.")

    p.add_argument("--depth-threshold", type=float, default=0.0,
                   help="Floodplain threshold (m). 0.0=any inundation; 0.1 reduces speckle.")

    p.add_argument("--lulc-path", default=str(LANDCOVER_DEFAULT),
                   help="Landcover raster aligned to depth rasters.")
    p.add_argument("--mapping-json", required=True,
                   help=("JSON mapping LULC IDs to SDA classes. Preferred keys: artif/crop/pasture/ignore.\n"
                         "Supports aliases: built_up->artif, cropland->crop"))

    p.add_argument("--include-pasture", action="store_true",
                   help="Include mapping['pasture'] IDs as SDA class=3.")
    p.add_argument("--use-roads", action="store_true",
                   help="Include roads raster (binary) as SDA class=4 (overrides other classes where roads=1).")
    p.add_argument("--roads-path", default=str(ROADS_DEFAULT),
                   help="Roads mask raster aligned to depth grid (binary 1=road).")

    p.add_argument("--with-pop", action="store_true",
                   help="Compute population in SDA (requires --pop-path).")
    p.add_argument("--pop-path", default=str(POP_DEFAULT),
                   help="Population counts raster.")

    p.add_argument("--write-depthbin", action="store_true",
                   help="Write depth-bin index raster for later monetary joins.")
    p.add_argument("--depthbin-max", type=float, default=6.0,
                   help="Max depth for binning (m). Values above clip to last bin.")

    # NEW: depth inputs
    p.add_argument("--depth-dir", default=str(DEPTH_DIR_DEFAULT),
                   help="Directory containing RP depth rasters (auto-detected).")
    p.add_argument("--depth-json", default="",
                   help="Optional JSON file mapping rp->path. If provided, overrides --depth-dir auto-detect.")
    p.add_argument("--rps", default="",
                   help='Optional RP list to run, e.g. "10,20,50,100,200,500". If blank: run all detected RPs.')

    return p.parse_args()


# -----------------------------------------------------------------------------#
# Admin0 loader
# -----------------------------------------------------------------------------#
def load_admin0() -> gpd.GeoDataFrame:
    if not ADMIN0_PATH.exists():
        raise FileNotFoundError(f"Admin0 file not found:\n  {ADMIN0_PATH}")

    admin0 = gpd.read_file(ADMIN0_PATH)
    if admin0.crs is None:
        raise ValueError(f"Admin0 has no CRS: {ADMIN0_PATH}")

    iso_candidates = ["iso3", "ISO3", "ADM0_A3", "ISO_A3", "adm0_a3"]
    iso_col = next((c for c in iso_candidates if c in admin0.columns), None)
    if iso_col is None:
        raise ValueError(f"No ISO3-like column found. Columns: {list(admin0.columns)}")

    admin0["iso3"] = admin0[iso_col].astype(str).str.upper().str.strip()
    admin0["geometry"] = admin0["geometry"].buffer(0)
    admin0 = admin0[admin0.geometry.notna() & ~admin0.geometry.is_empty].copy()
    return admin0


# -----------------------------------------------------------------------------#
# Depth RP map builders
# -----------------------------------------------------------------------------#
def parse_rps_arg(rps: str) -> list[int]:
    if not rps.strip():
        return []
    out = []
    for tok in rps.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return sorted(set(out))


def load_depth_json(depth_json: str) -> dict[int, Path]:
    p = Path(depth_json)
    if not p.exists():
        raise FileNotFoundError(f"--depth-json not found:\n  {p}")
    d = json.loads(p.read_text())
    out = {}
    for k, v in d.items():
        out[int(k)] = Path(v)
    return out


def autodetect_depth_rasters(depth_dir: Path) -> dict[int, Path]:
    """
    Auto-detect RP depth rasters in a directory using filename patterns.

    Recognized patterns (case-insensitive):
      - rp10y, rp20y, rp100y, rp200y, rp500y
      - rp10, rp100 (fallback)

    If multiple rasters match an RP, we pick the shortest filename (usually the canonical one).
    """
    if not depth_dir.exists():
        raise FileNotFoundError(f"--depth-dir not found:\n  {depth_dir}")

    tifs = sorted(depth_dir.glob("*.tif"))
    if not tifs:
        raise FileNotFoundError(f"No .tif files found in:\n  {depth_dir}")

    rp_hits: dict[int, list[Path]] = {}

    # Prefer rp###y (strong)
    pat_strong = re.compile(r"rp[_-]?(?P<rp>\d{1,4})y", re.IGNORECASE)
    # Fallback rp### (weak)
    pat_weak = re.compile(r"rp[_-]?(?P<rp>\d{1,4})", re.IGNORECASE)

    for f in tifs:
        name = f.name
        m = pat_strong.search(name)
        if m:
            rp = int(m.group("rp"))
            rp_hits.setdefault(rp, []).append(f)
            continue
        m = pat_weak.search(name)
        if m:
            rp = int(m.group("rp"))
            rp_hits.setdefault(rp, []).append(f)

    # Choose best candidate per RP
    rp_map = {}
    for rp, files in rp_hits.items():
        files_sorted = sorted(files, key=lambda x: (len(x.name), x.name))
        rp_map[int(rp)] = files_sorted[0]

    return rp_map


def finalize_rp_map(args) -> dict[int, Path]:
    """
    Priority:
      1) --depth-json (explicit)
      2) --depth-dir auto-detect
      3) RP_MAP_DEFAULT (fallback if auto-detect finds nothing)
    Then apply --rps filtering (if provided).
    """
    if args.depth_json and args.depth_json.strip():
        rp_map = load_depth_json(args.depth_json.strip())
        source = f"depth-json: {args.depth_json}"
    else:
        rp_map = autodetect_depth_rasters(Path(args.depth_dir))
        source = f"depth-dir: {args.depth_dir}"

        # If auto-detect returns empty (rare), fallback
        if not rp_map:
            rp_map = {int(k): Path(v) for k, v in RP_MAP_DEFAULT.items()}
            source = "RP_MAP_DEFAULT fallback"

    # Filter to requested RPs
    req = parse_rps_arg(args.rps)
    if req:
        rp_map = {rp: p for rp, p in rp_map.items() if int(rp) in set(req)}

    # Validate existence
    missing = [rp for rp, p in rp_map.items() if not Path(p).exists()]
    if missing:
        msg = "\n".join([f"  rp{rp}: {rp_map[rp]}" for rp in sorted(missing)])
        raise FileNotFoundError(
            "[ERROR] Some depth rasters are missing after RP map construction.\n"
            f"Depth source: {source}\n"
            "Missing:\n" + msg
        )

    if not rp_map:
        raise FileNotFoundError(
            "[ERROR] No depth rasters selected.\n"
            "Check --depth-dir/--depth-json and/or your --rps filter."
        )

    print(f"[INFO] Depth source: {source}")
    print(f"[INFO] RPs selected: {sorted(rp_map.keys())}")
    for rp in sorted(rp_map.keys()):
        print(f"       rp{rp}: {rp_map[rp]}")
    return dict(sorted(rp_map.items(), key=lambda kv: int(kv[0])))


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#
def pixel_area_km2(transform) -> float:
    return abs(float(transform.a) * float(transform.e)) / 1e6


def load_mapping(mapping_path: Path) -> dict:
    if not mapping_path.exists():
        raise FileNotFoundError(f"mapping JSON not found:\n  {mapping_path}")

    mapping = json.loads(mapping_path.read_text())

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


def reproject_pop_to_target(pop_src: rasterio.io.DatasetReader, target_profile: dict) -> np.ndarray:
    dst = np.zeros((target_profile["height"], target_profile["width"]), dtype=np.float32)

    try:
        resamp = Resampling.sum
    except Exception:
        resamp = Resampling.nearest
        print("[WARN] Resampling.sum not available; using nearest. Totals may drift if grids differ substantially.")

    reproject(
        source=rasterio.band(pop_src, 1),
        destination=dst,
        src_transform=pop_src.transform,
        src_crs=pop_src.crs,
        dst_transform=target_profile["transform"],
        dst_crs=target_profile["crs"],
        src_nodata=pop_src.nodata,
        dst_nodata=0.0,
        resampling=resamp,
    )
    dst[dst < 0] = 0.0
    return dst


def build_depthbin_index(depth_m: np.ndarray, nodata_mask: np.ndarray, max_depth: float = 6.0) -> np.ndarray:
    edges = np.arange(0, max_depth + 0.5, 0.5)
    d = depth_m.copy().astype("float32")

    d[~np.isfinite(d)] = np.nan
    d[(d < 0) & np.isfinite(d)] = 0.0
    d[(d > max_depth) & np.isfinite(d)] = max_depth

    idx = np.searchsorted(edges, d, side="left").astype(np.int16)
    idx[nodata_mask] = -1
    return idx


def assert_same_grid(src_a: rasterio.io.DatasetReader, src_b: rasterio.io.DatasetReader, label_a="A", label_b="B"):
    if src_a.crs != src_b.crs:
        raise ValueError(f"CRS mismatch: {label_a}={src_a.crs} vs {label_b}={src_b.crs}")
    if src_a.transform != src_b.transform:
        raise ValueError(
            f"Transform mismatch between {label_a} and {label_b}. "
            "Pre-align rasters before running (do not silently warp SDA)."
        )
    if src_a.width != src_b.width or src_a.height != src_b.height:
        raise ValueError(
            f"Shape mismatch: {label_a}={src_a.height}x{src_a.width} vs {label_b}={src_b.height}x{src_b.width}"
        )


# -----------------------------------------------------------------------------#
# Process one ISO3
# -----------------------------------------------------------------------------#
def process_country(
    iso3: str,
    admin0: gpd.GeoDataFrame,
    rp_map: dict[int, Path],
    lulc_path: Path,
    mapping: dict,
    depth_threshold_m: float,
    all_touched: bool,
    include_pasture: bool,
    use_roads: bool,
    roads_path: Path,
    with_pop: bool,
    pop_path: Path,
    write_depthbin: bool,
    depthbin_max: float,
) -> pd.DataFrame:

    iso3 = iso3.upper().strip()
    aoi = admin0[admin0["iso3"] == iso3]
    if aoi.empty:
        raise ValueError(f"ISO3 {iso3} not found in Admin0.")

    out_dir = OUT / iso3
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"sda_summary_{iso3}.csv"

    artif_ids   = set(mapping.get("artif", []))
    crop_ids    = set(mapping.get("crop", []))
    pasture_ids = set(mapping.get("pasture", [])) if include_pasture else set()
    ignore_ids  = set(mapping.get("ignore", []))

    if (len(artif_ids) + len(crop_ids) + len(pasture_ids) == 0) and (not use_roads):
        raise ValueError(
            "Your mapping JSON provides no SDA codes (artif/crop/pasture all empty) and roads are disabled.\n"
            "Fix mapping-json or run with --use-roads."
        )

    if not lulc_path.exists():
        raise FileNotFoundError(f"LULC raster not found:\n  {lulc_path}")
    if use_roads and not roads_path.exists():
        raise FileNotFoundError(f"Roads raster not found:\n  {roads_path}")
    if with_pop and not pop_path.exists():
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
                if not depth_path.exists():
                    print(f"[WARN] Missing depth raster for RP{rp}; skipping:\n  {depth_path}")
                    continue

                with rasterio.open(depth_path) as depth_src:
                    # Reproject AOI geometry to raster CRS (depth is the driver)
                    if admin0.crs != depth_src.crs:
                        geom_r = gpd.GeoDataFrame(geometry=[country_geom], crs=admin0.crs).to_crs(depth_src.crs).geometry.values[0]
                    else:
                        geom_r = country_geom

                    # HARD LOCK: LULC and depth must be aligned globally
                    assert_same_grid(depth_src, lulc_src, label_a="DEPTH", label_b="LULC")

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

                    pix_km2 = pixel_area_km2(depth_tr)
                    area_total_km2 = float(sda_mask.sum() * pix_km2)
                    area_artif_km2 = float((sda_class == 1).sum() * pix_km2)
                    area_crop_km2  = float((sda_class == 2).sum() * pix_km2)
                    area_past_km2  = float((sda_class == 3).sum() * pix_km2)
                    area_roads_km2 = float((sda_class == 4).sum() * pix_km2)

                    pop_in_sda = np.nan
                    if with_pop:
                        target_profile = depth_src.profile.copy()
                        target_profile.update(height=depth.shape[0], width=depth.shape[1], transform=depth_tr, crs=depth_src.crs)
                        pop_reproj = reproject_pop_to_target(pop_src, target_profile)
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

                    class_tif = out_dir / f"sda_class_{iso3}_rp{rp}.tif"
                    prof_class = prof_base.copy()
                    prof_class.update(dtype="uint8", count=1, nodata=0)
                    atomic_write_raster(class_tif, prof_class, sda_class.astype("uint8"), band=1)

                    mask_tif = out_dir / f"sda_mask_{iso3}_rp{rp}.tif"
                    atomic_write_raster(mask_tif, prof_class, sda_mask.astype("uint8"), band=1)

                    if write_depthbin:
                        nodata_mask = ~valid_depth
                        depthbin = build_depthbin_index(depth_q, nodata_mask=nodata_mask, max_depth=depthbin_max)
                        db_tif = out_dir / f"sda_depthbin_idx_{iso3}_rp{rp}.tif"
                        prof_db = prof_base.copy()
                        prof_db.update(dtype="int16", count=1, nodata=-1)
                        atomic_write_raster(db_tif, prof_db, depthbin.astype("int16"), band=1)

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


# -----------------------------------------------------------------------------#
# MAIN
# -----------------------------------------------------------------------------#
def main():
    args = parse_args()

    print(f"[INFO] ROOT:   {ROOT}")
    print(f"[INFO] INPUTS: {INPUTS}")
    print(f"[INFO] OUT:    {OUT}")
    print(f"[INFO] CODE_VERSION:      {CODE_VERSION}")
    print(f"[INFO] depth_threshold_m: {args.depth_threshold:.3f}")
    print(f"[INFO] all_touched:       {args.all_touched}")
    print(f"[INFO] include_pasture:   {args.include_pasture}")
    print(f"[INFO] use_roads:         {args.use_roads}")
    print(f"[INFO] with_pop:          {args.with_pop}")
    print(f"[INFO] write_depthbin:    {args.write_depthbin} (max={args.depthbin_max})")
    print(f"[INFO] skip_done (smart): {args.skip_done}")

    admin0 = load_admin0()
    iso_all = sorted(set(admin0["iso3"].astype(str).str.upper().tolist()))
    print(f"[INFO] Admin0 ISO3 count: {len(iso_all):,}")

    lulc_path = Path(args.lulc_path)
    mapping = load_mapping(Path(args.mapping_json))

    # NEW: build RP map from --depth-json/--depth-dir and optional --rps filter
    rp_map = finalize_rp_map(args)

    # Build run list
    if args.iso3.strip():
        run_list = [x.strip().upper() for x in args.iso3.split(",") if x.strip()]
    else:
        s = max(args.start, 0)
        run_list = iso_all[s:(s + args.n) if args.n and args.n > 0 else None]

    all_rows = []

    # IMPORTANT: signature depends on rp_map (not a hard-coded default)
    new_sig = build_run_signature(args, rp_map)

    for iso3 in run_list:
        out_dir = OUT / iso3
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"sda_summary_{iso3}.csv"

        if args.skip_done and should_skip_iso3(out_dir, iso3, new_sig, rp_map=rp_map, write_depthbin=args.write_depthbin):
            print(f"[SKIP] {iso3} (signature matches + outputs complete)")
            try:
                all_rows.append(pd.read_csv(out_csv))
            except Exception:
                pass
            continue

        if args.skip_done and out_csv.exists():
            old = read_old_signature(out_dir, iso3)
            old_hash = old.get("signature_sha256") if isinstance(old, dict) else None
            print(f"[RE-RUN] {iso3} (signature differs or outputs incomplete) old={old_hash} new={new_sig.get('signature_sha256')}")

        try:
            df = process_country(
                iso3=iso3,
                admin0=admin0,
                rp_map=rp_map,
                lulc_path=lulc_path,
                mapping=mapping,
                depth_threshold_m=args.depth_threshold,
                all_touched=args.all_touched,
                include_pasture=args.include_pasture,
                use_roads=args.use_roads,
                roads_path=Path(args.roads_path),
                with_pop=args.with_pop,
                pop_path=Path(args.pop_path),
                write_depthbin=args.write_depthbin,
                depthbin_max=args.depthbin_max,
            )

            write_signature(out_dir, iso3, new_sig)

            if not df.empty:
                all_rows.append(df)

        except Exception as e:
            print(f"[ERROR] {iso3}: {e}")

    if all_rows:
        g = pd.concat(all_rows, ignore_index=True)
        out_global = OUT / "global_sda_summary_countries.csv"
        g.to_csv(out_global, index=False)
        print(f"\n[DONE] Global summary → {out_global}")
    else:
        print("[WARN] No outputs produced.")


if __name__ == "__main__":
    main()

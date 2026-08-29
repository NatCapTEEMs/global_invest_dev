#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Flood GEP — Step 4D: GLOBAL EXPORT + PACKAGING (CSV + GPKG + optional rasters)
NatCap TEEMs / GEP Flood (JRC-INCA–style depth-based damages)
UPDATED: Dec 2025 — SCRATCH outputs + WB region enrichment + robust file discovery
===============================================================================

PURPOSE (WHAT STEP 4D PRODUCES)
------------------------------
Step 4D is the GLOBAL “packaging” step that consolidates per-country results into
publication-ready global products.

It produces:

A) Global tabular outputs (CSV)
   1) One row per ISO3 with Expected Annual Damage (EAD) in USD2019:
        step4d_country_ead_USD2019.csv
   2) One-row global total:
        step4d_global_totals_USD2019.csv

B) Global vector output (GeoPackage, GPKG)
   - Admin0 polygons joined with EAD + metadata fields:
        step4d_country_ead_USD2019.gpkg  (layer: country_ead_usd2019)

C) Optional per-country EAD rasters + optional global mosaic
   - If Step 4B wrote per-RP damage rasters, Step 4D can integrate them over
     exceedance probability to compute per-pixel EAD rasters, and mosaic them globally.

-------------------------------------------------------------------------------
SCIENTIFIC DEFINITIONS (PUBLICATION LANGUAGE)
---------------------------------------------
Return Period (RP):
  - A recurrence interval in years (e.g., RP=100 means 1% annual exceedance).

Exceedance Probability:
  - p = 1 / RP

Damage–Probability Function:
  - For each country, Step 4B provides damages for discrete return periods:
      D(RP) in USD2019
    which we treat as D(p) after mapping p=1/RP.

Expected Annual Damage (EAD):
  - EAD is the expected value of annual flood damage:
      EAD = ∫_0^1 D(p) dp
  - Step 4D uses trapezoidal numerical integration in probability space.

Anchors / Tail Behavior (Defensible defaults used across this pipeline):
  - Frequent-event anchor at p=1: add point (p=1, D=0) by default.
    Rationale: Step 4B uses a flood depth threshold + SDA masking; extremely
    frequent shallow inundation is conservatively treated as zero-damage in
    this global JRC-style setup.
  - Rare-event tail as p→0: default "flat" from max RP to p=0 (hold last damage).
    Rationale: avoids forcing damages to zero in the extreme tail; contribution
    is small because tail width is 1/RPmax.

-------------------------------------------------------------------------------
INPUTS (DATA DEPENDENCIES)
-------------------------
1) Step 4C outputs (OPTIONAL for Step 4D; REQUIRED if you want country EAD tables)
   Typical locations:
     outputs_root/<ISO3>/step4c_ead_USD2019.csv
   BUT THIS SCRIPT IS ROBUST and will also search recursively under each ISO3 dir.

2) Admin0 country boundaries GPKG (REQUIRED for GPKG output and/or completeness)
   Example:
     $GEP_FLOOD_ROOT/inputs/country_vector/country_boundary_r250_with_iso3.gpkg

   Must include:
     - ISO3 field (commonly: iso3, adm0_a3, iso_a3)
     - Region field: region_wb     <-- REQUIRED BY DEFAULT IN THIS SCRIPT
     - A country name field (optional; auto-detected if present)

3) Step 4B per-RP damage rasters (ONLY if computing raster EAD)
   Under:
     outputs_root/<ISO3>/rasters/

   This script supports multiple naming conventions, including:
     - damage_USD2019_rp10.tif    (your current pipeline)
     - step4b_damage_rp10_USD2019.tif
     - damage_rp10_USD2019.tif
     - and robust glob fallbacks

-------------------------------------------------------------------------------
MSI DIRECTORY CONTEXT (YOUR CURRENT PATHS)
-----------------------------------------
You are writing model outputs to SCRATCH:

  outputs_root = /scratch.global/gep_flood/damph002/outputs

Step 4D global outputs will be written to:

  out_dir      = /scratch.global/gep_flood/damph002/outputs/_global

-------------------------------------------------------------------------------
USAGE EXAMPLES (Jupyter)
------------------------
A) CSV + totals + GPKG (recommended first)
%run "$GEP_FLOOD_ROOT/code_final/global_invest/flood/scripts/flood_gep_step4d_export_global_USD2019.py" \
  --admin0-gpkg "$GEP_FLOOD_ROOT/inputs/country_vector/country_boundary_r250_with_iso3.gpkg" \
  --outputs-root "/scratch.global/gep_flood/damph002/outputs" \
  --out-dir "/scratch.global/gep_flood/damph002/outputs/_global" \
  --write-csvs --write-gpkg \
  --fill-missing-zero \
  --region-col region_wb

B) Also compute per-country EAD rasters + global mosaic
%run "$GEP_FLOOD_ROOT/code_final/global_invest/flood/scripts/flood_gep_step4d_export_global_USD2019.py" \
  --admin0-gpkg "$GEP_FLOOD_ROOT/inputs/country_vector/country_boundary_r250_with_iso3.gpkg" \
  --outputs-root "/scratch.global/gep_flood/damph002/outputs" \
  --out-dir "/scratch.global/gep_flood/damph002/outputs/_global" \
  --write-csvs --write-gpkg \
  --compute-ead-rasters --mosaic-global-raster \
  --rps 10,20,50,500 \
  --fill-missing-zero \
  --region-col region_wb

Notes:
- If you want strict backward compatibility with older notebooks:
    --include-missing-as-zero  (alias for --fill-missing-zero)
    --status-report            (no-op; status is always included in output tables)

-------------------------------------------------------------------------------
PUBLICATION / METHODS NOTES
---------------------------
- Units:
    EAD is in absolute USD2019 per year (not “millions” unless you scale it).
- Step 4D reports:
    (i) country-level EAD (USD2019/year)
    (ii) global total EAD (sum across ISO3; USD2019/year)
- Missing-country handling:
    Optionally “fill missing with zero” so every Admin0 polygon appears in the
    final CSV/GPKG, with a status flag that documents completeness.

===============================================================================
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional (only needed if raster flags enabled)
try:
    import rasterio
    from rasterio.merge import merge as rio_merge
    _HAS_RIO = True
except Exception:
    _HAS_RIO = False

# Optional (only needed if --write-gpkg or --fill-missing-zero)
try:
    import geopandas as gpd
    _HAS_GPD = True
except Exception:
    _HAS_GPD = False


# -----------------------------------------------------------------------------
# Helpers: robust parsing / column matching
# -----------------------------------------------------------------------------

def _to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


def _norm(s: str) -> str:
    s = str(s).strip().lower().replace("_", " ")
    s = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)
    return " ".join(s.split())


def _find_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    """
    Robust column matching:
      - exact normalized match
      - fallback "contains" match
    """
    norm_map = {_norm(c): c for c in df.columns}
    for cand in candidates:
        k = _norm(cand)
        if k in norm_map:
            return norm_map[k]
    for cand in candidates:
        k = _norm(cand)
        for kk, orig in norm_map.items():
            if k in kk:
                return orig
    return None


def list_iso3_dirs(outputs_root: Path) -> List[Path]:
    """
    ISO3 dirs are direct children of outputs_root with 3-letter names.
    Example: outputs/GMB, outputs/USA, outputs/ZAF
    """
    if not outputs_root.exists():
        raise FileNotFoundError(f"outputs_root does not exist: {outputs_root}")
    return sorted([p for p in outputs_root.iterdir() if p.is_dir() and len(p.name) == 3])


# -----------------------------------------------------------------------------
# Step 4C discovery + reading (robust)
# -----------------------------------------------------------------------------

def find_step4c_file(iso3_dir: Path, ead_filename: str) -> Optional[Path]:
    """
    Find Step 4C EAD CSV within an ISO3 directory.

    Priority:
      1) exact filename at ISO3 root
      2) exact filename anywhere under ISO3 (recursive)
      3) heuristic glob patterns (recursive)

    Returns the newest match by modified time if multiple are found.
    """
    p0 = iso3_dir / ead_filename
    if p0.exists():
        return p0

    hits = list(iso3_dir.rglob(ead_filename))
    if hits:
        return max(hits, key=lambda p: p.stat().st_mtime)

    patterns = [
        "*step4c*ead*USD2019*.csv",
        "*step4c*EAD*USD2019*.csv",
        "*ead*USD2019*.csv",
        "*step4c*ead*.csv",
    ]
    cand: List[Path] = []
    for pat in patterns:
        cand.extend(list(iso3_dir.rglob(pat)))
    if cand:
        return max(cand, key=lambda p: p.stat().st_mtime)

    return None


def read_step4c_ead_robust(step4c_csv: Path, iso3_hint: str) -> pd.DataFrame:
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
    c_metric = _find_col(df, ("metric",))
    c_value = _find_col(df, ("value", "val"))
    if c_metric is not None and c_value is not None:
        m = df[c_metric].astype(str).str.strip().str.lower()
        # Accept any row whose metric contains 'ead' and 'usd2019'
        mask = m.str.contains("ead") & (m.str.contains("usd2019") | m.str.contains("2019"))
        if mask.any():
            v = _to_float(df.loc[mask, c_value].iloc[0])
            return pd.DataFrame([{"iso3": iso3_hint, "ead_usd2019": v}])

    # Case A: table with EAD column
    c_iso = _find_col(df, ("iso3", "country", "country_iso3"))
    c_ead = _find_col(df, (
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
    out["ead_usd2019"] = out["ead_usd2019"].apply(_to_float)
    out = out.dropna(subset=["iso3", "ead_usd2019"])
    return out


def compute_global_totals(country_df: pd.DataFrame) -> pd.DataFrame:
    """
    Global total EAD is the sum across ISO3 entries.
    Units: USD2019/year (absolute, not scaled).
    """
    total = float(country_df["ead_usd2019"].sum(skipna=True))
    return pd.DataFrame([{"metric": "global_total_ead_usd2019", "value": total}])


# -----------------------------------------------------------------------------
# Raster EAD from Step 4B rasters (optional)
# -----------------------------------------------------------------------------

def rp_to_p(rp: float) -> float:
    return 1.0 / float(rp)


def integrate_ead_pixelwise(
    rasters_by_rp: Dict[int, Path],
    out_tif: Path,
    *,
    add_p1_zero: bool = True,
    tail_mode: str = "flat",
    compress: str = "deflate"
) -> None:
    """
    Compute per-pixel EAD raster from per-RP damage rasters using trapezoid
    integration in probability space p = 1/RP over [0,1].

    Requirements:
      - >= 2 rasters
      - All rasters aligned (same grid/transform/shape)
    """
    if not _HAS_RIO:
        raise ImportError("rasterio not available; cannot compute rasters.")
    if tail_mode not in {"flat", "zero"}:
        raise ValueError("tail_mode must be 'flat' or 'zero'")

    rps = sorted(rasters_by_rp.keys())
    if len(rps) < 2:
        raise ValueError("Need at least 2 RP rasters to integrate EAD pixelwise.")

    srcs = {rp: rasterio.open(rasters_by_rp[rp]) for rp in rps}
    ref = srcs[rps[0]]

    meta = ref.meta.copy()
    nod = meta.get("nodata", None)
    meta.update(dtype="float32", count=1, compress=compress, nodata=np.float32(-9999.0 if nod is None else nod))

    out_tif.parent.mkdir(parents=True, exist_ok=True)

    p = np.array([rp_to_p(rp) for rp in rps], dtype="float64")

    with rasterio.open(out_tif, "w", **meta) as dst:
        for _, window in ref.block_windows(1):
            stack = []
            for rp in rps:
                arr = srcs[rp].read(1, window=window).astype("float64")
                nod_rp = srcs[rp].nodata
                if nod_rp is not None:
                    # Conservative: treat nodata as 0 damage
                    arr = np.where(arr == nod_rp, 0.0, arr)
                stack.append(arr)

            Y = np.stack(stack, axis=0)  # (n_rp, h, w)

            # Frequent-event anchor p=1, D=0
            if add_p1_zero:
                p_ext = np.concatenate([[1.0], p])
                Y_ext = np.concatenate([np.zeros_like(Y[:1, :, :]), Y], axis=0)
            else:
                p_ext = p.copy()
                Y_ext = Y

            # Tail to p=0
            p_ext = np.concatenate([p_ext, [0.0]])
            if tail_mode == "flat":
                Y_ext = np.concatenate([Y_ext, Y_ext[-1:, :, :]], axis=0)
            else:
                Y_ext = np.concatenate([Y_ext, np.zeros_like(Y_ext[-1:, :, :])], axis=0)

            # Integrate over p in increasing order 0->1
            x = p_ext[::-1]
            y = Y_ext[::-1, :, :]
            ead = np.trapezoid(y, x, axis=0).astype("float32")

            dst.write(ead, 1, window=window)

    for rp in rps:
        srcs[rp].close()


def find_step4b_raster(iso3_dir: Path, rp: int) -> Optional[Path]:
    """
    Locate Step 4B per-RP rasters under outputs/<ISO3>/rasters/.

    Supports multiple naming conventions, including your current pipeline:
      - damage_USD2019_rp10.tif
      - step4b_damage_rp10_USD2019.tif
      - damage_rp10_USD2019.tif
      - generic glob fallback

    Returns the newest match by modified time if multiple are found.
    """
    rasdir = iso3_dir / "rasters"
    if not rasdir.exists():
        return None

    candidates = [
        rasdir / f"damage_USD2019_rp{rp}.tif",           # ✅ your pipeline
        rasdir / f"step4b_damage_rp{rp}_USD2019.tif",
        rasdir / f"damage_rp{rp}_USD2019.tif",
        rasdir / f"damage_USD2019_rp{rp}y.tif",
        rasdir / f"damage_USD2019_rp{rp}yr.tif",
        rasdir / f"damage_rp{rp}.tif",
    ]
    for p in candidates:
        if p.exists():
            return p

    hits = list(rasdir.glob(f"*rp{rp}*USD2019*.tif"))
    if hits:
        return max(hits, key=lambda p: p.stat().st_mtime)

    hits2 = list(rasdir.glob(f"*rp{rp}*.tif"))
    if hits2:
        return max(hits2, key=lambda p: p.stat().st_mtime)

    return None


def mosaic_global(ead_rasters: List[Path], out_tif: Path) -> None:
    """
    Mosaic per-ISO3 EAD rasters into one global raster.
    Assumes all rasters share a common CRS/resolution (as produced by the pipeline).
    """
    if not _HAS_RIO:
        raise ImportError("rasterio not available; cannot mosaic rasters.")
    if not ead_rasters:
        raise ValueError("No rasters provided for mosaicking.")

    out_tif.parent.mkdir(parents=True, exist_ok=True)

    srcs = [rasterio.open(p) for p in ead_rasters]
    mosaic, out_transform = rio_merge(srcs)

    out_meta = srcs[0].meta.copy()
    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "count": 1,
        "dtype": mosaic.dtype,
        "compress": "deflate",
    })

    with rasterio.open(out_tif, "w", **out_meta) as dst:
        dst.write(mosaic[0], 1)

    for s in srcs:
        s.close()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 4D: Global export (CSV + GPKG + optional EAD rasters) for Flood GEP."
    )

    p.add_argument("--admin0-gpkg", required=True, type=str,
                   help="Admin0 polygons GPKG that contains ISO3 and region_wb fields.")
    p.add_argument("--admin0-layer", default=None, type=str,
                   help="Optional: layer name if the GPKG has multiple layers.")

    p.add_argument("--outputs-root", required=True, type=str,
                   help="Root directory with ISO3 subfolders containing Step 4C outputs and/or Step 4B rasters. "
                        "Example: /scratch.global/gep_flood/damph002/outputs")

    p.add_argument("--out-dir", required=True, type=str,
                   help="Directory to write global Step 4D products. "
                        "Example: /scratch.global/gep_flood/damph002/outputs/_global")

    p.add_argument("--ead-filename", default="step4c_ead_USD2019.csv", type=str,
                   help="Default Step 4C EAD filename; script also searches recursively under each ISO3 dir.")

    # Enrichment fields from admin0
    p.add_argument("--iso3-col", default=None, type=str,
                   help="ISO3 column name in Admin0 GPKG. If omitted, auto-detect among {iso3, adm0_a3, iso_a3}.")
    p.add_argument("--name-col", default=None, type=str,
                   help="Country name column in Admin0 GPKG. If omitted, auto-detect among common candidates.")
    p.add_argument("--region-col", default="region_wb", type=str,
                   help="Region column in Admin0 GPKG. Default is 'region_wb' (required for WB regions).")

    # Output toggles
    p.add_argument("--write-csvs", action="store_true", help="Write global CSV outputs.")
    p.add_argument("--write-gpkg", action="store_true", help="Write global GPKG output (requires geopandas).")

    # Completeness behavior
    p.add_argument("--fill-missing-zero", action="store_true",
                   help="Ensure all Admin0 ISO3 appear in global outputs. "
                        "Countries missing Step 4C will be included with EAD=0 and status='missing_step4c'.")

    # Backward-compatible alias (older notebooks)
    p.add_argument("--include-missing-as-zero", dest="fill_missing_zero", action="store_true",
                   help="Alias for --fill-missing-zero (backward compatible).")

    # Raster options
    p.add_argument("--compute-ead-rasters", action="store_true",
                   help="Compute per-ISO3 EAD rasters from Step 4B per-RP damage rasters (requires rasterio).")
    p.add_argument("--mosaic-global-raster", action="store_true",
                   help="Mosaic per-ISO3 EAD rasters into one global raster (requires rasterio).")
    p.add_argument("--rps", default="10,20,50,100,200,500", type=str,
                   help="Comma-separated RPs expected for Step 4B rasters.")
    p.add_argument("--tail-mode", default="flat", choices=["flat", "zero"],
                   help="Tail mode for raster EAD integration (consistent with Step 4C philosophy).")
    p.add_argument("--no-add-p1-zero", dest="add_p1_zero", action="store_false",
                   help="Disable p=1, D=0 anchor in raster EAD integration.")
    p.set_defaults(add_p1_zero=True)

    # Backward-compatible no-op (status is always included when writing tables)
    p.add_argument("--status-report", action="store_true",
                   help="Backward compatible no-op. Status column is always included in the country output table.")

    return p.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    admin0_gpkg = Path(args.admin0_gpkg)
    outputs_root = Path(args.outputs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rps = [int(x) for x in args.rps.split(",") if str(x).strip()]

    print(f"[INFO] Step 4D GLOBAL")
    print(f"[INFO] outputs_root: {outputs_root}")
    print(f"[INFO] out_dir:      {out_dir}")
    print(f"[INFO] Step4C file (preferred name): {args.ead_filename}")
    print(f"[INFO] RPs:          {rps}")
    print(f"[INFO] write_csvs:   {bool(args.write_csvs)}")
    print(f"[INFO] write_gpkg:   {bool(args.write_gpkg)}")
    print(f"[INFO] fill_missing_zero: {bool(args.fill_missing_zero)}")
    print(f"[INFO] rasters:      compute={bool(args.compute_ead_rasters)}, mosaic={bool(args.mosaic_global_raster)}")

    if args.compute_ead_rasters and not _HAS_RIO:
        print("[WARN] rasterio not available; raster flags will fail.")
    if (args.write_gpkg or args.fill_missing_zero) and not _HAS_GPD:
        print("[WARN] geopandas not available; --write-gpkg / --fill-missing-zero will fail.")

    # -------------------------------------------------------------------------
    # Load Admin0 boundary file (for names + region + completeness)
    # -------------------------------------------------------------------------
    admin0 = None
    enrich = None
    iso_col = None

    if args.write_gpkg or args.fill_missing_zero:
        if not _HAS_GPD:
            raise ImportError("geopandas is required for --write-gpkg or --fill-missing-zero.")
        if not admin0_gpkg.exists():
            raise FileNotFoundError(f"Admin0 GPKG not found: {admin0_gpkg}")

        admin0 = gpd.read_file(admin0_gpkg, layer=args.admin0_layer) if args.admin0_layer else gpd.read_file(admin0_gpkg)

        # ISO3 column detection
        if args.iso3_col:
            iso_col = args.iso3_col
            if iso_col not in admin0.columns:
                raise ValueError(f"--iso3-col '{iso_col}' not found. Available columns: {list(admin0.columns)}")
        else:
            # One list, in flood_functions. A second one here drifts.
            from global_invest.flood.flood_functions import pick_iso3_column
            iso_col = pick_iso3_column(admin0)
            if iso_col is None:
                raise ValueError(
                    f"No ISO3 column in the Admin0 layer. Columns={list(admin0.columns)}")

        # Country name column (best effort)
        if args.name_col:
            name_col = args.name_col
            if name_col not in admin0.columns:
                raise ValueError(f"--name-col '{name_col}' not found. Available columns: {list(admin0.columns)}")
        else:
            name_col = _find_col(admin0, ("name", "country", "country_name", "admin", "name_long", "name_en", "sovereignt"))

        # Region column STRICT (default region_wb)
        region_col = args.region_col
        if region_col not in admin0.columns:
            raise ValueError(
                f"Requested region column '{region_col}' not found in Admin0 GPKG. "
                f"Available columns: {list(admin0.columns)}"
            )

        admin0[iso_col] = admin0[iso_col].astype(str).str.upper().str.strip()

        keep = [iso_col, region_col]
        if name_col is not None:
            keep.append(name_col)

        enrich = admin0[keep].drop_duplicates(subset=[iso_col]).copy()
        enrich = enrich.rename(columns={iso_col: "iso3", region_col: "region_wb"})
        if name_col is not None:
            enrich = enrich.rename(columns={name_col: "country_name"})

    # -------------------------------------------------------------------------
    # Harvest Step 4C per-country files (robust)
    # -------------------------------------------------------------------------
    rows = []
    n_seen = 0
    iso3_dirs = list_iso3_dirs(outputs_root)

    for iso3_dir in iso3_dirs:
        iso3 = iso3_dir.name.upper()
        n_seen += 1

        f = find_step4c_file(iso3_dir, args.ead_filename)
        if f is None:
            continue

        try:
            part = read_step4c_ead_robust(f, iso3_hint=iso3)
            if not part.empty:
                rows.append(part)
        except Exception as e:
            warnings.warn(f"[WARN] {iso3}: failed reading Step4C at {f}: {e}")

    if rows:
        country = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["iso3"], keep="last")
    else:
        country = pd.DataFrame(columns=["iso3", "ead_usd2019"])

    country["iso3"] = country["iso3"].astype(str).str.upper().str.strip()

    print(f"[INFO] ISO3 folders scanned: {n_seen}")
    print(f"[INFO] Step4C files loaded:  {len(country):,} ISO3 with EAD")

    # -------------------------------------------------------------------------
    # Ensure completeness (include all Admin0 ISO3, even if missing Step 4C)
    # -------------------------------------------------------------------------
    if args.fill_missing_zero:
        if enrich is None:
            raise RuntimeError("Internal error: enrich table not prepared, but fill_missing_zero is True.")
        all_iso3 = enrich[["iso3"]].drop_duplicates().copy()
        out_all = all_iso3.merge(country, on="iso3", how="left")

        out_all["status"] = np.where(out_all["ead_usd2019"].isna(), "missing_step4c", "ok")
        out_all["ead_usd2019"] = out_all["ead_usd2019"].fillna(0.0)

        # Enrichment columns
        if "country_name" in enrich.columns:
            out_all = out_all.merge(enrich[["iso3", "country_name"]], on="iso3", how="left")
        out_all = out_all.merge(enrich[["iso3", "region_wb"]], on="iso3", how="left")

        # Column order (publication-friendly)
        col_order = ["iso3"]
        if "country_name" in out_all.columns:
            col_order.append("country_name")
        col_order += ["region_wb", "ead_usd2019", "status"]

        country_out = out_all[col_order].copy()

        n_missing = int((country_out["status"] == "missing_step4c").sum())
        if n_missing:
            print(f"[INFO] fill-missing-zero enabled: {n_missing} Admin0 entries missing Step4C → EAD set to 0")

    else:
        # No forced completeness; enrich if available
        country_out = country.copy()
        country_out["status"] = "ok"
        if enrich is not None:
            if "country_name" in enrich.columns:
                country_out = country_out.merge(enrich[["iso3", "country_name"]], on="iso3", how="left")
            country_out = country_out.merge(enrich[["iso3", "region_wb"]], on="iso3", how="left")

        col_order = ["iso3"]
        if "country_name" in country_out.columns:
            col_order.append("country_name")
        if "region_wb" in country_out.columns:
            col_order.append("region_wb")
        col_order += ["ead_usd2019", "status"]
        country_out = country_out[[c for c in col_order if c in country_out.columns]]

    country_out = country_out.sort_values("iso3").reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Write CSV outputs
    # -------------------------------------------------------------------------
    if args.write_csvs:
        out_country_csv = out_dir / "step4d_country_ead_USD2019.csv"
        out_totals_csv = out_dir / "step4d_global_totals_USD2019.csv"

        country_out.to_csv(out_country_csv, index=False)

        totals = compute_global_totals(country_out)
        totals.to_csv(out_totals_csv, index=False)

        print(f"[OK] Wrote country CSV → {out_country_csv} (rows={len(country_out):,})")
        print(f"[OK] Wrote totals CSV  → {out_totals_csv}")

    # -------------------------------------------------------------------------
    # Write GPKG output (Admin0 geometry + EAD + enrichment)
    # -------------------------------------------------------------------------
    if args.write_gpkg:
        if admin0 is None or iso_col is None:
            raise RuntimeError("Admin0 was not loaded; cannot write GPKG.")
        if not _HAS_GPD:
            raise ImportError("geopandas not available; cannot write GPKG.")

        g = admin0.copy()
        g[iso_col] = g[iso_col].astype(str).str.upper().str.strip()

        g2 = g.merge(country_out, left_on=iso_col, right_on="iso3", how="left")

        out_gpkg = out_dir / "step4d_country_ead_USD2019.gpkg"
        layer = "country_ead_usd2019"
        g2.to_file(out_gpkg, layer=layer, driver="GPKG")
        print(f"[OK] Wrote GPKG → {out_gpkg} (layer={layer})")

    # -------------------------------------------------------------------------
    # Optional: per-ISO3 EAD rasters + global mosaic
    # -------------------------------------------------------------------------
    ead_rasters: List[Path] = []
    if args.compute_ead_rasters:
        if not _HAS_RIO:
            raise ImportError("rasterio not available; cannot compute EAD rasters.")

        for iso3_dir in iso3_dirs:
            iso3 = iso3_dir.name.upper()

            rasters_by_rp: Dict[int, Path] = {}
            for rp in rps:
                pth = find_step4b_raster(iso3_dir, rp)
                if pth is not None:
                    rasters_by_rp[rp] = pth

            if len(rasters_by_rp) < 2:
                warnings.warn(f"[WARN] {iso3}: <2 RP rasters found; skipping EAD raster.")
                continue

            out_tif = iso3_dir / "rasters" / "step4d_ead_USD2019.tif"
            integrate_ead_pixelwise(
                rasters_by_rp,
                out_tif,
                add_p1_zero=args.add_p1_zero,
                tail_mode=args.tail_mode
            )
            print(f"[OK] {iso3}: wrote EAD raster → {out_tif}")
            ead_rasters.append(out_tif)

        if args.mosaic_global_raster:
            if not ead_rasters:
                warnings.warn("[WARN] No per-ISO3 EAD rasters available to mosaic.")
            else:
                out_global_tif = out_dir / "step4d_global_ead_USD2019_mosaic.tif"
                mosaic_global(ead_rasters, out_global_tif)
                print(f"[OK] Wrote global mosaic EAD raster → {out_global_tif}")

    print("[DONE] Step 4D export complete.")


if __name__ == "__main__":
    warnings.simplefilter("default")
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Build JRC Flood Depth–Damage Tables in USD2019 (Global, by ISO3)
NatCap TEEMs / GEP Flood — Updated Dec 16 2025
===============================================================================

WHAT THIS SCRIPT DOES
---------------------
This script produces depth–damage lookup tables (USD2019 per m²) for use in Flood
GEP Step 4 (monetary damages / avoided damages). It supports TWO equivalent
starting points:

(A) START FROM A "CANONICAL" EUR2010 TABLE (WIDE)
    - A prebuilt table with depth columns like: 0m, 0.5m, 1m, ...
    - Example columns:
        ISO3, JRC_Region, LandType, Max_Damage_Euro_per_m2, 0m, 0.5m, 1m, ...

(B) BUILD THE CANONICAL TABLE FROM COMPONENTS
    - fractional curves (region × landtype × depth → fraction)
    - max damages (country × landtype → max_damage_eur_m2)
    - iso3 to JRC region mapping (iso3 → JRC_Region)

OUTPUTS
-------
1) Per-sector (LandType) damage functions:
   - --out-long : long table  [iso3, landtype, depth_m, damage_per_m2, currency]
   - --out-wide : wide table  [iso3, landtype, 0m, 0.5m, 1m, ... , 6m]

2) Aggregated SDA damage functions (for raster SDA types used in Step 4):
   - --out-sda-long : long table [iso3, sda_type, depth_m, damage_per_m2, currency]
   - --out-sda-wide : wide table [iso3, sda_type, 0m, 0.5m, 1m, ... , 6m]

SDA aggregation is conservative and configurable:
- Agriculture → "crop"
- Residential/Commercial/Industrial/Transport/Roads → "artif"
- Optional: pasture can be set equal to crop (recommended if pasture is used)

IMPORTANT FIXES INCLUDED (YOUR ERRORS)
-------------------------------------
1) Robust column detection that tolerates underscore→space normalization.
   - Your fractional curves have 'depth_m' but a "normalize_label" helper often
     turns it into 'depth m'. This script explicitly checks BOTH forms.

2) Robust "combined_multiplier" loading:
   - Supports factors CSV *or* JSON.
   - From CSV: accepts either:
       (i) a tidy 'name,value' format with combined_multiplier row, OR
       (ii) columns like fx_usd_per_eur_2010_avg and inflator_us_2010_to_2019
   - From JSON: searches recursively for:
       combined_multiplier
       fx_usd_per_eur_2010_avg + inflator_us_2010_to_2019

3) NaN-safe pivoting:
   - Drops rows with missing depth_m before formatting "0.5m" etc.

SECTORS SUPPORTED
-----------------
The script supports ALL JRC landtypes present in your fraction curves:
- Agriculture
- Residential buildings
- Commercial buildings
- Industrial buildings
- Infrastructure - roads
- Transport

These appear in sector outputs as cleaned sector labels:
- Agriculture, Residential, Commercial, Industrial, Roads, Transport

USAGE EXAMPLES
--------------
(1) Convert an existing canonical EUR table:
    python build_damage_table_USD2019.py \
      --canonical-eur /.../country_landtype_flood_damage_JRC_EUR_m2.csv \
      --factors-csv  /.../_currency_audit/currency_conversion_factors_EUR2010_to_USD2019.csv \
      --out-long     /.../damage_functions_depth_USD2019_long.csv \
      --out-wide     /.../damage_functions_depth_USD2019_wide.csv \
      --out-sda-long /.../damage_functions_sda_depth_USD2019_long.csv \
      --out-sda-wide /.../damage_functions_sda_depth_USD2019_wide.csv \
      --set-pasture-equal-crop \
      --audit-dir /.../_currency_audit

(2) Build canonical from components, then convert:
    python build_damage_table_USD2019.py \
      --fractional-long-csv /.../jrc_fractional_curves_long.csv \
      --maxdamage-long-csv  /.../jrc_max_damage_values_long_with_iso3.csv \
      --iso3-region-csv     /.../iso3_to_jrc_region.csv \
      --factors-json        /.../_currency_audit/currency_conversion_factors_used.json \
      --out-long            /.../damage_functions_depth_USD2019_long.csv \
      --out-wide            /.../damage_functions_depth_USD2019_wide.csv \
      --out-sda-long        /.../damage_functions_sda_depth_USD2019_long.csv \
      --out-sda-wide        /.../damage_functions_sda_depth_USD2019_wide.csv \
      --set-pasture-equal-crop

===============================================================================
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Helpers: cleaning / normalization
# =============================================================================

def clean_missing(x):
    """Convert common string-missing tokens to np.nan; keep other values unchanged."""
    if x is None:
        return np.nan
    if isinstance(x, float) and np.isnan(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "na", "n/a", "null"}:
        return np.nan
    return x


def normalize_label(s: str) -> str:
    """
    Normalize a column name into a comparison key:
    - lowercase
    - underscores -> spaces
    - remove non-alphanum (keep spaces)
    - collapse spaces
    """
    s = str(s).strip().lower().replace("_", " ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_country_name(s: str) -> str:
    """Loose normalization for country-name matching."""
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def to_float(x) -> float:
    """Safe float conversion; returns np.nan on failure."""
    try:
        return float(x)
    except Exception:
        return np.nan


def parse_depth_colname(col: str) -> Optional[float]:
    """
    Parse wide depth column names like '0m', '0.5m', '1m' into numeric meters.
    Returns None if not a depth column.
    """
    c = str(col).strip().lower()
    if c.endswith("m"):
        c2 = c[:-1].strip()
        try:
            return float(c2)
        except Exception:
            return None
    return None


def fmt_depth(d: float) -> str:
    """Format numeric depth into a stable wide-column label, e.g., 1 -> '1m', 0.5 -> '0.5m'."""
    if d is None or (isinstance(d, float) and np.isnan(d)):
        raise ValueError("depth is NaN")
    d = float(d)
    return f"{int(d)}m" if abs(d - int(d)) < 1e-9 else f"{d}m"


def make_colmap(df: pd.DataFrame) -> Dict[str, str]:
    """Return dict: normalized_label -> actual_column_name."""
    out = {}
    for c in df.columns:
        out[normalize_label(c)] = c
    return out


def normalize_landtype_label(x: str) -> str:
    """
    Normalize JRC LandType labels to a stable set for merging and output.

    IMPORTANT:
    - Fraction curves landtypes include:
      'Agriculture', 'Residential buildings', 'Commercial buildings',
      'Industrial buildings', 'Infrastructure - roads', 'Transport'
    - Max damages (your rebuilt file) uses 'Residential' and 'Agriculture' etc.
      We standardize to the fraction landtypes as much as possible.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip()

    # Common canonicalization
    s0 = s.lower().strip()

    # Map short forms to JRC fraction labels
    if s0 == "residential":
        return "Residential buildings"
    if s0 == "commercial":
        return "Commercial buildings"
    if s0 == "industrial":
        return "Industrial buildings"
    if s0 in {"roads", "road", "infrastructure roads", "infrastructure - road"}:
        return "Infrastructure - roads"
    if s0 in {"transport", "transportation"}:
        return "Transport"
    if s0 == "agriculture":
        return "Agriculture"

    # Pass through already-correct JRC labels
    # (keeps exact fraction-curve landtypes)
    return s


def sector_label_from_landtype(landtype: str) -> str:
    """
    Output-friendly sector label for wide/long sector tables.
    """
    lt = normalize_landtype_label(landtype)
    m = {
        "Agriculture": "Agriculture",
        "Residential buildings": "Residential",
        "Commercial buildings": "Commercial",
        "Industrial buildings": "Industrial",
        "Infrastructure - roads": "Roads",
        "Transport": "Transport",
    }
    return m.get(lt, lt)


# =============================================================================
# Currency factors: load combined_multiplier from CSV or JSON
# =============================================================================

def load_combined_multiplier(
    factors_csv: Optional[Path] = None,
    factors_json: Optional[Path] = None
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
        df = df.applymap(clean_missing)

        cols = make_colmap(df)

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
        data = json.loads(Path(factors_json).read_text())

        def find_key(obj: Any, keynames: Iterable[str]) -> Optional[float]:
            """DFS search for numeric leaf with a matching key name."""
            if isinstance(obj, dict):
                for k, v in obj.items():
                    k0 = normalize_label(k)
                    if k0 in {normalize_label(kn) for kn in keynames}:
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


# =============================================================================
# Canonical EUR table: load from file or build from components
# =============================================================================

def load_canonical_eur_table(canonical_eur: Path, audit_dir: Optional[Path] = None) -> pd.DataFrame:
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
    df = df.applymap(clean_missing)

    cols = make_colmap(df)

    # If already long:
    has_long = ("depth_m" in cols or "depth m" in cols) and ("damage_per_m2" in cols or "damage per m2" in cols)
    if has_long:
        iso_col = cols.get("iso3") or cols.get("iso 3") or cols.get("iso")
        land_col = cols.get("landtype") or cols.get("land type") or cols.get("sector")
        depth_col = cols.get("depth_m") or cols.get("depth m") or cols.get("depth")
        dmg_col = cols.get("damage_per_m2") or cols.get("damage per m2") or cols.get("damage")
        out = df[[iso_col, land_col, depth_col, dmg_col]].copy()
        out.columns = ["iso3", "landtype", "depth_m", "damage_per_m2"]
        out["depth_m"] = out["depth_m"].apply(to_float)
        out["damage_per_m2"] = out["damage_per_m2"].apply(to_float)
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
            audit_dir.mkdir(parents=True, exist_ok=True)
            (audit_dir / "diagnostics_canonical_missing_iso_or_landtype.txt").write_text(msg)
        raise ValueError(msg)

    # depth columns are like '0m','0.5m'... OR numeric strings etc
    depth_cols = []
    depth_vals = []
    for c in df.columns:
        d = parse_depth_colname(c)
        if d is not None:
            depth_cols.append(c)
            depth_vals.append(d)

    if len(depth_cols) == 0:
        msg = (
            "Canonical EUR table appears wide but has no depth columns like '0m','0.5m',... "
            f"Columns found: {list(df.columns)}"
        )
        if audit_dir:
            audit_dir.mkdir(parents=True, exist_ok=True)
            (audit_dir / "diagnostics_canonical_no_depth_cols.txt").write_text(msg)
        raise ValueError(msg)

    # Melt wide -> long
    tmp = df[[iso_col, land_col] + depth_cols].copy()
    tmp = tmp.rename(columns={iso_col: "iso3", land_col: "landtype"})
    long = tmp.melt(id_vars=["iso3", "landtype"], var_name="depth_col", value_name="damage_per_m2")
    long["depth_m"] = long["depth_col"].apply(parse_depth_colname).astype(float)
    long = long.drop(columns=["depth_col"])

    long["iso3"] = long["iso3"].astype(str).str.strip()
    long["landtype"] = long["landtype"].astype(str).str.strip()
    long["damage_per_m2"] = long["damage_per_m2"].apply(to_float)

    return long


def build_canonical_from_components(
    fractional_long_csv: Path,
    maxdamage_long_csv: Path,
    iso3_region_csv: Path,
    audit_dir: Optional[Path] = None
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
    frac = pd.read_csv(fractional_long_csv).applymap(clean_missing)
    maxd = pd.read_csv(maxdamage_long_csv).applymap(clean_missing)
    iso3reg = pd.read_csv(iso3_region_csv).applymap(clean_missing)

    fcols = make_colmap(frac)
    mcols = make_colmap(maxd)
    rcols = make_colmap(iso3reg)

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
    frac2["LandType"] = frac2["LandType"].astype(str).apply(normalize_landtype_label).str.strip()
    frac2["depth_m"] = frac2["depth_m"].apply(to_float)
    frac2["fraction"] = frac2["fraction"].apply(to_float)

    max2 = maxd[[m_iso, m_land, m_max]].copy()
    max2.columns = ["iso3", "LandType", "max_damage_eur_m2"]
    max2["iso3"] = max2["iso3"].astype(str).str.strip()
    max2["LandType"] = max2["LandType"].astype(str).apply(normalize_landtype_label).str.strip()
    max2["max_damage_eur_m2"] = max2["max_damage_eur_m2"].apply(to_float)

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
            audit_dir.mkdir(parents=True, exist_ok=True)
            missing_reg.to_csv(audit_dir / "diagnostics_missing_jrc_region_for_iso3.csv", index=False)

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
            audit_dir.mkdir(parents=True, exist_ok=True)
            missing_frac.to_csv(audit_dir / "diagnostics_missing_fraction_landtype_region.csv", index=False)

    # Compute depth damages
    out["damage_per_m2"] = out["max_damage_eur_m2"] * out["fraction"]

    eur_long = out[["iso3", "LandType", "depth_m", "damage_per_m2"]].copy()
    eur_long = eur_long.rename(columns={"LandType": "landtype"})
    eur_long["landtype"] = eur_long["landtype"].apply(normalize_landtype_label)
    eur_long["depth_m"] = eur_long["depth_m"].apply(to_float)
    eur_long["damage_per_m2"] = eur_long["damage_per_m2"].apply(to_float)

    # Drop bad rows
    eur_long = eur_long.dropna(subset=["iso3", "landtype", "depth_m", "damage_per_m2"])

    return eur_long


# =============================================================================
# Wide pivot + SDA mapping
# =============================================================================

def pivot_wide(long_df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    """
    Pivot long table into wide depth columns, dropping NaN depth rows.
    """
    tmp = long_df.copy()
    tmp = tmp.dropna(subset=["depth_m"])
    tmp["depth_col"] = tmp["depth_m"].apply(lambda d: fmt_depth(d))
    wide = tmp.pivot_table(
        index=group_cols,
        columns="depth_col",
        values=value_col,
        aggfunc="mean",
        fill_value=0.0,
    ).reset_index()
    wide.columns = [str(c) for c in wide.columns]
    return wide


def landtype_to_sda(landtype: str) -> Optional[str]:
    """
    Map sector landtypes to SDA classes used in flood exposure accounting.

    Conservative JRC-style SDA:
      - Cropland assets: Agriculture -> crop
      - Built/asset surfaces: Residential/Commercial/Industrial/Transport/Roads -> artif

    Returns None if the landtype is not used for SDA aggregation.
    """
    lt = normalize_landtype_label(landtype)

    if lt == "Agriculture":
        return "crop"
    if lt in {"Residential buildings", "Commercial buildings", "Industrial buildings", "Transport", "Infrastructure - roads"}:
        return "artif"

    return None


# =============================================================================
# CLI + Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build depth–damage functions in USD2019 (JRC-style)")

    # Path to canonical (optional)
    p.add_argument("--canonical-eur", type=str, default=None,
                   help="Canonical EUR2010 depth-damage table (wide or long). If omitted, build from components.")

    # Components (required if no canonical-eur)
    p.add_argument("--fractional-long-csv", type=str, default=None,
                   help="Fractional curves long CSV: region×landtype×depth -> fraction")
    p.add_argument("--maxdamage-long-csv", type=str, default=None,
                   help="Max damages long CSV WITH ISO3: iso3×landtype -> max_damage_eur_m2")
    p.add_argument("--iso3-region-csv", type=str, default=None,
                   help="ISO3 to JRC region mapping CSV")

    # Currency factors (either CSV or JSON)
    p.add_argument("--factors-csv", type=str, default=None,
                   help="Currency conversion factors CSV (must include combined_multiplier OR fx+inflator)")
    p.add_argument("--factors-json", type=str, default=None,
                   help="Currency conversion factors JSON (must include combined_multiplier OR fx+inflator)")

    # Outputs
    p.add_argument("--out-long", type=str, required=True,
                   help="Output sector long CSV (USD2019)")
    p.add_argument("--out-wide", type=str, required=True,
                   help="Output sector wide CSV (USD2019)")
    p.add_argument("--out-sda-long", type=str, required=True,
                   help="Output SDA long CSV (USD2019)")
    p.add_argument("--out-sda-wide", type=str, required=True,
                   help="Output SDA wide CSV (USD2019)")

    # Optional diagnostics/audit
    p.add_argument("--audit-dir", type=str, default=None, help="Directory for diagnostics outputs")

    # Options
    p.add_argument("--set-pasture-equal-crop", action="store_true",
                   help="If pasture SDA is used, set pasture damage equal to crop damage (recommended).")

    return p.parse_args()


def main():
    args = parse_args()

    canonical_eur = Path(args.canonical_eur) if args.canonical_eur else None
    fractional_long_csv = Path(args.fractional_long_csv) if args.fractional_long_csv else None
    maxdamage_long_csv = Path(args.maxdamage_long_csv) if args.maxdamage_long_csv else None
    iso3_region_csv = Path(args.iso3_region_csv) if args.iso3_region_csv else None

    factors_csv = Path(args.factors_csv) if args.factors_csv else None
    factors_json = Path(args.factors_json) if args.factors_json else None

    out_long = Path(args.out_long)
    out_wide = Path(args.out_wide)
    out_sda_long = Path(args.out_sda_long)
    out_sda_wide = Path(args.out_sda_wide)

    audit_dir = Path(args.audit_dir) if args.audit_dir else None

    print(f"[INFO] Canonical EUR: {canonical_eur if canonical_eur else '(will be built from components)'}")
    print(f"[INFO] Factors CSV:  {factors_csv}")
    print(f"[INFO] Factors JSON: {factors_json}")
    print(f"[INFO] OUT_LONG:     {out_long}")
    print(f"[INFO] OUT_WIDE:     {out_wide}")
    print(f"[INFO] OUT_SDA_LONG: {out_sda_long}")
    print(f"[INFO] OUT_SDA_WIDE: {out_sda_wide}")
    print(f"[INFO] AUDIT_DIR:    {audit_dir}")
    print(f"[INFO] set_pasture_equal_crop: {bool(args.set_pasture_equal_crop)}")

    # --- currency multiplier
    combined_multiplier = load_combined_multiplier(factors_csv=factors_csv, factors_json=factors_json)
    print(f"[OK] Loaded combined_multiplier (EUR2010->USD2019): {combined_multiplier}")

    # --- build/load canonical EUR in LONG form
    if canonical_eur:
        eur_long = load_canonical_eur_table(canonical_eur, audit_dir=audit_dir)
    else:
        if not (fractional_long_csv and maxdamage_long_csv and iso3_region_csv):
            raise ValueError(
                "Either provide --canonical-eur OR provide all of: "
                "--fractional-long-csv, --maxdamage-long-csv, --iso3-region-csv"
            )
        eur_long = build_canonical_from_components(
            fractional_long_csv=fractional_long_csv,
            maxdamage_long_csv=maxdamage_long_csv,
            iso3_region_csv=iso3_region_csv,
            audit_dir=audit_dir
        )

    # Standardize landtype labels to fraction-curve labels where possible
    eur_long["landtype"] = eur_long["landtype"].apply(normalize_landtype_label)

    # --- Convert to USD2019
    sector_usd = eur_long.copy()
    sector_usd["damage_per_m2"] = sector_usd["damage_per_m2"] * float(combined_multiplier)
    sector_usd["currency"] = "USD2019"

    # Clean types
    sector_usd["iso3"] = sector_usd["iso3"].astype(str).str.strip()
    sector_usd["depth_m"] = sector_usd["depth_m"].apply(to_float)
    sector_usd["damage_per_m2"] = sector_usd["damage_per_m2"].apply(to_float)

    # Drop invalid
    sector_usd = sector_usd.dropna(subset=["iso3", "landtype", "depth_m", "damage_per_m2"])

    # --- Sector outputs (keep all landtypes)
    sector_long = sector_usd[["iso3", "landtype", "depth_m", "damage_per_m2", "currency"]].copy()
    sector_long["landtype"] = sector_long["landtype"].apply(sector_label_from_landtype)

    out_long.parent.mkdir(parents=True, exist_ok=True)
    out_wide.parent.mkdir(parents=True, exist_ok=True)

    sector_long.to_csv(out_long, index=False)
    sector_wide = pivot_wide(sector_long, group_cols=["iso3", "landtype"], value_col="damage_per_m2")
    sector_wide.to_csv(out_wide, index=False)

    print(f"[OK] Wrote sector long depth-damage table → {out_long}  (rows={len(sector_long):,})")
    print(f"[OK] Wrote sector wide depth-damage table → {out_wide}  (rows={len(sector_wide):,})")

    # --- SDA aggregation
    # Map each sector landtype to an SDA type
    tmp = sector_usd.copy()
    tmp["sda_type"] = tmp["landtype"].apply(landtype_to_sda)

    unmapped = sorted(set(tmp["landtype"].dropna().unique()) - set(
        lt for lt in tmp["landtype"].dropna().unique() if landtype_to_sda(lt) is not None
    ))
    if len(unmapped) > 0:
        print(f"[INFO] Unmapped landtypes (kept in sector outputs, ignored in SDA aggregation): {len(unmapped)}")
        if audit_dir:
            audit_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"unmapped_landtype": unmapped}).to_csv(audit_dir / "diagnostics_unmapped_landtypes.csv", index=False)

    sda = tmp.dropna(subset=["sda_type"]).copy()
    sda_long = sda[["iso3", "sda_type", "depth_m", "damage_per_m2", "currency"]].copy()

    # Optional: create pasture equal to crop
    if args.set_pasture_equal_crop:
        crop = sda_long[sda_long["sda_type"] == "crop"].copy()
        pasture = crop.copy()
        pasture["sda_type"] = "pasture"
        sda_long = pd.concat([sda_long, pasture], ignore_index=True)

    out_sda_long.parent.mkdir(parents=True, exist_ok=True)
    out_sda_wide.parent.mkdir(parents=True, exist_ok=True)

    sda_long.to_csv(out_sda_long, index=False)
    sda_wide = pivot_wide(sda_long, group_cols=["iso3", "sda_type"], value_col="damage_per_m2")
    sda_wide.to_csv(out_sda_wide, index=False)

    print(f"[OK] Wrote SDA long depth-damage table → {out_sda_long}  (rows={len(sda_long):,})")
    print(f"[OK] Wrote SDA wide depth-damage table → {out_sda_wide}  (rows={len(sda_wide):,})")

    # --- QA preview
    print("\n[QA] Preview (sector long):")
    print(sector_long.head(12).to_string(index=False))

    print("\n[QA] Preview (sector wide):")
    print(sector_wide.head(6).to_string(index=False))

    print("\n[QA] Preview (SDA long):")
    print(sda_long.head(12).to_string(index=False))

    print("\n[QA] Preview (SDA wide):")
    print(sda_wide.head(6).to_string(index=False))

    print("\n[DONE] Use sector outputs for per-sector analysis, SDA outputs for Step4 lookup (USD2019/m²).")


if __name__ == "__main__":
    warnings.simplefilter("default")
    main()

# -*- coding: utf-8 -*-
"""The flood-control account: the equations, over arrays and frames.

Everything here takes arrays or frames and returns them. Nothing opens a file, reads a raster or
consults a module global set somewhere else, which is what lets the account's science be tested on
four countries and a handful of return periods rather than on a global grid. The file handling
lives in `flood_tasks`, and that separation is the reason `test_flood` needs no monkeypatched
readers: a test that has to replace a file reader is testing the wiring.

What the account rests on is small. The depth-damage lookup, the expected-damage integral and its
two boundary assumptions, the truncation that splits damage at a country's design standard, and
the country valuation. The rest of the module -- fifteen hundred lines of it -- is preparing
inputs, windowing rasters and writing results, and none of that changes an answer.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import warnings
from datetime import datetime
from typing import Dict, List, Optional

import geopandas as gpd
import hazelbean as hb
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import pandas as pd

from global_invest import utilities


INCA_DEPTH_BANDS = np.array([0.25, 0.5, 1.0, 1.5, 2.5, 3.5, 4.5, 5.5], dtype="float32")


def integrate_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """
    Trapezoidal integral, with the x axis sorted first.
    np.trapezoid weights each segment by diff(x). Handing it a descending or
    unordered x makes backwards segments contribute negatively, and the partial
    cancellation returns a small number of the wrong sign rather than an error.
    In this pipeline x is exceedance probability derived from return periods,
    which arrive in descending-p order naturally, so the sort is not optional.
    Both current call sites happen to sort beforehand -- one by argsort, one via
    a pandas groupby whose sorted-key behaviour is a default rather than a
    guarantee. Sorting here makes the property hold by construction. Sorting an
    already-sorted array is a no-op, so this cannot change existing results.
    """
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    if x.size != y.size:
        raise ValueError(f"integrate_trapezoid: length mismatch {x.size} vs {y.size}")
    o = np.argsort(x)
    x, y = x[o], y[o]
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _fmt_depth_col(d: float) -> str:
    return f"{int(d)}m" if abs(d - int(d)) < 1e-9 else f"{d}m"


def _band_depth_inca(depth_m: np.ndarray) -> np.ndarray:
    """Round each depth up to the next INCA band boundary; above 5.5 m, hold at 5.5."""
    idx = np.searchsorted(INCA_DEPTH_BANDS, depth_m, side="left")
    return INCA_DEPTH_BANDS[np.minimum(idx, INCA_DEPTH_BANDS.size - 1)]


def interp_damage_per_m2(depth_m: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                         mode: str = "interpolated") -> np.ndarray:
    """
    Damage per square metre at a given inundation depth.

    Args:
        depth_m: inundation depth, in metres.
        xs, ys: the tabulated depth-damage curve for one country and land class.
        mode: "interpolated" reads the curve as the continuous function its
            publication intends. "banded" reproduces the reference
            implementation, which rounds each depth up to the next of nine band
            boundaries -- a cell under 0.7 m pays the 1.0 m rate. Reported as a
            sensitivity only: rounding up raises the level of damage while
            suppressing the difference between the two counterfactual worlds,
            because amplification moves most cells by less than a band width.

    The mode is a parameter rather than a module global so that this function
    answers only to its arguments; the task layer passes whichever the run is
    configured for.
    """
    d = _band_depth_inca(depth_m) if mode == "banded" else depth_m
    return np.interp(np.clip(d, xs.min(), xs.max()), xs, ys).astype("float32")


def _attach_service_flow(rec_df: pd.DataFrame, iso3: str,
                         flow: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Join the SPA service-flow fraction onto the per-RP damage table and carry
    an attributed damage series alongside the gross one.

    Args:
        rec_df: the per-return-period damage table for one country.
        iso3: the country, used to select rows from `flow`.
        flow: Section C's service-flow summary, keyed by (iso3, rp). Passed in
            rather than read here, so this stays a join over frames: a function
            that opens its own inputs can only be tested by replacing the file
            reader, which tests the wiring rather than the join.

    When flood_apply_service_flow is on, this attributes residual
    damage to naturally-served floodplains. It is not avoided damage.
    """
    if flow is None:
        return rec_df

    sub = flow.loc[flow.iso3 == iso3, ["rp", "mean_spa_ratio_on_sda"]]
    if sub.empty:
        warnings.warn(f"[WARN] {iso3}: no service-flow rows; attributed damages set to NaN.")
        rec_df["service_flow_frac"] = np.nan
    else:
        rec_df = rec_df.merge(sub.rename(columns={"mean_spa_ratio_on_sda": "service_flow_frac"}),
                              on="rp", how="left")

    rec_df["damage_attributed_to_spa_usd2019"] = (
        rec_df["damage_total_usd2019"] * rec_df["service_flow_frac"])
    return rec_df


def _integrate_truncated(x: np.ndarray, y: np.ndarray, p_max: float) -> float:
    """
    Trapezoid integral of D(p) over p in [0, p_max], interpolating D at p_max.

    Used for the natural-capital-only (NC) share of EAD: defences prevent damage
    from events more frequent than their design standard, so only p <= 1/RP_prot
    is attributable to natural capital alone.
    """
    if not np.isfinite(p_max) or p_max <= 0:
        return 0.0
    order = np.argsort(x)
    x, y = x[order], y[order]
    if p_max >= x.max():
        return integrate_trapezoid(y, x)
    keep = x < p_max
    xt = np.append(x[keep], p_max)
    yt = np.append(y[keep], float(np.interp(p_max, x, y)))
    return integrate_trapezoid(yt, xt)


def compute_ead_from_points(rp: np.ndarray, dmg: np.ndarray, *,
                            add_p1_zero: bool = False, tail_mode: str = "flat",
                            enforce_monotone: bool = False,
                            protection_rp: Optional[float] = None):
    """
    EAD = integral of D(p) dp over p in [0, 1], where p = 1/RP.

    Boundary assumptions (conservative and documented):
      A) Frequent-event anchor at p=1: add (p=1, D=0). Step 4B applies a depth
         threshold and SDA masking, so frequent shallow inundation is treated
         as zero-damage in this global setup.
      B) Rare-event tail p -> 0: "flat" holds D constant from p=1/RPmax to 0;
         "zero" sets D=0 there. Flat avoids forcing the extreme tail to zero;
         its contribution is small because the tail width is 1/RPmax.
      C) Monotonicity: damages should rise with RP. Default is warn-only.
    """
    msgs: List[str] = []
    if tail_mode not in {"flat", "zero"}:
        raise ValueError("tail_mode must be one of: flat, zero")

    df = pd.DataFrame({"rp": rp, "damage": dmg})
    df["rp"] = df["rp"].apply(utilities.to_float)
    df["damage"] = df["damage"].apply(utilities.to_float)
    df = df.dropna(subset=["rp", "damage"])
    df = df[(df["rp"] > 0) & (df["damage"] >= 0)]

    if df.empty:
        return 0.0, pd.DataFrame(columns=["p", "rp", "damage", "note"]), ["no_valid_points"], np.nan

    df = df.groupby("rp", as_index=False)["damage"].mean().sort_values("rp").reset_index(drop=True)

    if enforce_monotone:
        df["damage"] = np.maximum.accumulate(df["damage"].to_numpy())

    df["p"] = 1.0 / df["rp"]
    pts = df[["p", "rp", "damage"]].copy()
    pts["note"] = ""

    if add_p1_zero:
        anchor = pd.DataFrame({"p": [1.0], "rp": [1.0], "damage": [0.0],
                               "note": ["anchor_p1_zero"]})
        pts = pd.concat([anchor, pts], ignore_index=True)

    rp_max = float(df["rp"].max())
    d_at_rpmax = float(df.loc[df["rp"] == rp_max, "damage"].iloc[0])
    tail = pd.DataFrame({"p": [0.0], "rp": [np.inf],
                         "damage": [d_at_rpmax if tail_mode == "flat" else 0.0],
                         "note": [f"tail_{tail_mode}_to_p0"]})
    pts = pd.concat([pts, tail], ignore_index=True)
    pts = pts.sort_values("p", ascending=False).reset_index(drop=True)

    p_series = pts["p"].to_numpy()
    d_series = pts["damage"].to_numpy()
    for i in range(len(pts) - 1):
        if p_series[i + 1] < p_series[i] and d_series[i + 1] + 1e-9 < d_series[i]:
            msgs.append("non_monotone_in_p_space")
            break

    tmp = (pd.DataFrame({"p": pts["p"].to_numpy()[::-1], "damage": pts["damage"].to_numpy()[::-1]})
           .groupby("p", as_index=False)["damage"].max())
    xs, ys = tmp["p"].to_numpy(), tmp["damage"].to_numpy()
    ead = integrate_trapezoid(ys, xs)

    # Natural-capital-only share: events rarer than the protection standard.
    ead_nc = np.nan
    if protection_rp is not None and np.isfinite(protection_rp):
        if protection_rp <= 0:
            # No defences: the entire EAD is attributable to natural capital
            # alone. FLOPROS carries MerL_Riv = 0 for 1,154 of its 4,650
            # polygons, so this is not an edge case -- treating it as NaN would
            # silently drop every unprotected country from the NC column, which
            # is exactly where the natural-capital share should be largest.
            ead_nc = ead
        else:
            ead_nc = _integrate_truncated(xs, ys, 1.0 / float(protection_rp))

    return ead, pts, msgs, ead_nc


# ---------------------------------------------------------------------------------------------
# Country valuation, in the column set every service publishes. Erosion multiplies a shock share
# by crop output; flood subtracts one damage integral from another and derives the share from the
# result. Opposite direction, same published columns -- which is what makes a combined table a
# concatenation rather than a reconciliation.
# ---------------------------------------------------------------------------------------------
MIN_GEP_FLOOR = 1.0  # USD/yr


def prevention_share(ead_current, ead_degraded):
    """
    The fraction of potential damage that ecosystems prevent.

    Args:
        ead_current: expected annual damage under current land cover.
        ead_degraded: expected annual damage in the counterfactual world.

    Returns:
        The share on [0, 1]. A country whose degraded world holds no damage has
        no share rather than a zero one: zero would say ecosystems prevent
        nothing there, which is a finding, where missing says there is nothing to
        take a share of, which is the truth, and it keeps the country out of a
        mean.
    """
    cur = np.asarray(ead_current, dtype="float64")
    deg = np.asarray(ead_degraded, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(deg > 0, (deg - cur) / deg, np.nan)
    # Degradation cannot reduce damage. A negative share is a routing or
    # alignment fault, not a result, and clipping silently would hide it -- so
    # this floors at zero and leaves the diagnostic to the caller.
    return np.clip(share, 0.0, 1.0)


def country_gep(df_ead, df_gdp, component):
    """
    The value of the damage ecosystems prevent, and what it is as a share of GDP.

    Args:
        df_ead (pandas.DataFrame): `iso3`, `ead_current_const2019_usd` and
            `ead_degraded_const2019_usd`, one row per country.
        df_gdp (pandas.DataFrame): `iso3` and `gdp_const2019_2019`.
        component (str): which counterfactual this is -- `bare` or `insitu` --
            carried onto every row so the two runs can be concatenated and still
            told apart, as erosion does for its three prevention channels.

    Returns:
        pandas.DataFrame with the flood quantity columns followed by the four
        every service shares: `gdp_const2019_2019`, `gep_const2019_usd`,
        `gdp_loss_pct`, and `component` at the front.

    Note that GEP here is a difference rather than a product. Erosion multiplies
    a shock share by crop output; flood subtracts one expected-damage integral
    from another, and the share is derived from the result rather than producing
    it. The published columns are the same either way, which is the point.
    """
    out = df_ead.merge(df_gdp, on="iso3", how="left").copy()

    out["gep_const2019_usd"] = (out["ead_degraded_const2019_usd"].fillna(0.0)
                                - out["ead_current_const2019_usd"].fillna(0.0))
    out.loc[out["gep_const2019_usd"] < MIN_GEP_FLOOR, "gep_const2019_usd"] = 0.0

    out["flood_prevention_share"] = prevention_share(
        out["ead_current_const2019_usd"], out["ead_degraded_const2019_usd"])

    # A country with no GDP figure gets no percentage rather than an infinite
    # one, and its value stays visible so the gap reads as a missing denominator
    # rather than as a country where flood regulation does not matter.
    out["gdp_loss_pct"] = np.where(
        out["gdp_const2019_2019"].notna() & (out["gdp_const2019_2019"] > 0),
        100.0 * out["gep_const2019_usd"] / out["gdp_const2019_2019"],
        np.nan)

    out["component"] = component
    return out[["component", "iso3",
                "ead_current_const2019_usd", "ead_degraded_const2019_usd",
                "flood_prevention_share",
                "gdp_const2019_2019", "gep_const2019_usd", "gdp_loss_pct"]]


def combine_components(*frames):
    """
    Stack the counterfactual runs into one table.

    Concatenation rather than a merge, because the two are alternative answers to
    different questions rather than components of one total. Summing bare-soil
    and in-situ would double-count the same prevented damage under two
    assumptions about how much land cover is lost.
    """
    return pd.concat(frames, ignore_index=True).sort_values(
        ["component", "iso3"]).reset_index(drop=True)


# =============================================================================
# Grid, geometry and integration.
# =============================================================================


# -----------------------------------------------------------------------------
# Raster metadata / geometry
# -----------------------------------------------------------------------------
def pixel_area_m2(transform) -> float:
    return abs(float(transform.a) * float(transform.e))


def pixel_area_km2(transform) -> float:
    return pixel_area_m2(transform) / 1e6


def mercator_area_scale(transform, row_off: int, height: int) -> np.ndarray:
    """
    Areal scale factor for EPSG:3857, as a column vector of shape (height, 1).

        true_ground_area = nominal_pixel_area * cos^2(latitude)

    Web Mercator is conformal, not equal-area: it preserves shape and inflates
    area by 1/cos^2(lat), exactly. Reading pixel area off the affine transform
    therefore gives the value at the EQUATOR ONLY, and applying it globally
    overstates area everywhere else:

        latitude    overstatement
           0            1.00x
          45            2.00x
          60            4.00x
          65            5.60x
          70            8.55x

    Every country outside the tropics is affected, progressively worse toward
    the poles. This is analytic, not an approximation -- cos^2(lat) IS the areal
    scale factor for Web Mercator -- so correcting per pixel gives true ground
    area without reprojecting anything.

    Area depends only on latitude, hence only on raster ROW, so one value per
    row broadcasts across the whole tile.
    """
    R = 6378137.0                      # WGS84 semi-major axis, EPSG:3857 sphere
    rows = np.arange(row_off, row_off + height, dtype="float64") + 0.5
    y = float(transform.f) + rows * float(transform.e)
    lat = 2.0 * np.arctan(np.exp(y / R)) - np.pi / 2.0
    return (np.cos(lat) ** 2).astype("float32").reshape(-1, 1)


# -----------------------------------------------------------------------------
# Admin0 / ISO3 handling
# -----------------------------------------------------------------------------
def pick_iso3_column(gdf: gpd.GeoDataFrame) -> Optional[str]:
    candidates = ["iso3", "ISO3", "iso_a3", "ISO_A3", "ADM0_A3", "adm0_a3", "iso3_r250_label"]
    for c in candidates:
        if c in gdf.columns:
            return c
    return None


def pick_name_column(gdf: gpd.GeoDataFrame) -> Optional[str]:
    candidates = [
        "country_name", "NAME_EN", "ADMIN", "NAME_LONG", "NAME",
        "COUNTRY", "NAME_0", "ADM0_NAME", "GEOUNIT", "iso3_r250_name",
    ]
    for c in candidates:
        if c in gdf.columns:
            return c
    return None




def integrate_trapezoid(y: np.ndarray, x: np.ndarray) -> float:

    """

    Trapezoidal integral, with the x axis sorted first.



    np.trapezoid weights each segment by diff(x). Handing it a descending or

    unordered x makes backwards segments contribute negatively, and the partial

    cancellation returns a small number of the wrong sign rather than an error.

    In this pipeline x is exceedance probability derived from return periods,

    which arrive in descending-p order naturally, so the sort is not optional.



    Both current call sites happen to sort beforehand -- one by argsort, one via

    a pandas groupby whose sorted-key behaviour is a default rather than a

    guarantee. Sorting here makes the property hold by construction. Sorting an

    already-sorted array is a no-op, so this cannot change existing results.

    """

    x = np.asarray(x, dtype="float64")

    y = np.asarray(y, dtype="float64")

    if x.size != y.size:

        raise ValueError(f"integrate_trapezoid: length mismatch {x.size} vs {y.size}")

    o = np.argsort(x)

    x, y = x[o], y[o]

    if hasattr(np, "trapezoid"):

        return float(np.trapezoid(y, x))

    return float(np.trapz(y, x))




# =============================================================================
# Depth-damage table shaping: label normalisation, the depth-column parser and
# the long-to-wide pivot. Section 4A's arithmetic, with no IO in it.
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
SDA_CODE_VERSION = "2025-12-15_sda_step2_smartskip_v2_depth_inputs"


# Section B: the run signature that decides whether a country can be skipped,
# the depth-raster discovery and the return-period map.
# =============================================================================


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


# -----------------------------------------------------------------------------#
# Depth RP map builders
# -----------------------------------------------------------------------------#








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

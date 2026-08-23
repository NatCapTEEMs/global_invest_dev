"""Erosion-control ES science helpers (sediment-retention shock).

STATIC helper (read_erosion_dependency): parse the frozen per-scenario dependency table
(raw_dependencies/erosion_prevention_dependency.csv). Scenario-name resolution is shared across services
in global_invest.utilities.resolve_raw_scenario. DYNAMIC helpers (#26): the SPAM->coefficient
crosswalk (load_erosion_yield_coefficients, get_erosion_yield_coefficient, SPAM_ALIAS_MAP) and the per-country
severe-threshold policy (build_severe_threshold_raster) used by the dynamic exposure and shock tasks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import hazelbean as hb
import requests

from global_invest.erosion import erosion_chain
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch


def read_erosion_dependency(ero_path):
    """Load + normalize the erosion dependency table; return the df.

    Base extraction happens in the CALLER after resolving the configured base scenario through
    utilities.resolve_base_scenario (this function previously hardcoded 'baseline_ignore_damages'
    as the base, silently ignoring p.es_shock_base_scenario -- right only by spelling coincidence).
    """
    df = pd.read_csv(ero_path)
    df['scenario'] = df['scenario'].str.replace('_2050', '').str.replace('2023.0', 'baseline_2023')
    return df


# ---------------------------------------------------------------------------
# DYNAMIC helpers (#26): SPAM crop -> erosion-to-yield coefficient crosswalk.
# The 4-letter SPAM band codes are aliased to the coefficient table's crop names.
# ---------------------------------------------------------------------------
# SPAM2020 crop code -> candidate keys in the crop-coefficient table, tried in order.
# The FIRST alias of each entry is the EXACT FAO item name as it appears in
# elasticity_crops_fao_revised.csv; the looser stems after it are kept as fallbacks for other tables.
# This matters: the lookup is exact-match, so stems alone ("maize") never hit the FAO names
# ("Maize (corn)") and every crop silently took the 0.08 fallback -- which is the table MINIMUM, not
# its average (mean 0.163), so the miss biased the erosion shock low across the board.
# Six SPAM codes have NO counterpart in the table and correctly keep the default: grou (groundnut),
# ocer, orts, pige, vege and rest -- the n.e.c. aggregates FAO does not carry.
SPAM_ALIAS_MAP = {
    "whea": ["wheat"], "rice": ["rice"], "maiz": ["maize (corn)", "maize", "corn"],
    "barl": ["barley"], "sorg": ["sorghum"], "mill": ["millet", "small millet"],
    "pmil": ["millet", "pearl millet"], "pota": ["potatoes", "potato"],
    "cass": ["cassava, fresh", "cassava"], "soyb": ["soya beans", "soybean", "soy"],
    "grou": ["groundnut", "peanut"], "cott": ["seed cotton, unginned", "cotton"],
    "sugc": ["sugar cane", "sugarcane"], "bana": ["bananas", "banana"],
    "plnt": ["plantains and cooking bananas", "plantain"], "coco": ["cocoa beans", "cocoa"],
    "coff": ["coffee, green", "arabica coffee", "coffee"], "rcof": ["coffee, green", "robusta coffee"],
    "teas": ["tea leaves", "tea"], "toba": ["unmanufactured tobacco", "tobacco"],
    "toma": ["tomatoes", "tomato"],
    "onio": ["onions and shallots, dry (excluding dehydrated)", "onion"],
    "vege": ["vegetable", "other vegetables"], "sunf": ["sunflower seed", "sunflower"],
    "rape": ["rape or colza seed", "rapeseed", "canola"], "sesa": ["sesame seed", "sesame"],
    "citr": ["oranges", "citrus"], "lent": ["lentils, dry", "lentil"],
    "bean": ["beans, dry", "bean"], "chic": ["chick peas, dry", "chickpea"],
    "cowp": ["cow peas, dry", "cowpea"], "pige": ["peas, dry", "pigeon pea"], "yams": ["yams"],
    "swpo": ["sweet potatoes", "sweet potato"], "sugb": ["sugar beet", "sugarbeet"],
    "oilp": ["oil palm fruit", "oilpalm", "oil palm"], "cnut": ["coconuts, in shell", "coconut"],
    "ocer": ["other cereals"], "orts": ["other roots"],
    "opul": ["other pulses n.e.c.", "other pulses"], "ooil": ["castor oil seeds", "other oil crops"],
    "ofib": ["agave fibres, raw, n.e.c.", "other fibre crops"],
    "rubb": ["natural rubber in primary forms", "rubber"],
    "trof": ["other tropical fruits, n.e.c.", "other tropical fruit"],
    "temf": ["apples", "temperate fruit"], "rest": ["rest of crops"],
}


def load_erosion_yield_coefficients(elasticity_csv):
    """Return {crop_key (lowercased) -> erosion-to-yield coefficient in [0,1]} from the coefficient CSV.

    Accepts a crop-name column among crop/monfreda_crop/item/item_name plus an 'elasticity' column.
    """
    df = pd.read_csv(elasticity_csv, encoding='utf-8-sig')
    df.columns = [str(c).strip().lower() for c in df.columns]
    crop_col = next((c for c in ('crop', 'monfreda_crop', 'item', 'item_name') if c in df.columns), None)
    if crop_col is None or 'elasticity' not in df.columns:
        return {}
    df['elasticity'] = pd.to_numeric(df['elasticity'], errors='coerce').clip(0.0, 1.0)
    key = df[crop_col].astype(str).str.strip().str.lower()
    keep = key.ne('') & df['elasticity'].notna()
    df = df[keep].assign(__k=key[keep]).drop_duplicates('__k', keep='last')
    return dict(zip(df['__k'], df['elasticity']))


def get_erosion_yield_coefficient(crop_key, coef_map, fallback=0.08):
    """crop_key -> erosion-to-yield coefficient: direct hit, else SPAM alias, else the flat fallback."""
    k = str(crop_key).strip().lower()
    v = coef_map.get(k, np.nan)
    if np.isfinite(v):
        return float(np.clip(v, 0.0, 1.0))
    for alias in SPAM_ALIAS_MAP.get(k, []):
        v2 = coef_map.get(str(alias).strip().lower(), np.nan)
        if np.isfinite(v2):
            return float(np.clip(v2, 0.0, 1.0))
    return float(np.clip(fallback, 0.0, 1.0))


def build_seals7_biophysical_table(src_csv, out_csv):
    """Re-key a biophysical table from ESA lucodes onto SEALS7 classes, for InVEST SDR.

    SDR matches the table's `lucode` against the LULC raster's values, but the shipped table is keyed on
    ESA-CCI codes while our maps are SEALS7 (1-7), so SDR would match nothing. The table already carries
    a `seals_lucode` column, and usle_c/usle_p are CONSTANT within each SEALS class (verified: min == max
    for all 7), so the collapse is unambiguous -- no area weighting to choose. Returns the written path.
    """
    df = pd.read_csv(src_csv)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if 'seals_lucode' not in df.columns:
        raise ValueError('%s has no seals_lucode column, so it cannot be re-keyed onto SEALS7 classes; '
                         'supply an already-SEALS-keyed table via p.erosion_biophysical_table_path.'
                         % src_csv)
    df = df.dropna(subset=['seals_lucode'])
    out = (df.groupby(df['seals_lucode'].astype(int))[['usle_c', 'usle_p']].mean()
             .reset_index().rename(columns={'seals_lucode': 'lucode'}))
    out['description'] = ['seals7_class_%d' % c for c in out['lucode']]
    out.to_csv(out_csv, index=False)
    return out_csv


def repair_watersheds(src_path, out_path):
    """Repair self-intersecting watershed rings so InVEST SDR can finish.

    SDR's last step (_generate_report) unions the watershed polygons to test for overlap, and GEOS
    RAISES TopologyException on an invalid ring rather than warning -- so a single bad geometry kills a
    run whose rasters are already computed. HydroBASINS reprojected to an equal-area CRS carries ring
    self-intersections (1192 of 16397 in hybas_global_lev06_v1c), which is why this never showed up on a
    clipped AOI: the small subset happened to exclude them.

    make_valid (not buffer(0), which can silently drop slivers) clears all of them, leaves the union
    computable, and preserves total area. Returns out_path.
    """
    import geopandas as gpd

    gdf = gpd.read_file(src_path, engine='pyogrio')
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, 'geometry'] = gdf.loc[invalid, 'geometry'].make_valid()
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf.to_file(out_path, driver='GPKG')
    print('  erosion watersheds: repaired %d of %d invalid geometries -> %s'
          % (int(invalid.sum()), len(gdf), out_path))
    return out_path


def build_severe_threshold_raster(grid_da, country_boundary_path, dem_path=None,
                                  thresh_high=11.0, thresh_low=2.0,
                                  small_area_km2=50_000, low_elevation_mean_m=250,
                                  mask_below_sea=True, max_valid_elevation_m=9000.0):
    """Per-pixel soil-loss-tolerance threshold aligned to grid_da (the SES-11 policy).

    T = thresh_low for a country that is small-area (geometry area < small_area_km2) OR low-elevation
    (mean DEM elevation < low_elevation_mean_m); else thresh_high. grid_da must be on an equal-area
    grid (its CRS units are used for the km2 area). If dem_path is None the elevation rule is skipped.
    Returns a float32 array of shape grid_da.shape.
    """
    import geopandas as gpd
    import rioxarray as rxr
    from rasterio.features import rasterize
    from rasterio.enums import Resampling

    gdf = gpd.read_file(country_boundary_path, engine='pyogrio').to_crs(grid_da.rio.crs)
    gdf = gdf[gdf.geometry.notnull()].reset_index(drop=True)
    gdf['iid'] = range(1, len(gdf) + 1)
    shape, transform = grid_da.shape, grid_da.rio.transform()
    iso_id = rasterize([(g, int(i)) for g, i in zip(gdf.geometry, gdf['iid'])],
                       out_shape=shape, transform=transform, fill=0, dtype='int32')
    max_id = int(gdf['iid'].max())

    area_km2 = gdf.set_index('iid').geometry.area / 1e6      # equal-area CRS -> m2 -> km2
    iso_low = set(int(i) for i in area_km2[area_km2 < small_area_km2].index)

    if dem_path:
        dem = rxr.open_rasterio(dem_path, masked=True).squeeze().rio.reproject_match(
            grid_da, resampling=Resampling.bilinear)
        v = dem.values.astype('float64')
        if mask_below_sea:
            v[v < 0.0] = np.nan
        v[v > max_valid_elevation_m] = np.nan
        m = np.isfinite(v) & (iso_id > 0)
        s = np.bincount(iso_id[m], weights=v[m], minlength=max_id + 1).astype('float64')
        c = np.bincount(iso_id[m], minlength=max_id + 1).astype('float64')
        with np.errstate(invalid='ignore'):
            mean_elev = np.where(c > 0, s / c, np.nan)
        iso_low |= set(int(i) for i in range(1, max_id + 1)
                       if np.isfinite(mean_elev[i]) and mean_elev[i] < low_elevation_mean_m)

    thr = np.full(shape, float(thresh_high), dtype='float32')
    if iso_low:
        thr[np.isin(iso_id, np.fromiter(iso_low, dtype='int32'))] = float(thresh_low)
    return thr


# #############################################################################
# GEP VALUATION (folded from global_erosion_gep, the prevention-share erosion
# valuation this module's dynamic shock already cites as its method source).
# Straight append per the fold-in recipe in erosion_initialize.py, with exactly
# these changes, each import-hygiene or a supersede:
#   - from __future__ import annotations moved to the top of this file;
#   - module-level natcap imports made local to run_invest_sdr;
#   - the import-time env/logging block wrapped in _setup_sdr_environment()
#     (root logging config + gdal.UseExceptions no longer run at import);
#   - the two module-level output-dir mkdirs deleted (configure_* recreate
#     them at run time);
#   - the duplicate SPAM_ALIAS_MAP dropped in favour of the corrected map above.
# Heavy module-level path constants remain: configure_sdr/configure_prevention_
# shares/configure_maps override them from p at run time.
# #############################################################################

# =============================================================================
# erosion_functions.py
#
# Key science functions for the GEP erosion / sediment retention service.
# Follows the same run_x.py / x_tasks.py / x_functions.py / x_initialize.py
# structure Justin described (2026-07-07 email) for the terrestrial_carbon
# module. Three original scripts are represented here as three sections:
#
#   A) InVEST SDR run              (from step1_sdr_invest_run.ipynb)
#   B) On-farm + upstream erosion   (from Combine_PS_SES11_3_3_2026.ipynb)
#      prevention-share GEP valuation
#   C) Maps & figures               (from combined_maps_figures_SES.ipynb)
#
# Generic raster/plotting helpers reused across all three (and likely
# duplicated in other people's GEP code, per Justin's email) live in
# the generic-helpers section below.
#
# DESIGN NOTE ON CONFIG: the original notebooks hardcoded all paths/knobs
# as module-level constants (each script had its own ROOT). To keep the
# original, working science logic byte-for-byte intact (rather than risk
# introducing bugs while threading a config object through ~40 nested
# references), each section keeps its constants as module-level globals
# with the *same defaults the original notebooks used*, and exposes a
# `configure_*(p)` function that a task calls first to override any of
# them from the ProjectFlow object `p`. This mirrors how run_erosion.py
# sets p.erosion_* attributes, the same pattern used in run_terrestrial_carbon.py.
# =============================================================================

import os
import sys
import time
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.features
import rioxarray as rxr
import xarray as xr
from rasterio.enums import Resampling
from rasterio.crs import CRS as rioCRS

# =============================================================================
# Generic helpers extracted from the source notebooks (formerly erosion_utils.py; folded in to
# keep the five-file structure -- Chiara's call). Each is a promotion candidate for
# global_invest.utilities / hazelbean on its SECOND caller, or dies when the hazelbean-first
# pass replaces it with an existing hb equivalent. This section should only shrink.
# =============================================================================
_HTTP_TIMEOUT = 60
_RETRY = 4

# Defaults for the plotting helpers below (overridden at run time by
# erosion_functions.configure_maps(p) if the project sets different values).
EXCLUDE_ISO3 = {"ATA"}
ROBINSON_CRS = "+proj=robin"
USD_TO_MILLIONS = 1e6
TOP_N = 20


def assert_exists(p: Path, hint: str = ""):
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}\n{hint}")



def _normcols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def _http_get(url, params=None, headers=None, stream=False):
    last_err = None
    for attempt in range(_RETRY):
        try:
            r = requests.get(url, params=params, headers=headers,
                             timeout=_HTTP_TIMEOUT, stream=stream)
            if r.status_code == 200:
                return r
            last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            last_err = e
        time.sleep(1 + attempt)
    raise last_err

def open_raster_1band(path: Path) -> xr.DataArray:
    """Open a single-band raster as a 2D DataArray (masked)."""
    return rxr.open_rasterio(path, masked=True).squeeze()

def _ensure_crs(da: xr.DataArray, name: str) -> xr.DataArray:
    if da.rio.crs is None:
        raise ValueError(f"{name} raster has no CRS. Fix metadata before running.")
    return da

def reproject_to_analysis_grid(da: xr.DataArray, analysis_crs: rioCRS, resampling: Resampling) -> xr.DataArray:
    """Reproject to equal-area CRS if needed (does NOT match a template grid)."""
    _ensure_crs(da, "input")
    if da.rio.crs == analysis_crs:
        return da
    return da.rio.reproject(analysis_crs, resampling=resampling)

def pixel_area_hectares(da: xr.DataArray) -> float:
    """Pixel area (ha) in a projected CRS."""
    if da.rio.crs is None or (not da.rio.crs.is_projected):
        raise ValueError("pixel_area_hectares requires a projected CRS.")
    res_x, res_y = map(abs, da.rio.resolution())
    return (res_x * res_y) / 10_000.0

def _clean_nonneg(da: xr.DataArray) -> xr.DataArray:
    """Convert negative to 0, keep NaNs as NaN."""
    out = da.copy()
    vals = out.values
    vals = np.where(np.isfinite(vals), np.maximum(vals, 0.0), np.nan)
    out.values = vals
    return out

def _clip01_arr(arr: np.ndarray) -> np.ndarray:
    out = arr.astype("float32", copy=False)
    out = np.where(np.isfinite(out), np.clip(out, 0.0, 1.0), np.nan).astype("float32")
    return out

def _write_share(path: Path, template: xr.DataArray, arr01: np.ndarray):
    """Write a float32 share raster (0–1) aligned to template."""
    da = xr.DataArray(arr01.astype("float32"), coords=template.coords, dims=template.dims)
    da = da.rio.write_crs(template.rio.crs, inplace=False)
    da = da.rio.write_transform(template.rio.transform(), inplace=False)
    da.rio.to_raster(path, compress="deflate", nodata=np.float32(-9999))

def _bincount_weighted_mean(ids: np.ndarray, x: np.ndarray, max_id: int) -> np.ndarray:
    """Compute mean(x) by integer id (1..max_id). ids and x must be 1D aligned."""
    ok = np.isfinite(x) & (ids > 0)
    if not np.any(ok):
        return np.full(max_id + 1, np.nan, dtype="float64")
    ids_ok = ids[ok].astype("int32", copy=False)
    x_ok   = x[ok].astype("float64", copy=False)
    s = np.bincount(ids_ok, weights=x_ok, minlength=max_id + 1).astype("float64")
    c = np.bincount(ids_ok, minlength=max_id + 1).astype("float64")
    return np.divide(s, c, out=np.full_like(s, np.nan), where=c > 0)


def _write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def savefig(path: Path, dpi: int = 300):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def top_n(df: pd.DataFrame, col: str, n: int = TOP_N) -> pd.DataFrame:
    d = df[np.isfinite(df[col])].copy()
    return d.sort_values(col, ascending=False).head(n)


def pick_iso3_column(gdf: gpd.GeoDataFrame) -> str | None:
    candidates = ["iso3", "ISO3", "iso_a3", "ADM0_A3", "adm0_a3", "ISO_A3", "iso3_r250_label"]
    for c in candidates:
        if c in gdf.columns:
            return c
    return None


def pick_name_column(gdf: gpd.GeoDataFrame) -> str | None:
    candidates = [
        "country_name", "NAME_EN", "ADMIN", "NAME_LONG", "NAME",
        "COUNTRY", "NAME_0", "ADM0_NAME", "GEOUNIT", "iso3_r250_name"
    ]
    for c in candidates:
        if c in gdf.columns:
            return c
    return None


def fmt_usd_millions(x: float) -> str:
    if not np.isfinite(x):
        return "NA"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 100:
        return f"{x:,.0f}"
    if abs(x) >= 10:
        return f"{x:,.1f}"
    if abs(x) >= 1:
        return f"{x:,.1f}"
    return f"{x:,.2f}"


def fmt_percent(x: float) -> str:
    if not np.isfinite(x):
        return "NA"
    if abs(x) >= 10:
        return f"{x:.1f}"
    if abs(x) >= 1:
        return f"{x:.2f}"
    return f"{x:.3f}"


def fmt_usd(x: float) -> str:
    if not np.isfinite(x):
        return "NA"
    return f"${x:,.0f}"


def build_interval_labels(edges: np.ndarray, label_format: str = "usd_millions") -> list[str]:
    labels = []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if label_format == "usd_millions":
            lo_txt = fmt_usd_millions(lo)
            hi_txt = fmt_usd_millions(hi)
        else:
            lo_txt = fmt_percent(lo)
            hi_txt = fmt_percent(hi)
        labels.append(f"{lo_txt} – {hi_txt}")
    return labels


def compute_classification(values: pd.Series, scheme: str = "fisher_jenks", k: int = 5):
    s = pd.to_numeric(values, errors="coerce")
    m = np.isfinite(s)
    clean = s[m]

    if clean.empty:
        return pd.Series(index=values.index, dtype="float64"), np.array([0.0, 1.0])

    try:
        import mapclassify

        scheme = (scheme or "fisher_jenks").lower()
        k_eff = min(k, int(clean.nunique()))
        k_eff = max(k_eff, 1)

        if scheme == "fisher_jenks":
            classifier = mapclassify.FisherJenks(clean.to_numpy(), k=k_eff)
        elif scheme == "equal_interval":
            classifier = mapclassify.EqualInterval(clean.to_numpy(), k=k_eff)
        elif scheme == "quantiles":
            classifier = mapclassify.Quantiles(clean.to_numpy(), k=k_eff)
        else:
            classifier = mapclassify.FisherJenks(clean.to_numpy(), k=k_eff)

        edges = np.concatenate(([clean.min()], np.asarray(classifier.bins, dtype=float)))
        class_ids = pd.Series(np.nan, index=values.index)
        class_ids.loc[m] = classifier.yb
        return class_ids, edges

    except Exception:
        warnings.warn("mapclassify unavailable or failed; falling back to qcut quantiles.")
        q = min(k, max(1, int(clean.nunique())))
        cats = pd.qcut(clean, q=q, duplicates="drop")
        codes = pd.Series(np.nan, index=values.index)
        codes.loc[m] = cats.cat.codes.astype(float)

        intervals = cats.cat.categories
        edges = [intervals[0].left]
        for iv in intervals:
            edges.append(iv.right)
        return codes, np.asarray(edges, dtype=float)



def plot_raster_global(tif_path: Path, title: str, out_png: Path, downsample_factor: int = 6):
    assert_exists(tif_path)
    da = rxr.open_rasterio(tif_path, masked=True).squeeze()

    if downsample_factor and downsample_factor > 1:
        da = da.isel(
            y=slice(None, None, downsample_factor),
            x=slice(None, None, downsample_factor),
        )

    arr = da.values.astype("float32", copy=False)
    arr = np.where(np.isfinite(arr), arr, np.nan)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(arr, interpolation="nearest")
    ax.set_title(title, fontsize=16, pad=12)
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Share (0–1)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    savefig(out_png, dpi=300)


def plot_publication_choropleth_categorical(
    world_joined: gpd.GeoDataFrame,
    value_col: str,
    title: str,
    out_png: Path,
    legend_title: str,
    scheme: str = "fisher_jenks",
    k: int = 5,
    value_unit: str = "raw",
    label_format: str = "usd_millions",
    legend_loc: str = "lower left",
):
    gdf = world_joined.copy()

    if "iso3" in gdf.columns:
        gdf = gdf[~gdf["iso3"].isin(EXCLUDE_ISO3)].copy()

    gdf = gdf[gdf.geometry.notna()].copy()

    if value_col not in gdf.columns:
        warnings.warn(f"Column not found for map: {value_col}")
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_axis_off()
        ax.set_title(f"{title}\n[missing column: {value_col}]", fontsize=16, pad=14)
        savefig(out_png, dpi=300)
        return

    if value_unit == "usd_millions":
        gdf["_plot_value"] = pd.to_numeric(gdf[value_col], errors="coerce") / USD_TO_MILLIONS
    else:
        gdf["_plot_value"] = pd.to_numeric(gdf[value_col], errors="coerce")

    try:
        gdf = gdf.to_crs(ROBINSON_CRS)
    except Exception as e:
        warnings.warn(f"CRS transform failed ({e}). Plotting in native CRS.")

    minx, miny, maxx, maxy = gdf.total_bounds
    class_ids, edges = compute_classification(gdf["_plot_value"], scheme=scheme, k=k)

    valid_codes = pd.Series(class_ids).dropna()
    if valid_codes.empty:
        warnings.warn(f"No valid data for map: {value_col}")
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_axis_off()
        ax.set_title(title, fontsize=16, pad=14)
        savefig(out_png, dpi=300)
        return

    n_classes = int(valid_codes.max()) + 1
    labels = build_interval_labels(edges[:n_classes + 1], label_format=label_format)

    gdf["_class_id"] = pd.Series(class_ids, index=gdf.index)
    gdf["_class_label"] = pd.Categorical(
        [labels[int(x)] if np.isfinite(x) and int(x) < len(labels) else np.nan for x in gdf["_class_id"]],
        categories=labels,
        ordered=True,
    )

    cmap = mpl.cm.get_cmap(mpl.rcParams["image.cmap"], n_classes)
    color_list = [mpl.colors.to_hex(cmap(i)) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_axis_off()

    gdf.plot(
        column="_class_label",
        ax=ax,
        cmap=mpl.colors.ListedColormap(color_list),
        legend=False,
        linewidth=0.35,
        edgecolor="white",
        missing_kwds={"color": "lightgrey", "edgecolor": "white"},
    )

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_title(title, fontsize=16, pad=14)

    handles = [Patch(facecolor=color_list[i], edgecolor="none", label=labels[i]) for i in range(n_classes)]
    handles.append(Patch(facecolor="lightgrey", edgecolor="none", label="No data"))

    leg = ax.legend(
        handles=handles,
        title=legend_title,
        loc=legend_loc,
        frameon=True,
        fontsize=10,
        title_fontsize=11,
        borderpad=0.8,
        labelspacing=0.5,
        handlelength=1.6,
        handletextpad=0.6,
    )
    leg.get_frame().set_alpha(0.95)

    savefig(out_png, dpi=300)


# SDR-specific imports (Section A)
from osgeo import gdal  # ensures GDAL sees env vars early


def configure_sdr(p):
    """
    Override the Section-A (InVEST SDR) module-level path constants from the
    ProjectFlow object, if the project set them. Called by
    erosion_tasks.invest_sdr() before run_invest_sdr().
    """
    global ROOT, BASE_IN, BASE_OUT, BIOPHYS_CSV, DEM_TIF, LULC_TIF, K_TIF, R_TIF
    global WATERSHEDS_RAW, WATERSHEDS_SANITIZED, WATERSHEDS_SAN_LAYER, DRAINAGE_PATH

    ROOT = Path(getattr(p, 'erosion_sdr_root', ROOT))
    BASE_IN = Path(getattr(p, 'erosion_sdr_input_dir', BASE_IN))
    BASE_OUT = Path(getattr(p, 'erosion_sdr_output_dir', BASE_OUT))
    BASE_OUT.mkdir(parents=True, exist_ok=True)

    BIOPHYS_CSV = Path(getattr(p, 'erosion_biophysical_table_path', BIOPHYS_CSV))
    DEM_TIF = Path(getattr(p, 'erosion_dem_path', DEM_TIF))
    LULC_TIF = Path(getattr(p, 'erosion_lulc_path', LULC_TIF))
    K_TIF = Path(getattr(p, 'erosion_erodibility_path', K_TIF))
    R_TIF = Path(getattr(p, 'erosion_erosivity_path', R_TIF))
    WATERSHEDS_RAW = Path(getattr(p, 'erosion_watersheds_path', WATERSHEDS_RAW))
    WATERSHEDS_SANITIZED = Path(getattr(p, 'erosion_watersheds_sanitized_path', WATERSHEDS_SANITIZED))
    WATERSHEDS_SAN_LAYER = getattr(p, 'erosion_watersheds_sanitized_layer', WATERSHEDS_SAN_LAYER)
    DRAINAGE_PATH = getattr(p, 'erosion_drainage_path', DRAINAGE_PATH)


# =============================================================================
# SECTION A — InVEST SDR run (from step1_sdr_invest_run.ipynb)
# =============================================================================
# 0) PATH CONFIG (CANONICAL)
# =============================================================================
ROOT = Path("/users/3/damph002/GEP/sediment")

BASE_IN  = ROOT / "inputs_v2" / "sdr"
BASE_OUT = ROOT / "outputs_v2" / "sdr" / "invest_sdr_2019"

BIOPHYS_CSV = BASE_IN / "biophysical_esa_sdr_InVEST_Global_GEP_Oct_27_25_nkd.csv"
DEM_TIF     = BASE_IN / "global_dem_reproj.tif"
LULC_TIF    = BASE_IN / "lulc_esa_2019_int_reproj.tif"
K_TIF       = BASE_IN / "erodibility_30s_reproj.tif"
R_TIF       = BASE_IN / "erosivity_30s_reproj.tif"

# ✅ Use MERGED watershed ONLY
WATERSHEDS_RAW = BASE_IN / "wshed_global_reproj_CLEAN_MERGED.gpkg"

# Sanitized watersheds used to prevent overflow + CRS parse failures in report
WATERSHEDS_SANITIZED = BASE_IN / "wshed_for_sdr_report_sanitized.gpkg"
WATERSHEDS_SAN_LAYER = "watersheds"

# Optional drainage raster (leave empty to disable)
DRAINAGE_PATH = ""


# =============================================================================
# 1) ENV HARDENING FOR PROJ/GDAL (FIXES CRS PARSE FAILURES)
# =============================================================================
def _set_proj_gdal_env():
    """
    Fixes errors like:
      "PROJ: proj_create: no database context specified"
      "Cannot parse CRS http://www.opengis.net/def/crs/EPSG/0/2193"

    by ensuring PROJ_LIB and GDAL_DATA point to the *active* conda env.
    """
    prefix = Path(sys.prefix)
    proj_lib = prefix / "share" / "proj"
    gdal_data = prefix / "share" / "gdal"

    if proj_lib.exists():
        os.environ["PROJ_LIB"] = str(proj_lib)
    if gdal_data.exists():
        os.environ["GDAL_DATA"] = str(gdal_data)

    # Prefer official EPSG parameters vs GeoTIFF geokeys when there is mismatch
    os.environ.setdefault("GTIFF_SRS_SOURCE", "EPSG")

    # Cluster-safe: avoid network calls for grids/definitions
    os.environ.setdefault("PROJ_NETWORK", "OFF")

    # Helpful with some odd reprojection edge-cases
    os.environ.setdefault("OGR_ENABLE_PARTIAL_REPROJECTION", "YES")


def _print_env_banner():
    print("\n" + "=" * 78)
    print("[env] Python exe:", sys.executable)
    print("[env] Python ver:", sys.version.replace("\n", " "))
    print("[env] CWD       :", os.getcwd())
    print("[env] sys.prefix:", sys.prefix)
    print("[env] PROJ_LIB  :", os.environ.get("PROJ_LIB", "(not set)"))
    print("[env] GDAL_DATA :", os.environ.get("GDAL_DATA", "(not set)"))
    print("[env] PROJ_NETWORK:", os.environ.get("PROJ_NETWORK", "(not set)"))
    print("[env] GTIFF_SRS_SOURCE:", os.environ.get("GTIFF_SRS_SOURCE", "(not set)"))
    print("=" * 78 + "\n")


# =============================================================================
# 2) ENV + LOGGING (INVEST TEMPLATE STYLE) -- lazy, NOT at import
# =============================================================================
LOGGER = logging.getLogger(__name__)


def _setup_sdr_environment():
    """PROJ/GDAL env, gdal exceptions and InVEST-style logging. Called by run_invest_sdr, never at
    import: importing global_invest.erosion must not mutate process-wide state (root logging
    config, GDAL error behaviour). In the source repo this ran at module import."""
    import natcap.invest.utils
    _set_proj_gdal_env()
    _print_env_banner()
    gdal.UseExceptions()
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt=natcap.invest.utils.LOG_FMT, datefmt="%m/%d/%Y %H:%M:%S ")
    handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[handler])


# =============================================================================
# 3) HELPERS
# =============================================================================
def _assert_exists(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"[missing] {label}: {path}")


def _dem_wkt(dem_path: Path) -> str:
    """Return DEM CRS as WKT. We trust DEM CRS as the canonical target CRS. (hazelbean raster-info;
    swap verified by the bit-identical section-A rerun.)"""
    wkt = hb.get_raster_info_hb(str(dem_path))['projection']
    if not wkt:
        raise ValueError(f"DEM has no CRS: {dem_path}")
    return wkt


def sanitize_watersheds_for_report(
    watersheds_in: Path,
    watersheds_out: Path,
    target_wkt: str,
    layer: str = "watersheds",
) -> Path:
    """
    Creates a minimal watersheds GeoPackage for InVEST SDR reporting/zonal stats.

    NOTE:
    - SDR does NOT need HydroBASINS IDs; it only needs polygons.
    - We keep only geometry + a safe integer ws_id, dropping everything else.
    - Reproject to DEM CRS (clean WKT) to avoid CRS parse failures.
    - Attempt to fix invalid geometries via buffer(0) (common in global vectors).
    """
    if not watersheds_in.exists():
        raise FileNotFoundError(f"Missing watersheds: {watersheds_in}")

    gdf = gpd.read_file(watersheds_in)
    if gdf.empty:
        raise ValueError(f"Watersheds are empty: {watersheds_in}")

    # Fix invalid geometries defensively
    try:
        bad = ~gdf.geometry.is_valid
        if bad.any():
            LOGGER.warning("Watersheds: fixing %d invalid geometries via buffer(0).", int(bad.sum()))
            gdf.loc[bad, "geometry"] = gdf.loc[bad, "geometry"].buffer(0)
    except Exception:
        LOGGER.warning("Watersheds: geometry validity check/fix skipped (non-fatal).")

    # Reproject to DEM CRS using WKT -> CRS
    gdf = gdf.to_crs(target_wkt)

    # Keep only geometry + safe ID
    gdf_out = gdf[["geometry"]].copy()
    gdf_out["ws_id"] = range(1, len(gdf_out) + 1)

    # Write a fresh GPKG
    if watersheds_out.exists():
        watersheds_out.unlink()

    gdf_out.to_file(watersheds_out, layer=layer, driver="GPKG")
    return watersheds_out


# =============================================================================
# 4) SDR ARGS (INVEST 3.17.x TEMPLATE)
# =============================================================================
def build_args(watersheds_path: Path) -> dict:
    return {
        "workspace_dir": str(BASE_OUT),
        "results_suffix": "2019_revised_dec_14",

        "biophysical_table_path": str(BIOPHYS_CSV),
        "dem_path": str(DEM_TIF),
        "lulc_path": str(LULC_TIF),
        "erodibility_path": str(K_TIF),
        "erosivity_path": str(R_TIF),
        "watersheds_path": str(watersheds_path),

        "drainage_path": str(DRAINAGE_PATH) if DRAINAGE_PATH else "",

        "flow_dir_algorithm": "D8",           # "D8" or "MFD"
        "threshold_flow_accumulation": 100,   # int
        "k_param": 2.0,                       # float
        "sdr_max": 0.5,                       # float
        "ic_0_param": 0.8,                    # float
        "l_max": 122.0,                       # float

        # Parallelism (remove if your InVEST build rejects unknown keys)
        "n_workers": -1,
    }


# =============================================================================
# 5) MAIN RUN
# =============================================================================
def run_invest_sdr():
    _setup_sdr_environment()
    _assert_exists(BASE_IN, "BASE_IN directory")
    _assert_exists(BIOPHYS_CSV, "biophysical_table_path")
    _assert_exists(DEM_TIF, "dem_path")
    _assert_exists(LULC_TIF, "lulc_path")
    _assert_exists(K_TIF, "erodibility_path")
    _assert_exists(R_TIF, "erosivity_path")
    _assert_exists(WATERSHEDS_RAW, "watersheds_path (MERGED raw)")

    print("[paths] ROOT     :", ROOT)
    print("[paths] BASE_IN  :", BASE_IN)
    print("[paths] BASE_OUT :", BASE_OUT)
    print("[paths] WATERSHEDS_RAW (MERGED):", WATERSHEDS_RAW)

    dem_wkt = _dem_wkt(DEM_TIF)

    print("[prep] Sanitizing MERGED watersheds for SDR report/zonal stats …")
    ws_sanitized = sanitize_watersheds_for_report(
        watersheds_in=WATERSHEDS_RAW,
        watersheds_out=WATERSHEDS_SANITIZED,
        target_wkt=dem_wkt,
        layer=WATERSHEDS_SAN_LAYER,
    )
    print("[prep] ✅ Watersheds sanitized:", ws_sanitized)

    args = build_args(ws_sanitized)

    print("\n" + "=" * 78)
    print("[run] Starting InVEST SDR …")
    print("[run] workspace_dir :", args["workspace_dir"])
    print("[run] results_suffix:", args["results_suffix"])
    print("[run] flow_dir_algorithm:", args["flow_dir_algorithm"])
    print("[run] threshold_flow_accumulation:", args["threshold_flow_accumulation"])
    print("[run] params: k_param=%.3f, sdr_max=%.3f, ic_0_param=%.3f, l_max=%.3f"
          % (args["k_param"], args["sdr_max"], args["ic_0_param"], args["l_max"]))
    print("[run] n_workers:", args.get("n_workers", "(not set)"))
    print("=" * 78 + "\n")

    import natcap.invest.sdr.sdr
    file_registry = natcap.invest.sdr.sdr.execute(args)

    print("\n[done] ✅ InVEST SDR finished.")
    print("[done] Results in:", args["workspace_dir"])
    print("[done] MERGED watersheds used (raw):", WATERSHEDS_RAW)
    print("[done] Sanitized watersheds used   :", ws_sanitized)
    return args, file_registry


# ==============================
# 1) CONFIG — EDIT AS NEEDED
# ==============================
SCENARIO_NAME = "SES — On-farm + Upstream (decomposed)"
RUN_TAG = "ses11_onfarm_upstream_combined_20260305"

# =============================================================================
# SECTION B -- On-farm + upstream erosion prevention-share GEP valuation
#              (from Combine_PS_SES11_3_3_2026.ipynb)
# =============================================================================
ROOT = Path("/projects/standard/jajohns/shared/sediment_gep/sediment_feb_2026")

# Primary erosion inputs (SDR-derived)
IN_DIR = ROOT / "inputs_v2" / "erosion_gep"
USLE_PATH  = IN_DIR / "usle_2019_revised_feb_13.tif"              # actual erosion rate (t/ha/yr)
AVOID_PATH = IN_DIR / "avoided_erosion_2019_revised_feb_13.tif"   # avoided erosion rate (t/ha/yr)

# Precomputed upstream prevention share raster (from your DEM+D8 workflow)
UPS_DIR   = ROOT / "upstream_prevention_attribution_v3"
UPS_PATH  = UPS_DIR / "upstream_prevention_share.tif"

# Optional upstream LULC attribution shares (diagnostics only)
UPS_FOREST_SHARE = UPS_DIR / "upslope_forest_share.tif"
UPS_GRASS_SHARE  = UPS_DIR / "upslope_grass_share.tif"
UPS_CROP_SHARE   = UPS_DIR / "upslope_cropland_share.tif"
UPS_BARE_SHARE   = UPS_DIR / "upslope_bare_share.tif"
USE_UPSLOPE_LULC_ATTRIBUTION_DIAGNOSTICS = True

# Boundary with ISO3 column
BOUNDARY_GPKG = IN_DIR / "country_boundary_r250_with_iso3.gpkg"
BOUNDARY_SOURCE_EPSG = None  # set to 4326 if you know it is WGS84

# Optional DEM used ONLY for threshold policy (low-elevation rule). If not present, rule is skipped.
ELEVATION_PATH = ROOT / "inputs_v2" / "sdr" / "global_dem_reproj.tif"

# Crops: SPAM2020 stacks
CROP_DIR   = IN_DIR / "crops"
YIELD_STACK = CROP_DIR / "spam2020_yield_stack_TA.tif"             # t/ha
AREA_STACK  = CROP_DIR / "spam2020_harvested_area_stack_TA.tif"    # ha per pixel (EXTENSIVE)
BANDMAP_CSV = CROP_DIR / "spam2020_bandmap.csv"                    # band,crop
AREA_STACK_IS_HA_PER_PIXEL = True

# Elasticity table (used only in valuation Option A)
ELASTICITY_CSV = IN_DIR / "elasticity_crops_fao_revised.csv"

# Valuation inputs
FAO_GPV_ISO3_CSV     = IN_DIR / "faostat_gpv_2019_iso3.csv"
FAO_PRICES_FULL_CSV  = IN_DIR / "faostat_prices_2019_completed_revised.csv"
GDP_CURRENT_2019_CSV = IN_DIR / "worldbank_gdp_2019.csv"

# Output directory
OUT_DIR = ROOT / f"output_{RUN_TAG}"

# -------- Scientific knobs --------
THRESH_LOW  = 2.0
THRESH_HIGH = 11.0   # SES-11 default threshold
SMALL_COUNTRY_AREA_KM2 = 50_000
LOW_ELEVATION_MEAN_M   = 250

# If True: PS is computed only for severe pixels on cropland (recommended)
APPLY_SEVERE_FILTER = True

# Elasticity fallback if crop not found
YIELD_REDUCTION_FOR_SHOCK = 0.08

# Shock floor (applied only to (0 < shock < floor))
MIN_SHOCK_FLOOR = erosion_chain.MIN_SHOCK_FLOOR

BASE_YEAR_FOR_CONSTANT = 2019

# World Bank API toggles
AUTO_DOWNLOAD_WB = True
FORCE_REFRESH_WB = False
_HTTP_TIMEOUT = 60
_RETRY = 4

# Rasterization behavior
RASTERIZE_ALL_TOUCHED = True

# Crops-only filter for FAO GPV
CROPS_ONLY = True

# DEM validity rules (prevents low-elevation misclassification)
DEM_MASK_BELOW_SEA_LEVEL = True
DEM_MAX_VALID_ELEV_M = 9000.0

# -------- Enforce equal-area analysis CRS --------
ANALYSIS_EPSG = 8857

# Reprojection strategies
RESAMPLE_USLE_AVOID = Resampling.average
RESAMPLE_UPS        = Resampling.average
RESAMPLE_YIELD      = Resampling.average
RESAMPLE_AREA       = Resampling.sum
RESAMPLE_DEM        = Resampling.average



# ==========================================================
# 3) Elasticity loader (with SPAM support)
# ==========================================================
# SPAM_ALIAS_MAP: uses the corrected map defined ABOVE in this module (exact FAO item names
# first; the source repo's stem-only aliases never exact-matched the FAO names, so every crop
# silently took the 0.08 fallback).

def accumulate_upstream_prevention_share(dem_path, avoided_path, potential_path, work_dir,
                                         output_path):
    """Route the two erosion rasters down the drainage network and write the upstream share.

    A field is protected not only by its own cover but by everything upslope of it, so the share
    upstream cover prevents is the flow-accumulated avoided erosion over the flow-accumulated
    potential erosion. Both are accumulated on the same D8 network, so the pixel area cancels and
    the result is a share the on-farm one can be combined with.

    This is the layer the valuation used to read from the source repo's cluster workspace. It is
    computed here instead, from the DEM and the SDR outputs, which is what lets the account run
    without that workspace.

    Args:
        dem_path (str): elevation, resampled here to the erosion rasters' grid.
        avoided_path (str): avoided erosion, t/ha/yr.
        potential_path (str): potential (bare-soil) erosion, t/ha/yr.
        work_dir (str): where the filled DEM, flow direction and accumulation rasters go.
        output_path (str): where to write the share.

    Returns:
        str: `output_path`.
    """
    import pygeoprocessing as pgp
    import pygeoprocessing.routing as routing

    os.makedirs(work_dir, exist_ok=True)
    info = pgp.get_raster_info(avoided_path)
    pixel_size, geotransform, wkt = (
        info['pixel_size'], info['geotransform'], info['projection_wkt'])
    origin = (geotransform[0], geotransform[3])

    def _weight(source_path, destination_path):
        # Flow accumulation sums its weight raster, so nodata and negatives have to become zero
        # first: left as they are, a nodata sentinel would be routed downstream as if it were mass.
        source_ndv = pgp.get_raster_info(source_path)['nodata'][0]
        array = hb.as_array(source_path).astype('float64')
        array = np.where(np.isfinite(array) & (array != source_ndv) & (np.abs(array) < 1e30),
                         np.maximum(array, 0.0), 0.0)
        pgp.numpy_array_to_raster(array.astype('float32'), -1.0, pixel_size, origin, wkt,
                                  destination_path)
        return destination_path

    dem_on_grid = os.path.join(work_dir, 'dem_grid.tif')
    hb.resample_to_match(dem_path, avoided_path, dem_on_grid, resample_method='bilinear')
    avoided_weight = _weight(avoided_path, os.path.join(work_dir, 'avoided_w.tif'))
    potential_weight = _weight(potential_path, os.path.join(work_dir, 'rkls_w.tif'))

    filled = os.path.join(work_dir, 'filled.tif')
    flow_direction = os.path.join(work_dir, 'fdir.tif')
    accumulated_avoided = os.path.join(work_dir, 'acc_avoided.tif')
    accumulated_potential = os.path.join(work_dir, 'acc_rkls.tif')
    routing.fill_pits((dem_on_grid, 1), filled)
    routing.flow_dir_d8((filled, 1), flow_direction)
    routing.flow_accumulation_d8((flow_direction, 1), accumulated_avoided,
                                 weight_raster_path_band=(avoided_weight, 1))
    routing.flow_accumulation_d8((flow_direction, 1), accumulated_potential,
                                 weight_raster_path_band=(potential_weight, 1))

    share = erosion_chain.upstream_prevention_share(
        hb.as_array(accumulated_avoided), hb.as_array(accumulated_potential))
    pgp.numpy_array_to_raster(share, -9999.0, pixel_size, origin, wkt, output_path)
    return output_path


def load_elasticity_map(elasticity_csv: Path, fallback_value: float) -> tuple[dict, pd.DataFrame]:
    assert_exists(elasticity_csv, "Provide elasticity CSV in inputs/.")
    df = pd.read_csv(elasticity_csv, encoding="utf-8-sig")
    df = _normcols(df)

    crop_col = None
    for cand in ["crop", "monfreda_crop", "item", "item_name"]:
        if cand in df.columns:
            crop_col = cand
            break

    if crop_col is None:
        warnings.warn(
            "No crop-name column found in elasticity CSV. "
            "Accepted: 'crop'|'monfreda_crop'|'item'|'item_name'. "
            f"All crops will fall back to {fallback_value:.2%}."
        )
        return {}, df

    if "elasticity" not in df.columns:
        raise ValueError("Elasticity CSV must contain 'elasticity'.")

    df["elasticity"] = pd.to_numeric(df["elasticity"], errors="coerce").clip(0.0, 1.0)
    df["__key__"] = df[crop_col].astype(str).str.strip().str.lower()
    df = df[df["__key__"].ne("") & df["elasticity"].notna()].drop_duplicates(subset="__key__", keep="last")
    mapping = dict(zip(df["__key__"], df["elasticity"]))

    audit = df.rename(columns={crop_col: "crop_key"})[["crop_key", "elasticity"]].copy()
    return mapping, audit

def get_elasticity_for_crop(crop_key: str, elast_map: dict, fallback: float) -> float:
    crop_key = str(crop_key).strip().lower()
    v = elast_map.get(crop_key, np.nan)
    if np.isfinite(v):
        return float(np.clip(v, 0.0, 1.0))
    for alias in SPAM_ALIAS_MAP.get(crop_key, []):
        v2 = elast_map.get(alias.strip().lower(), np.nan)
        if np.isfinite(v2):
            return float(np.clip(v2, 0.0, 1.0))
    return float(np.clip(fallback, 0.0, 1.0))


# ==============================
# 4) Countries utilities
# ==============================
def load_countries_iso3_alpha(in_crs: rioCRS):
    assert_exists(BOUNDARY_GPKG, "Provide boundary with ISO3 column.")
    gdf = gpd.read_file(str(BOUNDARY_GPKG))
    gdf = gdf[gdf.geometry.notnull()].copy()

    if gdf.crs is None:
        if BOUNDARY_SOURCE_EPSG is not None:
            gdf = gdf.set_crs(BOUNDARY_SOURCE_EPSG)
            warnings.warn(f"Boundary had no CRS; set to EPSG:{BOUNDARY_SOURCE_EPSG}.")
        else:
            minx, miny, maxx, maxy = gdf.total_bounds
            looks_like_wgs84 = (-180.01 <= minx <= 180.01 and -180.01 <= maxx <= 180.01 and
                                -90.01  <= miny <=  90.01 and  -90.01  <= maxy <=  90.01)
            if looks_like_wgs84:
                gdf = gdf.set_crs(4326)
                warnings.warn("Boundary had no CRS; heuristically set to EPSG:4326 (WGS84).")
            else:
                raise ValueError(
                    "Boundary file has no CRS and does not look like lon/lat. "
                    "Set BOUNDARY_SOURCE_EPSG to the file’s source CRS."
                )

    iso_col = None
    for cand in ["iso3", "ISO3", "iso_a3", "adm0_a3", "ADM0_A3", "iso3_r250_label"]:
        if cand in gdf.columns:
            iso_col = cand
            break
    if not iso_col:
        raise ValueError("Boundary file must contain an ISO3 column (e.g., 'iso3').")
    gdf = gdf.rename(columns={iso_col: "ISO3"})
    gdf["ISO3"] = gdf["ISO3"].astype(str).str.upper()

    name_col = None
    for cand in ["country_name","NAME_EN","ADMIN","NAME_LONG","NAME",
                 "COUNTRY","NAME_0","ADM0_NAME","GEOUNIT","iso3_r250_name"]:
        if cand in gdf.columns:
            name_col = cand
            break
    gdf["country_name"] = gdf[name_col].astype(str) if name_col else gdf["ISO3"]

    gdf = gdf.to_crs(in_crs)
    return gdf[["ISO3","country_name","geometry"]]

def rasterize_iso3(gdf: gpd.GeoDataFrame, like_da: xr.DataArray):
    gdf = gdf.copy()
    gdf["ISO3"] = gdf["ISO3"].astype(str).str.upper()
    lut = (
        gdf[["ISO3"]].drop_duplicates().reset_index(drop=True)
        .assign(iso_id=lambda d: np.arange(1, len(d) + 1, dtype=np.int32))
    )
    gdf = gdf.merge(lut, on="ISO3", how="left")

    arr = rasterio.features.rasterize(
        shapes=zip(gdf.geometry, gdf["iso_id"].astype(int)),
        out_shape=(like_da.rio.height, like_da.rio.width),
        transform=like_da.rio.transform(),
        fill=0,
        dtype="int32",
        all_touched=RASTERIZE_ALL_TOUCHED,
    )
    return arr, lut

def compute_country_areas_km2(gdf: gpd.GeoDataFrame) -> pd.Series:
    area_km2 = gdf.geometry.area.values / 1_000_000.0
    return pd.Series(area_km2, index=gdf.index, dtype="float64")

def compute_country_mean_elevation(
    usle_like_da: xr.DataArray,
    iso_id_raster: np.ndarray,
    iso_lut: pd.DataFrame,
    elev_path: Path | None
) -> dict[int, float]:
    if elev_path is None or not elev_path.exists():
        warnings.warn("No DEM provided — elevation rule will be skipped.")
        return {}

    dem_native = open_raster_1band(elev_path)
    _ensure_crs(dem_native, "DEM")
    dem = dem_native.rio.reproject_match(usle_like_da, resampling=RESAMPLE_DEM)

    vals = dem.values.astype("float64", copy=True)
    ids  = iso_id_raster.astype("int32", copy=False)

    nodata = dem.rio.nodata
    if nodata is not None and np.isfinite(nodata):
        vals[vals == float(nodata)] = np.nan
    vals[~np.isfinite(vals)] = np.nan

    if DEM_MASK_BELOW_SEA_LEVEL:
        vals[vals < 0.0] = np.nan
    if DEM_MAX_VALID_ELEV_M is not None and np.isfinite(DEM_MAX_VALID_ELEV_M):
        vals[vals > float(DEM_MAX_VALID_ELEV_M)] = np.nan

    m = np.isfinite(vals) & (ids > 0)
    max_id = int(iso_lut["iso_id"].max())

    if not np.any(m):
        warnings.warn("DEM masking removed all elevation samples; skipping elevation rule.")
        return {}

    sum_elev = np.bincount(ids[m], weights=vals[m], minlength=max_id + 1).astype("float64")
    cnt_elev = np.bincount(ids[m], minlength=max_id + 1).astype("float64")
    mean_elev = np.divide(sum_elev, cnt_elev, out=np.full_like(sum_elev, np.nan), where=cnt_elev > 0)

    return {i: float(mean_elev[i]) for i in range(1, max_id + 1)}


# ==========================================================
# 5) FAO GPV loader + crops-only filter + fallback
# ==========================================================
def _filter_crops_only(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns
    item_col = "item" if "item" in cols else ("item name" if "item name" in cols else None)
    group_col = "domain" if "domain" in cols else None
    if not item_col and not group_col:
        warnings.warn("FAO GPV: no item/domain labels; cannot restrict to crops — using all rows.")
        return df

    def _s(s): return s.astype(str).str.lower()

    include = pd.Series(False, index=df.index)
    if item_col:
        s = _s(df[item_col])
        include = include | s.str.contains(r"\bcrop", na=False)
        include = include | s.str.contains(
            "apple|apricot|barley|bean|cabbage|carrot|cereal|cherr|chilli|cocoa|coffee|cotton|"
            "cucumber|date|eggplant|fig|fruit|grape|lentil|maize|melon|millet|nut|oat|onion|pea|peanut|"
            "pepper|potato|rice|rye|sesame|sorghum|soy|sunflower|tomato|vegetable|wheat|yuca|cassava",
            na=False
        )
    if group_col:
        s = _s(df[group_col])
        include = include | s.str.contains(r"\bcrop", na=False)

    neg = pd.Series(False, index=df.index)
    for col in [c for c in [item_col, group_col] if c]:
        s = _s(df[col])
        neg = neg | s.str.contains("livestock|animal|fish|fisher|forestry|forest", na=False)
        neg = neg | (s.str.contains("agriculture", na=False) & ~s.str.contains("crop", na=False))
        neg = neg | s.str.contains("crops? and livestock", na=False)

    out = df[include & ~neg].copy()
    if out.empty:
        warnings.warn("FAO GPV: crops-only filter removed all rows — falling back to original dataset.")
        return df
    return out

def load_fao_prices_full(path: Path) -> pd.DataFrame:
    assert_exists(path, "Provide prices CSV for GPV fallback.")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = _normcols(df)

    if "quantity_tons" not in df.columns and "crop_quantity_fao" in df.columns:
        df = df.rename(columns={"crop_quantity_fao": "quantity_tons"})
    if "item_name" not in df.columns and "item" in df.columns:
        df = df.rename(columns={"item": "item_name"})
    if "iso3" not in df.columns:
        raise ValueError("Prices CSV must contain 'iso3' (or '*' rows for global).")

    for need in ["iso3", "price_usd_per_ton", "quantity_tons"]:
        if need not in df.columns:
            raise ValueError(f"Prices CSV missing '{need}' column.")

    df["iso3"] = df["iso3"].astype(str).str.upper()
    df["price_usd_per_ton"] = pd.to_numeric(df["price_usd_per_ton"], errors="coerce")
    df["quantity_tons"] = pd.to_numeric(df["quantity_tons"], errors="coerce")
    return df

def load_fao_gpv_iso3_const2019_with_fallback(
    fao_csv_iso3: Path,
    prices_full_csv: Path,
    base_year: int = 2019
) -> pd.DataFrame:
    assert_exists(fao_csv_iso3, "Provide iso3-based FAO file (faostat_gpv_2019_iso3.csv).")
    base = pd.read_csv(fao_csv_iso3, encoding="utf-8-sig")
    base = _normcols(base)

    needed = {"iso3", "year", "unit", "value", "element"}
    miss = needed - set(base.columns)
    if miss:
        raise ValueError(f"FAO CSV missing required columns {miss}. Found: {list(base.columns)}")

    base = base[base["year"].astype(str) == str(base_year)].copy()
    el_ok = base["element"].astype(str).str.lower().str.contains("gross production value")
    usd_ok = base["unit"].astype(str).str.lower().str.contains("1000") & base["unit"].astype(str).str.upper().str.contains("USD")
    base = base[el_ok & usd_ok].copy()
    if CROPS_ONLY:
        base = _filter_crops_only(base)

    base["iso3"] = base["iso3"].astype(str).str.upper()
    base["value_thousand_usd"] = pd.to_numeric(base["value"], errors="coerce")
    fao_out = base.groupby("iso3", as_index=False)["value_thousand_usd"].sum()
    fao_out["crop_gpv_const2019_2019"] = fao_out["value_thousand_usd"] * 1000.0
    fao_out = fao_out[["iso3","crop_gpv_const2019_2019"]]

    prices = load_fao_prices_full(prices_full_csv)
    key = "item_code_fao" if "item_code_fao" in prices.columns else "item_name"

    if (prices["iso3"] == "*").any():
        global_price = (
            prices[prices["iso3"] == "*"]
            .groupby(key, as_index=False)["price_usd_per_ton"]
            .mean()
            .rename(columns={"price_usd_per_ton":"price_usd_per_ton_global"})
        )
    else:
        global_price = (
            prices.groupby(key, as_index=False)["price_usd_per_ton"]
            .mean()
            .rename(columns={"price_usd_per_ton":"price_usd_per_ton_global"})
        )

    dfp = prices[prices["iso3"] != "*"].copy()
    dfp = dfp.merge(global_price, on=key, how="left")
    dfp["price_used"] = dfp["price_usd_per_ton"]
    dfp.loc[dfp["price_used"].isna() | (dfp["price_used"] <= 0), "price_used"] = dfp["price_usd_per_ton_global"]
    dfp["fallback_value_usd"] = (dfp["price_used"].clip(lower=0.0) * dfp["quantity_tons"].clip(lower=0.0))

    gpv_fb = dfp.groupby("iso3", as_index=False)["fallback_value_usd"].sum()
    gpv_fb = gpv_fb.rename(columns={"fallback_value_usd":"crop_gpv_const2019_2019_fallback"})

    out = fao_out.merge(gpv_fb, on="iso3", how="outer")
    out["crop_gpv_const2019_2019"] = out["crop_gpv_const2019_2019"].fillna(out["crop_gpv_const2019_2019_fallback"])

    # extra diagnostic (publication transparency)
    out_diag = out.merge(gpv_fb, on="iso3", how="left")
    out_diag.to_csv(OUT_DIR / "gpv_fallback_diagnostic.csv", index=False)

    return out[["iso3","crop_gpv_const2019_2019"]]


# ------------------------------------------------------
# 6) World Bank GDP loader
# ------------------------------------------------------
def _write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def fetch_worldbank_gdp_2019_current(cache_path: Path, force: bool=False) -> pd.DataFrame:
    if cache_path.exists() and not force:
        df = pd.read_csv(cache_path, encoding="utf-8-sig")
        df = _normcols(df)
        if {"iso3","gdp_current_2019"}.issubset(df.columns):
            df["iso3"] = df["iso3"].astype(str).str.upper()
            return df[["iso3","gdp_current_2019"]]
    url = "https://api.worldbank.org/v2/country/all/indicator/NY.GDP.MKTP.CD"
    params = {"format":"json","per_page":20000}
    r = _http_get(url, params=params)
    payload = r.json()
    if not isinstance(payload, list) or len(payload) < 2:
        raise RuntimeError("Unexpected World Bank response for GDP.")
    df = pd.DataFrame(payload[1])[["countryiso3code","date","value"]]
    df = df[(df["date"]=="2019") & df["countryiso3code"].notna()]
    df = df[df["countryiso3code"].str.len()==3].rename(columns={"countryiso3code":"iso3","value":"gdp_current_2019"})
    df["iso3"] = df["iso3"].astype(str).str.upper()
    df["gdp_current_2019"] = pd.to_numeric(df["gdp_current_2019"], errors="coerce")
    _write_csv(df[["iso3","gdp_current_2019"]], cache_path)
    return df[["iso3","gdp_current_2019"]]

def load_wb_gdp_current_2019(gdp_csv: Path) -> pd.DataFrame:
    if gdp_csv.exists():
        df = pd.read_csv(gdp_csv, encoding="utf-8-sig")
        df = _normcols(df)
    else:
        if not AUTO_DOWNLOAD_WB:
            raise FileNotFoundError(f"Missing {gdp_csv} and AUTO_DOWNLOAD_WB=False.")
        df = fetch_worldbank_gdp_2019_current(gdp_csv, force=FORCE_REFRESH_WB)
    if "iso3" not in df.columns:
        raise ValueError("GDP CSV must contain column 'iso3'.")
    df["iso3"] = df["iso3"].astype(str).str.upper()
    col = "gdp_const2019_2019"
    if "gdp_current_2019" in df.columns:
        out = df[["iso3", "gdp_current_2019"]].copy()
        out[col] = pd.to_numeric(out["gdp_current_2019"], errors="coerce")
        return out[["iso3", col]]
    if col in df.columns:
        out = df[["iso3", col]].copy()
        out[col] = pd.to_numeric(out[col], errors="coerce")
        return out
    dld = fetch_worldbank_gdp_2019_current(gdp_csv, force=True)
    dld = dld.rename(columns={"gdp_current_2019": col})
    return dld[["iso3", col]]


# ==========================================================
# 7) CORE: SPAM production aggregation given a PS raster
# ==========================================================
def aggregate_country_crop_production(
    ps_arr01: np.ndarray,                        # 2D array aligned to USLE grid
    usle_like: xr.DataArray,                     # template for coords/dims
    iso_id_raster: np.ndarray,                   # int32 country ids aligned to grid
    id2iso: dict[int, str],
    bandmap: pd.DataFrame,
    elast_map: dict,
    max_id: int,
    out_prefix: str,
) -> pd.DataFrame:
    """
    Computes (for each country & crop):
      - protected_production_tons = PS * yield * harvested_area
      - total_production_tons = yield * harvested_area
      - share_protected_production = protected/total
      - elasticity_used
    """
    rows = []

    with rasterio.open(YIELD_STACK) as ds_y, rasterio.open(AREA_STACK) as ds_a:
        for _, row in bandmap.iterrows():
            b = int(row["band"])
            crop_raw = str(row["crop"]).strip()
            crop_key = crop_raw.lower()

            if b < 1 or b > ds_y.count or b > ds_a.count:
                continue

            elast = get_elasticity_for_crop(crop_key, elast_map, YIELD_REDUCTION_FOR_SHOCK)

            y_native = rxr.open_rasterio(ds_y.name, masked=True).sel(band=b).squeeze()
            a_native = rxr.open_rasterio(ds_a.name, masked=True).sel(band=b).squeeze()

            y_tgt  = y_native.rio.reproject_match(usle_like, resampling=RESAMPLE_YIELD).fillna(0.0)
            ha_tgt = a_native.clip(min=0.0).fillna(0.0).rio.reproject_match(usle_like, resampling=RESAMPLE_AREA).fillna(0.0)

            totalprod = (y_tgt * ha_tgt).fillna(0.0)
            protprod  = xr.DataArray(ps_arr01, coords=usle_like.coords, dims=usle_like.dims) * y_tgt * ha_tgt
            protprod  = protprod.fillna(0.0)

            tp_vals = totalprod.values.astype("float64", copy=False)
            pp_vals = protprod.values.astype("float64", copy=False)
            ids = iso_id_raster.astype("int32", copy=False)

            m_tp = np.isfinite(tp_vals) & (ids > 0)
            m_pp = np.isfinite(pp_vals) & (ids > 0)

            sums_tp = np.bincount(ids[m_tp], weights=tp_vals[m_tp], minlength=max_id + 1)
            sums_pp = np.bincount(ids[m_pp], weights=pp_vals[m_pp], minlength=max_id + 1)

            nz = np.flatnonzero((sums_tp + sums_pp) > 0)
            nz = nz[nz > 0]
            for iso_id in nz:
                iso_code = id2iso.get(int(iso_id))
                if iso_code is None:
                    continue
                protected_tons = float(sums_pp[iso_id])
                total_tons     = float(sums_tp[iso_id])
                share = (protected_tons / total_tons) if total_tons > 1e-12 else np.nan
                rows.append({
                    "component": out_prefix,  # onfarm / upstream / combined
                    "ISO3": iso_code,
                    "crop": crop_raw,
                    "elasticity_used": elast,
                    "protected_production_tons": protected_tons,
                    "total_production_tons": total_tons,
                    "share_protected_production": float(np.clip(share, 0.0, 1.0)) if pd.notna(share) else np.nan,
                })

    return pd.DataFrame(rows)


# ==========================================================
# 8) Valuation: elasticity-weighted shock + GEP (Option A)
# ==========================================================
def compute_country_gep_from_country_crop(
    df_country_crop_component: pd.DataFrame,
    fao_iso3_csv: Path,
    prices_full_csv: Path,
    base_year: int,
    gdp_current_2019_csv: Path,
    component: str,
) -> pd.DataFrame:
    """Country protected production, shock, value and GDP share, for one prevention component.

    Reads the crop gross production value and GDP tables, then hands the frames to
    `erosion_chain`, which holds the arithmetic.
    """
    df_shock = erosion_chain.country_erosion_shock(df_country_crop_component, MIN_SHOCK_FLOOR)
    df_crop_gpv = load_fao_gpv_iso3_const2019_with_fallback(
        fao_iso3_csv, prices_full_csv, base_year=base_year)
    df_gdp = load_wb_gdp_current_2019(gdp_current_2019_csv)
    return erosion_chain.country_gep(df_shock, df_crop_gpv, df_gdp, component)


# ==========================================================
# 9) BIOPHYSICAL — compute PS_onfarm, load UPS, build PS_eff
#     then compute onfarm/upstream/combined in parallel
# ==========================================================
def run_biophysical_decomposed():
    # ---- Required inputs
    assert_exists(USLE_PATH,  "Expected USLE raster.")
    assert_exists(AVOID_PATH, "Expected avoided_erosion raster.")
    assert_exists(UPS_PATH,   "Expected upstream_prevention_share.tif from upstream workflow.")
    assert_exists(YIELD_STACK, "Missing SPAM yield stack.")
    assert_exists(AREA_STACK,  "Missing SPAM harvested area stack.")
    assert_exists(BANDMAP_CSV, "Missing SPAM band map CSV.")
    assert_exists(ELASTICITY_CSV, "Missing elasticity table.")
    assert_exists(BOUNDARY_GPKG, "Missing country boundary GPKG.")

    analysis_crs = rioCRS.from_epsg(ANALYSIS_EPSG)

    # ---- Load native erosion layers
    usle_native = open_raster_1band(USLE_PATH)
    avo_native  = open_raster_1band(AVOID_PATH)
    ups_native  = open_raster_1band(UPS_PATH)

    _ensure_crs(usle_native, "USLE")
    _ensure_crs(avo_native,  "AVOID")
    _ensure_crs(ups_native,  "UPS")

    # ---- Reproject to equal-area CRS
    usle = reproject_to_analysis_grid(usle_native, analysis_crs, RESAMPLE_USLE_AVOID)
    avo  = reproject_to_analysis_grid(avo_native,  analysis_crs, RESAMPLE_USLE_AVOID)
    ups  = reproject_to_analysis_grid(ups_native,  analysis_crs, RESAMPLE_UPS)

    # ---- Force exact alignment to USLE grid
    avo  = avo.rio.reproject_match(usle, resampling=RESAMPLE_USLE_AVOID)
    ups  = ups.rio.reproject_match(usle, resampling=RESAMPLE_UPS)

    usle = _clean_nonneg(usle)
    avo  = _clean_nonneg(avo)
    ups_vals = _clip01_arr(ups.values)

    # ---- Countries raster
    gdf_countries = load_countries_iso3_alpha(usle.rio.crs)
    gdf_countries["area_km2"] = compute_country_areas_km2(gdf_countries)
    iso_id_raster, iso_lut = rasterize_iso3(gdf_countries, usle)
    max_id = int(iso_lut["iso_id"].max())
    id2iso = dict(zip(iso_lut["iso_id"].to_numpy(), iso_lut["ISO3"].to_numpy()))
    name_by_iso = dict(zip(gdf_countries["ISO3"], gdf_countries["country_name"]))

    # ---- Threshold policy (optional DEM)
    mean_elev_by_id = compute_country_mean_elevation(
        usle, iso_id_raster, iso_lut,
        ELEVATION_PATH if (ELEVATION_PATH and ELEVATION_PATH.exists()) else None
    )

    # A country's area is the sum of the sub-regions the boundary file splits it into, not any
    # one of them. The boundary file is r264, which splits 13 countries out into territories, so
    # taking a sub-region's area let China and India qualify as small on the strength of an
    # island: they were being given the low soil-loss tolerance, which counts far more of their
    # cropland as severely eroding. Seven countries changed on this: AUS, CHN, FRA, IND, NOR,
    # NZL, SRB.
    df_country_area = (gdf_countries[["ISO3", "area_km2"]].rename(columns={"ISO3": "iso3"})
                       .groupby("iso3", as_index=False)["area_km2"].sum(min_count=1))
    df_threshold = erosion_chain.country_threshold_policy(
        iso_lut.rename(columns={"ISO3": "iso3"})
               .merge(df_country_area, on="iso3", how="left")
               .assign(mean_elevation_m=lambda d: [mean_elev_by_id.get(int(i), np.nan)
                                                   for i in d["iso_id"]]),
        THRESH_HIGH, THRESH_LOW, SMALL_COUNTRY_AREA_KM2, LOW_ELEVATION_MEAN_M)
    if df_threshold["iso3"].duplicated().any():
        raise ValueError("the threshold policy has more than one row for a country, so the "
                         "per-country threshold raster would depend on row order.")

    threshold_by_id = np.full(max_id + 1, THRESH_HIGH, dtype="float32")
    id_by_iso = dict(zip(iso_lut["ISO3"], iso_lut["iso_id"].astype(int)))
    for iso3, threshold in zip(df_threshold["iso3"], df_threshold["threshold_t_ha_yr"]):
        threshold_by_id[id_by_iso[iso3]] = threshold
    threshold_map = threshold_by_id[iso_id_raster.astype("int32")]

    severe = (usle.values > threshold_map) if APPLY_SEVERE_FILTER else np.ones_like(usle.values, dtype=bool)

    df_threshold.insert(1, "country_name", [name_by_iso.get(i, i) for i in df_threshold["iso3"]])
    df_threshold.rename(columns={"iso3": "ISO3"}).to_csv(OUT_DIR / "threshold_policy.csv", index=False)

    # ---- Bandmap + elasticity
    bandmap = pd.read_csv(BANDMAP_CSV)
    bandmap = _normcols(bandmap)
    if "band" not in bandmap.columns or "crop" not in bandmap.columns:
        raise ValueError("BANDMAP_CSV must have columns: 'band', 'crop'.")
    bandmap["crop"] = bandmap["crop"].astype(str).str.strip()
    bandmap["band"] = pd.to_numeric(bandmap["band"], errors="coerce").astype("Int64")

    elast_map, elast_audit = load_elasticity_map(ELASTICITY_CSV, fallback_value=YIELD_REDUCTION_FOR_SHOCK)
    elast_audit.to_csv(OUT_DIR / "elasticity_audit.csv", index=False)

    # ---- Cropland mask built from SPAM area (union across bands) + area conservation audit
    cropland_mask = None
    area_conservation_rows = []

    with rasterio.open(AREA_STACK) as ds_a:
        for _, row in bandmap.iterrows():
            b = int(row["band"])
            if b < 1 or b > ds_a.count:
                continue
            a_native = rxr.open_rasterio(ds_a.name, masked=True).sel(band=b).squeeze()
            if a_native.rio.crs is None:
                raise ValueError(f"Area stack band {b} has no CRS. Fix the stack metadata.")
            a_pos  = a_native.clip(min=0.0).fillna(0.0)
            ha_tgt = a_pos.rio.reproject_match(usle, resampling=RESAMPLE_AREA).fillna(0.0)

            tot_src = float(np.nansum(a_pos.values))
            tot_tgt = float(np.nansum(ha_tgt.values))
            ratio = (tot_tgt / tot_src) if tot_src > 0 else np.nan
            area_conservation_rows.append({
                "band": b,
                "crop": str(row["crop"]).strip(),
                "total_area_ha_src": tot_src,
                "total_area_ha_tgt": tot_tgt,
                "ratio_tgt_over_src": ratio
            })

            band_mask = (ha_tgt > 0)
            cropland_mask = band_mask if cropland_mask is None else (cropland_mask | band_mask)

    pd.DataFrame(area_conservation_rows).to_csv(OUT_DIR / "area_conservation_audit.csv", index=False)

    cm = cropland_mask.values.astype(bool)

    # ---- Prevention shares, on cropland where soil loss is severe (see erosion_chain)
    ps_onfarm = erosion_chain.restrict_to_valued_pixels(
        erosion_chain.onfarm_prevention_share(avo.values, usle.values), cm, severe).astype("float32")
    ps_upstream = erosion_chain.restrict_to_valued_pixels(ups_vals, cm, severe).astype("float32")
    ps_combined = erosion_chain.combined_prevention_share(ps_onfarm, ps_upstream).astype("float32")

    # ---- Save PS rasters for transparency
    _write_share(OUT_DIR / "ps_onfarm_cropland_severe.tif", usle, ps_onfarm)
    _write_share(OUT_DIR / "ps_upstream_cropland_severe.tif", usle, ps_upstream)
    _write_share(OUT_DIR / "ps_combined_union_cropland_severe.tif", usle, ps_combined)

    # ---- Country PS diagnostics (means on cropland&severe)
    diag_mask = cm & severe & (iso_id_raster > 0)
    ids_1d = iso_id_raster[diag_mask].astype("int32", copy=False)

    mean_onfarm = _bincount_weighted_mean(ids_1d, ps_onfarm[diag_mask], max_id)
    mean_up     = _bincount_weighted_mean(ids_1d, ps_upstream[diag_mask], max_id)
    mean_comb   = _bincount_weighted_mean(ids_1d, ps_combined[diag_mask], max_id)

    diag_rows = []
    for i in range(1, max_id + 1):
        iso = id2iso.get(i)
        if not iso:
            continue
        diag_rows.append({
            "ISO3": iso,
            "country_name": name_by_iso.get(iso, iso),
            "mean_ps_onfarm_cropland_severe": float(mean_onfarm[i]),
            "mean_ps_upstream_cropland_severe": float(mean_up[i]),
            "mean_ps_combined_cropland_severe": float(mean_comb[i]),
        })
    df_diag = pd.DataFrame(diag_rows)

    # Optional: include upstream LULC attribution shares as country means (diagnostics only)
    if USE_UPSLOPE_LULC_ATTRIBUTION_DIAGNOSTICS:
        def _load_attr(p: Path, nm: str) -> np.ndarray | None:
            if not p.exists():
                warnings.warn(f"[ATTR] Missing {nm}: {p}")
                return None
            da0 = open_raster_1band(p)
            _ensure_crs(da0, nm)
            da1 = reproject_to_analysis_grid(da0, analysis_crs, Resampling.average).rio.reproject_match(usle, resampling=Resampling.average)
            return _clip01_arr(da1.values)

        attrs = {
            "upslope_forest_share": _load_attr(UPS_FOREST_SHARE, "upslope_forest_share"),
            "upslope_grass_share":  _load_attr(UPS_GRASS_SHARE,  "upslope_grass_share"),
            "upslope_cropland_share": _load_attr(UPS_CROP_SHARE, "upslope_cropland_share"),
            "upslope_bare_share":   _load_attr(UPS_BARE_SHARE,   "upslope_bare_share"),
        }
        for nm, arr in attrs.items():
            if arr is None:
                continue
            mean_attr = _bincount_weighted_mean(ids_1d, arr[diag_mask], max_id)
            df_diag[f"mean_{nm}_cropland_severe"] = [float(mean_attr[i]) for i in range(1, max_id + 1) if id2iso.get(i)]

    df_diag.to_csv(OUT_DIR / "country_ps_diagnostics.csv", index=False)

    # ---- Soil retained on cropland (tons): AE rate * pixel area, cropland mask (independent of decomposition)
    px_ha = pixel_area_hectares(usle)
    soil_retained_tons = xr.where(cropland_mask, (avo * px_ha), 0.0).fillna(0.0)
    vals = soil_retained_tons.values.astype("float64")
    ids_all = iso_id_raster.astype("int32")
    m = np.isfinite(vals) & (ids_all > 0)
    soil_sums = np.bincount(ids_all[m], weights=vals[m], minlength=max_id + 1)

    df_soil = pd.DataFrame({
        "ISO3": [id2iso[i] for i in range(1, max_id + 1) if i in id2iso],
        "soil_retained_cropland_tons": [float(soil_sums[i]) for i in range(1, max_id + 1) if i in id2iso]
    })

    # ---- Compute country-crop protected production for each component
    df_cc_onfarm   = aggregate_country_crop_production(ps_onfarm,   usle, iso_id_raster, id2iso, bandmap, elast_map, max_id, "onfarm")
    df_cc_upstream = aggregate_country_crop_production(ps_upstream, usle, iso_id_raster, id2iso, bandmap, elast_map, max_id, "upstream")
    df_cc_combined = aggregate_country_crop_production(ps_combined, usle, iso_id_raster, id2iso, bandmap, elast_map, max_id, "combined")

    # ---- Save per-country-crop (long form; publication transparency)
    df_country_crop_long = pd.concat([df_cc_onfarm, df_cc_upstream, df_cc_combined], ignore_index=True)
    df_country_crop_long.to_csv(OUT_DIR / "country_crop_protected_production_long.csv", index=False)

    # Optional: also write 3 separate files (handy for reviewers)
    df_cc_onfarm.to_csv(OUT_DIR / "country_crop_protected_production_onfarm.csv", index=False)
    df_cc_upstream.to_csv(OUT_DIR / "country_crop_protected_production_upstream.csv", index=False)
    df_cc_combined.to_csv(OUT_DIR / "country_crop_protected_production_combined.csv", index=False)

    # Country name master
    country_master = gdf_countries[["ISO3","country_name"]].drop_duplicates().reset_index(drop=True)

    return {
        "usle_template": usle,
        "iso_id_raster": iso_id_raster,
        "iso_lut": iso_lut,
        "id2iso": id2iso,
        "country_master": country_master,
        "df_soil": df_soil,
        "df_diag": df_diag,
        "df_country_crop": {
            "onfarm": df_cc_onfarm,
            "upstream": df_cc_upstream,
            "combined": df_cc_combined,
        }
    }


# ==============================================
# 10) INTEGRATE + WRITE (decomposed outputs)
# ==============================================
def integrate_and_write():
    t0 = time.time()

    # ---- Run biophysical + produce country-crop tables per component
    pack = run_biophysical_decomposed()
    country_master = pack["country_master"]
    df_soil = pack["df_soil"]
    df_diag = pack["df_diag"]
    dcc = pack["df_country_crop"]

    # ---- Compute valuation per component
    df_gep_onfarm = compute_country_gep_from_country_crop(
        dcc["onfarm"], FAO_GPV_ISO3_CSV, FAO_PRICES_FULL_CSV, BASE_YEAR_FOR_CONSTANT, GDP_CURRENT_2019_CSV, "onfarm"
    )
    df_gep_upstream = compute_country_gep_from_country_crop(
        dcc["upstream"], FAO_GPV_ISO3_CSV, FAO_PRICES_FULL_CSV, BASE_YEAR_FOR_CONSTANT, GDP_CURRENT_2019_CSV, "upstream"
    )
    df_gep_combined = compute_country_gep_from_country_crop(
        dcc["combined"], FAO_GPV_ISO3_CSV, FAO_PRICES_FULL_CSV, BASE_YEAR_FOR_CONSTANT, GDP_CURRENT_2019_CSV, "combined"
    )

    # ---- Pivot to wide (one row per country) for integrated_country_gep.csv
    def _pivot_component(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        x = df.copy()
        x = x.rename(columns={
            "iso3": "iso3",
            "protected_production_tons": f"protected_production_tons_{prefix}",
            "total_production_tons": f"total_production_tons_{prefix}",
            "share_protected_production": f"share_protected_production_{prefix}",
            "erosion_shock_share": f"erosion_shock_share_{prefix}",
            "gep_const2019_usd": f"gep_const2019_usd_{prefix}",
            "gdp_loss_pct": f"gdp_loss_pct_{prefix}",
        })
        keep = ["iso3",
                f"protected_production_tons_{prefix}",
                f"total_production_tons_{prefix}",
                f"share_protected_production_{prefix}",
                f"erosion_shock_share_{prefix}",
                "crop_gpv_const2019_2019",
                "gdp_const2019_2019",
                f"gep_const2019_usd_{prefix}",
                f"gdp_loss_pct_{prefix}"]
        return x[keep]

    w_on  = _pivot_component(df_gep_onfarm, "onfarm")
    w_up  = _pivot_component(df_gep_upstream, "upstream")
    w_com = _pivot_component(df_gep_combined, "combined")

    # ---- Merge wide tables + add soil retained + add diagnostics
    out = country_master.rename(columns={"ISO3":"iso3"}).merge(w_on, on="iso3", how="left")
    out = out.merge(w_up.drop(columns=["crop_gpv_const2019_2019","gdp_const2019_2019"], errors="ignore"), on="iso3", how="left")
    out = out.merge(w_com.drop(columns=["crop_gpv_const2019_2019","gdp_const2019_2019"], errors="ignore"), on="iso3", how="left")

    out = out.merge(df_soil.rename(columns={"ISO3":"iso3"}), on="iso3", how="left")
    out = out.merge(df_diag.rename(columns={"ISO3":"iso3"}), on=["iso3","country_name"], how="left")

    # ---- Optional: “incremental” attribution (not additive; shown for interpretation only)
    #   incremental_upstream = combined - onfarm
    #   incremental_onfarm   = combined - upstream
    out["gep_incremental_upstream_usd"] = out["gep_const2019_usd_combined"] - out["gep_const2019_usd_onfarm"]
    out["gep_incremental_onfarm_usd"]   = out["gep_const2019_usd_combined"] - out["gep_const2019_usd_upstream"]

    # ---- Write final integrated file
    out_cols_front = ["iso3","country_name",
                      "crop_gpv_const2019_2019","gdp_const2019_2019",
                      "soil_retained_cropland_tons"]
    # keep everything else after
    out = out[out_cols_front + [c for c in out.columns if c not in out_cols_front]]
    out = out.sort_values(["country_name","iso3"], na_position="last")
    out.to_csv(OUT_DIR / "integrated_country_gep.csv", index=False)

    # ---- Also write a “long” valuation table (nice for figures/tables)
    df_gep_long = pd.concat([df_gep_onfarm, df_gep_upstream, df_gep_combined], ignore_index=True)
    df_gep_long.to_csv(OUT_DIR / "country_gep_decomposition_long.csv", index=False)

    # ---- Manifest + run metadata
    manifest = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "scenario_name": SCENARIO_NAME,
        "run_tag": RUN_TAG,
        "analysis_epsg": ANALYSIS_EPSG,
        "apply_severe_filter": APPLY_SEVERE_FILTER,
        "threshold_high_t_ha_yr": THRESH_HIGH,
        "threshold_low_t_ha_yr": THRESH_LOW,
        "small_country_area_km2": SMALL_COUNTRY_AREA_KM2,
        "low_elevation_mean_m": LOW_ELEVATION_MEAN_M,
        "definitions": {
            "PS_onfarm": "AE/(AE+USLE) on cropland pixels (and severe if enabled); else 0",
            "UPS": "upstream_prevention_share evaluated at pixel j (and restricted to cropland & severe); else 0",
            "PS_combined": "1 - (1-PS_onfarm)*(1-UPS) (union-of-protection; avoids double counting)",
        },
        "inputs": {
            "usle_path": str(USLE_PATH),
            "avoided_path": str(AVOID_PATH),
            "ups_path": str(UPS_PATH),
            "boundary_gpkg": str(BOUNDARY_GPKG),
            "dem_path_for_thresholds": str(ELEVATION_PATH),
            "yield_stack": str(YIELD_STACK),
            "area_stack": str(AREA_STACK),
            "bandmap_csv": str(BANDMAP_CSV),
            "elasticity_csv": str(ELASTICITY_CSV),
            "fao_gpv_iso3_csv": str(FAO_GPV_ISO3_CSV),
            "fao_prices_csv": str(FAO_PRICES_FULL_CSV),
            "worldbank_gdp_csv": str(GDP_CURRENT_2019_CSV),
        },
        "outputs": {
            "integrated_country_gep": str(OUT_DIR / "integrated_country_gep.csv"),
            "country_crop_protected_production_long": str(OUT_DIR / "country_crop_protected_production_long.csv"),
            "country_gep_decomposition_long": str(OUT_DIR / "country_gep_decomposition_long.csv"),
            "country_ps_diagnostics": str(OUT_DIR / "country_ps_diagnostics.csv"),
            "ps_onfarm_raster": str(OUT_DIR / "ps_onfarm_cropland_severe.tif"),
            "ps_upstream_raster": str(OUT_DIR / "ps_upstream_cropland_severe.tif"),
            "ps_combined_raster": str(OUT_DIR / "ps_combined_union_cropland_severe.tif"),
            "area_conservation_audit": str(OUT_DIR / "area_conservation_audit.csv"),
            "elasticity_audit": str(OUT_DIR / "elasticity_audit.csv"),
            "gpv_fallback_diagnostic": str(OUT_DIR / "gpv_fallback_diagnostic.csv"),
            "threshold_policy": str(OUT_DIR / "threshold_policy.csv"),
        },
        "elapsed_minutes": round((time.time() - t0) / 60.0, 3),
    }
    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    (OUT_DIR / "run_metadata.txt").write_text(f"""
============================================================
Integrated GEP run — {SCENARIO_NAME}
Run tag: {RUN_TAG}
Timestamp (UTC): {manifest['timestamp_utc']}

Analysis CRS:
  EPSG:{ANALYSIS_EPSG} (equal-area required for tons conversions)

Severe erosion definition:
  severe = usle_actual > threshold(country)
  Default threshold (high): {THRESH_HIGH:g} t/ha/yr
  Low threshold (special cases): {THRESH_LOW:g} t/ha/yr

Decomposed prevention shares (cropland-only; severe-only if enabled):
  On-farm:   PS_onfarm = AE/(AE+USLE)
  Upstream:  PS_upstream = UPS(j)
  Combined:  PS_combined = 1 - (1-PS_onfarm)(1-PS_upstream)

Primary output:
  {OUT_DIR / 'integrated_country_gep.csv'}

Also written:
  - country_crop_protected_production_long.csv (country-crop, all components)
  - country_gep_decomposition_long.csv (country totals, all components)
  - country_ps_diagnostics.csv (country mean PS values on cropland&severe)
  - gpv_fallback_diagnostic.csv
  - area_conservation_audit.csv
  - elasticity_audit.csv

Elapsed minutes: {manifest['elapsed_minutes']}
============================================================
""", encoding="utf-8")

    print(f"✅ Done → {OUT_DIR / 'integrated_country_gep.csv'}")
    print(f"Manifest → {OUT_DIR / 'manifest.json'}")



def configure_prevention_shares(p):
    """
    Override the Section-B (on-farm + upstream prevention share GEP
    valuation) module-level constants from the ProjectFlow object.
    Called by erosion_tasks.prevention_shares() before
    integrate_and_write().

    NOTE (organizational quirk to flag to Justin/Chiara): the original
    Combine_PS_SES notebook points ROOT at a different filesystem root
    (/projects/standard/jajohns/shared/sediment_gep/sediment_feb_2026)
    than the SDR script's ROOT (/users/3/damph002/GEP/sediment) even
    though Section B's inputs (USLE, avoided erosion) are meant to be
    Section A's SDR outputs. p.erosion_gep_root defaults to the former;
    override it once both stages share one project layout.
    """
    global SCENARIO_NAME, RUN_TAG, ROOT, IN_DIR, USLE_PATH, AVOID_PATH
    global UPS_DIR, UPS_PATH, UPS_FOREST_SHARE, UPS_GRASS_SHARE, UPS_CROP_SHARE, UPS_BARE_SHARE
    global USE_UPSLOPE_LULC_ATTRIBUTION_DIAGNOSTICS, BOUNDARY_GPKG, BOUNDARY_SOURCE_EPSG
    global ELEVATION_PATH, CROP_DIR, YIELD_STACK, AREA_STACK, BANDMAP_CSV, AREA_STACK_IS_HA_PER_PIXEL
    global ELASTICITY_CSV, FAO_GPV_ISO3_CSV, FAO_PRICES_FULL_CSV, GDP_CURRENT_2019_CSV
    global OUT_DIR, THRESH_LOW, THRESH_HIGH, SMALL_COUNTRY_AREA_KM2, LOW_ELEVATION_MEAN_M
    global APPLY_SEVERE_FILTER, YIELD_REDUCTION_FOR_SHOCK, MIN_SHOCK_FLOOR, BASE_YEAR_FOR_CONSTANT
    global AUTO_DOWNLOAD_WB, FORCE_REFRESH_WB, RASTERIZE_ALL_TOUCHED, CROPS_ONLY
    global DEM_MASK_BELOW_SEA_LEVEL, DEM_MAX_VALID_ELEV_M, ANALYSIS_EPSG

    SCENARIO_NAME = getattr(p, 'erosion_scenario_name', SCENARIO_NAME)
    RUN_TAG = getattr(p, 'erosion_run_tag', RUN_TAG)
    ROOT = Path(getattr(p, 'erosion_gep_root', ROOT))

    IN_DIR = Path(getattr(p, 'erosion_gep_input_dir', ROOT / "inputs_v2" / "erosion_gep"))
    USLE_PATH = Path(getattr(p, 'erosion_usle_path', IN_DIR / "usle_2019_revised_feb_13.tif"))
    AVOID_PATH = Path(getattr(p, 'erosion_avoided_erosion_path', IN_DIR / "avoided_erosion_2019_revised_feb_13.tif"))

    UPS_DIR = Path(getattr(p, 'erosion_upstream_dir', ROOT / "upstream_prevention_attribution_v3"))
    UPS_PATH = Path(getattr(p, 'erosion_upstream_prevention_share_path', UPS_DIR / "upstream_prevention_share.tif"))
    UPS_FOREST_SHARE = Path(getattr(p, 'erosion_upslope_forest_share_path', UPS_DIR / "upslope_forest_share.tif"))
    UPS_GRASS_SHARE = Path(getattr(p, 'erosion_upslope_grass_share_path', UPS_DIR / "upslope_grass_share.tif"))
    UPS_CROP_SHARE = Path(getattr(p, 'erosion_upslope_cropland_share_path', UPS_DIR / "upslope_cropland_share.tif"))
    UPS_BARE_SHARE = Path(getattr(p, 'erosion_upslope_bare_share_path', UPS_DIR / "upslope_bare_share.tif"))
    USE_UPSLOPE_LULC_ATTRIBUTION_DIAGNOSTICS = getattr(p, 'erosion_use_upslope_lulc_diagnostics', True)

    BOUNDARY_GPKG = Path(getattr(p, 'erosion_country_boundary_path', IN_DIR / "country_boundary_r250_with_iso3.gpkg"))
    BOUNDARY_SOURCE_EPSG = getattr(p, 'erosion_boundary_source_epsg', None)
    ELEVATION_PATH = Path(getattr(p, 'erosion_elevation_path', ROOT / "inputs_v2" / "sdr" / "global_dem_reproj.tif"))

    CROP_DIR = Path(getattr(p, 'erosion_crop_dir', IN_DIR / "crops"))
    YIELD_STACK = Path(getattr(p, 'erosion_yield_stack_path', CROP_DIR / "spam2020_yield_stack_TA.tif"))
    AREA_STACK = Path(getattr(p, 'erosion_area_stack_path', CROP_DIR / "spam2020_harvested_area_stack_TA.tif"))
    BANDMAP_CSV = Path(getattr(p, 'erosion_bandmap_csv_path', CROP_DIR / "spam2020_bandmap.csv"))
    AREA_STACK_IS_HA_PER_PIXEL = getattr(p, 'erosion_area_stack_is_ha_per_pixel', True)

    ELASTICITY_CSV = Path(getattr(p, 'erosion_elasticity_csv_path', IN_DIR / "elasticity_crops_fao_revised.csv"))
    FAO_GPV_ISO3_CSV = Path(getattr(p, 'erosion_fao_gpv_iso3_csv_path', IN_DIR / "faostat_gpv_2019_iso3.csv"))
    FAO_PRICES_FULL_CSV = Path(getattr(p, 'erosion_fao_prices_csv_path', IN_DIR / "faostat_prices_2019_completed_revised.csv"))
    GDP_CURRENT_2019_CSV = Path(getattr(p, 'erosion_gdp_csv_path', IN_DIR / "worldbank_gdp_2019.csv"))

    OUT_DIR = Path(getattr(p, 'erosion_gep_output_dir', ROOT / f"output_{RUN_TAG}"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    THRESH_LOW = getattr(p, 'erosion_threshold_low_t_ha_yr', 2.0)
    THRESH_HIGH = getattr(p, 'erosion_threshold_high_t_ha_yr', 11.0)
    SMALL_COUNTRY_AREA_KM2 = getattr(p, 'erosion_small_country_area_km2', 50_000)
    LOW_ELEVATION_MEAN_M = getattr(p, 'erosion_low_elevation_mean_m', 250)
    APPLY_SEVERE_FILTER = getattr(p, 'erosion_apply_severe_filter', True)
    YIELD_REDUCTION_FOR_SHOCK = getattr(p, 'erosion_yield_reduction_for_shock', 0.08)
    MIN_SHOCK_FLOOR = getattr(p, 'erosion_min_shock_floor', 8e-10)
    BASE_YEAR_FOR_CONSTANT = getattr(p, 'erosion_base_year', 2019)
    AUTO_DOWNLOAD_WB = getattr(p, 'erosion_auto_download_wb', True)
    FORCE_REFRESH_WB = getattr(p, 'erosion_force_refresh_wb', False)
    RASTERIZE_ALL_TOUCHED = getattr(p, 'erosion_rasterize_all_touched', True)
    CROPS_ONLY = getattr(p, 'erosion_crops_only', True)
    DEM_MASK_BELOW_SEA_LEVEL = getattr(p, 'erosion_dem_mask_below_sea_level', True)
    DEM_MAX_VALID_ELEV_M = getattr(p, 'erosion_dem_max_valid_elev_m', 9000.0)
    ANALYSIS_EPSG = getattr(p, 'erosion_analysis_epsg', 8857)



# =============================================================================
# SECTION C — Maps & figures (from combined_maps_figures_SES_final.ipynb)
# =============================================================================

# =============================================================================
# 0) CONFIG
# =============================================================================
ROOT = Path("/projects/standard/jajohns/shared/sediment_gep/sediment_feb_2026")
RUN_DIR = ROOT / "output_eps5_onfarm_upstream_combined_20260305"

FIG_DIR = RUN_DIR / "figures_20260306_extended"
# (no mkdir at import -- configure_maps re-resolves FIG_DIR from p and creates it at run time)

INTEGRATED_CSV = RUN_DIR / "integrated_country_gep.csv"
COUNTRY_CROP_LONG_CSV = RUN_DIR / "country_crop_protected_production_long.csv"

PS_ONFARM_TIF   = RUN_DIR / "ps_onfarm_cropland_severe.tif"
PS_UPSTREAM_TIF = RUN_DIR / "ps_upstream_cropland_severe.tif"
PS_COMBINED_TIF = RUN_DIR / "ps_combined_union_cropland_severe.tif"

RUN_BOUNDARY_GPKG = ROOT / "inputs_v2" / "erosion_gep" / "country_boundary_r250_with_iso3.gpkg"

TOP_N = 20
TOP_N_LABELS = 25
RASTER_DOWNSAMPLE_FACTOR = 6
ROBINSON_CRS = "+proj=robin"
EXCLUDE_ISO3 = {"ATA"}

USD_TO_MILLIONS = 1e6
MAP_K_CLASSES = 5
MONEY_UNIT_LABEL = "2019 USD million"

def load_world_boundary_prefer_run() -> gpd.GeoDataFrame:
    if RUN_BOUNDARY_GPKG.exists():
        world = gpd.read_file(RUN_BOUNDARY_GPKG)
        iso_col = pick_iso3_column(world)
        if not iso_col:
            raise ValueError(f"Boundary has no ISO3 column. Columns: {list(world.columns)}")
        world = world.rename(columns={iso_col: "iso3"})
        world["iso3"] = world["iso3"].astype(str).str.upper()

        name_col = pick_name_column(world)
        if name_col and name_col != "country_name":
            world = world.rename(columns={name_col: "country_name"})
        if "country_name" not in world.columns:
            world["country_name"] = world["iso3"]

        world = world[world.geometry.notna()].copy()
        return world[["iso3", "country_name", "geometry"]]

    raise FileNotFoundError(f"Boundary GPKG not found: {RUN_BOUNDARY_GPKG}")


def generate_all_maps_and_figures():
    """Driver that produces every map/figure/CSV described in the module docstring below (originally a flat script)."""
    # =============================================================================
    # 2) LOAD DATA
    # =============================================================================
    assert_exists(INTEGRATED_CSV, "Run the latest integrated pipeline first.")
    df = pd.read_csv(INTEGRATED_CSV)
    df.columns = [c.strip() for c in df.columns]
    
    if "ISO3" in df.columns and "iso3" not in df.columns:
        df = df.rename(columns={"ISO3": "iso3"})
    df["iso3"] = df["iso3"].astype(str).str.upper()
    
    NUM_COLS = [
        "crop_gpv_const2019_2019", "gdp_const2019_2019", "soil_retained_cropland_tons",
        "protected_production_tons_onfarm", "total_production_tons_onfarm", "share_protected_production_onfarm",
        "erosion_shock_share_onfarm", "gep_const2019_usd_onfarm", "gdp_loss_pct_onfarm",
        "protected_production_tons_upstream", "total_production_tons_upstream", "share_protected_production_upstream",
        "erosion_shock_share_upstream", "gep_const2019_usd_upstream", "gdp_loss_pct_upstream",
        "protected_production_tons_combined", "total_production_tons_combined", "share_protected_production_combined",
        "erosion_shock_share_combined", "gep_const2019_usd_combined", "gdp_loss_pct_combined",
        "mean_ps_onfarm_cropland_severe", "mean_ps_upstream_cropland_severe", "mean_ps_combined_cropland_severe",
        "gep_incremental_upstream_usd", "gep_incremental_onfarm_usd",
    ]
    df = to_num(df, NUM_COLS)
    
    if {"gep_const2019_usd_onfarm", "gep_const2019_usd_upstream", "gep_const2019_usd_combined"}.issubset(df.columns):
        df["gep_const2019_usd_overlap"] = (
            df["gep_const2019_usd_onfarm"].fillna(0.0)
            + df["gep_const2019_usd_upstream"].fillna(0.0)
            - df["gep_const2019_usd_combined"].fillna(0.0)
        )
        sum_components = df["gep_const2019_usd_onfarm"].fillna(0.0) + df["gep_const2019_usd_upstream"].fillna(0.0)
        df["overlap_pct_of_sum_components"] = (
            100.0 * df["gep_const2019_usd_overlap"] / sum_components.where(sum_components > 0)
        ).where(sum_components > 0)
    else:
        df["gep_const2019_usd_overlap"] = np.nan
        df["overlap_pct_of_sum_components"] = np.nan
    
    for c in [
        "crop_gpv_const2019_2019",
        "gdp_const2019_2019",
        "gep_const2019_usd_onfarm",
        "gep_const2019_usd_upstream",
        "gep_const2019_usd_combined",
        "gep_const2019_usd_overlap",
        "gep_incremental_upstream_usd",
        "gep_incremental_onfarm_usd",
    ]:
        if c in df.columns:
            df[f"{c}_million"] = df[c] / USD_TO_MILLIONS
    
    if "country_name" not in df.columns:
        df["country_name"] = df["iso3"]
    
    if COUNTRY_CROP_LONG_CSV.exists():
        df_crop_long = pd.read_csv(COUNTRY_CROP_LONG_CSV)
        df_crop_long.columns = [c.strip() for c in df_crop_long.columns]
        if "ISO3" in df_crop_long.columns and "iso3" not in df_crop_long.columns:
            df_crop_long = df_crop_long.rename(columns={"ISO3": "iso3"})
    else:
        df_crop_long = None
    
    df.to_csv(FIG_DIR / "integrated_country_gep_plus_overlap.csv", index=False)
    
    
    # =============================================================================
    # 3) WORLD GEOMETRY
    # =============================================================================
    world = load_world_boundary_prefer_run()
    world["iso3"] = world["iso3"].astype(str).str.upper()
    g = world.merge(df, on="iso3", how="left")
    
    
    # =============================================================================
    # 4) BAR FIGURES
    # =============================================================================
    
    # 4.1 Top countries: Combined GEP
    col = "gep_const2019_usd_combined"
    if col in df.columns:
        top = top_n(df, col, TOP_N).copy()
        top["label"] = top["country_name"].fillna(top["iso3"])
        top = top.sort_values(col, ascending=True)
    
        plt.figure(figsize=(12, 10))
        plt.barh(top["label"], top[f"{col}_million"])
        plt.xlabel(f"Combined GEP ({MONEY_UNIT_LABEL})", fontsize=12)
        plt.title(f"Top {TOP_N}: Combined GEP from severe erosion protection", fontsize=16, pad=12)
        plt.grid(axis="x", alpha=0.25)
        savefig(FIG_DIR / "fig1_top20_combined_gep_2019usd_million.png", dpi=300)
    
    # 4.2 Decomposition to combined
    if {"gep_const2019_usd_onfarm_million", "gep_const2019_usd_combined_million"}.issubset(df.columns):
        top2 = top_n(df, "gep_const2019_usd_combined", TOP_N).copy()
        top2["label"] = top2["country_name"].fillna(top2["iso3"])
        top2 = top2.sort_values("gep_const2019_usd_combined", ascending=True)
    
        on = top2["gep_const2019_usd_onfarm_million"].fillna(0.0)
        comb = top2["gep_const2019_usd_combined_million"].fillna(0.0)
        incr_up = (comb - on).clip(lower=0.0)
    
        plt.figure(figsize=(12, 10))
        plt.barh(top2["label"], on, label="On-farm protection (standalone)")
        plt.barh(top2["label"], incr_up, left=on, label="Incremental upstream protection (given on-farm)")
        plt.xlabel(f"GEP ({MONEY_UNIT_LABEL})", fontsize=12)
        plt.title(f"Top {TOP_N}: Decomposition summing to Combined GEP", fontsize=16, pad=12)
        plt.grid(axis="x", alpha=0.25)
        plt.legend(loc="lower right", frameon=True)
        savefig(FIG_DIR / "fig2_top20_decomposition_to_combined_2019usd_million.png", dpi=300)
    
    # 4.3 Top overlap percent
    if "overlap_pct_of_sum_components" in df.columns:
        top_ov = top_n(df, "gep_const2019_usd_overlap", TOP_N).copy()
        top_ov["label"] = top_ov["country_name"].fillna(top_ov["iso3"])
        top_ov = top_ov.sort_values("gep_const2019_usd_overlap", ascending=True)
    
        plt.figure(figsize=(12, 10))
        plt.barh(top_ov["label"], top_ov["overlap_pct_of_sum_components"])
        plt.xlabel("Overlap as % of (On-farm + Upstream)", fontsize=12)
        plt.title(f"Top {TOP_N}: Overlap removed by union-of-protection", fontsize=16, pad=12)
        plt.grid(axis="x", alpha=0.25)
        savefig(FIG_DIR / "fig3_top20_overlap_pct_of_sum.png", dpi=300)
    
    # 4.4 Top overlap absolute
    if "gep_const2019_usd_overlap_million" in df.columns:
        top_ov_abs = top_n(df, "gep_const2019_usd_overlap", TOP_N).copy()
        top_ov_abs["label"] = top_ov_abs["country_name"].fillna(top_ov_abs["iso3"])
        top_ov_abs = top_ov_abs.sort_values("gep_const2019_usd_overlap", ascending=True)
    
        plt.figure(figsize=(12, 10))
        plt.barh(top_ov_abs["label"], top_ov_abs["gep_const2019_usd_overlap_million"])
        plt.xlabel(f"Overlap removed ({MONEY_UNIT_LABEL})", fontsize=12)
        plt.title(f"Top {TOP_N}: Overlap removed in absolute terms", fontsize=16, pad=12)
        plt.grid(axis="x", alpha=0.25)
        savefig(FIG_DIR / "fig4_top20_overlap_removed_2019usd_million.png", dpi=300)
    
    # 4.5 Macro exposure
    if "gdp_loss_pct_combined" in df.columns:
        top_gdp = top_n(df, "gdp_loss_pct_combined", TOP_N).copy()
        top_gdp["label"] = top_gdp["country_name"].fillna(top_gdp["iso3"])
        top_gdp = top_gdp.sort_values("gdp_loss_pct_combined", ascending=True)
    
        plt.figure(figsize=(12, 10))
        plt.barh(top_gdp["label"], top_gdp["gdp_loss_pct_combined"])
        plt.xlabel("Combined GEP as % of GDP", fontsize=12)
        plt.title(f"Top {TOP_N}: Macro exposure (Combined GEP / GDP)", fontsize=16, pad=12)
        plt.grid(axis="x", alpha=0.25)
        savefig(FIG_DIR / "fig5_top20_gdp_loss_pct_combined.png", dpi=300)
    
    # 4.6 Top countries by combined protected production
    if "protected_production_tons_combined" in df.columns:
        top_prot = top_n(df, "protected_production_tons_combined", TOP_N).copy()
        top_prot["label"] = top_prot["country_name"].fillna(top_prot["iso3"])
        top_prot = top_prot.sort_values("protected_production_tons_combined", ascending=True)
    
        plt.figure(figsize=(12, 10))
        plt.barh(top_prot["label"], top_prot["protected_production_tons_combined"])
        plt.xlabel("Protected production (tons)", fontsize=12)
        plt.title(f"Top {TOP_N}: Countries by protected production (combined)", fontsize=16, pad=12)
        plt.grid(axis="x", alpha=0.25)
        savefig(FIG_DIR / "fig6_top20_protected_production_tons_combined.png", dpi=300)
    
    # 4.7 Top countries by crop GPV
    if "crop_gpv_const2019_2019_million" in df.columns:
        top_gpv = top_n(df, "crop_gpv_const2019_2019", TOP_N).copy()
        top_gpv["label"] = top_gpv["country_name"].fillna(top_gpv["iso3"])
        top_gpv = top_gpv.sort_values("crop_gpv_const2019_2019", ascending=True)
    
        plt.figure(figsize=(12, 10))
        plt.barh(top_gpv["label"], top_gpv["crop_gpv_const2019_2019_million"])
        plt.xlabel(f"Crop production value ({MONEY_UNIT_LABEL})", fontsize=12)
        plt.title(f"Top {TOP_N}: Countries by crop production value", fontsize=16, pad=12)
        plt.grid(axis="x", alpha=0.25)
        savefig(FIG_DIR / "fig7_top20_crop_gpv_2019usd_million.png", dpi=300)
    
    # 4.8 On-farm vs upstream standalone
    if {"gep_const2019_usd_onfarm_million", "gep_const2019_usd_upstream_million"}.issubset(df.columns):
        top_cmp = top_n(df, "gep_const2019_usd_combined", TOP_N).copy()
        top_cmp["label"] = top_cmp["country_name"].fillna(top_cmp["iso3"])
        top_cmp = top_cmp.sort_values("gep_const2019_usd_combined", ascending=True)
    
        y = np.arange(len(top_cmp))
        h = 0.38
    
        plt.figure(figsize=(12, 10))
        plt.barh(y - h/2, top_cmp["gep_const2019_usd_onfarm_million"].fillna(0.0), height=h, label="On-farm")
        plt.barh(y + h/2, top_cmp["gep_const2019_usd_upstream_million"].fillna(0.0), height=h, label="Upstream")
        plt.yticks(y, top_cmp["label"])
        plt.xlabel(f"GEP ({MONEY_UNIT_LABEL})", fontsize=12)
        plt.title(f"Top {TOP_N}: Standalone On-farm vs Upstream GEP", fontsize=16, pad=12)
        plt.grid(axis="x", alpha=0.25)
        plt.legend(frameon=True)
        savefig(FIG_DIR / "fig8_top20_onfarm_vs_upstream_2019usd_million.png", dpi=300)
    
    
    # =============================================================================
    # 5) HISTOGRAMS
    # =============================================================================
    
    if "share_protected_production_combined" in df.columns:
        m = np.isfinite(df["share_protected_production_combined"])
        plt.figure(figsize=(9, 6))
        plt.hist(df.loc[m, "share_protected_production_combined"].clip(lower=0, upper=1), bins=30)
        plt.xlabel("Share of protected production (combined)", fontsize=12)
        plt.ylabel("Number of countries", fontsize=12)
        plt.title("Distribution of share of protected production (combined)", fontsize=16, pad=12)
        plt.grid(alpha=0.25)
        savefig(FIG_DIR / "hist_share_protected_production_combined.png", dpi=300)
    
    if "erosion_shock_share_combined" in df.columns:
        m = np.isfinite(df["erosion_shock_share_combined"])
        plt.figure(figsize=(9, 6))
        plt.hist(df.loc[m, "erosion_shock_share_combined"].clip(lower=0), bins=30)
        plt.xlabel("Erosion shock share (combined)", fontsize=12)
        plt.ylabel("Number of countries", fontsize=12)
        plt.title("Distribution of erosion shock shares (combined)", fontsize=16, pad=12)
        plt.grid(alpha=0.25)
        savefig(FIG_DIR / "hist_erosion_shock_share_combined.png", dpi=300)
    
    if "overlap_pct_of_sum_components" in df.columns:
        m = np.isfinite(df["overlap_pct_of_sum_components"])
        plt.figure(figsize=(9, 6))
        plt.hist(df.loc[m, "overlap_pct_of_sum_components"], bins=30)
        plt.xlabel("Overlap as % of (On-farm + Upstream)", fontsize=12)
        plt.ylabel("Number of countries", fontsize=12)
        plt.title("Distribution of overlap removed by union-of-protection", fontsize=16, pad=12)
        plt.grid(alpha=0.25)
        savefig(FIG_DIR / "hist_overlap_pct_of_sum.png", dpi=300)
    
    
    # =============================================================================
    # 6) SCATTERS
    # =============================================================================
    
    # 6.1 Combined GEP vs Crop GPV
    if {"crop_gpv_const2019_2019_million", "gep_const2019_usd_combined_million"}.issubset(df.columns):
        m = (
            np.isfinite(df["crop_gpv_const2019_2019"]) &
            np.isfinite(df["gep_const2019_usd_combined"]) &
            (df["crop_gpv_const2019_2019"] > 0) &
            (df["gep_const2019_usd_combined"] > 0)
        )
        d = df[m].copy()
    
        plt.figure(figsize=(9, 7))
        plt.scatter(
            d["crop_gpv_const2019_2019_million"],
            d["gep_const2019_usd_combined_million"],
            s=18
        )
        plt.xlabel(f"Crop GPV ({MONEY_UNIT_LABEL})", fontsize=12)
        plt.ylabel(f"Combined GEP ({MONEY_UNIT_LABEL})", fontsize=12)
        plt.title("Combined GEP vs Crop GPV (log-log)", fontsize=16, pad=12)
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(alpha=0.25)
        savefig(FIG_DIR / "scatter_combined_gep_vs_crop_gpv_loglog_2019usd_million.png", dpi=300)
    
    # 6.2 Combined GEP vs GDP with labels
    if {"gdp_const2019_2019", "gep_const2019_usd_combined"}.issubset(df.columns):
        d = df.copy()
        mask = (
            np.isfinite(d["gdp_const2019_2019"]) &
            np.isfinite(d["gep_const2019_usd_combined"]) &
            (d["gdp_const2019_2019"] > 0) &
            (d["gep_const2019_usd_combined"] > 0)
        )
        d = d.loc[mask].copy()
    
        if len(d) > 0:
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.scatter(
                d["gdp_const2019_2019"],
                d["gep_const2019_usd_combined"],
                s=28,
                alpha=0.75,
                edgecolors="white",
                linewidths=0.3
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("GDP 2019 (USD, log scale)")
            ax.set_ylabel("Combined GEP (USD, log scale)")
            ax.set_title("Combined GEP vs GDP (log-log)")
    
            label_subset = d.sort_values("gep_const2019_usd_combined", ascending=False).head(TOP_N_LABELS)
            for _, r in label_subset.iterrows():
                ax.text(
                    r["gdp_const2019_2019"] * 1.03,
                    r["gep_const2019_usd_combined"] * 1.03,
                    str(r["country_name"])[:18],
                    fontsize=7,
                    color="gray",
                    alpha=0.85
                )
    
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            diag_min = max(xmin, ymin)
            diag_max = min(xmax, ymax)
            if diag_max > diag_min:
                ax.plot([diag_min, diag_max], [diag_min, diag_max], linestyle="--", linewidth=0.8, color="black", alpha=0.4)
    
            savefig(FIG_DIR / "scatter_combined_gep_vs_gdp_log_countrynames.png", dpi=300)
    
    # 6.3 Income group scatter plots
    income_map = {
        "AUS":"High income","AUT":"High income","BEL":"High income","CAN":"High income","CHE":"High income",
        "CHL":"High income","CZE":"High income","DEU":"High income","DNK":"High income","ESP":"High income",
        "EST":"High income","FIN":"High income","FRA":"High income","GBR":"High income","GRC":"High income",
        "HKG":"High income","HRV":"High income","HUN":"High income","IRL":"High income","ISL":"High income",
        "ISR":"High income","ITA":"High income","JPN":"High income","KOR":"High income","LTU":"High income",
        "LUX":"High income","LVA":"High income","NLD":"High income","NOR":"High income","NZL":"High income",
        "POL":"High income","PRT":"High income","SGP":"High income","SVN":"High income","SWE":"High income",
        "USA":"High income",
    
        "ARG":"Upper middle income","BRA":"Upper middle income","CHN":"Upper middle income","COL":"Upper middle income",
        "CRI":"Upper middle income","DOM":"Upper middle income","GAB":"Upper middle income","IDN":"Upper middle income",
        "IRN":"Upper middle income","KAZ":"Upper middle income","MEX":"Upper middle income","MYS":"Upper middle income",
        "PER":"Upper middle income","SRB":"Upper middle income","THA":"Upper middle income","TUR":"Upper middle income",
        "ZAF":"Upper middle income","BGR":"Upper middle income","JOR":"Upper middle income","PRY":"Upper middle income",
        "ECU":"Upper middle income","VNM":"Upper middle income","BOL":"Upper middle income","ALB":"Upper middle income",
    
        "BGD":"Lower middle income","CIV":"Lower middle income","CMR":"Lower middle income","COD":"Lower middle income",
        "EGY":"Lower middle income","GHA":"Lower middle income","IND":"Lower middle income","KEN":"Lower middle income",
        "MAR":"Lower middle income","MNG":"Lower middle income","NGA":"Lower middle income","PAK":"Lower middle income",
        "PHL":"Lower middle income","SLV":"Lower middle income","SEN":"Lower middle income","TZA":"Lower middle income",
        "UKR":"Lower middle income","UZB":"Lower middle income","VUT":"Lower middle income","LAO":"Lower middle income",
        "PNG":"Lower middle income","DJI":"Lower middle income","HND":"Lower middle income","NIC":"Lower middle income",
        "BTN":"Lower middle income","KHM":"Lower middle income","LKA":"Lower middle income","ZMB":"Lower middle income",
        "AGO":"Lower middle income","NAM":"Lower middle income",
    
        "AFG":"Low income","BFA":"Low income","BDI":"Low income","CAF":"Low income","ETH":"Low income","GMB":"Low income",
        "GIN":"Low income","GNB":"Low income","LBR":"Low income","MDG":"Low income","MLI":"Low income","MOZ":"Low income",
        "MWI":"Low income","NER":"Low income","NPL":"Low income","RWA":"Low income","SLE":"Low income","SOM":"Low income",
        "SSD":"Low income","SYR":"Low income","TCD":"Low income","TGO":"Low income","UGA":"Low income","YEM":"Low income",
        "ZWE":"Low income",
    }
    
    income_colors = {
        "Low income": "#d73027",
        "Lower middle income": "#fc8d59",
        "Upper middle income": "#fee08b",
        "High income": "#1a9850",
    }
    
    if {"gdp_const2019_2019", "gep_const2019_usd_combined"}.issubset(df.columns):
        d0 = df.copy()
        d0["income_group"] = d0["iso3"].map(income_map)
        d0 = d0.dropna(subset=["income_group"]).copy()
    
        mask = (
            np.isfinite(d0["gdp_const2019_2019"]) &
            np.isfinite(d0["gep_const2019_usd_combined"]) &
            (d0["gdp_const2019_2019"] > 0) &
            (d0["gep_const2019_usd_combined"] > 0)
        )
        d = d0.loc[mask].copy()
    
        if len(d) > 0:
            order = ["Low income", "Lower middle income", "Upper middle income", "High income"]
    
            # Log-log
            fig, ax = plt.subplots(figsize=(10, 7))
            for group in order:
                subset = d[d["income_group"] == group]
                if subset.empty:
                    continue
                ax.scatter(
                    subset["gdp_const2019_2019"],
                    subset["gep_const2019_usd_combined"],
                    s=30,
                    alpha=0.78,
                    edgecolors="white",
                    linewidths=0.4,
                    label=group,
                    color=income_colors.get(group, "gray")
                )
    
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("GDP 2019 (USD, log scale)")
            ax.set_ylabel("Combined GEP (USD, log scale)")
            ax.set_title("Combined GEP vs GDP (log-log), by income group")
    
            label_subset = d.sort_values("gep_const2019_usd_combined", ascending=False).head(TOP_N_LABELS)
            for _, r in label_subset.iterrows():
                ax.text(
                    r["gdp_const2019_2019"] * 1.03,
                    r["gep_const2019_usd_combined"] * 1.03,
                    str(r["country_name"])[:18],
                    fontsize=7, color="gray", alpha=0.85
                )
    
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            diag_min = max(xmin, ymin)
            diag_max = min(xmax, ymax)
            if diag_max > diag_min:
                ax.plot([diag_min, diag_max], [diag_min, diag_max], "k--", lw=0.8, alpha=0.4)
    
            ax.legend(title="Income Group", fontsize=8, title_fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
            plt.tight_layout()
            plt.savefig(FIG_DIR / "scatter_combined_gep_vs_gdp_log_income_groups.png", dpi=300, bbox_inches="tight")
            plt.close()
    
            # Linear capped
            CAP_AT_PCTL = 99
            x_cap = np.nanpercentile(d["gdp_const2019_2019"], CAP_AT_PCTL)
            y_cap = np.nanpercentile(d["gep_const2019_usd_combined"], CAP_AT_PCTL)
    
            fig, ax = plt.subplots(figsize=(12, 7))
            for group in order:
                subset = d[d["income_group"] == group]
                if subset.empty:
                    continue
                ax.scatter(
                    subset["gdp_const2019_2019"],
                    subset["gep_const2019_usd_combined"],
                    s=30,
                    alpha=0.78,
                    edgecolors="white",
                    linewidths=0.4,
                    label=group,
                    color=income_colors.get(group, "gray")
                )
    
            ax.set_xlabel("GDP 2019 (USD, linear)")
            ax.set_ylabel("Combined GEP (USD, linear)")
            ax.set_title(f"Combined GEP vs GDP (linear), by income group (axes capped at p{CAP_AT_PCTL})")
            ax.set_xlim(0, x_cap)
            ax.set_ylim(0, y_cap)
    
            label_subset = d.sort_values("gep_const2019_usd_combined", ascending=False).head(TOP_N_LABELS)
            for _, r in label_subset.iterrows():
                if r["gdp_const2019_2019"] <= x_cap and r["gep_const2019_usd_combined"] <= y_cap:
                    ax.text(
                        r["gdp_const2019_2019"] * 1.01,
                        r["gep_const2019_usd_combined"] * 1.01,
                        str(r["country_name"])[:18],
                        fontsize=7, color="gray", alpha=0.85
                    )
    
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            diag_min = max(xmin, ymin)
            diag_max = min(xmax, ymax)
            if diag_max > diag_min:
                ax.plot([diag_min, diag_max], [diag_min, diag_max], "k--", lw=0.8, alpha=0.4)
    
            ax.legend(title="Income Group", fontsize=8, title_fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
            fig.subplots_adjust(right=0.78)
            plt.savefig(FIG_DIR / "scatter_combined_gep_vs_gdp_linear_income_groups.png", dpi=300, bbox_inches="tight")
            plt.close()
    
    
    # =============================================================================
    # 7) GLOBAL CROP-LEVEL FIGURE
    # =============================================================================
    if df_crop_long is not None:
        needed = {"component", "crop", "protected_production_tons"}
        if needed.issubset(df_crop_long.columns):
            dcc = df_crop_long.copy()
            dcc["protected_production_tons"] = pd.to_numeric(dcc["protected_production_tons"], errors="coerce")
    
            dcc_comb = dcc[dcc["component"].astype(str).str.lower() == "combined"].copy()
            top_crop = (
                dcc_comb.groupby("crop", as_index=False)["protected_production_tons"]
                .sum()
                .sort_values("protected_production_tons", ascending=False)
                .head(TOP_N)
                .copy()
            )
    
            if len(top_crop) > 0:
                top_crop = top_crop.sort_values("protected_production_tons", ascending=True)
    
                plt.figure(figsize=(11, 8))
                plt.barh(top_crop["crop"], top_crop["protected_production_tons"])
                plt.xlabel("Protected production (tons)")
                plt.title(f"Top {TOP_N} crops by nature protected production (combined)", fontsize=16, pad=12)
                plt.grid(axis="x", alpha=0.25)
                savefig(FIG_DIR / "bar_top20_crops_protected_tons_combined.png", dpi=300)
    
    
    # =============================================================================
    # 8) CHOROPLETH MAPS
    # =============================================================================
    
    # Monetary maps
    plot_publication_choropleth_categorical(
        g, "gep_const2019_usd_combined",
        "Combined GEP from severe erosion protection",
        FIG_DIR / "map1_country_combined_gep_5class_2019usd_million.png",
        f"Combined GEP ({MONEY_UNIT_LABEL})",
        scheme="fisher_jenks", k=MAP_K_CLASSES, value_unit="usd_millions", label_format="usd_millions"
    )
    
    plot_publication_choropleth_categorical(
        g, "gep_const2019_usd_onfarm",
        "On-farm GEP from severe erosion protection",
        FIG_DIR / "map2_country_onfarm_gep_5class_2019usd_million.png",
        f"On-farm GEP ({MONEY_UNIT_LABEL})",
        scheme="fisher_jenks", k=MAP_K_CLASSES, value_unit="usd_millions", label_format="usd_millions"
    )
    
    plot_publication_choropleth_categorical(
        g, "gep_const2019_usd_upstream",
        "Upstream GEP from severe erosion protection",
        FIG_DIR / "map3_country_upstream_gep_5class_2019usd_million.png",
        f"Upstream GEP ({MONEY_UNIT_LABEL})",
        scheme="fisher_jenks", k=MAP_K_CLASSES, value_unit="usd_millions", label_format="usd_millions"
    )
    
    plot_publication_choropleth_categorical(
        g, "gep_const2019_usd_overlap",
        "Overlap removed = On-farm + Upstream - Combined",
        FIG_DIR / "map4_country_overlap_5class_2019usd_million.png",
        f"Overlap ({MONEY_UNIT_LABEL})",
        scheme="fisher_jenks", k=MAP_K_CLASSES, value_unit="usd_millions", label_format="usd_millions"
    )
    
    plot_publication_choropleth_categorical(
        g, "crop_gpv_const2019_2019",
        "Total crop production value (FAO 2019)",
        FIG_DIR / "map5_country_crop_gpv_5class_2019usd_million.png",
        f"Crop GPV ({MONEY_UNIT_LABEL})",
        scheme="fisher_jenks", k=MAP_K_CLASSES, value_unit="usd_millions", label_format="usd_millions"
    )
    
    # Shares / percentages
    plot_publication_choropleth_categorical(
        g, "overlap_pct_of_sum_components",
        "Overlap as % of (On-farm + Upstream)",
        FIG_DIR / "map6_country_overlap_pct_5class.png",
        "Overlap (% of On-farm + Upstream)",
        scheme="equal_interval", k=MAP_K_CLASSES, value_unit="raw", label_format="percent"
    )
    
    plot_publication_choropleth_categorical(
        g, "gdp_loss_pct_combined",
        "Combined GEP as % of GDP (indicative macro exposure)",
        FIG_DIR / "map7_country_gdp_loss_pct_combined_5class.png",
        "Combined GEP / GDP (%)",
        scheme="equal_interval", k=MAP_K_CLASSES, value_unit="raw", label_format="percent"
    )
    
    plot_publication_choropleth_categorical(
        g, "share_protected_production_combined",
        "Share of protected production (combined)",
        FIG_DIR / "map8_country_share_protected_combined_5class.png",
        "Share protected production",
        scheme="equal_interval", k=MAP_K_CLASSES, value_unit="raw", label_format="percent"
    )
    
    plot_publication_choropleth_categorical(
        g, "share_protected_production_onfarm",
        "Share of protected production (on-farm)",
        FIG_DIR / "map9_country_share_protected_onfarm_5class.png",
        "Share protected production",
        scheme="equal_interval", k=MAP_K_CLASSES, value_unit="raw", label_format="percent"
    )
    
    plot_publication_choropleth_categorical(
        g, "share_protected_production_upstream",
        "Share of protected production (upstream)",
        FIG_DIR / "map10_country_share_protected_upstream_5class.png",
        "Share protected production",
        scheme="equal_interval", k=MAP_K_CLASSES, value_unit="raw", label_format="percent"
    )
    
    plot_publication_choropleth_categorical(
        g, "erosion_shock_share_combined",
        "Erosion shock share (combined)",
        FIG_DIR / "map11_country_erosion_shock_share_combined_5class.png",
        "Shock share",
        scheme="equal_interval", k=MAP_K_CLASSES, value_unit="raw", label_format="percent"
    )
    
    # Mean PS maps
    plot_publication_choropleth_categorical(
        g, "mean_ps_onfarm_cropland_severe",
        "Mean prevention share on cropland severe pixels (on-farm)",
        FIG_DIR / "map12_country_mean_ps_onfarm_5class.png",
        "Mean PS_onfarm (0–1)",
        scheme="equal_interval", k=MAP_K_CLASSES, value_unit="raw", label_format="percent"
    )
    
    plot_publication_choropleth_categorical(
        g, "mean_ps_upstream_cropland_severe",
        "Mean prevention share on cropland severe pixels (upstream)",
        FIG_DIR / "map13_country_mean_ps_upstream_5class.png",
        "Mean PS_upstream (0–1)",
        scheme="equal_interval", k=MAP_K_CLASSES, value_unit="raw", label_format="percent"
    )
    
    plot_publication_choropleth_categorical(
        g, "mean_ps_combined_cropland_severe",
        "Mean prevention share on cropland severe pixels (combined)",
        FIG_DIR / "map14_country_mean_ps_combined_5class.png",
        "Mean PS_combined (0–1)",
        scheme="equal_interval", k=MAP_K_CLASSES, value_unit="raw", label_format="percent"
    )
    
    # Log10 combined GEP map
    if "gep_const2019_usd_combined" in g.columns:
        g_log = g.copy()
        g_log["log10_gep_million_usd_combined"] = np.log10(
            (pd.to_numeric(g_log["gep_const2019_usd_combined"], errors="coerce") / USD_TO_MILLIONS)
            .where(pd.to_numeric(g_log["gep_const2019_usd_combined"], errors="coerce") > 0)
        )
        plot_publication_choropleth_categorical(
            g_log, "log10_gep_million_usd_combined",
            "Combined GEP (log10 USD million)",
            FIG_DIR / "map15_country_log10_combined_gep_5class.png",
            "log10(USD million)",
            scheme="fisher_jenks", k=MAP_K_CLASSES, value_unit="raw", label_format="percent"
        )
    
    
    # =============================================================================
    # 9) RASTER PREVIEWS
    # =============================================================================
    if PS_ONFARM_TIF.exists():
        plot_raster_global(
            PS_ONFARM_TIF,
            "PS_onfarm on cropland & severe",
            FIG_DIR / "raster1_ps_onfarm_cropland_severe.png",
            downsample_factor=RASTER_DOWNSAMPLE_FACTOR,
        )
    
    if PS_UPSTREAM_TIF.exists():
        plot_raster_global(
            PS_UPSTREAM_TIF,
            "PS_upstream on cropland & severe",
            FIG_DIR / "raster2_ps_upstream_cropland_severe.png",
            downsample_factor=RASTER_DOWNSAMPLE_FACTOR,
        )
    
    if PS_COMBINED_TIF.exists():
        plot_raster_global(
            PS_COMBINED_TIF,
            "PS_combined (union-of-protection) on cropland & severe",
            FIG_DIR / "raster3_ps_combined_union_cropland_severe.png",
            downsample_factor=RASTER_DOWNSAMPLE_FACTOR,
        )
    
    
    # =============================================================================
    # 10) SUMMARY
    # =============================================================================
    print(f"✅ Done. Figures saved to: {FIG_DIR}")
    print("Created files:")
    for fp in sorted(FIG_DIR.glob("*")):
        if fp.suffix.lower() in {".png", ".csv"}:
            print(" -", fp.name)

def configure_maps(p):
    """
    Override the Section-C (maps & figures) module-level constants from the
    ProjectFlow object. Also pushes EXCLUDE_ISO3 / ROBINSON_CRS /
    USD_TO_MILLIONS / TOP_N as module globals, since
    plot_publication_choropleth_categorical() (which lives there) reads
    those as module globals. Called by
    erosion_tasks.maps_and_figures() before
    generate_all_maps_and_figures().

    NOTE (organizational quirk to flag to Justin/Chiara): the original
    combined_maps_figures notebook pointed RUN_DIR at
    'output_eps5_onfarm_upstream_combined_20260305', while the
    Combine_PS_SES notebook that produces those outputs wrote to
    'output_ses11_onfarm_upstream_combined_20260305' (RUN_TAG mismatch,
    "eps5" vs "ses11"). p.erosion_run_tag below defaults to the
    Combine_PS_SES value so the two stages line up; double check this
    against whichever run you actually want to map.
    """
    global ROOT, RUN_DIR, FIG_DIR, INTEGRATED_CSV, COUNTRY_CROP_LONG_CSV
    global PS_ONFARM_TIF, PS_UPSTREAM_TIF, PS_COMBINED_TIF, RUN_BOUNDARY_GPKG
    global TOP_N_LABELS, RASTER_DOWNSAMPLE_FACTOR, MAP_K_CLASSES, MONEY_UNIT_LABEL

    ROOT = Path(getattr(p, 'erosion_gep_root', ROOT))
    run_tag = getattr(p, 'erosion_run_tag', 'ses11_onfarm_upstream_combined_20260305')
    RUN_DIR = Path(getattr(p, 'erosion_gep_output_dir', ROOT / f"output_{run_tag}"))

    FIG_DIR = Path(getattr(p, 'erosion_figures_dir', RUN_DIR / "figures"))
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    INTEGRATED_CSV = Path(getattr(p, 'erosion_integrated_country_gep_csv', RUN_DIR / "integrated_country_gep.csv"))
    COUNTRY_CROP_LONG_CSV = Path(getattr(p, 'erosion_country_crop_long_csv', RUN_DIR / "country_crop_protected_production_long.csv"))

    PS_ONFARM_TIF = Path(getattr(p, 'erosion_ps_onfarm_tif', RUN_DIR / "ps_onfarm_cropland_severe.tif"))
    PS_UPSTREAM_TIF = Path(getattr(p, 'erosion_ps_upstream_tif', RUN_DIR / "ps_upstream_cropland_severe.tif"))
    PS_COMBINED_TIF = Path(getattr(p, 'erosion_ps_combined_tif', RUN_DIR / "ps_combined_union_cropland_severe.tif"))

    RUN_BOUNDARY_GPKG = Path(getattr(p, 'erosion_country_boundary_path', ROOT / "inputs_v2" / "erosion_gep" / "country_boundary_r250_with_iso3.gpkg"))

    TOP_N_LABELS = getattr(p, 'erosion_top_n_labels', 25)
    RASTER_DOWNSAMPLE_FACTOR = getattr(p, 'erosion_raster_downsample_factor', 6)
    MAP_K_CLASSES = getattr(p, 'erosion_map_k_classes', 5)
    MONEY_UNIT_LABEL = getattr(p, 'erosion_money_unit_label', "2019 USD million")

    # Shared plotting constants (module globals since the utils fold).
    global EXCLUDE_ISO3, ROBINSON_CRS, USD_TO_MILLIONS, TOP_N
    EXCLUDE_ISO3 = getattr(p, 'erosion_exclude_iso3', {"ATA"})
    ROBINSON_CRS = getattr(p, 'erosion_robinson_crs', "+proj=robin")
    USD_TO_MILLIONS = getattr(p, 'erosion_usd_to_millions', 1e6)
    TOP_N = getattr(p, 'erosion_top_n', 20)

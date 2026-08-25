"""The erosion account's science, as functions over arrays and frames.

Nothing here opens a file. The account values the crop production that soil retention protects,
in four steps, and each one is a function below:

1. A country's severity threshold. Soil loss counts as severe above a tolerance rate, which is
   lower for small and low-lying countries because the default rate is calibrated on deep upland
   soils (`country_threshold_policy`).
2. The share of gross soil loss that is prevented, per pixel. On-farm cover prevents
   `avoided / (avoided + gross)`; the upstream layer prevents its own share; the two combine as a
   union rather than a sum, so a pixel protected twice is not counted twice
   (`onfarm_prevention_share`, `combined_prevention_share`, `restrict_to_valued_pixels`).
3. A country's production shock, as the production-weighted mean of each crop's protected share
   times that crop's yield elasticity to erosion (`country_erosion_shock`).
4. The value, as the country's crop gross production value times that shock (`country_gep`).

The threshold and prevention steps run over rasters the module reads elsewhere; the shock and
value steps run over frames. Splitting them out is what lets the arithmetic be tested on four
countries instead of on a global grid.

The steps that open rasters and write tables, including the InVEST SDR run, the D8
routing and the figure driver, are in the task module.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import os
import sys
import time
import logging
import warnings
import geopandas as gpd
import rasterio
import rasterio.features
import xarray as xr
from rasterio.enums import Resampling
from rasterio.crs import CRS as rioCRS


# Shocks below this are numerical residue rather than economics, but a country with a genuinely
# tiny protected share should not fall to exactly zero and drop out of the account.
MIN_SHOCK_FLOOR = 8e-10


def country_threshold_policy(df_countries, threshold_high, threshold_low,
                             small_country_area_km2, low_elevation_mean_m):
    """Each country's soil-loss tolerance rate, and why it got the one it got.

    The high rate is the default. A country takes the low rate if it is small or if its mean
    elevation is low, because the high rate assumes deep soils on upland slopes.

    Args:
        df_countries (pandas.DataFrame): one row per country, with `iso3`, `area_km2` and
            `mean_elevation_m`. Either measure may be missing, in which case it cannot trigger
            the low rate.
        threshold_high (float): the default tolerance, t/ha/yr.
        threshold_low (float): the tolerance for small and low-lying countries, t/ha/yr.
        small_country_area_km2 (float): area below which a country is small.
        low_elevation_mean_m (float): mean elevation below which a country is low-lying.

    Returns:
        pandas.DataFrame: `iso3`, `threshold_t_ha_yr`, `reason`.
    """
    df = df_countries.copy()
    area = pd.to_numeric(df.get('area_km2'), errors='coerce')
    elevation = pd.to_numeric(df.get('mean_elevation_m'), errors='coerce')

    is_small = area.notna() & (area < small_country_area_km2)
    is_low = elevation.notna() & np.isfinite(elevation) & (elevation < low_elevation_mean_m)

    reasons = []
    for small, low in zip(is_small, is_low):
        parts = (['small-area'] if small else []) + (['low-elevation'] if low else [])
        reasons.append(' & '.join(parts) if parts else 'default-high')

    return pd.DataFrame({
        'iso3': df['iso3'].to_numpy(),
        'threshold_t_ha_yr': np.where(is_small | is_low, threshold_low, threshold_high),
        'reason': reasons,
    })


def onfarm_prevention_share(avoided_erosion, gross_erosion, epsilon=1e-9):
    """The share of gross soil loss that the field's own cover prevents.

    `avoided / (avoided + gross)`, which is a rate rather than a tonnage, so it can be applied to
    production without carrying the pixel's area through.

    Args:
        avoided_erosion (numpy.ndarray): soil loss avoided by present cover, t/ha/yr.
        gross_erosion (numpy.ndarray): soil loss that still occurs, t/ha/yr.
        epsilon (float): floor on the denominator, so a pixel with neither reads as zero
            prevention rather than dividing by zero.

    Returns:
        numpy.ndarray: the share, in [0, 1].
    """
    denominator = np.maximum(np.asarray(avoided_erosion) + np.asarray(gross_erosion), epsilon)
    return np.clip(np.asarray(avoided_erosion) / denominator, 0.0, 1.0)


def combined_prevention_share(onfarm_share, upstream_share):
    """On-farm and upstream prevention combined as a union, not a sum.

    `1 - (1 - onfarm)(1 - upstream)`: the soil that gets through both. Summing the two shares
    would double-count the loss that either alone would have prevented, and could exceed 1.

    Returns:
        numpy.ndarray: the combined share, in [0, 1].
    """
    combined = 1.0 - (1.0 - np.asarray(onfarm_share)) * (1.0 - np.asarray(upstream_share))
    return np.clip(combined, 0.0, 1.0)


def restrict_to_valued_pixels(share, is_cropland, is_severe):
    """Zero a prevention share wherever the account does not value it.

    The account values prevention only on cropland, and only where soil loss exceeds the
    country's tolerance, so prevention of a loss the soil could absorb is not counted.

    Returns:
        numpy.ndarray: the share where both hold, zero elsewhere.
    """
    return np.where(np.asarray(is_cropland) & np.asarray(is_severe), np.asarray(share), 0.0)


def country_erosion_shock(df_country_crop, min_shock_floor=MIN_SHOCK_FLOOR):
    """A country's crop production shock, from its per-crop protected shares.

    The shock is the production-weighted mean over crops of `protected share x elasticity`, so a
    crop that is a large part of the country's output moves the shock more than a small one, and a
    crop whose yield barely responds to soil loss contributes little however well protected it is.
    Both factors are clipped into [0, 1]: an elasticity above 1 would claim yield falls faster
    than soil is lost.

    Args:
        df_country_crop (pandas.DataFrame): one row per country and crop, with `ISO3`,
            `protected_production_tons`, `total_production_tons`, `share_protected_production`
            and `elasticity_used`.
        min_shock_floor (float): shocks between zero and this floor to it.

    Returns:
        pandas.DataFrame: `iso3`, `protected_production_tons`, `total_production_tons`,
        `share_protected_production` and `erosion_shock_share`. A country with no production
        carries a missing share and a missing shock rather than a zero, because there is nothing
        to take a share of.
    """
    df = df_country_crop.copy()

    physical = df.groupby('ISO3', as_index=False)[
        ['protected_production_tons', 'total_production_tons']].sum()
    physical['share_protected_production'] = np.where(
        physical['total_production_tons'] > 0,
        np.clip(physical['protected_production_tons'] / physical['total_production_tons'], 0.0, 1.0),
        np.nan)
    physical = physical.rename(columns={'ISO3': 'iso3'})

    for column in ['total_production_tons', 'share_protected_production', 'elasticity_used']:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    df['weight'] = df['total_production_tons'].clip(lower=0.0)
    df['term'] = (df['weight']
                  * df['share_protected_production'].clip(0.0, 1.0)
                  * df['elasticity_used'].clip(0.0, 1.0))

    weighted = df.groupby('ISO3', as_index=False).agg(
        weight_sum=('weight', 'sum'), term_sum=('term', 'sum'))
    weighted['erosion_shock_share'] = np.where(
        weighted['weight_sum'] > 0,
        (weighted['term_sum'] / weighted['weight_sum']).clip(0.0, 1.0),
        np.nan)

    tiny = (weighted['erosion_shock_share'].notna()
            & (weighted['erosion_shock_share'] > 0)
            & (weighted['erosion_shock_share'] < min_shock_floor))
    weighted.loc[tiny, 'erosion_shock_share'] = min_shock_floor
    weighted = weighted.rename(columns={'ISO3': 'iso3'})

    return physical.merge(weighted[['iso3', 'erosion_shock_share']], on='iso3', how='left')


def country_gep(df_shock, df_crop_gpv, df_gdp, component):
    """The value of the protected production, and what it is as a share of GDP.

    Args:
        df_shock (pandas.DataFrame): `country_erosion_shock`'s output.
        df_crop_gpv (pandas.DataFrame): `iso3` and `crop_gpv_const2019_2019`, the country's crop
            gross production value in constant 2019 dollars.
        df_gdp (pandas.DataFrame): `iso3` and `gdp_const2019_2019`.
        component (str): which prevention channel this is, carried onto every row so the on-farm,
            upstream and combined runs can be concatenated and still told apart.

    Returns:
        pandas.DataFrame: the shock columns plus `crop_gpv_const2019_2019`,
        `gdp_const2019_2019`, `gep_const2019_usd` and `gdp_loss_pct`.
    """
    out = df_shock.merge(df_crop_gpv, on='iso3', how='left').merge(df_gdp, on='iso3', how='left')
    out['gep_const2019_usd'] = (out['crop_gpv_const2019_2019'].fillna(0.0)
                                * out['erosion_shock_share'].fillna(0.0))
    out['gdp_loss_pct'] = np.where(
        out['gdp_const2019_2019'].notna() & (out['gdp_const2019_2019'] > 0),
        100.0 * out['gep_const2019_usd'] / out['gdp_const2019_2019'],
        np.nan)
    out['component'] = component
    return out[['component', 'iso3', 'protected_production_tons', 'total_production_tons',
                'share_protected_production', 'erosion_shock_share', 'crop_gpv_const2019_2019',
                'gdp_const2019_2019', 'gep_const2019_usd', 'gdp_loss_pct']]


def upstream_prevention_share(accumulated_avoided, accumulated_potential, ndv=-9999.0):
    """The share of soil loss that upslope land cover prevents, at each pixel.

    Both arguments are flow-accumulated down the drainage network, so each pixel carries what its
    whole catchment contributes: `accumulated_avoided` the soil that upslope cover held back, and
    `accumulated_potential` what bare soil would have lost. Their ratio is a share, so the pixel
    area cancels and the result can be combined with the on-farm share directly.

    Args:
        accumulated_avoided (numpy.ndarray): flow-accumulated avoided erosion.
        accumulated_potential (numpy.ndarray): flow-accumulated potential (bare-soil) erosion.
        ndv (float): what to write where nothing drains, so a ridge pixel reads as no data rather
            than as no prevention.

    Returns:
        numpy.ndarray: the share in [0, 1], or `ndv` where there is no potential erosion to hold.
    """
    accumulated_avoided = np.asarray(accumulated_avoided, dtype='float64')
    accumulated_potential = np.asarray(accumulated_potential, dtype='float64')
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(accumulated_potential > 0,
                        np.clip(accumulated_avoided / accumulated_potential, 0.0, 1.0),
                        ndv).astype('float32')



def _required_path(p, attribute, constant_name):
    """The project's value for a path, or a failure that names what is missing.

    These used to read `getattr(p, attribute, SOME_CONSTANT)`, where the constant was an absolute
    path on the machine the source scripts were written on. A project that did not set the
    attribute therefore fell back to a directory that does not exist here, and the run failed
    later with a missing-file error naming somebody else's home directory. Failing here instead
    names the parameter to add to es_parameters.csv.

    Args:
        p (ProjectFlow): the project.
        attribute (str): the attribute the project should carry.
        constant_name (str): what this path is, for the error message.

    Returns:
        str: the resolved path.

    Raises:
        NameError: when the project does not carry the attribute.
    """
    value = getattr(p, attribute, None)
    if value is None:
        raise NameError('erosion needs %s (%s). Add a row for it to es_parameters.csv; it used to '
                        'default to a path on the machine the source scripts came from.'
                        % (attribute, constant_name))
    return str(value)

def _output_path(p, attribute, default_name):
    """Where the run writes something, from the project or under the task's own directory.

    An output is not an input: the project need not be told where to put it, only where to find
    what it reads. So this defaults into the task directory rather than raising the way
    _required_path does, and a project that wants the file somewhere else still sets the
    attribute. Before this, the defaults were absolute paths on the machines the source scripts
    came from, so outputs of a run here were addressed to a cluster.

    Args:
        p (ProjectFlow): the project.
        attribute (str): the attribute a project may set to override the default.
        default_name (str): the file or directory name under the task directory.

    Returns:
        str: where to write.
    """
    value = getattr(p, attribute, None)
    if value is not None:
        return str(value)
    directory = getattr(p, 'cur_dir', None) or getattr(p, 'project_dir', None) or '.'
    return os.path.join(str(directory), default_name)


def configure_sdr(p):
    """
    Override the Section-A (InVEST SDR) module-level path constants from the
    ProjectFlow object, if the project set them. Called by
    erosion_tasks.invest_sdr() before run_invest_sdr().
    """
    global ROOT, BASE_IN, BASE_OUT, BIOPHYS_CSV, DEM_TIF, LULC_TIF, K_TIF, R_TIF
    global WATERSHEDS_RAW, WATERSHEDS_SANITIZED, WATERSHEDS_SAN_LAYER, DRAINAGE_PATH

    ROOT = _output_path(p, 'erosion_sdr_root', 'sdr')
    BASE_IN = _output_path(p, 'erosion_sdr_input_dir', 'sdr_input')
    BASE_OUT = _output_path(p, 'erosion_sdr_output_dir', 'sdr_output')
    os.makedirs(BASE_OUT, exist_ok=True)

    BIOPHYS_CSV = _required_path(p, 'erosion_biophysical_table_path', 'BIOPHYS_CSV')
    DEM_TIF = _required_path(p, 'erosion_dem_path', 'DEM_TIF')
    LULC_TIF = _required_path(p, 'erosion_lulc_path', 'LULC_TIF')
    K_TIF = _required_path(p, 'erosion_erodibility_path', 'K_TIF')
    R_TIF = _required_path(p, 'erosion_erosivity_path', 'R_TIF')
    WATERSHEDS_RAW = _required_path(p, 'erosion_watersheds_path', 'WATERSHEDS_RAW')
    WATERSHEDS_SANITIZED = _output_path(p, 'erosion_watersheds_sanitized_path', 'watersheds_sanitized.gpkg')
    WATERSHEDS_SAN_LAYER = getattr(p, 'erosion_watersheds_sanitized_layer', WATERSHEDS_SAN_LAYER)
    DRAINAGE_PATH = getattr(p, 'erosion_drainage_path', DRAINAGE_PATH)


# =============================================================================
# SECTION A — InVEST SDR run (from step1_sdr_invest_run.ipynb)
# =============================================================================
# 0) PATH CONFIG (CANONICAL)
# =============================================================================
# Set by configure_sdr / configure_prevention_shares / configure_maps from the project's
# es_parameters rows. None until then: these used to hold absolute paths on the machines the
# source scripts were written on, so a project that forgot a row ran against a directory that
# does not exist here. _required_path now names the missing row instead.
ROOT = None

BASE_IN = None
BASE_OUT = None

BIOPHYS_CSV = None
DEM_TIF = None
LULC_TIF = None
K_TIF = None
R_TIF = None

# ✅ Use MERGED watershed ONLY
WATERSHEDS_RAW = None

# Sanitized watersheds used to prevent overflow + CRS parse failures in report
WATERSHEDS_SANITIZED = None
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
    prefix = sys.prefix
    proj_lib = os.path.join(prefix, "share", "proj")
    gdal_data = os.path.join(prefix, "share", "gdal")

    if os.path.exists(proj_lib):
        os.environ["PROJ_LIB"] = str(proj_lib)
    if os.path.exists(gdal_data):
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


# =============================================================================
# 3) HELPERS
# =============================================================================
def _assert_exists(path, label: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[missing] {label}: {path}")


# =============================================================================
# 4) SDR ARGS (INVEST 3.17.x TEMPLATE)
# =============================================================================
def build_args(watersheds_path) -> dict:
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


# ==============================
# 1) CONFIG — EDIT AS NEEDED
# ==============================
SCENARIO_NAME = "SES — On-farm + Upstream (decomposed)"
RUN_TAG = "ses11_onfarm_upstream_combined_20260305"

# =============================================================================
# SECTION B -- On-farm + upstream erosion prevention-share GEP valuation
#              (from Combine_PS_SES11_3_3_2026.ipynb)
# =============================================================================
ROOT = None

# Primary erosion inputs (SDR-derived)
IN_DIR = None
USLE_PATH = None
AVOID_PATH = None

# Precomputed upstream prevention share raster (from your DEM+D8 workflow)
UPS_DIR = None
UPS_PATH = None

# Optional upstream LULC attribution shares (diagnostics only)
UPS_FOREST_SHARE = None
UPS_GRASS_SHARE = None
UPS_CROP_SHARE = None
UPS_BARE_SHARE = None
USE_UPSLOPE_LULC_ATTRIBUTION_DIAGNOSTICS = True

# Boundary with ISO3 column
BOUNDARY_GPKG = None
BOUNDARY_SOURCE_EPSG = None  # set to 4326 if you know it is WGS84

# Optional DEM used ONLY for threshold policy (low-elevation rule). If not present, rule is skipped.
ELEVATION_PATH = None

# Crops: SPAM2020 stacks
CROP_DIR = None
YIELD_STACK = None
AREA_STACK = None
BANDMAP_CSV = None
AREA_STACK_IS_HA_PER_PIXEL = True

# Elasticity table (used only in valuation Option A)
ELASTICITY_CSV = None

# Valuation inputs
FAO_GPV_ISO3_CSV = None
FAO_PRICES_FULL_CSV = None
GDP_CURRENT_2019_CSV = None

# Output directory
OUT_DIR = None

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


def compute_country_areas_km2(gdf: gpd.GeoDataFrame) -> pd.Series:
    area_km2 = gdf.geometry.area.values / 1_000_000.0
    return pd.Series(area_km2, index=gdf.index, dtype="float64")


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
    ROOT = _output_path(p, 'erosion_gep_root', 'gep')

    IN_DIR = _output_path(p, 'erosion_gep_input_dir', 'gep_input')
    USLE_PATH = _required_path(p, 'erosion_usle_path', 'IN_DIR / "usle_2019_revised_feb_13.tif"')
    AVOID_PATH = _required_path(p, 'erosion_avoided_erosion_path', 'IN_DIR / "avoided_erosion_2019_revised_feb_13.tif"')

    UPS_DIR = _output_path(p, 'erosion_upstream_dir', 'upstream')
    UPS_PATH = _output_path(p, 'erosion_upstream_prevention_share_path', 'upstream_prevention_share.tif')
    UPS_FOREST_SHARE = _output_path(p, 'erosion_upslope_forest_share_path', 'upslope_forest_share.tif')
    UPS_GRASS_SHARE = _output_path(p, 'erosion_upslope_grass_share_path', 'upslope_grass_share.tif')
    UPS_CROP_SHARE = _output_path(p, 'erosion_upslope_cropland_share_path', 'upslope_cropland_share.tif')
    UPS_BARE_SHARE = _output_path(p, 'erosion_upslope_bare_share_path', 'upslope_bare_share.tif')
    USE_UPSLOPE_LULC_ATTRIBUTION_DIAGNOSTICS = getattr(p, 'erosion_use_upslope_lulc_diagnostics', True)

    BOUNDARY_GPKG = _required_path(p, 'erosion_country_boundary_path', 'IN_DIR / "country_boundary_r250_with_iso3.gpkg"')
    BOUNDARY_SOURCE_EPSG = getattr(p, 'erosion_boundary_source_epsg', None)
    ELEVATION_PATH = _required_path(p, 'erosion_dem_path', 'the elevation model')

    CROP_DIR = _output_path(p, 'erosion_crop_dir', 'crops')
    YIELD_STACK = _required_path(p, 'erosion_yield_stack_path', 'CROP_DIR / "spam2020_yield_stack_TA.tif"')
    AREA_STACK = _required_path(p, 'erosion_area_stack_path', 'CROP_DIR / "spam2020_harvested_area_stack_TA.tif"')
    BANDMAP_CSV = _required_path(p, 'erosion_bandmap_csv_path', 'CROP_DIR / "spam2020_bandmap.csv"')
    AREA_STACK_IS_HA_PER_PIXEL = getattr(p, 'erosion_area_stack_is_ha_per_pixel', True)

    ELASTICITY_CSV = _required_path(p, 'erosion_elasticity_csv_path', 'IN_DIR / "elasticity_crops_fao_revised.csv"')
    FAO_GPV_ISO3_CSV = _required_path(p, 'erosion_fao_gpv_iso3_csv_path', 'IN_DIR / "faostat_gpv_2019_iso3.csv"')
    FAO_PRICES_FULL_CSV = _required_path(p, 'erosion_fao_prices_csv_path', 'IN_DIR / "faostat_prices_2019_completed_revised.csv"')
    GDP_CURRENT_2019_CSV = _required_path(p, 'erosion_gdp_csv_path', 'IN_DIR / "worldbank_gdp_2019.csv"')

    OUT_DIR = _output_path(p, 'erosion_gep_output_dir', 'gep_output')
    os.makedirs(OUT_DIR, exist_ok=True)

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
ROOT = None
RUN_DIR = None

FIG_DIR = None
# (no mkdir at import -- configure_maps re-resolves FIG_DIR from p and creates it at run time)

INTEGRATED_CSV = None
COUNTRY_CROP_LONG_CSV = None

PS_ONFARM_TIF = None
PS_UPSTREAM_TIF = None
PS_COMBINED_TIF = None

RUN_BOUNDARY_GPKG = None

TOP_N = 20
TOP_N_LABELS = 25
RASTER_DOWNSAMPLE_FACTOR = 6
ROBINSON_CRS = "+proj=robin"
EXCLUDE_ISO3 = {"ATA"}

USD_TO_MILLIONS = 1e6
MAP_K_CLASSES = 5
MONEY_UNIT_LABEL = "2019 USD million"


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

    ROOT = _output_path(p, 'erosion_gep_root', 'gep')
    run_tag = getattr(p, 'erosion_run_tag', 'ses11_onfarm_upstream_combined_20260305')
    RUN_DIR = _output_path(p, 'erosion_gep_output_dir', 'gep_output')

    FIG_DIR = _output_path(p, 'erosion_figures_dir', 'figures')
    os.makedirs(FIG_DIR, exist_ok=True)

    INTEGRATED_CSV = _output_path(p, 'erosion_integrated_country_gep_csv', 'integrated_country_gep.csv')
    COUNTRY_CROP_LONG_CSV = _output_path(p, 'erosion_country_crop_long_csv', 'country_crop_protected_production_long.csv')

    PS_ONFARM_TIF = _output_path(p, 'erosion_ps_onfarm_tif', 'ps_onfarm_cropland_severe.tif')
    PS_UPSTREAM_TIF = _output_path(p, 'erosion_ps_upstream_tif', 'ps_upstream_cropland_severe.tif')
    PS_COMBINED_TIF = _output_path(p, 'erosion_ps_combined_tif', 'ps_combined_union_cropland_severe.tif')

    RUN_BOUNDARY_GPKG = _required_path(p, 'erosion_country_boundary_path', 'ROOT / "inputs_v2" / "erosion_gep" / "country_boundary_r250_with_iso3.gpkg"')

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

# ---------------------------------------------------------------------------------------------
# Helpers the rest of the module shares: array cleaning, formatting and column picking.
# ---------------------------------------------------------------------------------------------

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


def assert_exists(p, hint: str = ""):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing: {p}\n{hint}")


def _normcols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


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


def _write_share(path, template: xr.DataArray, arr01: np.ndarray):
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


def to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


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

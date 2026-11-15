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
3. Each SPAM crop's protected share of its production in a country, from the prevention shares.
4. The value, crop by crop: each FAO item's gross production value, the value crop_provision
   carries before its rental rate, times its SPAM crop's protected share times its yield
   elasticity to erosion, summed over the country's items (`per_crop_gep`). The earlier reading,
   one production-weighted shock times the country's crop total (`country_erosion_shock`,
   `country_gep`), is still computed beside it.

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

import hazelbean as hb
from global_invest import utilities

import sys
import logging
import geopandas as gpd
import xarray as xr
from rasterio.enums import Resampling
from rasterio.crs import CRS as rioCRS



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


def country_erosion_shock(df_country_crop, min_shock_floor):
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
        `gdp_const2019_2019`, `gep_const2019_usd` and `gdp_loss_pct`. A country missing its crop
        gross production value or its shock has a missing value, not a zero. A zero comes only
        from a present zero price or a present zero shock.
    """
    out = df_shock.merge(df_crop_gpv, on='iso3', how='left').merge(df_gdp, on='iso3', how='left')
    out['gep_const2019_usd'] = out['crop_gpv_const2019_2019'] * out['erosion_shock_share']
    out['gdp_loss_pct'] = np.where(
        out['gdp_const2019_2019'].notna() & (out['gdp_const2019_2019'] > 0),
        100.0 * out['gep_const2019_usd'] / out['gdp_const2019_2019'],
        np.nan)
    out['component'] = component
    return out[['component', 'iso3', 'protected_production_tons', 'total_production_tons',
                'share_protected_production', 'erosion_shock_share', 'crop_gpv_const2019_2019',
                'gdp_const2019_2019', 'gep_const2019_usd', 'gdp_loss_pct']]


def crop_values_by_country_item(df_crop_values, year):
    """crop_provision's gross production value per country and FAO item, in USD.

    Erosion values a crop at the number crop_provision carries before its rental rate. Under
    crop_provision's production_x_price basis that is production times a 2019 USD producer price
    from the pollination chain's values table for every pair, and under its reported_first basis
    FAOSTAT's reported current-USD value where FAOSTAT reports one. Either way this reads the table
    as crop_provision wrote it. A dissolved state can leave a second, empty row on a pair, so rows
    are collapsed with a sum that keeps an all-empty pair missing and a zero a zero.

    Args:
        df_crop_values (pd.DataFrame): crop_provision's gep_by_country_year_crop table, with
            `iso3_r250_label`, `crop_code`, `crop`, `year`, `value_source` and
            `gross_production_value` (thousand USD, before the rental rate).
        year (int): the valuation year.

    Returns:
        pd.DataFrame: one row per country and item, `iso3`, `fao_item_code`, `fao_item_name`,
        `value_source` and `crop_value_usd`. A pair crop_provision leaves blank stays missing, and
        a row without an r250 country is not valued.

    Raises:
        ValueError: if a pair carries a value on two rows, since summing them would count one crop
            twice.
    """
    df = df_crop_values[(df_crop_values['year'] == year) & df_crop_values['iso3_r250_label'].notna()].copy()
    df['fao_item_code'] = df['crop_code'].astype(int)
    key = ['iso3_r250_label', 'fao_item_code']
    valued = df[df['gross_production_value'].notna()]
    twice = valued[valued.duplicated(key, keep=False)]
    if not twice.empty:
        raise ValueError('crop_provision values %d country-item pairs on two rows at %d, for example %s.'
                         % (twice.groupby(key).ngroups, year, sorted(set(zip(twice[key[0]], twice[key[1]])))[:5]))
    out = df.groupby(key, as_index=False).agg(
        fao_item_name=('crop', 'first'), value_source=('value_source', 'first'),
        gross_production_value=('gross_production_value', lambda s: s.sum(min_count=1)))
    out['crop_value_usd'] = out['gross_production_value'] * utilities.FAOSTAT_THOUSAND_USD
    return out.rename(columns={'iso3_r250_label': 'iso3'})[
        ['iso3', 'fao_item_code', 'fao_item_name', 'value_source', 'crop_value_usd']]


def cpc_key(code):
    """A CPC code as one number, so FAOSTAT's '01491.01' and the elasticity table's 1491.01 meet."""
    return round(float(str(code).strip().lstrip("'")), 2)


def own_elasticity_by_cpc(df_elasticity):
    """Each item's own erosion-to-yield elasticity, keyed on its CPC code.

    The table's `Item Code` is the CPC code without its leading zero. Its column `Item_code_fao`
    holds the same CPC code, not the FAO item code, so joining it on FAO item codes would attach
    rye's row (CPC 0116) to potatoes (FAO 116) and miss every other item.

    Args:
        df_elasticity (pd.DataFrame): the elasticity table, with `Item Code` and `elasticity`.

    Returns:
        dict: `cpc_key(code)` to the elasticity, clipped into [0, 1].

    Raises:
        ValueError: if two rows carry the same CPC code.
    """
    codes = df_elasticity['Item Code'].map(cpc_key)
    if codes.duplicated().any():
        raise ValueError('the elasticity table repeats CPC codes %s' % sorted(codes[codes.duplicated()])[:5])
    return dict(zip(codes, pd.to_numeric(df_elasticity['elasticity'], errors='coerce').clip(0.0, 1.0)))


def per_crop_gep(df_values, df_correspondence, df_country_crop, own_elasticity):
    """Each crop valued at its own protected share and elasticity: GEP_k = sum_i V_ki s_k,m(i) e_i.

    V is the item's gross production value, m(i) the SPAM 2020 crop or crops the correspondence
    gives it, s that crop's protected share of its production in the country, and e the item's own
    elasticity where the elasticity table lists its CPC code, otherwise the SPAM crop's. An item
    placed on two crops (millet, coffee) takes their shares pooled by SPAM tonnes. No floor applies.

    Args:
        df_values (pd.DataFrame): `crop_values_by_country_item`'s output.
        df_correspondence (pd.DataFrame): one row per FAO item and SPAM crop, with `fao_item_code`,
            `fao_cpc_code` and `spam2020_crop_label`, blank for an item that is not a crop.
        df_country_crop (pd.DataFrame): one prevention component's rows, one per country and SPAM
            crop, with `ISO3`, `crop`, `total_production_tons`, `share_protected_production` and
            `elasticity_used` (the library elasticity of that crop).
        own_elasticity (dict): `own_elasticity_by_cpc`'s output.

    Returns:
        pd.DataFrame: one row per country and item, the value columns plus `spam2020_crops`,
        `status` ('valued', 'not a crop', 'no value' or 'no SPAM production of the crop'),
        `protected_share`, `elasticity_library`, `elasticity_own`, `elasticity_used` and
        `gep_const2019_usd`, missing unless the pair is valued.

    Raises:
        ValueError: if a valued item has no row in the correspondence.
    """
    corr = df_correspondence.copy()
    corr['spam2020_crop_label'] = corr['spam2020_crop_label'].fillna('').astype(str).str.strip()
    unmapped = (set(df_values.loc[df_values['crop_value_usd'].notna(), 'fao_item_code'])
                - set(corr['fao_item_code']))
    if unmapped:
        raise ValueError('FAO items %s carry a crop value and no row in the SPAM 2020 '
                         'correspondence.' % sorted(unmapped))
    crops = corr.loc[corr['spam2020_crop_label'] != '', ['fao_item_code', 'spam2020_crop_label']]
    prot = df_country_crop.rename(columns={'ISO3': 'iso3', 'crop': 'spam2020_crop_label'}).copy()
    prot['t'] = pd.to_numeric(prot['total_production_tons'], errors='coerce')
    prot['ts'] = prot['t'] * pd.to_numeric(prot['share_protected_production'], errors='coerce').clip(0.0, 1.0)
    prot['tse'] = prot['ts'] * pd.to_numeric(prot['elasticity_used'], errors='coerce').clip(0.0, 1.0)
    library = prot.groupby('spam2020_crop_label')['elasticity_used'].first().clip(0.0, 1.0)

    key = ['iso3', 'fao_item_code']
    long = df_values[key].merge(crops, on='fao_item_code', how='left')
    long = long.merge(prot[['iso3', 'spam2020_crop_label', 't', 'ts', 'tse']],
                      on=['iso3', 'spam2020_crop_label'], how='left')
    long['e_map'] = long['spam2020_crop_label'].map(library)
    grouped = long.groupby(key)
    agg = grouped[['t', 'ts', 'tse']].sum(min_count=1)
    agg['e_map'] = grouped['e_map'].mean()
    agg['spam2020_crops'] = grouped['spam2020_crop_label'].agg(lambda s: '+'.join(sorted(s.dropna())))

    out = df_values.merge(agg.reset_index(), on=key, how='left')
    out['protected_share'] = out['ts'] / out['t']
    # The order of operations is the checked bridge's (audit_runs/erosion_william_bridge_20261103),
    # so its per-pair values reproduce to the last bit.
    se_library = out['tse'] / out['t']
    out['elasticity_library'] = se_library / out['protected_share']
    zero = out['protected_share'] == 0
    out.loc[zero, 'elasticity_library'] = out.loc[zero, 'e_map']
    own_by_item = (corr.drop_duplicates('fao_item_code').set_index('fao_item_code')['fao_cpc_code']
                   .map(cpc_key).map(own_elasticity))
    out['elasticity_own'] = out['fao_item_code'].map(own_by_item)
    out['status'] = np.select(
        [out['spam2020_crops'].fillna('') == '', out['crop_value_usd'].isna(), ~(out['t'] > 0)],
        ['not a crop', 'no value', 'no SPAM production of the crop'], 'valued')
    out['elasticity_used'] = out['elasticity_own'].where(out['elasticity_own'].notna(), out['elasticity_library'])
    valued = out['status'] == 'valued'
    out['gep_const2019_usd'] = np.where(
        valued, out['crop_value_usd'] * out['protected_share'] * out['elasticity_used'], np.nan)
    return out[key + ['fao_item_name', 'value_source', 'crop_value_usd', 'spam2020_crops', 'status',
                      'protected_share', 'elasticity_library', 'elasticity_own', 'elasticity_used',
                      'gep_const2019_usd']]


def per_crop_country_gep(df_pairs):
    """Country totals of the per-crop valuation, with the value it leaves out and why.

    Args:
        df_pairs (pd.DataFrame): `per_crop_gep`'s output.

    Returns:
        pd.DataFrame: one row per country, `crop_value_usd` (every valued item),
        `crop_value_valued_usd`, `crop_value_without_spam_share_usd`, `crop_value_not_a_crop_usd`,
        the pair counts, `partial_sum` (True where items with a value have no share) and
        `gep_const2019_usd`, missing where no pair is valued.
    """
    df = df_pairs.copy()
    for status, column in (('valued', 'valued'), ('no SPAM production of the crop', 'without_spam_share'),
                           ('not a crop', 'not_a_crop')):
        df['crop_value_%s_usd' % column] = df['crop_value_usd'].where(df['status'] == status)
        df['pairs_%s' % column] = (df['status'] == status).astype(int)
    df['pairs_no_value'] = (df['status'] == 'no value').astype(int)
    out = df.groupby('iso3').agg(
        crop_value_usd=('crop_value_usd', lambda s: s.sum(min_count=1)),
        crop_value_valued_usd=('crop_value_valued_usd', lambda s: s.sum(min_count=1)),
        crop_value_without_spam_share_usd=('crop_value_without_spam_share_usd', lambda s: s.sum(min_count=1)),
        crop_value_not_a_crop_usd=('crop_value_not_a_crop_usd', lambda s: s.sum(min_count=1)),
        pairs_valued=('pairs_valued', 'sum'), pairs_without_spam_share=('pairs_without_spam_share', 'sum'),
        pairs_not_a_crop=('pairs_not_a_crop', 'sum'), pairs_no_value=('pairs_no_value', 'sum'),
        gep_const2019_usd=('gep_const2019_usd', lambda s: s.sum(min_count=1))).reset_index()
    out['partial_sum'] = out['crop_value_without_spam_share_usd'].fillna(0) > 0
    return out


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






# =============================================================================
# SECTION A — InVEST SDR run (from step1_sdr_invest_run.ipynb)
# =============================================================================



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

    if hb.path_exists(proj_lib):
        os.environ["PROJ_LIB"] = str(proj_lib)
    if hb.path_exists(gdal_data):
        os.environ["GDAL_DATA"] = str(gdal_data)

    # Prefer official EPSG parameters vs GeoTIFF geokeys when there is mismatch
    os.environ.setdefault("GTIFF_SRS_SOURCE", "EPSG")

    # Cluster-safe: avoid network calls for grids/definitions
    os.environ.setdefault("PROJ_NETWORK", "OFF")

    # Helpful with some odd reprojection edge-cases
    os.environ.setdefault("OGR_ENABLE_PARTIAL_REPROJECTION", "YES")


def proj_gdal_env():
    """The PROJ/GDAL settings that decide how a CRS is read and a reprojection is done.

    Erosion reads rasters stored in one CRS and computes in another, so these four settings are
    part of a run's provenance: they change what a stored CRS resolves to and which datum grids
    are used. `run_metadata.txt` records them beside the analysis EPSG.

    Returns:
        dict: setting name -> value, with "(not set)" where the variable is absent.
    """
    return {name: os.environ.get(name, "(not set)")
            for name in ("PROJ_LIB", "GDAL_DATA", "PROJ_NETWORK", "GTIFF_SRS_SOURCE")}


# =============================================================================
# 2) ENV + LOGGING (INVEST TEMPLATE STYLE) -- lazy, NOT at import
# =============================================================================
LOGGER = logging.getLogger(__name__)


# =============================================================================
# 3) HELPERS
# =============================================================================
def _assert_exists(path, label: str):
    if not hb.path_exists(path):
        raise FileNotFoundError(f"[missing] {label}: {path}")


# =============================================================================
# 4) SDR ARGS (INVEST 3.17.x TEMPLATE)
# =============================================================================
def build_args(p, watersheds_path) -> dict:
    """The InVEST SDR argument dictionary for Section A.

    Args:
        p: the ProjectFlow object, for the erosion_sdr_* settings and the input references.
        watersheds_path (str): the sanitized watersheds layer.

    Returns:
        dict: the arguments InVEST SDR takes.
    """
    return {
        "workspace_dir": str(p.erosion_sdr_output_dir),
        "results_suffix": p.erosion_sdr_results_suffix,

        "biophysical_table_path": str(p.erosion_biophysical_table_path),
        "dem_path": str(p.erosion_dem_path),
        "lulc_path": str(p.erosion_lulc_path),
        "erodibility_path": str(p.erosion_erodibility_path),
        "erosivity_path": str(p.erosion_erosivity_path),
        "watersheds_path": str(watersheds_path),

        "drainage_path": str(p.erosion_drainage_path) if p.erosion_drainage_path else "",

        "flow_dir_algorithm": p.erosion_sdr_flow_dir_algorithm,
        "threshold_flow_accumulation": p.erosion_sdr_threshold_flow_accumulation,
        "k_param": p.erosion_sdr_k_param,
        "sdr_max": p.erosion_sdr_max,
        "ic_0_param": p.erosion_sdr_ic_0_param,
        "l_max": p.erosion_sdr_l_max,
        "n_workers": p.erosion_sdr_n_workers,
    }


# =============================================================================
# SECTION B -- On-farm + upstream erosion prevention-share GEP valuation
#              (from Combine_PS_SES11_3_3_2026.ipynb)
# =============================================================================

# Reprojection strategies
RESAMPLE_USLE_AVOID = Resampling.average
RESAMPLE_UPS        = Resampling.average
RESAMPLE_YIELD      = Resampling.average
RESAMPLE_AREA       = Resampling.sum
RESAMPLE_DEM        = Resampling.average


# ==========================================================
# 3) Elasticity loader (with SPAM support)
# ==========================================================
# The alias map is passed in, not read from a module constant (exact FAO item names
# first; the source repo's stem-only aliases never exact-match the FAO names, so there every crop
# silently takes the 0.08 fallback).


def get_elasticity_for_crop(crop_key: str, elast_map: dict, fallback: float,
                            spam_aliases: dict) -> float:
    crop_key = str(crop_key).strip().lower()
    v = elast_map.get(crop_key, np.nan)
    if np.isfinite(v):
        return float(np.clip(v, 0.0, 1.0))
    for alias in spam_aliases.get(crop_key, []):
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
# FAOSTAT's own definition of a crop: the members of its QV item group 2041, "Crops".
FAOSTAT_QV_CROPS_GROUP = '2041'


def crop_item_codes(item_groups):
    """FAOSTAT's crop item codes: the members of the QV item group "Crops" (2041).

    Args:
        item_groups (pd.DataFrame): FAOSTAT's item-group definitions, with `Domain Code`,
            `Item Group Code` and `Item Code`.

    Returns:
        set: the integer item codes of the crop group. A code that is itself a QV group total
        (Crops, Cereals primary and the like) is never returned: its value repeats its members'
        values, so counting it would count them twice.

    Raises:
        ValueError: if the table has no QV "Crops" members, since filtering on an empty set
            would value no crop at all.
    """
    qv = item_groups[item_groups['Domain Code'].astype(str) == 'QV']
    rows = qv[qv['Item Group Code'].astype(str).str.strip() == FAOSTAT_QV_CROPS_GROUP]
    codes = set(pd.to_numeric(rows['Item Code'], errors='coerce').dropna().astype(int))
    codes -= set(pd.to_numeric(qv['Item Group Code'], errors='coerce').dropna().astype(int))
    if not codes:
        raise ValueError('FAOSTAT item groups carry no QV "Crops" (2041) members.')
    return codes


def crop_item_mask(df, crop_codes):
    """True on the rows of a FAOSTAT table whose item code is a crop.

    Args:
        df (pd.DataFrame): rows with `item_code_fao` (normalized column names).
        crop_codes (set): `crop_item_codes`' output.

    Returns:
        pd.Series: a boolean mask. A row without a numeric item code is not a crop.
    """
    return pd.to_numeric(df['item_code_fao'], errors='coerce').isin(crop_codes)


def crop_items_only(df, crop_codes):
    """The rows of a FAOSTAT value table whose item code is a crop.

    The code decides, not the item name. A name pattern misses sugar cane, oranges, bananas and
    other crops, and its "oat" also keeps goat meat.

    Args:
        df (pd.DataFrame): FAOSTAT rows with `item_code_fao` (normalized column names).
        crop_codes (set): `crop_item_codes`' output.

    Returns:
        pd.DataFrame: the crop rows. A row without a numeric item code is not a crop.

    Raises:
        ValueError: if a country lists the same crop twice, because summing both rows would
            double that crop's value in the country total.
    """
    out = df[crop_item_mask(df, crop_codes)].copy()
    if 'iso3' in out.columns:
        repeated = out[out.duplicated(['iso3', 'item_code_fao'], keep=False)]
        if not repeated.empty:
            raise ValueError('FAOSTAT crop value listed twice for country-item pairs: %s'
                             % sorted(set(zip(repeated['iso3'], repeated['item_code_fao'])))[:5])
    return out



# ---------------------------------------------------------------------------------------------
# Helpers the rest of the module shares: array cleaning, formatting and column picking.
# ---------------------------------------------------------------------------------------------


def get_erosion_yield_coefficient(crop_key, coef_map, fallback, spam_aliases):
    """crop_key -> erosion-to-yield coefficient: direct hit, else SPAM alias, else the flat fallback."""
    k = str(crop_key).strip().lower()
    v = coef_map.get(k, np.nan)
    if np.isfinite(v):
        return float(np.clip(v, 0.0, 1.0))
    for alias in spam_aliases.get(k, []):
        v2 = coef_map.get(str(alias).strip().lower(), np.nan)
        if np.isfinite(v2):
            return float(np.clip(v2, 0.0, 1.0))
    return float(np.clip(fallback, 0.0, 1.0))






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


# --- Area-damage arithmetic (the NGFS damage_area seam) ---
# Panagos-style area damage and physical erosion-prevention stress.
# 
# Pure arithmetic for the NGFS damage specification. Missing data remain missing;
# the caller must aggregate severe and total cropland on the same valid support.
def productivity_level(severe_crop_ha, valid_crop_ha, alpha=0.08):
    """Negative productivity level in percent; zero cropland is undefined."""
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be a fraction in [0, 1]')
    severe, total = np.broadcast_arrays(np.asarray(severe_crop_ha, dtype=float),
                                      np.asarray(valid_crop_ha, dtype=float))
    if np.any(np.isinf(severe) | np.isinf(total)):
        raise ValueError('Areas must be finite or missing')
    valid = np.isfinite(severe) & np.isfinite(total)
    if np.any(valid & ((severe < 0) | (total < 0) | (severe > total))):
        raise ValueError('Require 0 <= severe cropland <= valid total cropland')
    result = np.full(total.shape, np.nan)
    np.divide(-100 * alpha * severe, total, out=result, where=valid & (total > 0))
    return result


def summarize_damage_areas(usle, potential, crop_fraction, hectares, zones,
                           common_support=None, threshold=11., provision_loss=.2):
    """Zonal areas and damage on aligned SDR rasters in t/ha/year.

    Zone 0 is outside the analysis domain. Missing soil data are reported as
    excluded cropland hectares, never recoded as zero erosion. Thresholding occurs
    before zonal averaging; a cell at exactly 11 is not classified as severe.
    """
    import pandas as pd
    loss, bare, crop, ha, ids = np.broadcast_arrays(*[np.asarray(x) for x in
        (usle, potential, crop_fraction, hectares, zones)])
    if np.any(np.isfinite(crop) & ((crop<0)|(crop>1))):
        raise ValueError('Cropland fraction outside [0,1]')
    if np.any(np.isfinite(ha)&(ha<=0)):
        raise ValueError('Cell hectares must be positive')
    if np.any(~np.isfinite(ids)) or np.any(ids<0) or np.any(ids!=np.floor(ids)):
        raise ValueError('Zone ids must be nonnegative integers')
    footprint=(ids>0)&np.isfinite(crop)&np.isfinite(ha)
    valid=footprint&np.isfinite(loss)&np.isfinite(bare)
    if common_support is not None:
        valid &= np.broadcast_to(np.asarray(common_support,dtype=bool),ids.shape)
    stress=stressed_soil_loss(np.where(valid,loss,np.nan),np.where(valid,bare,np.nan),provision_loss)
    weight=np.where(footprint,crop*ha,0.)
    def aggregate(mask):
        return np.bincount(ids.ravel().astype(int),weights=np.where(mask,weight,0.).ravel())
    total=aggregate(valid); severe=aggregate(valid&(loss>threshold))
    stressed=aggregate(valid&(stress>threshold)); excluded=aggregate(footprint&~valid)
    active=np.flatnonzero((total+excluded)>0)
    return pd.DataFrame(dict(zone_id=active,valid_crop_ha=total[active],
        severe_crop_ha=severe[active],stressed_severe_crop_ha=stressed[active],
        excluded_crop_ha=excluded[active],
        level=productivity_level(severe[active],total[active]),
        stressed_level=productivity_level(stressed[active],total[active])))


def stressed_soil_loss(usle, rkls, provision_loss=0.2):
    """Reduce on-site prevented soil loss, holding potential erosion fixed.

All quantities are t/ha/year. The caller applies the agreed onset and thresholds
the resulting soil loss, rather than scaling a signed productivity level.
"""
    if not 0 <= provision_loss <= 1:
        raise ValueError('provision_loss must be a fraction in [0, 1]')
    actual, potential = np.broadcast_arrays(np.asarray(usle, dtype=float),
                                            np.asarray(rkls, dtype=float))
    valid = np.isfinite(actual) & np.isfinite(potential)
    if np.any(np.isinf(actual) | np.isinf(potential)):
        raise ValueError('Soil loss must be finite or missing')
    if np.any(valid & ((actual < 0) | (potential < actual))):
        raise ValueError('Require 0 <= actual soil loss <= potential soil loss')
    return np.where(valid, actual + provision_loss * (potential - actual), np.nan)


def annual_damage_change(base_level, anchor_years, ordinary_levels, years,
                         base_year=2023, stressed_levels=None, stress_from=2030):
    """Interpolate negative productivity levels, then subtract the 2023 level.

    A physical stress begins at its onset, with no anticipation in preceding
    years. Stress onset must be an anchor so its physical level is known.
    """
    anchors = np.asarray(anchor_years, dtype=int)
    levels = np.asarray(ordinary_levels, dtype=float)
    dates = np.asarray(years, dtype=int)
    if (anchors.ndim != 1 or levels.shape != anchors.shape or anchors.size == 0
            or np.any(np.diff(anchors) <= 0) or np.any(anchors <= base_year)):
        raise ValueError('Need sorted unique anchor years after the base year and matching levels')
    if np.any(dates < base_year) or np.any(dates > anchors[-1]):
        raise ValueError('Requested dates are outside the trajectory')
    if not np.isfinite(base_level) or not np.all(np.isfinite(levels)):
        raise ValueError('Missing damage levels cannot define a trajectory')
    result = np.interp(dates, np.r_[base_year, anchors], np.r_[base_level, levels])
    if stressed_levels is not None:
        stress = np.asarray(stressed_levels, dtype=float)
        if stress.shape != levels.shape or not np.all(np.isfinite(stress)):
            raise ValueError('Need finite stressed levels at every anchor')
        if stress_from not in anchors:
            raise ValueError('Stress onset needs a physical anchor')
        if np.any(stress > levels + 1e-10):
            raise ValueError('Losing prevention must not improve productivity on the same support')
        active = dates >= stress_from
        result[active] = np.interp(dates[active], anchors, stress)
    return result - base_level

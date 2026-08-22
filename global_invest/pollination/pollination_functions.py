"""Pollination science: a thin driver over William Sidemo-Holm's crop_benefits chain, plus the
frame arithmetic the two shock tasks and the GEP valuation share.

The raster science (300 m sufficiency, 5 km valuation, PNAS diff) is imported unchanged from
crop_benefits; the driver functions at the bottom only run it per scenario on our SEALS maps and
zonal-aggregate the diff to GTAP r50xAEZ regions, reproducing the percent change definition in
his pnas_ngfs pipeline (crop_benefits/project_flow/pnas_ngfs.py, which is not part of the
installed package).

Everything above the driver section is pure frame arithmetic over per-zone series and needs
neither crop_benefits nor a raster, which is what the tests exercise. The crop_benefits and
raster imports are made inside the driver functions so that importing this module (or running the
static shock task, which needs none of it) does not require the package to be installed.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASELINE_LABEL = '2023_pnas'

# The country attributes every GEP per-country CSV carries, in the order the CSV writes them.
POLLINATION_ATTR_COLS = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
                         'continent', 'region_un', 'region_wb', 'income_grp', 'subregion']

# The zonal percent changes arrive already expressed in percent, so a percent change of x means a
# growth factor of 1 + x/100.
PERCENT = 100.0


# =============================================================================
# Shock arithmetic. Both shock tasks turn per-zone values into one row per
# (zone, sector, year); these functions are that turn, with no IO in them.
# =============================================================================

def anchor_shock_tables(scenario_pct_by_year, baseline_pct_by_year):
    """The two shock measures at the anchor years, per zone.

    Both inputs are percent changes against the SAME fixed base-year value, so subtracting them
    gives the scenario's departure from its baseline measured against that fixed base. Dividing
    that by the baseline's own growth factor rebases it onto the contemporaneous baseline, which
    is what the economic model reads. The rescale is exact and needs no extra rasters, because
    the ratio of the two denominators is precisely the baseline's growth factor.

    Args:
        scenario_pct_by_year (dict): anchor year -> pd.Series of per-zone percent change.
        baseline_pct_by_year (dict): anchor year -> pd.Series of per-zone percent change for the
            nature-off baseline, over the same zones.

    Returns:
        tuple: (fixedbase, contemporaneous) DataFrames indexed by zone with one column per anchor
        year. Zones missing from any anchor year are dropped from both, so the two tables always
        carry the same rows. A baseline that has shrunk exactly to zero gives NaN rather than an
        infinite contemporaneous shock.
    """
    anchor_years = sorted(scenario_pct_by_year)
    fixedbase = pd.DataFrame({y: scenario_pct_by_year[y] - baseline_pct_by_year[y]
                              for y in anchor_years}).dropna()
    base_factor = pd.DataFrame({y: 1.0 + baseline_pct_by_year[y] / PERCENT
                                for y in anchor_years}).reindex(fixedbase.index).replace(0, np.nan)
    return fixedbase, fixedbase / base_factor


def dynamic_shock_rows(fixedbase, contemporaneous, level_usd, scenario, sectors, base_year):
    """Anchor-year shocks expanded to one row per zone, sector and year.

    The chain computes a shock only at the years the scenario maps exist for; the economic model
    reads one value per year, so the anchors are joined by straight lines with the base year
    pinned at no shock. shock_pct is the contemporaneous measure, matching carbon: afeall is a
    productivity deviation from the baseline path, so it is normalised by the year's own baseline
    rather than by the fixed base-year value.

    Args:
        fixedbase (pd.DataFrame): per-zone percent change against the fixed base year, one column
            per anchor year.
        contemporaneous (pd.DataFrame): the same zones rebased onto each year's baseline.
        level_usd (pd.Series): per-zone absolute baseline pollination value in base-year USD, or
            None. Carried through so the GEP chain can consume this task instead of rerunning the
            same rasters.
        scenario (str): the scenario label written into every row.
        sectors (iterable): the GTAP activities the shock applies to.
        base_year (int): the year the ramp starts from, at zero.

    Returns:
        list: dicts, one per zone, sector and year from base_year through the last anchor year.
    """
    anchor_years = list(fixedbase.columns)
    all_years = list(range(base_year, anchor_years[-1] + 1))
    interp_years = [base_year] + anchor_years

    rows = []
    for zone, fixed_series in fixedbase.iterrows():
        endw, reg = zone
        annual = np.interp(all_years, interp_years, [0.0] + list(fixed_series.values))
        annual_contemp = np.interp(all_years, interp_years,
                                   [0.0] + list(contemporaneous.loc[zone].values))
        base_usd = float(level_usd.get(zone, float('nan'))) if level_usd is not None else float('nan')
        for year, fixed_value, contemp_value in zip(all_years, annual, annual_contemp):
            for sector in sectors:
                rows.append({'ENDW': endw, 'ACTS': sector, 'REG': reg, 'scenario': scenario,
                             'year': year, 'shock_pct': contemp_value,
                             'shock_pct_fixedbase': fixed_value,
                             'shock_pct_contemp': contemp_value,
                             'value_usd_base': base_usd})
    return rows


def static_shock_rows(baseline_values, scenario_values, scenario, sectors, base_year, end_year):
    """The frozen table's scenario-minus-baseline difference, ramped linearly from the base year.

    Only zones present in both series are shocked: a zone the scenario or the baseline does not
    cover has no difference to ramp, and inventing one would put a fabricated number into the
    economic model.

    Args:
        baseline_values (pd.Series): per-zone nature-off baseline value, indexed by (ENDW, REG).
        scenario_values (pd.Series): per-zone scenario value over the same index.
        scenario (str): the scenario label written into every row.
        sectors (iterable): the GTAP activities the shock applies to.
        base_year (int): the year the ramp starts from, at zero.
        end_year (int): the year the ramp reaches the full difference.

    Returns:
        list: dicts, one per zone, sector and year from base_year through end_year.
    """
    common = baseline_values.index.intersection(scenario_values.index)
    shock = (scenario_values.loc[common] - baseline_values.loc[common]).dropna()
    n_years = end_year - base_year

    rows = []
    for year in range(base_year, end_year + 1):
        fraction = (year - base_year) / n_years
        for (endw, reg), value in shock.items():
            for sector in sectors:
                rows.append({'ENDW': endw, 'ACTS': sector, 'REG': reg,
                             'scenario': scenario, 'year': year, 'shock_pct': value * fraction})
    return rows


# =============================================================================
# GEP valuation arithmetic. The value raster is USD per cell with crop prices
# and pollination dependence already embedded upstream, so there is no price
# join here: the region totals ARE value, and the only work is getting from the
# r264 aggregation surface to one row per country.
# =============================================================================

def collapse_regions_to_countries(df_regions):
    """Per-region USD totals summed to one row per country and year.

    Summing the r264-expanded table as it stands would count a split country once per sub-region,
    so the sum is taken on the r250 country id and the country attributes are attached afterwards
    from one representative sub-region each.

    Args:
        df_regions (pd.DataFrame): the zonal summary, one row per r264 region, with
            iso3_r250_id, year, total and the country attribute columns.

    Returns:
        pd.DataFrame: the attribute columns, year and pollination_gep, one row per country.
    """
    df_countries = (df_regions.groupby(['iso3_r250_id', 'year'], as_index=False)['total'].sum()
                    .rename(columns={'total': 'pollination_gep'}))
    attributes = df_regions[POLLINATION_ATTR_COLS].drop_duplicates('iso3_r250_id')
    return df_countries.merge(attributes, how='left', on='iso3_r250_id')[
        POLLINATION_ATTR_COLS + ['year', 'pollination_gep']]


def expand_country_values_to_regions(df_regions, df_gep_by_country):
    """Each r264 region carrying its COUNTRY's value, for the map only.

    The sub-region rows repeat the national value rather than splitting it, so this table is
    never summed. It exists because the choropleth draws r264 polygons.

    Args:
        df_regions (pd.DataFrame): the zonal summary, one row per r264 region.
        df_gep_by_country (pd.DataFrame): the collapsed per-country table.

    Returns:
        pd.DataFrame: df_regions with pollination_gep attached.
    """
    return df_regions.merge(df_gep_by_country[['iso3_r250_id', 'pollination_gep']],
                            how='left', on='iso3_r250_id')


# =============================================================================
# Driver over the crop_benefits raster chain. Nothing below here is arithmetic
# this module owns; it runs the imported science and aggregates its output.
# =============================================================================

def configure_crop_benefits(p, target_year):
    """crop_benefits Config pointed at OUR base_data + task dir -- from p, no hardcoded machine paths.

    validate=False: our chain skips the FAO/CropGrids tabular stages, so the ONLY external input it
    needs is the precomputed baseline pollination-value raster under base_data/crop_benefits/. The 5 km
    resample template only has to define the target grid, and run_pollination_valuation_5km requires
    sufficiency and value to share that grid -- so country_raster points at the value raster itself,
    which removes the separate country-raster dependency. Sufficiency outputs go to the task dir.
    """
    from crop_benefits.config import load_config

    cfg = load_config(validate=False)
    cb = p.get_path('crop_benefits')
    # poll_value_global_<year>usd.tif is a PRECOMPUTED baseline input, not built here. It is the
    # output of the crop_benefits raster pipeline, run once over Monfreda yields x CropGrids area x
    # FAO producer prices x pollination-dependence ratios:
    #   crop_benefits: scripts/pipelines/run_fao_pipeline.py  then
    #                  scripts/pipelines/run_raster_pipeline.py --step poll_value
    #                  (src/crop_benefits/raster/pollination_value.py::run_pollination_value)
    # Reuses that raster and skips the FAO/CropGrids tabular stages (validate=False).
    cfg.paths.country_raster = Path(os.path.join(cb, 'poll_value_global_%dusd.tif' % int(target_year)))
    cfg.outputs.pollination_value_2020 = Path(cb)
    cfg.outputs.pollination_sufficiency = Path(p.cur_dir)
    return cfg


def baseline_denominator(cfg, baseline_lulc_path, target_year):
    """Unpaired 2023 pollination value (the % change denominator), computed once."""
    from crop_benefits.pollination.sufficiency_poll import (
        run_pollination_sufficiency_300m, run_pollination_sufficiency_5km)
    from crop_benefits.pollination.sufficiency_value import run_pollination_valuation_5km

    run_pollination_sufficiency_300m(cfg, lulc_path=baseline_lulc_path, scenario=BASELINE_LABEL)
    run_pollination_sufficiency_5km(cfg, scenario=BASELINE_LABEL)
    run_pollination_valuation_5km(cfg, scenario=BASELINE_LABEL, target_year=target_year)
    return cfg.outputs.pollination_sufficiency / f'value_pollination_sufficiency_{BASELINE_LABEL}_5km.tif'


def scenario_region_pct_change(cfg, scenario, lulc_path, baseline_lulc_path,
                               denominator_path, correspondence_gpkg, target_year):
    """Per-region % change of pollination value, scenario vs 2023 baseline (stable cropland only).

    Returns (pct_change, baseline_value_usd); see _zonal_pct_change for why the level comes free.
    """
    from crop_benefits.pollination.sufficiency_poll import (
        run_pollination_sufficiency_300m_stable_ag, run_pollination_sufficiency_5km)
    from crop_benefits.pollination.sufficiency_value import run_pollination_valuation_5km
    from crop_benefits.pollination.sufficiency_diff import run_pollination_diff_5km_pnas

    stab, b_stab = f'{scenario}_stab', f'{BASELINE_LABEL}_stab_{scenario}'
    for suff_scen, lulc, other in [(stab, lulc_path, baseline_lulc_path),
                                   (b_stab, baseline_lulc_path, lulc_path)]:
        run_pollination_sufficiency_300m_stable_ag(cfg, lulc_path=lulc, other_lulc_path=other, scenario=suff_scen)
        run_pollination_sufficiency_5km(cfg, scenario=suff_scen)
        run_pollination_valuation_5km(cfg, scenario=suff_scen, target_year=target_year)

    suff_dir = cfg.outputs.pollination_sufficiency
    diff_path = run_pollination_diff_5km_pnas(
        cfg, scenario=scenario, baseline_scenario=BASELINE_LABEL,
        scenario_value_path=suff_dir / f'value_pollination_sufficiency_{stab}_5km.tif',
        baseline_value_path=suff_dir / f'value_pollination_sufficiency_{b_stab}_5km.tif')
    return _zonal_pct_change(diff_path, denominator_path, correspondence_gpkg)


def _zonal_pct_change(diff_path, denominator_path, correspondence_gpkg, region_id_field='ee_r50_aez18_id'):
    """Per r50xAEZ zone, keyed to (ENDW, REG): the % change AND the absolute baseline value.

    Returns (pct_change, baseline_value_usd) as two aligned Series.

    The second is the GEP quantity and costs nothing to emit: it is the DENOMINATOR of the first
    (sum of the baseline pollination-value raster x pixel area over the zone, in target-year USD),
    so it is already computed here and was previously discarded. GEP wants the level, GTAP wants
    the ratio; returning both means one task serves both rather than two chains recomputing the
    same rasters. See pollination_shock, which writes it as `value_usd_base`.
    """
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from crop_benefits.raster.spatial import build_area_km2_raster

    gdf = gpd.read_file(correspondence_gpkg, engine='pyogrio')
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    gdf[region_id_field] = gdf[region_id_field].astype(int)

    with rasterio.open(denominator_path) as src:
        base = src.read(1).astype(np.float64)
        base[base == src.nodata] = np.nan
        transform, shape, meta = src.transform, src.shape, src.meta.copy()
    with rasterio.open(diff_path) as src:
        diff = src.read(1).astype(np.float64)
        diff[diff == src.nodata] = np.nan

    zones = rasterize(
        ((g, int(r)) for g, r in zip(gdf.geometry, gdf[region_id_field]) if g is not None and not g.is_empty),
        out_shape=shape, transform=transform, fill=0, dtype=np.int32)
    area = build_area_km2_raster(meta)

    out, level = {}, {}
    for rid, sub in gdf.drop_duplicates(region_id_field).set_index(region_id_field).iterrows():
        mask = zones == rid
        denom = np.nansum(base[mask] * area[mask]) if mask.any() else 0.0
        if not denom:
            continue
        key = (f"AEZ{int(sub['aez18_id'])}", sub['gtapv7_r50_label'])
        out[key] = np.nansum(diff[mask] * area[mask]) / denom * PERCENT
        level[key] = denom
    return pd.Series(out), pd.Series(level)

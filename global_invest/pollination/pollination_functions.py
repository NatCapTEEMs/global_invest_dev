"""Pollination science: the sufficiency and value calculation, plus the
frame and array arithmetic the two shock tasks and the GEP valuation share.

The raster science (300 m sufficiency, 5 km valuation, PNAS diff) lives here rather than in an
outside package, so the module needs nothing beyond this repo; a test blocks that import and
asserts the module still loads. The driver functions at the bottom run
that science per scenario on our SEALS maps, and the task layer reads the rasters it leaves
behind. Nothing here opens a file: the zonal step takes arrays in and hands per-zone series back,
which is what the tests exercise.

What crop_benefits still supplies is data, not code. The shock path resamples a precomputed
baseline value raster from base_data/crop_benefits/, declared in es_parameters like any other
input.
"""
from __future__ import annotations
import os
import re

import logging
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict
import rasterio

import pandas as pd
from global_invest import utilities

logger = logging.getLogger(__name__)


# The zonal rasters are geographic, and the burned zone raster reserves this id for the cells
# that fall outside every zone.
LATLON_EPSG = 4326
NO_ZONE_ID = 0


# =============================================================================
# Zonal arithmetic. The task reads the rasters and burns the zone ids; this
# turns those arrays into one percent change and one absolute value per zone.
# =============================================================================

def zone_labels_from_boundary(gdf_boundary, zone_id_col, endw_col, reg_col, endw_format):
    """Zone id -> the (ENDW, REG) pair every shock row is keyed on.

    Args:
        gdf_boundary (pd.DataFrame): the r50xAEZ boundary attributes. One row per zone is kept.
        zone_id_col, endw_col, reg_col, endw_format (str): which columns carry the zone id, the
            AEZ band and the region label, and how the endowment label is spelled. They are
            es_parameters rows, the same ones terrestrial_carbon has always read.

    Returns:
        dict: zone id -> (ENDW, REG), in the boundary's own row order.
    """
    unique_zones = gdf_boundary.drop_duplicates(zone_id_col).set_index(zone_id_col)
    return {int(zone_id): (endw_format % int(row[endw_col]), row[reg_col])
            for zone_id, row in unique_zones.iterrows()}


def zonal_pct_change(diff_arr, baseline_arr, area_arr, zones_arr, zone_labels):
    """Per zone, the percent change in pollination value AND the absolute baseline value.

    The second costs nothing to compute because it is the DENOMINATOR of the first (the baseline
    pollination-value raster summed over the zone, weighted by pixel area, in target-year USD).
    GEP wants the level, the economic model wants the ratio, so returning both means one task
    serves both rather than two calculations recomputing the same rasters. A zone with no baseline value
    is dropped from both series: there is nothing for a change to be a share of.

    Args:
        diff_arr (np.ndarray): the scenario-minus-baseline value raster, nodata as NaN.
        baseline_arr (np.ndarray): the baseline value raster on the same grid, nodata as NaN.
        area_arr (np.ndarray): pixel area in km2 on the same grid.
        zones_arr (np.ndarray): the zone id burned onto the same grid, NO_ZONE_ID outside.
        zone_labels (dict): zone id -> (ENDW, REG).

    Returns:
        tuple: (pct_change, baseline_value_usd) as two aligned pd.Series keyed on (ENDW, REG).
    """
    pct_change, baseline_value = {}, {}
    for zone_id, key in zone_labels.items():
        mask = zones_arr == zone_id
        denominator = np.nansum(baseline_arr[mask] * area_arr[mask])
        if not denominator:
            continue
        pct_change[key] = np.nansum(diff_arr[mask] * area_arr[mask]) / denominator * 100.0
        baseline_value[key] = denominator
    return pd.Series(pct_change), pd.Series(baseline_value)


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
    base_factor = pd.DataFrame({y: 1.0 + baseline_pct_by_year[y] / 100.0
                                for y in anchor_years}).reindex(fixedbase.index).replace(0, np.nan)
    return fixedbase, fixedbase / base_factor


def dynamic_shock_rows(fixedbase, contemporaneous, level_usd, scenario, sectors, base_year):
    """Anchor-year shocks expanded to one row per zone, sector and year.

    The calculation computes a shock only at the years the scenario maps exist for; the economic model
    reads one value per year, so the anchors are joined by straight lines with the base year
    pinned at no shock. shock_pct is the contemporaneous measure, matching carbon: afeall is a
    productivity deviation from the baseline path, so it is normalised by the year's own baseline
    rather than by the fixed base-year value.

    Args:
        fixedbase (pd.DataFrame): per-zone percent change against the fixed base year, one column
            per anchor year.
        contemporaneous (pd.DataFrame): the same zones rebased onto each year's baseline.
        level_usd (pd.Series): per-zone absolute baseline pollination value in base-year USD, or
            None. Carried through so the GEP calculation can consume this task instead of rerunning the
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
    df_countries = utilities.collapse_regions_to_countries(
        df_regions, utilities.GEP_COUNTRY_ATTR_COLS, 'pollination_gep')
    return df_countries[utilities.GEP_COUNTRY_ATTR_COLS + ['year', 'pollination_gep']]


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
    return utilities.expand_country_values_to_regions(
        df_regions, df_gep_by_country, 'pollination_gep')


# =============================================================================
# Driver over the crop_benefits raster chain. Nothing below here is arithmetic
# this module owns; it runs the imported science and names what it left behind.
# =============================================================================

def configure_sufficiency(p, target_year):
    """The settings the sufficiency and value steps need, filled from the ProjectFlow object.

    The shock side weights a value raster that is fixed across scenarios, so it reads the finished
    raster rather than rebuilding it; the GEP side builds its own through the yield, production and
    value tasks. The 5 km template points at the value raster itself, because the valuation needs
    sufficiency and value on one grid and that removes a separate country-raster input.
    """
    crop_benefits_dir = p.pollination_value_raster_dir
    return SufficiencySettings(
        output_dir=str(p.cur_dir),
        value_raster_dir=str(crop_benefits_dir),
        country_raster_path=os.path.join(
            crop_benefits_dir, 'poll_value_global_%dusd.tif' % int(target_year)),
        tile_size=int(p.pollination_sufficiency_kernel_tile_rows),
        n_workers=int(p.pollination_sufficiency_n_workers),
        lulc_classes_path=p.pollination_lulc_classes_path)


# ---------------------------------------------------------------------------------------------

@dataclass
class SufficiencySettings:
    """What the raster steps below need, in place of the crop_benefits Config they used to read.

    That Config was loaded from a gitignored local.yaml with `validate=False`, so a missing or
    wrong file did not fail up front, it just proceeded. These seven fields are everything the
    four raster modules ever read off it, and the pollination task fills them from the
    ProjectFlow object.

    Attributes:
        output_dir (str): where the sufficiency and value rasters are written, the task's own dir.
        value_raster_dir (str): where the precomputed baseline pollination-value raster lives.
        country_raster_path (str): the raster defining the 5 km target grid. The valuation needs
            sufficiency and value on one grid, so this points at the value raster itself.
        pa_raster_300m_path (str): the protected-area raster, for the protected-area summary.
        tile_size (int): rows per block when streaming the 300 m land cover. ⚠⚠ **This changes
            the result, it is not a performance setting.** The foraging-radius kernel takes its
            latitude from the tile's midpoint and rounds the 2 km radius to an integer pixel
            count, so the tile height decides the kernel. At 2048 our raster agreed with the
            source pipeline's on 78.67 percent of cells, in bands; at its own 8192 every one of
            331,871,070 cells matches. The es_parameters row is named
            `pollination_sufficiency_kernel_tile_rows` to say so, since it sits beside n_workers,
            which really is about speed.
        n_workers (int): parallel workers for the tiled sufficiency pass.
    """
    output_dir: str
    value_raster_dir: str
    country_raster_path: str
    tile_size: int
    n_workers: int
    lulc_classes_path: str
    pa_raster_300m_path: str = None


# The compression profiles the raster writers ask for.
COMPRESSION_PROFILES = {
    'continuous': {'compress': 'DEFLATE', 'predictor': 3, 'zlevel': 6, 'tiled': True,
                   'blockxsize': 256, 'blockysize': 256, 'BIGTIFF': 'IF_SAFER'},
    'categorical': {'compress': 'DEFLATE', 'predictor': 2, 'zlevel': 6, 'tiled': True,
                    'blockxsize': 256, 'blockysize': 256, 'BIGTIFF': 'IF_SAFER'},
    'defaults': {'compress': 'DEFLATE', 'tiled': True, 'BIGTIFF': 'IF_SAFER'},
}


def build_area_km2_raster(meta: dict) -> np.ndarray:
    """
    Build a 2-D pixel-area raster (km²) from a rasterio profile.
    
    The resulting raster has the same shape as defined in 'meta'.
    """
    transform = meta["transform"]
    nrows = meta["height"]
    ncols = meta["width"]
    
    # Latitude of pixel centers: y = f + e * (row + 0.5)
    # Note: 'e' is usually negative (pixel height) in north-up images
    latitudes = transform.f + transform.e * (np.arange(nrows) + 0.5)
    
    # hazelbean returns m2 per cell on the WGS84 ellipsoid, which is the convention the
    # rest of the account uses through the ha_per_cell pyramid.
    import hazelbean as hb

    # resolution from the transform, not the module constant: the constant is the grid this
    # normally runs on, but the meta is the grid it was actually handed.
    area_per_row = np.asarray(hb.get_area_of_pixel_column_from_center_lats(
        abs(transform.a), np.asarray(latitudes, dtype='float64'))) / 1e6
    # Broadcast row areas across all columns
    return np.repeat(area_per_row[:, None], ncols, axis=1).astype(np.float32)


def convert_density_to_mass(density_raster: np.ndarray, area_km2_raster: np.ndarray) -> np.ndarray:
    """
    Convert density (e.g. tonnes/km²) to mass (e.g. tonnes).
    
    mass = density * area
    """
    mass = np.full_like(density_raster, np.nan, dtype=np.float32)
    valid = np.isfinite(density_raster) & (area_km2_raster > 0)
    mass[valid] = density_raster[valid] * area_km2_raster[valid]
    return mass


def get_compression_profile(
    cfg: SufficiencySettings,
    profile_name: str = "continuous"
) -> Dict[str, Any]:
    """
    Retrieve compression settings for a given profile name.

    Parameters
    ----------
    cfg : Config
        Pipeline configuration.
    profile_name : str
        Name of the profile (e.g., 'continuous', 'categorical', 'defaults').

    Returns
    -------
    dict
        Dictionary of rasterio creation options (compress, predictor, etc.).
    """
    if profile_name == "continuous":
        return COMPRESSION_PROFILES['continuous'].copy()
    elif profile_name == "categorical":
        return COMPRESSION_PROFILES['categorical'].copy()
    else:
        return COMPRESSION_PROFILES['defaults'].copy()

@dataclass
class FaoPriceSettings:
    """What the FAO price steps need, in place of the crop_benefits Config they used to read.

    The prices are a base-data input rather than a per-run result: the pipeline reads the FAOSTAT
    production and producer-price bulks, reconstructs local currency where FAOSTAT reports only the
    old series, converts to USD against World Bank exchange rates, and writes a median price per
    crop over `price_years`. The pollination value raster is built against that table.

    The two FAOSTAT bulks are staged into base data rather than pulled on each run. FAOSTAT
    revises them, so a fresh pull would make the price table depend on the day it was built, and
    a compute node without outbound network could not run the stage at all. es_parameters carries
    a url and an archive member for each, so the shared download task can fetch them once.

    Attributes:
        crosswalk_m49_iso3_path (str): FAO M49 area codes to ISO3.
        fao_classification_path (str): the FAO item classification.
        crosswalk_fao_cropgrids_path (str): FAO items to CropGrids crop names.
        fao_production_bulk_path (str): the staged FAOSTAT production and yield bulk CSV.
        fao_prices_bulk_path (str): the staged FAOSTAT producer-price bulk CSV.
        fx_lcu_per_usd_path (str): the staged World Bank exchange rates.
        output_dir (str): where the production, price, value and median-price tables are written.
        fao_start_year (int): first year of FAOSTAT data to keep.
        fao_end_year (int): last year.
        price_years (tuple): the years the median price is taken over.
    """
    crosswalk_m49_iso3_path: str
    fao_classification_path: str
    crosswalk_fao_cropgrids_path: str
    fao_production_bulk_path: str
    fao_prices_bulk_path: str
    fx_lcu_per_usd_path: str
    output_dir: str
    # No defaults on the three windows: es_parameters is the only place they are set, so a run
    # at a different base year cannot silently inherit one built for another. price_years kept a
    # 2018-2022 default through the move to a 2019 account, which is what put the price window
    # two years off the base year and made a deflator necessary to hide it.
    fao_start_year: int
    fao_end_year: int
    price_years: tuple
    # Tables the price path needs, read from CSVs beside the service rather than held as
    # dictionaries in this module: FAO aggregate codes to leave out, countries with no
    # exchange rate of their own, and IMF country names in the FAO spelling.
    excluded_item_codes: set
    fx_inherit_iso3: dict
    imf_fao_names: dict
    # Price quality-control tolerances. _QC_START_YEAR was a separate constant holding 1993,
    # the same value as fao_start_year, so the panel's span and the QC window could disagree.
    tol_near_equal_lcu_slc: float
    anchor_max_year_dist: int
    anchor_max_times_off: float
    global_median_max_times_off: float
    qc_min_overlap: int
    qc_bad_median_times_off: float
    qc_bad_share_over_3x: float



# ---------------------------------------------------------------------------------------------
# The FAO price path, the parts that hold no file handling. These build the per-crop median
# producer prices the pollination value raster is priced at.
# ---------------------------------------------------------------------------------------------

# Two bulks, two element vocabularies. Upstream keeps these in separate modules, where both are
# called _ELEMENTS_KEEP; flattening those modules into this one let the price list overwrite the
# production list, so the production filter kept price elements from a production bulk and the
# panel came out empty.
_PRODUCTION_ELEMENTS_KEEP = {"Yield", "Production"}


_PRICE_ELEMENTS_KEEP = [
    "Producer Price (USD/tonne)",
    "Producer Price (SLC/tonne)",
    "Producer Price (LCU/tonne)",
    "Producer Price Index (2014-2016 = 100)",
]




def _convert_yield_units(df: pd.DataFrame) -> pd.DataFrame:
    """Convert yield from kg/ha -> t/ha."""
    mask = df["Element"] == "Yield"
    df.loc[mask, "Value"] = df.loc[mask, "Value"] / 1000
    logger.info("Converted %d yield rows from kg/ha to t/ha", mask.sum())
    return df


def _recode_country(series: pd.Series, mapping: dict) -> pd.Series:
    return series.map(mapping).fillna(series)


def _log_df(name: str, df: pd.DataFrame) -> None:
    logger.info("%s: %d rows, %d cols", name, df.shape[0], df.shape[1])


def _reshape_prices(pp_raw: pd.DataFrame) -> pd.DataFrame:
    """Keep relevant elements and pivot to wide format."""
    logger.info("=== 2-3) FILTER ELEMENTS + RESHAPE TO WIDE ===")

    pp = pp_raw[pp_raw["Element"].isin(_PRICE_ELEMENTS_KEEP)].copy()

    pp_wide = (
        pp.pivot_table(
            index=["Area", "Area Code (M49)", "Item", "Item Code", "Year"],
            columns="Element", values="Value", aggfunc="first",
        )
        .reset_index()
        .rename(columns={
            "Area": "country", "Year": "year",
            "Producer Price (USD/tonne)": "usd_tonne_obs",
            "Producer Price (SLC/tonne)": "slc_tonne",
            "Producer Price (LCU/tonne)": "lcu_tonne",
            "Producer Price Index (2014-2016 = 100)": "price_index",
        })
    )
    _log_df("pp_wide", pp_wide)
    return pp_wide


def _reconstruct_slc_lcu(pp_wide: pd.DataFrame) -> pd.DataFrame:
    """Steps 4-7: base SLC, fill SLC, LCU/SLC ratios, fill LCU."""

    # 4) Base SLC at index = 100
    logger.info("=== 4) ESTIMATING BASE SLC AT INDEX=100 ===")
    slc_base = (
        pp_wide.dropna(subset=["slc_tonne", "price_index"])
        .query("price_index > 0")
        .assign(base_slc_implied=lambda d: d["slc_tonne"] * 100 / d["price_index"])
        .groupby(["country", "Item"], as_index=False)
        .agg(
            slc_at_100_index=("base_slc_implied", "median"),
            n_years_base=("base_slc_implied", "size"),
        )
    )

    # 5) Fill SLC
    logger.info("=== 5) FILLING SLC USING INDEX ===")
    pp2 = pp_wide.merge(slc_base, on=["country", "Item"], how="left")

    pp2["slc_filled"] = np.where(
        pp2["slc_tonne"].notna(), pp2["slc_tonne"],
        np.where(
            pp2["price_index"].notna() & pp2["slc_at_100_index"].notna(),
            pp2["slc_at_100_index"] * pp2["price_index"] / 100, np.nan,
        ),
    )
    pp2["slc_source"] = np.select(
        [pp2["slc_tonne"].notna(),
         pp2["slc_tonne"].isna() & pp2["price_index"].notna() & pp2["slc_at_100_index"].notna()],
        ["observed", "index_imputed"], default="missing",
    )

    # 6) LCU/SLC ratios
    logger.info("=== 6) ESTIMATING LCU/SLC RATIOS ===")
    lcu_slc_year = (
        pp2.dropna(subset=["lcu_tonne", "slc_tonne"])
        .query("slc_tonne > 0")
        .assign(ratio=lambda d: d["lcu_tonne"] / d["slc_tonne"])
        .groupby(["country", "year"], as_index=False)
        .agg(lcu_per_slc_year_country=("ratio", "median"))
    )
    lcu_slc_country = (
        lcu_slc_year.groupby("country", as_index=False)
        .agg(lcu_per_slc_country=("lcu_per_slc_year_country", "median"))
    )

    # 7) Fill LCU
    logger.info("=== 7) FILLING LCU USING SLC + RATIOS ===")
    pp3 = (
        pp2.merge(lcu_slc_year, on=["country", "year"], how="left")
        .merge(lcu_slc_country, on="country", how="left")
    )

    pp3["lcu_filled"] = np.select(
        [
            pp3["lcu_tonne"].notna(),
            pp3["lcu_tonne"].isna() & pp3["slc_filled"].notna() & pp3["lcu_per_slc_year_country"].notna(),
            pp3["lcu_tonne"].isna() & pp3["slc_filled"].notna()
            & pp3["lcu_per_slc_year_country"].isna() & pp3["lcu_per_slc_country"].notna()
            & (pp3["lcu_per_slc_country"] - 1).abs() <= cfg.tol_near_equal_lcu_slc,
            pp3["lcu_tonne"].isna() & pp3["slc_filled"].notna()
            & pp3["lcu_per_slc_year_country"].isna() & pp3["lcu_per_slc_country"].notna(),
            pp3["lcu_tonne"].isna() & pp3["slc_filled"].notna(),
        ],
        [
            pp3["lcu_tonne"],
            pp3["slc_filled"] * pp3["lcu_per_slc_year_country"],
            pp3["slc_filled"],
            pp3["slc_filled"] * pp3["lcu_per_slc_country"],
            pp3["slc_filled"],
        ],
        default=np.nan,
    )

    pp3["lcu_source"] = np.select(
        [
            pp3["lcu_tonne"].notna(),
            pp3["lcu_tonne"].isna() & pp3["slc_filled"].notna() & pp3["lcu_per_slc_year_country"].notna(),
            pp3["lcu_tonne"].isna() & pp3["slc_filled"].notna()
            & pp3["lcu_per_slc_year_country"].isna() & pp3["lcu_per_slc_country"].notna()
            & (pp3["lcu_per_slc_country"] - 1).abs() <= cfg.tol_near_equal_lcu_slc,
            pp3["lcu_tonne"].isna() & pp3["slc_filled"].notna()
            & pp3["lcu_per_slc_year_country"].isna() & pp3["lcu_per_slc_country"].notna(),
            pp3["lcu_tonne"].isna() & pp3["slc_filled"].notna(),
        ],
        [
            "observed", "slc_filled_country_year_factor",
            "slc_filled_country_factor_near_equal",
            "slc_filled_country_factor", "slc_filled_assume_equal",
        ],
        default="missing",
    )

    logger.info("LCU source counts:\n%s", pp3["lcu_source"].value_counts())
    return pp3


def _build_usd_with_qc(
    pp3: pd.DataFrame,
    fx: pd.DataFrame,
) -> pd.DataFrame:
    """Merge FX, compute USD, apply 4 QC filters."""

    # 14) Merge FX
    logger.info("=== 14) MERGING FX INTO PRICE PANEL ===")
    pp_usd = pp3.merge(fx, on=["iso3", "year"], how="left")

    # 15) FX-implied USD
    logger.info("=== 15) COMPUTING FX-IMPLIED USD ===")
    pp_usd["usd_fx_implied"] = np.where(
        pp_usd["lcu_filled"].notna() & pp_usd["lcu_per_usd"].notna() & (pp_usd["lcu_per_usd"] > 0),
        pp_usd["lcu_filled"] / pp_usd["lcu_per_usd"], np.nan,
    )
    pp_usd["usd_filled"] = pp_usd["usd_tonne_obs"].combine_first(pp_usd["usd_fx_implied"])
    pp_usd["usd_source"] = np.select(
        [pp_usd["usd_tonne_obs"].notna(),
         pp_usd["usd_tonne_obs"].isna() & pp_usd["usd_fx_implied"].notna()],
        ["observed", "filled"], default="missing",
    )

    # 15a) Invalidate USD from bad LCU sources
    logger.info("=== 15a) INVALIDATING USD FROM BAD LCU SOURCES ===")
    bad_lcu = {"slc_filled_country_factor", "missing", "slc_filled_assume_equal"}
    mask_bad = (pp_usd["usd_source"] == "filled") & pp_usd["lcu_source"].isin(bad_lcu)
    pp_usd.loc[mask_bad, ["usd_fx_implied", "usd_filled", "usd_source"]] = [
        np.nan, np.nan, "invalid_currency_mapping",
    ]

    # 15b) QC: closest observed USD anchor
    logger.info("=== 15b) QC: CLOSEST OBSERVED USD ANCHOR ===")
    obs_anchor = (
        pp_usd.loc[pp_usd["usd_tonne_obs"].notna(), ["iso3", "Item", "year", "usd_tonne_obs"]]
        .rename(columns={"year": "anchor_year", "usd_tonne_obs": "anchor_usd"})
    )
    tmp = pp_usd.merge(obs_anchor, on=["iso3", "Item"], how="left")
    tmp["anchor_year_dist"] = (tmp["year"] - tmp["anchor_year"]).abs()
    tmp = tmp.sort_values("anchor_year_dist")
    closest_idx = tmp.groupby(["iso3", "Item", "year"])["anchor_year_dist"].idxmin().dropna().astype(int)
    nearest = tmp.loc[closest_idx, ["iso3", "Item", "year", "anchor_year", "anchor_usd", "anchor_year_dist"]].copy()
    pp_usd = pp_usd.merge(nearest, on=["iso3", "Item", "year"], how="left")

    use_anchor = (
        (pp_usd["usd_source"] == "filled")
        & (pp_usd["anchor_year_dist"] <= cfg.anchor_max_year_dist)
        & pp_usd["anchor_usd"].notna() & (pp_usd["anchor_usd"] > 0)
        & pp_usd["usd_fx_implied"].notna() & (pp_usd["usd_fx_implied"] > 0)
    )
    pp_usd.loc[use_anchor, "times_off_closest_obs"] = np.maximum(
        pp_usd.loc[use_anchor, "usd_fx_implied"] / pp_usd.loc[use_anchor, "anchor_usd"],
        pp_usd.loc[use_anchor, "anchor_usd"] / pp_usd.loc[use_anchor, "usd_fx_implied"],
    )
    bad_anchor = use_anchor & (pp_usd["times_off_closest_obs"] > cfg.anchor_max_times_off)
    logger.warning("Anchor QC dropping %d rows", bad_anchor.sum())
    pp_usd.loc[bad_anchor, ["usd_fx_implied", "usd_filled", "usd_source"]] = [
        np.nan, np.nan, "implausible_vs_closest_observed_10x",
    ]

    # 15c) QC: global item-year median band
    logger.info("=== 15c) QC: GLOBAL ITEM-YEAR MEDIAN BAND ===")
    global_med = (
        pp_usd.loc[pp_usd["usd_tonne_obs"].notna()]
        .groupby(["Item", "year"])
        .agg(global_median_usd=("usd_tonne_obs", "median"), n_obs=("usd_tonne_obs", "size"))
        .reset_index()
    )
    pp_usd = pp_usd.merge(global_med, on=["Item", "year"], how="left")
    use_global = (
        (pp_usd["usd_source"] == "filled")
        & pp_usd["global_median_usd"].notna() & (pp_usd["global_median_usd"] > 0)
        & pp_usd["usd_fx_implied"].notna() & (pp_usd["usd_fx_implied"] > 0)
    )
    pp_usd.loc[use_global, "global_ratio"] = (
        pp_usd.loc[use_global, "usd_fx_implied"] / pp_usd.loc[use_global, "global_median_usd"]
    )
    bad_global = use_global & (
        (pp_usd["global_ratio"] > cfg.global_median_max_times_off)
        | (pp_usd["global_ratio"] < 1 / cfg.global_median_max_times_off)
    )
    logger.warning("Global median QC dropping %d rows", bad_global.sum())
    pp_usd.loc[bad_global, ["usd_fx_implied", "usd_filled", "usd_source"]] = [
        np.nan, np.nan, "implausible_vs_global_median_10x",
    ]

    # 15d) QC: FX reliability by country
    logger.info("=== 15d) QC: FX RELIABILITY BY COUNTRY ===")
    qc = (
        pp_usd.loc[
            (pp_usd["year"] >= cfg.fao_start_year)
            & pp_usd["usd_tonne_obs"].notna() & pp_usd["usd_fx_implied"].notna()
            & (pp_usd["usd_tonne_obs"] > 0) & (pp_usd["usd_fx_implied"] > 0),
            ["iso3", "usd_tonne_obs", "usd_fx_implied"],
        ].copy()
    )
    qc["times_off"] = np.maximum(
        qc["usd_tonne_obs"] / qc["usd_fx_implied"],
        qc["usd_fx_implied"] / qc["usd_tonne_obs"],
    )
    country_qc = (
        qc.groupby("iso3").agg(
            n_overlap=("times_off", "size"),
            median_times_off=("times_off", "median"),
            share_over_3x=("times_off", lambda s: (s > 3).mean()),
        ).reset_index()
    )
    bad_fx_countries = set(
        country_qc.loc[
            (country_qc["n_overlap"] >= cfg.qc_min_overlap)
            & (
                (country_qc["median_times_off"] > cfg.qc_bad_median_times_off)
                | (country_qc["share_over_3x"] > cfg.qc_bad_share_over_3x)
            ),
            "iso3",
        ]
    )
    logger.warning("Bad FX countries: %d", len(bad_fx_countries))
    bad_fx_mask = (pp_usd["usd_source"] == "filled") & pp_usd["iso3"].isin(bad_fx_countries)
    pp_usd.loc[bad_fx_mask, ["usd_fx_implied", "usd_filled", "usd_source"]] = [
        np.nan, np.nan, "invalid_fx_country",
    ]

    # 15e) Re-attach country column
    country_lookup = pp3[["iso3", "region_fao", "subregion_fao", "year", "Item", "country"]].drop_duplicates()
    pp_usd = pp_usd.merge(country_lookup, on=["iso3", "year", "Item"], how="left", suffixes=("", "_from_pp3"))
    if "country_from_pp3" in pp_usd.columns:
        pp_usd["country"] = pp_usd["country"].combine_first(pp_usd["country_from_pp3"])
        pp_usd = pp_usd.drop(columns=["country_from_pp3"])

    logger.info("USD source counts (final):\n%s", pp_usd["usd_source"].value_counts())
    return pp_usd


def _compute_annual_prices(prices: pd.DataFrame, cw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute annual prices hierarchically: country, subregion, region, world.
    """
    if "region_fao" not in prices.columns:
        prices = prices.merge(cw[["area_code_m49", "region_fao", "subregion_fao"]], on="area_code_m49", how="left")

    pf = prices[
        prices["price_usd_tonne"].notna()
        & (prices["price_usd_tonne"] > 0)
    ].copy()

    # 1. Annual country prices
    price_country = (
        pf.groupby(
            ["year", "area_code_m49", "region_fao", "subregion_fao", "item_code_fao"],
            as_index=False, dropna=False
        )
        .agg(
            price_usd_tonne=("price_usd_tonne", "mean"),
            n_obs=("price_usd_tonne", "size"),
        )
    )
    
    # 2. Annual subregion median prices
    price_subregion = (
        price_country.dropna(subset=["subregion_fao"])
        .groupby(["year", "subregion_fao", "item_code_fao"], as_index=False)
        .agg(
            subreg_price_usd_tonne=("price_usd_tonne", "median")
        )
    )

    # 3. Annual region median prices
    price_region = (
        price_country.dropna(subset=["region_fao"])
        .groupby(["year", "region_fao", "item_code_fao"], as_index=False)
        .agg(
            reg_price_usd_tonne=("price_usd_tonne", "median")
        )
    )
    
    # 4. Annual world median prices
    price_world = (
        price_country.groupby(["year", "item_code_fao"], as_index=False)
        .agg(
            world_price_usd_tonne=("price_usd_tonne", "median"),
            n_countries_price=("area_code_m49", "nunique"),
        )
    )
    
    return price_country, price_subregion, price_region, price_world


# The CropGrids and yield grid, half a degree at 0.05, which pixel_area_km2 is written for.
PIXEL_RES_DEG = 0.05


def pixel_area_km2_spherical(lat_deg: np.ndarray, res_deg: float = PIXEL_RES_DEG) -> np.ndarray:
    """Pixel area in km2 on a 6371 km sphere, the convention the source pipeline uses.

    Kept so the replication check against crop_benefits can be run on its own terms. Production
    calls hazelbean directly, which is WGS84 and agrees with the rest of the account.
    """
    R = 6371.0  # Earth radius, km
    lat_rad = np.deg2rad(lat_deg)
    dlat = np.deg2rad(res_deg)
    dlon = np.deg2rad(res_deg)
    return (R ** 2) * dlon * (
        np.sin(lat_rad + dlat / 2.0) - np.sin(lat_rad - dlat / 2.0)
    )


# =============================================================================
# The pollination value raster: production x price x pollination dependence.
#
# This is the arithmetic behind poll_value_global_<year>usd.tif, which the GEP
# valuation used to take as a finished input. Everything here is array and scalar
# maths so the task layer can open the per-crop rasters and these functions stay
# testable without one.
# =============================================================================


def price_window_centre_year(price_years):
    """The year the median price is denominated in: the centre of the window it is taken over.

    A median over 2017-2021 is 2019 money and needs no deflating for a 2019 account; a median over
    2018-2022 is 2020 money and does. The window used to be a hardcoded 2018-2022 with the centre
    written out as a constant called PRODUCTION_RASTER_YEAR, which named neither the thing it was
    nor the thing it was used for -- the deflator is applied to `price`, never to production.
    """
    years = sorted(int(y) for y in price_years)
    return years[len(years) // 2]


def usd_deflator(from_year, to_year, cpi_by_year):
    """CPI ratio converting dollars of one year into dollars of another.

    Args:
        from_year (int): the year the value is currently denominated in.
        to_year (int): the year wanted.
        cpi_by_year (dict): year to price index, read from the CPI table by the task layer.

    Returns:
        float: multiply a `from_year` dollar amount by this to get `to_year` dollars.

    Raises:
        KeyError: if either year is outside the table, which is deliberate. Silently returning
            1.0 for an unknown year would leave the value undeflated and indistinguishable from
            a correct one.
    """
    return cpi_by_year[int(to_year)] / cpi_by_year[int(from_year)]


def crop_pollination_value_density(production_density, price_usd_per_tonne, dependence_ratio):
    """Per-crop pollination value, as a density, from a production density.

    The production rasters carry tonnes per square kilometre, not tonnes per pixel, so what
    comes out here is USD per square kilometre and must be multiplied by cell area before it
    can be added up. That distinction is the whole reason this function returns a name
    ending in `_density`: summing the result directly gives a number with no meaning, and
    it is a plausible-looking number, which is worse.

    Args:
        production_density (np.ndarray): tonnes per square kilometre.
        price_usd_per_tonne (float): producer price, already in the target year's dollars.
        dependence_ratio (float or np.ndarray): the share of this crop's output attributable
            to animal pollination, 0 for a wind-pollinated crop and up to 1 for one that sets
            no fruit without pollinators. An array when the ratio varies over space, which is
            coffee, where one FAO item covers arabica and robusta.

    Returns:
        tuple: (pollination value density, total crop value density), both USD per square
        kilometre. The second is returned because the share of crop value that pollination
        accounts for is a headline check on the first.
    """
    import numpy as np
    production = np.where(production_density < 0, np.nan, production_density)
    crop_value_density = production * float(price_usd_per_tonne)
    return crop_value_density * np.asarray(dependence_ratio, dtype='float64'), crop_value_density




def find_source_value_raster(p, gep_base_year):
    """Locate the source author's pollination value raster, preferring the GEP base year.

    His files are named `poll_value_global_<year>usd.tif`, one per price year, and he does not
    publish every year. Take the exact year when it exists, which needs no deflation at all.

    ⚠ Otherwise take the LATEST year he publishes, not the nearest. The files are separate vintages
    of his model, not one raster restated in different dollars: measured on 2026-08-28, his 2024
    file deflates to $386.76bn at 2019 prices while his 2023 file deflates to $398.74bn, a three
    percent spread that a price index cannot produce. The later file is the later method, and it is
    the one that lands on the figure he reports for 2019. Choosing by proximity would silently pick
    the older model whenever the base year sits below the newest release.

    Args:
        p (ProjectFlow): the project, used for path resolution.
        gep_base_year (int): the year the GEP account reports in.

    Returns:
        tuple: (path to the raster, the year its dollars are stated in).

    Raises:
        NameError: if the source directory holds no `poll_value_global_<year>usd.tif` at all.
    """
    import glob
    import re
    source_dir = p.pollination_value_raster_dir
    candidates = {}
    for path in glob.glob(os.path.join(str(source_dir), 'poll_value_global_*usd.tif')):
        match = re.search(r'poll_value_global_(\d{4})usd\.tif$', os.path.basename(path))
        if match:
            candidates[int(match.group(1))] = path
    if not candidates:
        raise NameError('No poll_value_global_<year>usd.tif in %s. The GEP pollination value comes '
                        'from the source author\'s raster; without it there is nothing to read.'
                        % source_dir)
    year = gep_base_year if gep_base_year in candidates else max(candidates)
    return candidates[year], year


def available_source_value_years(p):
    """The price years the source author's directory actually holds, for a clear error message."""
    import glob
    source_dir = p.pollination_value_raster_dir
    years = []
    for path in glob.glob(os.path.join(str(source_dir), 'poll_value_global_*usd.tif')):
        match = re.search(r'poll_value_global_(\d{4})usd\.tif$', os.path.basename(path))
        if match:
            years.append(int(match.group(1)))
    return sorted(years)


def value_density_to_per_cell(value_density, area_km2):
    """USD per square kilometre to USD per cell, which is what a zonal sum can add.

    A cell near the pole covers a fraction of the area of one at the equator, so a density
    raster summed over a country weights every cell alike and answers a question nobody
    asked. Multiplying by the cell's own area is what turns it into money.

    Args:
        value_density (np.ndarray): USD per square kilometre.
        area_km2 (np.ndarray): the area each cell covers, same shape.

    Returns:
        np.ndarray: USD in the cell, with cells of no area left missing rather than zero.
    """
    import numpy as np
    out = np.full(np.shape(value_density), np.nan, dtype='float64')
    valid = np.isfinite(value_density) & (area_km2 > 0)
    out[valid] = value_density[valid] * area_km2[valid]
    return out


def local_pollination_share(pollination_value, crop_value):
    """The fraction of crop value that pollination accounts for, cell by cell.

    Undefined where there is no crop value: a cell growing nothing has no share, which is
    not the same as a share of zero, and zero would pull a mean down as though the cell
    were farmland that pollinators do nothing for.

    Args:
        pollination_value (np.ndarray): the pollination-attributable part.
        crop_value (np.ndarray): total crop value, same units and shape.

    Returns:
        np.ndarray: the ratio, missing where crop value is missing or not positive.
    """
    import numpy as np
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(np.isfinite(crop_value) & (crop_value > 0),
                        pollination_value / crop_value, np.nan)


# FAO gives coffee one item code but two dependence ratios, because arabica and robusta are
# different plants: arabica largely self-pollinates and robusta largely does not.
COFFEE_ITEM_CODE_FAO = 656
COFFEE_DEPENDENCE = {'arabica': 0.25, 'robusta': 0.65}
# The split file carries one row per growing season plus a multi-year summary, so each country
# appears six times. This names the summary row, the way the fisheries reader names the CWoN
# columns it needs: it is the source file's own label, not a setting anyone would tune.
COFFEE_SPLIT_SUMMARY_YEAR = '2021_2025'


def coffee_dependence_by_country(df_arabica_robusta):
    """Each country's coffee pollination dependence, from what it actually grows.

    Arabica depends on pollinators for a quarter of its yield and robusta for near two
    thirds, and FAO files both under item 656. A lookup keyed on the item code therefore has
    two values for coffee, and the source pipeline's `drop_duplicates` kept whichever came
    first, which was arabica: every coffee-growing country was valued as though it grew no
    robusta. Globally that puts coffee's pollination value at $5.55bn where the crops
    actually grown put it at $9.39bn.

    Blending by country rather than globally matters because the mix is not a detail of the
    average: Colombia is all arabica and Vietnam is 97 percent robusta, so one global ratio
    would be wrong in opposite directions for the two largest producers.

    The file gives each country five seasons and a `2021_2025` summary, and the author averages
    over that window. Taking the rows as they come and keeping the last one returns the summary
    today, because the exporter happens to write it last for all 21 countries -- so the number is
    right and nothing says why. Re-export the file in another order and a single season would be
    substituted silently. The summary row is therefore selected by name.

    Args:
        df_arabica_robusta (pd.DataFrame): area_code_m49, year, prop_arabica and prop_robusta.

    Returns:
        dict: M49 country code (int) to the blended dependence ratio.

    Raises:
        NameError: if no row carries the summary year, since the alternative is to average
            whatever rows are present and quietly report a different window than the author.
    """
    import pandas as pd
    df = df_arabica_robusta.dropna(subset=['area_code_m49']).copy()
    summary = df[df['year'].astype(str) == COFFEE_SPLIT_SUMMARY_YEAR]
    if summary.empty:
        raise NameError(
            'No %r row in the coffee split file; it carries %s. That row is the multi-year '
            'window the author blends over, so without it the seasons would have to be averaged '
            'here and the two pipelines would silently differ.'
            % (COFFEE_SPLIT_SUMMARY_YEAR, sorted(df['year'].astype(str).unique())))
    df = summary
    blended = (df['prop_arabica'].astype(float) * COFFEE_DEPENDENCE['arabica']
               + df['prop_robusta'].astype(float) * COFFEE_DEPENDENCE['robusta'])
    codes = pd.to_numeric(df['area_code_m49'], errors='coerce').astype('Int64')
    return dict(zip(codes, blended))


def value_weighted_by_sufficiency(value_array, sufficiency_array, value_nodata=None):
    """The part of the pollination value that the habitat actually present delivers.

    The account's headline is the crop output at stake if pollinators vanished, which is a property
    of the crop mix and does not move when land use does. Multiplying it by habitat sufficiency
    gives the other definition on the table: the service the landscape supplies today, which does
    move. Both are wanted, so both are computed on one grid at one price year.

    Sufficiency is NaN off cropland. That is not zero service, it is no cropland, and those cells
    carry no value either, so treating it as zero drops them from both sums identically rather than
    biasing one.

    Args:
        value_array (np.ndarray): pollination value per cell, at the base year's prices.
        sufficiency_array (np.ndarray): habitat sufficiency on the same grid, 0 to 1.
        value_nodata: the value raster's nodata, treated as no value rather than as a number.

    Returns:
        tuple: (weighted array, unweighted total, weighted total).
    """
    import numpy as np
    value = np.where(np.isfinite(value_array), value_array, 0.0)
    if value_nodata is not None:
        value = np.where(value_array == value_nodata, 0.0, value)
    sufficiency = np.where(np.isfinite(sufficiency_array),
                           np.clip(sufficiency_array, 0.0, 1.0), 0.0)
    weighted = value * sufficiency
    return weighted, float(value.sum()), float(weighted.sum())


def dependence_raster_from_country_lookup(country_id_array, dependence_by_country, default):
    """A per-pixel dependence ratio, looked up by which country the pixel is in.

    Args:
        country_id_array (np.ndarray): the country code in each cell.
        dependence_by_country (dict): country code to ratio.
        default (float): the ratio for a country the lookup does not cover, which is a
            coffee grower we have no arabica-robusta split for rather than one that grows
            none: the production raster is what decides whether a pixel carries coffee.

    Returns:
        np.ndarray: the ratio in each cell, float64.
    """
    import numpy as np
    out = np.full(country_id_array.shape, float(default), dtype='float64')
    for code, ratio in dependence_by_country.items():
        if code is None or ratio is None or not np.isfinite(ratio):
            continue
        out[country_id_array == int(code)] = float(ratio)
    return out


# =============================================================================
# The yield and production chain: Monfreda 2000 yields carried to a target year
# by FAO country ratios, then multiplied by CropGrids harvested area.
#
# This is the half of the source author's pipeline that produces the production
# rasters the value raster is built from. Everything here is array and table
# maths so the task layer can open the per-crop rasters and these stay testable
# without one.
# =============================================================================

# Pixel sentinels, the source author's values. A cell outside the crop's harvested extent and a
# cell inside it with no Monfreda yield are different facts, and the summary counts them apart.
OUTSIDE_CROP = -1.0
UNRESOLVED = -2.0

# M49 country codes run to 894; the lookup is a dense array indexed by code, so it needs a ceiling.
MAX_M49_CODE = 1000

# Which level of the hierarchy supplied a cell's value, written into the provenance raster.
PROVENANCE_NONE, PROVENANCE_COUNTRY, PROVENANCE_SUBREGION = 0, 1, 2
PROVENANCE_REGION, PROVENANCE_WORLD = 3, 4


def yield_change_ratios(production_df, crosswalk_m49, early_years, late_years):
    """FAO yield in a late window over yield in an early window, per country and crop.

    Monfreda's yields are for 2000. Carrying them to a target year needs a per-country, per-crop
    factor, and this is it: the median FAO yield over the late window divided by the median over
    the early one. Medians rather than means, so one drought or one bumper harvest does not set
    the factor.

    A country-crop pair FAO does not report gets no row, so the table also carries subregion,
    region and world medians for `build_hierarchical_lookup` to fall back through.

    Args:
        production_df (pd.DataFrame): FAO production rows, with `element`, `area_code_m49`,
            `item_code_fao`, `year`, `value` and `item_fao`.
        crosswalk_m49 (pd.DataFrame): M49 to `region_fao` and `subregion_fao`.
        early_years (iterable): the base window, around Monfreda's 2000.
        late_years (iterable): the window around the target year.

    Returns:
        pd.DataFrame: one row per country-crop plus the aggregate rows, with `yield_ratio`,
        `late_yield`, `early_yield` and an `agg_level` of country, subregion, region or world.
    """
    df = production_df[production_df['element'] == 'Yield'].copy()
    df['area_code_m49'] = df['area_code_m49'].astype(str).str.zfill(3)
    df['item_code_fao'] = df['item_code_fao'].astype(int)
    df['year'] = df['year'].astype(int)
    # The FAO panel reads back as object, because a blank cell makes the whole column one. A
    # median over strings sorts them, so this has to be numeric before the groupby, not after.
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df[np.isfinite(df['value'])]

    def window_yield(sub, prefix):
        return (sub.groupby(['area_code_m49', 'item_code_fao'], as_index=False)
                .agg(**{'%s_yield' % prefix: ('value', 'median'),
                        '%s_years_count' % prefix: ('year', 'nunique'),
                        'item_fao': ('item_fao', 'first')}))

    early = window_yield(df[df['year'].isin(list(early_years))], 'early')
    late = window_yield(df[df['year'].isin(list(late_years))], 'late')

    merged = early.merge(late, on=['area_code_m49', 'item_code_fao'], how='inner',
                         suffixes=('_x', '_y'))
    if 'item_fao_y' in merged.columns:
        merged = merged.drop(columns=['item_fao_y']).rename(columns={'item_fao_x': 'item_fao'})

    merged['yield_ratio'] = merged['late_yield'] / merged['early_yield']
    merged = merged[np.isfinite(merged['yield_ratio'])]

    crosswalk = crosswalk_m49.copy()
    crosswalk['area_code_m49'] = crosswalk['area_code_m49'].astype(str).str.zfill(3)
    merged = merged.merge(crosswalk[['area_code_m49', 'iso3', 'region_fao', 'subregion_fao']],
                          on='area_code_m49', how='left')

    def aggregate(level_column, level_name, sentinel):
        agg = (merged.dropna(subset=[level_column])
               .groupby(['item_code_fao', level_column], as_index=False)
               .agg(yield_ratio=('yield_ratio', 'median'),
                    late_yield=('late_yield', 'median'),
                    early_yield=('early_yield', 'median'),
                    item_fao=('item_fao', 'first')))
        agg['area_code_m49'] = sentinel
        agg['agg_level'] = level_name
        return agg

    subregion = aggregate('subregion_fao', 'subregion', 'SUB').assign(region_fao=np.nan)
    region = aggregate('region_fao', 'region', 'REG').assign(subregion_fao=np.nan)
    world = (merged.groupby('item_code_fao', as_index=False)
             .agg(yield_ratio=('yield_ratio', 'median'),
                  late_yield=('late_yield', 'median'),
                  early_yield=('early_yield', 'median'),
                  item_fao=('item_fao', 'first')))
    world['area_code_m49'], world['agg_level'] = 'WOR', 'world'
    world = world.assign(subregion_fao=np.nan, region_fao=np.nan)

    merged['agg_level'] = 'country'
    columns = ['area_code_m49', 'item_code_fao', 'yield_ratio', 'late_yield', 'early_yield',
               'agg_level', 'subregion_fao', 'region_fao', 'item_fao']
    return pd.concat([merged[columns], subregion[columns], region[columns], world[columns]],
                     ignore_index=True)


def build_hierarchical_lookup(rows, crosswalk_m49, value_column, default=np.nan):
    """A value per M49 country code, falling back country to subregion to region to world.

    FAO does not report every crop in every country, so a country with no row for a crop takes
    its subregion's median, then its region's, then the world's. The fallback level is recorded
    per country rather than left implicit, because a world-median yield ratio applied to a whole
    country is a much weaker number than its own, and the summary needs to say how many cells
    rest on each.

    Args:
        rows (pd.DataFrame): one crop's slice of `yield_change_ratios`.
        crosswalk_m49 (pd.DataFrame): every M49 code with its `region_fao` and `subregion_fao`.
        value_column (str): which column to map, `yield_ratio` or `late_yield`.
        default (float): the value for a code no level covers.

    Returns:
        tuple: (values, provenance), both arrays indexed by M49 code, provenance carrying
        PROVENANCE_COUNTRY through PROVENANCE_WORLD and PROVENANCE_NONE.
    """
    lookup = np.full(MAX_M49_CODE, default, dtype=np.float32)
    provenance = np.zeros(MAX_M49_CODE, dtype=np.uint8)

    world_rows = rows[rows['agg_level'] == 'world']
    world_value = world_rows[value_column].iloc[0] if not world_rows.empty else np.nan
    region_values = rows[rows['agg_level'] == 'region'].set_index('region_fao')[value_column]
    subregion_values = rows[rows['agg_level'] == 'subregion'].set_index('subregion_fao')[value_column]

    country_values = {}
    for _, row in rows[rows['agg_level'] == 'country'].iterrows():
        try:
            code = int(row['area_code_m49'])
        except (ValueError, TypeError):
            continue
        if code < MAX_M49_CODE and np.isfinite(row[value_column]):
            country_values[code] = row[value_column]

    for _, row in crosswalk_m49.iterrows():
        try:
            code = int(row['area_code_m49'])
        except (ValueError, TypeError):
            continue
        if code >= MAX_M49_CODE:
            continue
        subregion, region = row.get('subregion_fao'), row.get('region_fao')
        if code in country_values:
            lookup[code], provenance[code] = country_values[code], PROVENANCE_COUNTRY
        elif subregion in subregion_values.index and np.isfinite(subregion_values[subregion]):
            lookup[code], provenance[code] = subregion_values[subregion], PROVENANCE_SUBREGION
        elif region in region_values.index and np.isfinite(region_values[region]):
            lookup[code], provenance[code] = region_values[region], PROVENANCE_REGION
        elif np.isfinite(world_value):
            lookup[code], provenance[code] = world_value, PROVENANCE_WORLD
    return lookup, provenance


def lookup_by_country(lookup, country_id_array):
    """Read a per-M49-code lookup onto the grid, with codes past the lookup treated as absent."""
    codes = country_id_array.copy()
    codes[codes >= MAX_M49_CODE] = 0
    return lookup[codes]


def convert_mass_to_density(mass_raster, area_km2_raster):
    """Mass per cell to mass per square kilometre.

    Args:
        mass_raster (np.ndarray): tonnes in the cell.
        area_km2_raster (np.ndarray): the area each cell covers, same shape.

    Returns:
        np.ndarray: tonnes per square kilometre, NaN where the cell has no area or no mass.
    """
    density = np.full_like(mass_raster, np.nan, dtype=np.float32)
    valid = np.isfinite(mass_raster) & (area_km2_raster > 0)
    density[valid] = mass_raster[valid] / area_km2_raster[valid]
    return density


def apply_yield_change(yield_base, ratio, mask_crop):
    """Carry base-year yields to the target year by one country-crop ratio."""
    out = np.full_like(yield_base, OUTSIDE_CROP)
    out[mask_crop] = yield_base[mask_crop] * ratio
    return out


def normalize_yield_to_target(yield_base, harvested_area, mask_crop, country_ids,
                              target_lookup, provenance_lookup):
    """Scale each country's yields so its area-weighted mean matches the FAO target yield.

    The alternative, `apply_yield_change`, multiplies by a ratio and inherits whatever level
    Monfreda's 2000 mean sat at. This instead pins the country mean to what FAO reports for the
    late window and keeps Monfreda only for the shape of the variation within the country.

    A country with no valid Monfreda cell keeps its base values, and a crop cell inside a country
    with no Monfreda yield takes the country target flat, which is the `UNRESOLVED` case: there
    is a country-level number but nothing to spread it over.

    Args:
        yield_base (np.ndarray): Monfreda yield, t/ha, NaN where absent.
        harvested_area (np.ndarray): CropGrids harvested area, ha, the weight in the mean.
        mask_crop (np.ndarray): True where the crop is grown.
        country_ids (np.ndarray): M49 code per cell.
        target_lookup (np.ndarray): target yield per M49 code, from `build_hierarchical_lookup`
            on `late_yield`.
        provenance_lookup (np.ndarray): the matching fallback level per code.

    Returns:
        tuple: (yield_target, provenance), the second being which level backed each cell.
    """
    yield_target = np.full_like(yield_base, np.nan)
    yield_target[~mask_crop] = OUTSIDE_CROP
    provenance = lookup_by_country(provenance_lookup, country_ids)

    has_base = mask_crop & (yield_base >= 0) & (~np.isnan(yield_base))
    for code in np.unique(country_ids):
        if code == 0 or code >= MAX_M49_CODE:
            continue
        target = target_lookup[code]
        in_country = (country_ids == code)
        valid = in_country & has_base

        if np.isnan(target) or target <= 0:
            # No FAO number at any level: Monfreda's own values are the best available.
            if np.any(valid):
                yield_target[valid] = yield_base[valid]
            continue

        gaps = in_country & mask_crop & (~has_base)
        if np.any(valid):
            area = harvested_area[valid]
            total_area = np.sum(area)
            mean_yield = np.sum(yield_base[valid] * area) / total_area if total_area > 0 else 0.0
            if mean_yield > 0:
                yield_target[valid] = yield_base[valid] * (target / mean_yield)
            else:
                yield_target[valid] = target
        if np.any(gaps):
            yield_target[gaps] = target
    return yield_target, provenance


def compute_production(yield_raster, harvested_area, mask_crop):
    """Production in tonnes: yield in tonnes per hectare times harvested hectares.

    Args:
        yield_raster (np.ndarray): t/ha.
        harvested_area (np.ndarray): ha in the cell.
        mask_crop (np.ndarray): True where the crop is grown.

    Returns:
        np.ndarray: tonnes, NaN outside the crop mask.
    """
    out = np.full_like(yield_raster, np.nan)
    out[mask_crop] = (yield_raster * harvested_area)[mask_crop]
    return out


def assign_nearest_country(country_ids, mask_crop):
    """Give a cropped cell with no country the code of the nearest cell that has one.

    A cell can carry harvested area and fall outside every country polygon, on a coastline or a
    small island. Left at zero it would be skipped by every per-country step, silently dropping
    its production; the nearest real country is the closest available answer.
    """
    from scipy.spatial import cKDTree
    out = country_ids.copy()
    needs = mask_crop & (country_ids == 0)
    has = country_ids > 0
    if not np.any(needs) or not np.any(has):
        return out
    rows, cols = np.indices(country_ids.shape)
    tree = cKDTree(np.column_stack((rows[has], cols[has])))
    _, nearest = tree.query(np.column_stack((rows[needs], cols[needs])))
    out[needs] = country_ids[has][nearest]
    return out


def fill_nearest_by_country(data, mask_to_fill, country_ids):
    """Fill each missing cell from the nearest valid cell in its own country.

    Within a country rather than globally: a missing yield next to a border would otherwise take
    a neighbouring country's value, which is exactly the variation the country step is about to
    normalise away.
    """
    from scipy.spatial import cKDTree
    filled = data.copy()
    rows, cols = np.indices(data.shape)
    for code in np.unique(country_ids):
        if code == 0:
            continue
        in_country = (country_ids == code)
        missing = mask_to_fill & in_country & np.isnan(filled)
        valid = in_country & (~np.isnan(filled))
        if not missing.any() or not valid.any():
            continue
        tree = cKDTree(np.column_stack((rows[valid], cols[valid])))
        _, nearest = tree.query(np.column_stack((rows[missing], cols[missing])))
        filled[missing] = filled[valid][nearest]
    return filled


def align_to_reference(src_data, src_meta, ref_meta, resampling=None, dst_nodata=np.nan):
    """Put an array on the reference grid. Arrays only; nothing is opened."""
    from rasterio.warp import reproject, Resampling
    destination = np.full((ref_meta['height'], ref_meta['width']), dst_nodata, dtype=np.float32)
    reproject(source=src_data.astype(np.float32), destination=destination,
              src_transform=src_meta['transform'], src_crs=src_meta.get('crs', 'EPSG:4326'),
              src_nodata=src_meta.get('nodata'),
              dst_transform=ref_meta['transform'], dst_crs=ref_meta.get('crs', 'EPSG:4326'),
              dst_nodata=dst_nodata,
              resampling=Resampling.nearest if resampling is None else resampling)
    return destination

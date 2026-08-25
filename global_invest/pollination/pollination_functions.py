"""Pollination science: the sufficiency and value calculation, plus the
frame and array arithmetic the two shock tasks and the GEP valuation share.

The raster science (300 m sufficiency, 5 km valuation, PNAS diff) is vendored here from
crop_benefits rather than imported from it, so the module needs no package outside this repo; a
test blocks the import and asserts the module still loads. The driver functions at the bottom run
that science per scenario on our SEALS maps, and the task layer reads the rasters it leaves
behind. Nothing here opens a file: the zonal step takes arrays in and hands per-zone series back,
which is what the tests exercise.

What crop_benefits still supplies is data, not code. The shock path resamples a precomputed
baseline value raster from base_data/crop_benefits/, declared in es_parameters like any other
input.
"""
from __future__ import annotations
import os

import logging
import numpy as np
import logging
from dataclasses import dataclass
from typing import Any, Dict
import rasterio

import pandas as pd

logger = logging.getLogger(__name__)

BASELINE_LABEL = '2023_pnas'

# The country attributes every GEP per-country CSV carries, in the order the CSV writes them.
POLLINATION_ATTR_COLS = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
                         'continent', 'region_un', 'region_wb', 'income_grp', 'subregion']

# The zonal percent changes arrive already expressed in percent, so a percent change of x means a
# growth factor of 1 + x/100.
PERCENT = 100.0

# The zone the shock is reported on: one GTAP r50 region crossed with one AEZ18 band. The
# boundary carries this id plus the two columns the (ENDW, REG) key is built from.
REGION_ID_FIELD = 'ee_r50_aez18_id'
AEZ_ID_COLUMN = 'aez18_id'
REGION_LABEL_COLUMN = 'gtapv7_r50_label'
ENDW_LABEL_FORMAT = 'AEZ%d'

# The zonal rasters are geographic, and the burned zone raster reserves this id for the cells
# that fall outside every zone.
LATLON_EPSG = 4326
NO_ZONE_ID = 0


# =============================================================================
# Zonal arithmetic. The task reads the rasters and burns the zone ids; this
# turns those arrays into one percent change and one absolute value per zone.
# =============================================================================

def zone_labels_from_boundary(gdf_boundary):
    """Zone id -> the (ENDW, REG) pair every shock row is keyed on.

    Args:
        gdf_boundary (pd.DataFrame): the r50xAEZ boundary attributes, carrying REGION_ID_FIELD,
            AEZ_ID_COLUMN and REGION_LABEL_COLUMN. One row per zone is kept.

    Returns:
        dict: zone id -> (ENDW, REG), in the boundary's own row order.
    """
    unique_zones = gdf_boundary.drop_duplicates(REGION_ID_FIELD).set_index(REGION_ID_FIELD)
    return {int(zone_id): (ENDW_LABEL_FORMAT % int(row[AEZ_ID_COLUMN]), row[REGION_LABEL_COLUMN])
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
        pct_change[key] = np.nansum(diff_arr[mask] * area_arr[mask]) / denominator * PERCENT
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
    base_factor = pd.DataFrame({y: 1.0 + baseline_pct_by_year[y] / PERCENT
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
# this module owns; it runs the imported science and names what it left behind.
# =============================================================================

def configure_sufficiency(p, target_year):
    """The settings the sufficiency and value steps need, filled from the ProjectFlow object.

    Our calculation skips the FAO and CropGrids tabular stages, so the only external input it
    needs is the precomputed baseline pollination-value raster under base_data/crop_benefits/.
    That raster is the output of the source pipeline's FAO and raster stages, run once over
    Monfreda yields times CropGrids area times FAO producer prices times pollination-dependence
    ratios, and it is reused rather than rebuilt.

    The 5 km resample template only has to define the target grid, and the valuation requires
    sufficiency and value to share that grid, so the template points at the value raster itself.
    That removes the separate country-raster input. Sufficiency outputs go to the task's own dir.

    This replaces a crop_benefits Config loaded from a gitignored local.yaml with validate=False,
    which meant a missing or wrong config did not fail up front, it proceeded.
    """
    crop_benefits_dir = p.get_path('crop_benefits')
    return SufficiencySettings(
        output_dir=str(p.cur_dir),
        value_raster_dir=str(crop_benefits_dir),
        country_raster_path=os.path.join(
            crop_benefits_dir, 'poll_value_global_%dusd.tif' % int(target_year)))


# ---------------------------------------------------------------------------------------------
# Vendored from crop_benefits: the pieces of its sufficiency and value calculation that hold
# no file handling. The raster steps they belong to are in the task module.
# ---------------------------------------------------------------------------------------------

@dataclass
class SufficiencySettings:
    """What the raster steps below need, in place of the crop_benefits Config they used to read.

    That Config was loaded from a gitignored local.yaml with `validate=False`, so a missing or
    wrong file did not fail up front, it just proceeded. These seven fields are everything the
    four vendored modules ever read off it, and the pollination task fills them from the
    ProjectFlow object.

    Attributes:
        output_dir (str): where the sufficiency and value rasters are written, the task's own dir.
        value_raster_dir (str): where the precomputed baseline pollination-value raster lives.
        country_raster_path (str): the raster defining the 5 km target grid. The valuation needs
            sufficiency and value on one grid, so this points at the value raster itself.
        lulc_path (str): the land-cover map the sufficiency is computed from.
        pa_raster_300m_path (str): the protected-area raster, for the protected-area summary.
        tile_size (int): rows per block when streaming a raster.
        n_workers (int): parallel workers for the tiled sufficiency pass.
    """
    output_dir: str
    value_raster_dir: str
    country_raster_path: str
    lulc_path: str = None
    pa_raster_300m_path: str = None
    tile_size: int = 2048
    n_workers: int = 4


# The compression profiles the vendored writers ask for, which used to come off the same Config.
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
    
    area_per_row = pixel_area_km2(latitudes)
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
    fao_start_year: int = 1991
    fao_end_year: int = 2023
    price_years: tuple = (2018, 2019, 2020, 2021, 2022)

    # The vendored code reads these as nested attributes; these keep its call sites unchanged.
    @property
    def paths(self):
        return self

    @property
    def outputs(self):
        return self

    @property
    def run(self):
        return self

    @property
    def crosswalk_m49_iso3(self):
        return self.crosswalk_m49_iso3_path

    @property
    def fao_classification(self):
        return self.fao_classification_path

    @property
    def crosswalk_fao_cropgrids(self):
        return self.crosswalk_fao_cropgrids_path

    @property
    def fao_production(self):
        return self.output_dir

    @property
    def fao_prices(self):
        return self.output_dir

    @property
    def fao_values(self):
        return self.output_dir

    @property
    def fao_median_prices(self):
        return os.path.join(self.output_dir, 'median_prices')


# ---------------------------------------------------------------------------------------------
# Vendored from crop_benefits: the FAO price path, the parts that hold no file handling.
# These build the per-crop median producer prices the pollination value raster is priced at.
# ---------------------------------------------------------------------------------------------

_ELEMENTS_KEEP = {"Yield", "Production"}

_EXCLUDE_ITEM_CODES = {
    836,    # Natural rubber in primary forms (processed latex)
    1717,   # Cereals, primary (aggregate)
    1720,   # Roots and Tubers, Total (aggregate)
    1723,   # Sugar Crops Primary (aggregate)
    1726,   # Pulses, Total (aggregate)
    1729,   # Treenuts, Total (aggregate)
    1732,   # Oilcrops, Oil Equivalent (derived equivalent)
    1735,   # Vegetables Primary (aggregate)
    1738,   # Fruit Primary (aggregate)
    1804,   # Citrus Fruit, Total (aggregate)
    1841,   # Oilcrops, Cake Equivalent (derived equivalent)
    17530,  # Fibre Crops, Fibre Equivalent (derived equivalent)
}

_ELEMENTS_KEEP = [
    "Producer Price (USD/tonne)",
    "Producer Price (SLC/tonne)",
    "Producer Price (LCU/tonne)",
    "Producer Price Index (2014-2016 = 100)",
]

_TOL_NEAR_EQUAL_LCU_SLC = 0.01

_ANCHOR_MAX_YEAR_DIST = 10

_ANCHOR_MAX_TIMES_OFF = 10

_GLOBAL_MED_MAX_TIMES_OFF = 10

_QC_START_YEAR = 1993

_QC_MIN_OVERLAP = 50

_QC_BAD_MEDIAN_TIMES_OFF = 3

_QC_BAD_SHARE_OVER_3X = 0.5

_FX_INHERIT_ISO3 = {
    "COK": "NZL",
    "PRI": "USA",
    "REU": "FRA",
    "KNA": "DMA",
    "VCT": "DMA",
    "LCA": "DMA",
}

_IMF_RECODE = {
    "Afghanistan, Islamic Republic of": "Afghanistan",
    "Armenia, Republic of": "Armenia",
    "Azerbaijan, Republic of": "Azerbaijan",
    "Bahrain, Kingdom of": "Bahrain",
    "Belarus, Republic of": "Belarus",
    "Bolivia": "Bolivia (Plurinational State of)",
    "Hong Kong Special Administrative Region, People's Republic of China": "China, Hong Kong SAR",
    "China, People's Republic of": "China, mainland",
    "Comoros, Union of the": "Comoros",
    "Congo, Republic of": "Congo",
    "Côte d'Ivoire": "Ivory Coast",
    "Croatia, Republic of": "Croatia",
    "Czech Republic": "Czechia",
    "Egypt, Arab Republic of": "Egypt",
    "Equatorial Guinea, Republic of": "Equatorial Guinea",
    "Eritrea, The State of": "Eritrea",
    "Estonia, Republic of": "Estonia",
    "Ethiopia, The Federal Democratic Republic of": "Ethiopia",
    "Fiji, Republic of": "Fiji",
    "Gambia, The": "Gambia",
    "Iran, Islamic Republic of": "Iran (Islamic Republic of)",
    "Kazakhstan, Republic of": "Kazakhstan",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Latvia, Republic of": "Latvia",
    "Lesotho, Kingdom of": "Lesotho",
    "Lithuania, Republic of": "Lithuania",
    "Madagascar, Republic of": "Madagascar",
    "Mauritania, Islamic Republic of": "Mauritania",
    "Mozambique, Republic of": "Mozambique",
    "Netherlands, The": "Netherlands (Kingdom of the)",
    "North Macedonia, Republic of": "North Macedonia",
    "Poland, Republic of": "Poland",
    "Korea, Republic of": "Republic of Korea",
    "Moldova, Republic of": "Republic of Moldova",
    "Serbia, Republic of": "Serbia",
    "Slovak Republic": "Slovakia",
    "Slovenia, Republic of": "Slovenia",
    "Tajikistan, Republic of": "Tajikistan",
    "Timor-Leste, Democratic Republic of": "Timor-Leste",
    "Türkiye, Republic of": "Turkey",
    "United Kingdom": "United Kingdom of Great Britain and Northern Ireland",
    "Tanzania, United Republic of": "United Republic of Tanzania",
    "United States": "United States of America",
    "Uzbekistan, Republic of": "Uzbekistan",
    "Venezuela, República Bolivariana de": "Venezuela (Bolivarian Republic of)",
    "Vietnam": "Viet Nam",
    "Yemen, Republic of": "Yemen",
}


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

    pp = pp_raw[pp_raw["Element"].isin(_ELEMENTS_KEEP)].copy()

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
            & (pp3["lcu_per_slc_country"] - 1).abs() <= _TOL_NEAR_EQUAL_LCU_SLC,
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
            & (pp3["lcu_per_slc_country"] - 1).abs() <= _TOL_NEAR_EQUAL_LCU_SLC,
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
        & (pp_usd["anchor_year_dist"] <= _ANCHOR_MAX_YEAR_DIST)
        & pp_usd["anchor_usd"].notna() & (pp_usd["anchor_usd"] > 0)
        & pp_usd["usd_fx_implied"].notna() & (pp_usd["usd_fx_implied"] > 0)
    )
    pp_usd.loc[use_anchor, "times_off_closest_obs"] = np.maximum(
        pp_usd.loc[use_anchor, "usd_fx_implied"] / pp_usd.loc[use_anchor, "anchor_usd"],
        pp_usd.loc[use_anchor, "anchor_usd"] / pp_usd.loc[use_anchor, "usd_fx_implied"],
    )
    bad_anchor = use_anchor & (pp_usd["times_off_closest_obs"] > _ANCHOR_MAX_TIMES_OFF)
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
        (pp_usd["global_ratio"] > _GLOBAL_MED_MAX_TIMES_OFF)
        | (pp_usd["global_ratio"] < 1 / _GLOBAL_MED_MAX_TIMES_OFF)
    )
    logger.warning("Global median QC dropping %d rows", bad_global.sum())
    pp_usd.loc[bad_global, ["usd_fx_implied", "usd_filled", "usd_source"]] = [
        np.nan, np.nan, "implausible_vs_global_median_10x",
    ]

    # 15d) QC: FX reliability by country
    logger.info("=== 15d) QC: FX RELIABILITY BY COUNTRY ===")
    qc = (
        pp_usd.loc[
            (pp_usd["year"] >= _QC_START_YEAR)
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
            (country_qc["n_overlap"] >= _QC_MIN_OVERLAP)
            & (
                (country_qc["median_times_off"] > _QC_BAD_MEDIAN_TIMES_OFF)
                | (country_qc["share_over_3x"] > _QC_BAD_SHARE_OVER_3X)
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


def pixel_area_km2(lat_deg: np.ndarray, res_deg: float = PIXEL_RES_DEG) -> np.ndarray:
    """
    Pixel area in km² for each row of a lat/lon grid (spherical Earth).

    Parameters
    ----------
    lat_deg : 1-D array
        Latitude of each pixel-row centre (degrees, north-positive).
    res_deg : float
        Angular resolution in degrees (default 0.05° ≈ 5 km).

    Returns
    -------
    1-D array of areas in km², one per row.

    See Also
    --------
    crop_benefits.raster.grid.pixel_area_km2 :
        Complementary function that accepts a rasterio Affine transform and
        grid dimensions instead of a raw latitude array.  Use *that* function
        when you only have a rasterio metadata dict; use *this* function when
        you already have an explicit latitude array (e.g. inside
        ``build_area_km2_raster``).
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

# US CPI-U annual averages (BLS, all urban consumers). Prices are nominal in the year
# FAOSTAT reports them, so a target year needs a deflator, and a deflator needs an index.
CPI_BY_YEAR = {
    1993: 144.5,
    1994: 148.2,
    1995: 152.4,
    1996: 156.9,
    1997: 160.5,
    1998: 163.0,
    1999: 166.6,
    2000: 172.2,
    2001: 177.1,
    2002: 179.9,
    2003: 184.0,
    2004: 188.9,
    2005: 195.3,
    2006: 201.6,
    2007: 207.3,
    2008: 215.3,
    2009: 214.5,
    2010: 218.1,
    2011: 224.9,
    2012: 229.6,
    2013: 232.9,
    2014: 236.7,
    2015: 237.0,
    2016: 240.0,
    2017: 245.1,
    2018: 251.1,
    2019: 255.7,
    2020: 258.8,
    2021: 270.9,
    2022: 292.7,
    2023: 305.1,
    2024: 314.9,
    2025: 321.9,
}

# The production rasters are dated 2020, so a price expressed in any other year's dollars
# has to be brought to 2020 before it multiplies them, or the year of the money and the
# year of the harvest disagree.
PRODUCTION_RASTER_YEAR = 2020


def usd_deflator(from_year, to_year):
    """CPI ratio converting dollars of one year into dollars of another.

    Args:
        from_year (int): the year the value is currently denominated in.
        to_year (int): the year wanted.

    Returns:
        float: multiply a `from_year` dollar amount by this to get `to_year` dollars.

    Raises:
        KeyError: if either year is outside the CPI table, which is deliberate. Silently
            returning 1.0 for an unknown year would leave the value undeflated and
            indistinguishable from a correct one.
    """
    return CPI_BY_YEAR[int(to_year)] / CPI_BY_YEAR[int(from_year)]


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

    Args:
        df_arabica_robusta (pd.DataFrame): area_code_m49 and prop_arabica, prop_robusta.

    Returns:
        dict: M49 country code (int) to the blended dependence ratio.
    """
    import pandas as pd
    df = df_arabica_robusta.dropna(subset=['area_code_m49']).copy()
    blended = (df['prop_arabica'].astype(float) * COFFEE_DEPENDENCE['arabica']
               + df['prop_robusta'].astype(float) * COFFEE_DEPENDENCE['robusta'])
    codes = pd.to_numeric(df['area_code_m49'], errors='coerce').astype('Int64')
    return dict(zip(codes, blended))


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

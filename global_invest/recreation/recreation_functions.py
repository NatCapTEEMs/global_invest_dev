"""Recreation/tourism science: site quality, gravity-model visits, travel-cost valuation.

Ported from the GEP recreation repo (NatCapTEEMs shared drive, Recreation subfolder); the method
constants below ARE the published method, so they live in code -- a change is a reviewed commit,
not an input/-copy edit (the sorting rule; landslide's constants follow the same pattern).

Method in one pass: LULC class shares + protected-area share -> a 0-3 environment class;
urban share + distance-to-road -> a 1-5 accessibility class; the two cross into a 1-9 site rank,
of which class 9 is a high-quality site. Population (residents for daily recreation, allocated
UNWTO overnights for tourism) generates visits into sites within 1-4 km Chebyshev rings via a
gravity model, and visits are valued at national per-km fuel cost times ring distance.

Raster ops are pure array functions (unit-testable) wrapped into pygeoprocessing.raster_calculator
closures by the callers below; nodata semantics are kept exactly as in the source repo so the
port stays anchor-comparable against its reference output.

todo A units question is FLAGGED, not fixed: the travel-cost distance is buffer_zone * pixel_size
with pixel_size read from the raster (degrees on a 4326 grid), while the fuel-cost column is USD
per KM. Port is faithful; resolve against the reference results_by_country.csv when the data and
anchor are staged, then fix here and in the method qmd together if confirmed.
"""
import os

import numpy as np
import pandas as pd
import pygeoprocessing
from osgeo import gdal

# --- Method constants (published science; see module docstring) ---
RECREATION_LULC_CLASSES = ('cropland', 'forest', 'grassland', 'othernat', 'urban', 'water')
RECREATION_LULC_SCORES = {'cropland': 0.4, 'forest': 1.0, 'grassland': 0.7,
                          'othernat': 0.85, 'urban': 0.05, 'water': 1.0}
RECREATION_PA_SCORE = 1.0
RECREATION_ENV_BINS = (1.5, 1.0, 0.5, 0.0)          # combined score > bin -> class 3, 2, 1, 0
RECREATION_URBAN_BINS = (0.9, 0.5, 0.1, 0.001, 0.0)  # urban share > bin -> class 4, 3, 2, 1, 0
RECREATION_ROAD_BINS = (50.0, 20.0, 5.0, 2.0, 0.0)   # road distance > bin -> class 0, 1, 2, 3, 4
RECREATION_URBAN_ROAD_MATRIX = np.array([            # rows: urban class, cols: road class -> 1-5
    [1, 1, 2, 3, 4],
    [1, 1, 2, 3, 4],
    [2, 2, 2, 4, 5],
    [3, 3, 4, 5, 5],
    [3, 4, 4, 5, 5]])
RECREATION_SITE_MATRIX = np.array([                  # rows: accessibility 1-5, cols: env 0-3 -> 1-9
    [1, 1, 4, 7],
    [1, 4, 4, 7],
    [2, 2, 8, 8],
    [3, 5, 5, 9],
    [3, 6, 6, 9]])
RECREATION_HQ_SITE_CLASS = 9
RECREATION_DIST_BUFFERS = (1, 2, 3, 4)               # Chebyshev rings, grid cells (~km at 1 km)
RECREATION_GRAVITY_K = (0.0132, 0.0267, 0.0518, 0.1067)      # per ring
RECREATION_GRAVITY_ALPHA = (0.00155, 0.00115, 0.00098, 0.00067)
RECREATION_WEEKS_PER_YEAR = 52
RECREATION_FUEL_COST_COL = 'gasoline_cost_usd_per_km_2019_gppdata'

INT_NDV = -1
FLOAT_NDV = -9999.0


# --- Pure array ops (the science; wrapped into raster_calculator closures below) ---
def environment_class_array(crop, forest, grass, othernat, urban, water, pa):
    """Weighted LULC-share composite + protected-area share -> environment class 0-3."""
    combined = (RECREATION_LULC_SCORES['cropland'] * crop
                + RECREATION_LULC_SCORES['forest'] * forest
                + RECREATION_LULC_SCORES['grassland'] * grass
                + RECREATION_LULC_SCORES['othernat'] * othernat
                + RECREATION_LULC_SCORES['urban'] * urban
                + RECREATION_LULC_SCORES['water'] * water
                + RECREATION_PA_SCORE * pa)
    return np.select(
        condlist=[combined > RECREATION_ENV_BINS[0], combined > RECREATION_ENV_BINS[1],
                  combined > RECREATION_ENV_BINS[2], combined >= RECREATION_ENV_BINS[3]],
        choicelist=[3, 2, 1, 0], default=0)


def accessibility_class_array(urban, roads):
    """Urban share x distance-to-road -> accessibility class 1-5 via the urban/road matrix."""
    urban_class = np.select(
        condlist=[urban > RECREATION_URBAN_BINS[0], urban > RECREATION_URBAN_BINS[1],
                  urban > RECREATION_URBAN_BINS[2], urban > RECREATION_URBAN_BINS[3],
                  urban >= RECREATION_URBAN_BINS[4]],
        choicelist=[4, 3, 2, 1, 0], default=0)
    road_class = np.select(
        condlist=[roads > RECREATION_ROAD_BINS[0], roads > RECREATION_ROAD_BINS[1],
                  roads > RECREATION_ROAD_BINS[2], roads > RECREATION_ROAD_BINS[3],
                  roads >= RECREATION_ROAD_BINS[4]],
        choicelist=[0, 1, 2, 3, 4], default=0)
    return RECREATION_URBAN_ROAD_MATRIX[urban_class, road_class].astype(np.int32)


def site_rank_array(acc_class, env_class):
    """Accessibility class 1-5 x environment class 0-3 -> site rank 1-9."""
    return RECREATION_SITE_MATRIX[acc_class - 1, env_class].astype(np.int16)


def visit_potential_array(population, country_id, k, alpha):
    """Gravity model: annual visits a population cell would generate if in the given ring."""
    result = np.full_like(population, FLOAT_NDV, dtype=np.float32)
    valid_mask = (~np.isnan(population) & ~np.isnan(country_id)
                  & (population > 0) & (country_id >= 0))
    if np.any(valid_mask):
        weekly_visits = (1 + k) / (k + np.exp(-alpha * population[valid_mask]))
        result[valid_mask] = weekly_visits * RECREATION_WEEKS_PER_YEAR
        result[~valid_mask] = 0
    return result


def value_potential_array(population, country_id, cost_lookup, k, alpha, buffer_zone, pixel_size):
    """Visits x national travel cost for the given ring (see the flagged units note above)."""
    max_country_id = len(cost_lookup) - 1
    result = np.full_like(population, FLOAT_NDV, dtype=np.float32)
    valid_mask = (~np.isnan(population) & ~np.isnan(country_id)
                  & (population > 0) & (country_id >= 0) & (country_id <= max_country_id))
    if np.any(valid_mask):
        weekly_visits = (1 + k) / (k + np.exp(-alpha * population[valid_mask]))
        annual_visits = weekly_visits * RECREATION_WEEKS_PER_YEAR
        ring_distance = buffer_zone * pixel_size
        travel_costs = cost_lookup[country_id[valid_mask].astype(int)] * (ring_distance - 0.5 * pixel_size)
        result[valid_mask] = annual_visits * travel_costs
        result[~valid_mask] = 0
    return result


def allocate_overnights_array(hotel_array, country_array, country_overnights_map):
    """National overnight totals spread proportionally over each country's hotel pixels."""
    result = np.zeros_like(hotel_array, dtype=np.float32)
    for country_id, total_overnights in country_overnights_map.items():
        country_mask = (country_array == country_id)
        hotels_in_country = np.sum(hotel_array[country_mask])
        if hotels_in_country > 0:
            result[country_mask] = (hotel_array[country_mask] / hotels_in_country) * total_overnights
    return result


# --- Overnights (UNWTO panel + hotel allocation) ---
# The workbook keeps domestic and inbound accommodation on their own sheets, and the panel is
# the two stacked.
UNWTO_ACCOMMODATION_SHEETS = {'domestic': 'Domestic Tourism-Accommodation',
                              'international': 'Inbound Tourism-Accommodation'}
# Each sheet opens with banner rows above its real header, so the table is located by its first
# country row rather than by a fixed offset.
UNWTO_FIRST_COUNTRY = 'AFGHANISTAN'
UNWTO_COUNTRY_COLUMN_INDEX = 3


def unwto_header_row_index(banner_sheet):
    """Rows to skip so that a re-read starts at the sheet's own header.

    Args:
        banner_sheet (pd.DataFrame): the sheet read straight through, so its banner row is
            standing in as the header and the real header is its first row.

    Returns:
        int: the position of the first country row, which is the header row's position once the
        banner is no longer consuming a row.
    """
    country_column = banner_sheet.iloc[:, UNWTO_COUNTRY_COLUMN_INDEX]
    return banner_sheet[country_column == UNWTO_FIRST_COUNTRY].index[0]


def extract_clean_overnights(df, tourism_type):
    """One UNWTO accommodation sheet -> tidy (iso3_r250_id, name, type, year, overnights)."""
    df = df.copy()
    df.columns = df.columns.map(str)
    iso3_r250_id_col = 'C.'
    country_col = 'Basic data and indicators'
    indicator_col = 'Unnamed: 6'
    type_col = 'Unnamed: 5'
    for col in (iso3_r250_id_col, country_col, indicator_col, type_col):
        df[col] = df[col].ffill()

    overnight_df = df[df[indicator_col].str.contains('Overnights', na=False)]
    year_cols = [col for col in df.columns if col.isdigit()]
    overnight_df = overnight_df[[iso3_r250_id_col, country_col, type_col] + year_cols]

    tidy = overnight_df.melt(id_vars=[iso3_r250_id_col, country_col, type_col],
                             var_name='year', value_name='overnights')
    tidy = tidy.rename(columns={iso3_r250_id_col: 'iso3_r250_id', country_col: 'unwto_name',
                                type_col: 'overnight_type'})
    tidy['tourism_type'] = tourism_type
    tidy['year'] = tidy['year'].astype(int)
    tidy['overnights'] = pd.to_numeric(tidy['overnights'], errors='coerce')
    return tidy


def clean_unwto_data(sheets_by_tourism_type):
    """UNWTO accommodation sheets -> a wide overnight panel per country-year.

    The sheets are tidied and pivoted; per tourism type the combined overnights column prefers
    the Total row and falls back to Hotels-and-similar where Total is missing.

    Args:
        sheets_by_tourism_type (dict): tourism type -> that type's accommodation sheet, read
            from its own header row down (the task module's read_unwto_sheets).

    Returns:
        pd.DataFrame: iso3_r250_id, unwto_name, year and the per-type overnight columns.
    """
    rows_by_type = {tourism_type: extract_clean_overnights(sheet, tourism_type)
                    for tourism_type, sheet in sheets_by_tourism_type.items()}
    panel_df = pd.concat(list(rows_by_type.values()), ignore_index=True)
    panel_df['overnight_type'] = panel_df['overnight_type'].str.lower().str.strip()

    pivoted = panel_df.pivot_table(index=['iso3_r250_id', 'unwto_name', 'year'],
                                   columns=['overnight_type', 'tourism_type'],
                                   values='overnights', aggfunc='first').reset_index()
    pivoted.columns = ['iso3_r250_id', 'unwto_name', 'year'] + [
        f'{overnight_type}_overnights_{tourism_type}'
        for overnight_type, tourism_type in pivoted.columns[3:]]

    for tourism_type in ('domestic', 'international'):
        total_col = f'total_overnights_{tourism_type}'
        hotel_col = f'hotels and similar establishments_overnights_{tourism_type}'
        pivoted[f'overnights_{tourism_type}'] = np.where(
            pivoted[total_col].notna(), pivoted[total_col], pivoted[hotel_col])

    pivoted = pivoted.rename(columns={
        'hotels and similar establishments_overnights_domestic': 'hotel_overnights_domestic',
        'hotels and similar establishments_overnights_international': 'hotel_overnights_international'})
    final_cols = ['iso3_r250_id', 'unwto_name', 'year']
    for tourism_type in ('domestic', 'international'):
        final_cols += [f'total_overnights_{tourism_type}', f'hotel_overnights_{tourism_type}',
                       f'overnights_{tourism_type}']
    final_df = pivoted[[col for col in final_cols if col in pivoted.columns]]
    return final_df


def build_country_overnights_map(overnight_df, target_year):
    """Panel -> {iso3_r250_id: domestic + international overnights} for the target year."""
    year_data = overnight_df[overnight_df['year'] == target_year]
    country_overnights_map = {}
    for _, row in year_data.iterrows():
        total_overnights = 0
        for col in ('overnights_domestic', 'overnights_international'):
            if col in row and pd.notna(row[col]):
                total_overnights += row[col]
        if total_overnights > 0:
            country_overnights_map[row['iso3_r250_id']] = (
                country_overnights_map.get(row['iso3_r250_id'], 0) + total_overnights)
    return country_overnights_map



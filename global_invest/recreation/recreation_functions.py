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


# --- Raster-chain steps ---
def calculate_environment_index(lulc_share_paths, pa_share_path, output_path):
    """LULC class-share rasters + PA share -> environment class raster (0-3)."""
    def op(crop, forest, grass, othernat, urban, water, pa):
        return environment_class_array(crop, forest, grass, othernat, urban, water, pa)
    base_list = [(lulc_share_paths[c], 1) for c in RECREATION_LULC_CLASSES] + [(pa_share_path, 1)]
    pygeoprocessing.raster_calculator(base_list, op, output_path, gdal.GDT_Int32, INT_NDV,
                                      calc_raster_stats=True)


def calculate_accessibility_index(urban_share_path, road_distance_path, output_path):
    """Urban share + distance-to-road rasters -> accessibility class raster (1-5)."""
    pygeoprocessing.raster_calculator(
        [(urban_share_path, 1), (road_distance_path, 1)], accessibility_class_array,
        output_path, gdal.GDT_Int32, INT_NDV)


def rank_recreation_sites(accessibility_index_path, environment_index_path, output_path):
    pygeoprocessing.raster_calculator(
        [(accessibility_index_path, 1), (environment_index_path, 1)], site_rank_array,
        output_path, gdal.GDT_Int32, INT_NDV)


def extract_high_quality_sites(ranked_sites_path, output_path):
    def op(site_class):
        return np.where(site_class == RECREATION_HQ_SITE_CLASS, 1, 0).astype(np.int32)
    pygeoprocessing.raster_calculator([(ranked_sites_path, 1)], op, output_path,
                                      gdal.GDT_Int32, INT_NDV)


def create_distance_kernel(kernel_path, buffer_zone):
    """Chebyshev ring kernel: ring 1 is the full 3x3 block (center included), ring n>1 the
    cells exactly n steps away."""
    kernel_size = 2 * buffer_zone + 1
    offsets = np.abs(np.arange(kernel_size) - buffer_zone)
    chebyshev = np.maximum.outer(offsets, offsets)
    if buffer_zone == 1:
        kernel_array = (chebyshev <= buffer_zone).astype(np.float32)
    else:
        kernel_array = (chebyshev == buffer_zone).astype(np.float32)
    pygeoprocessing.numpy_array_to_raster(
        kernel_array, target_nodata=None, pixel_size=None, origin=None,
        projection_wkt=None, target_path=kernel_path)


def add_to_total(total_path, addition_path):
    """total += addition in place, nodata-aware, clamped at zero against float drift."""
    def op(total, addition):
        valid_total = total != FLOAT_NDV
        valid_addition = addition != FLOAT_NDV
        result = np.full_like(total, FLOAT_NDV, dtype=np.float32)
        both = valid_total & valid_addition
        result[both] = total[both] + addition[both]
        result[valid_total & ~valid_addition] = total[valid_total & ~valid_addition]
        result[~valid_total & valid_addition] = addition[~valid_total & valid_addition]
        valid_result = result != FLOAT_NDV
        result[valid_result] = np.maximum(result[valid_result], 0.0)
        return result
    temporary_path = total_path.replace('.tif', '_partial.tif')
    pygeoprocessing.raster_calculator([(total_path, 1), (addition_path, 1)], op,
                                      temporary_path, gdal.GDT_Float32, FLOAT_NDV)
    os.replace(temporary_path, total_path)


def mask_to_sites(input_path, sites_path, output_path):
    """Keep values only where the high-quality-site raster is 1."""
    def op(values, sites):
        result = np.full_like(values, FLOAT_NDV, dtype=np.float32)
        result[sites == 1] = values[sites == 1]
        return result
    pygeoprocessing.raster_calculator([(input_path, 1), (sites_path, 1)], op,
                                      output_path, gdal.GDT_Float32, FLOAT_NDV)


def calculate_kernel_recreation(population_path, country_id_path, fuel_cost_df,
                                hq_sites_path, visits_output_path, value_output_path, working_dir):
    """The visit/value engine shared by daily recreation and tourism: for each Chebyshev ring,
    build the potential surfaces and convolve them over the population, then mask the summed
    totals to high-quality sites. population_path is residents (daily) or allocated overnights
    (tourism); everything else is identical between the two."""
    country_costs = dict(zip(fuel_cost_df['iso3_r250_id'], fuel_cost_df[RECREATION_FUEL_COST_COL]))
    max_country_id = max(country_costs.keys()) if country_costs else 0
    cost_lookup = np.zeros(max_country_id + 1, dtype=np.float32)
    for country_id, cost in country_costs.items():
        cost_lookup[country_id] = cost

    pixel_size = abs(pygeoprocessing.get_raster_info(population_path)['pixel_size'][0])
    os.makedirs(working_dir, exist_ok=True)
    total_visits_path = os.path.join(working_dir, 'total_visits.tif')
    total_value_path = os.path.join(working_dir, 'total_value.tif')
    for total_path in (total_visits_path, total_value_path):
        pygeoprocessing.new_raster_from_base(population_path, total_path, gdal.GDT_Float32,
                                             [FLOAT_NDV], fill_value_list=[0.0])

    for buffer_index, buffer_zone in enumerate(RECREATION_DIST_BUFFERS):
        k = RECREATION_GRAVITY_K[buffer_index]
        alpha = RECREATION_GRAVITY_ALPHA[buffer_index]

        kernel_path = os.path.join(working_dir, f'kernel_buffer_{buffer_zone}.tif')
        create_distance_kernel(kernel_path, buffer_zone)

        visit_potential_path = os.path.join(working_dir, f'pop_visit_potential_buffer_{buffer_zone}.tif')
        pygeoprocessing.raster_calculator(
            [(population_path, 1), (country_id_path, 1)],
            lambda population, country_id: visit_potential_array(population, country_id, k, alpha),
            visit_potential_path, gdal.GDT_Float32, FLOAT_NDV)

        value_potential_path = os.path.join(working_dir, f'pop_value_potential_buffer_{buffer_zone}.tif')
        pygeoprocessing.raster_calculator(
            [(population_path, 1), (country_id_path, 1)],
            lambda population, country_id: value_potential_array(
                population, country_id, cost_lookup, k, alpha, buffer_zone, pixel_size),
            value_potential_path, gdal.GDT_Float32, FLOAT_NDV)

        for potential_path, buffer_output_name, total_path in (
                (visit_potential_path, f'visits_buffer_{buffer_zone}.tif', total_visits_path),
                (value_potential_path, f'value_buffer_{buffer_zone}.tif', total_value_path)):
            buffer_output_path = os.path.join(working_dir, buffer_output_name)
            pygeoprocessing.convolve_2d(
                (potential_path, 1), (kernel_path, 1), buffer_output_path,
                ignore_nodata_and_edges=False, mask_nodata=True, normalize_kernel=False,
                target_datatype=gdal.GDT_Float32, target_nodata=FLOAT_NDV, working_dir=working_dir)
            add_to_total(total_path, buffer_output_path)

    mask_to_sites(total_visits_path, hq_sites_path, visits_output_path)
    mask_to_sites(total_value_path, hq_sites_path, value_output_path)


# --- Overnights (UNWTO panel + hotel allocation) ---
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


def clean_unwto_data(unwto_xlsx_path):
    """UNWTO all-data workbook -> a wide overnight panel per country-year, written as CSV.

    Domestic and inbound accommodation sheets are located by their AFGHANISTAN header row,
    tidied, and pivoted; per tourism type the combined overnights column prefers the Total
    row and falls back to Hotels-and-similar where Total is missing."""
    rows_by_type = {}
    for tourism_type, sheet_name in (('domestic', 'Domestic Tourism-Accommodation'),
                                     ('international', 'Inbound Tourism-Accommodation')):
        raw = pd.read_excel(unwto_xlsx_path, sheet_name=sheet_name)
        start_index = raw[raw.iloc[:, 3] == 'AFGHANISTAN'].index[0]
        data = pd.read_excel(unwto_xlsx_path, sheet_name=sheet_name, skiprows=start_index)
        rows_by_type[tourism_type] = extract_clean_overnights(data, tourism_type)
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


def rasterize_presence(vector_path, ref_raster_path, output_path):
    """Vector features -> additive presence counts on the reference grid (hotel points)."""
    ref_info = pygeoprocessing.geoprocessing.get_raster_info(ref_raster_path)
    pygeoprocessing.geoprocessing.create_raster_from_bounding_box(
        target_bounding_box=ref_info['bounding_box'], target_raster_path=output_path,
        target_pixel_size=ref_info['pixel_size'], target_pixel_type=gdal.GDT_Int16,
        target_srs_wkt=ref_info['projection_wkt'], target_nodata=INT_NDV, fill_value=0)
    pygeoprocessing.geoprocessing.rasterize(
        vector_path=vector_path, target_raster_path=output_path, burn_values=[1],
        option_list=['ALL_TOUCHED=TRUE', 'MERGE_ALG=ADD'])


def rasterize_id_column(vector_path, ref_raster_path, id_column, output_path):
    """Vector id column -> raster on the reference grid (country ids)."""
    ref_info = pygeoprocessing.geoprocessing.get_raster_info(ref_raster_path)
    pygeoprocessing.geoprocessing.create_raster_from_bounding_box(
        target_bounding_box=ref_info['bounding_box'], target_raster_path=output_path,
        target_pixel_size=ref_info['pixel_size'], target_pixel_type=ref_info['datatype'],
        target_srs_wkt=ref_info['projection_wkt'], target_nodata=ref_info['nodata'][0],
        fill_value=0)
    pygeoprocessing.geoprocessing.rasterize(
        vector_path=vector_path, target_raster_path=output_path, burn_values=None,
        option_list=['ALL_TOUCHED=TRUE', f'ATTRIBUTE={id_column}'])


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


def allocate_overnights_raster(country_overnights_map, hotel_raster_path, country_id_path,
                               output_path):
    pygeoprocessing.raster_calculator(
        [(hotel_raster_path, 1), (country_id_path, 1)],
        lambda hotels, countries: allocate_overnights_array(hotels, countries,
                                                           country_overnights_map),
        output_path, gdal.GDT_Float32, FLOAT_NDV)


def sum_rasters_by_country_id(value_raster_paths, country_id_path):
    """Country-id raster + named value rasters -> one row per iso3_r250_id with nansums.

    Whole-array aggregation (the source repo's approach, kept for anchor fidelity); at 1 km
    global this holds several ~4 GB float arrays, so run it on a machine with headroom.
    """
    country_id_array = pygeoprocessing.raster_to_numpy_array(country_id_path)
    ndv = pygeoprocessing.get_raster_info(country_id_path)['nodata'][0]
    valid_mask = (country_id_array != ndv) & (~np.isnan(country_id_array))
    country_ids = np.unique(country_id_array[valid_mask])
    country_ids = country_ids[country_ids > 0]

    results_arrays = {}
    for key, path in value_raster_paths.items():
        array = pygeoprocessing.raster_to_numpy_array(path)
        array_ndv = pygeoprocessing.get_raster_info(path)['nodata'][0]
        results_arrays[key] = np.where(array == array_ndv, np.nan, array)

    rows = []
    for country_id in country_ids:
        mask = (country_id_array == country_id)
        row = {'iso3_r250_id': int(country_id)}
        for key, array in results_arrays.items():
            row[key] = np.nansum(array[mask])
        rows.append(row)
    return pd.DataFrame(rows)

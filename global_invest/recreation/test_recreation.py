"""Unit tests for the recreation GEP port (synthetic, self-contained).

Pins the ported science against hand-derived values so the port stays anchor-comparable: the
classification ops and matrices, the Chebyshev ring kernels, the gravity model, the overnight
allocation, and the per-country aggregation (born-r250 -- the id raster IS iso3_r250_id).
The reference-output comparison (results_by_country.csv from the source pipeline) is the
LIVE-run gate once the drive data is staged; these tests cover the math, not the data.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from global_invest.recreation import recreation_functions as rf
from global_invest.recreation import recreation_tasks as rt


def test_environment_class_matches_hand_derived_bins():
    # Pure forest + PA share 1 -> combined 2.0 -> class 3; pure urban, no PA -> 0.05 -> class 0;
    # pure grassland (0.7) -> class 1; grassland + PA 0.5 -> 1.2 -> class 2.
    zeros = np.zeros(4, dtype='float32')
    shares = {c: zeros.copy() for c in rf.RECREATION_LULC_CLASSES}
    pa = np.array([1.0, 0.0, 0.0, 0.5], dtype='float32')
    shares['forest'][0] = 1.0
    shares['urban'][1] = 1.0
    shares['grassland'][2] = 1.0
    shares['grassland'][3] = 1.0
    env = rf.environment_class_array(shares['cropland'], shares['forest'], shares['grassland'],
                                     shares['othernat'], shares['urban'], shares['water'], pa)
    assert list(env) == [3, 0, 1, 2]


def test_accessibility_and_site_rank_follow_the_matrices():
    # Dense urban (0.95) next to a road (1 km) -> urban class 4, road class 4 -> matrix 5;
    # remote wilderness (urban 0, road 100 km) -> urban 0, road 0 -> matrix 1.
    urban = np.array([0.95, 0.0], dtype='float32')
    roads = np.array([1.0, 100.0], dtype='float32')
    acc = rf.accessibility_class_array(urban, roads)
    assert list(acc) == [5, 1]
    # Site rank: accessibility 5 x env 3 -> 9 (high-quality); accessibility 1 x env 0 -> 1.
    rank = rf.site_rank_array(np.array([5, 1]), np.array([3, 0]))
    assert list(rank) == [rf.RECREATION_HQ_SITE_CLASS, 1]


def test_distance_kernels_ring_one_is_full_block_rings_beyond_are_hollow(tmp_path):
    import pygeoprocessing
    k1 = str(tmp_path / 'k1.tif')
    k3 = str(tmp_path / 'k3.tif')
    rt.create_distance_kernel(k1, 1)
    rt.create_distance_kernel(k3, 3)
    a1 = pygeoprocessing.raster_to_numpy_array(k1)
    a3 = pygeoprocessing.raster_to_numpy_array(k3)
    assert a1.shape == (3, 3) and a1.sum() == 9          # ring 1: full block, center included
    assert a3.shape == (7, 7) and a3.sum() == 24         # ring 3: hollow ring only
    assert a3[3, 3] == 0 and a3[0, 0] == 1


def test_gravity_visits_are_bounded_and_monotone_in_population():
    country_id = np.zeros(3, dtype='float32')
    population = np.array([1.0, 1000.0, 1e9], dtype='float32')
    k, alpha = rf.RECREATION_GRAVITY_K[0], rf.RECREATION_GRAVITY_ALPHA[0]
    visits = rf.visit_potential_array(population, country_id, k, alpha)
    assert np.all(np.diff(visits) > 0)                                   # more people, more visits
    assert np.isclose(visits[-1], (1 + k) / k * rf.RECREATION_WEEKS_PER_YEAR, rtol=1e-4)
    # zero population generates nothing WHEN the block has any valid pixel (source semantics,
    # kept for anchor fidelity: an entirely-invalid block stays nodata, and the downstream
    # convolution treats nodata and 0 identically -- both contribute nothing).
    mixed = rf.visit_potential_array(np.array([0.0, 10.0], dtype='float32'),
                                     np.zeros(2, dtype='float32'), k, alpha)
    assert mixed[0] == 0 and mixed[1] > 0
    all_invalid = rf.visit_potential_array(np.zeros(1, dtype='float32'), country_id[:1], k, alpha)
    assert all_invalid[0] == rf.FLOAT_NDV


def test_value_potential_is_visits_times_national_cost_times_ring_distance():
    population = np.array([1000.0], dtype='float32')
    country_id = np.array([2.0], dtype='float32')
    cost_lookup = np.array([0.0, 0.0, 0.5], dtype='float32')             # country 2: 0.5 USD/km
    k, alpha = rf.RECREATION_GRAVITY_K[1], rf.RECREATION_GRAVITY_ALPHA[1]
    buffer_zone, pixel_size = 2, 1.0
    visits = rf.visit_potential_array(population, country_id, k, alpha)
    value = rf.value_potential_array(population, country_id, cost_lookup, k, alpha,
                                     buffer_zone, pixel_size)
    expected = visits[0] * 0.5 * (buffer_zone * pixel_size - 0.5 * pixel_size)
    assert np.isclose(value[0], expected, rtol=1e-5)


def test_overnight_allocation_splits_national_totals_by_hotel_share():
    # Country 1: 3 hotel pixels weighted 1/1/2 sharing 400 overnights -> 100/100/200.
    # Country 2 has overnights but no hotels -> allocates nothing (stays 0).
    hotels = np.array([1, 1, 2, 0], dtype='float32')
    countries = np.array([1, 1, 1, 2], dtype='float32')
    out = rf.allocate_overnights_array(hotels, countries, {1: 400.0, 2: 999.0})
    assert list(out) == [100.0, 100.0, 200.0, 0.0]


def test_build_country_overnights_map_sums_domestic_and_international_for_target_year():
    panel = pd.DataFrame({
        'iso3_r250_id': [8, 8, 12],
        'year': [2019, 2018, 2019],
        'overnights_domestic': [100.0, 5.0, np.nan],
        'overnights_international': [50.0, 5.0, 30.0]})
    result = rf.build_country_overnights_map(panel, 2019)
    assert result == {8: 150.0, 12: 30.0}


def test_unwto_extraction_tidies_the_sheet_layout():
    # The UNWTO sheet layout: merged cells arrive as NaN (ffilled), year columns as strings,
    # the Overnights indicator row carries the values.
    raw = pd.DataFrame({
        'C.': [8.0, np.nan, np.nan],
        'Basic data and indicators': ['ALBANIA', np.nan, np.nan],
        'Unnamed: 5': ['Total', np.nan, 'Hotels and similar establishments'],
        'Unnamed: 6': ['Arrivals', 'Overnights', 'Overnights'],
        '2019': [10.0, 200.0, 150.0]})
    tidy = rf.extract_clean_overnights(raw, 'domestic')
    assert set(tidy['overnight_type']) == {'Total', 'Hotels and similar establishments'}
    assert tidy['overnights'].tolist() == [200.0, 150.0]
    assert (tidy['iso3_r250_id'] == 8.0).all() and (tidy['year'] == 2019).all()


def _unwto_sheet(total_overnights, hotel_overnights):
    """One accommodation sheet as read_unwto_sheets hands it over: two countries, each with a
    Total row and a Hotels row, and the merged cells under a country name arriving as NaN. The
    second country always reports a Total, which is what holds the Total column open in the
    pivot when the first country's Total is missing."""
    return pd.DataFrame({
        'C.': [8.0, np.nan, 12.0, np.nan],
        'Basic data and indicators': ['ALBANIA', np.nan, 'BELGIUM', np.nan],
        'Unnamed: 5': ['Total', 'Hotels and similar establishments'] * 2,
        'Unnamed: 6': ['Overnights'] * 4,
        '2019': [total_overnights, hotel_overnights, 400.0, 300.0]})


def test_clean_unwto_data_prefers_the_total_row_and_falls_back_to_hotels():
    panel = rf.clean_unwto_data({'domestic': _unwto_sheet(200.0, 150.0),
                                 'international': _unwto_sheet(np.nan, 50.0)})
    albania = panel[panel['iso3_r250_id'] == 8.0].iloc[0]
    assert albania['overnights_domestic'] == 200.0        # Total present, so Total is used
    assert albania['overnights_international'] == 50.0    # Total missing, so Hotels stands in
    assert albania['hotel_overnights_domestic'] == 150.0


def test_task_reader_finds_each_sheet_header_below_the_banner_row(tmp_path):
    """The sheets open with a banner row, so a straight read puts the banner in the header and
    the real header in the first row. The country column is what locates the table."""
    path = str(tmp_path / 'unwto_all_data.xlsx')
    with pd.ExcelWriter(path) as writer:
        for tourism_type, sheet_name in rf.UNWTO_ACCOMMODATION_SHEETS.items():
            total = 200.0 if tourism_type == 'domestic' else np.nan
            pd.DataFrame([
                ['UNWTO country data', None, None, None, None, None, None, None],
                ['S.', 'x', 'C.', 'Basic data and indicators', 'y', None, None, 2019],
                [1, None, 4, rf.UNWTO_FIRST_COUNTRY, None, 'Total', 'Overnights', total],
                [None, None, None, None, None, 'Hotels and similar establishments',
                 'Overnights', 150.0],
                [2, None, 8, 'ALBANIA', None, 'Total', 'Overnights', 400.0],
                [None, None, None, None, None, 'Hotels and similar establishments',
                 'Overnights', 300.0],
            ]).to_excel(writer, sheet_name=sheet_name, header=False, index=False)

    sheets = rt.read_unwto_sheets(path)
    assert sorted(sheets) == ['domestic', 'international']
    panel = rf.clean_unwto_data(sheets)
    afghanistan = panel[panel['iso3_r250_id'] == 4.0].iloc[0]
    assert afghanistan['overnights_domestic'] == 200.0
    assert afghanistan['overnights_international'] == 150.0


def _write_tif(path, array, nodata=-9999.0):
    from osgeo import gdal, osr
    array = np.asarray(array, dtype='float32')
    h, w = array.shape
    ds = gdal.GetDriverByName('GTiff').Create(str(path), w, h, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((-180.0, 360.0 / w, 0.0, 90.0, 0.0, -180.0 / h))
    srs = osr.SpatialReference(); srs.ImportFromEPSG(4326); ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1); band.SetNoDataValue(nodata)
    band.WriteArray(array); band.FlushCache(); ds = None


def test_country_sums_key_on_the_id_raster_and_respect_nodata(tmp_path):
    ids = np.array([[1, 1], [2, -9999]], dtype='float32')
    vals = np.array([[10.0, 5.0], [2.0, 100.0]], dtype='float32')       # 100 sits on nodata id
    _write_tif(tmp_path / 'ids.tif', ids)
    _write_tif(tmp_path / 'vals.tif', vals)
    df = rt.sum_rasters_by_country_id({'daily_value': str(tmp_path / 'vals.tif')},
                                      str(tmp_path / 'ids.tif'))
    assert df.set_index('iso3_r250_id')['daily_value'].to_dict() == {1: 15.0, 2: 2.0}


def test_es_config_and_parameters_rows_hydrate_the_recreation_surface(tmp_path):
    from global_invest import utilities
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'recreation', log=lambda *a: None)
    assert p.gep_base_year == 2019
    assert p.gep_regions_id_col == 'iso3_r250_id'                        # born-r250 aggregation
    assert p.gep_regions_input_path.endswith('cartographic/ee/ee_r250.gpkg')
    utilities.hydrate_es_parameters(p, 'recreation', log=lambda *a: None)
    assert '{lulc_class}' in p.recreation_lulc_share_path_template       # formatted at use
    assert p.recreation_fuel_cost_path.endswith('feul_cost_per_km_2019_2datasources_iso3.csv')

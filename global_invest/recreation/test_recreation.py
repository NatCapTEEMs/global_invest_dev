"""Unit tests for the recreation GEP port (synthetic, self-contained).

Pins the ported science against hand-derived values: the classification matrices, the band
maths and its conservation, the demand-parameter validation, the UNWTO machinery with its
nearest-year fallback, the air-travel valuation, and the one structural change the port
makes -- the flow engine windows by country where the source windows by parameter group, and
one test computes both formulations on a shared toy world and requires the same rasters.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import signal

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
    rank = rf.site_rank_array(np.array([5, 1]), np.array([3, 0]))
    assert list(rank) == [rf.RECREATION_HQ_SITE_CLASS, 1]


def test_band_edges_are_grid_aware_and_shares_sum_to_the_cutoff_cdf():
    cutoff, edges = rf.compute_group_bands(pixel_size_km=1.0, max_distance_km=50.0, max_bands=8)
    assert cutoff == 50.0 and len(edges) == 9 and edges[0] == 0.0 and edges[-1] == 50.0
    # A cutoff the grid cannot subdivide collapses to one band.
    _, coarse = rf.compute_group_bands(pixel_size_km=80.0, max_distance_km=50.0, max_bands=8)
    assert len(coarse) == 2
    # The band shares plus the beyond-cutoff share partition the whole budget.
    b = 0.35
    shares = [rf.cdf_exp(edges[k + 1], b) - rf.cdf_exp(edges[k], b) for k in range(8)]
    assert np.isclose(sum(shares) + (1.0 - rf.cdf_exp(cutoff, b)), 1.0)


def test_ring_kernels_partition_the_disc():
    # The first band includes the centre; later rings are hollow; together they tile the
    # cutoff disc with no overlap, so no pixel offset is counted in two bands.
    _, edges = rf.compute_group_bands(pixel_size_km=1.0, max_distance_km=4.0, max_bands=4)
    kernels = [rf.ring_kernel_array(edges[k], edges[k + 1], 1.0) for k in range(4)]
    assert kernels[0][1, 1] == 1 and kernels[1].sum() > 0
    size = kernels[-1].shape[0]
    stacked = np.zeros((size, size))
    for kernel in kernels:
        pad = (size - kernel.shape[0]) // 2
        stacked[pad:pad + kernel.shape[0], pad:pad + kernel.shape[0]] += kernel
    assert stacked.max() == 1.0                          # no offset in two bands
    yy, xx = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    inside = np.sqrt(xx ** 2 + yy ** 2) <= 4.0
    assert np.array_equal(stacked > 0, inside)           # and the disc is fully tiled


def test_param_validation_rejects_a_group_whose_countries_disagree():
    good = pd.DataFrame({
        'iso3_r250_id': [1, 2, 3], 'iso3_r250_name': ['A', 'B', 'C'],
        'param_group': [1, 1, 2], 'participation_param': [6.0, 6.0, 2.0],
        'distance_param': [0.35, 0.35, 0.45]})
    country_to_group, group_params = rf.validate_recreation_params(good)
    assert country_to_group == {1: 1, 2: 1, 3: 2}
    assert group_params[1] == {'a': 6.0, 'b': 0.35}
    bad = good.copy()
    bad.loc[1, 'participation_param'] = 7.0
    with pytest.raises(ValueError):
        rf.validate_recreation_params(bad)


def test_tourist_rate_prices_a_person_night_at_the_daily_share():
    group_params = {1: {'a': 102.50, 'b': 0.167}}
    night = rf.per_night_group_params(group_params)
    assert night[1]['a'] == pytest.approx(102.50 / 365.0)
    assert night[1]['b'] == 0.167
    assert group_params[1]['a'] == 102.50


def test_nearest_year_fallback_prefers_closer_then_later_years():
    panel = pd.DataFrame({
        'iso3_r250_id': [8, 8, 8, 12, 12],
        'unwto_name': ['ALBANIA'] * 3 + ['BELGIUM'] * 2,
        'year': [2016, 2018, 2020, 2015, 2021],
        'overnights_domestic': [1.0, 2.0, 3.0, 4.0, 5.0],
        'overnights_international': [0.0, 0.0, 0.0, 0.0, 0.0]})
    result, substitutions = rf.build_country_overnights_map(panel, 2019)
    # 2018 and 2020 tie at distance 1; the later year wins, as in the source's sort.
    assert result[8] == 3.0 * rf.UNWTO_OVERNIGHTS_UNIT_NIGHTS
    assert result[12] == 5.0 * rf.UNWTO_OVERNIGHTS_UNIT_NIGHTS   # 2021 at distance 2 beats 2015
    by_id = {row['iso3_r250_id']: row for row in substitutions}
    assert by_id[8]['year_used'] == 2020 and by_id[8]['year_distance'] == 1
    # A country with no positive overnights in any year is absent, never zero.
    empty, _ = rf.build_country_overnights_map(panel.assign(
        overnights_domestic=0.0, overnights_international=np.nan), 2019)
    assert empty == {}


def test_overnight_allocation_splits_national_totals_by_hotel_share():
    # The denominator is the NATIONAL hotel count handed in, so a raster block holding only
    # part of a country's hotels allocates only that part's share -- the per-block closure
    # it replaces handed the full national total to every such block (Spain: exactly 22x).
    hotels = np.array([1, 1, 2, 0], dtype='float32')
    countries = np.array([1, 1, 1, 2], dtype='float32')
    out = rf.allocate_overnights_array(hotels, countries, {1: 400.0, 2: 999.0}, {1: 8.0})
    assert list(out) == [50.0, 50.0, 100.0, 0.0]    # this block holds half of country 1's hotels
    full = rf.allocate_overnights_array(hotels, countries, {1: 400.0, 2: 999.0}, {1: 4.0})
    assert list(full) == [100.0, 100.0, 200.0, 0.0]


def test_overnights_map_reads_the_workbook_thousands_as_nights():
    panel = pd.DataFrame({
        'iso3_r250_id': [724], 'unwto_name': ['SPAIN'], 'year': [2019],
        'overnights_domestic': [170721.0], 'overnights_international': [299092.0]})
    result, _ = rf.build_country_overnights_map(panel, 2019)
    # The Units column says Thousands, so Spain's 469,813 panel units are 469.8M nights.
    assert result[724] == pytest.approx(469813.0 * 1000.0)


def test_unwto_extraction_tidies_the_sheet_layout():
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
    Total row and a Hotels row, and the merged cells under a country name arriving as NaN."""
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


def test_fuel_cost_prices_from_the_per_litre_column_at_the_gfei_economy():
    # The staged cost-per-km column multiplies the per-litre price by 7.1, the GFEI economy
    # read per kilometre instead of per hundred; pricing from the price column at 0.071 L/km
    # is the source read as labelled (USA 2019: $0.66/L -> $0.047/km).
    assert rf.RECREATION_FUEL_COST_COL == 'gasoline_price_usd_per_liter_2019_gppdata'
    assert rf.GFEI_LITERS_PER_KM == pytest.approx(7.1 / 100.0)
    assert 0.659750 * rf.GFEI_LITERS_PER_KM == pytest.approx(0.04684, abs=1e-4)


def test_air_travel_value_prices_arrivals_at_the_region_pair_fare():
    arrivals = pd.DataFrame({
        'iso3_r250_id': [8, 8, 8, 12],
        'origin_region': ['Europe', 'Other', 'Europe', 'Africa'],
        'year': [2019, 2019, 2017, 2018],
        'arrivals': [1000.0, 500.0, 999.0, 200.0]})
    crosswalk = pd.DataFrame({'iso3_r250_id': [8, 12], 'unwto_region': ['Europe', 'Africa']})
    airfare = pd.DataFrame({
        'origin_region': ['Europe', 'Africa'], 'destination_region': ['Europe', 'Africa'],
        'predicted_fare_2019_usd': [300.0, 400.0]})
    out = rf.air_travel_value_by_country(arrivals, crosswalk, airfare, 2019).set_index('iso3_r250_id')
    # Country 8 uses its exact 2019 rows: Europe arrivals priced, Other counted but unpriced.
    assert out.at[8, 'air_travel_value'] == 1000.0 * 300.0
    assert out.at[8, 'total_arrivals'] == 1500.0
    assert out.at[8, 'arrivals_coverage_share'] == pytest.approx(1000.0 / 1500.0)
    # Country 12 falls back to its nearest year (2018) and prices fully.
    assert out.at[12, 'air_travel_value'] == 200.0 * 400.0


def _write_tif(path, array, nodata=-9999.0):
    from osgeo import gdal, osr
    array = np.asarray(array, dtype='float32')
    h, w = array.shape
    ds = gdal.GetDriverByName('GTiff').Create(str(path), w, h, 1, gdal.GDT_Float32)
    # A toy geographic grid centred on the equator, so pixel size in km is uniform enough
    # for the ring geometry; 0.01 degrees to match the production grid's spacing.
    ds.SetGeoTransform((0.0, 0.01, 0.0, 0.2, 0.0, -0.01))
    srs = osr.SpatialReference(); srs.ImportFromEPSG(4326); ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1); band.SetNoDataValue(nodata)
    band.WriteArray(array); band.FlushCache(); ds = None


def _group_window_reference(pop, country_id, sites, country_to_group, group_params, cost_map,
                            pixel_size_km, max_distance_km, max_bands):
    """The SOURCE formulation, transcribed: one pass per parameter group over the full grid,
    the group's pixels selected by a group mask. What the port changes is only the windowing
    (per country), so this reference is what the engine must reproduce."""
    visits = np.zeros_like(pop, dtype=np.float64)
    value = np.zeros_like(pop, dtype=np.float64)
    unmet = np.zeros_like(pop, dtype=np.float64)
    site_ind = (sites == 1).astype(np.float64)
    cutoff, edges = rf.compute_group_bands(pixel_size_km, max_distance_km, max_bands)
    max_cid = max(cost_map)
    cost_lookup = np.zeros(max_cid + 1)
    for cid, price in cost_map.items():
        cost_lookup[cid] = price * rf.GFEI_LITERS_PER_KM
    for gid in sorted(set(country_to_group.values())):
        a, b = group_params[gid]['a'], group_params[gid]['b']
        group_countries = [c for c, g in country_to_group.items() if g == gid]
        group_mask = np.isin(country_id, group_countries)
        valid = (pop > 0) & group_mask
        budget = np.where(valid, pop * a, 0.0)
        cost_per_km = np.zeros_like(pop, dtype=np.float64)
        in_range = valid & (country_id >= 0) & (country_id <= max_cid)
        cost_per_km[in_range] = cost_lookup[country_id[in_range].astype(int)]
        for k in range(len(edges) - 1):
            r_lo, r_hi = edges[k], edges[k + 1]
            r_mid = 0.5 * (r_lo + r_hi)
            share = rf.cdf_exp(r_hi, b) - rf.cdf_exp(r_lo, b)
            q_band = budget * share
            kernel = rf.ring_kernel_array(r_lo, r_hi, pixel_size_km).astype(np.float64)
            denom = signal.fftconvolve(site_ind, kernel, mode='same')
            denom_safe = np.where(denom > 1e-6, denom, np.nan)
            q_visits = np.where(np.isfinite(denom_safe), q_band / denom_safe, 0.0)
            q_value = q_visits * cost_per_km * 2.0 * r_mid
            no_site = ~np.isfinite(denom_safe) & valid
            unmet[no_site] += q_band[no_site]
            visits += signal.fftconvolve(q_visits, kernel, mode='same') * site_ind
            value += signal.fftconvolve(q_value, kernel, mode='same') * site_ind
        unmet += np.where(valid, budget * (1.0 - rf.cdf_exp(cutoff, b)), 0.0)
    return visits, value, unmet


def test_country_windowed_engine_reproduces_the_group_windowed_formulation(tmp_path):
    """The port's one structural change, pinned: windowing by country gives the same visits,
    value and unmet rasters as the source's one-pass-per-group formulation, because country
    ids partition each group's pixels and the maths is pixel-local within the cutoff."""
    rng = np.random.default_rng(7)
    shape = (36, 54)
    country_id = np.full(shape, -9999.0, dtype='float32')
    country_id[4:30, 3:20] = 1
    country_id[4:30, 20:38] = 2
    country_id[8:26, 40:52] = 3
    pop = np.where(country_id > 0, rng.uniform(0, 80, shape), -9999.0).astype('float32')
    pop[6, 5] = 0.0                                       # a zero-population pixel stays inert
    sites = np.zeros(shape, dtype='float32')
    for row, col in [(10, 10), (11, 10), (20, 25), (5, 37), (12, 45), (25, 50)]:
        sites[row, col] = 1.0                             # one site just across a border
    country_to_group = {1: 1, 2: 1, 3: 2}                 # two countries share a group
    group_params = {1: {'a': 6.0, 'b': 0.35}, 2: {'a': 2.0, 'b': 0.6}}
    cost_map = {1: 0.1, 2: 0.25, 3: 0.4}

    paths = {}
    for name, array in (('pop', pop), ('country', country_id), ('sites', sites)):
        paths[name] = str(tmp_path / f'{name}.tif')
        _write_tif(paths[name], array)
    costs_path = str(tmp_path / 'costs.csv')
    pd.DataFrame({'iso3_r250_id': list(cost_map), rf.RECREATION_FUEL_COST_COL:
                  list(cost_map.values())}).to_csv(costs_path, index=False)
    outputs = {key: str(tmp_path / f'{key}.tif') for key in ('visits', 'value', 'unmet')}
    outputs.update({key: str(tmp_path / f'{key}.csv')
                    for key in ('site_table', 'group_table', 'missing_cost_iso3')})

    max_distance_km, max_bands = 5.0, 4
    rt.calculate_recreation_flows(
        {'hq_sites': paths['sites'], 'population': paths['pop'],
         'country_id': paths['country'], 'country_costs': costs_path},
        outputs, country_to_group, group_params, max_distance_km, max_bands)

    import pygeoprocessing
    pixel_size_km = rt.convert_pixel_size_to_km(paths['pop'], 0.01)
    ref_visits, ref_value, ref_unmet = _group_window_reference(
        pop.astype(np.float64), country_id, sites, country_to_group, group_params, cost_map,
        pixel_size_km, max_distance_km, max_bands)
    got = {key: pygeoprocessing.raster_to_numpy_array(outputs[key]) for key in
           ('visits', 'value', 'unmet')}
    for key, reference in (('visits', ref_visits), ('value', ref_value), ('unmet', ref_unmet)):
        assert np.allclose(got[key], reference.astype(np.float32), rtol=1e-4, atol=1e-3), key

    # Conservation, per group: budget = realized + unmet, and the visits raster carries
    # exactly the realized trips.
    qa = pd.read_csv(outputs['group_table'])
    assert np.allclose(qa['budget_total'], qa['realized_total'] + qa['unmet_total'], rtol=1e-9)
    assert np.isclose(got['visits'].sum(), qa['realized_total'].sum(), rtol=1e-4)


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
    assert float(p.recreation_max_distance_km) == 50.0
    assert p.recreation_airfare_matrix_path.endswith('airfare_2019_regions.csv')

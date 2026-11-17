"""Unit tests for the timber_provision port (committed-output anchored)."""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from global_invest import utilities
from global_invest.timber_provision import timber_provision_functions as tp


def _base_data_project():
    """A bare ProjectFlow, only for its base_data_dir.

    The anchors are inputs, so a test finds them the way a run does rather than by walking
    directories of its own.
    """
    import tempfile
    import hazelbean as hb
    return hb.ProjectFlow(project_dir=os.path.join(tempfile.mkdtemp(), 'anchors'))


REFERENCE_DIR = utilities.service_data_dir(_base_data_project(), 'timber_provision')


def test_join_reproduces_the_committed_anchor():
    timber = pd.read_csv(os.path.join(REFERENCE_DIR, 'timber_provision_gep.csv'))
    countries = timber[['iso3_r250_id', 'iso3_r250_label']]
    out = tp.timber_gep_by_country(timber, countries)
    assert len(out) == 250
    assert out['timber_provision_gep'].notna().sum() == 166
    assert np.isclose(out['timber_provision_gep'].sum(), timber['forestry_gep'].sum())
    assert np.isclose(out['timber_provision_gep'].sum() / 1e9, 88.74, rtol=1e-2)


def test_net_forest_return_is_the_appendix_decomposition():
    biomass = np.array([[10.0, 0.0], [4.0, 2.0]])
    price = np.array([[50.0, 50.0], [80.0, 80.0]])
    tcost = np.array([[5.0, 1.0], [0.5, 200.0]])
    net = tp.net_forest_return(biomass, price, tcost, 0.3)
    assert np.allclose(net, [[145.0, -1.0], [95.5, -152.0]])
    # A regional share raster broadcasts the same way a scalar does.
    share = np.array([[0.3, 0.3], [0.5, 0.5]])
    net_regional = tp.net_forest_return(biomass, price, tcost, share)
    assert np.allclose(net_regional, [[145.0, -1.0], [159.5, -120.0]])


def test_forest_value_keeps_only_managed_positive_net_return():
    net = np.array([[100.0, -3.0], [tp.NET_RETURN_NDV, 250.0]])
    managed = np.array([[True, True], [True, False]])
    value = tp.forest_value_from_net_return(net, managed)
    # Kept: managed and positive. Zeroed: negative, ndv, and positive-but-unmanaged.
    assert value.dtype == np.float32
    assert np.array_equal(value, np.float32([[100.0, 0.0], [0.0, 0.0]]))


def test_gep_by_zone_sums_pixels_per_country_id():
    value = np.array([[10.0, 20.0], [0.0, 5.0]], dtype=np.float32)
    zone_ids = np.array([[3, 3], [0, 5]])
    sums = utilities.sum_by_zone(value, zone_ids, n_zones=5)
    assert sums.shape == (6,)
    assert np.allclose(sums, [0.0, 0.0, 0.0, 30.0, 0.0, 5.0])
    # Blockwise accumulation is plain addition of successive blocks' arrays.
    total = sums + utilities.sum_by_zone(value, zone_ids, n_zones=5)
    countries = pd.DataFrame({'iso3_r250_id': [3, 5], 'iso3_r250_label': ['AAA', 'BBB']})
    df = tp.timber_gep_from_zone_sums(total, countries)
    assert df['timber_provision_gep'].tolist() == [60.0, 10.0]


def test_chain_round_trip_on_synthetic_layers():
    biomass = np.array([[10.0, 8.0], [6.0, 1.0]])
    price = np.full((2, 2), 60.0)
    tcost = np.array([[2.0, 2.0], [2.0, 100.0]])
    share = 0.3
    managed = np.array([[True, False], [True, True]])
    net = tp.net_forest_return(biomass, price, tcost, share)
    value = tp.forest_value_from_net_return(net, managed)
    # px (0,0) and (1,0) kept; (0,1) unmanaged; (1,1) negative net return.
    assert np.allclose(value, [[178.0, 0.0], [106.0, 0.0]])
    sums = utilities.sum_by_zone(value, np.array([[1, 1], [2, 2]]), n_zones=2)
    assert np.allclose(sums, [0.0, 178.0, 106.0])


def test_es_config_and_parameters_rows_hydrate_timber_provision(tmp_path):
    from global_invest import utilities
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'timber_provision', log=lambda *a: None)
    assert p.gep_base_year == 2019
    utilities.hydrate_es_parameters(p, 'timber_provision', log=lambda *a: None)
    assert p.timber_provision_gep_path.endswith('timber_provision_gep.csv')


def test_the_units_plausibility_diagnostic_favours_the_flat_sum_over_a_per_hectare_reading():
    """A recorded plausibility diagnostic on the raster's units, not a proof of them.

    The value raster is EPSG:4326 at 10 arc-second and carries NO unit metadata, so whether a cell
    holds a value or a per-hectare density cannot be read off the file. The committed output cannot
    settle it either, because it was produced by summing the same raster the same way.

    FAOSTAT gives an outside comparison. World industrial roundwood production in 2019 was 1.985
    billion m3, and the world export unit value was $111.01/m3, so the gross value of industrial
    roundwood at that proxy price was about $220bn. The pipeline's flat sum, $88.74bn, is 40
    percent of that, a plausible land factor share. Reading the raster as $/ha gives $708bn, 321
    percent of it, which a land share of the gross value is not expected to reach. The check
    records that the flat sum sits below the proxy gross and the per-hectare reading above three
    times it. It favours the flat reading. It does not prove the units, because the comparison
    rests on an export-price proxy.
    """
    import os
    path = os.path.join(os.path.expanduser('~'), 'Files', 'base_data', 'global_invest',
                        'timber_provision', 'input', 'faostat_forestry_roundwood_2019.csv')
    if not os.path.exists(path):
        pytest.skip('the staged FAOSTAT roundwood slice is not on this machine')
    fao = pd.read_csv(path, encoding='utf-8-sig')
    world = fao[(fao['Area'] == 'World') & (fao['Item'] == 'Industrial roundwood')]
    produced = float(world[world['Element'] == 'Production']['Value'].iloc[0])
    export_q = float(world[world['Element'] == 'Export quantity']['Value'].iloc[0])
    export_v = float(world[world['Element'] == 'Export value']['Value'].iloc[0]) * 1000.0
    gross = produced * (export_v / export_q)

    flat, per_hectare = 88_743_966_326.0, 708_472_545_086.0
    assert flat < gross, 'the recorded diagnostic expects the flat reading below the proxy gross'
    assert per_hectare > 3 * gross, (
        'the recorded diagnostic expects the $/ha reading above three times the proxy gross: it '
        'is %.0f%% of it' % (100 * per_hectare / gross))


def test_the_cwon_rent_is_read_from_its_named_series_and_missing_stays_missing():
    """The CWoN rent, the account's timber figure, read from its named series.

    CWoN's file is wide by year and carries a `series` label. Reading the year column without
    checking the label would publish whatever series happened to be in the file under the name
    `timber_provision_gep_cwon_rent`, which is the failure this guards.
    """
    rent = pd.DataFrame({
        'countrycode': ['AAA', 'BBB'],
        'series': ['Forest, rents (current US$)'] * 2,
        'YR2019': [100.0, 250.0],
    })
    countries = pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB', 'CCC']})
    out = tp.cwon_forest_rent_by_country(rent, countries, 2019).set_index('iso3_r250_label')
    assert out.loc['AAA', 'timber_provision_gep_cwon_rent'] == pytest.approx(100.0)
    # CWoN does not value CCC; no rent published is not a rent of nothing.
    assert pd.isna(out.loc['CCC', 'timber_provision_gep_cwon_rent'])

    wrong = rent.assign(series='Coal rents (current US$)')
    with pytest.raises(ValueError, match='Forest, rents'):
        tp.cwon_forest_rent_by_country(wrong, countries, 2019)


def _cwon_table(series, values_by_country):
    return pd.DataFrame({'countrycode': list(values_by_country), 'series': series,
                         'YR2019': list(values_by_country.values())})


def test_the_computed_rent_is_revenue_times_rate_matched_by_country_code():
    """The two CWoN tables list countries in different orders and one country only in the revenue
    table, so a product taken by row position would pair the wrong numbers."""
    revenue = _cwon_table(tp.CWON_FOREST_REVENUE_SERIES, {'AAA': 1000.0, 'BBB': 2000.0, 'DDD': 50.0})
    rate = _cwon_table(tp.CWON_FOREST_RENTAL_RATE_SERIES, {'BBB': 0.5, 'AAA': 0.1})
    countries = pd.DataFrame({'iso3_r250_label': ['BBB', 'AAA', 'DDD']})
    out = tp.cwon_forest_rent_from_revenue_and_rate(revenue, rate, countries, 2019).set_index('iso3_r250_label')
    assert out.loc['AAA', 'timber_provision_gep_cwon_rent'] == pytest.approx(100.0)
    assert out.loc['BBB', 'timber_provision_gep_cwon_rent'] == pytest.approx(1000.0)
    assert pd.isna(out.loc['DDD', 'timber_provision_gep_cwon_rent'])     # a revenue without a rate

    with pytest.raises(ValueError, match='rental rate'):
        tp.cwon_forest_rent_from_revenue_and_rate(
            revenue, rate.assign(series='Forest, rents (current US$)'), countries, 2019)
    with pytest.raises(ValueError, match='more than once'):
        tp.cwon_forest_rent_from_revenue_and_rate(pd.concat([revenue, revenue.head(1)]), rate, countries, 2019)


def test_a_named_country_without_inputs_is_zero_and_any_other_stays_missing():
    """CWoN sets a missing forest rent to zero only for the countries its gap-filling names. A named
    country with both inputs keeps its product, and an unnamed one without them stays missing."""
    nan = float('nan')
    revenue = _cwon_table(tp.CWON_FOREST_REVENUE_SERIES, {'SMR': nan, 'SYR': 100.0, 'NRU': nan})
    rate = _cwon_table(tp.CWON_FOREST_RENTAL_RATE_SERIES, {'SMR': 0.2, 'SYR': 0.22, 'NRU': 0.2})
    countries = pd.DataFrame({'iso3_r250_label': ['SMR', 'SYR', 'NRU', 'XKX']})
    out = tp.cwon_forest_rent_from_revenue_and_rate(
        revenue, rate, countries, 2019, assumed_zero_countries=['SMR', 'SYR']).set_index('iso3_r250_label')
    rent, assumed = out['timber_provision_gep_cwon_rent'], out['timber_provision_cwon_rent_assumed_zero']
    assert rent['SMR'] == 0.0 and assumed['SMR']
    assert rent['SYR'] == pytest.approx(22.0) and not assumed['SYR']
    assert pd.isna(rent['NRU']) and not assumed['NRU']        # no inputs, not named
    assert pd.isna(rent['XKX']) and not assumed['XKX']        # in neither table


def test_the_staged_cwon_tables_reproduce_the_published_2019_rent(tmp_path):
    """Revenue times rate on the staged CWoN 2024 tables, against the rent CWoN publishes for 2019.

    Every computed value equals the published one. The published table carries 16 further zeros,
    small islands and territories with no forest revenue in any year, which CWoN's island rule
    sets to zero from a list the package does not include, so the computed rent leaves them
    missing. IMN, NRU and XKX are missing in both, although they too have no revenue in any year.
    """
    revenue_path = os.path.join(REFERENCE_DIR, 'cwon2024', 'forest_timber_revenue.xlsx')
    rate_path = os.path.join(REFERENCE_DIR, 'cwon2024', 'forest_timber_rental_rate.xlsx')
    rent_path = os.path.join(REFERENCE_DIR, 'cwon', 'forest_timber_rent_cd.dta')
    if not all(os.path.exists(path) for path in (revenue_path, rate_path, rent_path)):
        pytest.skip('the staged CWoN timber tables are not on this machine')
    p = SimpleNamespace(input_dir=str(tmp_path / 'input'), get_path=lambda *a, **k: '/resolved/' + '/'.join(a))
    utilities.hydrate_es_parameters(p, 'timber_provision', log=lambda *a: None)
    assert p.timber_provision_cwon_assumed_zero_countries == ['SMR', 'MCO', 'GRL', 'PSE', 'SYR']

    published = pd.read_stata(rent_path)
    countries = pd.DataFrame({'iso3_r250_label': sorted(published['countrycode'].unique())})
    out = tp.cwon_forest_rent_from_revenue_and_rate(
        pd.read_excel(revenue_path, na_values=['..']), pd.read_excel(rate_path, na_values=['..']),
        countries, 2019, p.timber_provision_cwon_assumed_zero_countries).set_index('iso3_r250_label')
    rent = out['timber_provision_gep_cwon_rent']
    reference = tp.cwon_forest_rent_by_country(published, countries, 2019).set_index(
        'iso3_r250_label')['timber_provision_gep_cwon_rent']

    both = rent.notna() & reference.notna()
    assert both.sum() == 198 and rent.notna().sum() == 198     # 194 products and 4 named zeros
    assert np.allclose(rent[both], reference[both], rtol=1e-12, atol=1e-6)
    assert rent.sum() == pytest.approx(132_130_111_931.1806, rel=1e-12)
    assert sorted(out.index[out['timber_provision_cwon_rent_assumed_zero']]) == ['GRL', 'MCO', 'PSE', 'SMR']
    island_zeros = sorted(reference.index[reference.notna() & rent.isna()])
    assert island_zeros == ['ASM', 'ATG', 'BMU', 'CHI', 'CUW', 'CYM', 'GRD', 'GUM', 'KNA', 'MAF',
                            'MHL', 'MNP', 'PLW', 'SXM', 'TUV', 'VIR']
    assert reference[island_zeros].eq(0).all()
    assert sorted(reference.index[reference.isna()]) == ['IMN', 'NRU', 'XKX']


def test_both_timber_valuations_are_published_and_differ_by_the_recorded_ratio():
    """Every run writes both valuations, so the gap between them stays visible."""
    import glob
    import os
    pattern = os.path.join(os.path.expanduser('~'), 'Files', 'global_invest', 'projects',
                           'gep_timber_provision*', '**', 'gep_by_country_base_year.csv')
    produced = [f for f in glob.glob(pattern, recursive=True)]
    if not produced:
        pytest.skip('no timber_provision run on this machine')
    for path in produced:
        df = pd.read_csv(path)
        # timber_provision_gep IS the CWoN rent; the spatial estimate is kept beside it under its
        # own name.
        assert 'timber_provision_gep' in df.columns
        assert 'timber_provision_gep_spatial' in df.columns, (
            '%s publishes only one valuation' % path)
        spatial = df['timber_provision_gep_spatial'].sum()
        rental = df['timber_provision_gep'].sum()
        # The recorded ratio is 1.49. A wide band, because this pins that the two are the same
        # order and neither column has silently become the other, not the ratio itself.
        assert 1.3 < rental / spatial < 1.7, (
            '%s: rental/spatial is %.2f, not the recorded 1.49' % (path, rental / spatial))


def test_the_faostat_gross_diagnostic_is_published_and_neither_valuation_exceeds_it_wholesale():
    """The gross-value diagnostic is a third column, not a third estimate.

    A land factor share is a fraction of the gross value of the wood it comes from, so a country
    whose timber GEP exceeds the proxy gross value of the roundwood it produced is flagged. A
    handful of countries do exceed it and are named in the entry: an export unit value is a proxy
    for a domestic price, so an exceedance is a flag and a ratio near 1 is noise. What this asserts
    is a recorded plausibility diagnostic: fewer than 10 percent of the valued countries in each
    valuation sit above the proxy gross. More than that would flag the column for review. It
    would not by itself establish a units error or a bound.
    """
    import glob
    import os
    pattern = os.path.join(os.path.expanduser('~'), 'Files', 'global_invest', 'projects',
                           'gep_timber_provision*', '**', 'gep_by_country_base_year.csv')
    produced = glob.glob(pattern, recursive=True)
    if not produced:
        pytest.skip('no timber_provision run on this machine')
    for path in produced:
        df = pd.read_csv(path)
        assert 'timber_roundwood_gross_value' in df.columns, '%s lacks the gross-value diagnostic' % path
        priced = df[df['timber_roundwood_gross_value'] > 0]
        for column in ('timber_provision_gep', 'timber_provision_gep_spatial'):
            valued = priced[priced[column] > 0]
            over = valued[valued[column] > valued['timber_roundwood_gross_value']]
            assert len(over) < 0.1 * len(valued), (
                '%s: %s exceeds the proxy gross roundwood value in %d of %d countries, above the '
                'recorded 10 percent plausibility diagnostic' % (path, column, len(over), len(valued)))


def test_every_timber_run_reproduces_the_committed_anchor():
    """Condition 12: checked against what the RUNS wrote, not against the join alone.

    `test_join_reproduces_the_committed_anchor` reads the staged anchor and joins it, which tests
    the join. This sums the raster the way the pipeline does and compares every timber run on the
    machine, cold starts included, against the anchor.
    """
    import glob
    import os
    anchor_path = os.path.join(REFERENCE_DIR, 'timber_provision_gep.csv')
    if not os.path.exists(anchor_path):
        pytest.skip('the committed anchor is not on this machine')
    anchor = pd.read_csv(anchor_path)['forestry_gep'].sum()

    pattern = os.path.join(os.path.expanduser('~'), 'Files', 'global_invest', 'projects',
                           'gep_timber_provision*', '**', 'gep_by_country_base_year.csv')
    produced = glob.glob(pattern, recursive=True)
    if not produced:
        pytest.skip('no timber_provision run on this machine')
    for path in produced:
        df = pd.read_csv(path)
        # 1e-7: the raster is float32 and the anchor was written from a separate sum of it.
        assert df['timber_provision_gep_spatial'].sum() == pytest.approx(anchor, rel=1e-7), (
            '%s totals %r against the committed anchor %r'
            % (path, df['timber_provision_gep_spatial'].sum(), anchor))
        assert int(df['timber_provision_gep_spatial'].gt(0).sum()) == 166, (
            '%s values %d countries, not the anchor\'s 166'
            % (path, int(df['timber_provision_gep_spatial'].gt(0).sum())))


import pytest
from affine import Affine
from rasterio.crs import CRS

from global_invest.timber_provision.timber_provision_tasks import _assert_one_grid

SHAPE = (64800, 129600)
PIXEL = 0.002777777777777778
WGS84 = CRS.from_epsg(4326)
REF = Affine(PIXEL, 0.0, -180.0, 0.0, -PIXEL, 90.0)


def grids(other):
    return {'reference': (SHAPE, REF, WGS84), 'other': (SHAPE, other, WGS84)}


def test_the_reported_origin_noise_passes():
    """1e-14 degrees of y-origin noise: representation noise, not misalignment."""
    noisy = Affine(PIXEL, 0.0, -180.0, 0.0, -PIXEL, 90.00000000000001)
    _assert_one_grid(grids(noisy))


def test_a_meaningful_translation_fails():
    shifted = Affine(PIXEL, 0.0, -180.0 + PIXEL / 2, 0.0, -PIXEL, 90.0)
    with pytest.raises(ValueError, match='displaced'):
        _assert_one_grid(grids(shifted))


def test_a_scale_difference_that_accumulates_fails():
    """One part in 1e9 leaves the origin identical and is invisible there, but reaches ~1e-4 pixels
    by column 129,600 -- which is exactly why the test is at the corners, not the origin."""
    stretched = Affine(PIXEL * (1 + 1e-9), 0.0, -180.0, 0.0, -PIXEL, 90.0)
    with pytest.raises(ValueError, match='displaced'):
        _assert_one_grid(grids(stretched))


def test_a_rotation_term_fails():
    rotated = Affine(PIXEL, 1e-12, -180.0, 0.0, -PIXEL, 90.0)
    with pytest.raises(ValueError, match='displaced'):
        _assert_one_grid(grids(rotated))


def test_a_different_shape_fails():
    with pytest.raises(ValueError, match='is \\(100, 100\\)|needs one grid'):
        _assert_one_grid({'reference': (SHAPE, REF, WGS84), 'other': ((100, 100), REF, WGS84)})


def test_a_different_crs_fails():
    with pytest.raises(ValueError, match='CRS'):
        _assert_one_grid({'reference': (SHAPE, REF, WGS84),
                          'other': (SHAPE, REF, CRS.from_epsg(3857))})


def test_noise_far_below_the_tolerance_still_passes_at_every_corner():
    tiny = Affine(PIXEL, 0.0, -180.0 + 1e-16, 0.0, -PIXEL, 90.0 - 1e-16)
    _assert_one_grid(grids(tiny))


# --- age-aware harvestable flow (measure 'harvestable_flow') -------------------------------------
# Every growth value below is SYNTHETIC, chosen to exercise the accounting. None is a Brazil
# parameter or proposed for use.
SYNTHETIC_GROWTH = pd.DataFrame([
    dict(forest_type_id=1, harvest_mode='cycle', min_age=60, cycle_years=35, intensity_m3_per_ha=30.0,
         vmax_m3_per_ha=250.0, k=0.03, shape=2.0, rotation_age=np.nan, source='SYNTHETIC test value'),
    dict(forest_type_id=2, harvest_mode='cycle', min_age=25, cycle_years=35, intensity_m3_per_ha=10.0,
         vmax_m3_per_ha=180.0, k=0.04, shape=2.0, rotation_age=np.nan, source='SYNTHETIC test value'),
    dict(forest_type_id=3, harvest_mode='rotation', rotation_age=7, vmax_m3_per_ha=400.0, k=0.25, shape=2.0,
         min_age=np.nan, cycle_years=np.nan, intensity_m3_per_ha=np.nan, source='SYNTHETIC test value'),
])
GROWTH = tp.growth_parameters_from_table(SYNTHETIC_GROWTH)
NEW_FOREST = {'type': 2, 'managed': True, 'accessible': True}
SYNTHETIC_PROVENANCE = pd.DataFrame({'input': list(tp.PROVENANCE_INPUTS), 'status': 'synthetic',
                                     'source': 'SYNTHETIC test fixture'})


def _state(types, ages, since=None, managed=True, accessible=True, stock=None):
    t = np.asarray(types, dtype='float64')
    forest = t > 0
    age = np.where(forest, np.asarray(ages, dtype='float64'), np.nan)
    if stock is None:
        stock = np.zeros(t.shape)
        for code, prm in GROWTH.items():
            stock[t == code] = tp.merchantable_volume(age[t == code], prm)
    return dict(type=t, age=age, since_cut=np.where(forest, 0.0 if since is None else np.asarray(since, dtype='float64'), np.nan),
                stock=np.asarray(stock, dtype='float64'), managed=np.broadcast_to(managed, t.shape) & forest,
                accessible=np.broadcast_to(accessible, t.shape) & forest)


def _advance(state, n, forest_now, rule='scheduled', **kw):
    return tp.advance_forest_state(state, n, forest_now, NEW_FOREST, GROWTH, rule, 'curve_equivalent_age', 0.0, **kw)


def test_age_aware_growth_table_has_no_defaults():
    for drop in ('source', 'k', 'rotation_age'):
        bad = SYNTHETIC_GROWTH.copy()
        bad.loc[2, drop] = np.nan
        with pytest.raises(ValueError):
            tp.growth_parameters_from_table(bad)
    with pytest.raises(ValueError):
        tp.growth_parameters_from_table(SYNTHETIC_GROWTH.assign(harvest_mode='clearcut'))
    with pytest.raises(ValueError):
        _advance(_state([3], [6]), 1, [True], rule='assumed')
    with pytest.raises(ValueError):
        tp.advance_forest_state(_state([3], [6]), 1, [True], NEW_FOREST, GROWTH, 'scheduled', 'young_curve', 0.0)


def test_age_aware_selective_cut_depletes_the_stock():
    new, flows = _advance(_state([1], [200], since=[34]), 1, [True])
    uncut = float(tp.merchantable_volume(201, GROWTH[1]))
    assert flows['harvest'][0] == 30.0
    assert new['stock'][0] == pytest.approx(uncut - 30.0, rel=1e-12)       # not the uncut curve value
    assert new['age'][0] == 201.0 and new['since_cut'][0] == 0.0


def test_age_aware_plantation_resets_and_new_forest_waits_for_its_own_age():
    new, flows = _advance(_state([3], [6]), 1, [True])
    assert flows['harvest'][0] == pytest.approx(float(tp.merchantable_volume(7, GROWTH[3])), rel=1e-12)
    assert new['age'][0] == 0.0 and new['stock'][0] == 0.0 and new['type'][0] == 3     # label unchanged
    young, flows = _advance(_state([0], [0]), 7, [True])                  # new forest at the map year
    assert young['type'][0] == 2 and young['age'][0] == 0.0 and young['stock'][0] == 0.0 and flows['harvest'][0] == 0.0
    later, flows = _advance(young, 24, [True])
    assert flows['harvest'][0] == 0.0 and later['stock'][0] > 0.0          # below min_age 25: grows, no harvest


def test_age_aware_ledger_closes_through_harvest_clearing_conversion_and_new_forest():
    s = _state([1, 2, 3, 3, 0, 1], [200, 30, 3, 6, 0, 120], since=[34, 20, 0, 0, 0, 10])
    s, flows = _advance(s, 9, [True, True, True, False, True, True], type_now=[1, 2, 2, 3, 0, 1])
    assert np.abs(tp.stock_ledger_residual(flows)).max() < 1e-9
    assert flows['removed'][2] > 0 and flows['removed'][3] > 0 and flows['harvest'].sum() > 0
    assert s['type'].tolist() == [1, 2, 2, 0, 2, 1] and s['stock'][2] == 0.0 and s['stock'][4] == 0.0
    assert (s['stock'] >= 0).all()


def test_age_aware_step_length_does_not_change_the_result():
    s = _state([3, 2, 1], [2, 20, 300], since=[0, 30, 34])
    one, f_one = _advance(s, 10, [True] * 3)
    step, harvest = s, 0.0
    for _ in range(10):
        step, f = _advance(step, 1, [True] * 3)
        harvest = harvest + f['harvest']
    for key in ('age', 'stock', 'since_cut'):
        assert np.array_equal(one[key], step[key])
    assert np.allclose(f_one['harvest'], harvest, rtol=0, atol=1e-12)


def test_age_aware_rule_none_reports_eligible_volume_without_cutting():
    s = _state([3, 1], [7, 200], since=[0, 35])
    assert tp.eligible_volume(s, GROWTH).sum() > 0 and s['age'].tolist() == [7.0, 200.0]
    new, flows = _advance(s, 10, [True, True], rule='none')
    assert flows['harvest'].sum() == 0.0 and flows['eligible'].sum() > 0 and new['age'][0] == 17.0


def test_age_aware_logged_stand_regrows_along_its_curve_from_its_equivalent_age():
    prm = GROWTH[1]
    cut, flows = _advance(_state([1], [200], since=[34]), 1, [True])
    equivalent = float(tp.volume_equivalent_age(cut['stock'][0], prm))
    assert flows['harvest'][0] == 30.0 and equivalent < 201.0
    later, flows = _advance(cut, 10, [True])
    assert flows['harvest'][0] == 0.0 and later['age'][0] == 211.0 and later['since_cut'][0] == 10.0
    assert later['stock'][0] == pytest.approx(float(tp.merchantable_volume(equivalent + 10.0, prm)), rel=1e-9)
    assert later['stock'][0] < float(tp.merchantable_volume(211.0, prm))


def test_harvestable_flow_factors_are_undefined_at_a_zero_reference():
    f = tp.harvestable_flow_factors([5.0, 5.0, 0.0], [10.0, 10.0, 0.0], [0.0, 2.0, 2.0], [10.0, 0.0, 10.0])
    assert np.isnan(f['per_ha_factor']).all()
    f = tp.harvestable_flow_factors(6.0, 20.0, 4.0, 10.0)
    assert f['area_factor'] == 2.0 and f['per_ha_factor'] == pytest.approx(0.75) and f['total_factor'] == 1.5


# --- integration fixture: SYNTHETIC seals7 maps through the state, zonal and shock steps -----------
# PLUMBING ONLY. The maps, the initial state and the growth table are synthetic; no saved Brazil
# SEALS map has the provenance to serve as evidence. The shocks below test the accounting, not GTAP
# science.
FOREST, CROP = tp.SEALS7_FOREST_ID, 2
CELL = 1.0 / 360.0


def _tif(path, array, nodata=None):
    import rasterio
    from rasterio.transform import from_origin
    a = np.asarray(array)
    a = a[None] if a.ndim == 2 else a
    with rasterio.open(str(path), 'w', driver='GTiff', height=a.shape[1], width=a.shape[2], count=a.shape[0],
                       dtype=a.dtype, crs='EPSG:4326', transform=from_origin(0.0, 0.0, CELL, CELL), nodata=nodata) as dst:
        dst.write(a)
    return str(path)


def _fixture(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box
    rows, cols = 6, 12
    base = np.full((rows, cols), FOREST, dtype='uint8')
    base[5, :] = CROP
    t = np.zeros((rows, cols)); age = np.full((rows, cols), np.nan); since = np.full((rows, cols), np.nan)
    managed = np.zeros((rows, cols)); accessible = np.zeros((rows, cols))
    t[0:4, 0:4], age[0:4, 0:4] = 3, (np.arange(16).reshape(4, 4) % 7)          # AAA plantation, staggered ages
    t[4, 0:4], age[4, 0:4], since[4, 0:4] = 1, 200, 30                           # AAA managed native, cut in 2028
    t[0:3, 4:8], age[0:3, 4:8] = 2, 10                                           # BBB secondary, below min_age
    t[3:5, 4:8], age[3:5, 4:8], since[3:5, 4:8] = 1, 200, 34                     # BBB protected native
    t[0:5, 8:12], age[0:5, 8:12] = 1, 200                                        # CCC unmanaged native
    since[(t > 0) & np.isnan(since)] = 0.0
    managed[0:5, 0:8] = 1; managed[0:5, 8:12] = 0
    accessible[0:5, 0:12] = 1; accessible[3:5, 4:8] = 0
    stock = np.zeros((rows, cols))
    for code, prm in GROWTH.items():
        stock[t == code] = tp.merchantable_volume(age[t == code], prm)
    paths = SimpleNamespace(base=_tif(tmp_path / 'lulc_2023.tif', base, nodata=255),
                            state=_tif(tmp_path / 'initial_forest_state_2023.tif',
                                       np.stack([t, age, since, stock, managed, accessible]).astype('float32')),
                            ha=_tif(tmp_path / 'ha_per_cell.tif', (7.0 + 0.1 * np.arange(rows)[:, None] * np.ones(cols)).astype('float32')))
    SYNTHETIC_GROWTH.to_csv(tmp_path / 'growth.csv', index=False)
    SYNTHETIC_PROVENANCE.to_csv(tmp_path / 'provenance.csv', index=False)
    regions = gpd.GeoDataFrame({'zone_id': [1, 2, 3], 'endw': ['AEZ1'] * 3, 'endw_id': [1, 1, 1], 'reg': ['AAA', 'BBB', 'CCC']},
                               geometry=[box(c0 * CELL, -rows * CELL, c1 * CELL, 0.0) for c0, c1 in ((0, 4), (4, 8), (8, 12))],
                               crs='EPSG:4326')
    regions.to_file(tmp_path / 'regions.gpkg', driver='GPKG')
    clear_2030, regrow_2040 = base.copy(), base.copy()
    clear_2030[0, 0:4] = CROP                                                    # AAA plantation row cleared
    regrow_2040[5, 0:4] = FOREST                                                 # row 0 back, plus new forest
    maps = {'baseline': {2023: paths.base, 2030: _tif(tmp_path / 'baseline_2030.tif', base, 255), 2040: _tif(tmp_path / 'baseline_2040.tif', base, 255)},
            'policy_same': {2030: _tif(tmp_path / 'same_2030.tif', base, 255), 2040: _tif(tmp_path / 'same_2040.tif', base, 255)},
            'policy_clear': {2030: _tif(tmp_path / 'clear_2030.tif', clear_2030, 255), 2040: _tif(tmp_path / 'clear_2040.tif', regrow_2040, 255)}}
    work, shocks = tmp_path / 'work', tmp_path / 'es_shocks'
    work.mkdir(); shocks.mkdir()
    p = SimpleNamespace(cur_dir=str(work), region_boundary_path=str(tmp_path / 'regions.gpkg'),
                        terrestrial_carbon_shock_id_col='zone_id', terrestrial_carbon_shock_acts='frs',
                        ha_per_cell_10sec_path=paths.ha, scenario_lulc_paths=maps,
                        timber_provision_shock_output_path=str(shocks / 'timber_provision_interpolated.csv'),
                        timber_provision_resource_measure='harvestable_flow',
                        timber_provision_growth_parameters_path=str(tmp_path / 'growth.csv'),
                        timber_provision_initial_state_path=paths.state,
                        timber_provision_input_provenance_path=str(tmp_path / 'provenance.csv'),
                        timber_provision_new_forest=dict(NEW_FOREST),
                        timber_provision_harvest_rule='scheduled', timber_provision_recovery_rule='curve_equivalent_age',
                        timber_provision_salvage_fraction=0.0, timber_provision_reference_flow='base_map_first_interval')
    return p, paths


def _run(p, paths, scenarios=('baseline', 'policy_same', 'policy_clear'), anchors=(2030, 2040)):
    from global_invest.timber_provision import timber_provision_tasks as t
    t._harvestable_flow_shock(p, 'baseline', list(scenarios), list(anchors), 2023,
                              {1: ('AEZ1', 'AAA'), 2: ('AEZ1', 'BBB'), 3: ('AEZ1', 'CCC')}, paths.base)
    work = p.cur_dir
    return (pd.read_csv(p.timber_provision_shock_output_path), pd.read_csv(os.path.join(work, 'timber_harvestable_flow_by_region.csv')),
            pd.read_csv(os.path.join(work, 'timber_harvestable_flow_ledger_by_zone.csv')))


def _state_mtimes(p):
    return {f: os.stat(os.path.join(p.cur_dir, f)).st_mtime_ns for f in os.listdir(p.cur_dir) if f.startswith('timber_forest_state_') and f.endswith('.tif')}


def test_harvestable_flow_fixture_accounts_shocks_and_declares(tmp_path):
    import json
    import rasterio
    p, paths = _fixture(tmp_path)
    shock, regional, ledger = _run(p, paths)
    # The stock ledger closes in every zone and step.
    assert ledger['ledger_residual_m3'].abs().max() < 1e-6 * ledger['closing_m3'].abs().max()
    # Units: forest area is the sum of the cells' hectares.
    ha = 7.0 + 0.1 * np.arange(6)
    aaa = ledger[(ledger.scenario == 'baseline') & (ledger.year == 2030) & (ledger.region_id == 1)].iloc[0]
    assert aaa['forest_ha'] == pytest.approx(4 * ha[:5].sum(), rel=1e-6) and aaa['harvest_m3'] > 0
    # BBB: secondary forest is below its minimum age and the native forest is protected, so no harvest.
    assert ledger[ledger.region_id == 2]['harvest_m3'].sum() == 0.0
    # CCC has no timberland: no reference flow, declared and not shocked.
    declared = json.load(open(os.path.join(os.path.dirname(p.timber_provision_shock_output_path), 'timber_provision_excluded_regions.json')))
    assert declared['excluded_regions'] == ['BBB', 'CCC'] and 'CCC' not in set(shock['REG'])
    # Same maps as the baseline: zero against the baseline in every year; the baseline matches the reference at 2030.
    same = shock[shock.scenario == 'policy_same']
    assert (same['shock_pct_contemp'] == 0.0).all()
    base_2030 = regional[(regional.scenario == 'baseline') & (regional.year == 2030)].iloc[0]
    assert base_2030['per_ha_factor'] == pytest.approx(1.0, abs=1e-12) and base_2030['area_factor'] == pytest.approx(1.0)
    # Clearing changes area and flow; only the per-hectare factor is fed.
    clear = regional[(regional.scenario == 'policy_clear') & (regional.year == 2040)].iloc[0]
    assert clear['area_factor'] < 1.0
    assert clear['shock_pct_v3'] == pytest.approx(100 * (clear['per_ha_factor'] - 1)) != pytest.approx(100 * (clear['total_factor'] - 1))
    # Re-established and new forest start at age 0 with no stock.
    with rasterio.open(os.path.join(p.cur_dir, 'timber_forest_state_policy_clear_2040.tif')) as src:
        t, age, stock = src.read(1), src.read(2), src.read(4)
    assert (t[0, 0:4] == 2).all() and (age[0, 0:4] == 0).all() and (stock[0, 0:4] == 0).all() and (t[5, 0:4] == 2).all()
    assert set(shock['timber_provision_accounting']) == {'harvestable_flow_age_v1'} and set(shock['harvest_basis']) == {'scheduled'}


def test_harvestable_flow_fixture_reuses_and_invalidates_by_provenance(tmp_path):
    p, paths = _fixture(tmp_path)
    first = _run(p, paths)
    before = _state_mtimes(p)
    again = _run(p, paths)
    assert _state_mtimes(p) == before                                       # nothing recomputed
    for a, b in zip(first, again):
        pd.testing.assert_frame_equal(a, b)
    changed = np.full((6, 12), FOREST, dtype='uint8')
    changed[5, :] = CROP
    changed[0:2, 0:4] = CROP                                                 # the 2030 policy map changes
    _tif(p.scenario_lulc_paths['policy_clear'][2030], changed, 255)
    _run(p, paths)
    after = _state_mtimes(p)
    rebuilt = sorted(f for f in after if after[f] != before[f])
    assert rebuilt == ['timber_forest_state_policy_clear_2030.tif', 'timber_forest_state_policy_clear_2040.tif']


def test_harvestable_flow_table_is_signed_by_its_measure(tmp_path):
    from global_invest.timber_provision import timber_provision_tasks as t
    p, paths = _fixture(tmp_path)
    _run(p, paths)
    maps = sorted({m for by_year in p.scenario_lulc_paths.values() for m in by_year.values()})
    out = [p.timber_provision_shock_output_path]
    assert utilities.reuse_reason(p, 'timber_provision', out, t.TIMBER_SHOCK_SIGNATURE, light_inputs=maps) is None
    default = SimpleNamespace(**{k: v for k, v in vars(p).items() if not k.startswith('timber_provision_')
                                 or k == 'timber_provision_shock_output_path'})
    assert 'resource_measure' in utilities.reuse_reason(default, 'timber_provision', out, t.TIMBER_SHOCK_SIGNATURE, light_inputs=maps)


def test_harvestable_flow_settings_are_all_required(tmp_path):
    from global_invest.timber_provision import timber_provision_tasks as t
    p, _ = _fixture(tmp_path)
    for key in t.HARVESTABLE_FLOW_SETTINGS:
        q = SimpleNamespace(**vars(p))
        delattr(q, 'timber_provision_' + key)
        with pytest.raises(NameError, match=key):
            t._harvestable_flow_settings(q)


def test_existing_measures_and_default_dispatch_are_unchanged(tmp_path):
    """Regression fixture for the net_return default and the aboveground_carbon measure.

    The expected totals were produced by the code at commit 5f714b3, before the harvestable_flow
    measure was added, on the same synthetic fixture. Agreement shows the existing measures and the
    default selection are unchanged. It is a compatibility check, not a validation of either method.
    """
    import geopandas as gpd
    from shapely.geometry import box
    from global_invest.timber_provision import timber_provision_initialize as ti
    from global_invest.timber_provision import timber_provision_tasks as t
    rng = np.random.default_rng(20261112)
    base = np.full((6, 12), FOREST, dtype='uint8')
    base[5, :] = CROP
    scen = base.copy()
    scen[0, 0:4] = CROP
    scen[5, 8:12] = FOREST
    scen[2, 6] = 255
    value = rng.uniform(-50, 400, size=(6, 12)).astype('float32')
    value[1, 1] = np.nan
    density = rng.uniform(0, 150, size=(6, 12)).astype('float32')
    ha = (7.0 + 0.1 * np.arange(6)[:, None] * np.ones(12)).astype('float32')
    maps = {'base': _tif(tmp_path / 'base.tif', base, 255), 'scen': _tif(tmp_path / 'scen.tif', scen, 255)}
    boundary = str(tmp_path / 'regions.gpkg')
    gpd.GeoDataFrame({'zone_id': [1, 2, 3]}, geometry=[box(c0 * CELL, -6 * CELL, c1 * CELL, 0) for c0, c1 in ((0, 4), (4, 8), (8, 12))],
                     crs='EPSG:4326').to_file(boundary, driver='GPKG')
    eligible = str(tmp_path / 'eligible.tif')
    t._write_eligible_value(_tif(tmp_path / 'value.tif', value), maps['base'], eligible)
    base_forest = _tif(tmp_path / 'agbc_base.tif', tp.forest_biomass_carbon_base(density, ha, base, lulc_ndv=255))
    new_forest = _tif(tmp_path / 'agbc_new.tif', (np.full((6, 12), 80.0) * ha).astype('float32'))
    expected = {('net_return', 'base'): [2592.407908, 3204.373512, 3625.956979],
                ('net_return', 'scen'): [1795.638774, 3204.373512, 3625.956979],
                ('aboveground_carbon', 'base'): [11270.481535, 10053.503141, 10211.281868],
                ('aboveground_carbon', 'scen'): [9133.28314, 10053.503141, 12611.281868]}
    for (measure, label), totals in expected.items():
        summary = str(tmp_path / ('%s_%s.csv' % (measure, label)))
        raster = str(tmp_path / ('%s_%s.tif' % (measure, label)))
        if measure == 'net_return':
            t._value_one_timber_map((label, 2030, maps[label], eligible, raster, summary, boundary, 'zone_id'))
        else:
            t._biomass_one_map((label, 2030, maps[label], base_forest, new_forest, raster, summary, boundary, 'zone_id'))
        assert pd.read_csv(summary).sort_values('zone_id')['total'].round(6).tolist() == totals

    class Tree(SimpleNamespace):
        def add_task(self, fn, **kwargs):
            self.added.append(fn.__name__)

    for measure, tasks in ((None, ['timber_value_density_table', 'timber_provision_shock']),
                           ('net_return', ['timber_value_density_table', 'timber_provision_shock']),
                           ('aboveground_carbon', ['timber_provision_shock']),
                           ('harvestable_flow', ['timber_provision_shock'])):
        tree = Tree(added=[], **({} if measure is None else {'timber_provision_resource_measure': measure}))
        ti.add_timber_provision_shock_tasks(tree)
        assert tree.added == tasks


def _public(p, tmp_path, measure='harvestable_flow'):
    """The fixture's p, completed with what the public task reads, for timber_provision_shock(p) itself."""
    q = SimpleNamespace(**vars(p))
    q.input_dir = str(tmp_path / 'input')
    q.get_path = lambda *a, **k: str(tmp_path / 'unresolved' / '/'.join(a))
    q.run_this = True
    q.es_shock_base_year, q.es_shock_years, q.es_shock_base_scenario = 2023, [2023, 2030, 2040], 'baseline'
    q.terrestrial_carbon_shock_endw_col, q.terrestrial_carbon_shock_reg_col = 'endw_id', 'reg'
    q.terrestrial_carbon_shock_endw_format, q.terrestrial_carbon_shock_value_col = 'AEZ%d', 'mean'
    q.terrestrial_quantity_input_path = p.scenario_lulc_paths['baseline'][2023]
    if measure is None:
        del q.timber_provision_resource_measure
    return q


def test_harvestable_flow_runs_through_the_public_shock_task(tmp_path):
    from global_invest.timber_provision import timber_provision_tasks as t
    p, paths = _fixture(tmp_path)
    q = _public(p, tmp_path)
    assert t.timber_provision_shock(q) is True
    public = pd.read_csv(q.timber_provision_shock_output_path)
    assert set(public['scenario']) == {'policy_same', 'policy_clear'} and set(public['REG']) == {'AAA'}
    direct, _, _ = _run(p, paths)                                           # the helper on the same fixture
    pd.testing.assert_frame_equal(public.reset_index(drop=True),
                                  direct[direct.scenario != 'baseline'].reset_index(drop=True))


def test_a_skipped_timber_shock_task_only_publishes_its_path(tmp_path):
    from global_invest.timber_provision import timber_provision_tasks as t
    for measure in ('harvestable_flow', None):
        (tmp_path / str(measure)).mkdir()
        p, _ = _fixture(tmp_path / str(measure))
        q = _public(p, tmp_path / str(measure), measure)
        del q.timber_provision_shock_output_path
        q.run_this, q.es_shock_dir = 0, str(tmp_path / str(measure) / 'es_shocks')
        assert t.timber_provision_shock(q) is None
        assert q.timber_provision_shock_output_path == os.path.join(q.es_shock_dir, 'timber_provision_interpolated.csv')
        assert not os.listdir(q.cur_dir) and not os.path.exists(q.timber_provision_shock_output_path)


def test_the_default_measure_never_reaches_the_harvestable_flow_code(tmp_path, monkeypatch):
    from global_invest.timber_provision import timber_provision_tasks as t
    p, _ = _fixture(tmp_path)
    q = _public(p, tmp_path, measure=None)
    monkeypatch.setattr(t, '_harvestable_flow_shock', lambda *a, **k: pytest.fail('default reached harvestable_flow'))
    with pytest.raises(NameError, match='timber value raster'):              # the net_return branch's own input check
        t.timber_provision_shock(q)


def test_a_changed_cached_table_or_boundary_is_rebuilt(tmp_path):
    import geopandas as gpd
    p, paths = _fixture(tmp_path)
    first = _run(p, paths)
    before = _state_mtimes(p)
    table = os.path.join(p.cur_dir, 'timber_harvestable_flow_by_zone_policy_clear_2030.csv')
    tampered = pd.read_csv(table)
    tampered.loc[0, 'harvest_m3'] *= 2
    tampered.to_csv(table, index=False)
    again = _run(p, paths)
    after = _state_mtimes(p)
    assert sorted(f for f in after if after[f] != before[f]) == ['timber_forest_state_policy_clear_2030.tif']
    for a, b in zip(first, again):                                          # restored, and 2040 reused
        pd.testing.assert_frame_equal(a, b)
    zone_ids = os.path.join(p.cur_dir, 'timber_cell_area_by_zone_zone_ids.tif')
    zones_before = os.stat(zone_ids).st_mtime_ns
    regions = gpd.read_file(p.region_boundary_path)
    regions.loc[regions.reg == 'CCC', 'zone_id'] = 4                        # the boundary changes
    regions.to_file(p.region_boundary_path, driver='GPKG')
    from global_invest.timber_provision import timber_provision_tasks as t
    t._harvestable_flow_shock(p, 'baseline', ['baseline', 'policy_same', 'policy_clear'], [2030, 2040], 2023,
                              {1: ('AEZ1', 'AAA'), 2: ('AEZ1', 'BBB'), 4: ('AEZ1', 'CCC')}, paths.base)
    rebuilt = _state_mtimes(p)
    assert os.stat(zone_ids).st_mtime_ns != zones_before and all(rebuilt[f] != after[f] for f in after)


def test_an_initial_state_that_breaks_its_contract_is_refused(tmp_path):
    import rasterio
    from global_invest.timber_provision import timber_provision_tasks as t
    p, paths = _fixture(tmp_path)
    with rasterio.open(paths.state) as src:
        bands = src.read()
    for band, row, col, value, what in ((0, 0, 0, 0.0, r'contract: 1 base-map forest cell\(s\) carry no forest type, 0 typed'),
                                        (0, 5, 0, 3.0, r'contract: 0 base-map forest .* 1 typed cell\(s\) are not forest'),
                                        (3, 0, 1, 1e6, r'contract: 0 base-map .* 0 typed .* and 1 cell\(s\) have an invalid'),
                                        (4, 0, 2, 0.5, r'contract: 0 base-map .* 0 typed .* and 1 cell\(s\) have an invalid')):
        bad = bands.copy()
        bad[band, row, col] = value
        _tif(paths.state, bad)
        for f in os.listdir(p.cur_dir):
            os.remove(os.path.join(p.cur_dir, f))
        with pytest.raises(ValueError, match=what):
            _run(p, paths)


# --- contemporaneous reference, decompositions, provenance and a real ProjectFlow -----------------
def _set_stands(paths, cols, plantation_ages, native_managed):
    """Rebuild one region's columns of the fixture state: plantation rows 0-3 at the given ages, the
    same in every column, and native forest in row 4, cut five years before the base year. Cell
    areas vary by row only, so any set of whole columns has the same stand mix as any other."""
    import rasterio
    with rasterio.open(paths.state) as src:
        t, age, since, stock, managed, accessible = src.read().astype('float64')
    for row, a in enumerate(plantation_ages):
        t[row, cols], age[row, cols], since[row, cols] = 3, a, 0.0
    t[4, cols], age[4, cols], since[4, cols] = 1, 200.0, 30.0
    managed[0:4, cols], managed[4, cols], accessible[0:5, cols] = 1, float(native_managed), 1
    for code, prm in GROWTH.items():
        stock[t == code] = tp.merchantable_volume(age[t == code], prm)
    _tif(paths.state, np.stack([t, age, since, stock, managed, accessible]).astype('float32'))


def _uniform_stands(tmp_path, cleared):
    """Region AAA with the same stands in every column, and a policy that clears `cleared` in 2030."""
    import rasterio
    p, paths = _fixture(tmp_path)
    _set_stands(paths, slice(0, 4), (1.0, 3.0, 5.0, 6.0), native_managed=True)
    with rasterio.open(paths.base) as src:
        policy = src.read(1)
    for rows, cols in cleared:
        policy[rows, cols] = CROP
    p.scenario_lulc_paths['policy_loss'] = {y: _tif(tmp_path / ('loss_%d.tif' % y), policy, 255) for y in (2030, 2040)}
    return p, paths


def test_identical_maps_give_a_zero_contemporaneous_shock_whatever_the_harvest_timing(tmp_path):
    p, paths = _fixture(tmp_path)
    years = list(range(2024, 2041))
    p.scenario_lulc_paths = {'baseline': {y: paths.base for y in [2023] + years}, 'policy_same': {y: paths.base for y in years}}
    shock, regional, _ = _run(p, paths, ('baseline', 'policy_same'), years)
    same = regional[regional.scenario == 'policy_same']
    assert same['shock_pct_v3'].max() - same['shock_pct_v3'].min() > 10       # the age structure moves the reference measure
    assert (same['per_ha_factor_vs_baseline_scenario'] == 1.0).all()
    assert (shock['shock_pct_contemp'] == 0.0).all() and (shock['shock_pct'] == 0.0).all()
    assert (same['timing_per_ha_factor'] == same['per_ha_factor']).all()


def test_every_chain_starts_from_the_same_state_and_management(tmp_path):
    p, paths = _fixture(tmp_path)
    _, _, ledger = _run(p, paths)
    first = ledger[ledger.year == 2030].set_index(['scenario', 'region_id'])[['opening_m3', 'timberland_open_ha', 'harvest_m3']]
    for label in ('baseline', 'policy_same', 'policy_clear'):
        pd.testing.assert_frame_equal(first.loc[label], first.loc['base_map_reference'])


def test_land_lost_with_the_kept_stand_mix_moves_area_and_total_not_the_per_hectare_shock(tmp_path):
    p, paths = _uniform_stands(tmp_path, [(slice(0, 5), slice(2, 4))])
    shock, regional, _ = _run(p, paths, ('baseline', 'policy_loss'))
    lost = regional[(regional.scenario == 'policy_loss') & (regional.year == 2040)].iloc[0]
    assert lost['area_factor_vs_baseline_scenario'] == pytest.approx(0.5, rel=1e-9)
    assert lost['total_factor_vs_baseline_scenario'] == pytest.approx(0.5, rel=1e-9)
    assert lost['per_ha_factor_vs_baseline_scenario'] == pytest.approx(1.0, rel=1e-9)
    assert shock[shock.scenario == 'policy_loss']['shock_pct_contemp'].abs().max() < 1e-9


def test_composition_and_harvest_timing_are_reported_apart(tmp_path):
    p, paths = _uniform_stands(tmp_path, [(slice(3, 5), slice(0, 4))])     # the oldest plantation row and the native row
    _, regional, _ = _run(p, paths, ('baseline', 'policy_loss'))
    lost = regional[regional.scenario == 'policy_loss'].set_index('year')
    bau = regional[regional.scenario == 'baseline'].set_index('year')
    assert (lost['timing_per_ha_factor'] == bau['per_ha_factor']).all()     # timing is the baseline's, shared
    assert np.allclose(lost['per_ha_factor'], lost['timing_per_ha_factor'] * lost['per_ha_factor_vs_baseline_scenario'], rtol=1e-12)
    assert np.allclose(lost['total_factor_vs_baseline_scenario'],
                       lost['area_factor_vs_baseline_scenario'] * lost['per_ha_factor_vs_baseline_scenario'], rtol=1e-12)
    assert lost.loc[2040, 'area_factor_vs_baseline_scenario'] < 1.0
    assert abs(lost.loc[2040, 'per_ha_factor_vs_baseline_scenario'] - 1.0) > 1e-3   # which stands were lost


def test_a_baseline_without_harvest_in_an_interval_is_declared_not_given_an_epsilon(tmp_path):
    import json
    p, paths = _fixture(tmp_path)
    _set_stands(paths, slice(0, 4), (6.0,) * 4, native_managed=False)      # AAA: one cut in 2024, none to 2030
    _set_stands(paths, slice(4, 8), (0.0, 2.0, 4.0, 6.0), native_managed=False)   # BBB: a cut in every interval
    p.scenario_lulc_paths = {'baseline': {y: paths.base for y in (2023, 2024, 2030)},
                             'policy_same': {y: paths.base for y in (2024, 2030)}}
    shock, regional, ledger = _run(p, paths, ('baseline', 'policy_same'), (2024, 2030))
    aaa = ledger[(ledger.scenario == 'baseline') & (ledger.region_id == 1)].set_index('year')['harvest_m3']
    assert aaa[2024] > 0 and aaa[2030] == 0.0
    declared = json.load(open(os.path.join(os.path.dirname(p.timber_provision_shock_output_path), 'timber_provision_excluded_regions.json')))
    assert declared['reason']['AAA'].startswith('no defined contemporaneous per-hectare factor')
    # One undefined anchor leaves the region out of every scenario and year, including 2024 where it is defined.
    assert set(shock['REG']) == set(regional['REG']) == {'BBB'} and (shock['shock_pct_contemp'] == 0.0).all()


def test_a_map_change_first_changes_the_next_years_harvest(tmp_path):
    p, paths = _fixture(tmp_path)
    cleared = p.scenario_lulc_paths['policy_clear'][2030]
    years = list(range(2024, 2033))
    p.scenario_lulc_paths = {'baseline': {y: paths.base for y in [2023] + years},
                             'policy_clear': {y: paths.base if y < 2030 else cleared for y in years}}
    shock, _, ledger = _run(p, paths, ('baseline', 'policy_clear'), years)
    aaa = ledger[ledger.region_id == 1].set_index(['year', 'scenario'])[['harvest_m3', 'timberland_open_ha']].unstack()
    for column in ('harvest_m3', 'timberland_open_ha'):                     # 2030's harvest is cut before its map
        assert (aaa.loc[:2030, (column, 'policy_clear')] == aaa.loc[:2030, (column, 'baseline')]).all()
    assert aaa.loc[2031, ('timberland_open_ha', 'policy_clear')] < aaa.loc[2031, ('timberland_open_ha', 'baseline')]
    contemp = shock[(shock.scenario == 'policy_clear') & (shock.REG == 'AAA')].set_index('year')['shock_pct_contemp']
    assert (contemp.loc[:2030] == 0.0).all() and contemp.loc[2031] != 0.0


def test_input_provenance_is_required_complete_and_written_beside_the_shock(tmp_path):
    p, paths = _fixture(tmp_path)
    shock, _, _ = _run(p, paths)
    assert set(shock['input_provenance']) == {' '.join('%s:synthetic' % name for name in tp.PROVENANCE_INPUTS)}
    pd.testing.assert_frame_equal(pd.read_csv(os.path.join(p.cur_dir, 'timber_harvestable_flow_input_provenance.csv')),
                                  SYNTHETIC_PROVENANCE)
    for bad in (SYNTHETIC_PROVENANCE[SYNTHETIC_PROVENANCE['input'] != 'age'],
                pd.concat([SYNTHETIC_PROVENANCE, SYNTHETIC_PROVENANCE.head(1)]),
                SYNTHETIC_PROVENANCE.assign(status=['observed', 'measured'] + ['assumed'] * 5),
                SYNTHETIC_PROVENANCE.assign(source=[''] + ['cited'] * 6)):
        with pytest.raises(ValueError):
            tp.input_provenance_from_table(bad)
    # Ages observed where a map has them and assumed elsewhere: one row per subset, with its coverage.
    ages = pd.DataFrame({'input': ['age', 'age'], 'status': ['observed', 'assumed'], 'source': ['an age map', 'a stated fill rule'],
                         'subset': ['cells with a mapped age', 'cells without one'], 'coverage': ['92% of secondary cells', '8%']})
    mixed = pd.concat([SYNTHETIC_PROVENANCE[SYNTHETIC_PROVENANCE['input'] != 'age'], ages])
    assert tp.input_provenance_from_table(mixed)['age'] == 'observed+assumed'
    for bad in (mixed.assign(coverage=mixed['coverage'].where(mixed['input'] != 'age', None)),
                mixed.assign(subset=mixed['subset'].where(mixed['input'] != 'age', 'all cells'))):
        with pytest.raises(ValueError, match='subset and a coverage'):
            tp.input_provenance_from_table(bad)


# The terrestrial_carbon rows whose *_path values the shock task's hydration would resolve through
# get_path, and so possibly fetch from the cloud bucket. The harvestable_flow branch reads none.
HYDRATED_PATHS_NOT_READ = ('terrestrial_carbon_density_lookup_table_path', 'total_carbon_density_path',
                           'spawn_raw_input_path', 'carbon_density_lookup_lulc_path')


def test_harvestable_flow_runs_reuses_and_skips_in_a_real_projectflow(tmp_path):
    import hazelbean as hb
    from global_invest.timber_provision import timber_provision_initialize as ti

    def refuse_get_path(*args, **kwargs):
        raise AssertionError('get_path called with %r: the test resolves and downloads nothing' % (args,))

    fixture, paths = _fixture(tmp_path)

    def execute(project_dir, tasks_to_skip=None):
        p = hb.ProjectFlow(project_dir=str(project_dir))
        for key, value in vars(_public(fixture, tmp_path)).items():
            if key not in ('cur_dir', 'input_dir', 'get_path', 'run_this', 'timber_provision_shock_output_path'):
                setattr(p, key, value)
        for key in HYDRATED_PATHS_NOT_READ:
            setattr(p, key, 'not read by the harvestable_flow shock')
        p.get_path = refuse_get_path
        ti.add_timber_provision_shock_tasks(p)
        p.skip_tasks(tasks_to_skip)
        p.execute()
        return p, os.path.join(p.intermediate_dir, 'timber_provision_shock')

    p, task_dir = execute(tmp_path / 'project')
    assert p.timber_provision_shock_output_path == os.path.join(p.project_dir, 'timber_provision_interpolated.csv')
    shock = pd.read_csv(p.timber_provision_shock_output_path)
    direct, _, _ = _run(fixture, paths)
    pd.testing.assert_frame_equal(shock, direct[direct.scenario != 'baseline'].reset_index(drop=True))
    states = {f: os.stat(os.path.join(task_dir, f)).st_mtime_ns for f in os.listdir(task_dir) if f.startswith('timber_forest_state_')}
    assert 'timber_forest_state_policy_clear_2040.tif' in states
    again, _ = execute(tmp_path / 'project')                                # the same project reuses every state
    assert {f: os.stat(os.path.join(task_dir, f)).st_mtime_ns for f in states} == states
    pd.testing.assert_frame_equal(pd.read_csv(again.timber_provision_shock_output_path), shock)
    skipped, skipped_dir = execute(tmp_path / 'skipped', ['timber_provision_shock'])
    assert not os.path.exists(skipped.timber_provision_shock_output_path)
    assert not os.path.exists(skipped_dir) or not os.listdir(skipped_dir)

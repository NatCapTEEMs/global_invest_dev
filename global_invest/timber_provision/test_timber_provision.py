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


def test_the_flat_sum_is_the_only_reading_the_roundwood_market_can_support():
    """The units question, settled from outside the pipeline rather than from its own output.

    The value raster is EPSG:4326 at 10 arc-second and carries NO unit metadata, so whether a cell
    holds a value or a per-hectare density cannot be read off the file. The committed output cannot
    settle it either, because it was produced by summing the same raster the same way.

    FAOSTAT settles it. World industrial roundwood production in 2019 was 1.985 billion m3, and the
    world export unit value was $111.01/m3, so the gross value of all industrial roundwood produced
    on earth was about $220bn. The pipeline's flat sum, $88.74bn, is 40 percent of that -- a high
    but possible land factor share. Reading the raster as $/ha gives $708bn, which is 321 percent
    of gross. A land share cannot exceed the gross value it is a share of, so the flat sum is the
    only reading the market supports.
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
    assert flat < gross, 'the flat reading must be a fraction of gross roundwood value'
    assert per_hectare > 3 * gross, (
        'the $/ha reading must be impossible, not merely large: it is %.0f%% of gross'
        % (100 * per_hectare / gross))


def test_the_cwon_rent_is_read_from_its_named_series_and_missing_stays_missing():
    """The rental alternative the issues document recommends, published beside ours.

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


def test_both_timber_valuations_are_published_and_differ_by_the_recorded_ratio():
    """Publishing one silently would hide a $43bn choice, so every run writes both."""
    import glob
    import os
    pattern = os.path.join(os.path.expanduser('~'), 'Files', 'global_invest', 'projects',
                           'gep_timber_provision*', '**', 'gep_by_country_base_year.csv')
    produced = [f for f in glob.glob(pattern, recursive=True)]
    if not produced:
        pytest.skip('no timber_provision run on this machine')
    for path in produced:
        df = pd.read_csv(path)
        assert 'timber_provision_gep' in df.columns
        assert 'timber_provision_gep_cwon_rent' in df.columns, (
            '%s publishes only one valuation' % path)
        spatial = df['timber_provision_gep'].sum()
        rental = df['timber_provision_gep_cwon_rent'].sum()
        # 1.49 as measured on 2026-09-02. A wide band, because this pins that the two are the
        # same order and neither column has silently become the other, not the ratio itself.
        assert 1.3 < rental / spatial < 1.7, (
            '%s: rental/spatial is %.2f, not the recorded 1.49' % (path, rental / spatial))

"""Unit tests for the timber_provision port (committed-output anchored)."""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from global_invest.timber_provision import timber_provision_chain as tpc
from global_invest.timber_provision import timber_provision_functions as tp

REFERENCE_DIR = os.path.join(os.path.dirname(tp.__file__), 'reference')


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
    net = tpc.net_forest_return(biomass, price, tcost, 0.3)
    assert np.allclose(net, [[145.0, -1.0], [95.5, -152.0]])
    # A regional share raster broadcasts the same way a scalar does.
    share = np.array([[0.3, 0.3], [0.5, 0.5]])
    net_regional = tpc.net_forest_return(biomass, price, tcost, share)
    assert np.allclose(net_regional, [[145.0, -1.0], [159.5, -120.0]])


def test_forest_value_keeps_only_managed_positive_net_return():
    net = np.array([[100.0, -3.0], [tpc.NET_RETURN_NDV, 250.0]])
    managed = np.array([[True, True], [True, False]])
    value = tpc.forest_value_from_net_return(net, managed)
    # Kept: managed and positive. Zeroed: negative, ndv, and positive-but-unmanaged.
    assert value.dtype == np.float32
    assert np.array_equal(value, np.float32([[100.0, 0.0], [0.0, 0.0]]))


def test_gep_by_zone_sums_pixels_per_country_id():
    value = np.array([[10.0, 20.0], [0.0, 5.0]], dtype=np.float32)
    zone_ids = np.array([[3, 3], [0, 5]])
    sums = tpc.gep_by_zone(value, zone_ids, n_zones=5)
    assert sums.shape == (6,)
    assert np.allclose(sums, [0.0, 0.0, 0.0, 30.0, 0.0, 5.0])
    # Blockwise accumulation is plain addition of successive blocks' arrays.
    total = sums + tpc.gep_by_zone(value, zone_ids, n_zones=5)
    countries = pd.DataFrame({'iso3_r250_id': [3, 5], 'iso3_r250_label': ['AAA', 'BBB']})
    df = tpc.timber_gep_from_zone_sums(total, countries)
    assert df['timber_provision_gep'].tolist() == [60.0, 10.0]


def test_chain_round_trip_on_synthetic_layers():
    biomass = np.array([[10.0, 8.0], [6.0, 1.0]])
    price = np.full((2, 2), 60.0)
    tcost = np.array([[2.0, 2.0], [2.0, 100.0]])
    share = 0.3
    managed = np.array([[True, False], [True, True]])
    net = tpc.net_forest_return(biomass, price, tcost, share)
    value = tpc.forest_value_from_net_return(net, managed)
    # px (0,0) and (1,0) kept; (0,1) unmanaged; (1,1) negative net return.
    assert np.allclose(value, [[178.0, 0.0], [106.0, 0.0]])
    sums = tpc.gep_by_zone(value, np.array([[1, 1], [2, 2]]), n_zones=2)
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

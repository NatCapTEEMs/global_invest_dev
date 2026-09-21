"""Timber provision on the fixed eligible area: what changes it and what must not."""
import numpy as np
import pytest

from global_invest.timber_provision import timber_provision_functions as tp

F = tp.SEALS7_FOREST_ID   # forest
C = 2                     # cropland
NDV = 255


def _base():
    # value: managed net return (0 off the mask, negative already floored upstream; one nodata cell)
    value = np.array([[10., 20., 0., 5.],
                      [0., 8., np.nan, 3.]], dtype='float32')
    base_lulc = np.array([[F, F, F, C],
                          [C, F, F, F]], dtype='uint8')
    return value, base_lulc


def test_eligible_area_is_managed_and_forest_in_the_base_map():
    value, base_lulc = _base()
    e = tp.timber_eligible_value(value, base_lulc, lulc_ndv=NDV)
    # (0,3) managed but cropland in the base map -> excluded; (1,0) cropland, 0 -> 0; (1,2) nodata stays nodata
    assert e[0, 0] == 10 and e[0, 1] == 20 and e[0, 2] == 0 and e[0, 3] == 0
    assert e[1, 0] == 0 and e[1, 1] == 8 and np.isnan(e[1, 2]) and e[1, 3] == 3


def test_unchanged_map_gives_zero_change():
    value, base_lulc = _base()
    e = tp.timber_eligible_value(value, base_lulc, lulc_ndv=NDV)
    y = tp.timber_value_on_forest(e, base_lulc, lulc_ndv=NDV)
    assert np.nansum(y) == np.nansum(e)


def test_conversion_of_eligible_forest_removes_its_provision():
    value, base_lulc = _base()
    e = tp.timber_eligible_value(value, base_lulc, lulc_ndv=NDV)
    scen = base_lulc.copy(); scen[0, 1] = C          # the 20-valued eligible cell becomes cropland
    y = tp.timber_value_on_forest(e, scen, lulc_ndv=NDV)
    assert np.nansum(y) == pytest.approx(np.nansum(e) - 20)


def test_forest_gain_outside_the_eligible_area_gives_no_change():
    value, base_lulc = _base()
    e = tp.timber_eligible_value(value, base_lulc, lulc_ndv=NDV)
    scen = base_lulc.copy(); scen[1, 0] = F; scen[0, 3] = F   # new forest on unmanaged land and on managed-but-cropland-in-2023 land
    y = tp.timber_value_on_forest(e, scen, lulc_ndv=NDV)
    assert np.nansum(y) == np.nansum(e)


def test_nodata_in_the_scenario_map_is_not_a_conversion():
    value, base_lulc = _base()
    e = tp.timber_eligible_value(value, base_lulc, lulc_ndv=NDV)
    scen = base_lulc.copy(); scen[0, 0] = NDV
    y = tp.timber_value_on_forest(e, scen, lulc_ndv=NDV)
    assert y[0, 0] == 10 and np.nansum(y) == np.nansum(e)


def test_negative_net_returns_carry_no_value():
    # upstream: forest_value_from_net_return floors negatives and nodata to zero
    net = np.array([[-5., 4., -1e9]], dtype='float32'); mask = np.array([[True, True, True]])
    v = tp.forest_value_from_net_return(net, mask, ndv=-1e9)
    assert v.tolist() == [[0., 4., 0.]]
    # and the eligible area treats a zero as not managed
    e = tp.timber_eligible_value(v, np.array([[F, F, F]], dtype='uint8'))
    assert e.tolist() == [[0., 4., 0.]]

"""Unit tests for the NTFP skeleton (valuation function + configuration rows)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from global_invest.ntfp import ntfp_functions
from global_invest.ntfp import ntfp_functions as nf


def test_the_published_rate_is_applied_to_the_reachable_area():
    # The rate is CWoN's own, per hectare, and it is applied to the hectares people can reach.
    # AAA: $1/ha over 100 reachable hectares is $100, not the $200 CWoN spread over all 200.
    area = pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB'], 'accessible_forest_ha': [100.0, 50.0]})
    values = pd.DataFrame({'iso3_r250_label': ['AAA', 'AAA', 'BBB'],
                           'year': [2019, 2020, 2019],
                           'nwfp_value_per_ha': [1.0, 4.5, 2.0]})
    out = ntfp_functions.ntfp_gep_by_country(area, values, 2019).set_index('iso3_r250_label')
    assert out['ntfp_gep'].to_dict() == {'AAA': 100.0, 'BBB': 100.0}
    # Only the requested year is used: AAA's 2020 row would have given a different rate.
    assert out.loc['AAA', 'nwfp_value_per_ha'] == 1.0


def test_source_script_constants_are_pinned():
    assert ntfp_functions.NTFP_ACCESS_BUFFER_M == 10_000
    assert (ntfp_functions.ESA_FOREST_CLASS_MIN, ntfp_functions.ESA_FOREST_CLASS_MAX) == (50, 90)


def test_es_config_row_hydrates_ntfp(tmp_path):
    from global_invest import utilities
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'ntfp', log=lambda *a: None)
    assert p.gep_base_year == 2019


# ---------------------------------------------------------------------------
# Accessibility: forest within 10 km of a road or a river.
# ---------------------------------------------------------------------------

def test_forest_mask_covers_the_esa_forest_classes_only():
    block = np.array([10, 49, 50, 70, 90, 91, 200])
    assert nf.forest_mask(block).tolist() == [False, False, True, True, True, False, False]


def test_forest_mask_excludes_nodata_even_inside_the_class_range():
    block = np.array([70, 255])
    assert nf.forest_mask(block, ndv=255).tolist() == [True, False]


def test_access_sources_are_roads_or_rivers():
    roads = np.array([0.0, 120.0, 0.0, 0.0])
    rivers = np.array([0.0, 0.0, 1.0, 0.0])
    assert nf.access_source_mask(roads, rivers).tolist() == [False, True, True, False]


def test_a_cell_exactly_on_the_buffer_edge_counts_as_accessible():
    distance = np.array([0.0, 9.999, 10.0, 10.001])
    assert nf.accessible_mask(distance, buffer_km=10.0).tolist() == [True, True, True, False]


def test_accessible_forest_hectares_needs_both_forest_and_access():
    forest = np.array([True, True, False, False])
    access = np.array([True, False, True, False])
    ha = np.array([9.0, 9.0, 9.0, 9.0])
    assert nf.accessible_forest_hectares(forest, access, ha).tolist() == [9.0, 0.0, 0.0, 0.0]


def test_hectares_by_zone_sums_into_the_zone_id_and_keeps_outside_at_index_zero():
    ha = np.array([[5.0, 5.0], [2.0, 0.0]])
    zones = np.array([[1, 2], [1, 0]])
    out = nf.hectares_by_zone(ha, zones, n_zones=3)
    assert out[1] == 7.0        # two cells in zone 1
    assert out[2] == 5.0
    assert out[3] == 0.0        # a zone with no accessible forest is zero, not missing
    assert out[0] == 0.0        # outside every zone


def test_the_country_total_falls_below_cwons_because_unreachable_forest_is_dropped():
    # CWoN's rate covers all forest. Applying it to only the reachable part drops the value
    # notionally earned where nobody can go, which is the source module's choice and ours.
    accessible = pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB', 'NOACCESS'],
                               'accessible_forest_ha': [600.0, 250.0, 0.0]})
    rates = pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB', 'NOACCESS'], 'year': [2019] * 3,
                          'nwfp_value_per_ha': [12.0, 10.0, 2.0]})

    out = nf.ntfp_gep_by_country(accessible, rates, 2019).set_index('iso3_r250_label')

    assert out.loc['AAA', 'ntfp_gep'] == pytest.approx(7200.0)   # 600 ha at $12
    assert out.loc['BBB', 'ntfp_gep'] == pytest.approx(2500.0)   # 250 ha at $10
    # CWoN priced 1,000 and 500 hectares, so its own totals would be $12,000 and $5,000.
    assert out['ntfp_gep'].sum() == pytest.approx(9700.0)
    # A country with no reachable forest earns nothing rather than an undefined amount.
    assert out.loc['NOACCESS', 'ntfp_gep'] == pytest.approx(0.0)


def test_ndvi_floor_drops_bare_and_unseen_forest():
    # Four forest cells and one non-forest. Raw NDVI is the value times 10,000, so 2500 is 0.25
    # and passes, 1500 is 0.15 and fails, and -9999 is no data and fails with it. The non-forest
    # cell stays out however green it is, because the floor narrows the mask and cannot widen it.
    forest = np.array([True, True, True, True, False])
    ndvi = np.array([2500, 1500, -9999, 2000, 9000], dtype=np.int16)

    kept = nf.vegetated_forest_mask(forest, ndvi)

    assert list(kept) == [True, False, False, True, False]


def test_ndvi_floor_is_inclusive_at_the_threshold():
    # 0.20 exactly is kept: the threshold is the floor a cell must reach, not clear.
    forest = np.array([True, True])
    ndvi = np.array([2000, 1999], dtype=np.int16)
    assert list(nf.vegetated_forest_mask(forest, ndvi)) == [True, False]

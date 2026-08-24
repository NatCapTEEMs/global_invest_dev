"""Unit tests for the NTFP skeleton (valuation function + configuration rows)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from global_invest.ntfp import ntfp_functions
from global_invest.ntfp import ntfp_functions as nf


def test_the_rate_is_cwons_value_over_the_reachable_area_not_over_all_forest():
    # The reachable area is smaller than the forest CWoN priced, so dividing by it raises the rate.
    # AAA: $200 over 100 reachable hectares is $2/ha where CWoN spread the same $200 over 200.
    area = pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB'], 'accessible_forest_ha': [100.0, 50.0]})
    values = pd.DataFrame({'iso3_r250_label': ['AAA', 'AAA', 'BBB'],
                           'year': [2019, 2020, 2019],
                           'nwfp_value_usd': [200.0, 900.0, 200.0],
                           'nwfp_value_per_ha': [1.0, 4.5, 2.0]})
    out = ntfp_functions.ntfp_gep_by_country(area, values, 2019).set_index('iso3_r250_label')
    assert out['nwfp_value_per_accessible_ha'].to_dict() == {'AAA': 2.0, 'BBB': 4.0}
    assert out['ntfp_gep'].to_dict() == {'AAA': 200.0, 'BBB': 200.0}
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


def test_the_price_is_rescaled_to_reachable_forest_so_the_country_total_is_cwons():
    # CWoN publishes a value and the forest area it attributes it to. Applying that rate to only
    # the reachable forest would drop the part earned on forest nobody can reach, which is not
    # zero, it is unallocated. Rescaling to the reachable area keeps the total and raises the rate.
    accessible = pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB', 'NOACCESS'],
                               'accessible_forest_ha': [600.0, 250.0, 0.0]})
    cwon = pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB', 'NOACCESS'], 'year': [2019] * 3,
                         'nwfp_value_usd': [12000.0, 5000.0, 800.0],
                         'forest_ha': [1000.0, 500.0, 400.0],
                         'nwfp_value_per_ha': [12.0, 10.0, 2.0]})

    out = nf.ntfp_gep_by_country(accessible, cwon, 2019).set_index('iso3_r250_label')

    # AAA: $12,000 over 600 reachable hectares is $20/ha, against CWoN's $12 over 1,000.
    assert out.loc['AAA', 'nwfp_value_per_accessible_ha'] == pytest.approx(20.0)
    assert out.loc['AAA', 'ntfp_gep'] == pytest.approx(12000.0)
    assert out.loc['BBB', 'ntfp_gep'] == pytest.approx(5000.0)
    # The country total is CWoN's, because the reachable hectares divide out and multiply back in.
    assert out['ntfp_gep'].sum() == pytest.approx(17000.0)
    # A country with no reachable forest gets no rate rather than an infinite one.
    assert pd.isna(out.loc['NOACCESS', 'nwfp_value_per_accessible_ha'])
    assert pd.isna(out.loc['NOACCESS', 'ntfp_gep'])

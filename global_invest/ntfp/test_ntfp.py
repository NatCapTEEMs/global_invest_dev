"""Unit tests for the NTFP skeleton (valuation function + configuration rows)."""
from types import SimpleNamespace

import pandas as pd

from global_invest.ntfp import ntfp_functions


def test_ntfp_gep_is_area_times_value_at_the_base_year():
    area = pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB'], 'accessible_forest_ha': [100.0, 50.0]})
    values = pd.DataFrame({'iso3_r250_label': ['AAA', 'AAA', 'BBB'],
                           'year': [2019, 2020, 2019],
                           'nwfp_value_per_ha': [2.0, 9.0, 4.0]})
    out = ntfp_functions.ntfp_gep_by_country(area, values, 2019)
    assert out.set_index('iso3_r250_label')['ntfp_gep'].to_dict() == {'AAA': 200.0, 'BBB': 200.0}


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

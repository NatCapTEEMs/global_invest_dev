"""Unit tests for the stormwater skeleton (valuation function + configuration rows)."""
from types import SimpleNamespace

import pandas as pd

from global_invest.stormwater import stormwater_functions


def test_stormwater_gep_is_volume_times_price():
    volumes = pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB'], 'retention_m3': [1000.0, 250.0]})
    out = stormwater_functions.stormwater_gep_by_country(volumes, 2.5)
    assert out.set_index('iso3_r250_label')['stormwater_gep'].to_dict() == {'AAA': 2500.0, 'BBB': 625.0}


def test_the_committed_price_placeholder_is_pinned():
    assert stormwater_functions.STORMWATER_PRICE_PER_M3_PLACEHOLDER == 1.0


def test_es_config_row_hydrates_stormwater(tmp_path):
    from global_invest import utilities
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'stormwater', log=lambda *a: None)
    assert p.gep_base_year == 2019

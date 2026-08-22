"""Unit tests for crop_provision's valuation steps."""
from types import SimpleNamespace

import pandas as pd

from global_invest.crop_provision import crop_provision_functions as cp


def test_normalize_m49_codes_unquotes_casts_and_maps_successors():
    """FAOSTAT ships quoted codes and keeps dissolved states under their own code: both are
    resolved so every row joins to a current country."""
    df = pd.DataFrame({'area_code_M49': ["'156", "'159", "'891", "'076"], 'value': [1, 2, 3, 4]})
    out = cp.normalize_m49_codes(df)
    assert list(out['area_code_M49']) == [156, 156, 688, 76]
    assert list(out['value']) == [1, 2, 3, 4]
    assert list(df['area_code_M49']) == ["'156", "'159", "'891", "'076"]   # input untouched


def test_every_successor_maps_to_a_different_live_code():
    """A successor mapping that pointed at itself, or at another dissolved state, would leave
    production stranded."""
    for dissolved, successor in cp.M49_SUCCESSORS.items():
        assert dissolved != successor
        assert successor not in cp.M49_SUCCESSORS


def test_the_faostat_unit_factor_is_the_thousand_usd_conversion():
    assert cp.FAOSTAT_THOUSAND_USD == 1000.0


def test_es_config_row_hydrates_crop_provision(tmp_path):
    from global_invest import utilities
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'crop_provision', log=lambda *a: None)
    assert p.sheet_label == 'crop_provision'

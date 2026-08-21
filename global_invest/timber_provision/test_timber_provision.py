"""Unit tests for the timber_provision port (committed-output anchored)."""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

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


def test_es_config_and_parameters_rows_hydrate_timber_provision(tmp_path):
    from global_invest import utilities
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'timber_provision', log=lambda *a: None)
    assert p.gep_base_year == 2019
    utilities.hydrate_es_parameters(p, 'timber_provision', log=lambda *a: None)
    assert p.timber_provision_gep_path.endswith('timber_provision_gep.csv')

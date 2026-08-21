"""Unit tests for the extractive_energy port (three committed fuel anchors)."""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from global_invest.extractive_energy import extractive_energy_functions as xe

REFERENCE_DIR = os.path.join(os.path.dirname(xe.__file__), 'reference')


def test_components_and_sum_reproduce_the_committed_anchors():
    gas = pd.read_csv(os.path.join(REFERENCE_DIR, 'gep-gas.csv'))
    coal = pd.read_csv(os.path.join(REFERENCE_DIR, 'gep-coal.csv'))
    oil = pd.read_csv(os.path.join(REFERENCE_DIR, 'gep-petrolium.csv'))
    countries = gas[['iso3_r250_id', 'iso3_r250_label']]
    out = xe.extractive_energy_gep_by_country(gas, coal, oil, countries)
    assert len(out) == 250
    for src, col, n in ((gas, 'extractive_energy_gas_gep', 111),
                        (coal, 'extractive_energy_coal_gep', 211),
                        (oil, 'extractive_energy_oil_gep', 110)):
        assert out[col].notna().sum() == n
        assert np.isclose(out[col].sum(), src.select_dtypes('number').iloc[:, -1].sum())
    # The service total sums the components that exist; a coal-only country is not NaN.
    coal_only = out[out['extractive_energy_gas_gep'].isna()
                    & out['extractive_energy_coal_gep'].notna()]
    assert len(coal_only) > 0
    assert coal_only['extractive_energy_gep'].notna().all()
    # And a country in no table stays NaN, never zero.
    none = out[[f'extractive_energy_{f}_gep' for f in xe.EXTRACTIVE_ENERGY_FUELS]].isna().all(axis=1)
    assert out.loc[none, 'extractive_energy_gep'].isna().all()


def test_es_config_and_parameters_rows_hydrate_extractive_energy(tmp_path):
    from global_invest import utilities
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'extractive_energy', log=lambda *a: None)
    assert p.gep_base_year == 2019
    utilities.hydrate_es_parameters(p, 'extractive_energy', log=lambda *a: None)
    assert p.extractive_energy_gas_path.endswith('gep-gas.csv')
    assert p.extractive_energy_coal_path.endswith('gep-coal.csv')
    assert p.extractive_energy_oil_path.endswith('gep-petrolium.csv')

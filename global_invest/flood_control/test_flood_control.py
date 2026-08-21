"""Unit tests for the flood_control port (committed-output anchored).

reference/ ships the extracted per-country avoided-damage table AND the pipeline's own global
summary, so every run verifies the anchor against the pipeline's own bookkeeping: the total,
the country counts, and the zero/NaN distinction (assessed-zero vs never-assessed).
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from global_invest.flood_control import flood_control_functions as fc

REFERENCE_DIR = os.path.join(os.path.dirname(fc.__file__), 'reference')


def test_anchor_reproduces_the_pipelines_own_global_summary():
    avoided = pd.read_csv(os.path.join(REFERENCE_DIR, 'country_avoided_damage_usd2019.csv'))
    summary = pd.read_csv(os.path.join(REFERENCE_DIR, 'summary_global.csv')).iloc[0]
    assert np.isclose(avoided['avoided_damage_usd2019'].sum() / 1e9,
                      summary['global_total_expected_avoided_damage_usd2019_bil'], rtol=1e-9)
    assert int((avoided['avoided_damage_usd2019'] > 0).sum()) == int(summary['n_nonzero'])
    assert int(avoided['avoided_damage_usd2019'].notna().sum()) == int(summary['n_countries'])
    assert int((avoided['avoided_damage_usd2019'] == 0).sum()) == int(summary['n_zero'])


def test_join_keeps_all_countries_and_distinguishes_zero_from_unassessed():
    avoided = pd.read_csv(os.path.join(REFERENCE_DIR, 'country_avoided_damage_usd2019.csv'))
    countries = avoided[['iso3_r250_id', 'iso3_r250_label']]
    out = fc.flood_control_gep_by_country(avoided, countries)
    assert len(out) == 250
    assert (out['flood_control_gep'] == 0).sum() > 0        # assessed, zero avoided damage
    # ESH (Western Sahara) was never assessed: NaN, not zero.
    assert np.isnan(out.set_index('iso3_r250_label').loc['ESH', 'flood_control_gep'])


def test_es_config_and_parameters_rows_hydrate_flood_control(tmp_path):
    from global_invest import utilities
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'flood_control', log=lambda *a: None)
    assert p.gep_base_year == 2019
    utilities.hydrate_es_parameters(p, 'flood_control', log=lambda *a: None)
    assert p.flood_control_avoided_damage_path.endswith('country_avoided_damage_usd2019.csv')

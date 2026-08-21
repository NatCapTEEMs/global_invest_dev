"""Unit tests for the water_quality port (committed-chain anchored).

reference/ ships the drive's committed intermediates and final output, so every run verifies:
the two per-nutrient identities (retention x domestic fraction; x price) against the
committed columns, the element GEPs against the committed N/P tables, and the documented
NON-identity of the final international-dollar stage (per-country ratios, not a global
scalar) so the unidentified conversion cannot be silently mistaken for the USD sum.
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from global_invest.water_quality import water_quality_functions as wq

REFERENCE_DIR = os.path.join(os.path.dirname(wq.__file__), 'reference')


def test_identities_hold_against_the_committed_intermediates():
    retention = pd.read_csv(os.path.join(REFERENCE_DIR, 'retention_estimates.csv'))
    out = wq.element_service_values(retention)     # raises if an identity breaks
    nitrogen = pd.read_csv(os.path.join(REFERENCE_DIR, 'nitrogen_value_gep.csv'))
    phosphorus = pd.read_csv(os.path.join(REFERENCE_DIR, 'phosphorus_value_gep.csv'))
    m = (out[['iso3_r250_label', 'n_gep_usd', 'p_gep_usd']]
         .merge(nitrogen, on='iso3_r250_label').merge(phosphorus, on='iso3_r250_label'))
    for ours, committed in (('n_gep_usd', 'nitrogen_gep'), ('p_gep_usd', 'phosphorus_gep')):
        both = m[ours].notna() & m[committed].notna()
        assert both.sum() == 209
        assert np.allclose(m.loc[both, ours], m.loc[both, committed], rtol=1e-5)


def test_broken_identity_raises():
    retention = pd.read_csv(os.path.join(REFERENCE_DIR, 'retention_estimates.csv'))
    broken = retention.copy()
    broken.loc[broken['n_ServiceValue(usd)'].notna(), 'n_ServiceValue(usd)'] *= 2
    with pytest.raises(ValueError, match='identity'):
        wq.element_service_values(broken)


def test_the_international_stage_is_not_the_usd_sum_and_not_a_global_scalar():
    # Documents the unidentified conversion: anyone assuming final == USD sum (or a single
    # scaling factor) must hit this test and read the module docstring.
    retention = pd.read_csv(os.path.join(REFERENCE_DIR, 'retention_estimates.csv'))
    final = pd.read_csv(os.path.join(REFERENCE_DIR, 'water_quality_gep.csv'))
    usd = wq.element_service_values(retention)
    m = final.merge(usd[['iso3_r250_label', 'n_gep_usd', 'p_gep_usd']], on='iso3_r250_label')
    m['usd_sum'] = m['n_gep_usd'] + m['p_gep_usd']
    both = m['water_quality_gep'].notna() & m['usd_sum'].notna()
    ratio = (m['water_quality_gep'] / m['usd_sum'])[both]
    assert not np.allclose(ratio, 1.0, rtol=0.01)            # not the plain sum
    assert ratio.max() / ratio.min() > 5                     # not a global scalar either


def test_country_join_keeps_all_countries_and_carries_both_currencies():
    retention = pd.read_csv(os.path.join(REFERENCE_DIR, 'retention_estimates.csv'))
    final = pd.read_csv(os.path.join(REFERENCE_DIR, 'water_quality_gep.csv'))
    countries = final[['iso3_r250_id', 'iso3_r250_label']]
    out = wq.water_quality_gep_by_country(retention, final, countries)
    assert len(out) == 250
    assert out['water_quality_gep_usd'].notna().sum() == 209
    assert out['water_quality_gep'].notna().sum() == 178


def test_es_config_and_parameters_rows_hydrate_water_quality(tmp_path):
    from global_invest import utilities
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'water_quality', log=lambda *a: None)
    assert p.gep_base_year == 2019
    utilities.hydrate_es_parameters(p, 'water_quality', log=lambda *a: None)
    assert p.water_quality_retention_path.endswith('retention_estimates.csv')
    assert p.water_quality_international_path.endswith('water_quality_gep.csv')

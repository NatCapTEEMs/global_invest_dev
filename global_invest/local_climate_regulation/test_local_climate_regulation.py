"""Unit tests for the local_climate_regulation port (committed-final anchored).

reference/ ships the current final, the two lineage versions and a city-month sample, so every
run verifies: the city-level kwh x price identity, the merged-sum lineage identity (the
December 2025 version IS the plain city sum), the current final's parse and totals, and the
v04 correction's arithmetic on synthetic values. The correction's own inputs are the named
asks; until they land, the current final is the carried value.
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from global_invest.local_climate_regulation import local_climate_regulation_functions as lc

REFERENCE_DIR = os.path.join(os.path.dirname(lc.__file__), 'reference')


def test_city_savings_identity_and_the_lineage_sum():
    merged = pd.read_csv(os.path.join(REFERENCE_DIR, 'ITA_all_urban_valuations.csv'))
    ita_sum = lc.city_savings_identity(merged)          # raises if kwh x price breaks
    dec = pd.read_csv(os.path.join(REFERENCE_DIR, 'local_climate_regulation_gep_12_19_25.csv'))
    dec.columns = [c.strip() for c in dec.columns]
    dec_ita = lc.parse_dollar_column(dec.loc[dec['iso3_r250_label'] == 'ITA',
                                             'local_climate_regulation']).iloc[0]
    assert np.isclose(ita_sum, dec_ita, rtol=1e-6)      # the Dec 2025 version IS the city sum


def test_current_final_parses_to_the_known_totals():
    final = pd.read_csv(os.path.join(REFERENCE_DIR, 'local_climate_regulation_gep.csv'))
    final.columns = [c.strip() for c in final.columns]
    values = lc.parse_dollar_column(final['local_climate_regulation'])
    assert values.notna().sum() == 174
    assert np.isclose(values.sum() / 1e9, 14.19, rtol=1e-2)


def test_v04_correction_arithmetic_on_synthetic_values():
    corrected = lc.apply_country_marginal_consumption(
        np.array([lc.URBAN_COOLING_OLD_GLOBAL_MC * 10.0]), np.array([0.5]))
    assert np.isclose(corrected[0], 5.0)


def test_cdd_smoothstep_matches_hand_derived_points():
    # Far below base: 0. Far above: temp - (base + window). At the base: half the window
    # by the smoothstep's symmetry.
    assert lc.temp_degc_to_cdd_smooth(10.0) == 0.0
    assert np.isclose(lc.temp_degc_to_cdd_smooth(25.0), 2.0 + (25.0 - 19.0))  # 2w*s(1) + (T - base - w)
    assert np.isclose(lc.temp_degc_to_cdd_smooth(18.0), 1.0)                # 2w * s(0.5) = 1


def test_es_config_and_parameters_rows_hydrate_local_climate_regulation(tmp_path):
    from global_invest import utilities
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'local_climate_regulation', log=lambda *a: None)
    assert p.gep_base_year == 2019
    utilities.hydrate_es_parameters(p, 'local_climate_regulation', log=lambda *a: None)
    assert p.local_climate_regulation_final_path.endswith('local_climate_regulation_gep.csv')

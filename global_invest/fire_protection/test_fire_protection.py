"""Unit tests for the fire_protection port.

The centerpiece is EXACT REPLICATION: reference/ carries the source repo's committed
regression results (the chain's input) and its committed per-country output (the anchor),
so the whole calculation -- betas, avoided acres, valuation -- is verified against an
external anchor on every test run, no staged data needed. The synthetic tests pin the pieces
whose failure modes replication alone would not localize.
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from global_invest.fire_protection import fire_protection_functions as ffn

REFERENCE_DIR = os.path.join(os.path.dirname(__file__), 'reference')


def test_exact_replication_of_the_committed_reference_output():
    reference = pd.read_csv(os.path.join(REFERENCE_DIR, 'GEP_wildfire_2019_results.csv'))
    regression = pd.read_csv(os.path.join(REFERENCE_DIR, 'all_countries_regression_results_lag1.csv'))

    beta = ffn.compute_beta_differences(regression)
    avoided = ffn.compute_avoided_acres(
        beta, reference[['GID_0', 'total_burned_areas_ha_2018', 'n_adm2_units']])
    out = ffn.compute_gep(avoided, reference[['GID_0', 'damages_usd_per_acre_country']],
                          global_average=np.nan)

    merged = out.merge(reference, on='GID_0', suffixes=('_ours', '_ref'))
    assert len(out) == len(reference) == len(merged) == 161
    for col in ['Beta_baseline_coeff', 'Beta_diff_nn_hh', 'Beta_diff_ttn_tth',
                'additional_acres_2019_baseline', 'GEP_wildfire_2019_baseline',
                'GEP_wildfire_2019_nn_hh', 'GEP_wildfire_2019_ttn_tth']:
        ours, ref = merged[col + '_ours'], merged[col + '_ref']
        assert (ours.isna() == ref.isna()).all(), col
        both = ours.notna()
        assert np.allclose(ours[both], ref[both], rtol=1e-9), col
    for variant in ('baseline', 'nn_hh', 'ttn_tth'):
        col = f'GEP_wildfire_2019_{variant}'
        assert np.isclose(out[col].sum(), reference[col].sum())
    # The two negative-total variants are a property of the anchor, not a port bug: state it.
    assert out['GEP_wildfire_2019_baseline'].sum() < 0
    assert out['GEP_wildfire_2019_nn_hh'].sum() < 0
    assert out['GEP_wildfire_2019_ttn_tth'].sum() > 0


def test_reordered_regression_input_fails_loudly_instead_of_reassigning_betas():
    regression = pd.read_csv(os.path.join(REFERENCE_DIR, 'all_countries_regression_results_lag1.csv'))
    reordered = regression.sort_values('Y_var', ascending=False).reset_index(drop=True)
    with pytest.raises(ValueError, match='regression CSV order changed'):
        ffn.build_regression_type_ids(reordered)


def test_difference_stats_match_hand_derived_values():
    diff, se_diff, t_stat, p_value = ffn.difference_stats(
        np.array([0.5]), np.array([0.3]), np.array([0.2]), np.array([0.4]))
    assert np.isclose(diff[0], 0.3)
    assert np.isclose(se_diff[0], 0.5)            # sqrt(0.09 + 0.16)
    assert np.isclose(t_stat[0], 0.6)
    from scipy.stats import norm
    assert np.isclose(p_value[0], 2 * norm.cdf(-0.6))


def test_significance_levels_map_pvalues_to_star_counts():
    levels = ffn.significance_level([0.005, 0.03, 0.08, 0.5, np.nan])
    assert list(levels) == [3, 2, 1, 0, 0]


def test_damage_per_acre_global_average_fills_zero_and_missing():
    burned = pd.DataFrame({'GID_0': ['AAA', 'BBB', 'CCC'],
                           'total_burned_acres_country': [100.0, 200.0, 0.0]})
    damages = pd.DataFrame({'GID_0': ['AAA'], 'total_damages_country_usd': [1000.0]})
    rates, global_average = ffn.compute_damage_per_acre(burned, damages)
    assert np.isclose(global_average, 10.0)          # only AAA has a positive rate
    by_country = rates.set_index('GID_0')['damages_usd_per_acre_country']
    assert np.isclose(by_country['AAA'], 10.0)
    assert np.isclose(by_country['BBB'], 10.0)       # zero damages -> filled
    assert np.isclose(by_country['CCC'], 10.0)       # zero acres -> filled


def test_emdat_reader_filters_wildfire_and_scales_thousands(tmp_path):
    xlsx_path = str(tmp_path / 'emdat.xlsx')
    pd.DataFrame({'Disaster Type': ['Wildfire', 'Flood', 'Wildfire'],
                  'ISO': ['AAA', 'AAA', 'BBB'],
                  "Total Damage, Adjusted ('000 US$)": ['1,000', '999', '250']}).to_excel(
        xlsx_path, sheet_name='EM-DAT Data', index=False)
    damages = ffn.read_emdat_wildfire_damages(xlsx_path)
    by_country = damages.set_index('GID_0')['total_damages_country_usd']
    assert np.isclose(by_country['AAA'], 1_000_000)  # thousands -> USD; flood row excluded
    assert np.isclose(by_country['BBB'], 250_000)


def test_gep_sign_flips_additional_acres_into_avoided_damage():
    avoided = pd.DataFrame({'GID_0': ['AAA'],
                            'additional_acres_2019_baseline': [-10.0],
                            'additional_acres_2019_nn_hh': [5.0],
                            'additional_acres_2019_ttn_tth': [np.nan]})
    rates = pd.DataFrame({'GID_0': ['AAA'], 'damages_usd_per_acre_country': [2.0]})
    out = ffn.compute_gep(avoided, rates, global_average=np.nan)
    assert np.isclose(out['GEP_wildfire_2019_baseline'][0], 20.0)   # avoided acres -> positive GEP
    assert np.isclose(out['GEP_wildfire_2019_nn_hh'][0], -10.0)
    assert np.isnan(out['GEP_wildfire_2019_ttn_tth'][0])


def test_es_config_and_parameters_rows_hydrate_fire_protection(tmp_path):
    from global_invest import utilities
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'fire_protection', log=lambda *a: None)
    assert p.gep_base_year == 2019
    utilities.hydrate_es_parameters(p, 'fire_protection', log=lambda *a: None)
    assert p.fire_regression_results_path.endswith('all_countries_regression_results_lag1.csv')
    assert p.fire_emdat_path.endswith('.xlsx')
    assert p.fire_panel_path.endswith('gadm_adm2_panel_complete_2010_2023.csv')

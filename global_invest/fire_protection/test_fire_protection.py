"""Unit tests for the fire_protection port.

The centerpiece is EXACT REPLICATION: reference/ carries the source repo's committed
regression results (the calculation's input) and its committed per-country output (the anchor),
so the whole calculation -- betas, avoided acres, valuation -- is verified against an
external anchor on every test run, no staged data needed. The synthetic tests pin the pieces
whose failure modes replication alone would not localize.
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from global_invest import utilities
from global_invest.fire_protection import fire_protection_functions as ffn
from global_invest.fire_protection import fire_protection_tasks as fpt


def _base_data_project():
    """A bare ProjectFlow, only for its base_data_dir.

    The anchors are inputs, so a test finds them the way a run does rather than by walking
    directories of its own.
    """
    import tempfile
    import hazelbean as hb
    return hb.ProjectFlow(project_dir=os.path.join(tempfile.mkdtemp(), 'anchors'))


REFERENCE_DIR = utilities.service_data_dir(_base_data_project(), 'fire_protection')


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


def _emdat_frame():
    """An EM-DAT extract with one non-wildfire row and the damages as comma-separated thousands,
    plus a row whose ISO is missing (the source drops it rather than summing it namelessly)."""
    return pd.DataFrame({'Disaster Type': ['Wildfire', 'Flood', 'Wildfire', 'Wildfire'],
                         'ISO': ['AAA', 'AAA', 'BBB', None],
                         "Total Damage, Adjusted ('000 US$)": ['1,000', '999', '250', '500']})


def test_emdat_damages_filter_wildfire_and_scale_thousands():
    damages = ffn.emdat_wildfire_damages(_emdat_frame())
    by_country = damages.set_index('GID_0')['total_damages_country_usd']
    assert np.isclose(by_country['AAA'], 1_000_000)  # thousands -> USD; flood row excluded
    assert np.isclose(by_country['BBB'], 250_000)
    assert len(damages) == 2                         # the row without an ISO is dropped


def test_task_reader_totals_the_emdat_workbook(tmp_path):
    xlsx_path = str(tmp_path / 'emdat.xlsx')
    _emdat_frame().to_excel(xlsx_path, sheet_name=fpt.EMDAT_SHEET_NAME, index=False)
    damages = fpt.read_emdat_wildfire_damages(xlsx_path)
    by_country = damages.set_index('GID_0')['total_damages_country_usd']
    assert np.isclose(by_country['AAA'], 1_000_000)
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


# ---------------------------------------------------------------------------
# The GADM-to-country join.
# ---------------------------------------------------------------------------

COUNTRIES = pd.DataFrame({
    'gadm_r263_label': ['IND', 'Z01', 'Z07', 'XKX', 'KEN'],
    'iso3_r250_label': ['IND', 'IND', 'IND', 'XKX', 'KEN'],
})


def _gep_rows(rows):
    return pd.DataFrame(rows, columns=['GID_0'] + list(ffn.FIRE_VALUE_COLUMNS))


def test_gadm_subregions_sum_into_their_country():
    """India's value arrives as three GADM units. Joining on the label alone dropped Z01 and
    Z07, which left their value on a row with no country and still inside the total."""
    out = ffn.attach_countries(_gep_rows([
        ['IND', 1.0, 2.0, 3.0],
        ['Z01', 10.0, 20.0, 30.0],
        ['Z07', 100.0, 200.0, 300.0],
    ]), COUNTRIES).set_index('iso3_r250_label')

    assert len(out) == 1
    assert out.loc['IND', 'GEP_wildfire_2019_nn_hh'] == 222.0


def test_kosovos_gadm_code_resolves_to_the_accounts_label():
    out = ffn.attach_countries(_gep_rows([['XKO', 1.0, 2.0, 3.0]]), COUNTRIES)
    assert out['iso3_r250_label'].tolist() == ['XKX']
    assert out['GEP_wildfire_2019_nn_hh'].tolist() == [2.0]


def test_a_valued_code_that_resolves_to_no_country_raises():
    """Silently dropping it would take its value out of the account without saying so."""
    with pytest.raises(ValueError, match='resolve to no r250 country'):
        ffn.attach_countries(_gep_rows([['ZZZ', 1.0, 2.0, 3.0]]), COUNTRIES)


def test_an_unvalued_code_that_resolves_to_no_country_is_simply_dropped():
    out = ffn.attach_countries(_gep_rows([
        ['KEN', 1.0, 2.0, 3.0],
        ['ZZZ', np.nan, np.nan, np.nan],
    ]), COUNTRIES)
    assert out['iso3_r250_label'].tolist() == ['KEN']

"""Unit tests for health_hypertension's valuation.

Every function is pinned on a hand-built frame, so the odds-to-risk conversion and the
missing-input rule are stated as executable facts, and the staged tables anchor the joins.
"""
import numpy as np
import pandas as pd

from global_invest.health_hypertension import health_hypertension_functions as hf


def test_the_risk_ratio_conversion_is_grant_2014():
    # OR 0.95 at 30 percent baseline: RR = OR / (1 - P0 + P0*OR).
    assert np.isclose(hf.risk_ratio_from_odds_ratio(0.95, 0.30),
                      0.95 / (1.0 - 0.30 + 0.30 * 0.95))
    # at a vanishing baseline the risk ratio approaches the odds ratio
    assert np.isclose(hf.risk_ratio_from_odds_ratio(0.95, 1e-9), 0.95, atol=1e-6)


def test_the_valuation_is_cases_times_cost():
    df = pd.DataFrame({'prevalence': [0.4], 'urban_ndvi_pop_sum': [1e6],
                       'share_30_79': [0.5], 'cost_per_case_usd': [1000.0]})
    out = hf.health_hypertension_gep(df, odds_ratio=0.95)
    rr = hf.risk_ratio_from_odds_ratio(0.95, 0.4)
    expected_cases = 0.4 * (1e6 / 0.1) * (1.0 - rr) * 0.5
    assert np.isclose(out['avoided_cases'].iloc[0], expected_cases)
    assert np.isclose(out['health_hypertension_gep'].iloc[0], expected_cases * 1000.0)
    assert out['note'].iloc[0] == ''


def test_a_missing_input_is_an_na_with_its_absence_named():
    df = pd.DataFrame({'prevalence': [0.4, 0.4], 'urban_ndvi_pop_sum': [1e6, 1e6],
                       'share_30_79': [0.5, 0.5], 'cost_per_case_usd': [1000.0, np.nan]})
    out = hf.health_hypertension_gep(df, odds_ratio=0.95)
    assert pd.isna(out['health_hypertension_gep'].iloc[1])
    assert 'no treatment cost' in out['note'].iloc[1]


def test_the_staged_tables_join_on_the_account_labels():
    import os, tempfile, hazelbean as hb
    from global_invest import utilities
    base = utilities.service_data_dir(
        hb.ProjectFlow(project_dir=os.path.join(tempfile.mkdtemp(), 'anchors')),
        'health_hypertension')
    costs = pd.read_csv(os.path.join(base, 'hypertension_cost_per_case.csv'))
    prevalence = pd.read_csv(os.path.join(base, 'who_hypertension_prevalence_2019.csv'))
    shares = pd.read_csv(os.path.join(base, 'wb_share_age_30_79_2019.csv'))
    reference = pd.read_csv(os.path.join(base, 'health_hypertension_appendix_table6.csv'))
    # every cost country has a prevalence, an age share and (except Ghana) a reference row
    assert set(costs['iso3_r250_label']) <= set(prevalence['iso3'])
    assert set(costs['iso3_r250_label']) <= set(shares['iso3'])
    assert set(reference['iso3_r250_label']) == set(costs['iso3_r250_label']) - {'GHA'}


def test_the_cost_transfer_fits_observed_and_keeps_studies():
    # two observed countries on an exact ln-ln line: slope 1, smearing 1, R2 1;
    # the unobserved country gets the line's prediction, the observed keep their studies.
    costs = pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB'], 'cost_per_case_usd': [100.0, 1000.0]})
    gdppc = pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB', 'CCC'],
                          'gdp_pc_usd': [1000.0, 10000.0, 100000.0]})
    out, fit = hf.extrapolated_cost_per_case(costs, gdppc)
    out = out.set_index('iso3_r250_label')
    assert np.isclose(fit['slope'], 1.0) and np.isclose(fit['smearing'], 1.0)
    assert out.loc['AAA', 'cost_source'] == 'study' and out.loc['AAA', 'cost_per_case_usd'] == 100.0
    assert out.loc['CCC', 'cost_source'] == 'extrapolated'
    assert np.isclose(out.loc['CCC', 'cost_per_case_usd'], 10000.0)

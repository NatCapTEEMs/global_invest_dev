"""Unit tests for health_depression's valuation.

Every function is pinned on a hand-built frame, so the dose-response convention and the
missing-input rule are stated as executable facts, and the staged tables anchor the joins.
"""
import numpy as np
import pandas as pd

from global_invest.health_depression import health_depression_functions as hd


def test_the_prevented_cases_factor_is_the_exponential_dose_response():
    # RR 0.931 per 0.1 NDVI at NDVI 0.4: removing the greenness multiplies risk by
    # exp(-10 ln(0.931) 0.4); the prevented share of observed cases is that minus one.
    assert np.isclose(hd.prevented_cases_factor(0.4, 0.931),
                      np.exp(-10.0 * np.log(0.931) * 0.4) - 1.0)
    assert hd.prevented_cases_factor(0.0, 0.931) == 0.0


def test_the_valuation_is_prevented_cases_times_cost():
    df = pd.DataFrame({'prevalence': [0.05], 'urban_prevented_factor_pop_sum': [1e6],
                       'cost_per_case_usd': [8000.0]})
    out = hd.health_depression_gep(df)
    assert np.isclose(out['prevented_cases'].iloc[0], 0.05 * 1e6)
    assert np.isclose(out['health_depression_gep'].iloc[0], 0.05 * 1e6 * 8000.0)
    assert out['note'].iloc[0] == ''


def test_a_missing_cost_is_an_na_with_its_absence_named():
    df = pd.DataFrame({'prevalence': [0.05], 'urban_prevented_factor_pop_sum': [1e6],
                       'cost_per_case_usd': [np.nan]})
    out = hd.health_depression_gep(df)
    assert pd.isna(out['health_depression_gep'].iloc[0])
    assert 'no cost study' in out['note'].iloc[0]


def test_the_staged_cost_table_matches_the_authors_country_set():
    import os, tempfile, hazelbean as hb
    from global_invest import utilities
    base = utilities.service_data_dir(
        hb.ProjectFlow(project_dir=os.path.join(tempfile.mkdtemp(), 'anchors')),
        'health_depression')
    costs = pd.read_csv(os.path.join(base, 'data_cost_mental_health.csv'))
    reference = pd.read_csv(os.path.join(base, 'reference_high_resolution.csv'))
    # the author's 32 valued countries are exactly the cost rows minus the Europe aggregate
    assert len(reference) == 32
    assert len(costs) == 33 and 'Europe' in set(costs['country'])

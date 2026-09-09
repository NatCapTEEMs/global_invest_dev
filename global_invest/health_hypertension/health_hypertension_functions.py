# -*- coding: utf-8 -*-
"""Physical-health science: urban greenspace valued as avoided hypertension treatment cost.

Nothing here opens a file. The task layer reads the greenness, population, prevalence and cost
tables, hands the frames in and writes back what it gets, so every step can be pinned on a
hand-built input in the test suite.

The valuation: greener urban surroundings lower the odds of hypertension. Per country, avoided
cases are the hypertension prevalence times the population-weighted greenness (in 0.1 NDVI
increments, against a zero-vegetation counterfactual) times the per-increment risk reduction,
over the urban population aged 30 to 79; the value is those cases at the country's published
cost of treating hypertension. The odds ratio from the meta-analyses converts to a risk ratio
against the country's own baseline prevalence.
"""
import pandas as pd
import numpy as np


def risk_ratio_from_odds_ratio(odds_ratio, baseline_prevalence):
    """Convert an odds ratio to a risk ratio at a baseline prevalence (Grant 2014).

    Args:
        odds_ratio: odds ratio of hypertension per 0.1 NDVI increment.
        baseline_prevalence: the baseline risk, as a fraction (0.30 for 30 percent).

    Returns:
        The risk ratio, elementwise over whatever is passed.
    """
    return odds_ratio / (1.0 - baseline_prevalence + baseline_prevalence * odds_ratio)


def extrapolated_cost_per_case(costs_df, gdppc_df):
    """Treatment cost per case for every country: observed where a study exists, else transferred.

    The transfer is the ln-ln regression of cost on GDP per capita over the observed countries,
    retransformed with Duan smearing -- the same benefit-transfer construction the water-quality
    strand's owners use.

    Args:
        costs_df (pd.DataFrame): iso3_r250_label and cost_per_case_usd for the studied countries.
        gdppc_df (pd.DataFrame): iso3_r250_label and gdp_pc_usd for every country.

    Returns:
        (pd.DataFrame, dict): one row per gdppc country with `cost_per_case_usd` (the observed
        value where a study exists, the transferred one otherwise) and `cost_source`
        ('study' or 'extrapolated'); and the fit's {slope, r2, smearing, n} for the log.
    """
    m = costs_df.merge(gdppc_df, on='iso3_r250_label')
    x, y = np.log(m['gdp_pc_usd']), np.log(m['cost_per_case_usd'])
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x)
    smearing = float(np.mean(np.exp(resid)))
    fit = {'slope': float(slope), 'r2': float(1 - resid.var() / y.var()),
           'smearing': smearing, 'n': len(m)}
    out = gdppc_df.copy()
    out['cost_per_case_usd'] = np.exp(intercept + slope * np.log(out['gdp_pc_usd'])) * smearing
    out['cost_source'] = 'extrapolated'
    observed = dict(zip(costs_df['iso3_r250_label'], costs_df['cost_per_case_usd']))
    has_study = out['iso3_r250_label'].isin(observed)
    out.loc[has_study, 'cost_per_case_usd'] = out.loc[has_study, 'iso3_r250_label'].map(observed)
    out.loc[has_study, 'cost_source'] = 'study'
    return out[['iso3_r250_label', 'cost_per_case_usd', 'cost_source']], fit


def health_hypertension_gep(df, odds_ratio):
    """The service value per country: avoided cases times the cost of a case.

    Args:
        df (pd.DataFrame): one row per country, carrying `prevalence` (fraction of adults 30-79
            with hypertension), `urban_ndvi_pop_sum` (the sum of NDVI times population over the
            country's urban pixels), `share_30_79` (the 30-79 share of total population) and
            `cost_per_case_usd` (treatment cost of one case).
        odds_ratio: odds ratio of hypertension per 0.1 NDVI increment from the meta-analyses.

    Returns:
        pd.DataFrame: the input with `avoided_cases` and `health_hypertension_gep` in USD
        added. A country missing any input carries NA there, never a silent zero, and the note
        column names which.
    """
    out = df.copy()
    rr = risk_ratio_from_odds_ratio(odds_ratio, out['prevalence'])
    out['avoided_cases'] = (out['prevalence'] * (out['urban_ndvi_pop_sum'] / 0.1)
                            * (1.0 - rr) * out['share_30_79'])
    out['health_hypertension_gep'] = out['avoided_cases'] * out['cost_per_case_usd']
    absences = {'prevalence': 'no prevalence', 'urban_ndvi_pop_sum': 'no urban greenness',
                'share_30_79': 'no age structure', 'cost_per_case_usd': 'no treatment cost'}
    out['note'] = out.apply(
        lambda row: '; '.join(absences[c] for c in absences if pd.isna(row[c])), axis=1)
    return out

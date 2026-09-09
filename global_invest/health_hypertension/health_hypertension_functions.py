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

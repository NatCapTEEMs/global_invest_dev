# -*- coding: utf-8 -*-
"""Mental-health science: urban greenspace valued as the depression cases it prevents.

Nothing here opens a file. The task layer reads the greenness, population, prevalence and cost
tables, hands the frames and arrays in and writes back what it gets, so every step can be
pinned on a hand-built input in the test suite.

The valuation follows the InVEST urban mental health model: removing a pixel's greenness would
raise its residents' depression risk by exp(-10 ln(RR) NDVI), so the cases today's nature
prevents are the observed cases times that factor minus one, summed over urban pixels and
priced at the societal cost of a depression case. The meta-analytic odds ratio converts to a
risk ratio at the country's own prevalence as the baseline (Zhang and Yu 1998), the same
conversion the hypertension strand uses.
"""
import numpy as np
import pandas as pd

from global_invest.health_hypertension.health_hypertension_functions import risk_ratio_from_odds_ratio


def prevented_cases_factor(ndvi, risk_ratio_per_01):
    """The prevented-cases multiplier per person: exp(-10 ln(RR) NDVI) - 1.

    Args:
        ndvi: the pixel's annual NDVI (0 to 1), elementwise.
        risk_ratio_per_01: the risk ratio of depression per 0.1 NDVI increment.

    Returns:
        The factor observed cases are multiplied by to get cases the greenness prevents,
        0 where NDVI is 0.
    """
    return np.exp(-10.0 * np.log(risk_ratio_per_01) * ndvi) - 1.0


def health_depression_gep(df):
    """The service value per country: prevented cases times the cost of a case.

    Args:
        df (pd.DataFrame): one row per country, carrying `prevalence` (fraction with
            depression), `urban_prevented_factor_pop_sum` (the prevented-cases factor times
            population summed over urban pixels, already at the country's risk ratio) and
            `cost_per_case_usd` (societal cost of one case).

    Returns:
        pd.DataFrame: the input with `prevented_cases` and `health_depression_gep` added. A
        country missing any input carries NA there, never a silent zero, and the note column
        names which.
    """
    out = df.copy()
    out['prevented_cases'] = out['prevalence'] * out['urban_prevented_factor_pop_sum']
    out['health_depression_gep'] = out['prevented_cases'] * out['cost_per_case_usd']
    absences = {'prevalence': 'no prevalence',
                'urban_prevented_factor_pop_sum': 'no urban greenness',
                'cost_per_case_usd': 'no cost study'}
    out['note'] = out.apply(
        lambda row: '; '.join(absences[c] for c in absences if pd.isna(row[c])), axis=1)
    return out

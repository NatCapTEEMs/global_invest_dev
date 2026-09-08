# -*- coding: utf-8 -*-
"""Pest-control science: biological pest control valued as the yield it adds to organic crops.

Nothing here opens a file. The task layer reads the organic areas, yields, prices and the crop
classification, hands the joined frame in and writes back what it gets, so every step can be
pinned on a hand-built input in the test suite.

The valuation: predators and other natural enemies raise crop yields where synthetic pesticides
are absent, so the service is the extra production on organically managed area, at the organic
producer price. Per crop and country, the value is the proportional yield benefit (from a
meta-analytic log response ratio), times organic production (organic area times the
conventional yield scaled by an organic-to-conventional yield ratio), times the organic
producer price (the conventional price times an organic-to-conventional price premium). Crop
groups without a measured benefit carry a true zero, deliberately.
"""
import pandas as pd
import numpy as np

KG_PER_TONNE = 1000.0


def yield_benefit_from_lnrr(lnrr):
    """The proportional yield benefit implied by a log response ratio: exp(lnRR) - 1.

    Args:
        lnrr: the meta-analytic log response ratio, elementwise over whatever is passed.

    Returns:
        The proportional benefit (0.13 means yields are 13 percent higher with the service).
    """
    return np.exp(lnrr) - 1.0


def pest_control_value(df, price_premium):
    """The service value per row: benefit times organic production times organic price.

    Args:
        df (pd.DataFrame): one row per crop-country, carrying `organic_ha`, `yield_kg_ha`,
            `price_usd_t`, `effect_lnrr` (0 for crop groups without a measured benefit) and
            `yield_ratio` (organic-to-conventional).
        price_premium (float): the organic-to-conventional producer price ratio.

    Returns:
        pd.DataFrame: the input with `pest_control_value` in USD added. A row missing a yield
        or a price carries NA there, never a silent zero, and the note column names which; a
        zero-effect crop group is a true zero with an empty note.
    """
    out = df.copy()
    out['pest_control_value'] = (
        yield_benefit_from_lnrr(out['effect_lnrr'])
        * out['organic_ha']
        * (out['yield_kg_ha'] / KG_PER_TONNE) * out['yield_ratio']
        * out['price_usd_t'] * price_premium)
    absences = {'yield_kg_ha': 'no yield', 'price_usd_t': 'no price'}
    out['note'] = out.apply(
        lambda row: '; '.join(absences[c] for c in absences if pd.isna(row[c])), axis=1)
    return out

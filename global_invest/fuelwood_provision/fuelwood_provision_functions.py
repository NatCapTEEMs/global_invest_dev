"""Fuelwood-provision science: the CAS per-country results onto the account's country list.

The CAS team's method (their staged appendix: methodology, input data, code, output result)
values national fuelwood harvest volumes through their energy content at substitute-fuel
replacement prices, with the ecosystem share taken as gross value minus human inputs. The
library carries their per-country results; rebuilding the chain from FAO volumes is the
recorded later step.
"""
import numpy as np
import pandas as pd

# The results sheet's own title is '1000USD ($2019 USD)', so the values are thousands of
# constant-2019 dollars.
FUELWOOD_RESULTS_UNIT_USD = 1000.0

# Codes the account list cannot carry, each with the reason. Any OTHER unmatched code is an
# error, not a note.
FUELWOOD_UNALLOCATABLE = {
    'ANT': 'Netherlands Antilles, dissolved 2010; no defensible split across successors',
}


def fuelwood_gep_by_country(results_df, countries_df):
    """The CAS results joined to the account ids, scaled to dollars.

    Args:
        results_df (pd.DataFrame): the CAS sheet, columns `ISO-3` and `Value(2019 $)`.
        countries_df (pd.DataFrame): the account list with iso3_r250_id and iso3_r250_label.

    Returns:
        (pd.DataFrame, pd.DataFrame): per-id fuelwood_provision_gep in 2019 USD, and the
        rows the account list cannot carry (reported, never silently dropped).

    Raises:
        ValueError: on a missing column, a duplicated code, or an unmatched code that is not
            in FUELWOOD_UNALLOCATABLE.
    """
    for column in ('ISO-3', 'Value(2019 $)'):
        if column not in results_df.columns:
            raise ValueError('the CAS fuelwood sheet is missing column %r' % column)
    df = results_df.rename(columns={'ISO-3': 'iso3', 'Value(2019 $)': 'value_thousand_usd'})
    df['iso3'] = df['iso3'].astype(str).str.strip()
    if df['iso3'].duplicated().any():
        raise ValueError('duplicated ISO-3 rows: %s'
                         % sorted(df.loc[df['iso3'].duplicated(), 'iso3']))
    if (df['value_thousand_usd'] < 0).any():
        raise ValueError('negative fuelwood values')
    labels = countries_df[['iso3_r250_id', 'iso3_r250_label']].drop_duplicates()
    merged = df.merge(labels, how='left', left_on='iso3', right_on='iso3_r250_label')
    unmatched = merged[merged['iso3_r250_id'].isna()]
    unknown = sorted(set(unmatched['iso3']) - set(FUELWOOD_UNALLOCATABLE))
    if unknown:
        raise ValueError('CAS fuelwood codes not on the account list and not recorded as '
                         'unallocatable: %s' % unknown)
    matched = merged[merged['iso3_r250_id'].notna()].copy()
    matched['fuelwood_provision_gep'] = (matched['value_thousand_usd']
                                         * FUELWOOD_RESULTS_UNIT_USD)
    out = matched[['iso3_r250_id', 'fuelwood_provision_gep']].copy()
    out['iso3_r250_id'] = out['iso3_r250_id'].astype(int)
    unallocated = unmatched[['iso3', 'value_thousand_usd']].copy()
    unallocated['value_usd'] = unallocated['value_thousand_usd'] * FUELWOOD_RESULTS_UNIT_USD
    unallocated['reason'] = unallocated['iso3'].map(FUELWOOD_UNALLOCATABLE)
    return out, unallocated.drop(columns='value_thousand_usd')

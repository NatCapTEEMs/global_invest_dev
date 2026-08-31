# -*- coding: utf-8 -*-
"""Extractive-materials science: the mineral-rent share of GDP, valued.

Nothing here opens a file. The task layer reads the two World Bank CSVs, hands the frames in and
writes back what it gets, so every step below can be pinned on a hand-built input in the test
suite.
"""
import logging

import hazelbean as hb

from global_invest import utilities

# The World Bank publishes one indicator per file in wide form: four descriptive columns
# (country name, country code, indicator name, indicator code) and then one column per year.
WORLD_BANK_COUNTRY_COLUMN = 'Country Code'


def world_bank_wide_to_long(df_wide, value_column):
    """A wide World Bank indicator table reshaped to one row per country and year.

    Args:
        df_wide (pd.DataFrame): the indicator as published, descriptive columns ahead of one
            column per year.
        value_column (str): the name the indicator's values take in the long table.

    Returns:
        pd.DataFrame: WORLD_BANK_COUNTRY_COLUMN, year, and value_column. Melting the year
        columns and nothing else is what drops the descriptive ones, so a byte-order mark on
        the first header cannot reach the result.
    """
    year_columns = [column for column in df_wide.columns if str(column).isdigit()]
    return df_wide.melt(id_vars=[WORLD_BANK_COUNTRY_COLUMN], value_vars=year_columns,
                        var_name='year', value_name=value_column)


def group_countries(df):
    """Country-year rows summed to one global row per year."""
    return utilities.sum_countries_to_year(df, 'Value')



def mineral_rent_gep(mineral_rent_percent, gdp_current_usd, factor):
    """Mineral provision value: the rent share of GDP, times the attribution factor.

    Args:
        mineral_rent_percent: mineral rents as a percentage of GDP (the World Bank series).
        gdp_current_usd: GDP in current USD.
        factor: the attribution factor applied to the rent (the 0.49 whose source is the
            service's open question).

    Returns:
        The valued rent, in the same units as gdp_current_usd.
    """
    return (mineral_rent_percent / 100.0) * gdp_current_usd * factor

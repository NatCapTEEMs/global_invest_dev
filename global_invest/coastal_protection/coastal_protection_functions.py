# -*- coding: utf-8 -*-
"""Coastal-protection science: storm damage avoided by mangroves and by coral reefs.

Two components, valued upstream and combined here. The mangrove component is the World Bank
Changing Wealth of Nations mangrove table, which already reports an annual protection value per
country in current 2019 USD, so it is read, joined to the country correspondence and summed.
The coral-reef component is the annual expected flood-protection benefit table, reported in 2011
USD, so it is carried to the base year by the cumulative World Bank GDP deflator over the
intervening years before being combined with the mangrove component.

File reading is kept to three thin functions at the top; everything below them is a pure
transformation over frames, which is what the tests exercise.
"""
import logging

import pandas as pd
import hazelbean as hb

# The coral-reef benefit table reports 2011 USD, so the deflator runs over 2012 through the base
# year: eight annual rates whose product carries a 2011 value to 2019.
CORAL_REEF_VALUE_YEAR = 2011
# The year the service reports. Kept beside the coral year because the two define the deflator
# span together, and the mangrove table only carries this year.
COASTAL_PROTECTION_BASE_YEAR = 2019
# The World Bank deflator series is an annual percentage change, so each year contributes
# 1 + rate/100 to the cumulative multiplier.
DEFLATOR_PERCENT = 100.0


def read_mangrove_values(path):
    """The CWoN mangrove workbook read and renamed. See clean_mangrove_values."""
    df_raw = pd.read_excel(path, sheet_name='Sheet1', engine='openpyxl')
    logging.info(f'Loaded mangrove coastal protection values from {path} ({df_raw.shape[0]} rows).')
    return clean_mangrove_values(df_raw)


def read_coral_reef_values(path):
    """The coral-reef annual expected benefit workbook, read as shipped.

    The table already carries ee_r264_name, coral_reef_value and year, so nothing is renamed or
    rescaled on the way in. Its currency year is CORAL_REEF_VALUE_YEAR; deflating to the base
    year happens in coral_reef_gep_by_country.
    """
    df = pd.read_excel(path, sheet_name='Sheet1', engine='openpyxl')
    logging.info(f'Loaded coral reef coastal protection values from {path} ({df.shape[0]} rows).')
    return df


def read_gdp_inflation_deflator(path):
    """The World Bank GDP deflator workbook read and reshaped to long.

    Source: https://data.worldbank.org/indicator/NY.GDP.DEFL.KD.ZG
    """
    df_raw = pd.read_excel(path, engine='openpyxl')
    logging.info(f'Loaded GDP deflator from {path} ({df_raw.shape[0]} rows).')
    return reshape_gdp_inflation_deflator(df_raw)


def clean_mangrove_values(df_raw):
    """The mangrove table keyed the way the country join expects.

    The workbook is already one row per country and year with the annual protection value in
    current USD, so the only work is renaming its two key columns onto the correspondence's
    vocabulary and making the year an integer.

    Args:
        df_raw (pd.DataFrame): the CWoN mangrove table as shipped, with countrycode,
            annual_value_2019 and year.

    Returns:
        pd.DataFrame: the same rows with ee_r264_label and Value in place of those two columns.
    """
    df = df_raw.rename(columns={'countrycode': 'ee_r264_label', 'annual_value_2019': 'Value'})
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)
    logging.info(f'Finished cleaning up ({df.shape[0]} rows).')
    return df


def reshape_gdp_inflation_deflator(df_raw):
    """The World Bank deflator table melted from one column per year to one row per country-year.

    Args:
        df_raw (pd.DataFrame): the World Bank wide table as shipped.

    Returns:
        pd.DataFrame: Country Name, Country Code, Indicator Name, Indicator Code, year, value.
    """
    df = df_raw.melt(
        id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
        var_name='year',
        value_name='value',
    )
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
    return df


def deflator_multiplier_by_country(df_deflator_long, start_year, end_year):
    """The cumulative GDP deflator multiplier per country over an inclusive year span.

    Each year in the span contributes 1 + rate/100 and the years are multiplied together, so a
    value in start_year - 1 currency times this multiplier is that value in end_year currency.

    Args:
        df_deflator_long (pd.DataFrame): the long deflator table, with Country Code,
            Country Name, year and value (annual percent change).
        start_year (int): first year whose inflation is applied.
        end_year (int): last year whose inflation is applied.

    Returns:
        pd.DataFrame: ee_r264_label, ee_r264_name, deflator_multiplier.
    """
    df = df_deflator_long[df_deflator_long['year'].between(start_year, end_year)].copy()
    df['multiplier'] = 1 + df['value'] / DEFLATOR_PERCENT
    df = (df.groupby(['Country Code', 'Country Name'], as_index=False)['multiplier']
          .prod()
          .rename(columns={'multiplier': 'deflator_multiplier'}))
    return df.rename(columns={'Country Code': 'ee_r264_label', 'Country Name': 'ee_r264_name'})


def get_inflation_deflator_multiplier(path, start_year, end_year):
    """The deflator workbook read and reduced to a per-country multiplier."""
    return deflator_multiplier_by_country(read_gdp_inflation_deflator(path), start_year, end_year)


def mangrove_gep_by_country(gdf_countries, df_mangrove_value):
    """Mangrove protection value summed to one row per country and year.

    The mangrove table is keyed on ee_r264_label, so a split country arrives as its sub-region
    rows; summing on iso3_r250_label is what puts those back together into one country. Countries
    the correspondence does not carry drop out with the inner join, and rows the correspondence
    matched but that carry no country label are dropped rather than summed into a nameless total.

    Args:
        gdf_countries (pd.DataFrame): the r264 country correspondence.
        df_mangrove_value (pd.DataFrame): the cleaned mangrove table, with ee_r264_label,
            year and Value.

    Returns:
        pd.DataFrame: iso3_r250_label, year, coastal_protection_gep_mangrove.
    """
    df = hb.df_merge(gdf_countries, df_mangrove_value, how='inner', on='ee_r264_label')
    df = (df.groupby(['iso3_r250_label', 'year'], as_index=False, dropna=False)['Value'].sum())
    df = df.dropna(subset=['iso3_r250_label'])
    return df.rename(columns={'Value': 'coastal_protection_gep_mangrove'})


def coral_reef_gep_by_country(gdf_countries, df_coral_reef_value, df_deflator_multiplier,
                              base_year=COASTAL_PROTECTION_BASE_YEAR):
    """Coral-reef protection value carried to the base year, one row per country.

    The coral table is keyed on country NAME, so it is joined to the correspondence on
    ee_r264_name and then de-duplicated on the country/year/value triple, which collapses the
    sub-region rows a split country's name matches. The surviving rows are multiplied by their
    country's cumulative deflator and stamped with the base year.

    todo A country the World Bank deflator table does not cover gets a missing multiplier, so its
    coral value becomes missing and the sum below reports it as zero rather than as unvalued. On
    the shipped data that silently zeroes nine countries, Taiwan the largest. Flagged rather than
    fixed, because fixing it changes the published number.

    Args:
        gdf_countries (pd.DataFrame): the r264 country correspondence.
        df_coral_reef_value (pd.DataFrame): the coral table, with ee_r264_name, coral_reef_value
            and year.
        df_deflator_multiplier (pd.DataFrame): ee_r264_label and deflator_multiplier.
        base_year (int): the year the deflated values are stamped with and filtered to.

    Returns:
        pd.DataFrame: iso3_r250_label, year, coastal_protection_gep_coral_reef.
    """
    df = hb.df_merge(gdf_countries, df_coral_reef_value, how='inner', on='ee_r264_name')
    df = df.dropna(subset=['coral_reef_value'])
    df = df.drop_duplicates(subset=['iso3_r250_label', 'year', 'coral_reef_value'])

    deflated = hb.df_merge(df, df_deflator_multiplier, how='left', on='ee_r264_label')
    deflated['coral_reef_value'] = deflated['coral_reef_value'] * deflated['deflator_multiplier']
    deflated['year'] = base_year

    df = pd.concat([df, deflated], ignore_index=True)
    df = df.groupby(['iso3_r250_label', 'year'], as_index=False, dropna=False)['coral_reef_value'].sum()
    df = df.rename(columns={'coral_reef_value': 'coastal_protection_gep_coral_reef'})
    return df[df['year'] == base_year]


def combine_coastal_components(df_mangrove, df_coral_reef):
    """The two components on one row per country and year, plus their sum.

    The join is outer because the two tables cover different countries, and a country present in
    only one of them contributes that component alone rather than dropping out.

    Args:
        df_mangrove (pd.DataFrame): iso3_r250_label, year, coastal_protection_gep_mangrove.
        df_coral_reef (pd.DataFrame): iso3_r250_label, year, coastal_protection_gep_coral_reef.

    Returns:
        pd.DataFrame: both components, their sum as coastal_protection_gep, and a Value copy of
        the sum for the shared reporting templates.
    """
    df = pd.merge(df_mangrove, df_coral_reef, how='outer', on=['iso3_r250_label', 'year'])
    df = df.fillna(0)
    df['coastal_protection_gep'] = (df['coastal_protection_gep_mangrove']
                                    + df['coastal_protection_gep_coral_reef'])
    df['Value'] = df['coastal_protection_gep']
    return df


def attach_country_attributes(df_gep, gdf_countries):
    """Country rows carrying the correspondence's identifiers and attributes.

    The correspondence is collapsed to one canonical row per country first: joining against r264
    as shipped would repeat each country once per sub-region.

    Args:
        df_gep (pd.DataFrame): one row per country and year, keyed on iso3_r250_label.
        gdf_countries (pd.DataFrame): the r264 country correspondence.

    Returns:
        pd.DataFrame: df_gep with the identifier and attribute columns attached.
    """
    attribute_columns = [
        'ee_r264_id', 'iso3_r250_id', 'ee_r264_label', 'iso3_r250_label', 'ee_r264_name',
        'iso3_r250_name', 'continent', 'region_un', 'region_wb', 'income_grp', 'subregion',
        'area_code_M49', 'area_code', 'country',
    ]
    one_row_per_country = gdf_countries[
        gdf_countries['ee_r264_label'] == gdf_countries['iso3_r250_label']]
    one_row_per_country = one_row_per_country[
        [c for c in one_row_per_country.columns if c in attribute_columns]]
    return pd.merge(df_gep, one_row_per_country, how='left', on='iso3_r250_label')

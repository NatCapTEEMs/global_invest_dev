# -*- coding: utf-8 -*-
"""Coastal-protection science: storm damage avoided by mangroves and by coral reefs.

Two components, combined here, and only one of them is ours to compute.

The mangrove component IS computed here: the CWoN mangrove table carries the protected area in
hectares and the value per hectare beside its own published total, so the value is the product
of those two rather than the published column read through. That reproduces the published total
to within 2e-7 and, unlike reading it, would show us a disagreement if one appeared.

The coral-reef component is NOT computed here. Its table carries a finished annual benefit per
country and nothing underneath it, so there is no calculation to run; it is carried from 2011 to the
base year by the shared US CPI factor (utilities.usd_deflation_factor) and added. That is the open
ask on this service.

Every function here is a pure transformation over frames, which is what the tests exercise. The
task module reads the mangrove and coral-reef workbooks, the shared US CPI series and the list of
countries the mangrove source excluded, and passes them in.
"""
import logging

import pandas as pd
import hazelbean as hb

# The coral-reef benefit table reports 2011 USD, carried to the base year by the shared US CPI
# factor. The values are USD, not local currency, so one factor serves every country.
CORAL_REEF_VALUE_YEAR = 2011
# The year the service reports. Kept beside the coral year because the two define the CPI span
# together, and the mangrove table only carries this year.
COASTAL_PROTECTION_BASE_YEAR = 2019


# The two columns the mangrove value is computed from, and the published total kept beside it
# as the comparison anchor.
MANGROVE_AREA_COLUMN = 'mangrove_ha'
MANGROVE_VALUE_PER_HA_COLUMN = 'value_per_ha_2019'
MANGROVE_PUBLISHED_VALUE_COLUMN = 'annual_value_2019'

# The published column naming what each country's total covers, and the label a country
# carrying both fully measured components gets.
COVERAGE_COLUMN = 'coastal_protection_coverage'
BOTH_COMPONENTS_LABEL = 'mangrove and coral protection'


def clean_mangrove_values(df_raw):
    """The mangrove protection value, computed from area times value per hectare.

    The workbook publishes a finished annual value, but it also carries the two numbers that
    value is made of, so the value is computed here instead of read. A country missing either
    stays empty rather than becoming zero: no area or no price is not no protection. A zero
    area the table reports stays zero even where the rate is missing, because zero habitat earns
    zero whatever a hectare would have been worth. Those zeros are source-reported, not all
    independently measured (see combine_coastal_components).

    Args:
        df_raw (pd.DataFrame): the CWoN mangrove table as shipped, carrying countrycode, year,
            mangrove_ha, value_per_ha_2019 and annual_value_2019.

    Returns:
        pd.DataFrame: ee_r264_label, year, Value (ours) and Value_published (the anchor).
    """
    df = df_raw.rename(columns={'countrycode': 'ee_r264_label'})
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)
    area = pd.to_numeric(df[MANGROVE_AREA_COLUMN], errors='coerce')
    value_per_ha = pd.to_numeric(df[MANGROVE_VALUE_PER_HA_COLUMN], errors='coerce')
    df['Value'] = area * value_per_ha
    # A reported zero area stays zero whatever the rate column holds (0 x missing must not read
    # as unknown). A missing area stays missing: no quantity is not no protection.
    df.loc[area == 0, 'Value'] = 0.0
    df['Value_published'] = pd.to_numeric(df[MANGROVE_PUBLISHED_VALUE_COLUMN], errors='coerce')
    logging.info(f'Finished cleaning up ({df.shape[0]} rows).')
    return df


def mangrove_gep_by_country(gdf_countries, df_mangrove_value):
    """Mangrove protection value summed to one row per country and year.

    The mangrove table is keyed on ee_r264_label, so a split country arrives as its sub-region
    rows; summing on iso3_r250_label is what puts those back together into one country. Countries
    the correspondence does not carry drop out with the inner join, and rows the correspondence
    matched but that carry no country label are dropped rather than summed into a nameless total.
    A country whose every row carries a missing value sums to missing, not zero (min_count=1);
    a partially measured country sums its measured rows alone. `mangrove_fully_measured` says
    whether every row the country contributed carried a value -- a partially measured country
    keeps its measured sum and its coverage label calls the component partial.

    Args:
        gdf_countries (pd.DataFrame): the r264 country correspondence.
        df_mangrove_value (pd.DataFrame): the cleaned mangrove table, with ee_r264_label,
            year and Value.

    Returns:
        pd.DataFrame: iso3_r250_label, year, coastal_protection_gep_mangrove.
    """
    df = hb.df_merge(gdf_countries, df_mangrove_value, how='inner', on='ee_r264_label')
    df = (df.groupby(['iso3_r250_label', 'year'], as_index=False, dropna=False)
            .agg(Value=('Value', lambda v: v.sum(min_count=1)),
                 mangrove_fully_measured=('Value', lambda v: bool(v.notna().all()))))
    df = df.dropna(subset=['iso3_r250_label'])
    return df.rename(columns={'Value': 'coastal_protection_gep_mangrove'})


def coral_reef_gep_by_country(gdf_countries, df_coral_reef_value, deflation_factor,
                              base_year=COASTAL_PROTECTION_BASE_YEAR):
    """Coral-reef protection value carried to the base year, one row per country.

    The coral table is keyed on country NAME, so it is joined to the correspondence on
    ee_r264_name and then de-duplicated on the country/year/value triple, which collapses the
    sub-region rows a split country's name matches. The surviving rows are multiplied by the
    shared US CPI factor and stamped with the base year. The values are USD of the table's own
    year, not local currency, so one factor carries every country.

    Args:
        gdf_countries (pd.DataFrame): the r264 country correspondence.
        df_coral_reef_value (pd.DataFrame): the coral table, with ee_r264_name, coral_reef_value
            and year.
        deflation_factor (float): the US CPI factor from the coral table's year to base_year
            (utilities.usd_deflation_factor).
        base_year (int): the year the deflated values are stamped with.

    Returns:
        pd.DataFrame: iso3_r250_label, year, coastal_protection_gep_coral_reef.
    """
    df = hb.df_merge(gdf_countries, df_coral_reef_value, how='inner', on='ee_r264_name')
    df = df.drop_duplicates(subset=['iso3_r250_label', 'year', 'coral_reef_value'])

    df['coral_reef_value'] = df['coral_reef_value'] * deflation_factor
    df['year'] = base_year
    # A value the table carries as missing stays missing: the country keeps a missing
    # component (min_count=1) instead of dropping out. The flag says whether every surviving
    # row carried a value.
    df = (df.groupby(['iso3_r250_label', 'year'], as_index=False, dropna=False)
            .agg(coral_reef_value=('coral_reef_value', lambda v: v.sum(min_count=1)),
                 coral_fully_measured=('coral_reef_value', lambda v: bool(v.notna().all()))))
    return df.rename(columns={'coral_reef_value': 'coastal_protection_gep_coral_reef'})


def coverage_label(mangrove_part, coral_part):
    """What a country's coastal-protection total covers, from its two component descriptions.

    Args:
        mangrove_part (str or None): 'mangrove', 'partial mangrove', or missing when the
            component carries no value.
        coral_part (str or None): 'coral', 'partial coral', or missing.

    Returns:
        str or None: '<a> and <b> protection' with both, '<a> protection only' with one, and
        None with neither, so the label is missing exactly where the total is.
    """
    present = [part for part in (mangrove_part, coral_part) if isinstance(part, str)]
    if len(present) == 2:
        return f'{present[0]} and {present[1]} protection'
    if len(present) == 1:
        return f'{present[0]} protection only'
    return None


def combine_coastal_components(df_mangrove, df_coral_reef, mangrove_excluded_labels=()):
    """The two components on one row per country and year, two totals and a coverage label.

    The join is outer because the two tables cover different countries. A component ABSENT
    from its table stays missing: the coral table lists 80 countries and leaves out some that
    have reefs (Japan among them), so absence from a table is not evidence of zero protection.
    A component a table carries as MISSING (area with no rate, or no area at all) stays missing
    too. A zero the source table reports (zero mapped mangrove area) is preserved. Those zeros
    are source-reported, not all independently measured: the source's processing writes zero
    area for countries absent from its analysis, and 17 of the 18 countries it removed for
    holding under 100 ha keep that zero while their interpretation is open (the 18th, Anguilla,
    is in neither table).

    A country on the mangrove exclusion list was removed from the source's analysis, so its
    mangrove component is missing whatever its row shows: the source's own processing writes
    zero area for countries absent from its analysis. Each listed country must carry no
    positive mangrove value, so a list written for another table version fails instead of
    blanking a measured value.

    coastal_protection_gep is the known-component total: the sum of what is measured, missing
    only where nothing is. COVERAGE_COLUMN names what that total covers ('mangrove and coral
    protection', 'coral protection only', 'mangrove protection only', with 'partial' before a
    component measured on only some of a split country's rows) and is missing where the total
    is. Mangroves and coral reefs are not the only habitats that protect coasts, so even both
    components are not all of nature's coastal protection: the label states which of the two
    modelled components the total covers, not that it is complete.
    coastal_protection_gep_both_components carries the total only where both components are
    fully measured.

    Args:
        df_mangrove (pd.DataFrame): iso3_r250_label, year, coastal_protection_gep_mangrove.
        df_coral_reef (pd.DataFrame): iso3_r250_label, year, coastal_protection_gep_coral_reef.
        mangrove_excluded_labels (iterable): iso3_r250 labels the mangrove source removed from
            its analysis.

    Returns:
        pd.DataFrame: both components, the known-component sum as coastal_protection_gep, the
        both-components sum as coastal_protection_gep_both_components, the coverage label, and
        a Value copy of the known-component sum for the shared reporting templates.

    Raises:
        ValueError: if a listed country carries a positive mangrove value.
    """
    m, c = 'coastal_protection_gep_mangrove', 'coastal_protection_gep_coral_reef'
    excluded = set(mangrove_excluded_labels)
    valued = df_mangrove.loc[df_mangrove['iso3_r250_label'].isin(excluded) & (df_mangrove[m] > 0),
                             'iso3_r250_label']
    if len(valued):
        raise ValueError('listed as excluded from the mangrove analysis but the table carries a '
                         f'positive value: {sorted(set(valued))} -- the list does not match this table')
    df = pd.merge(df_mangrove, df_coral_reef, how='outer', on=['iso3_r250_label', 'year'])
    df.loc[df['iso3_r250_label'].isin(excluded), m] = float('nan')
    df['coastal_protection_gep'] = df[[m, c]].sum(axis=1, min_count=1)
    # Each component described as measured, partly measured (some of a split country's rows
    # carried no value) or missing, then the two descriptions combined into the label.
    parts = {}
    for column, name, flag in ((m, 'mangrove', 'mangrove_fully_measured'),
                               (c, 'coral', 'coral_fully_measured')):
        # A component absent from its table has no flag; only an explicit False marks it partial.
        fully = df[flag].ne(False) if flag in df.columns else pd.Series(True, index=df.index)
        parts[name] = fully.map({True: name, False: 'partial ' + name}).where(df[column].notna())
    df[COVERAGE_COLUMN] = [coverage_label(a, b) for a, b in zip(parts['mangrove'], parts['coral'])]
    df['coastal_protection_gep_both_components'] = df['coastal_protection_gep'].where(
        df[COVERAGE_COLUMN] == BOTH_COMPONENTS_LABEL)
    df = df.drop(columns=['mangrove_fully_measured', 'coral_fully_measured'], errors='ignore')
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

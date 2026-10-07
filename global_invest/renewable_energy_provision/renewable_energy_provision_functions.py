# -*- coding: utf-8 -*-
"""Renewable-energy science: generation, priced, times nature's contribution share.

Nothing here opens a file. The task layer reads the three source tables (IRENA generation, the
World Bank price series, the CWoN resource rents), hands the frames in and writes back what it
gets, so every step below can be pinned on a hand-built input in the test suite.
"""
import numpy as np
import pandas as pd

# Subservice key -> the 'Group Technology' label IRENA gives that resource. Also the order the
# per-resource frames are built and concatenated in.
SUBSERVICE_TECHNOLOGIES = {
    'wind_energy_provision': 'Wind energy',
    'solar_energy_provision': 'Solar energy',
    'geothermal_energy_provision': 'Geothermal energy',
}
# The IRENA table splits each resource across sub-technologies and producer types; generation is
# summed over both, within a country, year and resource.
GENERATION_GROUP_COLUMNS = ['Year', 'ISO3 code', 'Country', 'Group Technology']
GENERATION_COLUMN = 'Electricity Generation (GWh)'
PRICE_COLUMN = 'Price (USD/GWh)'
# The World Bank price series arrives in US cents per kilowatt-hour, and generation is in
# gigawatt-hours: 1e6 kWh per GWh over 100 cents per USD leaves this factor.
CENTS_PER_KWH_TO_USD_PER_GWH = 10000.0
# What a valued row carries, in write order. The country column is renamed to the r250 label
# after the selection, so it stays first.
VALUED_COLUMNS = ['ISO3 code', 'Country', 'Year', 'Group Technology', PRICE_COLUMN,
                  GENERATION_COLUMN, 'nat_contrib', 'renewable_energy_provision_gep']


def generation_by_technology(df_generation):
    """One generation frame per valued resource, summed over sub-technology and producer type.

    Args:
        df_generation (pd.DataFrame): the IRENA table, one row per country, year, technology,
            sub-technology and producer type.

    Returns:
        list: a frame per entry of SUBSERVICE_TECHNOLOGIES, in that order. Fossil and other
        non-valued technologies are dropped by keeping only those labels.
    """
    aggregated = df_generation.groupby(GENERATION_GROUP_COLUMNS,
                                       as_index=False)[GENERATION_COLUMN].sum()
    return [aggregated[aggregated['Group Technology'] == technology]
            for technology in SUBSERVICE_TECHNOLOGIES.values()]


def price_in_usd_per_gwh(df_price):
    """The World Bank price series converted to USD per GWh, on the column names the join uses.

    Args:
        df_price (pd.DataFrame): Economy ISO3, Economy Name, Year and Price in US cents per kWh.

    Returns:
        pd.DataFrame: the same rows with the price converted and the country columns renamed to
        the labels the generation frames carry.
    """
    df = df_price.copy()
    df['Price'] = df['Price'] * CENTS_PER_KWH_TO_USD_PER_GWH
    return df.rename(columns={'Economy ISO3': 'ISO3 code', 'Economy Name': 'Country',
                              'Price': PRICE_COLUMN})


def merge_price_onto_generation(df_price, generation_frames):
    """Each resource's generation frame carrying that country-year's electricity price.

    The join is inner on country and year, so a country-year the price series does not cover
    contributes no valued row. Both sides carry a Country column; the price table's copy is the
    one dropped, which is why the join is on the ISO3 code rather than on the name.

    Args:
        df_price (pd.DataFrame): the priced table from price_in_usd_per_gwh.
        generation_frames (list): the per-resource frames from generation_by_technology.

    Returns:
        list: one merged frame per input frame, in the same order.
    """
    merge_columns = ['ISO3 code', 'Year']
    merged_frames = []
    for df_generation in generation_frames:
        merged = pd.merge(df_price, df_generation, on=merge_columns, how='inner')
        merged = merged.drop('Country_y', axis=1)
        merged = merged.rename(columns={'Country_x': 'Country'})
        merged_frames.append(merged)
    return merged_frames


def renewable_energy_gep(nature_contribution, price_usd_per_gwh, generation_gwh):
    """Renewable provision value: generation, priced, times nature's contribution share.

    Args:
        nature_contribution: the resource-rent share attributable to nature.
        price_usd_per_gwh: electricity price in USD per gigawatt-hour.
        generation_gwh: electricity generated, gigawatt-hours.

    Returns:
        The valued generation, USD.
    """
    return nature_contribution * price_usd_per_gwh * generation_gwh


def regional_attribution_fill(df_attribution, country_regions):
    """A rent share for the countries the attribution table does not carry, from their region.

    The fill is the unweighted mean of the sub-region's available shares that year, falling back
    to the region and then the world. A country with a MEASURED non-positive share is not
    touched: negative rent is an answer, not a gap.

    Args:
        df_attribution (pd.DataFrame): Country, Year, nat_contrib.
        country_regions (pd.DataFrame): Country, Sub-region and Region, one row per country.

    Returns:
        pd.DataFrame: Country, Year, nat_contrib_filled -- a share for every country-region pair
        and year the attribution table's years cover.
    """
    att = df_attribution.merge(country_regions.drop_duplicates('Country'), on='Country', how='left')
    by_sub = att.groupby(['Sub-region', 'Year'])['nat_contrib'].mean().rename('sub_mean')
    by_region = att.groupby(['Region', 'Year'])['nat_contrib'].mean().rename('region_mean')
    by_world = att.groupby('Year')['nat_contrib'].mean().rename('world_mean')
    frame = country_regions.drop_duplicates('Country').merge(
        pd.DataFrame({'Year': sorted(df_attribution['Year'].unique())}), how='cross')
    frame = (frame.merge(by_sub, on=['Sub-region', 'Year'], how='left')
                  .merge(by_region, on=['Region', 'Year'], how='left')
                  .merge(by_world, on='Year', how='left'))
    frame['nat_contrib_filled'] = (frame['sub_mean']
                                   .fillna(frame['region_mean'])
                                   .fillna(frame['world_mean']))
    return frame[['Country', 'Year', 'nat_contrib_filled']]


def valued_generation(priced_frames, df_attribution, country_regions=None):
    """The priced resource frames stacked and valued at each country-year's rent share.

    The join onto the resource-rent table is on the country NAME, which the two sources spell
    alike, plus the year. With `country_regions`, a country the attribution table does not carry
    takes its region's mean share instead of silently dropping out -- the 71 solar and 50 wind
    generators with no rent row are real generators, and a join is not a valuation decision.

    Args:
        priced_frames (list): the merged frames from merge_price_onto_generation.
        df_attribution (pd.DataFrame): Country, Year and nat_contrib, the share of the resource
            rent attributable to nature.
        country_regions (pd.DataFrame): Country, Sub-region, Region -- enables the regional fill.
            None preserves the inner-join behaviour.

    Returns:
        pd.DataFrame: every valued row, with renewable_energy_provision_gep added and, when the
        fill is active, `attribution_source` saying which share each row used.
    """
    combined = pd.concat(priced_frames, ignore_index=True)
    if country_regions is None:
        df = combined.merge(df_attribution, on=['Country', 'Year'], how='inner')
    else:
        df = combined.merge(df_attribution, on=['Country', 'Year'], how='left')
        fill = regional_attribution_fill(df_attribution, country_regions)
        df = df.merge(fill, on=['Country', 'Year'], how='left')
        df['attribution_source'] = np.where(df['nat_contrib'].notna(), 'country', 'regional mean')
        df['nat_contrib'] = df['nat_contrib'].fillna(df['nat_contrib_filled'])
        df = df.drop(columns=['nat_contrib_filled'])
        df = df[df['nat_contrib'].notna()]
    df['renewable_energy_provision_gep'] = renewable_energy_gep(
        df['nat_contrib'], df[PRICE_COLUMN], df[GENERATION_COLUMN])
    return df


def base_year_valued_rows(df_valued, base_year):
    """The base year's valued rows, keyed by the r250 country label.

    Rows valued at zero or below are dropped. A non-positive value comes out of the resource-rent
    attribution rather than out of generation, so it is not a country that generated nothing.

    Args:
        df_valued (pd.DataFrame): the frame from valued_generation.
        base_year (int): the year the account values.

    Returns:
        pd.DataFrame: VALUED_COLUMNS, with ISO3 code renamed to iso3_r250_label.
    """
    df = df_valued[VALUED_COLUMNS]
    df = df.loc[df['Year'] == base_year]
    df = df.loc[df['renewable_energy_provision_gep'] > 0]
    return df.rename(columns={'ISO3 code': 'iso3_r250_label'})


def split_by_resource(df):
    """The valued rows as one frame per resource, keyed by the IRENA Group Technology label.

    The columns are renamed to the subservice tables' own vocabulary on the way out, so the
    per-resource CSVs read as standalone outputs rather than as slices of the parent table.
    """
    df_filtered = df.rename(columns={
        'Country': 'Country_Name',
        'Group Technology': 'Resource',
        PRICE_COLUMN: 'P_electricity_USD_per_GWh',
        GENERATION_COLUMN: 'energy_prod_GWh',
    })
    return {resource: df_resource.copy()
            for resource, df_resource in df_filtered.groupby('Resource')}

"""Water-supply science. First component: hydropower direct use, CWoN resource-rent method.

The consortium drive's Hydropower folder carries the committed 2019 output
(gep_hydro_directuse_CWONresrent_20260720.csv) but not the script behind it; the method was
identified by reverse-engineering against that anchor and is exact across all 94 valued
countries (ratio constant to 1e-10): the 2019 GEP is CWoN 2024's capitalized hydropower
wealth divided by the annuity factor at the CWoN-standard 4 percent CAPITALIZATION rate over a
100-year horizon — the constant annual rent the capitalized value implies. That rate is CWoN's
and is not the account's 2 percent social discount rate; inverting their capitalization with any
other rate returns a rent they never capitalized. The test suite asserts both
the annuity identity and the exact replication. An earlier direct-use script on the drive
(price x quantity from EIA/World Bank series, Nov 2024) is superseded by this method and not
ported.

The other water_supply components on the group's sheet (agriculture, household) are not yet
built anywhere we have seen; the module is laid out to receive them beside hydropower.
"""
import numpy as np
import pandas as pd

# The capitalization rate is `water_supply_hydropower_capitalization_rate` in es_parameters and is
# NOT duplicated here: a module constant beside a CSV row wins over it silently, which is what H19
# exists to catch and did catch when this was written both ways.
#
# ⚠ It is not the account's social discount rate, which is 2 percent with sensitivity at 1 and 3.
# It is the rate CWoN CAPITALIZED hydropower wealth with, and this module runs that capitalization
# backwards to recover the annual rent inside it, so inverting with any other rate returns a rent
# CWoN never capitalized. The two share a word and nothing else.
HYDROPOWER_HORIZON_YEARS = 100    # the CWoN capitalization horizon
HYDROPOWER_GEP_YEAR = 2019
# Countries the reference output leaves EMPTY although the CWoN wealth table values them.
# No available material explains the drop (the 2026 script is not on the drive; the package
# lacks its raw workbook), and the reason is an open ask on the deck.
#
# These are NOT dropped from the reported value. Blanking them would fit our number to the
# reference instead of deriving it, which is the one thing the account must not do: a
# discrepancy we silently reproduce can never surface. The module reports every country the
# CWoN table values, and emits the reference-matching variant beside it as the comparison
# anchor, so the replication check still runs and the gap stays visible and countable.
HYDROPOWER_REFERENCE_EXCLUDED = ('AZE', 'BGR', 'DOM', 'GRC', 'GTM', 'HTI', 'KAZ', 'LBR',
                                 'MAR', 'MKD', 'NIC', 'POL', 'PRT', 'SLV', 'TJK', 'UKR', 'ZAF')


def annuity_factor(rate, years=HYDROPOWER_HORIZON_YEARS):
    """Present value of one dollar per year for the horizon: sum of 1/(1+r)^t, t = 1..years."""
    return float(np.sum(1.0 / (1.0 + rate) ** np.arange(1, years + 1)))


def hydropower_rent_from_wealth(wealth_df, capitalization_rate, year=HYDROPOWER_GEP_YEAR):
    """CWoN capitalized hydropower wealth -> the implied constant annual rent per country.

    wealth_df: the CWoN hydro_wealth_cd table (countrycode + YR<year> columns, current USD --
    equal to real 2019 USD in the 2019 base year, and the only variant that covers Venezuela).
    Countries without wealth stay NaN: no data is not zero rent.

    Returns hydropower_gep, every country the wealth table values, and beside it
    hydropower_gep_reference_variant, the same figure with the reference's unexplained
    exclusions blanked. The first is what the account reports; the second exists only so the
    replication check has something to compare against."""
    df = wealth_df[['countrycode', f'YR{year}']].rename(
        columns={'countrycode': 'iso3_r250_label',
                 f'YR{year}': 'hydropower_wealth_usd'})
    df['hydropower_gep'] = df['hydropower_wealth_usd'] / annuity_factor(capitalization_rate)
    df['hydropower_gep_reference_variant'] = df['hydropower_gep'].mask(
        df['iso3_r250_label'].isin(HYDROPOWER_REFERENCE_EXCLUDED))
    return df


def water_supply_gep_by_country(hydropower_df, countries_df):
    """Join the hydropower rent onto the r250 country list by iso3 label, one row per country."""
    columns = ['iso3_r250_label', 'hydropower_gep']
    if 'hydropower_gep_reference_variant' in hydropower_df.columns:
        columns.append('hydropower_gep_reference_variant')
    return countries_df.merge(hydropower_df[columns], on='iso3_r250_label', how='left')


# =============================================================================
# Water-use chain (agriculture / industrial / municipal), ported from the
# drive's two-script R chain: script 01 cleans the raw AQUASTAT export into the
# four SDG 6.4.1 efficiency series (US$/m3, wide by country-year); script 02
# joins iso codes on from the withdrawal table's names, merges the two, and
# computes sector GEP = efficiency x withdrawal volume at the survey years.
# Both of the chain's outputs are committed on the drive and replicate: the
# cleaned table bit-exactly, the country-year GEP table to the anchor CSV's own
# 15-significant-digit rounding (max 3e-15 relative).
#
# The drive's PER-COUNTRY outputs are NOT this chain's outputs, and the library
# does not adopt them. The components it reports are computed here from the raw
# AQUASTAT export, by water_use_components_from_chain over script 02's
# country-year table (water_supply_tasks.water_use_components). The committed
# tables stay as comparison anchors, logged beside our values on every run and
# pinned in the test suite. The agriculture table equals 1000 x GEP_total from
# the crop-water valuation (share_20260618/gep_agwater_country_source.csv, a
# separate chain whose scripts are not on the drive), and the all-sector table
# was computed on a newer AQUASTAT vintage with the appendix's deflate-to-2015
# step -- only 3/183 countries reproduce from the staged inputs (median
# committed/computed-at-2019 ratio 0.36, range 0 to 1.25).
# =============================================================================
WATER_USE_YEARS = (2000, 2005, 2010, 2015)   # the AQUASTAT survey years the chain keeps
# The four AQUASTAT series script 01 keeps, renamed to the chain's column names.
WATER_USE_EFFICIENCY_SERIES = {
    'SDG 6.4.1. Water Use Efficiency': 'wue_general_usdpm3',
    'SDG 6.4.1. Irrigated Agriculture Water Use Efficiency': 'wue_irrigation_usdpm3',
    'SDG 6.4.1. Industrial Water Use Efficiency': 'wue_industrial_usdpm3',
    'SDG 6.4.1. Services Water Use Efficiency': 'wue_municipal_usdpm3',
}
# Script 02's name surgery, verbatim: three renames onto the withdrawal table's spellings,
# then iso codes for the names its country map does not carry. The lookup countries keep
# their AQUASTAT names, so their efficiency and withdrawal rows share an iso code but never
# a country name -- the merge leaves them as half-rows with no products (USA, RUS, TUR...).
# The anchor commits that behavior and the port reproduces it.
WATER_USE_COUNTRY_RENAMES = {
    'Cabo Verde': 'Cape Verde',
    "Côte d'Ivoire": "Cote d'Ivoire",
    'Timor-Leste': 'East Timor',
}
WATER_USE_ISO_LOOKUP = {
    'Bolivia (Plurinational State of)': 'BOL',
    'Brunei Darussalam': 'BRN',
    "Democratic People's Republic of Korea": 'PRK',
    'Democratic Republic of the Congo': 'COD',
    'Iran (Islamic Republic of)': 'IRN',
    "Lao People's Democratic Republic": 'LAO',
    'Liechtenstein': 'LIE',
    'Netherlands (Kingdom of the)': 'NLD',
    'Republic of Korea': 'KOR',
    'Republic of Moldova': 'MDA',
    'Russian Federation': 'RUS',
    'Sao Tome and Principe': 'STP',
    'Syrian Arab Republic': 'SYR',
    'Türkiye': 'TUR',
    'United Kingdom of Great Britain and Northern Ireland': 'GBR',
    'United Republic of Tanzania': 'TZA',
    'United States of America': 'USA',
    'Venezuela (Bolivarian Republic of)': 'VEN',
    'Viet Nam': 'VNM',
    # The spellings WATER_USE_COUNTRY_RENAMES produces. That map exists to make the efficiency and
    # withdrawal tables agree with each other, and the names it settles on are not the ones the
    # r250 list uses, so these six resolved to nothing until they were named here too.
    'Cape Verde': 'CPV',
    'Congo': 'COG',
    "Cote d'Ivoire": 'CIV',
    'Democratic Republic of Congo': 'COD',
    'East Timor': 'TLS',
    'Eswatini': 'SWZ',
}
WATER_USE_REGIONS_DROPPED = (
    'Australia and New Zealand', 'Central Asia', 'Central and Southern Asia', 'Eastern Asia',
    'Eastern and South-Eastern Asia', 'Europe', 'Europe and Northern America',
    'Land Locked Developing Countries', 'Latin America and the Caribbean',
    'Least Developed Countries', 'Northern Africa', 'Northern Africa and Western Asia',
    'Northern America', 'Oceania', 'Oceania (excluding Australia and New Zealand)',
    'Small Island Developing States', 'South-eastern Asia', 'Southern Asia',
    'Sub-Saharan Africa', 'Western Asia', 'World',
)


def clean_aquastat_water_efficiency(raw_df):
    """Script 01: the raw AQUASTAT export -> the wide country-year efficiency table.

    Args:
        raw_df (pd.DataFrame): the AQUASTAT dissemination export (Variable, Area, Year,
            Value, Unit columns).

    Returns:
        pd.DataFrame: country, year, and the four wue_*_usdpm3 columns -- the US$/m3 rows of
        the four SDG 6.4.1 series pivoted wide.
    """
    df = raw_df[(raw_df['Unit'] == 'US$/m3')
                & raw_df['Variable'].isin(WATER_USE_EFFICIENCY_SERIES)]
    wide = (df.pivot_table(index=['Area', 'Year'], columns='Variable', values='Value',
                           aggfunc='first')
              .rename(columns=WATER_USE_EFFICIENCY_SERIES)
              .rename_axis(index=['country', 'year'], columns=None)
              .reset_index())
    return wide[['country', 'year', 'wue_industrial_usdpm3', 'wue_irrigation_usdpm3',
                 'wue_municipal_usdpm3', 'wue_general_usdpm3']]


def water_use_gep_by_country_year(efficiency_df, withdrawal_df):
    """Script 02: sector GEP per country-year = efficiency (US$/m3) x withdrawal volume (m3).

    Args:
        efficiency_df (pd.DataFrame): clean_aquastat_water_efficiency's output.
        withdrawal_df (pd.DataFrame): the withdrawal table (country, iso_code, year,
            w_agriculture, w_industry, w_munucipal -- the source's spelling, kept).

    Returns:
        pd.DataFrame: the outer merge at the survey years with the three gep_water_* product
        columns, in the anchor's column order.
    """
    wue = efficiency_df.copy()
    wue['country'] = wue['country'].replace(WATER_USE_COUNTRY_RENAMES)
    country_map = withdrawal_df[['iso_code', 'country']].drop_duplicates()
    wue = wue.merge(country_map, on='country', how='left')
    wue['iso_code'] = wue['country'].map(WATER_USE_ISO_LOOKUP).fillna(wue['iso_code'])
    wue = wue[~wue['country'].isin(WATER_USE_REGIONS_DROPPED)]
    df = wue.merge(withdrawal_df, on=['iso_code', 'country', 'year'], how='outer')
    df = df[df['year'].isin(WATER_USE_YEARS)]
    df['gep_water_agricultural'] = df['wue_irrigation_usdpm3'] * df['w_agriculture']
    df['gep_water_industrial'] = df['wue_industrial_usdpm3'] * df['w_industry']
    df['gep_water_municipal'] = df['wue_municipal_usdpm3'] * df['w_munucipal']
    return df[['iso_code', 'country', 'year', 'wue_industrial_usdpm3', 'wue_irrigation_usdpm3',
               'wue_municipal_usdpm3', 'wue_general_usdpm3', 'w_agriculture', 'w_industry',
               'w_munucipal', 'gep_water_agricultural', 'gep_water_industrial',
               'gep_water_municipal']]


def water_use_components_by_country(agriculture_df, all_sector_df, countries_df):
    """The two committed outputs joined onto the r250 country list by label: the agriculture
    component (the sheet's agriculture subgroup) and the all-sector total."""
    df = countries_df.merge(
        agriculture_df[['iso3_r250_label', 'wateruse_ag_gep']].rename(
            columns={'wateruse_ag_gep': 'water_use_agriculture_gep'}),
        on='iso3_r250_label', how='left')
    df = df.merge(
        all_sector_df[['iso3_r250_label', 'wateruse_gep']].rename(
            columns={'wateruse_gep': 'water_use_all_sector_gep'}),
        on='iso3_r250_label', how='left')
    return df


def water_use_components_from_chain(gep_by_country_year_df, countries_df):
    """One row per country from OUR calculation: the latest survey year's agriculture value and
    all-sector sum, keyed to r250 by exact country-name match (iso3_r250_name, then
    name_long). Unmatched calculation countries keep an empty iso3 so a name drift is visible
    instead of silently dropped."""
    # One row, one year. `groupby.last()` returns the last NON-NULL value per column
    # independently, not the last row, so a country reporting only municipal water in 2015 took
    # its industrial and agricultural values from 2010 and the whole row was stamped 2015.
    # Slovakia is the clean example: agriculture from 2015, industry and municipal from 2010,
    # summed as if one year. No country's published figure matched any single year of its own
    # chain, and the all-sector total sat between 2010's $26.9T and 2015's $14.9T because it was
    # neither. The docstring already said "the latest survey year", which is what this now does.
    #
    # The row kept is the year reporting the most sectors, ties going to the later year. Latest-
    # year-outright would throw away a complete 2010 for a 2015 carrying one sector; complete-
    # years-only would drop a country like Norway, which reports no agricultural water use in any
    # year. Both of those trade a true year for a worse number, and this trades neither.
    sector_cols = ['gep_water_agricultural', 'gep_water_industrial', 'gep_water_municipal']
    sectors_present = gep_by_country_year_df[sector_cols].notna().sum(axis=1)
    ordered = (gep_by_country_year_df.assign(_sectors=sectors_present)
               .sort_values(['_sectors', 'year']))
    latest = ordered.groupby('country', as_index=False).tail(1).drop(columns='_sectors')
    latest['water_use_agriculture_gep'] = latest['gep_water_agricultural']
    latest['water_use_all_sector_gep'] = latest[sector_cols].sum(axis=1, min_count=1)
    # The two the account reports, matching the authors' split of 2026-08-31: irrigation, and
    # domestic covering industrial, commercial and residential. Published as their own columns
    # because the table used to carry agriculture and the total, so domestic existed only as a
    # subtraction -- and only worked because there happen to be three sectors and one of them was
    # published. A reader wanting industry apart from residential could not have got it at all.
    latest['water_use_irrigation_gep'] = latest['gep_water_agricultural']
    latest['water_use_domestic_gep'] = latest[
        ['gep_water_industrial', 'gep_water_municipal']].sum(axis=1, min_count=1)

    by_name = countries_df.drop_duplicates('iso3_r250_name').set_index('iso3_r250_name')['iso3_r250_label']
    by_long = (countries_df.drop_duplicates('name_long').set_index('name_long')['iso3_r250_label']
               if 'name_long' in countries_df.columns else by_name.iloc[0:0])
    # Exact name, then the long name, then the AQUASTAT spellings the module already lists. That
    # last fallback existed for the efficiency table and not for this one, so 19 countries the
    # lookup names -- the United States, Turkiye, Viet Nam, the United Kingdom, Iran, Tanzania,
    # Cote d'Ivoire among them -- resolved to nothing and dropped out of the account with their
    # water use, Cote d'Ivoire alone carrying $77.5M of agricultural value that reached no country.
    latest['iso3_r250_label'] = (latest['country'].map(by_name)
                                 .fillna(latest['country'].map(by_long))
                                 .fillna(latest['country'].map(WATER_USE_ISO_LOOKUP)))
    latest = latest.merge(countries_df[['iso3_r250_label', 'iso3_r250_id']].drop_duplicates('iso3_r250_label'),
                          on='iso3_r250_label', how='left')
    return one_row_per_country(latest[['country', 'iso3_r250_id', 'iso3_r250_label', 'year',
                                       'water_use_agriculture_gep', 'water_use_irrigation_gep',
                                       'water_use_domestic_gep', 'water_use_all_sector_gep']])


# The AQUASTAT export names some countries twice, once in full and once short ("Russian
# Federation" and "Russia", "Republic of Korea" and "South Korea"), and both spellings can
# resolve to the same r250 id. Left-merging that table onto the country list then FANS OUT:
# the country gets two rows and is counted twice in every total. Korea and Russia did exactly
# that, inflating the reported hydropower total by 1.89bn USD, which stayed invisible because
# the deck quoted the reference file's total rather than the module's own.
WATER_USE_VALUE_COLUMNS = ('water_use_agriculture_gep', 'water_use_irrigation_gep',
                           'water_use_domestic_gep', 'water_use_all_sector_gep')


def one_row_per_country(df_components):
    """The components table collapsed to one row per r250 country.

    Rows the name join could not resolve keep their empty id and pass through unchanged, so a
    name drift stays visible. Where two spellings resolved to the same country, their values
    are combined by taking the one non-empty value per column.

    Raises:
        ValueError: if two rows for the same country carry DIFFERENT values for a column.
            That is a genuine conflict about what the country's value is, and picking one
            silently would be the same class of error as the double-count this prevents.
    """
    resolved = df_components[df_components['iso3_r250_id'].notna()]
    for column in WATER_USE_VALUE_COLUMNS:
        distinct = resolved.groupby('iso3_r250_id')[column].nunique(dropna=True)
        conflicted = distinct[distinct > 1]
        if len(conflicted):
            raise ValueError(
                f"water_use components disagree on {column} for r250 id(s) "
                f"{sorted(conflicted.index.tolist())}: two spellings of the same country carry "
                f"different values, so which one the account reports is undecided.")

    unresolved = df_components[df_components['iso3_r250_id'].isna()]
    # groupby.last skips nulls, so a column empty on one spelling takes the other's value.
    collapsed = (resolved.sort_values('year')
                 .groupby('iso3_r250_id', as_index=False)
                 .agg({c: 'last' for c in df_components.columns if c != 'iso3_r250_id'}))
    return pd.concat([collapsed, unresolved], ignore_index=True)[list(df_components.columns)]

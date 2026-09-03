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

# The capitalization rate is `water_supply_hydropower_capitalization_rate` in es_parameters. It is
# CWoN's, not the account's discount rate; the method page says why.
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
            columns={'wateruse_ag_gep': 'water_use_agriculture_value_added'}),
        on='iso3_r250_label', how='left')
    df = df.merge(
        all_sector_df[['iso3_r250_label', 'wateruse_gep']].rename(
            columns={'wateruse_gep': 'water_use_all_sector_value_added'}),
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
    latest['water_use_agriculture_value_added'] = latest['gep_water_agricultural']
    latest['water_use_all_sector_value_added'] = latest[sector_cols].sum(axis=1, min_count=1)
    # The two the account reports, matching the authors' split of 2026-08-31: irrigation, and
    # domestic covering industrial, commercial and residential. Published as their own columns
    # because the table used to carry agriculture and the total, so domestic existed only as a
    # subtraction -- and only worked because there happen to be three sectors and one of them was
    # published. A reader wanting industry apart from residential could not have got it at all.
    #
    # ⚠ These are VALUE ADDED, not a value of water. SDG 6.4.1 is defined as value added over the
    # volume withdrawn, so multiplying the indicator back by the withdrawal returns the value added
    # it was built from. They are named for what they are; the account's figure is a share of them.
    latest['water_use_irrigation_value_added'] = latest['gep_water_agricultural']
    latest['water_use_domestic_value_added'] = latest[
        ['gep_water_industrial', 'gep_water_municipal']].sum(axis=1, min_count=1)
    return _with_country_labels(latest, countries_df)


# The AQUASTAT variables the premium reads, and FAOSTAT's cropland item.
AQUASTAT_IRRIGATED_AREA_CODE = 4313        # Area equipped for irrigation: total, 1000 ha
AQUASTAT_AG_VALUE_ADDED_CODE = 4548        # Agriculture, value added to GDP, current US$
AQUASTAT_IRRIGATED_GVA_SHARE_CODE = 4555   # % of agricultural GVA produced by irrigated agriculture
FAOSTAT_CROPLAND_ITEM_CODE = 6620          # Cropland, element Area, 1000 ha


def cropland_area_from_faostat(land_use_df):
    """FAOSTAT Land Use, long, reduced to the cropland area per country and year.

    Returns m49, Year and cropland_1000ha -- the unit AQUASTAT reports irrigated area in, so the
    two need no conversion between them. M49 is the join key rather than the country name.
    """
    df = land_use_df[(land_use_df['Item Code'] == FAOSTAT_CROPLAND_ITEM_CODE)
                     & (land_use_df['Element'] == 'Area')].copy()
    df['m49'] = df['Area Code (M49)'].astype(str).str.lstrip("'").astype(int)
    return (df.rename(columns={'Value': 'cropland_1000ha'})[['m49', 'Year', 'cropland_1000ha']]
            .dropna(subset=['cropland_1000ha']))


def irrigation_premium_by_country(aquastat_df, cropland_df, year):
    """What an irrigated hectare earns above the same land rainfed, per country.

    This is the quantity that is actually water. A rent is what is left of value added after
    labour and capital are paid, and in crop agriculture that residual is attributed to LAND --
    which rainfed cropland earns too, from soil, climate and terrain. Only the DIFFERENCE is
    attributable to irrigation, so only the difference is carried forward.
    Args:
        aquastat_df (pd.DataFrame): the staged AQUASTAT pull, long, with VariableCode, m49, Year,
            Value.
        cropland_df (pd.DataFrame): FAOSTAT Land Use, with the cropland area in 1000 ha, m49 and
            Year.
        year (int): the account's base year.

    Returns:
        pd.DataFrame: m49, irrigated_area_ha, premium_usd_per_ha and irrigation_premium_usd, one
        row per country that reports every piece at that year.
    """
    wide = (aquastat_df[aquastat_df['Year'] == year]
            .pivot_table(index=['m49'], columns='VariableCode', values='Value', aggfunc='first')
            .reset_index())
    wide.columns = [str(c) for c in wide.columns]
    wide = wide.rename(columns={str(AQUASTAT_IRRIGATED_AREA_CODE): 'irrigated_1000ha',
                                str(AQUASTAT_AG_VALUE_ADDED_CODE): 'ag_value_added_usd',
                                str(AQUASTAT_IRRIGATED_GVA_SHARE_CODE): 'irrigated_gva_percent'})
    needed = ['irrigated_1000ha', 'ag_value_added_usd', 'irrigated_gva_percent']
    missing = [c for c in needed if c not in wide.columns]
    if missing:
        raise ValueError('the AQUASTAT pull has no %s at %d, so the premium cannot be formed'
                         % (', '.join(missing), year))
    wide = wide.dropna(subset=needed)

    cropland = cropland_df[cropland_df['Year'] == year][['m49', 'cropland_1000ha']]
    df = wide.merge(cropland, on='m49', how='inner')
    # A country whose irrigated area is not smaller than its cropland has no rainfed side to
    # compare against, and a share outside [0, 100] is not a share.
    df = df[(df['cropland_1000ha'] > df['irrigated_1000ha'])
            & df['irrigated_gva_percent'].between(0, 100)].copy()

    df['irrigated_area_ha'] = df['irrigated_1000ha'] * 1000.0
    rainfed_area_ha = (df['cropland_1000ha'] - df['irrigated_1000ha']) * 1000.0
    irrigated_va = df['ag_value_added_usd'] * df['irrigated_gva_percent'] / 100.0
    rainfed_va = df['ag_value_added_usd'] * (1 - df['irrigated_gva_percent'] / 100.0)
    df['premium_usd_per_ha'] = (irrigated_va / df['irrigated_area_ha']
                                - rainfed_va / rainfed_area_ha)
    df['irrigation_premium_usd'] = df['premium_usd_per_ha'] * df['irrigated_area_ha']
    return df[['m49', 'irrigated_area_ha', 'premium_usd_per_ha', 'irrigation_premium_usd']]


def apply_water_share_of_value_added(df, water_share):
    """Turn the value added of water-using sectors into a value of water.

    The share is of sectoral value added, as aquaculture's is of fishing revenue and forestry's
    of forest revenue.

    ⚠ `water_share` has no default. When it is None the GEP columns are absent rather than zero,
    so a run without a chosen share publishes the value added and nothing that reads as an
    account figure.

    Args:
        df (pd.DataFrame): carries `water_use_irrigation_value_added` and
            `water_use_domestic_value_added`.
        water_share (float): the share of sectoral value added attributable to water, or None.

    Returns:
        pd.DataFrame: with `water_use_irrigation_gep` and `water_use_domestic_gep` added when a
        share is given, and untouched when it is not.
    """
    out = df.copy()
    if water_share is None:
        return out
    share = float(water_share)
    if not 0.0 <= share <= 1.0:
        raise ValueError('water_use_water_share_of_value_added is %r; a share of value added has '
                         'to sit in [0, 1]' % water_share)
    out['water_use_irrigation_gep'] = out['water_use_irrigation_value_added'] * share
    out['water_use_domestic_gep'] = out['water_use_domestic_value_added'] * share
    return out


def _with_country_labels(latest, countries_df):
    """The name-matching tail of water_use_components_from_chain, unchanged."""

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
                                       'water_use_agriculture_value_added',
                                       'water_use_irrigation_value_added',
                                       'water_use_domestic_value_added',
                                       'water_use_all_sector_value_added']])


# The AQUASTAT export names some countries twice, once in full and once short ("Russian
# Federation" and "Russia", "Republic of Korea" and "South Korea"), and both spellings can
# resolve to the same r250 id. Left-merging that table onto the country list then FANS OUT:
# the country gets two rows and is counted twice in every total. Korea and Russia did exactly
# that, inflating the reported hydropower total by 1.89bn USD, which stayed invisible because
# the deck quoted the reference file's total rather than the module's own.
# ⚠ The two `_value_added` columns are the denominators; the two `_gep` columns appear only once
# a water share is set. Both are listed because one_row_per_country has to carry whichever exist.
WATER_USE_VALUE_COLUMNS = ('water_use_agriculture_value_added', 'water_use_all_sector_value_added',
                           'water_use_irrigation_value_added', 'water_use_domestic_value_added',
                           'water_use_irrigation_gep', 'water_use_domestic_gep')


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
    # ⚠ Only the columns actually present. The two `_gep` columns exist only when a water share
    # is set, so a fixed list would make the no-share case -- the default -- raise.
    for column in [c for c in WATER_USE_VALUE_COLUMNS if c in df_components.columns]:
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

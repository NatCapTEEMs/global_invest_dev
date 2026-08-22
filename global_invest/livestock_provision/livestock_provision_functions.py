# -*- coding: utf-8 -*-
"""Livestock-provision science: FAOSTAT gross production value x the CWoN land rental rate.

The valuation runs the same chain crop_provision does, on the livestock items of the same
FAOSTAT Value of Production file: gross production value per country, item and year, attributed
to land by the Changing Wealth of Nations 2024 rental rate, which varies by decade and is applied
by a backward as-of merge on year.

Two differences from crop_provision are deliberate and load-bearing. The item selection is by FAO
item CODE rather than name (the service owner's convention, robust to FAO renaming items), and
the values are NOT converted out of FAOSTAT's thousand USD, so this service's totals are in
thousand USD where every other service reports plain USD. That second one is an open item flagged
in the tracker, not a decision: see attach_countries.

Step two of the port, feed_lambda_by_country, computes the ecosystem-provided share of livestock
feed from GLEAM 3 and is not yet wired into the task chain.

Every function here is a pure transformation over frames, which is what the tests exercise. The
task module reads the FAOSTAT bulk file and the rental-rate table and passes the frames in.
"""
import logging

import pandas as pd
import hazelbean as hb

from global_invest import utilities

# FAOSTAT's Value of Production table stacks several elements and units in one file. The
# valuation reads gross production value in current USD, which the file reports as element 57
# with unit "1000 USD".
FAOSTAT_VALUE_UNIT = '1000 USD'
# FAOSTAT ships value in thousand USD; the library reports plain USD, as crop_provision does.
FAOSTAT_THOUSAND_USD = 1000.0
FAOSTAT_GROSS_PRODUCTION_VALUE_ELEMENT = 57
# The bulk file's year columns run Y1961 to Y2022, each shadowed by a Y<year>F data-quality flag.
FAOSTAT_FIRST_YEAR = 1961
FAOSTAT_LAST_YEAR = 2022
# FAOSTAT area 223 is Turkiye, which recent releases spell several ways. The country join runs on
# the M49 code, so the name is normalised only to keep the item-level table readable.
FAOSTAT_TURKIYE_AREA_CODE = 223

# FAOSTAT mixes country rows with regional and thematic aggregates under the same Area column, and
# an aggregate row would be counted on top of its members. These are the aggregate and
# no-longer-existing areas the valuation drops. The three China sub-entities are dropped because
# the file also carries the "China" total, which is what the country correspondence expects.
FAOSTAT_AGGREGATE_AREAS = [
    "USSR",
    "Yugoslav SFR",
    "World",
    "Africa",
    "Eastern Africa",
    "Middle Africa",
    "Northern Africa",
    "Southern Africa",
    "Western Africa",
    "Americas",
    "Northern America",
    "Central America",
    "Caribbean",
    "South America",
    "Asia",
    "Central Asia",
    "Eastern Asia",
    "Southern Asia",
    "South-eastern Asia",
    "Western Asia",
    "Europe",
    "Eastern Europe",
    "Northern Europe",
    "Southern Europe",
    "Western Europe",
    "Oceania",
    "Australia and New Zealand",
    "Melanesia",
    "Micronesia",
    "Polynesia",
    "European Union (27)",
    "Least Developed Countries",
    "Land Locked Developing Countries",
    "Low Income Food Deficit Countries",
    "Small Island Developing States",
    "Czechoslovakia" "Low Income Food Deficit Countries",
    "Net Food Importing Developing Countries",
    "China, Hong Kong SAR",
    "China, mainland",
    "China, Macao SAR",
    "China, Taiwan Province of",
    "Belgium-Luxembourg",
]

# FAOSTAT keeps dissolved states under their own M49 codes. Each maps to the successor the
# country correspondence uses, so their production joins to a country instead of dropping.
M49_SUCCESSORS = {
    159: 156,   # China (mainland) -> China
    891: 688,   # Serbia and Montenegro -> Serbia
    200: 203,   # Czechoslovakia -> Czechia
    230: 231,   # Ethiopia PDR -> Ethiopia
    736: 729,   # Sudan (former) -> Sudan
}

# The columns an item-level row is identified by, before the year columns are melted down. The
# item columns keep crop_provision's names because both services read the same FAOSTAT file.
CROP_ID_COLUMNS = ['area_code', 'area_code_M49', 'country', 'crop_code', 'crop']


def clean_crop_values(df_raw, items):
    """One row per country-item-year of FAOSTAT gross production value, in thousand USD.

    The bulk file is wide (one column per year, each with a flag column beside it) and mixes
    elements, units, aggregate areas and items nobody asked for. This keeps the gross-production-
    value rows in USD, keeps the requested items, drops the aggregate areas, and melts the year
    columns into rows.

    Args:
        df_raw (pd.DataFrame): the FAOSTAT Value of Production bulk table as shipped.
        items (iterable): the items to keep. Integer entries select by FAO item code (the
            owner's convention, robust to FAO's item-name revisions); strings select by name.

    Returns:
        pd.DataFrame: columns area_code, area_code_M49, country, crop_code, crop, year,
        livestock_provision_gep.
    """
    years = range(FAOSTAT_FIRST_YEAR, FAOSTAT_LAST_YEAR + 1)
    df = df_raw[(df_raw['Unit'] == FAOSTAT_VALUE_UNIT)
                & (df_raw['Element Code'] == FAOSTAT_GROSS_PRODUCTION_VALUE_ELEMENT)].copy()
    df = df.drop(columns=[col for col in df.columns if col.endswith('F')])

    old_names = ['Area Code', 'Area Code (M49)', 'Area', 'Item Code', 'Item'] + [f'Y{y}' for y in years]
    new_names = CROP_ID_COLUMNS + [str(y) for y in years]
    df = df.rename(columns=dict(zip(old_names, new_names)))

    codes = [i for i in items if isinstance(i, int)]
    names = [i for i in items if isinstance(i, str)]
    df = df[df['crop_code'].isin(codes) | df['crop'].isin(names)]
    df = df[~df['country'].isin(FAOSTAT_AGGREGATE_AREAS)]
    logging.info(f'Finished cleaning up ({df.shape[0]} rows).')

    df = pd.melt(df, id_vars=CROP_ID_COLUMNS, value_vars=[str(y) for y in years],
                 var_name='year', value_name='livestock_provision_gep')
    df['area_code'] = pd.to_numeric(df['area_code'], errors='coerce').astype(int)
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)
    df.loc[df['area_code'] == FAOSTAT_TURKIYE_AREA_CODE, 'country'] = 'Turkey'

    logging.info(f'Reshaped to long format ({df.shape[0]} rows).')
    return df


def build_rental_rate_lookup(df_raw):
    """The CWoN rental rates as one row per country and decade start.

    The workbook is one column per decade ("1961-1970", "1971-1980", ...) keyed on the FAO area
    code. Melting it and keeping the decade's first year gives the lookup merge_crop_with_coefs
    reads as-of. Columns that are not a decade (the ISO3 label) carry no leading year, so they
    fall out with the rows whose decade start does not parse.

    Args:
        df_raw (pd.DataFrame): the CWoN coefficient table as shipped.

    Returns:
        pd.DataFrame: columns FAO, year, rental_rate.
    """
    df = df_raw.melt(id_vars=['Order', 'FAO', 'Country/territory'],
                     var_name='Decade', value_name='rental_rate')
    df['Decade_start'] = df['Decade'].str.extract(r'^(\d{4})').astype(float)
    df = df.dropna(subset=['Decade_start', 'FAO'])

    df = df[['FAO', 'Decade_start', 'rental_rate']].copy()
    df['FAO'] = df['FAO'].astype(int)
    df['Decade_start'] = df['Decade_start'].astype(int)
    df = df.rename(columns={'Decade_start': 'year'})
    logging.info(f'Prepared coef lookup ({df.shape[0]} rows).')
    return df


def merge_crop_with_coefs(df_crop_value, df_crop_coefs):
    """Production value attributed to land, country by country.

    Each year takes the rental rate of the most recent decade that has started, which is a
    backward as-of merge on year within a country. A country the CWoN table never covers keeps
    a missing rate, so its value becomes missing rather than being attributed in full.

    Args:
        df_crop_value (pd.DataFrame): long item values, with area_code, year and
            livestock_provision_gep.
        df_crop_coefs (pd.DataFrame): the rental-rate lookup, with FAO, year and rental_rate.

    Returns:
        pd.DataFrame: df_crop_value with rental_rate attached and livestock_provision_gep
        multiplied by it, sorted by country then year.
    """
    merged_parts = []
    for code, df_group in df_crop_value.groupby('area_code', sort=True):
        lookup_sub = df_crop_coefs[df_crop_coefs['FAO'] == code]
        if lookup_sub.empty:
            df_group = df_group.copy()
            df_group['rental_rate'] = pd.NA
            merged_parts.append(df_group)
            continue

        merged = pd.merge_asof(
            left=df_group.sort_values('year'),
            right=lookup_sub.sort_values('year')[['year', 'rental_rate']],
            on='year',
            direction='backward',
        )
        merged_parts.append(merged)

    df = pd.concat(merged_parts, ignore_index=True)
    df['livestock_provision_gep'] = df['livestock_provision_gep'] * df['rental_rate']
    df = df.sort_values(by=['area_code', 'year'], ascending=[True, True])
    logging.info(f'Merged values + coefs ({df.shape[0]} rows).')
    return df


def normalize_m49_codes(df, column='area_code_M49', successors=None):
    """FAOSTAT's M49 area codes as integers, with dissolved states mapped to their successor.

    The codes arrive quoted ("'156"), so they are unquoted and cast before the mapping.

    Args:
        df (pd.DataFrame): a frame holding FAOSTAT area codes.
        column (str): the code column.
        successors (dict): code -> successor code, defaulting to M49_SUCCESSORS.

    Returns:
        pd.DataFrame: the frame with that column as integers, successors applied.
    """
    out = df.copy()
    out[column] = out[column].astype(str).str.replace("'", '', regex=False).astype(int)
    out[column] = out[column].replace(M49_SUCCESSORS if successors is None else successors)
    return out


def attach_countries(df_crop_value, df_countries):
    """Item-level rows carrying their country's identifiers and attributes.

    The correspondence is collapsed to one row per country first: joining against r264 as shipped
    would repeat a split country's production once per sub-region. The join keeps every FAOSTAT
    row, so a row whose M49 code the correspondence does not carry keeps missing identifiers
    rather than disappearing.

    FAOSTAT ships these values in thousand USD, so they are converted here, at the same point
    crop_provision converts, and the service reports plain USD like the rest of the library.

    Args:
        df_crop_value (pd.DataFrame): long item values with integer area_code_M49.
        df_countries (pd.DataFrame): the r264 country correspondence.

    Returns:
        pd.DataFrame: the item rows with country identifiers attached.
    """
    ee_r264_to_250 = utilities.collapse_countries_to_r250(
        df_countries,
        keep_columns=['area_code_M49', 'area_code', 'country', 'crop_code', 'crop', 'year',
                      'rental_rate', 'livestock_provision_gep'])
    df = hb.df_merge(ee_r264_to_250, df_crop_value, how='right',
                     left_on='iso3_r250_id', right_on='area_code_M49')
    df['livestock_provision_gep'] = df['livestock_provision_gep'] * FAOSTAT_THOUSAND_USD
    return df


def group_crops(df):
    """Item rows summed to one row per country and year."""
    hb.log('Grouping GEP by country-year.')
    df_gep_by_year_country = hb.df_groupby(df, ['iso3_r250_id', 'year'],
                                           agg_dict={'livestock_provision_gep': 'sum'},
                                           preserve='keep_all_valid')
    df_gep_by_year_country = df_gep_by_year_country.sort_values(by=['iso3_r250_id', 'year'],
                                                                ascending=[True, True])
    df_gep_by_year_country['livestock_provision_gep'] = pd.to_numeric(
        df_gep_by_year_country['livestock_provision_gep'], errors='coerce')

    logging.info(f'Grouped by country-year ({df_gep_by_year_country.shape[0]} rows).')
    return df_gep_by_year_country


def group_countries(df):
    """Country-year rows summed to one global row per year."""
    df_gep_by_year = hb.df_groupby(df, groupby_cols='year', agg_cols='livestock_provision_gep',
                                   preserve='keep_all_valid')
    df_gep_by_year.sort_values('year', inplace=True)
    logging.info(f'Grouped total by year ({df_gep_by_year.shape[0]} rows).')
    return df_gep_by_year


# The source repo's step-two attribution (lambda.py): lambda = the ecosystem-provided share
# of livestock feed, from GLEAM 3 dry-matter intake by feed category. These category lists
# ARE the method -- the first four are ecosystem-provided, all eight are total intake.
GLEAM_ECOSYSTEM_FEED_COLS = ("By-products", "Crop residues", "Fodder crop", "Grass and leaves")
GLEAM_TOTAL_FEED_COLS = GLEAM_ECOSYSTEM_FEED_COLS + ("Grains", "Oil seed cakes",
                                                     "Other edible", "Other non-edible")
# What the public GLEAM 3 dashboard serves, confirmed against the live Input Data table on
# 2026-08-22: it carries all four ecosystem categories but only two of the four others, so
# "Other edible" and "Other non-edible" are missing. Both are DENOMINATOR-ONLY. A lambda
# computed without them divides by too small a total, so it is biased upward: it is an UPPER
# BOUND on nature's share of feed, not an estimate of it. That is also exactly why a run on
# dashboard data cannot reproduce the author's figures, whose extract carries all eight.
GLEAM_DASHBOARD_FEED_COLS = GLEAM_ECOSYSTEM_FEED_COLS + ("Grains", "Oil seed cakes")


def feed_lambda_by_country(gleam_dmi_df):
    """One row per country: lambda, the ecosystem-provided share of total feed intake.

    The denominator is whichever of GLEAM_TOTAL_FEED_COLS the frame actually carries. With all
    eight this is nature's share. With only the six the public dashboard serves it is an upper
    bound, because the two absent categories belong to the denominator alone; the returned
    `lambda_is_upper_bound` column says which case produced the number, so a downstream reader
    cannot mistake one for the other.

    Args:
        gleam_dmi_df (pd.DataFrame): GLEAM 3 dry-matter intake with iso3_r250_id,
            iso3_r250_label and one column per feed category, any number of rows per country
            (species, production systems).

    Returns:
        pd.DataFrame: iso3_r250_id, iso3_r250_label, lambda, lambda_is_upper_bound.

    Raises:
        ValueError: if a category that feeds the NUMERATOR is missing. Nature's share cannot be
            bounded in either direction without all four, so there is no honest number to give.
    """
    eco_cols = [c for c in GLEAM_ECOSYSTEM_FEED_COLS if c in gleam_dmi_df.columns]
    missing_eco = [c for c in GLEAM_ECOSYSTEM_FEED_COLS if c not in gleam_dmi_df.columns]
    if missing_eco:
        raise ValueError(
            f"GLEAM intake is missing ecosystem feed categories {missing_eco}, which are the "
            f"numerator: nature's share cannot be computed or bounded without them.")

    total_cols = [c for c in GLEAM_TOTAL_FEED_COLS if c in gleam_dmi_df.columns]
    missing_total = [c for c in GLEAM_TOTAL_FEED_COLS if c not in total_cols]
    grouped = (gleam_dmi_df.groupby(['iso3_r250_id', 'iso3_r250_label'], dropna=False)[total_cols]
               .sum().reset_index())
    grouped['lambda'] = grouped[eco_cols].sum(axis=1) / grouped[total_cols].sum(axis=1)
    grouped['lambda_is_upper_bound'] = bool(missing_total)
    return grouped[['iso3_r250_id', 'iso3_r250_label', 'lambda', 'lambda_is_upper_bound']]

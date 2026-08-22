# -*- coding: utf-8 -*-
"""Crop-provision science: FAOSTAT gross production value x the CWoN land rental rate.

The quantity-and-price stage is already fused in the source data: FAOSTAT's Value of Production
bulk file reports gross production value per country, crop and year, so no separate price join
happens here. Attribution is the Changing Wealth of Nations 2024 land rental rate, a per-country
share that varies by decade, applied by an as-of merge so each year takes the rate of the decade
it falls in. The result is converted from FAOSTAT's thousand USD to plain USD once, before any
grouping, and collapsed onto the r250 country list.

File reading is kept to three thin functions at the top; everything below them is a pure
transformation over frames, which is what the tests exercise.
"""
import logging

import pandas as pd
import hazelbean as hb

from global_invest import utilities

# FAOSTAT's Value of Production table stacks several elements and units in one file. The
# valuation reads gross production value in current USD, which the file reports as element 57
# with unit "1000 USD".
FAOSTAT_VALUE_UNIT = '1000 USD'
FAOSTAT_GROSS_PRODUCTION_VALUE_ELEMENT = 57
# FAOSTAT ships crop values in thousand USD; every service in the library reports plain USD.
FAOSTAT_THOUSAND_USD = 1000.0
# The bulk file's year columns run Y1961 to Y2022, each shadowed by a Y<year>F data-quality flag.
FAOSTAT_FIRST_YEAR = 1961
FAOSTAT_LAST_YEAR = 2022
# FAOSTAT area 223 is Turkiye, which recent releases spell several ways. The country join runs on
# the M49 code, so the name is normalised only to keep the crop-level table readable.
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

# The columns a crop-level row is identified by, before the year columns are melted down.
CROP_ID_COLUMNS = ['area_code', 'area_code_M49', 'country', 'crop_code', 'crop']


def read_crop_values(path, items):
    """The FAOSTAT bulk file read and cleaned. See clean_crop_values for the science."""
    df_raw = pd.read_csv(path, encoding='ISO-8859-1')
    logging.info(f'Loaded crop values from {path} ({df_raw.shape[0]} rows).')
    return clean_crop_values(df_raw, items)


def read_crop_coefs(path):
    """The CWoN rental-rate workbook read and reshaped. See build_rental_rate_lookup."""
    df_raw = pd.read_csv(path, delimiter=';', encoding='utf-8')
    logging.info(f'Loaded crop coefs from {path} ({df_raw.shape[0]} rows).')
    return build_rental_rate_lookup(df_raw)


def clean_crop_values(df_raw, items):
    """One row per country-crop-year of FAOSTAT gross production value, in thousand USD.

    The bulk file is wide (one column per year, each with a flag column beside it) and mixes
    elements, units, aggregate areas and crops nobody asked for. This keeps the gross-production-
    value rows in USD, keeps the requested crops, drops the aggregate areas, and melts the year
    columns into rows.

    Args:
        df_raw (pd.DataFrame): the FAOSTAT Value of Production bulk table as shipped.
        items (iterable): the crop names to keep, matched against the Item column.

    Returns:
        pd.DataFrame: columns area_code, area_code_M49, country, crop_code, crop, year,
        crop_provision_gep.
    """
    years = range(FAOSTAT_FIRST_YEAR, FAOSTAT_LAST_YEAR + 1)
    df = df_raw[(df_raw['Unit'] == FAOSTAT_VALUE_UNIT)
                & (df_raw['Element Code'] == FAOSTAT_GROSS_PRODUCTION_VALUE_ELEMENT)].copy()
    df = df.drop(columns=[col for col in df.columns if col.endswith('F')])

    old_names = ['Area Code', 'Area Code (M49)', 'Area', 'Item Code', 'Item'] + [f'Y{y}' for y in years]
    new_names = CROP_ID_COLUMNS + [str(y) for y in years]
    df = df.rename(columns=dict(zip(old_names, new_names)))

    df = df[df['crop'].isin(items)]
    df = df[~df['country'].isin(FAOSTAT_AGGREGATE_AREAS)]
    logging.info(f'Finished cleaning up ({df.shape[0]} rows).')

    df = pd.melt(df, id_vars=CROP_ID_COLUMNS, value_vars=[str(y) for y in years],
                 var_name='year', value_name='crop_provision_gep')
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
        df_crop_value (pd.DataFrame): long crop values, with area_code, year and
            crop_provision_gep.
        df_crop_coefs (pd.DataFrame): the rental-rate lookup, with FAO, year and rental_rate.

    Returns:
        pd.DataFrame: df_crop_value with rental_rate attached and crop_provision_gep multiplied
        by it, sorted by country then year.
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
    df['crop_provision_gep'] = df['crop_provision_gep'] * df['rental_rate']
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


def attach_countries_in_usd(df_crop_value, df_countries):
    """Crop-level rows carrying their country's identifiers and attributes, in plain USD.

    The correspondence is collapsed to one row per country first: joining against r264 as shipped
    would repeat a split country's production once per sub-region. The join keeps every FAOSTAT
    row (a row whose M49 code the correspondence does not carry keeps missing identifiers rather
    than disappearing), and the unit conversion happens here, once, before any grouping.

    Args:
        df_crop_value (pd.DataFrame): long crop values with integer area_code_M49 and
            crop_provision_gep in thousand USD.
        df_countries (pd.DataFrame): the r264 country correspondence.

    Returns:
        pd.DataFrame: the crop rows with country identifiers attached and crop_provision_gep
        in USD.
    """
    ee_r264_to_250 = utilities.collapse_countries_to_r250(
        df_countries,
        keep_columns=['area_code_M49', 'area_code', 'country', 'crop_code', 'crop', 'year',
                      'rental_rate', 'Value'])
    df = hb.df_merge(ee_r264_to_250, df_crop_value, how='right',
                     left_on='iso3_r250_id', right_on='area_code_M49')
    df['crop_provision_gep'] = df['crop_provision_gep'] * FAOSTAT_THOUSAND_USD
    return df


def group_crops(df):
    """Crop rows summed to one row per country and year."""
    hb.log('Grouping GEP by country-year.')
    df_gep_by_year_country = hb.df_groupby(df, ['iso3_r250_id', 'year'],
                                           agg_dict={'crop_provision_gep': 'sum'},
                                           preserve='keep_all_valid')
    df_gep_by_year_country = df_gep_by_year_country.sort_values(by=['iso3_r250_id', 'year'],
                                                                ascending=[True, True])
    df_gep_by_year_country['crop_provision_gep'] = pd.to_numeric(
        df_gep_by_year_country['crop_provision_gep'], errors='coerce')

    logging.info(f'Grouped by country-year ({df_gep_by_year_country.shape[0]} rows).')
    return df_gep_by_year_country


def group_countries(df):
    """Country-year rows summed to one global row per year."""
    df_gep_by_year = hb.df_groupby(df, groupby_cols='year', agg_cols='crop_provision_gep',
                                   preserve='keep_all_valid')
    df_gep_by_year.sort_values('year', inplace=True)
    logging.info(f'Grouped total by year ({df_gep_by_year.shape[0]} rows).')
    return df_gep_by_year

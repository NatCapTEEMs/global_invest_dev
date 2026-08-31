# -*- coding: utf-8 -*-
"""Livestock-provision science: FAOSTAT gross production value x the CWoN land rental rate.

The valuation runs the same calculation crop_provision does, on the livestock items of the same
FAOSTAT Value of Production file: gross production value per country, item and year, attributed
to land by the Changing Wealth of Nations 2024 rental rate, which varies by decade and is applied
by a backward as-of merge on year.

Two differences from crop_provision are deliberate and load-bearing. The item selection is by FAO
item CODE rather than name (the service owner's convention, robust to FAO renaming items), and
the values are NOT converted out of FAOSTAT's thousand USD, so this service's totals are in
thousand USD where every other service reports plain USD. That second one is an open item flagged
in the tracker, not a decision: see attach_countries.

Step two of the port, feed_lambda_by_country, computes the ecosystem-provided share of livestock
feed from GLEAM 3, wired through the task layer beside the rental-rate attribution.

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
# FAOSTAT area 223 is Turkiye, which recent releases spell several ways. The country join runs on
# the M49 code, so the name is normalised only to keep the item-level table readable.
FAOSTAT_TURKIYE_AREA_CODE = 223


# The columns an item-level row is identified by, before the year columns are melted down. The
# item columns keep crop_provision's names because both services read the same FAOSTAT file.
CROP_ID_COLUMNS = ['area_code', 'area_code_M49', 'country', 'crop_code', 'crop']






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
    if 'gross_production_value' in df.columns:
        # The same conversion, or the feed-share attribution downstream would multiply a share by
        # a figure still in thousands and come out a thousand times too small.
        df['gross_production_value'] = df['gross_production_value'] * FAOSTAT_THOUSAND_USD
    return df






# The source repo's step-two attribution (lambda.py): lambda = the ecosystem-provided share
# of livestock feed, from GLEAM 3 dry-matter intake by feed category. These category lists
# ARE the method -- the first four are ecosystem-provided, all eight are total intake.
GLEAM_ECOSYSTEM_FEED_COLS = ("By-products", "Crop residues", "Fodder crop", "Grass and leaves")
GLEAM_TOTAL_FEED_COLS = GLEAM_ECOSYSTEM_FEED_COLS + ("Grains", "Oil seed cakes",
                                                     "Other edible", "Other non-edible")
# The public GLEAM 3 dashboard serves ALL EIGHT categories, but its table shows a different
# subset per species, which is why a first look at one species suggested only six existed:
# ruminants (buffalo, cattle, goats, sheep) get fodder crop and crop residues and no "other"
# categories, while chickens and pigs get "Other edible" and "Other non-edible" and no fodder
# crop. Harvesting every species and unioning the columns gives the full eight, so a share
# built from the dashboard is an estimate rather than a bound. The per-species layouts are
# recorded in howto_harvest_gleam_intake.md.
GLEAM_RUMINANT_FEED_COLS = GLEAM_ECOSYSTEM_FEED_COLS + ("Grains", "Oil seed cakes")
GLEAM_MONOGASTRIC_FEED_COLS = ("By-products", "Crop residues", "Grains", "Grass and leaves",
                               "Oil seed cakes", "Other edible", "Other non-edible")


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


# The dashboard harvest arrives one row per country, species and production system, with the
# feed columns formatted for display: thousands separated by commas, and an empty cell where a
# system does not occur. The country column is the dashboard's own code, which is ISO3 for
# nearly every entry but carries a few territories GLEAM models separately.
GLEAM_DASHBOARD_ID_COLUMNS = ('country_code', 'species', 'Area', 'Animal', 'LPS')


def clean_gleam_dashboard_intake(df_raw, df_countries):
    """The harvested dashboard table, numeric and keyed to r250 countries.

    Args:
        df_raw (pd.DataFrame): the harvest, carrying country_code and one column per feed
            category, values formatted with thousands separators.
        df_countries (pd.DataFrame): the r264 correspondence, for iso3_r250_id and label.

    Returns:
        tuple: (cleaned frame ready for feed_lambda_by_country, list of country codes that
        matched no r250 country). The unmatched are returned rather than dropped silently,
        because a code that matches nothing is intake leaving the account unannounced.
    """
    import pandas as pd
    df = df_raw.copy()
    feed_columns = [c for c in df.columns if c in GLEAM_TOTAL_FEED_COLS]
    for column in feed_columns:
        df[column] = pd.to_numeric(
            df[column].astype(str).str.replace(',', '', regex=False).str.strip().replace('', None),
            errors='coerce')

    labels = (df_countries[['iso3_r250_label', 'iso3_r250_id']]
              .dropna().drop_duplicates('iso3_r250_label'))
    df = df.merge(labels, how='left', left_on='country_code', right_on='iso3_r250_label')
    unmatched = sorted(df.loc[df['iso3_r250_id'].isna(), 'country_code'].unique().tolist())
    matched = df[df['iso3_r250_id'].notna()]
    return matched[['iso3_r250_id', 'iso3_r250_label'] + feed_columns], unmatched


def feed_share_gep(df_country_year, df_lambda):
    """Livestock value attributed by the share of feed that ecosystems provided.

    The account currently attributes livestock value with the CWoN land rental rate, which is
    the crop method's factor and stands in for this one. Nature's contribution to a farmed animal
    is the feed it ate that nature grew, which is what GLEAM's intake table measures, so this
    applies that share to the same gross production value.

    Both columns are produced and neither replaces the other: which factor the account uses is a
    decision for the group, and the two differ by enough that making it silently would change the
    headline.

    Args:
        df_country_year (pd.DataFrame): one row per country and year, carrying iso3_r250_id and
            gross_production_value.
        df_lambda (pd.DataFrame): feed_lambda_by_country's output, with iso3_r250_id and lambda.

    Returns:
        pd.DataFrame: df_country_year with feed_share and livestock_provision_gep_feed_share
        added. A country GLEAM does not model keeps a missing share, so its feed-share value is
        missing rather than zero.
    """
    df = df_country_year.merge(
        df_lambda[['iso3_r250_id', 'lambda']].rename(columns={'lambda': 'feed_share'}),
        on='iso3_r250_id', how='left')
    df['livestock_provision_gep_feed_share'] = df['gross_production_value'] * df['feed_share']
    return df

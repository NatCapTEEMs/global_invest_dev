# -*- coding: utf-8 -*-
"""Livestock-provision science: CWoN livestock revenue x the CWoN land rental rate.

The valuation follows the authors' write-up (Livestock_Provision.docx, 26 Oct 2026) and the
agricultural-land framework of The Changing Wealth of Nations 2024: a country's livestock rent is
its total livestock production revenue times its land rental rate, both read for the base year
from the CWoN 2024 reproducibility package. CWoN builds the revenue from the products of its
Table 22 (indigenous meat, whole fresh milk, hides, skins, wool, hair), and the rate is land costs
as a share of the value of production. Revenue is in current US dollars, hence 2019 dollars for a
2019 run, and the rental rate is a dimensionless share, so nothing here converts currency.

The package also ships the rent product by product. Summed by country it must equal revenue x
rate, which `product_rent_check` measures: the companion tables are a check on each other, not a
second estimate.

feed_lambda_by_country computes the ecosystem-provided share of livestock feed from GLEAM 3. It
is applied to the same CWoN revenue as a diagnostic beside the headline, not in its place.

Every function here is a pure transformation over frames, which is what the tests exercise. The
task module reads the three CWoN tables and passes the frames in.
"""
import pandas as pd

from global_invest import utilities


def cwon_series_long(df, value_name):
    """A CWoN wide table (one YRyyyy column per year) as one row per country and year.

    Args:
        df (pd.DataFrame): a CWoN table with countrycode and YRyyyy columns, plus itemcode and
            item when it is product-level. Missing values are already NaN.
        value_name (str): the name for the value column.

    Returns:
        pd.DataFrame: countrycode, (itemcode, item,) year and value_name, with integer years.
    """
    keys = [c for c in ('countrycode', 'itemcode', 'item') if c in df.columns]
    years = [c for c in df.columns if str(c).startswith('YR')]
    out = df.melt(id_vars=keys, value_vars=years, var_name='year', value_name=value_name)
    out['year'] = out['year'].str[2:].astype(int)
    out[value_name] = pd.to_numeric(out[value_name], errors='coerce')
    return out


def cwon_livestock_rent(revenue_long, rate_long):
    """Livestock rent per country and year: CWoN revenue times the CWoN land rental rate.

    A country-year missing either input is missing, never zero. A zero comes only from a zero
    product of two present inputs, as the write-up specifies.

    Args:
        revenue_long (pd.DataFrame): countrycode, year, livestock_revenue (current USD).
        rate_long (pd.DataFrame): countrycode, year, rental_rate (share of production value).

    Returns:
        pd.DataFrame: countrycode, year, livestock_revenue, rental_rate and
        livestock_provision_gep, one row per country and year either table carries.
    """
    for frame, name in ((revenue_long, 'revenue'), (rate_long, 'rental-rate')):
        if frame.duplicated(['countrycode', 'year']).any():
            raise ValueError('The CWoN %s table repeats a country and year.' % name)
    out = revenue_long.merge(rate_long, on=['countrycode', 'year'], how='outer')
    out['livestock_provision_gep'] = out['livestock_revenue'] * out['rental_rate']
    return out


def product_rent_check(rent_long, product_rent_long):
    """Revenue x rate against the package's product-level rents summed by country and year.

    Args:
        rent_long (pd.DataFrame): cwon_livestock_rent's output.
        product_rent_long (pd.DataFrame): countrycode, itemcode, item, year, product_rent.

    Returns:
        pd.DataFrame: countrycode, year, livestock_provision_gep, product_rent_sum and their
        difference, for every country-year either side carries. A country-year whose products
        are all missing sums to missing, not zero.
    """
    summed = (product_rent_long.groupby(['countrycode', 'year'])['product_rent']
              .sum(min_count=1).rename('product_rent_sum').reset_index())
    out = rent_long[['countrycode', 'year', 'livestock_provision_gep']].merge(
        summed, on=['countrycode', 'year'], how='outer')
    out['difference'] = out['livestock_provision_gep'] - out['product_rent_sum']
    return out


def attach_countries(rent_long, df_countries):
    """Country-year rows carrying the account's r250 identifiers and attributes.

    CWoN keys countries by ISO3, which is the account's iso3_r250_label. The correspondence is
    collapsed to one row per country first, so a split country is not repeated. Every CWoN row is
    kept, so a code the correspondence lacks keeps missing identifiers rather than disappearing.

    Args:
        rent_long (pd.DataFrame): rows keyed by countrycode.
        df_countries (pd.DataFrame): the r264 country correspondence.

    Returns:
        pd.DataFrame: rent_long with the r250 identifiers and attributes joined on.
    """
    countries = utilities.collapse_countries_to_r250(df_countries)
    return countries.merge(rent_long, left_on='iso3_r250_label', right_on='countrycode',
                           how='right')


# The source repo's step-two attribution (lambda.py): lambda = the ecosystem-provided share
# of livestock feed, from GLEAM 3 dry-matter intake by feed category. These category lists
# ARE the method -- the first four are ecosystem-provided, all eight are total intake.
GLEAM_ECOSYSTEM_FEED_COLS = ("By-products", "Crop residues", "Fodder crop", "Grass and leaves")
GLEAM_TOTAL_FEED_COLS = GLEAM_ECOSYSTEM_FEED_COLS + ("Grains", "Oil seed cakes",
                                                     "Other edible", "Other non-edible")
# The public GLEAM 3 dashboard serves ALL EIGHT categories, but its table shows a different
# subset per species: ruminants (buffalo, cattle, goats, sheep) get fodder crop and crop
# residues and no "other" categories, while chickens and pigs get "Other edible" and "Other
# non-edible" and no fodder crop. Harvesting every species and unioning the columns gives the
# full eight, so a share built from the dashboard is an estimate rather than a bound. The
# per-species layouts are recorded in docs/howto_harvest_gleam_intake.md.
GLEAM_RUMINANT_FEED_COLS = GLEAM_ECOSYSTEM_FEED_COLS + ("Grains", "Oil seed cakes")
GLEAM_MONOGASTRIC_FEED_COLS = ("By-products", "Crop residues", "Grains", "Grass and leaves",
                               "Oil seed cakes", "Other edible", "Other non-edible")


def feed_lambda_by_country(gleam_dmi_df):
    """One row per country: lambda, the ecosystem-provided share of total feed intake.

    The denominator is whichever of GLEAM_TOTAL_FEED_COLS the frame actually carries. With all
    eight this is nature's share. A frame missing a denominator-only category gives an upper
    bound, because those categories belong to the denominator alone. A single species' table
    from the public dashboard is such a frame, while the union over species carries all eight.
    The returned `lambda_is_upper_bound` column says which case produced the number, so a
    downstream reader cannot mistake one for the other.

    Args:
        gleam_dmi_df (pd.DataFrame): GLEAM 3 dry-matter intake with iso3_r250_id,
            iso3_r250_label and one column per feed category, any number of rows per country
            (species, production systems).

    Returns:
        pd.DataFrame: iso3_r250_id, iso3_r250_label, lambda, lambda_is_upper_bound.

    Raises:
        ValueError: if a category that feeds the NUMERATOR is missing. Nature's share cannot be
            bounded in either direction without all four.
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


def feed_share_gep(df_country_year, df_lambda, gross_column='gross_production_value'):
    """Livestock value attributed by the share of feed that ecosystems provided, a diagnostic.

    The account's livestock value is CWoN revenue times the CWoN land rental rate, the authors'
    method. This column applies GLEAM's ecosystem-provided feed share to the same revenue instead
    and is written beside the headline as a diagnostic. It does not replace it.

    Args:
        df_country_year (pd.DataFrame): one row per country and year, carrying iso3_r250_id and
            the gross value column.
        df_lambda (pd.DataFrame): feed_lambda_by_country's output, with iso3_r250_id and lambda.
        gross_column (str): the gross value the share applies to. It needs no rental rate, so a
            caller whose `gross_production_value` keeps only rent-valued rows passes its own.

    Returns:
        pd.DataFrame: df_country_year with feed_share and livestock_provision_gep_feed_share
        added. A country GLEAM does not model keeps a missing share, so its feed-share value is
        missing rather than zero.
    """
    df = df_country_year.merge(
        df_lambda[['iso3_r250_id', 'lambda']].rename(columns={'lambda': 'feed_share'}),
        on='iso3_r250_id', how='left')
    df['livestock_provision_gep_feed_share'] = df[gross_column] * df['feed_share']
    return df

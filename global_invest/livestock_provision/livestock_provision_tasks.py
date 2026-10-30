import os

import pandas as pd
import hazelbean as hb
from global_invest import utilities

from global_invest.livestock_provision import livestock_provision_functions



def read_cwon_table(path, value_name):
    """A CWoN 2024 table read and reshaped long. See cwon_series_long for the science.

    The World Bank writes a missing value as "..", which is read as missing.
    """
    return livestock_provision_functions.cwon_series_long(
        pd.read_excel(path, na_values=['..']), value_name)


def publish_inputs(p):
    """Every task's first line: the livestock valuation's es_config row (defaults layer -- a caller-set value wins)
    plus the shared country references and the results registry."""
    utilities.hydrate_es_config(p, 'livestock_provision', log=hb.log)
    utilities.hydrate_es_parameters(p, 'livestock_provision', log=hb.log)
    utilities.initialize_country_paths(p, simplified='30sec')
    if not hasattr(p, 'results'):
        p.results = {}
    return p

def livestock_provision(p):
    """
    Parent task for livestock provision.
    """
    publish_inputs(p)

def gep_preprocess(p):
    """Preprocessing placeholder, not implemented and not meant to be run by users.

    A preprocess task's output is an input to the model, saved at the canonical project
    attribute p.livestock_provision_input_path and promoted to the base_data_dir users receive.
    """
    publish_inputs(p)
    if not p.run_this:
        return p
    pass # NYI

def gep_calculation(p):
    """ GEP calculation task for livestock provision."""
    publish_inputs(p)
    # Define at least the primary output for the service, which for this project is gep_by_country_base_year.   
    # The extra tables this service writes are registered up front, so the skip check
    # covers all of them and the results page can find every one on a skipped rerun.
    service_results, already_done = utilities.begin_gep_calculation(
        p, 'livestock_provision', extra_results={
         'gep_by_country_year_crop': os.path.join(p.cur_dir, 'gep_by_country_year_crop.csv'),
         'gep_by_country_year': os.path.join(p.cur_dir, 'gep_by_country_year.csv'),
         'gep_by_year': os.path.join(p.cur_dir, 'gep_by_year.csv')})
    if already_done:
        return

    if not p.run_this:
        return p
    # The authors' method: CWoN livestock revenue x the CWoN land rental rate, from the CWoN 2024
    # reproducibility package (see the functions module for the write-up it follows).
    revenue = read_cwon_table(p.get_path(p.livestock_provision_cwon_revenue_path), 'livestock_revenue')
    rate = read_cwon_table(p.get_path(p.livestock_provision_cwon_rental_rate_path), 'rental_rate')
    rent = livestock_provision_functions.cwon_livestock_rent(revenue, rate)

    # The package's product-level rents, summed by country, must give the same numbers.
    product_rent = read_cwon_table(p.get_path(p.livestock_provision_cwon_product_rent_path), 'product_rent')
    check = livestock_provision_functions.product_rent_check(rent, product_rent)
    checked = check[check['year'] == int(p.gep_base_year)]
    both = checked['livestock_provision_gep'].notna() & checked['product_rent_sum'].notna()
    hb.log('CWoN check, %s: revenue x rate against product rents summed, %d countries, max |diff| '
           '%.3g USD, same missing countries: %s'
           % (p.gep_base_year, int(both.sum()), checked.loc[both, 'difference'].abs().max(),
              bool((checked['livestock_provision_gep'].isna()
                    == checked['product_rent_sum'].isna()).all())))

    df_gep_by_country_year = livestock_provision_functions.attach_countries(rent, p.df_countries)
    unplaced = df_gep_by_country_year[df_gep_by_country_year['iso3_r250_id'].isna()]
    if len(unplaced):
        hb.log('  CWoN codes with no r250 country, not carried: %s (%d valued country-years)'
               % (', '.join(sorted(unplaced['countrycode'].unique())),
                  int(unplaced['livestock_provision_gep'].notna().sum())))
    df_gep_by_country_year = df_gep_by_country_year[
        df_gep_by_country_year['iso3_r250_id'].notna()].drop(columns=['countrycode'])
    df_gep_by_country_year_crop = livestock_provision_functions.attach_countries(
        product_rent, p.df_countries)
    df_gep_by_country_year_crop = df_gep_by_country_year_crop[
        df_gep_by_country_year_crop['iso3_r250_id'].notna()]

    # The feed share, beside the headline as a diagnostic: GLEAM 3 intake gives the share of what
    # livestock ate that ecosystems grew, applied here to the same CWoN revenue.
    df_gleam, df_gleam_unmatched = livestock_provision_functions.clean_gleam_dashboard_intake(
        hb.df_read(p.get_path(p.gleam_dmi_path), delimiter='|'), p.df_countries)
    df_lambda = livestock_provision_functions.feed_lambda_by_country(df_gleam)
    df_gep_by_country_year = livestock_provision_functions.feed_share_gep(
        df_gep_by_country_year, df_lambda, gross_column='livestock_revenue')
    hb.log('Feed share (diagnostic): %d countries carry one, %d GLEAM rows match no country in '
           'the account.' % (int(df_lambda['lambda'].notna().sum()), len(df_gleam_unmatched)))

    df_gep_by_year = utilities.sum_countries_to_year(df_gep_by_country_year, 'livestock_provision_gep')

    df_gep_by_country_base_year = df_gep_by_country_year.loc[
        df_gep_by_country_year['year'] == int(p.gep_base_year)].copy()

    # Write to CSVs
    hb.df_write(df_gep_by_country_year_crop, p.results['livestock_provision']['gep_by_country_year_crop'])
    hb.df_write(df_gep_by_country_year, p.results['livestock_provision']['gep_by_country_year'])
    utilities.write_gep_by_country(
        p, df_gep_by_country_base_year[utilities.published_country_columns(
            df_gep_by_country_base_year, 'livestock_provision')],
        p.results['livestock_provision']['gep_by_country_base_year'])
    hb.df_write(df_gep_by_year, p.results['livestock_provision']['gep_by_year'], handle_quotes='all')
    hb.df_write(df_gep_by_year, hb.replace_ext(p.results['livestock_provision']['gep_by_year'], 'xlsx'), handle_quotes='all')
    
    # Map only: the r264-expanded boundaries, each sub-region carrying its country's value.
    gdf_gep_by_country_base_year = hb.df_merge(p.gdf_countries_simplified, df_gep_by_country_base_year, how='outer', left_on='ee_r264_id', right_on='ee_r264_id')
    gdf_gep_by_country_base_year.to_file(p.results['livestock_provision']['gep_by_country_base_year'].replace('.csv', '.gpkg'), driver='GPKG')

    value_gep_base_year = df_gep_by_country_base_year['livestock_provision_gep'].sum()

    hb.log(f"Total GEP value for base year {p.gep_base_year}: {value_gep_base_year}")

    return value_gep_base_year

def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    if not p.run_this:
        return p
    utilities.render_service_results(p)

def gep_results_distribution(p):
    """Copy this service's results into the output directory. Shared implementation in
    utilities."""
    publish_inputs(p)
    if not p.run_this:
        return p
    utilities.distribute_results(p, 'livestock_provision')

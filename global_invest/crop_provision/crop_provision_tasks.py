import os

import pandas as pd
import hazelbean as hb
from global_invest import utilities

from global_invest.crop_provision import crop_provision_initialize
from global_invest.crop_provision import crop_provision_functions

# Project content/config (was crop_provision_defaults.py; the template keeps content in the task module).


def read_crop_values(path, items):
    """The FAOSTAT bulk file read and cleaned. See clean_crop_values for the science.

    The file ships Latin-1 encoded, so it is read as such rather than as UTF-8.
    """
    return crop_provision_functions.clean_crop_values(
        pd.read_csv(path, encoding='ISO-8859-1'), items)


def read_crop_coefs(path):
    """The CWoN rental-rate table read and reshaped. See build_rental_rate_lookup for the science.

    The table ships semicolon-delimited.
    """
    return utilities.build_rental_rate_lookup(
        pd.read_csv(path, delimiter=';', encoding='utf-8'))


def publish_inputs(p):
    """Every task's first line: the FAO crop-production valuation's es_config row (defaults layer -- a caller-set value wins)
    plus the shared country references and the results registry."""
    utilities.hydrate_es_config(p, 'crop_provision', log=hb.log)
    utilities.hydrate_es_parameters(p, 'crop_provision', log=hb.log)
    utilities.initialize_country_paths(p, simplified='30sec')
    if not hasattr(p, 'results'):
        p.results = {}
    return p

def crop_provision(p):
    """
    Parent task for commercial agriculture.
    """
    publish_inputs(p)

def gep_preprocess(p):
    """
    Preprocessing tasks are assumed NOT to be run by the user. Instead, it is assumed that the output of a preprocess
    task is an input to the actual model, saved at the canonical project attribute p.crop_provision_input_path.
    These are preprocessing tasks are still provided for reference, but are not intended to be run directly by the user.
    We will "promote" the data outputed by a preprocess task to the base_data_dir provided to users.
    """
    publish_inputs(p)
    pass # NYI

def gep_calculation(p):
    """ GEP calculation task for commercial agriculture."""
    publish_inputs(p)
    # Define at least the primary output for the service, which for this project is gep_by_country_base_year.   
    # The extra tables this service writes are registered up front, so the skip check
    # covers all of them and the results page can find every one on a skipped rerun.
    service_results, already_done = utilities.begin_gep_calculation(
        p, 'crop_provision', extra_results={
         'gep_by_country_year_crop': os.path.join(p.cur_dir, 'gep_by_country_year_crop.csv'),
         'gep_by_country_year': os.path.join(p.cur_dir, 'gep_by_country_year.csv'),
         'gep_by_year': os.path.join(p.cur_dir, 'gep_by_year.csv')})
    if already_done:
        return

    if not getattr(p, 'crop_provision_subservices', None):
        p.commercial_attribute_subservices = utilities.read_column(
            p.crop_provision_default_items_path, 'item_fao')

    df_crop_value = read_crop_values(p.fao_input_path, p.commercial_attribute_subservices)
    df_crop_coefs = read_crop_coefs(p.cwon_crop_coefficients_path)

    df_gep_by_country_year_crop = crop_provision_functions.merge_crop_with_coefs(df_crop_value, df_crop_coefs)
    df_gep_by_country_year_crop = utilities.normalize_m49_codes(df_gep_by_country_year_crop)
    df_gep_by_country_year_crop = crop_provision_functions.attach_countries_in_usd(
        df_gep_by_country_year_crop, p.df_countries)

    df_gep_by_country_year = crop_provision_functions.group_crops(df_gep_by_country_year_crop)

    df_gep_by_year = crop_provision_functions.group_countries(df_gep_by_country_year)

    base_year = int(p.gep_base_year)
    df_gep_by_country_base_year = df_gep_by_country_year.loc[
        df_gep_by_country_year['year'] == base_year].copy()

    # Write to CSVs
    hb.df_write(df_gep_by_country_year_crop, p.results['crop_provision']['gep_by_country_year_crop'])
    hb.df_write(df_gep_by_country_year, p.results['crop_provision']['gep_by_country_year'])
    hb.df_write(df_gep_by_country_base_year[utilities.published_country_columns(
        df_gep_by_country_base_year, 'crop_provision')],
        p.results['crop_provision']['gep_by_country_base_year'])
    hb.df_write(df_gep_by_year, p.results['crop_provision']['gep_by_year'], handle_quotes='all')
    hb.df_write(df_gep_by_year, hb.replace_ext(p.results['crop_provision']['gep_by_year'], 'xlsx'), handle_quotes='all')
    
    # Map only: the r264-expanded boundaries, each sub-region carrying its country's value.
    gdf_gep_by_country_base_year = hb.df_merge(p.gdf_countries_simplified, df_gep_by_country_base_year, how='outer', left_on='ee_r264_id', right_on='ee_r264_id')
    gdf_gep_by_country_base_year.to_file(p.results['crop_provision']['gep_by_country_base_year'].replace('.csv', '.gpkg'), driver='GPKG')

    value_gep_base_year = df_gep_by_country_base_year['crop_provision_gep'].sum()

    hb.log(f"Total GEP value for base year {base_year}: {value_gep_base_year}")

    return value_gep_base_year

def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)

def gep_results_distribution(p):
    """Distribute the results of the GEP calculation."""
    publish_inputs(p)
    # This task is intended to copy the results to the output directory.
    hb.log("Distributing GEP results...")
    
    for key, value in p.results['crop_provision'].items():
        output_path = os.path.join(p.output_dir, key)
        hb.path_copy(value, output_path)
        hb.log(f"Distributed {key} to {output_path}")
    
    hb.log("GEP results distribution complete.")
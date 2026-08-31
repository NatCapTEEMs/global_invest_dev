import os

import pandas as pd
import hazelbean as hb
from global_invest import utilities

from global_invest.coastal_protection import coastal_protection_initialize
from global_invest.coastal_protection import coastal_protection_functions

# Both valuation workbooks ship their table on a single sheet with this name.
SOURCE_SHEET_NAME = 'Sheet1'


def read_mangrove_values(path):
    """The CWoN mangrove workbook read and renamed. See clean_mangrove_values."""
    return coastal_protection_functions.clean_mangrove_values(
        pd.read_excel(path, sheet_name=SOURCE_SHEET_NAME, engine='openpyxl'))


def read_deflator_multiplier(path, start_year, end_year):
    """The World Bank GDP deflator workbook read, melted to long and compounded over the span.

    Source: https://data.worldbank.org/indicator/NY.GDP.DEFL.KD.ZG
    """
    df_long = coastal_protection_functions.reshape_gdp_inflation_deflator(
        pd.read_excel(path, engine='openpyxl'))
    return coastal_protection_functions.deflator_multiplier_by_country(
        df_long, start_year, end_year)


def publish_inputs(p):
    """Every task's first line: the CWoN coastal-protection valuation's es_config row (defaults layer -- a caller-set value wins)
    plus the shared country references and the results registry."""
    utilities.hydrate_es_config(p, 'coastal_protection', log=hb.log)
    utilities.hydrate_es_parameters(p, 'coastal_protection', log=hb.log)
    utilities.initialize_country_paths(p, simplified='30sec')
    # Auxiliary science inputs beside the quantity row: the coral-reef workbook (really a second
    # sheet subgroup) and the GDP deflator (the drive folder spells it 'gdp_inflation_delator',
    # sic; staged locally under the corrected name, exact case for case-sensitive filesystems).
    p.coral_reef_ref_path = p.get_path(p.coastal_protection_coral_reef_path)
    p.df_gdp_inflation_deflator_path = p.get_path(p.coastal_protection_gdp_deflator_path)
    if not hasattr(p, 'results'):
        p.results = {}
    return p

def coastal_protection(p):
    """
    Parent task for mangrove coastal protection. Inputs resolve in publish_inputs.
    """
    publish_inputs(p)


def gep_preprocess(p):
    """
    Preprocessing tasks are assumed NOT to be run by the user. Instead, it is assumed that the output of a preprocess
    task is an input to the actual model, saved at the canonical project attribute p.coastal_protection_input_path.
    These are preprocessing tasks are still provided for reference, but are not intended to be run directly by the user.
    We will "promote" the data outputed by a preprocess task to the base_data_dir provided to users.
    """
    publish_inputs(p)
    pass # NYI



def gep_calculation(p):
    """ GEP calculation task for coastal protection."""
    publish_inputs(p)
    # Define at least the primary output for the service, which for this project is gep_by_country_base_year.   
    service_results, already_done = utilities.begin_gep_calculation(p, 'coastal_protection')
    if already_done:
        return

    base_year = coastal_protection_functions.COASTAL_PROTECTION_BASE_YEAR
    p.gdf_countries = hb.read_vector(p.gdf_countries_vector_path)

    df_mangrove_value = read_mangrove_values(p.gep_quantity_input_path)
    # The coral table already carries ee_r264_name, coral_reef_value and year, so nothing is
    # renamed or rescaled on the way in.
    df_coral_reef_value = pd.read_excel(p.coral_reef_ref_path, sheet_name=SOURCE_SHEET_NAME,
                                        engine='openpyxl')
    # The coral table is in CORAL_REEF_VALUE_YEAR currency, so inflation is applied from the
    # year after that through the base year.
    df_gdp_inflation_deflator = read_deflator_multiplier(
        p.df_gdp_inflation_deflator_path,
        coastal_protection_functions.CORAL_REEF_VALUE_YEAR + 1, base_year)

    df_mangrove = coastal_protection_functions.mangrove_gep_by_country(
        p.gdf_countries, df_mangrove_value)
    df_coral_reef = coastal_protection_functions.coral_reef_gep_by_country(
        p.gdf_countries, df_coral_reef_value, df_gdp_inflation_deflator, base_year)

    df_gep_by_country_year = coastal_protection_functions.combine_coastal_components(
        df_mangrove, df_coral_reef)
    df_gep_by_country_year = coastal_protection_functions.attach_country_attributes(
        df_gep_by_country_year, p.gdf_countries)

    df_gep_by_country_base_year = df_gep_by_country_year.loc[
        df_gep_by_country_year['year'] == base_year].copy()
    # `Value` arrives from the source table as an exact copy of coastal_protection_gep. Dropping it
    # here rather than at the write means the map file below carries the account's name as well.
    df_gep_by_country_base_year = df_gep_by_country_base_year.drop(columns=['Value'], errors='ignore')

    # The frame keeps its r264 columns for the map merge below; the published table does not.
    utilities.write_gep_by_country(
        p, df_gep_by_country_base_year[utilities.published_country_columns(
            df_gep_by_country_base_year, 'coastal_protection')],
        p.results['coastal_protection']['gep_by_country_base_year'])

    # Map only: the r264-expanded boundaries, each sub-region carrying its country's value.
    gdf_gep_by_country_base_year = hb.df_merge(p.gdf_countries_vector_simplified_path, df_gep_by_country_base_year, how='outer', on='ee_r264_id')
    gdf_gep_by_country_base_year.to_file(p.results['coastal_protection']['gep_by_country_base_year'].replace('.csv', '.gpkg'), driver='GPKG')

    value_gep_base_year = df_gep_by_country_base_year['coastal_protection_gep'].sum()

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
    
    for key, value in p.results['coastal_protection'].items():
        output_path = os.path.join(p.output_dir, key)
        hb.path_copy(value, output_path)
        hb.log(f"Distributed {key} to {output_path}")
    
    hb.log("GEP results distribution complete.")
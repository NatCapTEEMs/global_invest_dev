import os
import sys
import pandas as pd
import hazelbean as hb
from global_invest import utilities
import subprocess

from global_invest.extractive_materials_provision import extractive_materials_provision_initialize
from global_invest.extractive_materials_provision import extractive_materials_provision_functions

# Applied to (mineral rents share x GDP) in the valuation. Provenance UNDOCUMENTED as of 2026-08-16:
# no source in the code, the drive submission, or its raw_data notes -- open question for the service
# owner. Do not change without an owner-blessed source; the staged reference output embeds it.
MINERAL_RENT_GEP_FACTOR = 0.49



def publish_inputs(p):
    """Every task's first line: the mineral-rents valuation's es_config row (defaults layer -- a caller-set value wins)
    plus the shared country references and the results registry."""
    utilities.hydrate_es_config(p, 'extractive_materials_provision', log=hb.log)
    utilities.hydrate_es_parameters(p, 'extractive_materials_provision', log=hb.log)
    utilities.initialize_country_paths(p, simplified='30sec')
    if not hasattr(p, 'results'):
        p.results = {}
    return p

def extractive_materials_provision(p):
    """
    Parent task for extractive materials provision.
    """
    publish_inputs(p)
    pass  # Inputs resolve in publish_inputs.

def gep_preprocess(p):
    """
    Preprocessing tasks are assumed NOT to be run by the user. Instead, it is assumed that the output of a preprocess
    task is an input to the actual model, saved at the canonical project attribute p.extractive_materials_provision_input_path.
    These are preprocessing tasks are still provided for reference, but are not intended to be run directly by the user.
    We will "promote" the data outputed by a preprocess task to the base_data_dir provided to users.
    """
    publish_inputs(p)
    pass # NYI

def gep_calculation(p):
    """ GEP calculation task for extractive materials provision."""
    publish_inputs(p)
    # Define at least the primary output for the service, which for this project is gep_by_country_base_year.   
    service_results = {}
    p.results['extractive_materials_provision'] = service_results  
    p.results['extractive_materials_provision']['gep_by_country_base_year'] = os.path.join(p.cur_dir, "gep_by_country_base_year.csv")
    
    # Optional additional results.
    p.results['extractive_materials_provision']['gep_by_country_year_mineral'] = os.path.join(p.cur_dir, "gep_by_country_year_mineral.csv")
    p.results['extractive_materials_provision']['gep_by_country_year'] = os.path.join(p.cur_dir, "gep_by_country_year.csv")
    p.results['extractive_materials_provision']['gep_by_year'] = os.path.join(p.cur_dir, "gep_by_year.csv")
            
    # Check if all results exist
    if hb.path_all_exist(list(service_results.values())):
        hb.log("All results already exist. Skipping GEP calculation for extractive materials provision.")
    else:
        hb.log("Starting GEP calculation for extractive materials provision.")
        
        # Optimization here,
        # p.gdf_countries = hb.read_vector(p.gdf_countries)
        p.gdf_countries = hb.read_vector(p.gdf_countries_simplified)


        # 1. Read and process data
        df_mineral_values = extractive_materials_provision_functions.read_mineral_values(p.gep_attribution_input_path)

        df_gdp_values = extractive_materials_provision_functions.read_GDP_values(p.gep_quantity_input_path)


        df_mineral_gdp_values = df_mineral_values.merge(df_gdp_values, on=['Country Code', 'year'], how='left')

        df_mineral_gdp_values['extractive_materials_provision_gep'] = extractive_materials_provision_functions.mineral_rent_gep(
            df_mineral_gdp_values['mineral_rent'], df_mineral_gdp_values['GDP_currentUSD'], MINERAL_RENT_GEP_FACTOR)

        df_mineral_gdp_values['Value'] = df_mineral_gdp_values['extractive_materials_provision_gep']

        df_gep_by_country_year_mineral = df_mineral_gdp_values.copy()

        df_gep_by_country_year_mineral.drop_duplicates(subset=['Country Code', 'year'], inplace=True)
        
        # One row per country: r264 splits large countries, so the correspondence is
        # collapsed before the join.
        ee_r264_to_250 = utilities.collapse_countries_to_r250(p.df_countries)

        # Merge so it has all the good labels from the  
        df_gep_by_country_year_mineral = hb.df_merge(ee_r264_to_250, df_gep_by_country_year_mineral, how='left', left_on='iso3_r250_label', right_on='Country Code')
        
        # Rename value to extractive_materials_provision_gep

        df_gep_by_country_year =  df_gep_by_country_year_mineral.copy()
        
        df_gep_by_country_base_year = df_gep_by_country_year.loc[df_gep_by_country_year['year'] == 2019].copy()

        df_gep_by_year = extractive_materials_provision_functions.group_countries(df_gep_by_country_year)

        
        # Write to CSVs
        hb.df_write(df_gep_by_country_year_mineral, p.results['extractive_materials_provision']['gep_by_country_year_mineral'])
        hb.df_write(df_gep_by_country_year, p.results['extractive_materials_provision']['gep_by_country_year'])
        hb.df_write(df_gep_by_country_base_year, p.results['extractive_materials_provision']['gep_by_country_base_year'])   
        hb.df_write(df_gep_by_year, p.results['extractive_materials_provision']['gep_by_year'], handle_quotes='all')
        hb.df_write(df_gep_by_year, hb.replace_ext(p.results['extractive_materials_provision']['gep_by_year'], 'xlsx'), handle_quotes='all')


        # Use geopandas to merge the df_gep_by_country_base_year with the  to get the country names and other attributes
        gdf_gep_by_country_base_year = hb.df_merge(p.gdf_countries_simplified, df_gep_by_country_base_year, how='outer', left_on='ee_r264_id', right_on='ee_r264_id')
        gdf_gep_by_country_base_year.to_file(p.results['extractive_materials_provision']['gep_by_country_base_year'].replace('.csv', '.gpkg'), driver='GPKG')

        # Then sum the values across all countries. 
        value_gep_base_year = df_gep_by_country_base_year['extractive_materials_provision_gep'].sum()
        
        hb.log(f"Total GEP value for base year 2019: {value_gep_base_year}")
        
        return value_gep_base_year

def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)


def gep_load_results(p):
    publish_inputs(p)
    
    # Learn the paths by creating a temp task treep
    p_temp = hb.ProjectFlow()
    extractive_materials_provision_initialize.build_gep_service_calculation_task_tree(p_temp)
    p_temp.set_all_tasks_to_skip_if_dir_exists()
    p_temp.execute()
    
    print(p_temp.results)
    pass
        
def gep_results_distribution(p):
    """Distribute the results of the GEP calculation."""
    publish_inputs(p)
    # This task is intended to copy the results to the output directory.
    hb.log("Distributing GEP results...")
    
    for key, value in p.results['extractive_materials_provision'].items():
        output_path = os.path.join(p.output_dir, key)
        hb.path_copy(value, output_path)
        hb.log(f"Distributed {key} to {output_path}")
    
    hb.log("GEP results distribution complete.")
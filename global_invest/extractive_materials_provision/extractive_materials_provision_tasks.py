import os
import sys
import pandas as pd
import hazelbean as hb
import subprocess
import csv

from global_invest.extractive_materials_provision import extractive_materials_provision_initialization
from global_invest.extractive_materials_provision import extractive_materials_provision_functions


def extractive_materials_provision(p):
    """
    Parent task for commercial agriculture.
    """
    p.wb_mineral_input_ref_path = os.path.join(p.base_data_dir, 'extractive_materials_provision', 'API_NY.GDP.MINR.RT.ZS_DS2_en_csv_v2_6559.csv')
    p.wb_GDP_ref_path = os.path.join(p.base_data_dir, 'extractive_materials_provision', "API_NY.GDP.MKTP.CD_DS2_en_csv_v2_130122.csv")

def gep_preprocess(p):
    """
    Preprocessing tasks are assumed NOT to be run by the user. Instead, it is assumed that the output of a preprocess
    task is an input to the actual model, saved at the canonical project attribute p.extractive_materials_provision_input_path.
    These are preprocessing tasks are still provided for reference, but are not intended to be run directly by the user.
    We will "promote" the data outputed by a preprocess task to the base_data_dir provided to users.
    """
    pass # NYI

def gep_calculation(p):
    """ GEP calculation task for commercial agriculture."""
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
        hb.log("All results already exist. Skipping GEP calculation for commercial agriculture.")
    else:
        hb.log("Starting GEP calculation for commercial agriculture.")
        
        # Optimization here,
        # p.gdf_countries = hb.read_vector(p.gdf_countries)
        p.gdf_countries = hb.read_vector(p.gdf_countries_simplified)


        # 1. Read and process data
        df_mineral_values = extractive_materials_provision_functions.read_mineral_values(p.wb_mineral_input_ref_path)

        df_gdp_values = extractive_materials_provision_functions.read_GDP_values(p.wb_GDP_ref_path)


        df_mineral_gdp_values = df_mineral_values.merge(df_gdp_values, on=['Country Code', 'year'], how='left')

        df_mineral_gdp_values['extractive_materials_provision_gep'] = (df_mineral_gdp_values['mineral_rent'] / 100) * df_mineral_gdp_values['GDP_currentUSD']*0.49

        df_mineral_gdp_values['Value'] = df_mineral_gdp_values['extractive_materials_provision_gep']

        df_gep_by_country_year_mineral = df_mineral_gdp_values.copy()

        df_gep_by_country_year_mineral.drop_duplicates(subset=['Country Code', 'year'], inplace=True)
        
        # Drop repeated ids in df_countries
        ee_r264_to_250 = p.df_countries.copy()
        ee_r264_to_250 = ee_r264_to_250[ee_r264_to_250['ee_r264_label'] == ee_r264_to_250['iso3_r250_label']]
        
        cols_to_keep = [
            'ee_r264_id',	
            'iso3_r250_id',
            'ee_r264_label',
            'iso3_r250_label',
            'ee_r264_name',
            'iso3_r250_name',
            'continent',
            'region_un',
            'region_wb',
            'income_grp',
            'subregion',
        ]

        ee_r264_to_250.drop([i for i in ee_r264_to_250.columns if i not in cols_to_keep], axis=1, inplace=True, errors='ignore')
        # ee_r264_to_250 = ee_r264_to_250[cols_to_keep]
        
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
    """Display the results of the GEP calculation."""
    
    # Set the quarto path to wherever the current script is running. This means that the environment used needs to have quarto, which may not be true on e.g. codespaces.
    os.environ['QUARTO_PYTHON'] = sys.executable
    
    # Get the  list of current services run
    services_run = list(p.results.keys())
    
    # Additional groupbys = []
    
    # Imply from the service name the file_path for the results_qmd
    module_root = hb.get_projectflow_module_root()
    
    for service_label in services_run:
        results_qmd_path = os.path.join(module_root, service_label, f'{service_label}_results.qmd')    
        results_qmd_project_path = os.path.join(p.cur_dir, f'{service_label}_results.qmd')
        hb.create_directories(results_qmd_project_path)  # Ensure the directory exists   
        
        # Copy it to the project dir for cmd line processing (but will be removed again later because it makes confusion when people try to edit it and then rerun the script which won't of course update the results.)
        hb.path_copy(results_qmd_path, results_qmd_project_path)
        
        
        quarto_command = f"quarto render {results_qmd_project_path}"
        hb.log(f"Running quarto command: {quarto_command}")     

        """Run quarto with debug information"""
        # Set environment for more verbose output
        env = os.environ.copy()
        env['QUARTO_LOG_LEVEL'] = 'DEBUG'
        
        cmd = ['quarto', 'render', results_qmd_project_path, '--verbose']
        
        # print(f"Running command: {' '.join(results_qmd_project_path)}")
        print(f"Working directory: {os.getcwd()}")
        print(f"File exists: {os.path.exists(results_qmd_project_path)}")
        
        
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr into stdout
            text=True,
            bufsize=1,  # Line buffering
            universal_newlines=True
        )
        
        # Read line by line as they come
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.rstrip())
                sys.stdout.flush()  # Force immediate display
        # remove results_qmd_project_path
        hb.path_remove(results_qmd_project_path)
        
def gep_load_results(p):
    
    # Learn the paths by creating a temp task treep
    p_temp = hb.ProjectFlow()
    extractive_materials_provision_initialization.build_gep_service_calculation_task_tree(p_temp)
    p_temp.set_all_tasks_to_skip_if_dir_exists()
    p_temp.execute()
    
    print(p_temp.results)
    pass
        
def gep_results_distribution(p):
    """Distribute the results of the GEP calculation."""
    # This task is intended to copy the results to the output directory.
    hb.log("Distributing GEP results...")
    
    for key, value in p.results['extractive_materials_provision'].items():
        output_path = os.path.join(p.output_dir, key)
        hb.path_copy(value, output_path)
        hb.log(f"Distributed {key} to {output_path}")
    
    hb.log("GEP results distribution complete.")
import os
import sys
import pandas as pd
import hazelbean as hb
import subprocess
import csv
import pyogrio

import sys
print(sys.executable)

from osgeo import gdal
print(gdal.__version__)

from global_invest.coastal_protection import coastal_protection_initialization
from global_invest.coastal_protection import coastal_protection_functions


def coastal_protection(p):
    """
    Parent task for mangrove coastal protection.
    """
    p.cwon_input_ref_path = os.path.join(p.base_data_dir, 'coastal_protection', 'data_mangroves_2019.xlsx')
    p.coral_reef_ref_path = os.path.join(p.base_data_dir, 'coastal_protection', 'coral_reefs_annual_expected_benefit_nfamara.xlsx')


def gep_preprocess(p):
    """
    Preprocessing tasks are assumed NOT to be run by the user. Instead, it is assumed that the output of a preprocess
    task is an input to the actual model, saved at the canonical project attribute p.coastal_protection_input_path.
    These are preprocessing tasks are still provided for reference, but are not intended to be run directly by the user.
    We will "promote" the data outputed by a preprocess task to the base_data_dir provided to users.
    """
    pass # NYI



def gep_calculation(p):
    """ GEP calculation task for coastal protection."""
    # Define at least the primary output for the service, which for this project is gep_by_country_base_year.   
    service_results = {}
    p.results['coastal_protection'] = service_results  
    p.results['coastal_protection']['gep_by_country_base_year'] = os.path.join(p.project_dir, 'gep_by_country_base_year.csv')
            
    # Check if all results exist
    if hb.path_all_exist(list(service_results.values())):
        hb.log("All results already exist. Skipping GEP calculation for coastal protection.")
    else:
        hb.log("Starting GEP calculation for coastal protection.")
        
        # Optimization here,
        # p.gdf_countries = hb.read_vector(p.gdf_countries)
        p.gdf_countries = hb.read_vector(p.gdf_countries_vector_path)

        # 1. Read and process data
        df_mangrove_value = coastal_protection_functions.read_mangrove_values(p.cwon_input_ref_path)
        df_coral_reef_value = coastal_protection_functions.read_coral_reef_values(p.coral_reef_ref_path)

        # LEARNING POINT: I wasted lots of time not realizing the a how='right' operates differently than I expect. The left had IDs that were not in right under r264_id, but they thus had the a 
        # repeated ID in the r250. I had wrongly thought that the how='right' would only then return 1 row for each r250_id, but it actually a duplicate row repeated for each unique r264_id
        # even tho the r_250_id was the same. Thus, I had to drop the repeated ones.
        
        
        # Merge so it has all the good labels from the  
        df_gep_by_country_year_mangrove = hb.df_merge(p.gdf_countries, df_mangrove_value, how='inner', on='ee_r264_label')
        df_gep_by_country_year_mangrove = (
            df_gep_by_country_year_mangrove
            .groupby(["iso3_r250_label", "year"], as_index=False, dropna=False)["Value"]
            .sum()
        )
        df_gep_by_country_year_mangrove.dropna(subset=['iso3_r250_label'], inplace=True)

        df_gep_by_country_year_coral_reef = hb.df_merge(p.gdf_countries, df_coral_reef_value, how='inner', on='ee_r264_name')
        df_gep_by_country_year_coral_reef.dropna(subset=['coral_reef_value'], inplace=True)
        df_gep_by_country_year_coral_reef.drop_duplicates(subset=['iso3_r250_label', 'year','coral_reef_value'], inplace=True)


        df_gdp_inflation_deflator = coastal_protection_functions.get_inflation_deflator_multiplier(p.df_gdp_inflation_deflator_path,2012,2019)
        df_gep_by_country_year_coral_reef2019 = hb.df_merge(df_gep_by_country_year_coral_reef, df_gdp_inflation_deflator, how='left', on='ee_r264_label')
        df_gep_by_country_year_coral_reef2019['coral_reef_value'] = df_gep_by_country_year_coral_reef2019 ['coral_reef_value'] * df_gep_by_country_year_coral_reef2019['deflator_multiplier']
        df_gep_by_country_year_coral_reef2019['year'] = 2019
        df_gep_by_country_year_coral_reef = pd.concat(
            [df_gep_by_country_year_coral_reef, df_gep_by_country_year_coral_reef2019],
            ignore_index=True
        )

        df_gep_by_country_year_coral_reef = (
            df_gep_by_country_year_coral_reef
            .groupby(["iso3_r250_label", "year"], as_index=False, dropna=False)["coral_reef_value"]
            .sum()
        )


        # Rename value to coastal_protection_gep
        df_gep_by_country_year_mangrove = df_gep_by_country_year_mangrove.rename(
            columns={'Value': 'coastal_protection_gep_mangrove'}
        )

        df_gep_by_country_year_coral_reef = df_gep_by_country_year_coral_reef.rename(
            columns={'coral_reef_value': 'coastal_protection_gep_coral_reef'}
        )
        df_gep_by_country_year_coral_reef = df_gep_by_country_year_coral_reef[
            df_gep_by_country_year_coral_reef['year'] == 2019
        ]
    
        #merge the two dataframes on the common columns
        df_gep_by_country_year = pd.merge(
            df_gep_by_country_year_mangrove,
            df_gep_by_country_year_coral_reef,
            how='outer',
            on=['iso3_r250_label', 'year']
        )

        df_gep_by_country_year = df_gep_by_country_year.fillna(0)

        df_gep_by_country_year['coastal_protection_gep'] = df_gep_by_country_year['coastal_protection_gep_mangrove'] + df_gep_by_country_year['coastal_protection_gep_coral_reef']
        # df_gep_by_country_year['coastal_protection_gep'] = df_gep_by_country_year['coastal_protection_gep_coral_reef'] # for mangrove or coral reef only testing
        df_gep_by_country_year['Value'] = df_gep_by_country_year['coastal_protection_gep']
                # Drop repeated ids in df_countries
        ee_r264_to_250 = p.gdf_countries.copy()
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
            'area_code_M49',
            'area_code',
            'country',
        ]
        ee_r264_to_250.drop([i for i in ee_r264_to_250.columns if i not in cols_to_keep], axis=1, inplace=True, errors='ignore')
        # ee_r264_to_250 = ee_r264_to_250[cols_to_keep]

        df_gep_by_country_year = pd.merge(
            df_gep_by_country_year,
            ee_r264_to_250,
            how='left',
            on=['iso3_r250_label']
        )

        df_gep_by_country_base_year = df_gep_by_country_year.loc[df_gep_by_country_year['year'] == 2019].copy()
        
        # Write to CSVs
        hb.df_write(df_gep_by_country_base_year, p.results['coastal_protection']['gep_by_country_base_year'])   


        # Use geopandas to merge the df_gep_by_country_base_year with the  to get the country names and other attributes
        gdf_gep_by_country_base_year = hb.df_merge(p.gdf_countries_vector_simplified_path, df_gep_by_country_base_year, how='outer', on='ee_r264_id')


        gdf_gep_by_country_base_year.to_file(p.results['coastal_protection']['gep_by_country_base_year'].replace('.csv', '.gpkg'), driver='GPKG')

        # Then sum the values across all countries. 
        value_gep_base_year = df_gep_by_country_base_year['coastal_protection_gep'].sum() 
        
        hb.log(f"Total GEP value for base year 2019: {value_gep_base_year}")
        #Total GEP value for base year 2019: 73004611295.3582
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
    coastal_protection_initialization.build_gep_service_calculation_task_tree(p_temp)
    p_temp.set_all_tasks_to_skip_if_dir_exists()
    p_temp.execute()
    
    print(p_temp.results)
    pass
        
def gep_results_distribution(p):
    """Distribute the results of the GEP calculation."""
    # This task is intended to copy the results to the output directory.
    hb.log("Distributing GEP results...")
    
    for key, value in p.results['coastal_protection'].items():
        output_path = os.path.join(p.output_dir, key)
        hb.path_copy(value, output_path)
        hb.log(f"Distributed {key} to {output_path}")
    
    hb.log("GEP results distribution complete.")
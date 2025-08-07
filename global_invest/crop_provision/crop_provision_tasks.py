import os
import sys
import pandas as pd
import hazelbean as hb
import subprocess
import csv

from global_invest.crop_provision import crop_provision_initialization
from global_invest.crop_provision import crop_provision_functions
from global_invest.crop_provision import crop_provision_defaults

def crop_provision(p):
    """
    Parent task for commercial agriculture.
    """
    p.fao_input_ref_path = os.path.join('global_invest', 'crop_provision', 'Value_of_Production_E_All_Data.csv')
    p.cwon_crop_coefficients_ref_path = os.path.join('global_invest', 'crop_provision', "CWON2024_crop_coef.csv")

def gep_preprocess(p):
    """
    Preprocessing tasks are assumed NOT to be run by the user. Instead, it is assumed that the output of a preprocess
    task is an input to the actual model, saved at the canonical project attribute p.crop_provision_input_path.
    These are preprocessing tasks are still provided for reference, but are not intended to be run directly by the user.
    We will "promote" the data outputed by a preprocess task to the base_data_dir provided to users.
    """
    pass # NYI

def gep_calculation(p):
    """ GEP calculation task for commercial agriculture."""
    # Define at least the primary output for the service, which for this project is gep_by_country_base_year.   
    service_results = {}
    p.results['crop_provision'] = service_results  
    p.results['crop_provision']['gep_by_country_base_year'] = os.path.join(p.cur_dir, "gep_by_country_base_year.csv")
    
    # Optional additional results.
    p.results['crop_provision']['gep_by_country_year_crop'] = os.path.join(p.cur_dir, "gep_by_country_year_crop.csv")
    p.results['crop_provision']['gep_by_country_year'] = os.path.join(p.cur_dir, "gep_by_country_year.csv")
    p.results['crop_provision']['gep_by_year'] = os.path.join(p.cur_dir, "gep_by_year.csv")
            
    # Check if all results exist
    if hb.path_all_exist(list(service_results.values())):
        hb.log("All results already exist. Skipping GEP calculation for commercial agriculture.")
    else:
        hb.log("Starting GEP calculation for commercial agriculture.")
        
        # Optimization here,
        # p.gdf_countries = hb.read_vector(p.gdf_countries)
        p.gdf_countries = hb.read_vector(p.gdf_countries_simplified)

        # TODOOO: Could automate this by inspecting all ref_paths in a task. Or, could formalize a tasks inputs in p.inputs = {} like results.
        if not getattr(p, 'fao_input_path', None):
            # TODO: ProjectFlow Feature: make a similar p.load_path(). Get gets the file to storage, load gets it to memory. Also consider extending this to p.load_metadata() where, for eg tifs, it just loads the gdal ds, not the array
            p.fao_input_path = p.get_path(p.fao_input_ref_path)

        if not getattr(p, 'cwon_crop_coefficients_path', None):
            p.cwon_crop_coefficients_path = p.get_path(p.cwon_crop_coefficients_ref_path)

        if not getattr(p, 'crop_provision_subservices', None):
            p.commercial_attribute_subservices = crop_provision_defaults.DEFAULT_CROP_ITEMS

        # 1. Read and process data
        df_crop_value = crop_provision_functions.read_crop_values(p.fao_input_path, p.commercial_attribute_subservices)
        df_crop_coefs = crop_provision_functions.read_crop_coefs(p.cwon_crop_coefficients_path)

        df_gep_by_country_year_crop = crop_provision_functions.merge_crop_with_coefs(df_crop_value, df_crop_coefs)
        # String mangle the FAO M49 codes to integers.
        df_gep_by_country_year_crop['area_code_M49'] = df_gep_by_country_year_crop['area_code_M49'].str.replace('\'', '')
        df_gep_by_country_year_crop['area_code_M49'] = df_gep_by_country_year_crop['area_code_M49'].astype(int)
    
        replacements = {
            159: 156,  # China
            891: 688,  # Serbia and Montenegro
            200: 203,  # Czechoslovakia
            230: 231,  # Ethiopia PDR
            736: 729,  # Sudan (former)     
        }
        
        # Replace wrong codes in the m49
        df_gep_by_country_year_crop['area_code_M49'] = df_gep_by_country_year_crop['area_code_M49'].replace(replacements)    

        # LEARNING POINT: I wasted lots of time not realizing the a how='right' operates differently than I expect. The left had IDs that were not in right under r264_id, but they thus had the a 
        # repeated ID in the r250. I had wrongly thought that the how='right' would only then return 1 row for each r250_id, but it actually a duplicate row repeated for each unique r264_id
        # even tho the r_250_id was the same. Thus, I had to drop the repeated ones.
        
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
            'area_code_M49',
            'area_code',
            'country',
            'crop_code',
            'crop',
            'year',
            'rental_rate',
            'Value',
        ]
        ee_r264_to_250.drop([i for i in ee_r264_to_250.columns if i not in cols_to_keep], axis=1, inplace=True, errors='ignore')
        # ee_r264_to_250 = ee_r264_to_250[cols_to_keep]
        
        # Merge so it has all the good labels from the  
        df_gep_by_country_year_crop = hb.df_merge(ee_r264_to_250, df_gep_by_country_year_crop, how='right', left_on='iso3_r250_id', right_on='area_code_M49')
        
        # Rename value to crop_provision_gep
        df_gep_by_country_year_crop.rename(columns={'Value': 'crop_provision_gep'}, inplace=True)
        
        df_gep_by_country_year = crop_provision_functions.group_crops(df_gep_by_country_year_crop)

        df_gep_by_year = crop_provision_functions.group_countries(df_gep_by_country_year)
        
        df_gep_by_country_base_year = df_gep_by_country_year.loc[df_gep_by_country_year['year'] == 2019].copy()
        
        # Write to CSVs
        hb.df_write(df_gep_by_country_year_crop, p.results['crop_provision']['gep_by_country_year_crop'])
        hb.df_write(df_gep_by_country_year, p.results['crop_provision']['gep_by_country_year'])
        hb.df_write(df_gep_by_country_base_year, p.results['crop_provision']['gep_by_country_base_year'])   
        hb.df_write(df_gep_by_year, p.results['crop_provision']['gep_by_year'], handle_quotes='all')
        hb.df_write(df_gep_by_year, hb.replace_ext(p.results['crop_provision']['gep_by_year'], 'xlsx'), handle_quotes='all')
        
        # Use geopandas to merge the df_gep_by_country_base_year with the  to get the country names and other attributes
        gdf_gep_by_country_base_year = hb.df_merge(p.gdf_countries_simplified, df_gep_by_country_base_year, how='outer', left_on='ee_r264_id', right_on='ee_r264_id')
        gdf_gep_by_country_base_year.to_file(p.results['crop_provision']['gep_by_country_base_year'].replace('.csv', '.gpkg'), driver='GPKG')

        # Then sum the values across all countries. 
        value_gep_base_year = df_gep_by_country_base_year['crop_provision_gep'].sum()
        
        hb.log(f"Total GEP value for base year 2019: {value_gep_base_year}")
        
        return value_gep_base_year

def gep_result(p):
    """Display the results of the GEP calculation."""
    
    # Set the quarto path to wherever the current script is running. This means that the environment used needs to have quarto, which may not be true on e.g. codespaces.
    os.environ['QUARTO_PYTHON'] = sys.executable
    
    # Get the  list of current services run
    services_run = list(p.results.keys())
    
    # Get the last (presumably most recent, but this is hackish) service run
    service_label = services_run[-1]
    
    # Imply from the service name the file_path for the results_qmd
    module_root = hb.get_projectflow_module_root()
    

    results_qmd_path = os.path.join(module_root, service_label, f'{service_label}_results.qmd')    
    results_qmd_project_path = os.path.join(p.cur_dir, f'{service_label}_results.qmd')
    hb.create_directories(results_qmd_project_path)  # Ensure the directory exists   
    
    # Copy it to the project dir for cmd line processing (but will be removed again later because it makes confusion when people try to edit it and then rerun the script which won't of course update the results.)
    hb.path_copy(results_qmd_path, results_qmd_project_path)
    
    # Set the Current Directory to an environment-level variable that can be used by quarto.
    os.environ['PROJECTFLOW_ROOT'] = p.project_dir
    
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
    crop_provision_initialization.build_gep_service_calculation_task_tree(p_temp)
    p_temp.set_all_tasks_to_skip_if_dir_exists()
    p_temp.execute()
    
    print(p_temp.results)
    pass
        
def gep_results_distribution(p):
    """Distribute the results of the GEP calculation."""
    # This task is intended to copy the results to the output directory.
    hb.log("Distributing GEP results...")
    
    for key, value in p.results['crop_provision'].items():
        output_path = os.path.join(p.output_dir, key)
        hb.path_copy(value, output_path)
        hb.log(f"Distributed {key} to {output_path}")
    
    hb.log("GEP results distribution complete.")
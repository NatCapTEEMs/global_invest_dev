import os
import sys
import pandas as pd
import hazelbean as hb
import subprocess
import csv       
import numpy as np
from pathlib import Path

from global_invest.renewable_energy_provision import renewable_energy_provision_initialization
from global_invest.renewable_energy_provision import renewable_energy_provision_functions
# from global_invest.renewable_energy_provision import renewable_energy_provision_defaults

def renewable_energy_provision(p):
    """
    Parent task for commercial agriculture.
    """
    p.fao_input_ref_path = os.path.join('global_invest', 'renewable_energy_provision', 'Value_of_Production_E_All_Data.csv')
    p.cwon_crop_coefficients_ref_path = os.path.join('global_invest', 'renewable_energy_provision', "CWON2024_crop_coef.csv")

def gep_preprocess(p):
    """
    Preprocessing tasks are assumed NOT to be run by the user. Instead, it is assumed that the output of a preprocess
    task is an input to the actual model, saved at the canonical project attribute p.renewable_energy_provision_input_path.
    These are preprocessing tasks are still provided for reference, but are not intended to be run directly by the user.
    We will "promote" the data outputed by a preprocess task to the base_data_dir provided to users.
    """
    pass # NYI

def gep_calculation(p):
    """ GEP calculation task for commercial agriculture."""
    # Define at least the primary output for the service, which for this project is gep_by_country_base_year.   
    service_results = {}
    p.results['renewable_energy_provision'] = service_results  
    p.results['renewable_energy_provision']['gep_by_country_base_year'] = os.path.join(p.cur_dir, "renewable_energy_provision_gep_by_country_base_year.csv")
    
    # Add subservices if present
    subservice_results = {}
    p.results['renewable_energy_provision']['subservices'] = subservice_results    
    p.results['renewable_energy_provision']['subservices']['wind_energy_provision'] = {}
    p.results['renewable_energy_provision']['subservices']['solar_energy_provision'] = {}
    p.results['renewable_energy_provision']['subservices']['geothermal_energy_provision'] = {}
    
    p.results['renewable_energy_provision']['subservices']['wind_energy_provision'] ['gep_by_country_base_year'] = os.path.join(p.cur_dir, "wind_energy_provision_gep_by_country_base_year.csv")
    p.results['renewable_energy_provision']['subservices']['solar_energy_provision']['gep_by_country_base_year'] = os.path.join(p.cur_dir, "solar_energy_provision_gep_by_country_base_year.csv")
    p.results['renewable_energy_provision']['subservices']['geothermal_energy_provision']['gep_by_country_base_year'] = os.path.join(p.cur_dir, "geothermal_energy_provision_gep_by_country_base_year.csv")

    

            
    # Check if all results exist
    if hb.path_all_exist(list(service_results.values())):
        hb.log("All results already exist. Skipping GEP calculation for commercial agriculture.")
    else:
        hb.log("Starting GEP calculation for commercial agriculture.")

        print('Calculating Gross Ecosystem Product (GEP) for Renewable Energy Production.')
        # set dir
        # hb.create_shortcut
        data_dir = hb.Path(p.base_data_dir, 'global_invest', 'renewable_energy_provision')
        output_dir = hb.Path(p.cur_dir)

        #############
        # DEMAND SIDE
        #############

        # load data
        df_path = os.path.join(data_dir, 'IRENA_prod_by_country.csv')
        df = pd.read_csv(df_path)

        # aggregate generation technologies
        aggregated_df = (
            df.groupby(['Year', 'ISO3 code', 'Country', 'Group Technology'], 
                    as_index=False)['Electricity Generation (GWh)'].sum()
        )

        # Create df for each resource of interest
        geo_df = aggregated_df[aggregated_df['Group Technology'] == 'Geothermal energy']
        solar_df = aggregated_df[aggregated_df['Group Technology'] == 'Solar energy']
        wind_df = aggregated_df[aggregated_df['Group Technology'] == 'Wind energy']

        # create list of dfs 
        df_list = [wind_df, solar_df, geo_df]

        #############
        # SUPPLY SIDE
        #############

        # Load World Bank data

        wb_path = os.path.join(data_dir, 'WB_price_data.csv')
        wb_df = pd.read_csv(wb_path)

        # Convert Price from cents/kWh to USD/GWh
        wb_df['Price'] = wb_df['Price'] * 10000

        # rename columns for merge
        wb_df.rename(columns={'Economy ISO3' : 'ISO3 code', 'Economy Name' : 'Country', 'Price' : 'Price (USD/GWh)'}, inplace=True)

        #############
        # P * Q
        #############        
            

        # call P * Q merge
        gep_dfs = renewable_energy_provision_functions.merge_dfs(wb_df, df_list)
        # print(len(gep_dfs))

        ########################
        # NATURE'S CONTRIBUTIONS
        ########################

        # load resource rent data
        alpha_data_name = 'CWON_resource_rent_data.csv'
        alpha_path = os.path.join(data_dir, alpha_data_name)
        a_df = pd.read_csv(alpha_path)

        merge_cols = ['Country', 'Year']

        # concatenate list of dfs with p and q data
        combined_df = pd.concat(gep_dfs, ignore_index=True)

        # merge the concatenated df with the resource rent df 
        gep_df = combined_df.merge(a_df, on = merge_cols, how = 'inner')

        # gep calculation: gep = nat_contrib * P * Q
        gep_df['renewable_energy_provision_gep'] = gep_df['nat_contrib'] * gep_df['Price (USD/GWh)'] * gep_df['Electricity Generation (GWh)']
        gep_df.head()

        # filter to columns of interest
        filter_cols = ['ISO3 code', 'Country', 'Year', 'Group Technology', 'Price (USD/GWh)', 'Electricity Generation (GWh)', 'nat_contrib', 'renewable_energy_provision_gep']
        df_gep_by_country_base_year = gep_df[filter_cols]
        
        
        # Filter years
        df_gep_by_country_base_year = df_gep_by_country_base_year.loc[df_gep_by_country_base_year['Year'] == 2019] # 2019 only base
        
        # Drop rows where gep <= 0 (happens from nat_contrib calc)
        df_gep_by_country_base_year = df_gep_by_country_base_year.loc[df_gep_by_country_base_year['renewable_energy_provision_gep'] > 0]
        # Rename to iso3_r250_label
        df_gep_by_country_base_year = df_gep_by_country_base_year.rename(columns={'ISO3 code': 'iso3_r250_label'})
        
        # Merge in ee spec.
        p.df_countries = hb.df_read(p.df_countries)
        
        
        
        # Drop repeated ids in df_countries.
        # TODO Need to make this more robust by having a r250 dataset as the starting point.
        ee_r264_to_250 = p.df_countries.copy()
        ee_r264_to_250 = ee_r264_to_250[ee_r264_to_250['ee_r264_label'] == ee_r264_to_250['iso3_r250_label']]
        
        df_gep_by_country_base_year = hb.df_merge(ee_r264_to_250, df_gep_by_country_base_year, left_on='iso3_r250_label', right_on='iso3_r250_label', how='left')
        
        
        
        hb.df_write(df_gep_by_country_base_year, p.results['renewable_energy_provision']['gep_by_country_base_year'], index=False)
        
        # Filter and split by subservice
        df_dict = renewable_energy_provision_functions.filter_and_split_by_resource(df_gep_by_country_base_year)
        
        for subservice_key, subservice_results in p.results['renewable_energy_provision']['subservices'].items():
            modkey = subservice_key.split('_')[0].title() + ' energy' # e.g. 'wind_energy_provision' -> 'wind'
            df_cur = df_dict[modkey]
            
            output_path = subservice_results['gep_by_country_base_year']
            hb.df_write(df_cur, output_path, index=False)
            
            
        # Use geopandas to merge the df_gep_by_country_base_year with the  to get the country names and other attributes
        gdf_gep_by_country_base_year = hb.df_merge(p.gdf_countries_simplified, df_gep_by_country_base_year, how='outer', left_on='ee_r264_id', right_on='ee_r264_id')
        gdf_gep_by_country_base_year.to_file(p.results['renewable_energy_provision']['gep_by_country_base_year'].replace('.csv', '.gpkg'), driver='GPKG')

        # Then sum the values across all countries. 
        value_gep_base_year = df_gep_by_country_base_year['renewable_energy_provision_gep'].sum()
        
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
    renewable_energy_provision_initialization.build_gep_service_calculation_task_tree(p_temp)
    p_temp.set_all_tasks_to_skip_if_dir_exists()
    p_temp.execute()
    
    print(p_temp.results)
    pass
        
def gep_results_distribution(p):
    """Distribute the results of the GEP calculation."""
    # This task is intended to copy the results to the output directory.
    hb.log("Distributing GEP results...")
    
    for key, value in p.results['renewable_energy_provision'].items():
        output_path = os.path.join(p.output_dir, key)
        hb.path_copy(value, output_path)
        hb.log(f"Distributed {key} to {output_path}")
    
    hb.log("GEP results distribution complete.")
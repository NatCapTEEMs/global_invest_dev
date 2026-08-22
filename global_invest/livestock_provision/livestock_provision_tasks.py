import os
import sys
import pandas as pd
import hazelbean as hb
from global_invest import utilities
import subprocess
import csv

from global_invest.livestock_provision import livestock_provision_initialize
from global_invest.livestock_provision import livestock_provision_functions

# The service owner's item selection, ported verbatim from the committed active list in
# Urgendalai/gep_livestock_provision (fao_data_livestock.py, livestock_value_codes): 65 FAO item
# codes including the milk items the name list below lacks. Code-based filtering is also robust
# to FAO's item-name revisions. PROVISIONAL until the owner confirms the list is current; his
# script filters value on Element Code 152 where this module uses 57 -- an open method question
# recorded in the tracker, not silently changed here.
OWNER_LIVESTOCK_VALUE_CODES = [
    1012, 1017, 1018, 1019, 1020, 1025, 1032, 1035, 1036, 1037,
    1044, 1055, 1058, 1062, 1069, 1070, 1073, 1077, 1080, 1084,
    1087, 1089, 1091, 1094, 1097, 1098, 1108, 1111, 1120, 1122,
    1124, 1127, 1128, 1129, 1130, 1137, 1141, 1144, 1151, 1154,
    1158, 1161, 1163, 1166, 1176, 1182, 1183, 1185,
    2044, 867, 868, 869, 882, 944, 947, 948, 949, 951, 972, 977,
    978, 979, 982, 987, 995,
]

DEFAULT_LIVESTOCK_ITEMS = [
    "Meat of asses, fresh or chilled (indigenous)",
    "Meat of buffalo, fresh or chilled (indigenous)",
    "Meat of camels, fresh or chilled (indigenous)",
    "Meat of chickens, fresh or chilled (indigenous)",
    "Meat of ducks, fresh or chilled (indigenous)",
    "Meat of geese, fresh or chilled (indigenous)",
    "Meat of goat, fresh or chilled (indigenous)",
    "Meat of mules, fresh or chilled (indigenous)",
    "Meat of other domestic camelids, fresh or chilled (indigenous)",
    "Meat of pig with the bone, fresh or chilled (indigenous)",
    "Meat of pigeons and other birds n.e.c., fresh, chilled or frozen (indigenous)",
    "Meat of rabbits and hares, fresh or chilled (indigenous)",
    "Meat of sheep, fresh or chilled (indigenous)",
    "Meat of turkeys, fresh or chilled (indigenous)",
    "Horse meat, fresh or chilled (indigenous)",
    "Hen eggs in shell, fresh",
    "Eggs from other birds in shell, fresh, n.e.c.",
    "Game meat, fresh, chilled or frozen",
    "Meat of cattle with the bone, fresh or chilled",
    "Meat of chickens, fresh or chilled",
    "Meat of goat, fresh or chilled",
    "Meat of pig with the bone, fresh or chilled",
    "Meat of sheep, fresh or chilled",
    "Meat of turkeys, fresh or chilled",
    "Meat of camels, fresh or chilled",
    "Horse meat, fresh or chilled",
    "Meat of rabbits and hares, fresh or chilled",
    "Natural honey",
    "Other meat n.e.c. (excluding mammals), fresh, chilled or frozen",
]



def publish_inputs(p):
    """Every task's first line: the FAO livestock-production valuation's es_config row (defaults layer -- a caller-set value wins)
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
    p.fao_input_ref_path = os.path.join('global_invest', 'livestock_provision', 'Value_of_Production_E_All_Data.csv')
    p.cwon_crop_coefficients_ref_path = os.path.join('global_invest', 'livestock_provision', "CWON2024_crop_coef.csv")

def gep_preprocess(p):
    """
    Preprocessing tasks are assumed NOT to be run by the user. Instead, it is assumed that the output of a preprocess
    task is an input to the actual model, saved at the canonical project attribute p.livestock_provision_input_path.
    These are preprocessing tasks are still provided for reference, but are not intended to be run directly by the user.
    We will "promote" the data outputed by a preprocess task to the base_data_dir provided to users.
    """
    publish_inputs(p)
    pass # NYI

def gep_calculation(p):
    """ GEP calculation task for livestock provision."""
    publish_inputs(p)
    # Define at least the primary output for the service, which for this project is gep_by_country_base_year.   
    service_results = {}
    p.results['livestock_provision'] = service_results  
    p.results['livestock_provision']['gep_by_country_base_year'] = os.path.join(p.cur_dir, "gep_by_country_base_year.csv")
    
    # Optional additional results.
    p.results['livestock_provision']['gep_by_country_year_crop'] = os.path.join(p.cur_dir, "gep_by_country_year_crop.csv")
    p.results['livestock_provision']['gep_by_country_year'] = os.path.join(p.cur_dir, "gep_by_country_year.csv")
    p.results['livestock_provision']['gep_by_year'] = os.path.join(p.cur_dir, "gep_by_year.csv")
            
    # Check if all results exist
    if hb.path_all_exist(list(service_results.values())):
        hb.log("All results already exist. Skipping GEP calculation for livestock provision.")
    else:
        hb.log("Starting GEP calculation for livestock provision.")
        
        # Optimization here,
        # p.gdf_countries = hb.read_vector(p.gdf_countries)
        p.gdf_countries = hb.read_vector(p.gdf_countries_simplified)

        # TODOOO: Could automate this by inspecting all ref_paths in a task. Or, could formalize a tasks inputs in p.inputs = {} like results.
        if not getattr(p, 'fao_input_path', None):
            # TODO: ProjectFlow Feature: make a similar p.load_path(). Get gets the file to storage, load gets it to memory. Also consider extending this to p.load_metadata() where, for eg tifs, it just loads the gdal ds, not the array
            p.fao_input_path = p.get_path(p.fao_input_ref_path)

        if not getattr(p, 'cwon_crop_coefficients_path', None):
            p.cwon_crop_coefficients_path = p.get_path(p.cwon_crop_coefficients_ref_path)

        if not getattr(p, 'livestock_provision_subservices', None):
            p.commercial_attribute_subservices = OWNER_LIVESTOCK_VALUE_CODES

        # 1. Read and process data
        df_crop_value = livestock_provision_functions.read_crop_values(p.fao_input_path, p.commercial_attribute_subservices)
        df_crop_coefs = livestock_provision_functions.read_crop_coefs(p.cwon_crop_coefficients_path)

        df_gep_by_country_year_crop = livestock_provision_functions.merge_crop_with_coefs(df_crop_value, df_crop_coefs)
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

        # One row per country: r264 splits large countries, so the correspondence is
        # collapsed before the join.
        ee_r264_to_250 = utilities.collapse_countries_to_r250(
            p.df_countries, keep_columns=['area_code_M49', 'area_code', 'country', 'crop_code', 'crop', 'year', 'rental_rate', 'livestock_provision_gep'])
        
        # Merge so it has all the good labels from the  
        df_gep_by_country_year_crop = hb.df_merge(ee_r264_to_250, df_gep_by_country_year_crop, how='right', left_on='iso3_r250_id', right_on='area_code_M49')
        
        # Rename Valu to livestock_provision_gep
        # df_gep_by_country_year_crop.rename(columns={'livestock_provision_gep': 'livestock_provision_gep'}, inplace=True)
        
        df_gep_by_country_year = livestock_provision_functions.group_crops(df_gep_by_country_year_crop)

        df_gep_by_year = livestock_provision_functions.group_countries(df_gep_by_country_year)
        
        df_gep_by_country_base_year = df_gep_by_country_year.loc[df_gep_by_country_year['year'] == 2019].copy()
        
        # Write to CSVs
        hb.df_write(df_gep_by_country_year_crop, p.results['livestock_provision']['gep_by_country_year_crop'])
        hb.df_write(df_gep_by_country_year, p.results['livestock_provision']['gep_by_country_year'])
        hb.df_write(df_gep_by_country_base_year, p.results['livestock_provision']['gep_by_country_base_year'])   
        hb.df_write(df_gep_by_year, p.results['livestock_provision']['gep_by_year'], handle_quotes='all')
        hb.df_write(df_gep_by_year, hb.replace_ext(p.results['livestock_provision']['gep_by_year'], 'xlsx'), handle_quotes='all')
        
        # Use geopandas to merge the df_gep_by_country_base_year with the  to get the country names and other attributes
        gdf_gep_by_country_base_year = hb.df_merge(p.gdf_countries_simplified, df_gep_by_country_base_year, how='outer', left_on='ee_r264_id', right_on='ee_r264_id')
        gdf_gep_by_country_base_year.to_file(p.results['livestock_provision']['gep_by_country_base_year'].replace('.csv', '.gpkg'), driver='GPKG')

        # Then sum the values across all countries. 
        value_gep_base_year = df_gep_by_country_base_year['livestock_provision_gep'].sum()
        
        hb.log(f"Total GEP value for base year 2019: {value_gep_base_year}")
        
        return value_gep_base_year

def gep_result(p):
    """Display the results of the GEP calculation."""
    publish_inputs(p)
    
    # Set the quarto path to wherever the current script is running. This means that the environment used needs to have quarto, which may not be true on e.g. codespaces.
    os.environ['QUARTO_PYTHON'] = sys.executable
    
    # Get the  list of current services run
    services_run = list(p.results.keys())
    
    # Get the last (presumably most recent, but this is hackish) service run
    service_label = services_run[-1]
    
    # Additional groupbys = []
    
    # Imply from the service name the file_path for the results_qmd
    module_root = hb.get_projectflow_module_root()

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
    publish_inputs(p)
    
    # Learn the paths by creating a temp task treep
    p_temp = hb.ProjectFlow()
    livestock_provision_initialize.build_gep_service_calculation_task_tree(p_temp)
    p_temp.set_all_tasks_to_skip_if_dir_exists()
    p_temp.execute()
    
    print(p_temp.results)
    pass
        
def gep_results_distribution(p):
    """Distribute the results of the GEP calculation."""
    publish_inputs(p)
    # This task is intended to copy the results to the output directory.
    hb.log("Distributing GEP results...")
    
    for key, value in p.results['livestock_provision'].items():
        output_path = os.path.join(p.output_dir, key)
        hb.path_copy(value, output_path)
        hb.log(f"Distributed {key} to {output_path}")
    
    hb.log("GEP results distribution complete.")
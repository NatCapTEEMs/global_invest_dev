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

        if not getattr(p, 'livestock_provision_subservices', None):
            p.commercial_attribute_subservices = OWNER_LIVESTOCK_VALUE_CODES

        df_crop_value = livestock_provision_functions.read_crop_values(p.fao_input_path, p.commercial_attribute_subservices)
        df_crop_coefs = livestock_provision_functions.read_crop_coefs(p.cwon_crop_coefficients_path)

        df_gep_by_country_year_crop = livestock_provision_functions.merge_crop_with_coefs(df_crop_value, df_crop_coefs)
        df_gep_by_country_year_crop = livestock_provision_functions.normalize_m49_codes(df_gep_by_country_year_crop)
        df_gep_by_country_year_crop = livestock_provision_functions.attach_countries(
            df_gep_by_country_year_crop, p.df_countries)

        df_gep_by_country_year = livestock_provision_functions.group_crops(df_gep_by_country_year_crop)

        df_gep_by_year = livestock_provision_functions.group_countries(df_gep_by_country_year)

        df_gep_by_country_base_year = df_gep_by_country_year.loc[
            df_gep_by_country_year['year'] == int(p.gep_base_year)].copy()

        # Write to CSVs
        hb.df_write(df_gep_by_country_year_crop, p.results['livestock_provision']['gep_by_country_year_crop'])
        hb.df_write(df_gep_by_country_year, p.results['livestock_provision']['gep_by_country_year'])
        hb.df_write(df_gep_by_country_base_year, p.results['livestock_provision']['gep_by_country_base_year'])   
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
    utilities.render_service_results(p)


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
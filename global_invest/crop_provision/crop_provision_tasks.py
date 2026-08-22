import os
import sys
import pandas as pd
import hazelbean as hb
from global_invest import utilities
import subprocess
import csv

from global_invest.crop_provision import crop_provision_initialize
from global_invest.crop_provision import crop_provision_functions


# Project content/config (was crop_provision_defaults.py; the template keeps content in the task module).

DEFAULT_CROP_ITEMS = [
    "Apples",
    "Apricots",
    "Barley",
    "Beans, dry",
    "Broad beans and horse beans, green",
    "Cabbages",
    "Cantaloupes and other melons",
    "Carrots and turnips",
    "Cauliflowers and broccoli",
    "Cherries",
    "Chestnuts, in shell",
    "Chillies and peppers, green (Capsicum spp. and Pimenta spp.)",
    "Cucumbers and gherkins",
    "Dates",
    "Eggplants (aubergines)",
    "Figs",
    "Grapes",
    "Green garlic",
    "Leeks and other alliaceous vegetables",
    "Lemons and limes",
    "Lettuce and chicory",
    "Maize (corn)",
    "Oats",
    "Okra",
    "Olives",
    "Onions and shallots, dry (excluding dehydrated)",
    "Onions and shallots, green",
    "Oranges",
    "Other beans, green",
    "Other fruits, n.e.c.",
    "Other nuts (excluding wild edible nuts and groundnuts), in shell, n.e.c.",
    "Other vegetables, fresh n.e.c.",
    "Peaches and nectarines",
    "Pears",
    "Peas, green",
    "Plums and sloes",
    "Potatoes",
    "Pumpkins, squash and gourds",
    "Quinces",
    "Raw milk of cattle",
    "Raw milk of goats",
    "Raw milk of sheep",
    "Rice",
    "Rye",
    "Shorn wool, greasy, including fleece-washed shorn wool",
    "Soya beans",
    "Spinach",
    "Sunflower seed",
    "Tangerines, mandarins, clementines",
    "Tomatoes",
    "Unmanufactured tobacco",
    "Watermelons",
    "Wheat",
    "Non Food",
    "Oilcrops Primary",
    "Roots and Tubers, Total",
    "Sugar Crops Primary",
    "Artichokes",
    "Broad beans and horse beans, dry",
    "Chick peas, dry",
    "Chillies and peppers, dry (Capsicum spp., Pimenta spp.), raw",
    "Lentils, dry",
    "Other citrus fruit, n.e.c.",
    "Other pulses n.e.c.",
    "Peas, dry",
    "Seed cotton, unginned",
    "Sorghum",
    "Triticale",
    "Vetches",
    "Bananas",
    "Cassava, fresh",
    "Coffee, green",
    "Groundnuts, excluding shelled",
    "Millet",
    "Pineapples",
    "Sweet potatoes",
    "Mangoes, guavas and mangosteens",
    "Yams",
    "Anise, badian, coriander, cumin, caraway, fennel and juniper berries, raw",
    "Asparagus",
    "Beeswax",
    "Canary seed",
    "Linseed",
    "MatÃ© leaves",
    "Pomelos and grapefruits",
    "Strawberries",
    "Sugar cane",
    "Tea leaves",
    "Walnuts, in shell",
    "Mushrooms and truffles",
    "Almonds, in shell",
    "Avocados",
    "Cow peas, dry",
    "Currants",
    "Green corn (maize)",
    "Hop cones",
    "Kiwi fruit",
    "Lupins",
    "Other berries and fruits of the genus vaccinium n.e.c.",
    "Papayas",
    "Persimmons",
    "Pistachios, in shell",
    "Rape or colza seed",
    "Raspberries",
    "Safflower seed",
    "Gooseberries",
    "Meat of ducks, fresh or chilled",
    "Meat of geese, fresh or chilled",
    "Other oil seeds, n.e.c.",
    "Poppy seed",
    "Sour cherries",
    "Sugar beet",
    "Hazelnuts, in shell",
    "Silk-worm cocoons suitable for reeling",
    "Areca nuts",
    "Coconuts, in shell",
    "Ginger, raw",
    "Jute, raw or retted",
    "Meat of buffalo, fresh or chilled",
    "Raw milk of buffalo",
    "Sesame seed",
    "String beans",
    "Buckwheat",
    "Cereals n.e.c.",
    "Cranberries",
    "Chicory roots",
    "Cashew nuts, in shell",
    "Cocoa beans",
    "Edible roots and tubers with high starch or inulin content, n.e.c., fresh",
    "Yautia",
    "Taro",
    "Mustard seed",
    "Nutmeg, mace, cardamoms, raw",
    "Other stimulant, spice and aromatic crops, n.e.c.",
    "Natural rubber in primary forms",
    "Pepper (Piper spp.), raw",
    "Plantains and cooking bananas",
    "Quinoa",
    "Brazil nuts, in shell",
    "Cashewapple",
    "Castor oil seeds",
    "Kenaf, and other textile bast fibres, raw or retted",
    "Oil palm fruit",
    "Other fibre crops, raw, n.e.c.",
    "Other tropical fruits, n.e.c.",
    "Ramie, raw or retted",
    "Sisal, raw",
    "Tung nuts",
    "Peppermint, spearmint",
    "Bambara beans, dry",
    "Fonio",
    "Karite nuts (sheanuts)",
    "Pigeon peas, dry",
    "Blueberries",
    "Meat of asses, fresh or chilled",
    "Meat of mules, fresh or chilled",
    "Melonseed",
    "Other stone fruits",
    "Kola nuts",
    "Mixed grain",
    "Locust beans (carobs)",
    "Meat of pigeons and other birds n.e.c., fresh, chilled or frozen",
    "Abaca, manila hemp, raw",
    "Other meat of mammals, fresh or chilled",
    "Agave fibres, raw, n.e.c.",
    "Cinnamon and cinnamon-tree flowers, raw",
    "Cloves (whole stems), raw",
    "Vanilla, raw",
    "Raw milk of camel",
    "Pyrethrum, dried flowers",
    "Meat of other domestic camelids, fresh or chilled",
    "True hemp, raw or retted",
    "Kapok fruit",
    "Other pome fruits",
    "Hempseed",
    "Meat of cattle with the bone, fresh or chilled (indigenous)",
    "Meat of other domestic rodents, fresh or chilled",
    "Other sugar crops n.e.c.",
]


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
    p.fao_input_ref_path = os.path.join('global_invest', 'crop_provision', 'Value_of_Production_E_All_Data.csv')
    p.cwon_crop_coefficients_ref_path = os.path.join('global_invest', 'crop_provision', "CWON2024_crop_coef.csv")

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
            p.commercial_attribute_subservices = DEFAULT_CROP_ITEMS

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

        # One row per country: r264 splits large countries, so the correspondence is
        # collapsed before the join.
        ee_r264_to_250 = utilities.collapse_countries_to_r250(
            p.df_countries, keep_columns=['area_code_M49', 'area_code', 'country', 'crop_code', 'crop', 'year', 'rental_rate', 'Value'])
        
        # Merge so it has all the good labels from the  
        df_gep_by_country_year_crop = hb.df_merge(ee_r264_to_250, df_gep_by_country_year_crop, how='right', left_on='iso3_r250_id', right_on='area_code_M49')
        
        # Rename value to crop_provision_gep
        df_gep_by_country_year_crop.rename(columns={'Value': 'crop_provision_gep'}, inplace=True)
        # FAOSTAT ships Value in thousand USD. The library convention is plain USD, so convert
        # here, once, before any grouping (the reference table stays in the source's units).
        df_gep_by_country_year_crop['crop_provision_gep'] *= 1000.0
        
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
    publish_inputs(p)
    
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
    publish_inputs(p)
    
    # Learn the paths by creating a temp task treep
    p_temp = hb.ProjectFlow()
    crop_provision_initialize.build_gep_service_calculation_task_tree(p_temp)
    p_temp.set_all_tasks_to_skip_if_dir_exists()
    p_temp.execute()
    
    print(p_temp.results)
    pass
        
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
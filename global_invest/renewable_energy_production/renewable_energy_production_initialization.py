import pandas as pd
import hazelbean as hb

from global_invest.renewable_energy_production import renewable_energy_production_tasks

def initialize_paths(p):
    p.df_countries = pd.read_csv(p.df_countries_csv_path)  
    
    # Notice optimization here: the GDFs are still just path_strings. hb.read_vector takes the string as an input and converts it to a GeoDataFrame when needed.
    p.gdf_countries = p.gdf_countries_vector_path 
    p.gdf_countries_simplified = p.gdf_countries_vector_simplified_path 
    
    # p.gdf_countries = hb.read_vector(p.gdf_countries_vector_path)  # Read the vector file for the countries.
    # p.countries_simplified_gdf = hb.read_vector(p.countries_simplified_vector_path)  # Read the vector file for the countries.

def build_gep_service_calculation_task_tree(p):
    """Build the default task tree for commercial agriculture."""
    p.renewable_energy_production_task = p.add_task(renewable_energy_production_tasks.renewable_energy_production)
    p.renewable_energy_production_gep_calculation_task = p.add_task(renewable_energy_production_tasks.gep_calculation, parent=p.renewable_energy_production_task)  
    return p

def build_gep_service_task_tree(p):
    """If you just want to load results, eg for reporting, this task tree inspects a different task tree and to learn paths and then loads results."""
    
    p = build_gep_service_calculation_task_tree(p)
    p.renewable_energy_production_gep_result_task = p.add_task(renewable_energy_production_tasks.gep_result, parent=p.renewable_energy_production_task)   

    
def build_gep_task_tree(p):
    """
    Build the default task tree forthe GEP application of commercial agriculture. In this case, it's very similar to the standard task tree
    but i've included it here for consistency with other models.
    """
    p.renewable_energy_production_task = p.add_task(renewable_energy_production_tasks.renewable_energy_production)
    p.renewable_energy_production_gep_preprocess_task = p.add_task(renewable_energy_production_tasks.gep_preprocess, parent=p.renewable_energy_production_task)  
    p.renewable_energy_production_gep_calculation_task = p.add_task(renewable_energy_production_tasks.gep_calculation, parent=p.renewable_energy_production_task)  
    p.renewable_energy_production_gep_result_task = p.add_task(renewable_energy_production_tasks.gep_result, parent=p.renewable_energy_production_task)   
    p.renewable_energy_production_gep_results_distribution_task = p.add_task(renewable_energy_production_tasks.gep_results_distribution, parent=p.renewable_energy_production_task)      
    return p
    

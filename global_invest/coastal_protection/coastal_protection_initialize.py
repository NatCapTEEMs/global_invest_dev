import pandas as pd
import hazelbean as hb

from global_invest.coastal_protection import coastal_protection_tasks

def initialize_paths(p):
    p.df_countries = pd.read_csv(p.df_countries_csv_path)  
    
    # Notice optimization here: the GDFs are still just path_strings. hb.read_vector takes the string as an input and converts it to a GeoDataFrame when needed.
    p.gdf_countries = p.gdf_countries_vector_path 
    p.gdf_countries_simplified = p.gdf_countries_vector_simplified_path 
    
    # p.gdf_countries = hb.read_vector(p.gdf_countries_vector_path)  # Read the vector file for the countries.
    # p.countries_simplified_gdf = hb.read_vector(p.countries_simplified_vector_path)  # Read the vector file for the countries.

def build_gep_service_calculation_task_tree(p):
    """Build the default task tree for commercial agriculture."""
    p.coastal_protection_task = p.add_task(coastal_protection_tasks.coastal_protection)
    p.coastal_protection_gep_calculation_task = p.add_task(coastal_protection_tasks.gep_calculation, parent=p.coastal_protection_task)  
    return p

def build_gep_service_task_tree(p):
    """If you just want to load results, eg for reporting, this task tree inspects a different task tree and to learn paths and then loads results."""
    
    
    # QUESTION!!!! If a task truly already inspects itself to not rerun, what's the difference between loading and just executing the tree on 
    # an existing project? The difference is that load will do more error checking and FAIL rather than recalculate if it didn't find, also reporting
    # that it didn't find it and giving information about how to put the data in so it does find it in the base data or a manually-built project data.
    # I might want to have methods for automatically putting an archive into the right spot and also extended functionality for finding results in base_data
    # and functionality for promoting project results to base data per the new documentation in ee_dev.
    # Actually, maybe it's just that load_results is more useful for notebooks?
    
    p = build_gep_service_calculation_task_tree(p)
    p.coastal_protection_gep_result_task = p.add_task(coastal_protection_tasks.gep_result, parent=p.coastal_protection_task)   

    
def build_gep_task_tree(p):
    """
    Build the default task tree forthe GEP application. In this case, it's very similar to the standard task tree
    but i've included it here for consistency with other models.
    """
    p.coastal_protection_task = p.add_task(coastal_protection_tasks.coastal_protection)
    p.coastal_protection_gep_preprocess_task = p.add_task(coastal_protection_tasks.gep_preprocess, parent=p.coastal_protection_task)  
    p.coastal_protection_gep_calculation_task = p.add_task(coastal_protection_tasks.gep_calculation, parent=p.coastal_protection_task)  
    p.coastal_protection_gep_result_task = p.add_task(coastal_protection_tasks.gep_result, parent=p.coastal_protection_task)   
    p.coastal_protection_gep_results_distribution_task = p.add_task(coastal_protection_tasks.gep_results_distribution, parent=p.coastal_protection_task)      
    return p
    

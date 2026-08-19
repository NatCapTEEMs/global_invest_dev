import pandas as pd
import hazelbean as hb

from global_invest import utilities
from global_invest.extractive_materials_provision import extractive_materials_provision_tasks

def initialize_paths(p):
    """One source of truth for the inputs (shared country block + service data, get_path references)."""
    utilities.initialize_country_paths(p, simplified='30sec')

    # World Bank series staged into base_data from the drive's submissions folder (see base_data CHANGELOG):
    # mineral rents (% of GDP) and GDP (current USD).
    p.wb_mineral_input_ref_path = p.get_path('global_invest', 'extractive_materials_provision', 'API_NY.GDP.MINR.RT.ZS_DS2_en_csv_v2_6559.csv')
    p.wb_GDP_ref_path = p.get_path('global_invest', 'extractive_materials_provision', 'API_NY.GDP.MKTP.CD_DS2_en_csv_v2_130122.csv')

def build_gep_service_calculation_task_tree(p):
    """Build the default task tree for commercial agriculture."""
    p.extractive_materials_provision_task = p.add_task(extractive_materials_provision_tasks.extractive_materials_provision)
    p.extractive_materials_provision_gep_calculation_task = p.add_task(extractive_materials_provision_tasks.gep_calculation, parent=p.extractive_materials_provision_task)  
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
    p.extractive_materials_provision_gep_result_task = p.add_task(extractive_materials_provision_tasks.gep_result, parent=p.extractive_materials_provision_task)   

    
def build_gep_task_tree(p):
    """
    Build the default task tree forthe GEP application of commercial agriculture. In this case, it's very similar to the standard task tree
    but i've included it here for consistency with other models.
    """
    p.extractive_materials_provision_task = p.add_task(extractive_materials_provision_tasks.extractive_materials_provision)
    p.extractive_materials_provision_gep_preprocess_task = p.add_task(extractive_materials_provision_tasks.gep_preprocess, parent=p.extractive_materials_provision_task)  
    p.extractive_materials_provision_gep_calculation_task = p.add_task(extractive_materials_provision_tasks.gep_calculation, parent=p.extractive_materials_provision_task)  
    p.extractive_materials_provision_gep_result_task = p.add_task(extractive_materials_provision_tasks.gep_result, parent=p.extractive_materials_provision_task)   
    p.extractive_materials_provision_gep_results_distribution_task = p.add_task(extractive_materials_provision_tasks.gep_results_distribution, parent=p.extractive_materials_provision_task)      
    return p
    

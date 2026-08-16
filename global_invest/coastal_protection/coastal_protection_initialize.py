import pandas as pd
import hazelbean as hb

from global_invest import utilities
from global_invest.coastal_protection import coastal_protection_tasks

def initialize_paths(p):
    """One source of truth for the inputs (shared country block + service data, get_path references)."""
    utilities.initialize_country_paths(p, simplified='30sec')

    # Service data staged into base_data from the drive's submissions folders (see base_data CHANGELOG).
    p.cwon_input_ref_path = p.get_path('coastal_protection', 'data_mangroves_2019.xlsx')
    p.coral_reef_ref_path = p.get_path('coastal_protection', 'coral_reefs_annual_expected_benefit_nfamara.xlsx')
    # The drive's submissions folder spells this 'gdp_inflation_delator' (sic); staged locally under
    # the corrected name. Filename case is exact so it resolves on case-sensitive filesystems too.
    p.df_gdp_inflation_deflator_path = p.get_path('gdp_inflation_deflator', 'GDP_Inflation_deflator.xlsx')

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
    

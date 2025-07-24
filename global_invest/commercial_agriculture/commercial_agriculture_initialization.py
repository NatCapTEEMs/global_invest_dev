import pandas as pd
import hazelbean as hb

from global_invest.commercial_agriculture import commercial_agriculture_tasks

def initialize_paths(p):
    p.ee_r264_df = pd.read_csv(p.countries_csv_path)  

def build_gep_service_calculation_task_tree(p):
    """Build the default task tree for commercial agriculture."""
    p.commercial_agriculture_task = p.add_task(commercial_agriculture_tasks.commercial_agriculture)
    p.commercial_agriculture_gep_calculation_task = p.add_task(commercial_agriculture_tasks.gep_calculation, parent=p.commercial_agriculture_task)  
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
    p.commercial_agriculture_gep_result_task = p.add_task(commercial_agriculture_tasks.gep_result, parent=p.commercial_agriculture_task)   

    
def build_gep_task_tree(p):
    """
    Build the default task tree forthe GEP application of commercial agriculture. In this case, it's very similar to the standard task tree
    but i've included it here for consistency with other models.
    """
    p.commercial_agriculture_task = p.add_task(commercial_agriculture_tasks.commercial_agriculture)
    p.commercial_agriculture_gep_preprocess_task = p.add_task(commercial_agriculture_tasks.gep_preprocess, parent=p.commercial_agriculture_task)  
    p.commercial_agriculture_gep_calculation_task = p.add_task(commercial_agriculture_tasks.gep_calculation, parent=p.commercial_agriculture_task)  
    p.commercial_agriculture_gep_result_task = p.add_task(commercial_agriculture_tasks.gep_result, parent=p.commercial_agriculture_task)   
    p.commercial_agriculture_gep_results_distribution_task = p.add_task(commercial_agriculture_tasks.gep_results_distribution, parent=p.commercial_agriculture_task)      
    return p
    

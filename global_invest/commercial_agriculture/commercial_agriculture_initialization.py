import pandas as pd
import hazelbean as hb

from global_invest.commercial_agriculture import commercial_agriculture_tasks

def initialize_paths(p):
    p.ee_r264_df = pd.read_csv(p.countries_csv_path)  

def build_standard_task_tree(p):
    """Build the default task tree for commercial agriculture."""
    p.commercial_agriculture_task = p.add_task(commercial_agriculture_tasks.commercial_agriculture)
    # p.commercial_agriculture_gep_preprocess_task = p.add_task(commercial_agriculture_tasks.gep_preprocess, parent=p.commercial_agriculture_task)  
    p.commercial_agriculture_gep_calculation_task = p.add_task(commercial_agriculture_tasks.gep_calculation, parent=p.commercial_agriculture_task)  
    # p.commercial_agriculture_gep_result_task = p.add_task(commercial_agriculture_tasks.gep_result, parent=p.commercial_agriculture_task)   
    # p.commercial_agriculture_gep_results_distribution_task = p.add_task(commercial_agriculture_tasks.gep_results_distribution, parent=p.commercial_agriculture_task)   
    return p

def build_gep_task_tree(p):
    """
    Build the default task tree forthe GEP application of commercial agriculture. In this case, it's very similar to the standard task tree
    but i've included it here for consistency with other models.
    """
    p.commercial_agriculture_task = p.add_task(commercial_agriculture_tasks.commercial_agriculture)
    # p.commercial_agriculture_preprocess_task = p.add_task(gep_preprocess, parent=p.commercial_agriculture_task)  
    p.commercial_agriculture_gep_calculation_task = p.add_task(commercial_agriculture_tasks.gep_calculation, parent=p.commercial_agriculture_task)  
    # p.commercial_agriculture_gep_result_task = p.add_task(gep_result, parent=p.commercial_agriculture_task)      
    return p
    

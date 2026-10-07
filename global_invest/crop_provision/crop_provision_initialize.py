
from global_invest.crop_provision import crop_provision_tasks

def build_gep_service_calculation_task_tree(p, parent=None):
    """GEP calculation tree: the commercial FAOSTAT x CWoN valuation, plus the separate
    subsistence component (FAO RuLIS own-consumption shares x Lowder smallholder area shares)."""
    p.crop_provision_gep_calculation_task = p.add_task(crop_provision_tasks.gep_calculation)
    p.crop_subsistence_gep_task = p.add_task(crop_provision_tasks.crop_subsistence_gep)
    return p

def build_gep_service_task_tree(p, parent=None):
    """If you just want to load results, eg for reporting, this task tree inspects a different task tree and to learn paths and then loads results."""
    
    
    # QUESTION!!!! If a task truly already inspects itself to not rerun, what's the difference between loading and just executing the tree on 
    # an existing project? The difference is that load will do more error checking and FAIL rather than recalculate if it didn't find, also reporting
    # that it didn't find it and giving information about how to put the data in so it does find it in the base data or a manually-built project data.
    # I might want to have methods for automatically putting an archive into the right spot and also extended functionality for finding results in base_data
    # and functionality for promoting project results to base data per the new documentation in ee_dev.
    # Actually, maybe it's just that load_results is more useful for notebooks?
    
    p = build_gep_service_calculation_task_tree(p, parent=parent)
    p.crop_provision_gep_result_task = p.add_task(crop_provision_tasks.gep_result) 
    return p

    
def build_gep_task_tree(p):
    """
    Build the results-oriented task tree (very similar to the standard tree
    but i've included it here for consistency with other models.
    """
    p.crop_provision_gep_preprocess_task = p.add_task(crop_provision_tasks.gep_preprocess)  
    p.crop_provision_gep_calculation_task = p.add_task(crop_provision_tasks.gep_calculation)  
    p.crop_provision_gep_result_task = p.add_task(crop_provision_tasks.gep_result)   
    p.crop_provision_gep_results_distribution_task = p.add_task(crop_provision_tasks.gep_results_distribution)      
    return p
    

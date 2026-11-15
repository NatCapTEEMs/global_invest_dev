
from global_invest.crop_provision import crop_provision_tasks

def build_gep_service_calculation_task_tree(p, parent=None):
    """GEP calculation tree: the commercial FAOSTAT x CWoN valuation, plus the separate
    subsistence component (FAO RuLIS own-consumption shares x Lowder smallholder area shares)."""
    p.crop_provision_gep_calculation_task = p.add_task(crop_provision_tasks.gep_calculation)
    p.crop_subsistence_gep_task = p.add_task(crop_provision_tasks.crop_subsistence_gep)
    return p

def build_gep_service_task_tree(p, parent=None):
    """Build the default GEP task tree: the calculation tree, then the results report."""
    p = build_gep_service_calculation_task_tree(p, parent=parent)
    p.crop_provision_gep_result_task = p.add_task(crop_provision_tasks.gep_result) 
    return p

    
def build_gep_task_tree(p):
    """Build the full task tree: preprocess, both calculations, results report and results distribution.

    Kept for the same builder set the other services define. No run file calls it.
    """
    p.crop_provision_gep_preprocess_task = p.add_task(crop_provision_tasks.gep_preprocess)  
    p.crop_provision_gep_calculation_task = p.add_task(crop_provision_tasks.gep_calculation)  
    p.crop_subsistence_gep_task = p.add_task(crop_provision_tasks.crop_subsistence_gep)
    p.crop_provision_gep_result_task = p.add_task(crop_provision_tasks.gep_result)   
    p.crop_provision_gep_results_distribution_task = p.add_task(crop_provision_tasks.gep_results_distribution)      
    return p
    

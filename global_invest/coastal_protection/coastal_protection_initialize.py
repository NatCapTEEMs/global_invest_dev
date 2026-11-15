
from global_invest.coastal_protection import coastal_protection_tasks

def build_gep_service_calculation_task_tree(p):
    """Build the calculation-only GEP task tree."""
    p.coastal_protection_gep_calculation_task = p.add_task(coastal_protection_tasks.gep_calculation)  
    return p

def build_gep_service_task_tree(p):
    """Build the default GEP task tree: the calculation, then the results report."""
    p = build_gep_service_calculation_task_tree(p)
    p.coastal_protection_gep_result_task = p.add_task(coastal_protection_tasks.gep_result)   
    return p

    
def build_gep_task_tree(p):
    """Build the full task tree: preprocess, calculation, results report and results distribution.

    Kept for the same builder set the other services define. No run file calls it.
    """
    p.coastal_protection_gep_preprocess_task = p.add_task(coastal_protection_tasks.gep_preprocess)  
    p.coastal_protection_gep_calculation_task = p.add_task(coastal_protection_tasks.gep_calculation)  
    p.coastal_protection_gep_result_task = p.add_task(coastal_protection_tasks.gep_result)   
    p.coastal_protection_gep_results_distribution_task = p.add_task(coastal_protection_tasks.gep_results_distribution)      
    return p
    

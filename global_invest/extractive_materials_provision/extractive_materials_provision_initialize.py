
from global_invest.extractive_materials_provision import extractive_materials_provision_tasks

def build_gep_service_calculation_task_tree(p):
    """Build the calculation-only GEP task tree."""
    p.extractive_materials_provision_gep_calculation_task = p.add_task(extractive_materials_provision_tasks.gep_calculation)  
    return p

def build_gep_service_task_tree(p):
    """Build the default GEP task tree: calculation then the results report."""
    p = build_gep_service_calculation_task_tree(p)
    p.extractive_materials_provision_gep_result_task = p.add_task(extractive_materials_provision_tasks.gep_result)
    return p


def build_gep_task_tree(p):
    """Build the full task tree: preprocess, calculation, results report and results distribution.

    Kept for the same builder set the other services define. No run file calls it.
    """
    p.extractive_materials_provision_gep_preprocess_task = p.add_task(extractive_materials_provision_tasks.gep_preprocess)  
    p.extractive_materials_provision_gep_calculation_task = p.add_task(extractive_materials_provision_tasks.gep_calculation)  
    p.extractive_materials_provision_gep_result_task = p.add_task(extractive_materials_provision_tasks.gep_result)   
    p.extractive_materials_provision_gep_results_distribution_task = p.add_task(extractive_materials_provision_tasks.gep_results_distribution)      
    return p
    

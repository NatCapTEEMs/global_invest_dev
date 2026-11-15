
from global_invest.livestock_provision import livestock_provision_tasks

def build_gep_service_calculation_task_tree(p, parent=None):
    """The calculation alone: gep_calculation writes the CWoN country tables."""
    p.livestock_provision_gep_calculation_task = p.add_task(livestock_provision_tasks.gep_calculation)
    return p

def build_gep_service_task_tree(p, parent=None):
    """Full GEP run: the calculation task plus the results report (gep_result)."""
    p = build_gep_service_calculation_task_tree(p, parent=parent)
    p.livestock_provision_gep_result_task = p.add_task(livestock_provision_tasks.gep_result)
    return p

def build_gep_task_tree(p):
    """The four-task tree: preprocess (not implemented, a no-op), calculation, results report and
    results distribution to the output directory.
    """
    p.livestock_provision_gep_preprocess_task = p.add_task(livestock_provision_tasks.gep_preprocess)  
    p.livestock_provision_gep_calculation_task = p.add_task(livestock_provision_tasks.gep_calculation)  
    p.livestock_provision_gep_result_task = p.add_task(livestock_provision_tasks.gep_result)   
    p.livestock_provision_gep_results_distribution_task = p.add_task(livestock_provision_tasks.gep_results_distribution)      
    return p
    

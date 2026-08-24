
from global_invest.renewable_energy_provision import renewable_energy_provision_tasks

def build_gep_service_calculation_task_tree(p, parent=None):
    """Build the default GEP task tree."""
    p.renewable_energy_provision_gep_calculation_task = p.add_task(renewable_energy_provision_tasks.gep_calculation)  
    return p

def build_gep_service_task_tree(p, parent=None):
    """If you just want to load results, eg for reporting, this task tree inspects a different task tree and to learn paths and then loads results."""
    
    p = build_gep_service_calculation_task_tree(p, parent=parent)
    p.renewable_energy_provision_gep_result_task = p.add_task(renewable_energy_provision_tasks.gep_result)   
    return p
    
def build_gep_task_tree(p):
    """
    Build the results-oriented task tree (very similar to the standard tree
    but i've included it here for consistency with other models.
    """
    p.renewable_energy_provision_gep_preprocess_task = p.add_task(renewable_energy_provision_tasks.gep_preprocess)  
    p.renewable_energy_provision_gep_calculation_task = p.add_task(renewable_energy_provision_tasks.gep_calculation)  
    p.renewable_energy_provision_gep_result_task = p.add_task(renewable_energy_provision_tasks.gep_result)   
    p.renewable_energy_provision_gep_results_distribution_task = p.add_task(renewable_energy_provision_tasks.gep_results_distribution)      
    return p
    

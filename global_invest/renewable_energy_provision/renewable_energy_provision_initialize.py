import pandas as pd
import hazelbean as hb

from global_invest import utilities
from global_invest.renewable_energy_provision import renewable_energy_provision_tasks

def initialize_paths(p):
    """One source of truth for the inputs (shared country block + service data, get_path references)."""
    utilities.initialize_country_paths(p, simplified='30sec')

    # IRENA generation, WB prices, CWON resource rents. This service's data lives under the
    # drive's base_data/global_invest/ prefix, unlike the other accounting services (see CHANGELOG).
    p.irena_production_ref_path = p.get_path('global_invest', 'renewable_energy_provision', 'IRENA_prod_by_country.csv')
    p.wb_price_ref_path = p.get_path('global_invest', 'renewable_energy_provision', 'WB_price_data.csv')
    p.cwon_resource_rent_ref_path = p.get_path('global_invest', 'renewable_energy_provision', 'CWON_resource_rent_data.csv')

def build_gep_service_calculation_task_tree(p, parent=None):
    """Build the default task tree for commercial agriculture."""
    p.renewable_energy_provision_task = p.add_task(renewable_energy_provision_tasks.renewable_energy_provision, parent=parent)
    p.renewable_energy_provision_gep_calculation_task = p.add_task(renewable_energy_provision_tasks.gep_calculation, parent=p.renewable_energy_provision_task)  
    return p

def build_gep_service_task_tree(p, parent=None):
    """If you just want to load results, eg for reporting, this task tree inspects a different task tree and to learn paths and then loads results."""
    
    p = build_gep_service_calculation_task_tree(p, parent=parent)
    p.renewable_energy_provision_gep_result_task = p.add_task(renewable_energy_provision_tasks.gep_result, parent=p.renewable_energy_provision_task)   
    return p
    
def build_gep_task_tree(p):
    """
    Build the default task tree forthe GEP application of commercial agriculture. In this case, it's very similar to the standard task tree
    but i've included it here for consistency with other models.
    """
    p.renewable_energy_provision_task = p.add_task(renewable_energy_provision_tasks.renewable_energy_provision)
    p.renewable_energy_provision_gep_preprocess_task = p.add_task(renewable_energy_provision_tasks.gep_preprocess, parent=p.renewable_energy_provision_task)  
    p.renewable_energy_provision_gep_calculation_task = p.add_task(renewable_energy_provision_tasks.gep_calculation, parent=p.renewable_energy_provision_task)  
    p.renewable_energy_provision_gep_result_task = p.add_task(renewable_energy_provision_tasks.gep_result, parent=p.renewable_energy_provision_task)   
    p.renewable_energy_provision_gep_results_distribution_task = p.add_task(renewable_energy_provision_tasks.gep_results_distribution, parent=p.renewable_energy_provision_task)      
    return p
    

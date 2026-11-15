from global_invest.fuelwood_provision import fuelwood_provision_tasks


def build_gep_service_calculation_task_tree(p):
    """Build the calculation-only GEP task tree."""
    p.fuelwood_provision_gep_calculation_task = p.add_task(fuelwood_provision_tasks.gep_calculation)
    return p


def build_gep_service_task_tree(p):
    """Build the default GEP task tree: calculation then the results report."""
    p = build_gep_service_calculation_task_tree(p)
    p.fuelwood_provision_gep_result_task = p.add_task(fuelwood_provision_tasks.gep_result)
    return p

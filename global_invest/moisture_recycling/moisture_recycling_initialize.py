from global_invest.moisture_recycling import moisture_recycling_tasks


def build_gep_service_calculation_task_tree(p):
    """Build the calculation-only GEP task tree."""
    p.moisture_recycling_gep_calculation_task = p.add_task(moisture_recycling_tasks.gep_calculation)
    return p


def build_gep_service_task_tree(p):
    """Build the default GEP task tree: calculation then the results report."""
    p = build_gep_service_calculation_task_tree(p)
    p.moisture_recycling_gep_result_task = p.add_task(moisture_recycling_tasks.gep_result)
    return p

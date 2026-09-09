from global_invest.health_depression import health_depression_tasks


def build_gep_service_calculation_task_tree(p):
    """Build the calculation-only GEP task tree."""
    p.health_depression_urban_prevented_factor_task = p.add_task(health_depression_tasks.urban_prevented_factor)
    p.health_depression_gep_calculation_task = p.add_task(health_depression_tasks.gep_calculation)
    return p


def build_gep_service_task_tree(p):
    """Build the default GEP task tree: calculation then the results report."""
    p = build_gep_service_calculation_task_tree(p)
    p.health_depression_gep_result_task = p.add_task(health_depression_tasks.gep_result)
    return p

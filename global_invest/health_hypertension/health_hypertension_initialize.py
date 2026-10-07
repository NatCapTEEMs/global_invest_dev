from global_invest.health_hypertension import health_hypertension_tasks


def build_gep_service_calculation_task_tree(p):
    """Build the calculation-only GEP task tree."""
    p.health_hypertension_urban_greenness_task = p.add_task(health_hypertension_tasks.urban_greenness)
    p.health_hypertension_gep_calculation_task = p.add_task(health_hypertension_tasks.gep_calculation)
    return p


def build_gep_service_task_tree(p):
    """Build the default GEP task tree: calculation then the results report."""
    p = build_gep_service_calculation_task_tree(p)
    p.health_hypertension_gep_result_task = p.add_task(health_hypertension_tasks.gep_result)
    return p

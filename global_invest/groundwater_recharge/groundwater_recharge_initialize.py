from global_invest.groundwater_recharge import groundwater_recharge_tasks


def build_gep_service_calculation_task_tree(p):
    """Build the calculation-only GEP task tree."""
    p.groundwater_recharge_gep_calculation_task = p.add_task(groundwater_recharge_tasks.gep_calculation)
    return p


def build_gep_service_task_tree(p):
    """Build the default GEP task tree: calculation then the results report."""
    p = build_gep_service_calculation_task_tree(p)
    p.groundwater_recharge_gep_result_task = p.add_task(groundwater_recharge_tasks.gep_result)
    return p

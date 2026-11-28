from global_invest.pest_control import pest_control_tasks


def build_gep_service_calculation_task_tree(p):
    """Build the calculation-only GEP task tree: the FiBL snapshot check, then the valuation."""
    p.pest_control_fibl_organic_area_snapshot_task = p.add_task(pest_control_tasks.fibl_organic_area_snapshot)
    p.pest_control_gep_calculation_task = p.add_task(pest_control_tasks.gep_calculation)
    return p


def build_gep_service_task_tree(p):
    """Build the default GEP task tree: calculation then the results report."""
    p = build_gep_service_calculation_task_tree(p)
    p.pest_control_gep_result_task = p.add_task(pest_control_tasks.gep_result)
    return p

from global_invest.nature_view_amenity import nature_view_amenity_tasks


def build_gep_service_calculation_task_tree(p):
    """Build the calculation-only GEP task tree."""
    p.nature_view_amenity_coordinate_countries_task = p.add_task(nature_view_amenity_tasks.coordinate_countries)
    p.nature_view_amenity_gep_calculation_task = p.add_task(nature_view_amenity_tasks.gep_calculation)
    return p


def build_gep_service_task_tree(p):
    """Build the default GEP task tree: calculation then the results report."""
    p = build_gep_service_calculation_task_tree(p)
    p.nature_view_amenity_gep_result_task = p.add_task(nature_view_amenity_tasks.gep_result)
    return p

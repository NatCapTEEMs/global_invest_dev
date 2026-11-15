"""Marine carbon wiring: GEP task trees (global_invest module layout).

GEP-only. The service is reported separately and kept out of the summed GEP total. Its column in
the comparison workbook is not yet mapped.
"""
from global_invest.marine_carbon import marine_carbon_tasks


def build_gep_service_calculation_task_tree(p):
    """GEP calculation tree: ocean areas -> the area-share valuation."""
    p.marine_carbon_ocean_areas_task = p.add_task(marine_carbon_tasks.ocean_areas)
    p.gep_calculation_task = p.add_task(marine_carbon_tasks.gep_calculation)
    return p


def build_gep_service_task_tree(p):
    """Full GEP run: the calculation plus the results/report task."""
    p = build_gep_service_calculation_task_tree(p)
    p.marine_carbon_gep_result_task = p.add_task(marine_carbon_tasks.gep_result)
    return p

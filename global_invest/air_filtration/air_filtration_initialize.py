"""Air-filtration wiring: GEP task trees (global_invest module layout).

GEP-only; one workbook valued through two channels (air filtration + sandstorm prevention).
"""
from global_invest.air_filtration import air_filtration_tasks


def build_gep_service_calculation_task_tree(p):
    """GEP calculation tree: the workbook valuation -> r250 one-row-per-country table."""
    p.gep_calculation_task = p.add_task(air_filtration_tasks.gep_calculation)
    return p


def build_gep_service_task_tree(p):
    """Full GEP run: the calculation plus the results/report task."""
    p = build_gep_service_calculation_task_tree(p)
    p.air_filtration_gep_result_task = p.add_task(air_filtration_tasks.gep_result)
    return p

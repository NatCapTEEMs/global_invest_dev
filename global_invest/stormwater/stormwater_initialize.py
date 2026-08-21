"""Stormwater wiring: GEP task trees (global_invest module layout)."""
from global_invest.stormwater import stormwater_tasks


def build_gep_service_calculation_task_tree(p):
    """GEP calculation tree: retention volume times price."""
    p.gep_calculation_task = p.add_task(stormwater_tasks.gep_calculation)
    return p


def build_gep_service_task_tree(p):
    """Full GEP run: the calculation plus the results/report task."""
    p = build_gep_service_calculation_task_tree(p)
    p.stormwater_gep_result_task = p.add_task(stormwater_tasks.gep_result)
    return p

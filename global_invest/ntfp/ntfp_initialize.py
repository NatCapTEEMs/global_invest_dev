"""NTFP wiring: GEP task trees (global_invest module layout)."""
from global_invest.ntfp import ntfp_tasks


def build_gep_service_calculation_task_tree(p):
    """GEP calculation tree: the accessibility stage, then the valuation it feeds."""
    p.accessible_forest_task = p.add_task(ntfp_tasks.accessible_forest)
    p.gep_calculation_task = p.add_task(ntfp_tasks.gep_calculation)
    return p


def build_gep_service_task_tree(p):
    """Full GEP run: the calculation plus the results/report task."""
    p = build_gep_service_calculation_task_tree(p)
    p.ntfp_gep_result_task = p.add_task(ntfp_tasks.gep_result)
    return p

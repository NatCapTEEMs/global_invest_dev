"""NTFP wiring: GEP task trees (global_invest module layout)."""
from global_invest.ntfp import ntfp_tasks


def build_gep_service_calculation_task_tree(p):
    """GEP calculation tree: accessible-forest valuation, both source-script variants."""
    p.gep_calculation_task = p.add_task(ntfp_tasks.gep_calculation)
    return p


def build_gep_service_task_tree(p):
    """Full GEP run: the calculation plus the results/report task."""
    p = build_gep_service_calculation_task_tree(p)
    p.ntfp_gep_result_task = p.add_task(ntfp_tasks.gep_result)
    return p

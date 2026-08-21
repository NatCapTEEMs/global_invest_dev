"""Timber-provision wiring: GEP task trees (global_invest module layout)."""
from global_invest.timber_provision import timber_provision_tasks


def build_gep_service_calculation_task_tree(p):
    """GEP calculation tree: the committed timber table -> r250 one-row-per-country."""
    p.gep_calculation_task = p.add_task(timber_provision_tasks.gep_calculation)
    return p


def build_gep_service_task_tree(p):
    """Full GEP run: the calculation plus the results/report task."""
    p = build_gep_service_calculation_task_tree(p)
    p.timber_provision_gep_result_task = p.add_task(timber_provision_tasks.gep_result)
    return p

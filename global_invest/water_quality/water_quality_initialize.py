"""Water-quality wiring: GEP task trees (global_invest module layout)."""
from global_invest.water_quality import water_quality_tasks


def build_gep_service_calculation_task_tree(p):
    """GEP calculation tree: the retention calculation -> r250 one-row-per-country table."""
    p.gep_calculation_task = p.add_task(water_quality_tasks.gep_calculation)
    return p


def build_gep_service_task_tree(p):
    """Full GEP run: the calculation plus the results/report task."""
    p = build_gep_service_calculation_task_tree(p)
    p.water_quality_gep_result_task = p.add_task(water_quality_tasks.gep_result)
    return p

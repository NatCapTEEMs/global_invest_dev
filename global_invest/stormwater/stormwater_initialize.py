"""Stormwater wiring: GEP task trees (global_invest module layout)."""
from global_invest.stormwater import stormwater_tasks


def build_gep_service_calculation_task_tree(p):
    """GEP calculation tree: the zonal sum of the InVEST retention raster, then its price.

    skip_existing=1 on the zonal step, which costs minutes over a 74,596-pixel-wide grid and is
    deterministic."""
    p.retention_by_country_task = p.add_task(stormwater_tasks.retention_by_country, skip_existing=1)
    p.gep_calculation_task = p.add_task(stormwater_tasks.gep_calculation)
    return p


def build_gep_service_task_tree(p):
    """Full GEP run: the calculation plus the results/report task."""
    p = build_gep_service_calculation_task_tree(p)
    p.stormwater_gep_result_task = p.add_task(stormwater_tasks.gep_result)
    return p

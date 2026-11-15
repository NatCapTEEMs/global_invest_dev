"""NTFP wiring: GEP task trees (global_invest module layout)."""
from global_invest.ntfp import ntfp_tasks


def build_gep_service_calculation_task_tree(p):
    """GEP calculation tree: FAO total forest per country, then the valuation it feeds."""
    p.fao_forest_task = p.add_task(ntfp_tasks.fao_forest)
    p.gep_calculation_task = p.add_task(ntfp_tasks.gep_calculation)
    return p


def build_gep_service_sensitivity_task_tree(p):
    """The alternatives beside the account's value, each its own task and table.

    The divide-by-five on the account's area, the rate times the area CWoN publishes, the rate
    times all mapped ESA forest, and the earlier valuation over accessible forest. The two forest
    raster tasks read global rasters and run on the cluster, and each skips once its table exists.
    """
    p.divide_by_five_sensitivity_task = p.add_task(ntfp_tasks.divide_by_five_sensitivity)
    p.cwon_forest_task = p.add_task(ntfp_tasks.cwon_forest)
    p.cwon_forest_comparison_task = p.add_task(ntfp_tasks.cwon_forest_comparison)
    p.accessible_forest_task = p.add_task(ntfp_tasks.accessible_forest)
    p.esa_forest_task = p.add_task(ntfp_tasks.esa_forest)
    p.esa_forest_sensitivity_task = p.add_task(ntfp_tasks.esa_forest_sensitivity)
    p.accessible_forest_valuation_task = p.add_task(ntfp_tasks.accessible_forest_valuation)
    return p


def build_gep_service_task_tree(p):
    """Full GEP run: the calculation, its alternatives, then the results/report task."""
    p = build_gep_service_calculation_task_tree(p)
    p = build_gep_service_sensitivity_task_tree(p)
    p.ntfp_gep_result_task = p.add_task(ntfp_tasks.gep_result)
    return p

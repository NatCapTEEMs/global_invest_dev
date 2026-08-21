"""Water-supply wiring: GEP task trees (global_invest module layout).

GEP-only service; the hydropower component is built (CWoN resource-rent method), agriculture
and household join beside it when their science surfaces.
"""
from global_invest.water_supply import water_supply_tasks


def build_gep_service_calculation_task_tree(p):
    """GEP calculation tree: hydropower rent -> r250 one-row-per-country valuation."""
    p.hydropower_rent_task = p.add_task(water_supply_tasks.hydropower_rent)
    p.water_use_components_task = p.add_task(water_supply_tasks.water_use_components)
    p.gep_calculation_task = p.add_task(water_supply_tasks.gep_calculation)
    return p


def build_gep_service_task_tree(p):
    """Full GEP run: the calculation chain plus the results/report task."""
    p = build_gep_service_calculation_task_tree(p)
    p.water_supply_gep_result_task = p.add_task(water_supply_tasks.gep_result)
    return p

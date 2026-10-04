"""Recreation wiring: GEP task trees (global_invest module layout).

GEP-only service (no GTAP shock seam): the calculation builds site quality from LULC shares
and accessibility, dissolves the high-quality pixels into discrete sites, allocates UNWTO
overnights to hotels, runs the banded travel-cost flows for residents and tourists, prices
the accommodation and air-travel channels, and aggregates to one row per country on r250 --
the source pipeline's own surface, so aggregation and country collapse coincide.
"""
from global_invest.recreation import recreation_tasks


def build_gep_service_calculation_task_tree(p):
    """GEP calculation tree: indices -> sites -> overnights -> banded flows -> the two added
    channels -> r250 one-row-per-country valuation."""
    p.environment_index_task = p.add_task(recreation_tasks.environment_index)
    p.accessibility_index_task = p.add_task(recreation_tasks.accessibility_index)
    p.recreation_sites_task = p.add_task(recreation_tasks.recreation_sites)
    p.overnight_allocation_task = p.add_task(recreation_tasks.overnight_allocation)
    p.daily_recreation_task = p.add_task(recreation_tasks.daily_recreation)
    p.tourist_recreation_task = p.add_task(recreation_tasks.tourist_recreation)
    p.accommodation_value_task = p.add_task(recreation_tasks.accommodation_value)
    p.air_travel_value_task = p.add_task(recreation_tasks.air_travel_value)
    p.gep_calculation_task = p.add_task(recreation_tasks.gep_calculation)
    return p


def build_gep_service_results_task_tree(p):
    """Results-only run: load a PRIOR calculation's results and render the report (fails loudly
    if the calculation has not run; does NOT recompute)."""
    p.recreation_gep_load_results_task = p.add_task(recreation_tasks.gep_load_results)
    p.recreation_gep_result_task = p.add_task(recreation_tasks.gep_result)
    return p


def build_gep_service_task_tree(p):
    """Full GEP run: the calculation calculation plus the results/report task."""
    p = build_gep_service_calculation_task_tree(p)
    p.recreation_gep_result_task = p.add_task(recreation_tasks.gep_result)
    return p

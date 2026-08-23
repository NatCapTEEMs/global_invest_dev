"""Fire-protection wiring: GEP task trees (global_invest module layout).

GEP-only service (no GTAP shock seam): committed per-country AR(1) persistence betas ->
avoided burned acres in the base year -> valued at EM-DAT damage per burned acre. The source
repo's committed output is the replication anchor (reference/, reproduced to the float).
"""
from global_invest.fire_protection import fire_protection_tasks


def build_gep_service_calculation_task_tree(p):
    """GEP calculation tree: betas -> avoided acres -> damage rates -> per-country valuation."""
    p.beta_differences_task = p.add_task(fire_protection_tasks.beta_differences)
    p.avoided_burned_area_task = p.add_task(fire_protection_tasks.avoided_burned_area)
    p.damage_per_acre_task = p.add_task(fire_protection_tasks.damage_per_acre)
    p.gep_calculation_task = p.add_task(fire_protection_tasks.gep_calculation)
    return p


def build_gep_service_task_tree(p):
    """Full GEP run: the calculation calculation plus the results/report task."""
    p = build_gep_service_calculation_task_tree(p)
    p.fire_protection_gep_result_task = p.add_task(fire_protection_tasks.gep_result)
    return p

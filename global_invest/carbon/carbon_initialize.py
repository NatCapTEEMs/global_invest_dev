"""Standard task-tree seam for the carbon ES model (global_invest module layout).

Consumers (ngfs_pnas, nff_global, brazil) set the project-specific inputs on p, then call
add_carbon_tasks(p) after their SEALS stitch task to graft the dynamic carbon ES-shock computation.
The science lives in carbon_functions.py / carbon_tasks.py; this file only wires it into a tree, so
every consumer grafts carbon the same way (mirrors add_pollination_tasks etc.).
"""
from global_invest.carbon import carbon_tasks


def add_carbon_tasks(p, parent=None):
    """Graft the dynamic carbon ES-shock task onto p.

    Caller sets on p before calling: carbon_lulc_path_template, carbon_shock_scenarios,
    carbon_shock_base_year, carbon_shock_years, carbon_shock_output_path. Standard GTAP-carbon
    inputs (r50xAEZ boundary, Spawn density, zones) default inside the task via p.get_path; override
    on p only when different. Writes the per-region FRS aoall shock CSV at carbon_shock_output_path.
    """
    p.compute_carbon_shock_task = p.add_task(carbon_tasks.task_compute_carbon_shock, parent=parent)
    return p

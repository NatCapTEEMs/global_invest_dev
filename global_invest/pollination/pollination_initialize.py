"""Standard task-tree seam for the pollination ES model (global_invest module layout).

Consumers (ngfs_pnas, nff_global) set the project-specific inputs on p, then call
add_pollination_tasks(p) after their SEALS stitch task to graft the dynamic pollination ES-shock
computation. The science lives in pollination_functions.py / pollination_tasks.py; this file only
wires it into a tree, so every consumer grafts pollination the same way (mirrors add_carbon_tasks).
"""
from global_invest.pollination import pollination_tasks


def add_pollination_tasks(p, parent=None):
    """Graft the dynamic pollination ES-shock task onto p.

    Caller sets on p before calling: pollination_shock_years (SEALS-map anchor years, from
    seals_years), pollination_shock_base_year, pollination_shock_scenarios,
    pollination_lulc_path_template, pollination_baseline_lulc_path,
    pollination_shock_output_path. Standard GTAP r50xAEZ boundary defaults inside the task via
    p.get_path; override on p only when different. Writes the per-region V_F/OSD shock CSV at
    pollination_shock_output_path.
    """
    p.compute_pollination_shock_task = p.add_task(pollination_tasks.task_compute_pollination_shock, parent=parent)
    return p

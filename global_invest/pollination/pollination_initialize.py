"""Standard task-tree seam for the pollination ES model (global_invest module layout).

Consumers (ngfs_pnas, nff_global) set the project-specific inputs on p, then call
add_pollination_tasks(p) after their SEALS stitch task to graft the dynamic pollination ES-shock
computation. The science lives in pollination_functions.py / pollination_tasks.py; this file only
wires it into a tree, so every consumer grafts pollination the same way (mirrors add_carbon_tasks).
"""
from global_invest.pollination import pollination_tasks


def add_pollination_tasks(p, parent=None):
    """Graft the pollination ES-shock task onto p, dispatching STATIC vs DYNAMIC on SEALS-map availability.

    DYNAMIC (>=2 SEALS map years in p.seals_years): recompute the sufficiency shock from our SEALS maps
    at each anchor year (task_compute_pollination_shock). STATIC (<2, the fallback -- the dynamic
    recompute needs >=2 anchors to measure change): read the frozen
    raw_dependencies/pollination_dependency.csv (task_compute_pollination_shock_static). Mirrors
    add_erosion_tasks / add_carbon_tasks; both paths write pollination_interpolated.csv.

    Caller sets on p before calling: pollination_shock_scenarios, pollination_shock_base_year,
    pollination_shock_output_path. DYNAMIC also: pollination_shock_years (SEALS-map anchor years, from
    seals_years), pollination_lulc_path_template, pollination_base_year_lulc_path,
    pollination_shock_base_scenario. STATIC also: pollination_shock_end_year. Standard GTAP r50xAEZ
    boundary defaults inside the task via p.get_path; override on p only when different.
    """
    dynamic = len(str(getattr(p, 'seals_years', '') or '').split()) >= 2
    task = (pollination_tasks.task_compute_pollination_shock if dynamic
            else pollination_tasks.task_compute_pollination_shock_static)
    p.compute_pollination_shock_task = p.add_task(task, parent=parent)
    return p

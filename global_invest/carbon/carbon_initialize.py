"""Standard task-tree seam for the carbon ES model (global_invest module layout).

Consumers (ngfs_pnas, nff_global, brazil) set the project-specific inputs on p, then call
add_carbon_tasks(p) after their SEALS stitch task to graft the dynamic carbon ES-shock computation.
The science lives in carbon_functions.py / carbon_tasks.py; this file only wires it into a tree, so
every consumer grafts carbon the same way (mirrors add_pollination_tasks etc.).
"""
from global_invest.carbon import carbon_tasks


def add_carbon_tasks(p, parent=None):
    """Graft the carbon ES-shock task onto p, dispatching STATIC vs DYNAMIC on SEALS-map availability.

    DYNAMIC (>=2 SEALS map years in p.seals_years): recompute the carbon-density shock from our SEALS
    maps at each anchor year (task_compute_carbon_shock). STATIC (<2, the fallback -- the dynamic
    recompute needs >=2 anchors to measure change): read the frozen
    raw_dependencies/carbon_storage_dependency.csv (task_compute_carbon_shock_static). Mirrors
    add_erosion_tasks / add_pollination_tasks; both paths write carbon_storage_interpolated.csv.

    Caller sets on p before calling: carbon_shock_scenarios, carbon_shock_base_year,
    carbon_shock_output_path. DYNAMIC also: carbon_lulc_path_template, carbon_shock_years,
    carbon_base_year_lulc_path, carbon_shock_base_scenario. STATIC also: carbon_shock_end_year.
    Standard GTAP-carbon inputs (r50xAEZ boundary, Spawn density, zones) default inside the task via
    p.get_path; override on p only when different.
    """
    dynamic = len(str(getattr(p, 'seals_years', '') or '').split()) >= 2
    if not dynamic:   # <2 SEALS map years -> read the frozen dependency table
        p.compute_carbon_shock_task = p.add_task(carbon_tasks.task_compute_carbon_shock_static, parent=parent)
        return p
    # dynamic: recompute from the SEALS maps (one task for carbon; cf. erosion's multi-task chain)
    p.compute_carbon_shock_task = p.add_task(carbon_tasks.task_compute_carbon_shock, parent=parent)
    return p

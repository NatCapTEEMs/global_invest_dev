"""Standard task-tree seam for the pollination ES model (global_invest module layout).

Consumers (ngfs_pnas, nff_global) set the project-specific inputs on p, then call
add_pollination_tasks(p) after their SEALS stitch task to graft the dynamic pollination ES-shock
computation. The science lives in pollination_functions.py / pollination_tasks.py; this file only
wires it into a tree, so every consumer grafts pollination the same way (mirrors add_carbon_tasks).
"""
from global_invest.pollination import pollination_tasks


def add_pollination_tasks(p, parent=None):
    """Graft the pollination ES-shock task onto p, dispatching STATIC vs DYNAMIC on p.dynamic_es.

    DYNAMIC ('pollination' in p.dynamic_es): recompute the sufficiency shock from our SEALS maps at each
    p.es_shock_years anchor (task_compute_pollination_shock). STATIC (the default): read the frozen
    raw_dependencies/pollination_dependency.csv (task_compute_pollination_shock_static). Mirrors
    add_erosion_tasks / add_terrestrial_carbon_tasks; both paths write pollination_interpolated.csv.

    Caller sets only the shared es_shock_* config (see run_ngfs_pnas STEP 6). Everything
    pollination-specific defaults in the task: the output CSV into p.es_shock_dir, the r50xAEZ
    boundary via p.get_path.
    """
    dynamic = 'pollination' in getattr(p, 'dynamic_es', [])
    if not dynamic:   # not requested dynamic -> read the frozen dependency table
        p.compute_pollination_shock_task = p.add_task(pollination_tasks.task_compute_pollination_shock_static, parent=parent)
        return p
    # dynamic: recompute from the SEALS maps (one task for pollination; cf. erosion's multi-task chain)
    p.compute_pollination_shock_task = p.add_task(pollination_tasks.task_compute_pollination_shock, parent=parent)
    return p

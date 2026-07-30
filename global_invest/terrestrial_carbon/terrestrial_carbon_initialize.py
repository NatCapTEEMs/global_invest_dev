"""Standard task-tree seam for the carbon ES model (global_invest module layout).

Consumers (ngfs_pnas, nff_global, brazil) set the project-specific inputs on p, then call
add_terrestrial_carbon_tasks(p) after their SEALS stitch task to graft the dynamic carbon ES-shock computation.
The science lives in terrestrial_carbon_functions.py / terrestrial_carbon_tasks.py; this file only wires it into a tree, so
every consumer grafts carbon the same way (mirrors add_pollination_tasks etc.).
"""
from global_invest.terrestrial_carbon import terrestrial_carbon_tasks


def add_terrestrial_carbon_tasks(p, parent=None):
    """Graft the carbon ES-shock task onto p, dispatching STATIC vs DYNAMIC on p.dynamic_es.

    DYNAMIC ('terrestrial_carbon' in p.dynamic_es): recompute the carbon-density shock from our SEALS
    maps at each p.es_shock_years anchor (task_compute_terrestrial_carbon_shock). STATIC (the default):
    read the frozen raw_dependencies/carbon_storage_dependency.csv
    (task_compute_terrestrial_carbon_shock_static). Mirrors add_erosion_tasks / add_pollination_tasks;
    both paths write terrestrial_carbon_interpolated.csv.

    Caller sets only the shared es_shock_* config (see run_ngfs_pnas STEP 6). Everything
    carbon-specific defaults in the task: the output CSV into p.es_shock_dir, the r50xAEZ boundary /
    Spawn density / carbon zones via p.get_path.
    """
    dynamic = 'terrestrial_carbon' in getattr(p, 'dynamic_es', [])
    if not dynamic:   # not requested dynamic -> read the frozen dependency table
        p.compute_terrestrial_carbon_shock_task = p.add_task(terrestrial_carbon_tasks.task_compute_terrestrial_carbon_shock_static, parent=parent)
        return p
    # dynamic: recompute from the SEALS maps (one task for carbon; cf. erosion's multi-task chain)
    p.compute_terrestrial_carbon_shock_task = p.add_task(terrestrial_carbon_tasks.task_compute_terrestrial_carbon_shock, parent=parent)
    return p

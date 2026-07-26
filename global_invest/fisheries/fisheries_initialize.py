"""Standard task-tree seam for the fisheries ES model (mirrors add_pollination_tasks / add_carbon_tasks).

Fisheries is STATIC (marine, by RCP), so add_fisheries_tasks grafts a task that reads the pre-computed
cwon_shocks.har FI headers rather than recomputing from SEALS maps. Consumers (ngfs_pnas, nff_global)
set the fisheries_shock_* inputs on p, then call add_fisheries_tasks(p) alongside the other ES seams.
"""
from global_invest.fisheries import fisheries_tasks


def add_fisheries_tasks(p, parent=None):
    """Graft the (static) fisheries ES-shock task onto p.

    Caller sets on p before calling: fisheries_shock_scenarios, fisheries_shock_base_year,
    fisheries_shock_end_year, fisheries_shock_output_path. cwon_shocks.har defaults via
    base_data_dir / aggregation_label. Writes the per-region FSH shock CSV at fisheries_shock_output_path.
    """
    p.compute_fisheries_shock_task = p.add_task(fisheries_tasks.task_compute_fisheries_shock, parent=parent)
    return p

"""Standard task-tree seam for the fisheries ES model (mirrors add_pollination_tasks / add_terrestrial_carbon_tasks).

Fisheries is STATIC (marine, by RCP), so add_fisheries_tasks grafts a task that reads the pre-computed
cwon_shocks.har FI headers rather than recomputing from SEALS maps. Consumers (ngfs_pnas, nff_global)
set the fisheries_shock_* inputs on p, then call add_fisheries_tasks(p) alongside the other ES seams.
"""
from global_invest.fisheries import fisheries_tasks


def add_fisheries_tasks(p, parent=None):
    """Graft the (static) fisheries ES-shock task onto p.

    Caller sets only the shared es_shock_* config (see run_ngfs_pnas STEP 6). Marine, so it never reads
    the SEALS maps and has no dynamic path: p.dynamic_es does not apply to it. cwon_shocks.har defaults
    via base_data_dir / aggregation_label; the per-region FSH shock CSV lands in p.es_shock_dir.
    """
    p.fisheries_shock_task = p.add_task(fisheries_tasks.fisheries_shock, parent=parent)
    return p

"""Standard task-tree seam for the fisheries ES model (mirrors add_pollination_tasks / add_terrestrial_carbon_tasks).

Fisheries is STATIC (marine, by RCP), so add_fisheries_tasks grafts a task that reads the pre-computed
cwon_shocks.har FI headers rather than recomputing from SEALS maps. Consumers (ngfs_pnas, nff_global)
set the fisheries_shock_* inputs on p, then call add_fisheries_tasks(p) alongside the other ES seams.
"""
from global_invest.fisheries import fisheries_tasks


def add_fisheries_tasks(p, parent=None):
    """Graft the (static) fisheries ES-shock task onto p.

    Caller sets only the shared es_shock_* config (see run_ngfs_pnas STEP 6). Marine, so it never reads
    the SEALS maps and has no dynamic path: p.dynamic_es does not apply to it. cwon_shocks.har resolves
    via get_path under the gtappy release layout (aggregation_label row); the per-region shock CSV
    lands in p.es_shock_dir.
    """
    p.fisheries_shock_task = p.add_task(fisheries_tasks.fisheries_shock, parent=parent)
    return p


# ---------------------------------------------------------------------------------------------
# GEP task trees (the valuation side; the shock seam above is unchanged).
# ---------------------------------------------------------------------------------------------
def build_gep_service_calculation_task_tree(p):
    """GEP calculation tree: CWoN rent trends -> r250 one-row-per-country valuation, plus
    the separate subsistence component (Lynch et al. 2024)."""
    p.fisheries_rent_trends_task = p.add_task(fisheries_tasks.fisheries_rent_trends)
    p.gep_calculation_task = p.add_task(fisheries_tasks.gep_calculation)
    p.fisheries_subsistence_gep_task = p.add_task(fisheries_tasks.fisheries_subsistence_gep)
    p.fisheries_aquaculture_gep_task = p.add_task(fisheries_tasks.fisheries_aquaculture_gep)
    return p


def build_gep_service_task_tree(p):
    """Full GEP run: the calculation calculation plus the results/report task."""
    p = build_gep_service_calculation_task_tree(p)
    p.fisheries_gep_result_task = p.add_task(fisheries_tasks.gep_result)
    return p

"""Pollination wiring: GEP task trees + the ES-shock seam (global_invest module layout).

GEP side: the valuation consumes the pollination value raster
(poll_value_global_<gep_base_year>usd.tif -- USD per cell, crop prices and pollination dependence
embedded upstream, so lambda = 1) and aggregates it to one row per country on r250.
Shock side: consumers (ngfs_pnas, nff_global) call add_pollination_tasks(p) after their SEALS
stitch task; it dispatches static vs dynamic on p.dynamic_es (mirrors add_terrestrial_carbon_tasks).
"""
from global_invest.pollination import pollination_tasks


def build_gep_service_calculation_task_tree(p):
    """GEP calculation tree: value raster -> per-r264 sums -> r250 one-row-per-country valuation.

    fao_median_prices is the first step towards building that value raster here rather than
    taking it as given. It downloads FAOSTAT production and producer prices and writes the
    per-crop median price the raster is priced at. skip_existing=1 because it is a download."""
    p.fao_median_prices = p.add_task(pollination_tasks.fao_median_prices, skip_existing=1)
    p.pollination_value_by_region = p.add_task(
        pollination_tasks.pollination_value_by_region, skip_existing=1)
    p.gep_calculation = p.add_task(pollination_tasks.gep_calculation)
    return p


def build_gep_service_results_task_tree(p):
    """Results-only run: load a PRIOR calculation's results and render the report (fails loudly
    if the calculation has not run; does NOT recompute)."""
    p.pollination_gep_load_results_task = p.add_task(pollination_tasks.gep_load_results)
    p.pollination_gep_result_task = p.add_task(pollination_tasks.gep_result)
    return p


def build_gep_service_task_tree(p):
    """Full GEP run: the calculation calculation plus the results/report task."""
    p = build_gep_service_calculation_task_tree(p)
    p.pollination_gep_result_task = p.add_task(pollination_tasks.gep_result)
    return p


# ---------------------------------------------------------------------------------------------
# ES-shock wiring (the consumer seam). Everything above builds the GEP task trees; this builds the ES-shock one.
# ---------------------------------------------------------------------------------------------
def add_pollination_tasks(p, parent=None):
    """Graft the pollination ES-shock task onto p, dispatching STATIC vs DYNAMIC on p.dynamic_es.

    DYNAMIC ('pollination' in p.dynamic_es): recompute the sufficiency shock from our SEALS maps at each
    p.es_shock_years anchor (pollination_shock). STATIC (the default): read the frozen
    raw_dependencies/pollination_dependency.csv (pollination_shock_static). Mirrors
    add_erosion_tasks / add_terrestrial_carbon_tasks; both paths write pollination_interpolated.csv.

    Caller sets only the shared es_shock_* config (see run_ngfs_pnas STEP 6). Everything
    pollination-specific defaults in the task: the output CSV into p.es_shock_dir, the r50xAEZ
    boundary via p.get_path.
    """
    dynamic = 'pollination' in getattr(p, 'dynamic_es', [])
    if not dynamic:   # not requested dynamic -> read the frozen dependency table
        p.pollination_shock_task = p.add_task(pollination_tasks.pollination_shock_static, parent=parent)
        return p
    # dynamic: recompute from the SEALS maps (one task for pollination; cf. erosion's multi-task chain)
    p.pollination_shock_task = p.add_task(pollination_tasks.pollination_shock, parent=parent)
    return p

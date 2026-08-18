"""Pollination wiring: GEP task trees + the ES-shock seam (global_invest module layout).

GEP side: the valuation consumes the crop_benefits value raster
(poll_value_global_<gep_base_year>usd.tif -- USD per cell, crop prices and pollination dependence
embedded upstream, so lambda = 1) and aggregates it to one row per country on r250.
Shock side: consumers (ngfs_pnas, nff_global) call add_pollination_tasks(p) after their SEALS
stitch task; it dispatches static vs dynamic on p.dynamic_es (mirrors add_terrestrial_carbon_tasks).
"""
from global_invest import utilities
from global_invest.pollination import pollination_tasks


def initialize_paths(p):
    """Resolve the pollination GEP inputs on p via get_path (machine-agnostic reference paths;
    one source of truth for every run file and the results report).

    gep_base_year and the value-raster reference come from es_config.csv (defaults layer: a
    caller-set value wins). They default to 2023 as a PAIR -- the crop_benefits value rasters on
    hand are 2023/2024. The GEP manuscript's base year is 2019: regenerate
    poll_value_global_2019usd.tif with the crop_benefits recipe and update BOTH cells of the
    pollination row (gep_base_year and gep_quantity_input_path name the year twice) before
    quoting a manuscript-aligned number.
    """
    utilities.hydrate_es_config(p, 'pollination')
    utilities.initialize_country_paths(p)   # shared r264 block (csv/gpkg/simplified + df_countries)
    return p


def build_gep_service_calculation_task_tree(p):
    """GEP calculation tree: value raster -> per-r264 sums -> r250 one-row-per-country valuation."""
    p.pollination_value_by_region = p.add_task(
        pollination_tasks.pollination_value_by_region, skip_existing=1)
    p.task_gep_calculation = p.add_task(pollination_tasks.gep_calculation)
    return p


def build_gep_service_results_task_tree(p):
    """Results-only run: load a PRIOR calculation's results and render the report (fails loudly
    if the calculation has not run; does NOT recompute)."""
    p.pollination_gep_load_results_task = p.add_task(pollination_tasks.gep_load_results)
    p.pollination_gep_result_task = p.add_task(pollination_tasks.gep_result)
    return p


def build_gep_service_task_tree(p):
    """Full GEP run: the calculation chain plus the results/report task."""
    p = build_gep_service_calculation_task_tree(p)
    p.pollination_gep_result_task = p.add_task(pollination_tasks.gep_result)
    return p


# ---------------------------------------------------------------------------------------------
# NGFS ES-shock wiring. Everything above builds the GEP task trees; this builds the ES-shock one.
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

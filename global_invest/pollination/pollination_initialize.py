"""Pollination wiring: GEP task trees + the ES-shock seam (global_invest module layout).

GEP side: the valuation builds the pollination value raster here
(pollination_value_raster_rebuilt: production times producer price times each crop's dependence
on animal pollination, written as USD in the cell), weights it by habitat sufficiency and
aggregates the reported variant to one row per country on r250. The author's raster carries USD
per square kilometre; a zonal sum over that adds densities and yields $18.28bn where the
area-weighted total is $476.29bn. Building the raster as USD in the cell is what makes the zonal
sum the total.
Shock side: consumers (ngfs_pnas, nff_global) call add_pollination_tasks(p) after their SEALS
stitch task; it dispatches static vs dynamic on p.dynamic_es (mirrors add_terrestrial_carbon_tasks).
"""
from global_invest import utilities
from global_invest.pollination import pollination_tasks


def build_gep_service_calculation_task_tree(p):
    """GEP calculation tree: value raster -> per-r264 sums -> r250 one-row-per-country valuation.

    fao_median_prices builds the per-crop median price the value raster is priced at, from the
    FAOSTAT bulks the download task stages. Both carry skip_existing=1: one is a download, the
    other is heavy and deterministic once its inputs exist."""
    p.download_pollination_inputs_task = p.add_task(
        utilities.download_inputs_task('pollination'), skip_existing=1)
    p.fao_median_prices_task = p.add_task(pollination_tasks.fao_median_prices, skip_existing=1)
    # The yield and production chain: Monfreda 2000 yields carried to the base year by FAO
    # country ratios, then multiplied by CropGrids harvested area. These are what the rebuilt
    # value raster is priced from.
    p.fao_yield_change_task = p.add_task(pollination_tasks.fao_yield_change, skip_existing=1)
    # No skip_existing on these two: each guards itself per crop, so a run that stopped part way
    # resumes at the crop it reached. skip_existing works on the task's directory, which exists as
    # soon as the first raster lands, so an interrupted run would come back and skip the other 145.
    p.pollination_yield_rasters_task = p.add_task(pollination_tasks.pollination_yield_rasters)
    p.pollination_production_rasters_task = p.add_task(
        pollination_tasks.pollination_production_rasters)
    p.pollination_source_value_raster_task = p.add_task(
        pollination_tasks.pollination_source_value_raster, skip_existing=1)
    p.pollination_value_raster_task = p.add_task(
        pollination_tasks.pollination_value_raster, skip_existing=1)
    # Our own construction of the value raster, which the account reports, and the comparison
    # against the author's. The comparison is the only check in the library that can disagree
    # with an author on method rather than on transcription, so it runs every time.
    # No skip_existing, same reasoning as sufficiency_weighted below: the directory exists
    # before the body runs, so a part-way failure wedges every later resume. The body's own
    # output checks say whether the work is done.
    p.pollination_value_raster_rebuilt_task = p.add_task(
        pollination_tasks.pollination_value_raster_rebuilt)
    p.pollination_value_independence_check_task = p.add_task(
        pollination_tasks.pollination_value_independence_check, skip_existing=1)
    # The weighting sits BEFORE the regional summary: the account's selected variant is the
    # sufficiency-weighted raster, and the country table is built from whichever raster
    # pollination_reported_variant names, so the weighted raster must exist first.
    # No skip_existing: that flag reads an existing task directory as work already done, and the
    # directory is created before the task body runs, so a run that fails part way through leaves
    # an empty directory every later run would take for a finished one. This task checks for the
    # two files it writes instead, which is what says whether it has run.
    p.pollination_sufficiency_weighted_task = p.add_task(
        pollination_tasks.pollination_sufficiency_weighted)
    p.pollination_value_by_region_task = p.add_task(
        pollination_tasks.pollination_value_by_region, skip_existing=1)
    p.gep_calculation_task = p.add_task(pollination_tasks.gep_calculation)
    return p


def build_gep_service_results_task_tree(p):
    """Results-only run: load a PRIOR calculation's results and render the report (fails loudly
    if the calculation has not run; does NOT recompute)."""
    p.pollination_gep_load_results_task = p.add_task(pollination_tasks.gep_load_results)
    p.pollination_gep_result_task = p.add_task(pollination_tasks.gep_result)
    return p


def build_gep_service_task_tree(p):
    """Full GEP run: the calculation plus the results/report task."""
    p = build_gep_service_calculation_task_tree(p)
    p.pollination_gep_result_task = p.add_task(pollination_tasks.gep_result)
    p.pollination_gep_results_distribution_task = p.add_task(
        pollination_tasks.gep_results_distribution)
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
    # Before the shock task and on both branches: es_shock_export reads this table whichever
    # pollination shock ran.
    p.pollination_zone_sector_values_task = p.add_task(
        pollination_tasks.pollination_zone_sector_values, parent=parent)
    dynamic = 'pollination' in getattr(p, 'dynamic_es', [])
    if not dynamic:   # not requested dynamic -> read the frozen dependency table
        p.pollination_shock_task = p.add_task(pollination_tasks.pollination_shock_static, parent=parent)
        return p
    # dynamic: recompute from the SEALS maps (one task for pollination; cf. erosion's multi-task chain)
    p.pollination_shock_task = p.add_task(pollination_tasks.pollination_shock, parent=parent)
    return p

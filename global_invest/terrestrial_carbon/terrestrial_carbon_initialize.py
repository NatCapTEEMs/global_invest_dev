from global_invest import utilities
from global_invest.terrestrial_carbon import terrestrial_carbon_tasks

def build_gep_service_preprocess_task_tree(p):
    """Base-data preprocessing: (re)build global_invest/terrestrial_carbon/spawn_total_biomass_carbon_2010.tif (scale raw
    Spawn to Mg C/ha, reproject, add aboveground+belowground). One-off job, kept OUT of the default run;
    run on demand, its product is promoted to base_data and consumed from there by the calculation tree.
    """
    p.terrestrial_carbon_gep_preprocess_task = p.add_task(terrestrial_carbon_tasks.gep_preprocess, skip_existing=1)
    return p


def build_gep_service_calculation_task_tree(p):
    """Build the default task tree for terrestrial carbon.

    The raw Spawn density build (convert dtype + combine aboveground/belowground) is a one-off
    base-data job, not part of the per-run tree -- its product (the total biomass-carbon density
    raster) is consumed from base_data by total_carbon_density.
    """
    # skip_existing=1 makes the chain re-runnable: each task's dir already present -> p.run_this=0 and
    # the task publishes its paths then returns early (cf. erosion's SDR chain).
    p.total_carbon_density = p.add_task(terrestrial_carbon_tasks.total_carbon_density, skip_existing=1)
    p.carbon_density_table = p.add_task(terrestrial_carbon_tasks.carbon_density_table, skip_existing=1)
    p.carbon_density_raster_base_year = p.add_task(terrestrial_carbon_tasks.carbon_density_raster_base_year, skip_existing=1)
    p.carbon_density_raster_per_cell_base_year = p.add_task(terrestrial_carbon_tasks.carbon_density_raster_per_cell_base_year, skip_existing=1)
    p.carbon_by_region = p.add_task(terrestrial_carbon_tasks.carbon_by_region, skip_existing=1)
    p.gep_calculation = p.add_task(terrestrial_carbon_tasks.gep_calculation)

    return p

def build_gep_service_results_task_tree(p):
    """Results-only ('load') run: load the GEP results from a PRIOR calculation run and render the report.
    Requires the calculation to have run already; fails loudly if the results are missing (does NOT recompute).
    """
    p.terrestrial_carbon_gep_load_results_task = p.add_task(terrestrial_carbon_tasks.gep_load_results)
    p.terrestrial_carbon_gep_result_task = p.add_task(terrestrial_carbon_tasks.gep_result)

    return p


def build_gep_service_task_tree(p):
    """Full GEP run: the calculation calculation plus the results/report task."""
    p = build_gep_service_calculation_task_tree(p)
    p.terrestrial_carbon_gep_result_task = p.add_task(terrestrial_carbon_tasks.gep_result)

    return p


# ---------------------------------------------------------------------------------------------
# ES-shock wiring (the consumer seam). Everything above builds the GEP task trees; this builds the ES-shock one.
# Both live here so a module has a single wiring file (cf. add_pollination_tasks / add_erosion_tasks
# / add_fisheries_tasks in their own <module>_initialization.py).
# ---------------------------------------------------------------------------------------------
def add_terrestrial_carbon_tasks(p, parent=None):
    """Graft the carbon ES-shock task onto p, dispatching STATIC vs DYNAMIC on p.dynamic_es.

    DYNAMIC ('terrestrial_carbon' in p.dynamic_es): recompute the carbon-density shock from our SEALS
    maps at each p.es_shock_years anchor (terrestrial_carbon_shock). STATIC (the default):
    read the frozen raw_dependencies/carbon_storage_dependency.csv
    (terrestrial_carbon_shock_static). Mirrors add_erosion_tasks / add_pollination_tasks;
    both paths write terrestrial_carbon_interpolated.csv.

    Caller sets only the shared es_shock_* config. Everything
    carbon-specific defaults in the task: the output CSV into p.es_shock_dir, the r50xAEZ boundary /
    Spawn density / carbon zones via p.get_path.
    """
    # skip_existing=1 makes the shock re-runnable via ProjectFlow (dir present -> p.run_this=0 and the task
    # publishes its output path then returns), matching the calc chain and erosion. Cached intermediates
    # inside the task are a second layer for partial re-runs.
    dynamic = 'terrestrial_carbon' in getattr(p, 'dynamic_es', [])
    if not dynamic:   # not requested dynamic -> read the frozen dependency table
        p.terrestrial_carbon_shock_task = p.add_task(terrestrial_carbon_tasks.terrestrial_carbon_shock_static, parent=parent, skip_existing=1)
        return p
    # dynamic: recompute from the SEALS maps (one task for carbon; cf. erosion's multi-task chain)
    p.terrestrial_carbon_shock_task = p.add_task(terrestrial_carbon_tasks.terrestrial_carbon_shock, parent=parent, skip_existing=1)
    return p

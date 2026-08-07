import pandas as pd
import hazelbean as hb

from global_invest.terrestrial_carbon import terrestrial_carbon_tasks

def initialize_paths(p):
    p.df_countries = pd.read_csv(p.df_countries_csv_path)

    # Notice optimization here: the GDFs are still just path_strings. hb.read_vector takes the string as an input and converts it to a GeoDataFrame when needed.
    p.gdf_countries = p.gdf_countries_vector_path
    p.gdf_countries_simplified = p.gdf_countries_vector_simplified_path

def build_gep_service_calculation_task_tree(p):
    """Build the default task tree for terrestrial carbon.

    The raw Spawn density build (convert dtype + combine aboveground/belowground) is a one-off
    base-data job, not part of the per-run tree -- its product (the total biomass-carbon density
    raster) is consumed from base_data by task_reproject_total_carbon_density.
    """
    # skip_existing=1 makes the chain re-runnable: each task's dir already present -> p.run_this=0 and
    # the task publishes its paths then returns early (cf. erosion's SDR chain).
    p.task_reproject_total_carbon_density = p.add_task(terrestrial_carbon_tasks.task_reproject_total_carbon_density, skip_existing=1)
    p.task_compute_carbon_density_table = p.add_task(terrestrial_carbon_tasks.task_compute_carbon_density_table, skip_existing=1)
    p.task_generate_carbon_density_raster_base_year = p.add_task(terrestrial_carbon_tasks.task_generate_carbon_density_raster_base_year, skip_existing=1)
    p.task_generate_carbon_density_raster_per_cell_base_year = p.add_task(terrestrial_carbon_tasks.task_generate_carbon_density_raster_per_cell_base_year, skip_existing=1)
    p.task_summarize_carbon_by_region = p.add_task(terrestrial_carbon_tasks.task_summarize_carbon_by_region, skip_existing=1)
    p.task_gep_calculation = p.add_task(terrestrial_carbon_tasks.gep_calculation)

    return p

def build_gep_service_task_tree(p):
    """Full GEP run: the calculation chain plus the results/report task."""
    p = build_gep_service_calculation_task_tree(p)
    p.terrestrial_carbon_gep_result_task = p.add_task(terrestrial_carbon_tasks.gep_result)

    return p


# ---------------------------------------------------------------------------------------------
# NGFS ES-shock wiring. Everything above builds the GEP task trees; this builds the ES-shock one.
# Both live here so a module has a single wiring file (cf. add_pollination_tasks / add_erosion_tasks
# / add_fisheries_tasks in their own <module>_initialization.py).
# ---------------------------------------------------------------------------------------------
def add_terrestrial_carbon_tasks(p, parent=None):
    """Graft the carbon ES-shock task onto p, dispatching STATIC vs DYNAMIC on p.dynamic_es.

    DYNAMIC ('terrestrial_carbon' in p.dynamic_es): recompute the carbon-density shock from our SEALS
    maps at each p.es_shock_years anchor (task_compute_terrestrial_carbon_shock). STATIC (the default):
    read the frozen raw_dependencies/carbon_storage_dependency.csv
    (task_compute_terrestrial_carbon_shock_static). Mirrors add_erosion_tasks / add_pollination_tasks;
    both paths write terrestrial_carbon_interpolated.csv.

    Caller sets only the shared es_shock_* config. Everything
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

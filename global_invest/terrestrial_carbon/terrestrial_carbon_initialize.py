import pandas as pd
import hazelbean as hb

from global_invest.terrestrial_carbon import terrestrial_carbon_tasks

def initialize_paths(p):
    p.df_countries = pd.read_csv(p.df_countries_csv_path)

    # Notice optimization here: the GDFs are still just path_strings. hb.read_vector takes the string as an input and converts it to a GeoDataFrame when needed.
    p.gdf_countries = p.gdf_countries_vector_path
    p.gdf_countries_simplified = p.gdf_countries_vector_simplified_path

    # p.gdf_countries = hb.read_vector(p.gdf_countries_vector_path)  # Read the vector file for the countries.
    # p.countries_simplified_gdf = hb.read_vector(p.countries_simplified_vector_path)  # Read the vector file for the countries.

def build_gep_service_calculation_task_tree(p):
    """Build the default task tree for terrestrial carbon."""
    p.task_convert_carbon_density_maps_dtype = p.add_task(terrestrial_carbon_tasks.task_convert_carbon_density_maps_dtype)
    p.task_combine_two_carbon_density_maps = p.add_task(terrestrial_carbon_tasks.task_combine_two_carbon_density_maps)
    p.task_reproject_total_carbon_density = p.add_task(terrestrial_carbon_tasks.task_reproject_total_carbon_density)
    p.task_compute_carbon_density_table = p.add_task(terrestrial_carbon_tasks.task_compute_carbon_density_table)
    p.task_generate_carbon_density_raster_base_year = p.add_task(terrestrial_carbon_tasks.task_generate_carbon_density_raster_base_year)
    p.task_generate_carbon_density_raster_per_cell_base_year = p.add_task(terrestrial_carbon_tasks.task_generate_carbon_density_raster_per_cell_base_year)
    p.task_summarize_carbon_by_region = p.add_task(terrestrial_carbon_tasks.task_summarize_carbon_by_region)
    p.task_gep_calculation = p.add_task(terrestrial_carbon_tasks.gep_calculation)

    return p

def build_gep_service_results_task_tree(p):
    """Build the default task tree for terrestrial carbon."""
    p.terrestrial_carbon_gep_result_task = p.add_task(terrestrial_carbon_tasks.gep_result)

    return p


def build_gep_service_task_tree(p):
    """If you just want to load results, eg for reporting, this task tree inspects a different task tree and to learn paths and then loads results."""


    # QUESTION!!!! If a task truly already inspects itself to not rerun, what's the difference between loading and just executing the tree on
    # an existing project? The difference is that load will do more error checking and FAIL rather than recalculate if it didn't find, also reporting
    # that it didn't find it and giving information about how to put the data in so it does find it in the base data or a manually-built project data.
    # I might want to have methods for automatically putting an archive into the right spot and also extended functionality for finding results in base_data
    # and functionality for promoting project results to base data per the new documentation in ee_dev.
    # Actually, maybe it's just that load_results is more useful for notebooks?

    p = build_gep_service_calculation_task_tree(p)
    p.terrestrial_carbon_gep_result_task = p.add_task(terrestrial_carbon_tasks.gep_result)

    return p


def build_gep_task_tree(p):
    """
    Build the default task tree forthe GEP application. In this case, it's very similar to the standard task tree
    but i've included it here for consistency with other models.
    """
    p.terrestrial_carbon_gep_preprocess_task = p.add_task(terrestrial_carbon_tasks.gep_preprocess, parent=p.terrestrial_carbon_task)
    p.terrestrial_carbon_gep_calculation_task = p.add_task(terrestrial_carbon_tasks.gep_calculation, parent=p.terrestrial_carbon_task)
    p.terrestrial_carbon_gep_result_task = p.add_task(terrestrial_carbon_tasks.gep_result, parent=p.terrestrial_carbon_task)
    p.terrestrial_carbon_gep_results_distribution_task = p.add_task(terrestrial_carbon_tasks.gep_results_distribution, parent=p.terrestrial_carbon_task)
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

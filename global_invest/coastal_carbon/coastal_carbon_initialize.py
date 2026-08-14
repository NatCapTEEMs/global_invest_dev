"""coastal_carbon wiring -- READ BEFORE CHANGING THIS MODULE.

The copy of coastal_carbon on develop is an early clone of terrestrial_carbon and CANNOT run
(reviewed file by file 2026-08; findings verified against the code, not fixed here -- see below for why):
- build_gep_service_calculation_task_tree registers SEVEN tasks that do not exist in
  coastal_carbon_tasks.py (task_convert_carbon_density_maps_dtype through
  task_summarize_carbon_by_region) -> AttributeError at build time.
- gep_calculation consumes p.carbon_by_region_base_year_path, which no coastal task writes
  (a dangling clone of the terrestrial chain), so the valuation has no quantity input.
- task_calculate_mangrove_area_within_countries reads p.gdf_countries__marine_vector_path
  (double underscore) while the run file sets p.gdf_countries_marine_vector_path; it also
  attaches country geometry to a groupby result by row position (misaligned geometries),
  writes into p.project_dir instead of p.cur_dir, and has no run_this guard.
- initialize_paths reads p.df_countries_csv_path, which run_coastal_carbon.py never sets
  (it sets the marine r566 paths) -> AttributeError in the main runner.
- gep_result renders a coastal_carbon_results.qmd that does not exist in this module, and
  creates directories at the module source path; gep_load_results is a temp-tree stub.
- coastal_carbon_functions.py (63 KB) is never called from any task: it holds one-off
  tile-merge/preprocessing scripts (its import, together with rasterstats -- absent from
  hazelbean_env, so the tasks module could not even be imported -- was pruned on this branch).
  developing.py is scratch that executes hardcoded personal paths at import. Both are dead
  weight in this copy.
- build_gep_service_task_tree lacks `return p` (same bug fixed in terrestrial_carbon).

The `develop_yanxu` branch carries a full REWORK of this module that supersedes this copy:
per-ecosystem task trees (mangrove + salt marsh implemented, seagrass stubbed) composed into
build_gep_service_calculation_task_tree, real area->stock->storage-value chains, a
coastal_carbon_results.qmd + references.bib, developing.py deleted, the functions file cut to
what the tasks use, and a gep_calculation that already enforces the r250-only rule via the
canonical `ee_r264_label == iso3_r250_label` filter (see global_invest/utilities.py).

THEREFORE this branch deliberately does NOT restructure this module in parallel -- that would
duplicate the rework and guarantee add/add merge conflicts. Changes here are limited to the
national-GEP double-count fix and the results-contract fix in gep_calculation (needed while
this copy is live) plus this review record. Fold-in recipe when the rework merges:
- take the develop_yanxu side wholesale for tasks/initialization/functions/run files;
- re-check its gep_calculation keeps the r250 canonical filter (it does at time of writing);
- conform its manual `_task_outputs_exist` guards to ProjectFlow-native skip
  (`p.add_task(..., skip_existing=1)` + `if not p.run_this: return` after publishing paths);
- de-hardcode the personal base_data_dir in its run files (resolve via machine.env/get_path);
- this file was RENAMED from coastal_carbon_initialization.py on this branch (to match the
  other services' `_initialize.py`), so the rework's edits to the old filename arrive as a
  modify/delete conflict: resolve by taking the rework's CONTENT into THIS filename and
  updating its imports; then update coastal_carbon/test_coastal_carbon.py to the reworked
  valuation interface.
"""
import pandas as pd
import hazelbean as hb

from global_invest.coastal_carbon import coastal_carbon_tasks

def initialize_paths(p):
    p.df_countries = pd.read_csv(p.df_countries_csv_path)

    # Notice optimization here: the GDFs are still just path_strings. hb.read_vector takes the string as an input and converts it to a GeoDataFrame when needed.
    p.gdf_countries = p.gdf_countries_vector_path
    p.gdf_countries_simplified = p.gdf_countries_vector_simplified_path

    # p.gdf_countries = hb.read_vector(p.gdf_countries_vector_path)  # Read the vector file for the countries.
    # p.countries_simplified_gdf = hb.read_vector(p.countries_simplified_vector_path)  # Read the vector file for the countries.

def build_gep_service_calculation_task_tree(p):
    """Build the default task tree for terrestrial carbon."""
    p.task_convert_carbon_density_maps_dtype = p.add_task(coastal_carbon_tasks.task_convert_carbon_density_maps_dtype)
    p.task_combine_two_carbon_density_maps = p.add_task(coastal_carbon_tasks.task_combine_two_carbon_density_maps)
    p.task_reproject_total_carbon_density = p.add_task(coastal_carbon_tasks.task_reproject_total_carbon_density)
    p.task_compute_carbon_density_table = p.add_task(coastal_carbon_tasks.task_compute_carbon_density_table)
    p.task_generate_carbon_density_raster_base_year = p.add_task(coastal_carbon_tasks.task_generate_carbon_density_raster_base_year)
    p.task_generate_carbon_density_raster_per_cell_base_year = p.add_task(coastal_carbon_tasks.task_generate_carbon_density_raster_per_cell_base_year)
    p.task_summarize_carbon_by_region = p.add_task(coastal_carbon_tasks.task_summarize_carbon_by_region)
    p.task_gep_calculation = p.add_task(coastal_carbon_tasks.gep_calculation)

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
    p.coastal_carbon_gep_result_task = p.add_task(coastal_carbon_tasks.gep_result)


# def build_gep_task_tree(p):
#     """
#     Build the default task tree forthe GEP application of commercial agriculture. In this case, it's very similar to the standard task tree
#     but i've included it here for consistency with other models.
#     """
#     p.coastal_carbon_gep_preprocess_task = p.add_task(coastal_carbon_tasks.gep_preprocess, parent=p.coastal_carbon_task)
#     p.coastal_carbon_gep_calculation_task = p.add_task(coastal_carbon_tasks.gep_calculation, parent=p.coastal_carbon_task)
#     p.coastal_carbon_gep_result_task = p.add_task(coastal_carbon_tasks.gep_result, parent=p.coastal_carbon_task)
#     p.coastal_carbon_gep_results_distribution_task = p.add_task(coastal_carbon_tasks.gep_results_distribution, parent=p.coastal_carbon_task)
#     return p


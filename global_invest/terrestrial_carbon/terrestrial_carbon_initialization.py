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


# def build_gep_task_tree(p):
#     """
#     Build the default task tree forthe GEP application of commercial agriculture. In this case, it's very similar to the standard task tree
#     but i've included it here for consistency with other models.
#     """
#     p.terrestrial_carbon_gep_preprocess_task = p.add_task(terrestrial_carbon_tasks.gep_preprocess, parent=p.terrestrial_carbon_task)
#     p.terrestrial_carbon_gep_calculation_task = p.add_task(terrestrial_carbon_tasks.gep_calculation, parent=p.terrestrial_carbon_task)
#     p.terrestrial_carbon_gep_result_task = p.add_task(terrestrial_carbon_tasks.gep_result, parent=p.terrestrial_carbon_task)
#     p.terrestrial_carbon_gep_results_distribution_task = p.add_task(terrestrial_carbon_tasks.gep_results_distribution, parent=p.terrestrial_carbon_task)
#     return p


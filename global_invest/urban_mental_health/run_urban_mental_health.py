import os, sys
import pandas as pd
import hazelbean as hb
from global_invest.urban_mental_health import urban_mental_health_tasks


# Create the project flow object
p = hb.ProjectFlow()


# Set directories
p.user_dir = os.path.expanduser('~')
p.extra_dirs = ['Files', 'global_invest', 'projects']
p.project_name = 'urban_mental_health_' + hb.pretty_time()  # Fixed circular reference
p.project_dir = os.path.join(p.user_dir, os.sep.join(p.extra_dirs), p.project_name)
p.set_project_dir(p.project_dir)

# Set base_data_dir.
p.base_data_dir = os.path.join(p.user_dir, 'Files', 'base_data')

# Set model paths for global processing
p.aoi = 'global'

p.base_year_lulc_path = p.get_path(os.path.join(p.base_data_dir, 'lulc/esa/lulc_esa_2019.tif'))  # ESA CCI 300m LULC
p.counterfactual_lulc_path = p.get_path(os.path.join(p.base_data_dir, 'lulc/esa/lulc_esa_2010.tif'))  # ESA CCI 300m LULC
p.population_2019_path = p.get_path(os.path.join(p.base_data_dir, 'population/worldpop/global_pop_2019_CN_1km_R2025A_UA_v1.tif'))  # WorldPop 1km population
p.lulc_attribute_table_path = p.get_path(os.path.join(p.base_data_dir, 'lulc/esa/esa_cci_attribute_table_processed.csv'))
p.effect_size_table_path = p.get_path(os.path.join(p.base_data_dir, 'files_specific_to_urban-mental-health/effect-size-table.xlsx'))
p.baseline_prevalence_rate = 0.05
p.urban_boundary_path = p.get_path(os.path.join(p.base_data_dir, 'cartographic/ee/urban_boundaries_2019.gpkg'))
p.health_cost_rate_path = p.get_path(os.path.join(p.base_data_dir, 'files_specific_to_urban-mental-health/cost_mental-health.xlsx'))


def build_task_tree(p):
    p.task_convert_population_raster_dtype = p.add_task(urban_mental_health_tasks.task_convert_population_raster_dtype)
    p.task_reproject_population_raster = p.add_task(urban_mental_health_tasks.task_reproject_population_raster)
    p.task_resample_lulc_to_population_grid = p.add_task(urban_mental_health_tasks.task_resample_lulc_to_population_grid)
    p.task_convert_lulc_to_ndvi_baseline = p.add_task(urban_mental_health_tasks.task_convert_lulc_to_ndvi_baseline)
    p.task_convert_lulc_to_ndvi_scenario = p.add_task(urban_mental_health_tasks.task_convert_lulc_to_ndvi_scenario)
    p.task_calculate_delta_nature_exposure = p.add_task(urban_mental_health_tasks.task_calculate_delta_nature_exposure)
    p.task_calculate_preventable_cases = p.add_task(urban_mental_health_tasks.task_calculate_preventable_cases)
    p.task_aggregate_preventable_cases_by_region = p.add_task(urban_mental_health_tasks.task_aggregate_preventable_cases_by_region)
    p.task_calculate_country_costs = p.add_task(urban_mental_health_tasks.task_calculate_country_costs)

# Build the task tree and execute it!
build_task_tree(p)
p.execute()

print("URBAN MENTAL HEALTH MODEL COMPLETED")
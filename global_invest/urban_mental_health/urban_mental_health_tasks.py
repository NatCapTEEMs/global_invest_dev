import os
import numpy as np
import rasterio
from global_invest.carbon import carbon_functions
from global_invest.urban_mental_health import urban_mental_health_functions


def task_convert_population_raster_dtype(p):
    """
    Task to convert population raster from uint to float32.
    """
    # Convert 2019 population raster
    p.population_2019_float_path = os.path.join(p.project_dir, "pop_2019_float.tif")

    with rasterio.open(p.population_2019_path) as src:
        if src.dtypes[0] == 'float32':
            print("Population raster is already in float32 format.")
            p.population_2019_float_path = p.population_2019_path
        else:
            carbon_functions.convert_uint_to_float_raster(
                input_path=p.population_2019_path,
                output_path=p.population_2019_float_path,
                scale_factor=1.0,  # WorldPop 1km data represents actual population counts and doesn't need scaling
                compress="lzw"
                )

    print(f"Float population raster saved to {p.population_2019_float_path}.")
    return p.population_2019_float_path


def task_reproject_population_raster(p):
    """
    Task to reproject population raster to match LULC reference CRS and resolution.
    """
    p.population_2019_float_path = p.get_path(os.path.join(p.project_dir, "pop_2019_float.tif"))
    print(f"Input 2019 population raster for reprojection: {p.population_2019_float_path}")
    p.population_reprojected_path = os.path.join(p.project_dir, "pop_2019_reprojected.tif")

    result = carbon_functions.reproject_raster(
        input_path=p.population_2019_float_path,
        reference_path=p.base_year_lulc_path,
        output_path=p.population_reprojected_path,
        compress="lzw",
        chunks={"x": 1024, "y": 1024},
        overwrite=True
    )

    print(f"Reprojected population raster saved to {p.population_reprojected_path}")
    return p.population_reprojected_path


def task_resample_lulc_to_population_grid(p):
    
    p.lulc_resampled_baseline_path = os.path.join(p.project_dir, "lulc_2019_resampled.tif")
    p.lulc_resampled_scenario_path = os.path.join(p.project_dir, "lulc_2010_resampled.tif")

    # Use nearest neighbor for categorical LULC data
    from rasterio.enums import Resampling

    # Resample baseline LULC
    urban_mental_health_functions.resample_raster_to_reference(
        input_path=p.base_year_lulc_path,
        reference_path=p.population_reprojected_path,
        out_path=p.lulc_resampled_baseline_path,
        resampling_method=Resampling.nearest,  # preserve categorical values
        compress="lzw"
    )
    print(f"Finished resampling baseline LULC to population grid: {p.lulc_resampled_baseline_path}")

    # Resample scenario LULC
    urban_mental_health_functions.resample_raster_to_reference(
        input_path=p.counterfactual_lulc_path,
        reference_path=p.population_reprojected_path,
        out_path=p.lulc_resampled_scenario_path,
        resampling_method=Resampling.nearest,  # preserve categorical values
        compress="lzw"
    )
    print(f"Finished resampling scenario LULC to population grid: {p.lulc_resampled_scenario_path}")

    return p.lulc_resampled_baseline_path, p.lulc_resampled_scenario_path


def task_convert_lulc_to_ndvi_baseline(p):
    """
    Task to convert baseline LULC (2019) to NDVI using processed attribute table.
    """
    p.ndvi_baseline_path = os.path.join(p.project_dir, 'ndvi_2019.tif')

    # Use resampled LULC (now at 1km resolution)
    lulc_path = getattr(p, 'lulc_resampled_baseline_path', p.base_year_lulc_path)
    print(f"Resampled LULC path for input: {lulc_path}")

    urban_mental_health_functions.map_lulc_to_ndvi(
        lulc_path=lulc_path,
        attr_table_path=p.lulc_attribute_table_path,
        ndvi_col='lc_ndvi',  # from processed attribute table
        out_path=p.ndvi_baseline_path,
        compress="lzw"
    )
    print(f"NDVI baseline raster saved to {p.ndvi_baseline_path}")

    return p.ndvi_baseline_path


def task_convert_lulc_to_ndvi_scenario(p):
    """
    Task to convert scenario LULC (2010) to NDVI using processed attribute table.
    """
    p.ndvi_scenario_path = os.path.join(p.project_dir, 'ndvi_2010.tif')

    # Use resampled LULC (now at 1km resolution)
    lulc_path = getattr(p, 'lulc_resampled_scenario_path', p.counterfactual_lulc_path)

    urban_mental_health_functions.map_lulc_to_ndvi(
        lulc_path=lulc_path,
        attr_table_path=p.lulc_attribute_table_path,
        ndvi_col='lc_ndvi',  # from processed attribute table
        out_path=p.ndvi_scenario_path,
        compress="lzw"
    )
    print(f"NDVI scenario raster saved to {p.ndvi_scenario_path}")

    return p.ndvi_scenario_path


def task_calculate_delta_nature_exposure(p):
    """
    Task to calculate delta nature exposure (NDVI_2019 - NDVI_2010).
    """
    p.ndvi_baseline_path = p.get_path(os.path.join(p.project_dir, 'ndvi_2019.tif'))
    p.ndvi_scenario_path = p.get_path(os.path.join(p.project_dir, 'ndvi_2010.tif'))
    p.delta_ne_path = os.path.join(p.project_dir, 'delta_ne_2019_vs_2010.tif')

    urban_mental_health_functions.calculate_delta_raster(
        raster1_path=p.ndvi_baseline_path,
        raster2_path=p.ndvi_scenario_path,
        out_path=p.delta_ne_path,
        operation=lambda a, b: a - b,  # NDVI_2019 - NDVI_2010
        fill_value=np.nan,
        compress="lzw"
    )
    print(f"Delta NDVI raster saved to {p.delta_ne_path}")

    return p.delta_ne_path


def task_calculate_preventable_cases(p):
    """
    Task to calculate preventable cases per pixel using delta nature exposure and population.
    Both datasets now at 1km resolution for perfect alignment.
    """
    p.delta_ne_path = p.get_path(os.path.join(p.project_dir, 'delta_ne_2019_vs_2010.tif'))

    # Use reprojected population (1km resolution)
    pop_path = p.get_path(os.path.join(p.project_dir, "pop_2019_reprojected.tif"))

    p.preventable_cases_path = os.path.join(p.project_dir, 'preventable_cases_2019.tif')

    urban_mental_health_functions.calculate_preventable_cases(
        delta_ne_path=p.delta_ne_path,
        pop_path=pop_path,
        effect_size_table_path=p.effect_size_table_path,
        prevalence=p.baseline_prevalence_rate,
        out_path=p.preventable_cases_path,
        compress="lzw"
    )
    print(f"Preventable cases raster saved to: {p.preventable_cases_path}")

    return p.preventable_cases_path


def task_aggregate_preventable_cases_by_region(p):
    """
    Task to aggregate preventable cases by urban regions.
    """
    p.preventable_cases_path = p.get_path(os.path.join(p.project_dir, 'preventable_cases_2019.tif'))
    p.preventable_cases_by_region_csv = os.path.join(p.project_dir, 'preventable_cases_by_region.csv')

    result = urban_mental_health_functions.aggregate_preventable_cases_by_region(
        preventable_cases_raster_path=p.preventable_cases_path,
        urban_region_boundary_path=p.urban_boundary_path,
        out_csv_path=p.preventable_cases_by_region_csv
    )

    # Return CSV path
    return result


def task_calculate_country_costs(p):
    """
    Task to apply country-specific cost rates to aggregated preventable cases.
    """
    p.preventable_cases_by_region_csv = p.get_path(os.path.join(p.project_dir, 'preventable_cases_by_region.csv'))
    p.preventable_cost_by_country_csv = os.path.join(p.project_dir, 'preventable_cost_by_country.csv')

    result = urban_mental_health_functions.apply_country_costs(
        regional_cases_csv_path=p.preventable_cases_by_region_csv,
        health_cost_rate_path=p.health_cost_rate_path,
        out_country_csv_path=p.preventable_cost_by_country_csv
    )

    # Return tuple of paths
    return result
# =============================================================================
# imports
# =============================================================================

import os
import numpy as np

from global_invest.carbon import carbon_functions


# =============================================================================
# define tasks
# =============================================================================

def task_convert_carbon_density_maps_dtype(p):
    """
    Task to convert all uint TIFF carbon density rasters in a folder
    to float32 with scaled values, saving them with '_float' suffix.

    Parameters
    ----------
    p : ProjectFlow-like object
        Must contain p.base_data_dir (input folder path).
    """
    input_folder = p.base_data_dir
    output_folder = p.project_dir

    raw_carbon_density_maps = [
        f for f in os.listdir(input_folder)
        if f.endswith("biomass_carbon_2010.tif") and not f.startswith("._")
    ]

    for file in raw_carbon_density_maps:
        input_path = os.path.join(input_folder, file)

        name_root, ext = os.path.splitext(file)
        output_name = f"{name_root}_float{ext}"
        output_path = os.path.join(output_folder, output_name)

        carbon_functions.convert_uint_to_float_raster(
            input_path=input_path,
            output_path=output_path,
            scale_factor=0.1,
            compress="lzw"
        )

    print("Finished converting all carbon density rasters to float.")



def task_combine_two_carbon_density_maps(p):
    """
    Task to combine aboveground and belowground biomass carbon maps using service function.
    """

    # Input and output paths
    p.agb_path = p.get_path(os.path.join(p.project_dir,"aboveground_biomass_carbon_2010_float.tif"))
    p.bgb_path = p.get_path(os.path.join(p.project_dir,"belowground_biomass_carbon_2010_float.tif"))
    p.total_carbon_output_path = os.path.join(p.project_dir, "total_biomass_carbon_2010_float.tif")

    # Run the function
    result = carbon_functions.combine_two_float_rasters(
        raster1_path=p.agb_path,
        raster2_path=p.bgb_path,
        out_path=p.total_carbon_output_path,
        operation=lambda a, b: a + b,  # Default operation: addition
        fill_value=np.nan,
        compress="lzw")

    return True


def task_reproject_total_carbon_density(p):
    """
    Task to reproject the total carbon density raster to the project's coordinate reference system (CRS).
    """

    # Input and output paths
    p.total_carbon_density_path = p.get_path(os.path.join(p.project_dir, "total_biomass_carbon_2010_float.tif"))
    p.reprojected_total_carbon_density_path = os.path.join(p.project_dir, "total_biomass_carbon_2010_float_reprojected.tif")

    # Run the function
    result = carbon_functions.reproject_raster(
        input_path=p.total_carbon_density_path,
        reference_path=p.base_year_lulc_path,
        output_path=p.reprojected_total_carbon_density_path,
        compress="lzw",
        chunks={"x": 1024, "y": 1024},
        overwrite=False
        )

    return True


def task_reproject_carbon_zones(p):
    """
    Task to reproject the total carbon density raster to the project's coordinate reference system (CRS).
    """

    # Input and output paths
    p.reprojected_carbon_zones_path = os.path.join(p.project_dir, "carbon_zones_rasterized_reprojected.tif")

    # Run the function
    result = carbon_functions.reproject_raster(
        input_path=p.carbon_zones_path,
        reference_path=p.base_year_lulc_path,
        output_path=p.reprojected_carbon_zones_path,
        compress="lzw",
        chunks={"x": 1024, "y": 1024},
        overwrite=False
        )
    return True


def task_compute_carbon_density_table(p):

    p.reprojected_total_carbon_density_path = p.get_path(os.path.join(p.project_dir, "total_biomass_carbon_2010_float_reprojected.tif"))
    p.carbon_density_lookup_table_path = os.path.join(p.project_dir, "carbon_density_lookup_table.csv")

    result = carbon_functions.stack_layers_to_csv(
        group_layer1_path=p.base_year_lulc_path,
        group_layer2_path=p.carbon_zones_path,
        value_layer_path=p.reprojected_total_carbon_density_path,
        output_path=p.carbon_density_lookup_table_path,
        group1_name="lulc_id",
        group2_name="carbon_zone_id",
        value_name="carbon_density",
        num_slices=100)
    return True


def task_generate_carbon_density_raster_base_year(p):
    p.reprojected_total_carbon_density_path = p.get_path(os.path.join(p.project_dir, "total_biomass_carbon_2010_float_reprojected.tif"))
    p.carbon_density_lookup_table_path = p.get_path(os.path.join(p.project_dir, "carbon_density_lookup_table.csv"))
    p.carbon_density_raster_output_path = os.path.join(p.project_dir, "carbon_density_2019.tif")
    result = carbon_functions.generate_carbon_density_raster(
        lulc_path=p.base_year_lulc_path,
        cz_path=p.carbon_zones_path,
        carbon_density_lookup_table_path=p.carbon_density_lookup_table_path,
        out_path=p.carbon_density_raster_output_path)
    return True


def task_summarize_carbon_density_by_region(p):
    p.carbon_density_raster_output_path = p.get_path(os.path.join(p.project_dir, "carbon_density_2019.tif"))
    p.carbon_density_by_region_path = os.path.join(p.project_dir, "carbon_density_by_region_2019.csv")
    result = carbon_functions.summarize_raster_by_region(
        value_raster_path=p.carbon_density_raster_output_path,
        region_boundary_path=p.region_boundary_path,
        out_path=p.carbon_density_by_region_path)
    return result


def task_print_hello(p):
    print("Hello World!")
    return True

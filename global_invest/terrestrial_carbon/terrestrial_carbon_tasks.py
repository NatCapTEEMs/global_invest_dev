import os
import sys
import pandas as pd
import hazelbean as hb # type: ignore
import subprocess
import csv
import numpy as np


from global_invest.terrestrial_carbon import terrestrial_carbon_functions
from global_invest.terrestrial_carbon import terrestrial_carbon_initialization


def terrestrial_carbon(p):
    """
    Parent task for terrestrial carbon.
    """
    return True


def task_convert_carbon_density_maps_dtype(p):
    """
    Task to convert all uint TIFF carbon density rasters in a folder
    to float32 with scaled values, saving them with '_float' suffix.

    Parameters
    ----------
    p : ProjectFlow
        Must contain p.base_data_dir (input folder path).
    """
    input_folder = p.get_path('terrestrial_carbon', 'spawn_2020')
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

        terrestrial_carbon_functions.convert_uint_to_float_raster(
            input_path=input_path,
            output_path=output_path,
            scale_factor=0.1,
            compress="lzw"
        )

    print("Finished converting all carbon density rasters to float.")



def task_combine_two_carbon_density_maps(p):
    """
    Task to combine aboveground and belowground biomass carbon maps using functions.
    """

    # Input and output paths
    p.agb_path = p.get_path(os.path.join(p.project_dir,"aboveground_biomass_carbon_2010_float.tif"))
    p.bgb_path = p.get_path(os.path.join(p.project_dir,"belowground_biomass_carbon_2010_float.tif"))
    p.total_carbon_output_path = os.path.join(p.project_dir, "total_biomass_carbon_2010_float.tif")

    # Run the function
    result = terrestrial_carbon_functions.combine_two_float_rasters(
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
    result = terrestrial_carbon_functions.reproject_raster(
        input_path=p.total_carbon_density_path,
        reference_path=p.base_year_lulc_path,
        output_path=p.reprojected_total_carbon_density_path,
        compress="lzw",
        chunks={"x": 1024, "y": 1024},
        overwrite=False
        )

    return True


def task_compute_carbon_density_table(p):

    p.reprojected_total_carbon_density_path = p.get_path(os.path.join(p.project_dir, "total_biomass_carbon_2010_float_reprojected.tif"))
    p.carbon_density_lookup_table_path = os.path.join(p.project_dir, "carbon_density_lookup_table.csv")

    result = terrestrial_carbon_functions.stack_layers_to_csv(
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
    p.carbon_density_raster_base_year_path = os.path.join(p.project_dir, "projected_carbon_density_maps_per_ha/projected_carbon_density_2019.tif")
    result = terrestrial_carbon_functions.generate_carbon_density_raster(
        lulc_path=p.base_year_lulc_path,
        cz_path=p.carbon_zones_path,
        carbon_density_lookup_table_path=p.carbon_density_lookup_table_path,
        out_path=p.carbon_density_raster_base_year_path)
    return True


def task_generate_carbon_density_raster_per_cell_base_year(p):
    p.ha_per_cell_10sec_ref_path = p.get_path('pyramids', 'ha_per_cell_10sec.tif')
    p.projected_carbon_density_2019_per_cell_path = os.path.join(p.project_dir, 'projected_carbon_density_maps_per_cell/projected_carbon_density_2019_per_cell.tif')
    hb.multiply(p.carbon_density_raster_base_year_path, p.ha_per_cell_10sec_ref_path, p.projected_carbon_density_2019_per_cell_path)
    return True


def task_summarize_carbon_density_by_region(p):
    p.carbon_density_by_region_path = os.path.join(p.project_dir, "gep_by_country_base_year.csv")
    result = terrestrial_carbon_functions.summarize_raster_by_region(
        value_raster_path=p.projected_carbon_density_2019_per_cell_path,
        region_boundary_path=p.gdf_countries_vector_path,
        out_path=p.carbon_density_by_region_path)
    return result


#%%

def gep_calculation(p):
    """ GEP calculation task for terrestrial carbon."""
    # Define at least the primary output for the service, which for this project is gep_by_country_base_year.
    service_results = {}
    p.results['terrestrial_carbon'] = service_results
    p.results['terrestrial_carbon']['gep_by_country_base_year'] = os.path.join(p.project_dir, "gep_by_country_base_year.csv")

    # Optional additional results.
    p.results['terrestrial_carbon']['gep_by_country_year'] = os.path.join(p.project_dir, "gep_by_country_year.csv")
    p.results['terrestrial_carbon']['gep_by_country_year'] = os.path.join(p.project_dir, "gep_by_country_year.csv")
    p.results['terrestrial_carbon']['gep_by_year'] = os.path.join(p.project_dir, "gep_by_year.csv")

    # Check if all results exist
    if hb.path_all_exist(list(service_results.values())):
        hb.log("All results already exist. Skipping GEP calculation for terrestrial carbon.")
    else:
        hb.log("Starting GEP calculation for terrestrial carbon.")

        # Optimization here,
        # p.gdf_countries = hb.read_vector(p.gdf_countries)
        # p.gdf_countries = hb.read_vector(p.gdf_countries_simplified)

        # 1. Read and process data
        df_carbon_q264 = pd.read_csv(os.path.join(p.project_dir, "gep_by_country_base_year.csv"))
        df_carbon_q250 = df_carbon_q264.groupby(['iso3_r250_id', 'year'])['total'].sum().reset_index()
        df_carbon_q = df_carbon_q250.rename(columns={'total': 'terrestrial_carbon_quantity'})
        df_carbon_p = pd.read_excel(p.carbon_prices_path)
        df_carbon_p = df_carbon_p[[p.carbon_price, 'year']]
        df_gep_by_country_base_year_terrestrial_carbon = df_carbon_q.merge(df_carbon_p,how='left',on='year')
        df_gep_by_country_base_year_terrestrial_carbon['terrestrial_carbon_gep'] = df_gep_by_country_base_year_terrestrial_carbon['terrestrial_carbon_quantity'] * df_gep_by_country_base_year_terrestrial_carbon[p.carbon_price]
        df_gep_by_country_base_year_terrestrial_carbon = df_gep_by_country_base_year_terrestrial_carbon.merge(df_carbon_q264,how='left',on='iso3_r250_id')
        df_gep_by_country_base_year_terrestrial_carbon['year'] = df_gep_by_country_base_year_terrestrial_carbon['year_x']

        cols_to_keep = [
            'ee_r264_id',
            'iso3_r250_id',
            'ee_r264_label',
            'iso3_r250_label',
            'ee_r264_name',
            'iso3_r250_name',
            'continent',
            'region_un',
            'region_wb',
            'income_grp',
            'subregion',
            'year',
            'terrestrial_carbon_quantity',
            p.carbon_price,
            'terrestrial_carbon_gep',
        ]

        df_gep_by_country_base_year = df_gep_by_country_base_year_terrestrial_carbon[cols_to_keep]


        # Write to CSVs
        hb.df_write(df_gep_by_country_base_year, p.results['terrestrial_carbon']['gep_by_country_base_year'])

        # Use geopandas to merge the df_gep_by_country_base_year with the  to get the country names and other attributes
        gdf_gep_by_country_base_year = hb.df_merge(p.gdf_countries_simplified, df_gep_by_country_base_year, how='outer', left_on='ee_r264_id', right_on='ee_r264_id')
        gdf_gep_by_country_base_year.to_file(p.results['terrestrial_carbon']['gep_by_country_base_year'].replace('.csv', '.gpkg'), driver='GPKG')

        # Then sum the values across all countries.
        value_gep_base_year = df_gep_by_country_base_year['terrestrial_carbon_gep'].sum()

        hb.log(f"Total GEP value for base year 2019: {value_gep_base_year}")

        return value_gep_base_year

def gep_result(p):
    """Display the results of the GEP calculation."""

    # Set the quarto path to wherever the current script is running. This means that the environment used needs to have quarto, which may not be true on e.g. codespaces.
    os.environ['QUARTO_PYTHON'] = sys.executable

    # Get the  list of current services run
    services_run = list(p.results.keys())

    # Additional groupbys = []

    # Imply from the service name the file_path for the results_qmd
    # module_root = hb.get_projectflow_module_root()

    for service_label in services_run:
        print(service_label)
        results_qmd_path = os.path.join(p.project_dir, f'{service_label}_results.qmd')
        #results_qmd_path = os.path.join(module_root, service_label, f'{service_label}_results.qmd')
        results_qmd_project_path = os.path.join(p.cur_dir, f'{service_label}_results.qmd')
        hb.create_directories(results_qmd_path)  # Ensure the directory exists

        # Copy it to the project dir for cmd line processing (but will be removed again later because it makes confusion when people try to edit it and then rerun the script which won't of course update the results.)
        # hb.path_copy(results_qmd_path, results_qmd_project_path)
        # hb.path_copy(results_qmd_path, results_qmd_project_path)

        quarto_command = f"quarto render {results_qmd_project_path}"
        hb.log(f"Running quarto command: {quarto_command}")

        """Run quarto with debug information"""
        # Set environment for more verbose output
        env = os.environ.copy()
        env['QUARTO_LOG_LEVEL'] = 'DEBUG'

        cmd = ['quarto', 'render', results_qmd_project_path, '--verbose']

        # print(f"Running command: {' '.join(results_qmd_project_path)}")
        print(f"Working directory: {os.getcwd()}")
        print(f"File exists: {os.path.exists(results_qmd_project_path)}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr into stdout
            text=True,
            bufsize=1,  # Line buffering
            universal_newlines=True
        )

        # Read line by line as they come
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.rstrip())
                sys.stdout.flush()  # Force immediate display
        # remove results_qmd_project_path
        hb.path_remove(results_qmd_project_path)

def gep_load_results(p):

    # Learn the paths by creating a temp task treep
    p_temp = hb.ProjectFlow()
    terrestrial_carbon_initialization.build_gep_service_calculation_task_tree(p_temp)
    p_temp.set_all_tasks_to_skip_if_dir_exists()
    p_temp.execute()

    print(p_temp.results)
    pass

def gep_results_distribution(p):
    """Distribute the results of the GEP calculation."""
    # This task is intended to copy the results to the output directory.
    hb.log("Distributing GEP results...")

    for key, value in p.results['terrestrial_carbon'].items():
        output_path = os.path.join(p.output_dir, key)
        hb.path_copy(value, output_path)
        hb.log(f"Distributed {key} to {output_path}")

    hb.log("GEP results distribution complete.")



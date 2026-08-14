# Only what the tasks below use. The old header also imported rasterstats (not in hazelbean_env, so
# the module could not even be imported), rasterio, gdal/ogr, numpy, csv, tqdm and the unused
# coastal_carbon_functions -- all dead weight from the terrestrial clone.
import os
import sys
import subprocess

import pandas as pd
import geopandas as gpd
import hazelbean as hb

from global_invest.coastal_carbon import coastal_carbon_initialization


def coastal_carbon(p):
    """
    Parent task for terrestrial carbon.
    """
    return True


def task_calculate_mangrove_area_within_countries(p):

    gdf_countries_marine_vector = gpd.read_file(p.gdf_countries__marine_vector_path)

    gdf_mangroves = gpd.read_file(p.mangrove_vector_path)

    gdf_mangroves = gdf_mangroves.to_crs(gdf_countries_marine_vector.crs)

    gdf_mangroves_within_countries = gpd.overlay(gdf_mangroves, gdf_countries_marine_vector, how='intersection')

    gdf_mangroves_within_countries = gdf_mangroves_within_countries.to_crs(gdf_countries_marine_vector.crs)

    gdf_mangroves_within_countries = gdf_mangroves_within_countries.to_crs(epsg=6933)

    gdf_mangroves_within_countries["area_m2"] = gdf_mangroves_within_countries.geometry.area
    gdf_mangroves_within_countries["area_ha"] = gdf_mangroves_within_countries["area_m2"] / 10000

    gdf_mangroves_within_countries_base_year_path = os.path.join(p.project_dir, "mangroves_within_countries2019.gpkg")
    gdf_mangroves_within_countries.to_file(gdf_mangroves_within_countries_base_year_path, driver="GPKG")

    mangrove_area_by_countries_base_year = gdf_mangroves_within_countries.groupby("eemarine_r566_id")["area_ha"].sum().reset_index()

    mangrove_area_by_countries_base_year = mangrove_area_by_countries_base_year.merge(
        gdf_countries_marine_vector, how="left", on="eemarine_r566_id"
    )

    mangrove_area_by_countries_base_year = gpd.GeoDataFrame(
        mangrove_area_by_countries_base_year,
        geometry=gdf_countries_marine_vector.geometry,  # make sure geometries are aligned
        crs=gdf_countries_marine_vector.crs             # preserve CRS
    )

    p.mangrove_area_by_countries_base_year_path = os.path.join(p.project_dir, "mangrove_area_by_countries2019.gpkg")


    mangrove_area_by_countries_base_year.to_file(
        p.mangrove_area_by_countries_base_year_path, driver="GPKG"
    )

    print("done saving mangrove area by countries")









def gep_calculation(p):
    """ GEP calculation task for terrestrial carbon."""
    # Define at least the primary output for the service, which for this project is gep_by_country_base_year.
    service_results = {}
    p.results['coastal_carbon'] = service_results
    p.results['coastal_carbon']['gep_by_country_base_year'] = os.path.join(p.cur_dir, "gep_by_country_base_year.csv")
    # Only register results this task actually writes. Per-year results (gep_by_country_year, gep_by_year)
    # belong to a multi-year run and are registered there, not in this base-year valuation. Registering
    # them here (as before) broke the path_all_exist skip below and the results contract downstream.

    # Check if all results exist
    if hb.path_all_exist(list(service_results.values())):
        hb.log("All results already exist. Skipping GEP calculation for coastal carbon.")
    else:
        hb.log("Starting GEP calculation for coastal carbon.")

        # Optimization here,
        # p.gdf_countries = hb.read_vector(p.gdf_countries)
        # p.gdf_countries = hb.read_vector(p.gdf_countries_simplified)

        # 1. Read and process data
        df_carbon_q264 = pd.read_csv(p.carbon_by_region_base_year_path)
        # 1. Per-region (r264) quantity -> aggregate to one row per COUNTRY (r250).
        df_carbon_q250 = (
            df_carbon_q264
            .groupby(['iso3_r250_id', 'year'], as_index=False)['total']
            .sum()
            .rename(columns={'total': 'coastal_carbon_quantity'})
        )
        # 2. Country-level (r250) GEP = quantity * price. ONE row per country: source of truth for the
        #    CSV, the national total and every aggregation. Summing the r264-expanded table instead would
        #    double-count split countries (China spans 6 r264 rows, India 6, etc.) -- the same bug fixed
        #    in terrestrial_carbon. Only the map below touches r264, and it never sums.
        df_carbon_p = pd.read_excel(p.carbon_prices_path)[[p.carbon_price, 'year']]
        df_gep_250 = df_carbon_q250.merge(df_carbon_p, how='left', on='year')
        df_gep_250['coastal_carbon_gep'] = df_gep_250['coastal_carbon_quantity'] * df_gep_250[p.carbon_price]

        # Attach per-country attributes (one row per iso3 from the correspondence) and write the CSV.
        attr_cols = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
                     'continent', 'region_un', 'region_wb', 'income_grp', 'subregion']
        keep_cols = attr_cols + ['year', 'coastal_carbon_quantity', p.carbon_price, 'coastal_carbon_gep']
        df_gep_by_country_base_year = df_gep_250.merge(
            df_carbon_q264[attr_cols].drop_duplicates('iso3_r250_id'), how='left', on='iso3_r250_id')[keep_cols]
        hb.df_write(df_gep_by_country_base_year, p.results['coastal_carbon']['gep_by_country_base_year'])

        # 3. Map only: attach each country's GEP to the r264 boundary geometry (each sub-region shows its
        #    country's value). These rows are for plotting and must never be summed.
        df_regions = df_carbon_q264.merge(df_gep_250[['iso3_r250_id', 'coastal_carbon_gep']], how='left', on='iso3_r250_id')
        gdf_gep_by_country_base_year = hb.df_merge(p.gdf_countries_simplified, df_regions, how='outer', left_on='ee_r264_id', right_on='ee_r264_id')
        gdf_gep_by_country_base_year.to_file(p.results['coastal_carbon']['gep_by_country_base_year'].replace('.csv', '.gpkg'), driver='GPKG')

        # 4. National total = sum over the one-row-per-country table.
        value_gep_base_year = df_gep_by_country_base_year['coastal_carbon_gep'].sum()

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
    module_root = hb.get_projectflow_module_root()

    for service_label in services_run:
        print(service_label)
        results_qmd_path = os.path.join(module_root, service_label, f'{service_label}_results.qmd')
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
    coastal_carbon_initialization.build_gep_service_calculation_task_tree(p_temp)
    p_temp.set_all_tasks_to_skip_if_dir_exists()
    p_temp.execute()

    print(p_temp.results)
    pass

def gep_results_distribution(p):
    """Distribute the results of the GEP calculation."""
    # This task is intended to copy the results to the output directory.
    hb.log("Distributing GEP results...")

    for key, value in p.results['coastal_carbon'].items():
        output_path = os.path.join(p.output_dir, key)
        hb.path_copy(value, output_path)
        hb.log(f"Distributed {key} to {output_path}")

    hb.log("GEP results distribution complete.")



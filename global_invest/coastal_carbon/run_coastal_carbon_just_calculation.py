import os
import pandas as pd
import hazelbean as hb

from global_invest.coastal_carbon import coastal_carbon_initialization

if __name__ == '__main__':

    # ProjectFlow object
    p = hb.ProjectFlow()
    p.project_name = 'gep_coastal_carbon'
    p.project_dir = os.path.join(
        os.path.expanduser('~'), 'Files', 'global_invest', 'projects', p.project_name
    )
    p.base_data_dir = (
        "/Users/long/Library/CloudStorage/GoogleDrive-yxlong@umn.edu/"
        "Shared drives/NatCapTEEMs/Files/base_data/submissions"
    )
    p.set_project_dir(p.project_dir)


    coastal_carbon_initialization.build_gep_service_calculation_task_tree(p, include_seagrass=True)

    # Project level attributes - Data Paths

    # Country boundaries (marine EEZ)
    p.df_countries_marine_csv_path = p.get_path(
        'cartographic', 'ee', 'eemarine_r566_correspondence.csv'
    )
    p.gdf_countries_marine_vector_path = p.get_path(
        'cartographic', 'ee', 'eemarine_r566_correspondence.gpkg'
    )

    # Mangrove data (Global Mangrove Watch v3)
    p.mangrove_vector_path = os.path.join(
        p.base_data_dir, 'coastal_carbon', 'gmw_v3_2019_vec', 'gmw_v3_2019_vec.shp'
    )

    # Salt marsh data (processed from GWL_FCS30D)
    p.salt_marsh_vector_path = os.path.join(
        p.base_data_dir, 'coastal_carbon', 'global_salt_marsh2019.gpkg'
    )

    # Seagrass extent (UNEP-WCMC013-014 SeagrassPtPy v7.1, polygon shapefile).
    p.seagrass_vector_path = os.path.join(
        p.base_data_dir, 'coastal_carbon',
        '014_001_WCMC013-014_SeagrassPtPy2021_v7_1', '01_Data',
        'WCMC013_014_Seagrasses_Py_v7_1.shp'
    )

    # Sanderman et al. 2018 mangrove SOC raster (top 1 m, Mg C/ha).
    # File downloaded from Zenodo 7727569; median typology, 2019-2020 period, EPSG:4326.
    p.mangrove_soc_path = os.path.join(
        p.base_data_dir, 'coastal_carbon',
        'soc.tha_tnc.mangroves.typology_m_30m_b0..100cm_2019_2020_go_epsg.4326_v1.2.tif'
    )

    # Maxwell et al. 2024 tidal marsh SOC raster (top 1 m, Mg C/ha).
    # Download: https://zenodo.org/records/10940066 -> need pred0.zip (0-30 cm)
    # and pred30.zip (30-100 cm), summed to a single 0-100 cm GeoTIFF.
    # If absent, the salt marsh stock task falls back to a latitude step function.
    p.salt_marsh_soc_path = os.path.join(
        p.base_data_dir, 'coastal_carbon', 'maxwell_2024_marsoc_0_100cm.tif'
    )

    # Optional: mean annual precipitation raster (mm/yr) for tropical wet/dry split
    p.precipitation_path = os.path.join(
        p.base_data_dir, 'coastal_carbon', 'mean_annual_precipitation_mm.tif'
    )

    # Reference raster for area calculation (ha per cell)
    p.ha_per_cell_10sec_path = p.get_path('pyramids', "ha_per_cell_10sec.tif")

    # Carbon prices
    p.carbon_prices_path = os.path.join(
        p.base_data_dir, 'coastal_carbon', 'carbon_prices.xlsx'
    )
    p.carbon_price = "rental scc r2%"

    # Results dictionary
    p.results = {}

    # Initialize paths
    coastal_carbon_initialization.initialize_paths(p)

    # Run the model
    hb.log(
        'Created ProjectFlow object at ' + p.project_dir +
        '\n    from script ' + p.calling_script +
        '\n    with base_data set at ' + p.base_data_dir
    )
    p.execute()

    result = 'Done!'

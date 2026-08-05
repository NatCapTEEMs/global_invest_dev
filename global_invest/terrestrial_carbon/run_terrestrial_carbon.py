"""Standalone runner for the terrestrial-carbon GEP valuation -- values the carbon stock per country.

Consumers (ngfs_pnas, nff_global) do NOT use this script -- they graft add_terrestrial_carbon_tasks(p)
into their own task tree. This is for standalone GEP runs and smoke tests. For the standalone GTAP
productivity shock, see run_terrestrial_carbon_shock.py (one run file per purpose, no MODE switch).

base_data_dir is resolved by ProjectFlow (its default, overridable per machine via
~/.config/hazelbean/machine.env) -- do not hardcode it here.
"""
import os
import pandas as pd
import hazelbean as hb

from global_invest.terrestrial_carbon import terrestrial_carbon_initialize


if __name__ == '__main__':

    # ProjectFlow object
    p = hb.ProjectFlow() # Create a ProjectFlow Object to organize directories and enable parallel processing.
    p.project_name = 'gep_terrestrial_carbon'  # Determines the folder created to store intermediate and final results.
    p.project_dir = os.path.join(os.path.expanduser('~'), 'Files', 'global_invest', 'projects', p.project_name) # Put it in the right location relative to the user's home directory.
    p.set_project_dir(p.project_dir) # Sets p.base_data_dir (default / machine.env), p.input_dir, p.intermediate_dir, p.output_dir.

    # Task tree
    terrestrial_carbon_initialize.build_gep_service_task_tree(p) # Defines the actual logic of the model. Navigate into here to see what the model does.

    # Project level attributes
    p.df_countries_csv_path = p.get_path('cartographic', 'ee', 'ee_r264_correspondence.csv') # ProjectFlow downloads all files automatically via the p.get_path() function.
    p.gdf_countries_vector_path = p.get_path('cartographic', 'ee', 'ee_r264_correspondence.gpkg')
    p.gdf_countries_vector_simplified_path = p.get_path('cartographic', 'ee', 'ee_r264_simplified300sec.gpkg')
    p.carbon_zones_path = os.path.join(p.base_data_dir, 'carbon', 'johnson_2019', 'decision_tree_combined_carbon', 'carbon_zones_rasterized.tif')
    p.projected_carbon_density_2019_per_cell_path = os.path.join(p.project_dir, 'projected_carbon_density_maps_per_cell/projected_carbon_density_2019_per_cell.tif')
    p.lulc_folder_path = os.path.join(p.base_data_dir, 'lulc/esa')
    p.base_year_lulc_path = os.path.join(p.base_data_dir, 'lulc/esa/lulc_esa_2019.tif')
    p.carbon_prices_path = os.path.join(p.base_data_dir, 'terrestrial_carbon', 'carbon_prices.xlsx')
    p.carbon_price = "rental scc r2%"
    p.results = {}  # All results will be stored here by each child task.
    terrestrial_carbon_initialize.initialize_paths(p)

    # Run the model
    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

    result = 'Done!'

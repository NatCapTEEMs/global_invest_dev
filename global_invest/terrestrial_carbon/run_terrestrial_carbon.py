"""Standalone runner for the terrestrial-carbon module, in either of its two modes.

The module serves two consumers, so this script does too. Set MODE below:

  'gep'    the GEP valuation -- values the carbon stock per country. Original behaviour of this
           script, unchanged.
  'shock'  the GTAP productivity shock -- recomputes carbon density from the pipeline's own SEALS
           300 m maps at each anchor year and writes terrestrial_carbon_interpolated.csv.

One script rather than one per mode, matching run_pollination.py / run_erosion.py /
run_fisheries.py, which each cover their whole module.

Consumers (ngfs_pnas, nff_global) do NOT use this script -- they call add_terrestrial_carbon_tasks(p)
from their own task tree. This is for standalone runs and smoke tests.

SHOCK MODE NOTES
  The upstream density-preparation tasks (task_convert_carbon_density_maps_dtype ->
  task_combine_two_carbon_density_maps -> task_reproject_total_carbon_density ->
  task_compute_carbon_density_table) are NOT run here. They build the carbon-density lookup the shock
  consumes, out of the raw Spawn biomass rasters -- a one-off base-data job rather than part of the
  per-scenario shock. They stay in terrestrial_carbon_tasks.py for that purpose.

  Requires base_data/carbon_storage/ (carbon_zones_rasterized.tif and the SEALS7 density lookup, both
  resolved inside the task via p.get_path) plus SEALS 300 m maps for the base scenario and each
  scenario x anchor year.
"""
import os
import pandas as pd
import hazelbean as hb

from global_invest.terrestrial_carbon import terrestrial_carbon_initialize

MODE = 'gep'          # 'gep' | 'shock'

if __name__ == '__main__':

    # ProjectFlow object
    p = hb.ProjectFlow() # Create a ProjectFlow Object to organize directories and enable parallel processing.
    p.project_name = 'gep_terrestrial_carbon'  # Determines the folder created to store intermediate and final results.
    p.project_dir = os.path.join(os.path.expanduser('~'), 'Files', 'global_invest', 'projects', p.project_name) # Put it in the right location relative to the user's home directory.

    if MODE == 'gep':
        p.base_data_dir = "/Users/long/Library/CloudStorage/GoogleDrive-yxlong@umn.edu/Shared drives/NatCapTEEMs/Files/base_data/submissions" # Set where data outside the project will be stored. CAUTION: For GEP we are using the shared Google Drive, but best practice is to use a local directory that you can control (also it's faster)
        p.set_project_dir(p.project_dir) # Set the project directory in the ProjectFlow object. Also defines p.input_dir, p.intermediate_dir, and p.output_dir based on the project_dir.

        # Task tree
        terrestrial_carbon_initialize.build_gep_service_task_tree(p) # Defines the actual logic of the model. Navigate into here to see what the model does.

        # Project level attributes
        p.df_countries_csv_path = p.get_path('cartographic', 'ee', 'ee_r264_correspondence.csv') # ProjectFlow downloads all files automatically via the p.get_path() function.
        p.gdf_countries_vector_path = p.get_path('cartographic', 'ee', 'ee_r264_correspondence.gpkg')
        p.gdf_countries_vector_simplified_path = p.get_path('cartographic', 'ee', 'ee_r264_simplified300sec.gpkg')
        p.carbon_zones_path =os.path.join(p.base_data_dir,'carbon', 'johnson_2019', 'decision_tree_combined_carbon', 'carbon_zones_rasterized.tif')
        p.projected_carbon_density_2019_per_cell_path = os.path.join(p.project_dir, 'projected_carbon_density_maps_per_cell/projected_carbon_density_2019_per_cell.tif')
        p.lulc_folder_path = os.path.join(p.base_data_dir, 'lulc/esa')
        p.base_year_lulc_path = os.path.join(p.base_data_dir, 'lulc/esa/lulc_esa_2019.tif')
        p.carbon_prices_path = os.path.join(p.base_data_dir, 'terrestrial_carbon', 'carbon_prices.xlsx')
        p.carbon_price = "rental scc r2%"
        p.results = {}  # All results will be stored here by each child task.
        terrestrial_carbon_initialize.initialize_paths(p)

    elif MODE == 'shock':
        p.set_project_dir(p.project_dir)

        # Edit for a local smoke test. In a consumer pipeline these same attributes are set by that
        # project's run script (e.g. run_ngfs_pnas.py STEP 6).
        _ngfs = os.path.join(os.path.expanduser('~'), 'Files', 'gtap_invest', 'projects', 'ngfs', 'ngfs_pnas')

        # SEALS 300 m maps, one per scenario x anchor year; resolved by globbing this template.
        p.es_lulc_path_template = os.path.join(
            _ngfs, 'intermediate', 'stitched_lulc_simplified_scenarios',
            'lulc_esa_seals7_*_magpie_{scenario}_{year}.tif')

        # MUST be the SEALS7-classified base map, not a raw ESA map: the density lookup is keyed on
        # SEALS7 classes, so an ESA-coded raster here yields all-NoData densities. SEALS writes this
        # itself into fine_processed_inputs (it is NOT in base_data). Never p.base_year_lulc_path,
        # which SEALS owns and overwrites at runtime with its raw-ESA source.
        p.es_base_year_lulc_path = os.path.join(
            _ngfs, 'intermediate', 'fine_processed_inputs', 'lulc', 'esa', 'seals7',
            'lulc_esa_seals7_2023.tif')

        # Reaches the dynamic chain. Without this, add_terrestrial_carbon_tasks grafts the static task
        # instead, and this would read the frozen dependency CSV rather than recomputing from the maps.
        p.dynamic_es = ['terrestrial_carbon']

        p.es_shock_years         = [2030, 2040, 2050]              # SEALS anchor years (= seals_years)
        p.es_shock_base_year     = 2023                            # interp 0-anchor (GTAP base year)
        p.es_shock_end_year      = 2050
        p.es_shock_scenarios     = ['below_2c']
        p.es_shock_base_scenario = 'baseline_ignore_dependencies'  # the nature-off counterfactual
        p.terrestrial_carbon_shock_output_path = os.path.join(p.project_dir, 'terrestrial_carbon_interpolated.csv')
        p.results = {}

        terrestrial_carbon_initialize.add_terrestrial_carbon_tasks(p)

    else:
        raise ValueError("MODE must be 'gep' or 'shock', not %r" % (MODE,))

    # Run the model
    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

    result = 'Done!'

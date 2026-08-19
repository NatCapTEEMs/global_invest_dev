"""Standalone runner for the terrestrial-carbon GTAP productivity shock.

Recomputes carbon density from the pipeline's own SEALS 300 m maps at each anchor year and writes
terrestrial_carbon_interpolated.csv. Mirrors run_pollination.py / run_erosion.py. Consumers (ngfs_pnas,
nff_global) do NOT use this script -- they graft add_terrestrial_carbon_tasks(p) into their own task
tree. This is for standalone smoke tests. For the GEP valuation, see run_terrestrial_carbon.py.

The raw-Spawn density build (scale to Mg/ha, add aboveground+belowground, reproject to the LULC grid)
is a one-off base-data job, not part of this shock -- see howto/rebuild_spawn_total_carbon_density.md.
Requires base_data/carbon_storage/ (carbon_zones_rasterized.tif and the SEALS7 density lookup, both
resolved inside the task via p.get_path) plus SEALS 300 m maps for the base scenario and each scenario
x anchor year.

base_data_dir is resolved by ProjectFlow (default / machine.env) -- do not hardcode it here.
"""
import os

import hazelbean as hb

from global_invest.terrestrial_carbon import terrestrial_carbon_initialize


if __name__ == '__main__':

    p = hb.ProjectFlow()
    p.project_name = 'gep_terrestrial_carbon'
    p.project_dir = os.path.join(os.path.expanduser('~'), 'Files', 'global_invest', 'projects', p.project_name)
    p.set_project_dir(p.project_dir)

    # -------------------------------------------------------------------
    # Config -- edit for a local smoke test. In a consumer pipeline these
    # same attributes are set by the run script (e.g. run_ngfs_pnas.py STEP 6).
    # -------------------------------------------------------------------
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

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

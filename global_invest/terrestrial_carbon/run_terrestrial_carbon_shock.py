"""Standalone runner for the dynamic terrestrial-carbon ES SHOCK.

Named _shock to sit alongside run_terrestrial_carbon.py, which runs the GEP valuation
through terrestrial_carbon_initialize. Different entry points into the same module:
that one values the carbon stock, this one produces the GTAP productivity shock.

Mirrors run_pollination.py / run_erosion.py / run_fisheries.py: build a ProjectFlow, point it at the
SEALS 300 m maps and base_data, graft add_terrestrial_carbon_tasks, execute. Consumers (ngfs_pnas,
nff_global) do NOT use this script -- they graft the same seam into their own task tree. This exists for
standalone smoke tests of the carbon model on one or two scenarios.

The upstream density-preparation tasks (task_convert_carbon_density_maps_dtype ->
task_combine_two_carbon_density_maps -> task_reproject_total_carbon_density ->
task_compute_terrestrial_carbon_density_table) are NOT run here. They build the carbon-density lookup
that this shock consumes out of the raw Spawn biomass rasters, which is a one-off base-data job rather
than part of the per-scenario shock; they remain in terrestrial_carbon_tasks.py for that purpose.

Requires:
  - base_data/carbon_storage/: carbon_zones_rasterized.tif and the SEALS7 density lookup
    (both resolved inside the task via p.get_path)
  - SEALS 300 m maps for the base scenario and each scenario x anchor year
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
    # same attributes are set by the run script (e.g. run_ngfs_pnas.py).
    # -------------------------------------------------------------------
    _ngfs = os.path.join(os.path.expanduser('~'), 'Files', 'gtap_invest', 'projects', 'ngfs', 'ngfs_pnas')

    # SEALS 300 m maps, one per scenario x anchor year; resolved by globbing this template.
    p.es_lulc_path_template = os.path.join(
        _ngfs, 'intermediate', 'stitched_lulc_simplified_scenarios',
        'lulc_esa_seals7_*_magpie_{scenario}_{year}.tif')

    # MUST be the SEALS7-classified base map, not a raw ESA map: the density lookup is keyed on SEALS7
    # classes, so an ESA-coded raster here yields all-NoData densities. SEALS writes this itself into
    # fine_processed_inputs (it is NOT in base_data). Never p.base_year_lulc_path, which SEALS owns and
    # overwrites at runtime with its raw-ESA source.
    p.es_base_year_lulc_path = os.path.join(
        _ngfs, 'intermediate', 'fine_processed_inputs', 'lulc', 'esa', 'seals7',
        'lulc_esa_seals7_2023.tif')

    # Reaches the dynamic chain. Without this, add_terrestrial_carbon_tasks grafts the static task
    # instead and this script would read the frozen dependency CSV rather than recomputing from the maps.
    p.dynamic_es = ['terrestrial_carbon']

    p.es_shock_years         = [2030, 2040, 2050]          # SEALS anchor years (= seals_years)
    p.es_shock_base_year     = 2023                        # interp 0-anchor (GTAP base year)
    p.es_shock_end_year      = 2050
    p.es_shock_scenarios     = ['below_2c']
    p.es_shock_base_scenario = 'baseline_ignore_dependencies'   # the nature-off counterfactual
    p.terrestrial_carbon_shock_output_path = os.path.join(p.project_dir, 'terrestrial_carbon_interpolated.csv')

    terrestrial_carbon_initialize.add_terrestrial_carbon_tasks(p)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

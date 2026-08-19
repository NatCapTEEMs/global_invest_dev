"""Standalone runner for the dynamic pollination ES shock.

Mirrors run_terrestrial_carbon.py / run_erosion.py: build a ProjectFlow, point it at the SEALS 300 m maps and
base_data, graft add_pollination_tasks, execute. Consumers (ngfs_pnas, nff_global) do NOT use this
script -- they graft the same seam into their own task tree. This exists for standalone smoke tests
of the pollination model on one or two scenarios.

Requires (Track 2) under base_data/crop_benefits/:
  - poll_value_global_<base_year>usd.tif      baseline pollination value raster (fixed, all scenarios)
"""
import os

import hazelbean as hb

from global_invest.pollination import pollination_initialize


def build_task_tree(p):
    # This runner's tree IS the consumer seam: graft the tasks exactly as a pipeline would.
    pollination_initialize.add_pollination_tasks(p)


def run_project(p):

    # -------------------------------------------------------------------
    # Config -- edit for a local smoke test. In a consumer pipeline these
    # same attributes are set by the run script (e.g. run_ngfs_pnas.py).
    # -------------------------------------------------------------------
    # SEALS 300 m maps, one per scenario x anchor year; resolved by globbing this template.
    # Read the same way the GEP runner reads its maps: a base_data-relative reference. Stage the
    # maps under base_data/lulc/esa/seals7/scenarios/ for standalone runs; a consumer pipeline
    # overrides p.es_lulc_path_template with its own project's maps before grafting.
    p.es_lulc_path_template = os.path.join(
        p.get_path('lulc', 'esa', 'seals7', 'scenarios'),
        'lulc_esa_seals7_*_{scenario}_{year}.tif')

    # MUST be a SEALS7-classified base map (classes 1-7), not a raw ESA map:
    # run_pollination_sufficiency_300m selects the SEALS class scheme whenever the scenario label is
    # not the literal "2020", so an ESA-coded raster here would be silently misread. Same
    # base_data-relative convention as the scenario maps: stage it under base_data/lulc/esa/seals7/;
    # a consumer pipeline overrides with its own copy. Note the two base years: the SEALS land-cover
    # base is seals_key_base_year (2020) while the GTAP/ES anchor is key_base_year (2023); the
    # pollination baseline follows the ES anchor.
    p.base_year_lulc_path = p.get_path('lulc', 'esa', 'seals7', 'lulc_esa_seals7_2023.tif')

    # Reaches the dynamic chain. Without this, add_pollination_tasks grafts the static task instead and
    # this script would read the frozen dependency CSV rather than recomputing from the maps above.
    p.dynamic_es = ['pollination']

    p.es_shock_years     = [2030, 2040, 2050]   # SEALS anchor years (= seals_years)
    p.es_shock_base_year = 2023                 # interp 0-anchor (GTAP base year)
    p.es_shock_scenarios = ['below_2c']
    p.pollination_shock_output_path = os.path.join(p.project_dir, 'pollination_interpolated.csv')

    build_task_tree(p)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

    return p


if __name__ == '__main__':

    # Create ProjectFlow object
    p = hb.ProjectFlow(project_name='gep_pollination_shock', run_mode='check')

    # Run the project
    run_project(p)

    result = 'Done!'

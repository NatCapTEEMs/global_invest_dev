"""Standalone runner for the dynamic erosion-control ES shock.

Mirrors run_terrestrial_carbon.py / run_pollination.py: build a ProjectFlow, point it at the SEALS 300 m
maps and base_data, graft add_erosion_tasks, execute. Consumers (ngfs_pnas, nff_global) do NOT use this
script -- they graft the same seam into their own task tree. This exists to run the SDR chain in
isolation, which is the practical way to debug it: the four dynamic tasks are the heaviest thing in the
ES fold, and going through run_ngfs_pnas.py would drag in GTAP solves and SEALS as well.

Runs the DYNAMIC path (p.dynamic_es includes 'erosion'), i.e. SDR -> upstream D8 -> exposure -> shock.
For the static path there is nothing to run standalone: it just reads the frozen dependency table.

Requires:
  - natcap.invest (conda-forge) for the SDR model, and pygeoprocessing for the D8 routing
  - base_data/global_invest/sdr/: the analysis grid, DEM, watersheds, biophysical table, SPAM2020
    yield/area stacks, bandmap and crop-coefficient table (all resolved inside add_erosion_tasks)
  - base_data/soil/: erosivity and erodibility
  - SEALS 300 m maps for the base scenario and each scenario x anchor year

On the default 6.45 km analysis grid this is a ~25M-pixel global run, which is tractable locally -- the
multi-hour figures quoted for SDR refer to NATIVE-resolution global runs, roughly 56x larger. Set
p.modality to 'sc' or 'msi' to run at native SEALS resolution on a cluster instead.

Note the base scenario must be present for the differencing, so a minimal run is 2 scenarios x 1 anchor
= 2 SDR calls; the long_test config implies 4 and the production config 18.
"""
import os

import hazelbean as hb

from global_invest.erosion import erosion_initialize


def build_task_tree(p):
    # This runner's tree IS the consumer seam: graft the tasks exactly as a pipeline would.
    erosion_initialize.add_erosion_tasks(p)


def run_project(p):

    # -------------------------------------------------------------------
    # Config -- edit for a local smoke test. In a consumer pipeline these
    # same attributes are set by the run script (e.g. run_ngfs_pnas.py).
    # -------------------------------------------------------------------
    # SEALS 300 m maps, one per scenario x anchor year; resolved by globbing this template. The base
    # scenario is globbed with the same pattern, so it must be present for the differencing to work.
    #
    # GLOBAL maps, deliberately. The analysis grid is global 6.45 km either way, so SDR costs the same
    # ~25M pixels whether the LULC spans the world or a small AOI -- an AOI run just leaves the grid
    # almost entirely nodata and pays the same price.
    # Read the same way the GEP runner reads its maps: a base_data-relative reference. Stage the
    # maps under base_data/lulc/esa/seals7/scenarios/ for standalone runs; a consumer pipeline
    # overrides p.es_lulc_path_template with its own project's maps before grafting.
    p.es_lulc_path_template = os.path.join(
        p.get_path('lulc', 'esa', 'seals7', 'scenarios'),
        'lulc_esa_seals7_*_{scenario}_{year}.tif')

    # Reaches the dynamic chain. Without this, add_erosion_tasks grafts the static task instead and
    # none of the SDR work runs.
    p.dynamic_es = ['erosion']

    p.es_shock_years         = [2050]           # the only anchor the global maps carry
    p.es_shock_base_year     = 2023                            # interp 0-anchor (GTAP base year)
    p.es_shock_end_year      = 2050
    p.es_shock_scenarios     = ['below_2c']
    p.es_shock_base_scenario = 'baseline_ignore_damages'       # the nature-off counterfactual
    p.erosion_shock_output_path = os.path.join(p.project_dir, 'erosion_interpolated.csv')

    # Every method is computed and reported side by side, and erosion_method only picks which one
    # becomes shock_pct. Left unset so this inherits the task default rather than pinning a second
    # copy of it here, which is how the standalone runner and the pipeline drift apart.

    build_task_tree(p)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

    return p


if __name__ == '__main__':

    # Create ProjectFlow object
    p = hb.ProjectFlow(project_name='gep_erosion', run_mode='check')

    # Run the project
    run_project(p)

    result = 'Done!'

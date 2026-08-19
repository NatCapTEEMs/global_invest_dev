"""Standalone runner for the marine-fisheries ES shock.

Mirrors run_erosion.py / run_pollination.py: build a ProjectFlow, graft add_fisheries_tasks, execute.
Consumers (ngfs_pnas, nff_global) do NOT use this script -- they graft the same seam into their own task
tree. This exists for a standalone check of the FSH shock.

Unlike the other services there is no dynamic path to select: fisheries is marine, so it never reads the
SEALS land-cover maps and p.dynamic_es does not apply to it. The task reads the pre-computed FI headers
out of cwon_shocks.har by RCP, which makes this the cheapest of the four to run -- no rasters, no
InVEST, seconds rather than hours.

Requires:
  - base_data/<aggregation_label>/cwon_shocks.har with the FI26 / FI45 / FI85 headers
"""
import os

import hazelbean as hb

from global_invest.fisheries import fisheries_initialize


def build_task_tree(p):
    # This runner's tree IS the consumer seam: graft the tasks exactly as a pipeline would.
    fisheries_initialize.add_fisheries_tasks(p)


if __name__ == '__main__':

    p = hb.ProjectFlow(project_name='gep_fisheries', run_mode='check')

    # -------------------------------------------------------------------
    # Config -- edit for a local smoke test. In a consumer pipeline these
    # same attributes are set by the run script (e.g. run_ngfs_pnas.py).
    # -------------------------------------------------------------------
    p.aggregation_label = 'v12-s26-r50'         # locates cwon_shocks.har under base_data

    p.es_shock_years         = [2050]
    p.es_shock_base_year     = 2023             # interp 0-anchor (GTAP base year)
    p.es_shock_end_year      = 2050
    p.es_shock_scenarios     = ['below_2c']
    p.es_shock_base_scenario = 'baseline_ignore_damages'
    p.fisheries_shock_output_path = os.path.join(p.project_dir, 'fisheries_interpolated.csv')

    build_task_tree(p)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

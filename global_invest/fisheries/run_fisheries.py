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
import hazelbean as hb

from global_invest import utilities
from global_invest.fisheries import fisheries_initialize


def build_task_tree(p):
    # This runner's tree IS the consumer seam: graft the tasks exactly as a pipeline would.
    fisheries_initialize.add_fisheries_tasks(p)


def run_project(p):

    # Fisheries is the static marine service: no maps, no dynamic path -- scenarios name the CWoN
    # table's own FI headers, so this reads its own scenarios file rather than the seals-map one
    # (the task's header map falls back to identity, so FI-native labels flow straight through).
    # The shared es_shock_* seam attributes come from that CSV as a defaults layer: anything the
    # caller already set on p wins. The output CSV needs no line here -- the task defaults it.
    p.es_scenario_definitions_filename = 'es_scenarios_fisheries_test.csv'
    utilities.hydrate_es_scenarios(p)

    build_task_tree(p)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

    return p


if __name__ == '__main__':

    # Create ProjectFlow object
    p = hb.ProjectFlow(project_name='gep_fisheries', run_mode='check')

    # Run the project
    run_project(p)

    result = 'Done!'

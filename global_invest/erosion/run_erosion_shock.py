"""Standalone runner for the dynamic erosion-control ES shock.

Mirrors run_terrestrial_carbon_shock.py / run_pollination_shock.py: build a ProjectFlow, graft
add_erosion_tasks, execute. Consumers (ngfs_pnas, nff_global) do NOT use this script -- they graft
the same seam into their own task tree and set the es_shock_* seam attributes from their own
scenarios CSV. This exists to run the SDR chain in isolation, which is the practical way to debug
it: the four dynamic tasks are the heaviest thing in the ES fold, and going through
run_ngfs_pnas.py would drag in GTAP solves and SEALS as well.

Scenario configuration (scenarios, years, the SEALS7 map references) comes from
input_template/es_scenarios_test.csv via hydrate_es_scenarios -- edit the project's input/ copy, or
point p.es_scenario_definitions_filename at another file. The shipped CSV uses the standard seals
scenario names, and the matching maps ship as input_template fixtures. The base scenario must be
present for the differencing, so the shipped CSV implies (1 base + 1 policy) x 2 anchors = 4 SDR
calls.

Runs the DYNAMIC path (p.dynamic_es includes 'erosion'), i.e. SDR -> upstream D8 -> exposure ->
shock. For the static path there is nothing to run standalone: it just reads the frozen dependency
table.

Requires:
  - natcap.invest (conda-forge) for the SDR model, and pygeoprocessing for the D8 routing
  - base_data/global_invest/sdr/: the analysis grid, DEM, watersheds, biophysical table, SPAM2020
    yield/area stacks, bandmap and crop-coefficient table (all es_parameters rows, hydrated by
    each task's publish_inputs)
  - base_data/soil/: erosivity and erodibility

On the default 6.45 km analysis grid this is a ~25M-pixel global run, which is tractable locally --
the multi-hour figures quoted for SDR refer to NATIVE-resolution global runs, roughly 56x larger.
Set the erosion_native_resolution row (or p.erosion_native_resolution) true to run at native
SEALS resolution on a cluster instead.
"""
import hazelbean as hb

from global_invest import utilities
from global_invest.erosion import erosion_initialize


def build_task_tree(p):
    # This runner's tree IS the consumer seam: graft the tasks exactly as a pipeline would.
    erosion_initialize.add_erosion_tasks(p)


def run_project(p):
    # Reaches the dynamic chain. Without this, add_erosion_tasks grafts the static task instead and
    # none of the SDR work runs.
    p.dynamic_es = ['erosion']

    # The shared es_shock_* seam attributes, from the scenarios CSV as a defaults layer:
    # anything the caller already set on p wins. The output CSV needs no line here -- the
    # task defaults it into the project dir.
    utilities.hydrate_es_scenarios(p)

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

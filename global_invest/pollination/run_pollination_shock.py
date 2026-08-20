"""Standalone runner for the dynamic pollination ES shock.

Mirrors run_terrestrial_carbon_shock.py / run_erosion_shock.py: build a ProjectFlow, graft
add_pollination_tasks, execute. Consumers (ngfs_pnas, nff_global) do NOT use this script -- they
graft the same seam into their own task tree and set the es_shock_* seam attributes from their own
scenarios CSV. This exists for standalone smoke tests of the pollination model on one or two
scenarios.

Scenario configuration (scenarios, years, the SEALS7 map references) comes from
input_template/es_scenarios_test.csv via hydrate_es_scenarios -- edit the project's input/ copy, or
point p.es_scenario_definitions_filename at another file. The shipped CSV uses the standard seals
scenario names, and the matching maps ship as input_template fixtures, so this runs self-contained.
The base-year map MUST be SEALS7-classified (classes 1-7), not raw ESA: the sufficiency task selects
the SEALS class scheme whenever the scenario label is not the literal "2020", so an ESA-coded raster
would be silently misread. The shipped fixture is SEALS7.

Requires (Track 2) under base_data/crop_benefits/:
  - poll_value_global_<base_year>usd.tif      baseline pollination value raster (fixed, all scenarios)
"""
import hazelbean as hb

from global_invest import utilities
from global_invest.pollination import pollination_initialize


def build_task_tree(p):
    # This runner's tree IS the consumer seam: graft the tasks exactly as a pipeline would.
    pollination_initialize.add_pollination_tasks(p)


def run_project(p):
    # Reaches the dynamic chain. Without this, add_pollination_tasks grafts the static task instead
    # and this script would read the frozen dependency CSV rather than recomputing from the maps.
    p.dynamic_es = ['pollination']

    # The shared es_shock_* seam attributes, from the scenarios CSV as a defaults layer:
    # anything the caller already set on p wins. The output CSV needs no line here -- the
    # task defaults it into the project dir.
    utilities.hydrate_es_scenarios(p)

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

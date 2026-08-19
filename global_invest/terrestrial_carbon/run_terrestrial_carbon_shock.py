"""Standalone runner for the terrestrial-carbon GTAP productivity shock.

Recomputes carbon density from SEALS 300 m maps at each anchor year and writes
terrestrial_carbon_interpolated.csv. Mirrors run_pollination_shock.py / run_erosion_shock.py.
Consumers (ngfs_pnas, nff_global) do NOT use this script -- they graft add_terrestrial_carbon_tasks(p)
into their own task tree and set the es_shock_* seam attributes from their own scenarios CSV. This is
for standalone smoke tests. For the GEP valuation, see run_terrestrial_carbon.py.

Scenario configuration (scenarios, years, the SEALS7 map references) comes from
input_template/es_scenarios_test.csv via hydrate_es_scenarios -- edit the project's input/ copy, or
point p.es_scenario_definitions_filename at another file. The shipped CSV uses the standard seals
scenario names (ssp2_rcp45_luh2-message_bau / _bau_shift), and the matching maps from a standard
seals test run ship as input_template fixtures, so this runs self-contained; to shock other maps,
put them in the project's input/ (or base_data) under the CSV's references.

The raw-Spawn density build (scale to Mg/ha, add aboveground+belowground, reproject to the LULC grid)
is a one-off base-data job, not part of this shock -- see howto/rebuild_spawn_total_carbon_density.md.
Requires base_data/global_invest/terrestrial_carbon/ (carbon_zones_rasterized.tif and the SEALS7
density lookup, both resolved inside the task via p.get_path) plus SEALS 300 m maps for the base
scenario and each scenario x anchor year.

base_data_dir is resolved by ProjectFlow (default / machine.env) -- do not hardcode it here.
"""
import hazelbean as hb

from global_invest import utilities
from global_invest.terrestrial_carbon import terrestrial_carbon_initialize


def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder,
    # which reads p.dynamic_es (set in run_project) to decide whether to graft
    # the dynamic recompute chain or the static frozen-CSV task.
    terrestrial_carbon_initialize.add_terrestrial_carbon_tasks(p)


def run_project(p):
    # Reaches the dynamic chain. Without this, add_terrestrial_carbon_tasks grafts the static task
    # instead, and this would read the frozen dependency CSV rather than recomputing from the maps.
    p.dynamic_es = ['terrestrial_carbon']

    # The shared es_shock_* seam attributes, from the scenarios CSV as a defaults layer:
    # anything the caller already set on p wins. The output CSV needs no line here -- the
    # task defaults it into the project dir.
    utilities.hydrate_es_scenarios(p)

    build_task_tree(p)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

    return p


if __name__ == '__main__':

    # Create ProjectFlow object. run_mode: 'check' resumes in place | 'fresh_intermediate'
    # rebuilds all computation but keeps input/ (test projects only) | 'full' timestamps a new dir.
    p = hb.ProjectFlow(project_name='gep_terrestrial_carbon', run_mode='check')

    # Run the project
    run_project(p)

    result = 'Done!'

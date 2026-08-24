"""Full flood-control GEP run: the committed avoided-damage valuation -> report.

Thin runner: builds ONE tree and executes it. Inputs are published by each task itself
(publish_inputs in the tasks module); base_data_dir is resolved by ProjectFlow (default /
machine.env), never hardcoded here.

Requires (es_parameters row): country_avoided_damage_usd2019.csv under base_data/global_invest/
flood_regulation/ (extracted from the drive's FINAL_GEP_FLOOD mapping geopackage; also shipped
in reference/ as the anchor). The upstream pipeline -- flood hazard x JRC depth-damage x
protection levels, with and without ecosystems -- is the drive's December 2025 pipeline, taken
as given; its code and method notes are on the drive.
"""
import hazelbean as hb

from global_invest.flood_control import flood_control_initialize


def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    flood_control_initialize.build_gep_service_task_tree(p)


def run_project(p):

    # Every task publishes its own inputs (publish_inputs in the tasks module): no setup call.
    build_task_tree(p)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

    return p


if __name__ == '__main__':

    # Create ProjectFlow object
    p = hb.ProjectFlow(project_name='gep_flood_control', run_mode='check')

    # Run the project
    run_project(p)

    result = 'Done!'

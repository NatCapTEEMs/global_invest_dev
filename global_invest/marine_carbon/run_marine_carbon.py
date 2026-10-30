"""Full marine carbon GEP run: ocean areas -> area-share valuation of four pools -> report.

Thin runner: builds ONE tree and executes it. Inputs are published by each task itself
(publish_inputs in the tasks module); base_data_dir is resolved by ProjectFlow (default /
machine.env), never hardcoded here.

Requires the marine correspondence layer named in es_config (cartographic/ee/
eemarine_r566_correspondence.gpkg) and the pool stocks and prices in es_parameters.
"""
import hazelbean as hb

from global_invest.marine_carbon import marine_carbon_initialize


def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    marine_carbon_initialize.build_gep_service_task_tree(p)


def run_project(p):

    # Every task publishes its own inputs (publish_inputs in the tasks module): no setup call.
    build_task_tree(p)

    p.skip_tasks(p.tasks_to_skip)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

    return p


if __name__ == '__main__':

    # Create ProjectFlow object
    p = hb.ProjectFlow(project_name='gep_marine_carbon', run_mode='check')

    # Run the project
    run_project(p)

    result = 'Done!'

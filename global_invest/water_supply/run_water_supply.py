"""Run hydropower, agricultural and domestic water valuations and their report.

Requires the commercial crop country-item table and staged service inputs.
Paths and parameters are supplied through the shared input templates.
"""
import hazelbean as hb

from global_invest.water_supply import water_supply_initialize


def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    water_supply_initialize.build_gep_service_task_tree(p)


def run_project(p):

    # Every task publishes its own inputs (publish_inputs in the tasks module): no setup call.
    build_task_tree(p)

    p.skip_tasks(p.tasks_to_skip)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

    return p


if __name__ == '__main__':

    # Create ProjectFlow object
    p = hb.ProjectFlow(project_name='gep_water_supply', run_mode='check')

    # Run the project
    run_project(p)

    result = 'Done!'

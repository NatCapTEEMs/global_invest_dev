"""Full stormwater GEP run: retention valuation -> report.

Thin runner: builds ONE tree and executes it. Inputs are published by each task itself
(publish_inputs in the tasks module); base_data_dir is resolved by ProjectFlow (default /
machine.env), never hardcoded here.

The InVEST retention run of the port is in progress (see stormwater_tasks.gep_calculation),
so a run currently stops there with the state spelled out.
"""
import hazelbean as hb

from global_invest.stormwater import stormwater_initialize


def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    stormwater_initialize.build_gep_service_task_tree(p)


def run_project(p):

    # Every task publishes its own inputs (publish_inputs in the tasks module): no setup call.
    build_task_tree(p)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

    return p


if __name__ == '__main__':

    p = hb.ProjectFlow(project_name='gep_stormwater', run_mode='check')
    run_project(p)

    result = 'Done!'

"""Full pollination GEP run: value-raster summarize -> per-country valuation -> results report.

Thin runner: builds ONE tree and executes it. Inputs are published by each task itself
(publish_inputs in the tasks module); base_data_dir is
resolved by ProjectFlow (default / machine.env), never hardcoded here. The ES-shock runner is
run_pollination_shock.py -- a separate tree, a separate thin runner, no mode switch.
"""
import hazelbean as hb

from global_invest.pollination import pollination_initialize

def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    pollination_initialize.build_gep_service_task_tree(p)


def run_project(p):

    # Every task publishes its own inputs (publish_inputs in the tasks module): no setup call.
    build_task_tree(p)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

    return p


if __name__ == '__main__':

    # Create ProjectFlow object
    p = hb.ProjectFlow(project_name='gep_pollination', run_mode='check')

    # Run the project
    run_project(p)

    result = 'Done!'

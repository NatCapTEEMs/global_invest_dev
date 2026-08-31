"""Full flood GEP run: inputs -> SDA -> service flow -> valuation -> GEP -> results report.

Thin runner: builds one tree and executes it. Every task publishes its own inputs, so nothing is
configured here. Inputs are es_parameters `*_path` rows resolved by `get_path`; outputs are named
by the task that writes them, under its own `cur_dir`.
"""
import hazelbean as hb

from global_invest.flood import flood_initialize


def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    flood_initialize.build_gep_service_task_tree(p)


def run_project(p):

    # Every task publishes its own inputs (publish_inputs in the tasks module): no setup call.
    build_task_tree(p)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

    return p


if __name__ == '__main__':

    # Create ProjectFlow object
    p = hb.ProjectFlow(project_name='gep_flood', run_mode='check')

    # Run the project
    run_project(p)

    result = 'Done!'

"""Full NTFP GEP run: FAO total forest valuation, its alternatives -> report.

Thin runner: builds ONE tree and executes it. Inputs are published by each task itself
(publish_inputs in the tasks module); base_data_dir is resolved by ProjectFlow (default /
machine.env), never hardcoded here.

A full run values FAO total forest from FRA 2020, writes the CWoN-area, divide-by-five, ESA-forest
and accessible-forest tables beside it, and renders the report. Every heavy step is guarded by an existence check, so
a rerun resumes where it stopped.
"""
import hazelbean as hb

from global_invest.ntfp import ntfp_initialize


def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    ntfp_initialize.build_gep_service_task_tree(p)


def run_project(p):

    # Every task publishes its own inputs (publish_inputs in the tasks module): no setup call.
    build_task_tree(p)

    p.skip_tasks(p.tasks_to_skip)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

    return p


if __name__ == '__main__':

    p = hb.ProjectFlow(project_name='gep_ntfp', run_mode='check')
    run_project(p)

    result = 'Done!'

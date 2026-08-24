"""Full fisheries GEP run (commercial capture, CWoN method): rent trends -> valuation -> report.

Thin runner: builds ONE tree and executes it. Inputs are published by each task itself
(publish_inputs in the tasks module); base_data_dir is resolved by ProjectFlow (default /
machine.env), never hardcoded here. The ES-shock runner is run_fisheries_shock.py -- a separate
tree, a separate thin runner, no mode switch.

Requires (es_parameters rows under base_data/global_invest/fisheries/cwon/): cpi2019.dta and
EconRent_Analysis_AllYears.dta from the CWoN 2024 reproducibility package (FR_WLD_2024_195,
public World Bank download) -- NOT yet staged; the open data ask, together with the source
pipeline's fish_provision_gep_20260720.csv as the replication anchor.
"""
import hazelbean as hb

from global_invest.fisheries import fisheries_initialize


def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    fisheries_initialize.build_gep_service_task_tree(p)


def run_project(p):

    # Every task publishes its own inputs (publish_inputs in the tasks module): no setup call.
    build_task_tree(p)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

    return p


if __name__ == '__main__':

    # Create ProjectFlow object
    p = hb.ProjectFlow(project_name='gep_fisheries', run_mode='check')

    # Run the project
    run_project(p)

    result = 'Done!'

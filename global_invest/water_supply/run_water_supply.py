"""Full water-supply GEP run (hydropower component): CWoN wealth -> implied rent -> report.

Thin runner: builds ONE tree and executes it. Inputs are published by each task itself
(publish_inputs in the tasks module); base_data_dir is resolved by ProjectFlow (default /
machine.env), never hardcoded here.

Requires (es_parameters row): cwon/hydro_wealth.dta under base_data/global_invest/water_supply/
hydropower/ — from the CWoN 2024 reproducibility package (public; already staged on this
machine). The committed anchor from the consortium drive replicates exactly; see the module
docstring for the identified method.
"""
import hazelbean as hb

from global_invest.water_supply import water_supply_initialize


def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    water_supply_initialize.build_gep_service_task_tree(p)


def run_project(p):

    # Every task publishes its own inputs (publish_inputs in the tasks module): no setup call.
    build_task_tree(p)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

    return p


if __name__ == '__main__':

    # Create ProjectFlow object
    p = hb.ProjectFlow(project_name='gep_water_supply', run_mode='check')

    # Run the project
    run_project(p)

    result = 'Done!'

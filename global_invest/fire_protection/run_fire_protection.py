"""Full fire-protection GEP run: persistence betas -> avoided acres -> avoided damage -> report.

Thin runner: builds ONE tree and executes it. Inputs are published by each task itself
(publish_inputs in the tasks module); base_data_dir is resolved by ProjectFlow (default /
machine.env), never hardcoded here.

Requires (es_parameters rows under base_data/global_invest/fire_protection/):
  - all_countries_regression_results_lag1.csv  (committed in the source repo; staged)
  - the EM-DAT wildfire extract xlsx            (committed in the source repo; staged)
  - gadm_adm2_panel_complete_2010_2023.csv      (NOT staged -- the open data ask; until it
    lands, the burned-area and damage-rate columns come from the frozen reference output,
    which this port reproduces to the float)
"""
import hazelbean as hb

from global_invest.fire_protection import fire_protection_initialize


def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    fire_protection_initialize.build_gep_service_task_tree(p)


def run_project(p):

    # Every task publishes its own inputs (publish_inputs in the tasks module): no setup call.
    build_task_tree(p)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

    return p


if __name__ == '__main__':

    # Create ProjectFlow object
    p = hb.ProjectFlow(project_name='gep_fire_protection', run_mode='check')

    # Run the project
    run_project(p)

    result = 'Done!'

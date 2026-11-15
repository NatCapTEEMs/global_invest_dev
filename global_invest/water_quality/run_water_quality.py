"""Full water-quality GEP run: retention priced at the benefit-transfer treatment cost -> report.

Thin runner: builds ONE tree and executes it. Inputs are published by each task itself
(publish_inputs in the tasks module); base_data_dir is resolved by ProjectFlow (default /
machine.env), never hardcoded here.

Requires the es_parameters rows under base_data/global_invest/water_quality/price_transfer/:
the authors' retention estimates, raw price observations and the three World Bank series their
script reads. The retention totals are upstream science taken as given.
"""
import hazelbean as hb

from global_invest.water_quality import water_quality_initialize


def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    water_quality_initialize.build_gep_service_task_tree(p)


def run_project(p):

    # Every task publishes its own inputs (publish_inputs in the tasks module): no setup call.
    build_task_tree(p)

    p.skip_tasks(p.tasks_to_skip)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

    return p


if __name__ == '__main__':

    # Create ProjectFlow object
    p = hb.ProjectFlow(project_name='gep_water_quality', run_mode='check')

    # Run the project
    run_project(p)

    result = 'Done!'

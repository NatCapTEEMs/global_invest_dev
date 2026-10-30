"""Full air-filtration GEP run: the drive workbook's avoided-mortality valuation -> report.

Thin runner: builds ONE tree and executes it. Inputs are published by each task itself
(publish_inputs in the tasks module); base_data_dir is resolved by ProjectFlow (default /
machine.env), never hardcoded here.

Requires (es_parameters rows): air_filtration_gep.xlsx, r250_gpkg_order.csv and the
not-estimated list under base_data/global_invest/air_filtration/ (pulled from the consortium
drive), and the shared VSL panel. The InMAP-based deaths columns are upstream science taken as
given; the values are priced at the shared panel, not the workbook's own VSL column.
"""
import hazelbean as hb

from global_invest.air_filtration import air_filtration_initialize


def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    air_filtration_initialize.build_gep_service_task_tree(p)


def run_project(p):

    # Every task publishes its own inputs (publish_inputs in the tasks module): no setup call.
    build_task_tree(p)

    p.skip_tasks(p.tasks_to_skip)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

    return p


if __name__ == '__main__':

    # Create ProjectFlow object
    p = hb.ProjectFlow(project_name='gep_air_filtration', run_mode='check')

    # Run the project
    run_project(p)

    result = 'Done!'

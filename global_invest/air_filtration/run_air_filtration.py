"""Full air-filtration GEP run: the drive workbook's avoided-mortality valuation -> report.

Thin runner: builds ONE tree and executes it. Inputs are published by each task itself
(publish_inputs in the tasks module); base_data_dir is resolved by ProjectFlow (default /
machine.env), never hardcoded here.

Requires (es_parameters row): air_filtration_gep.xlsx under base_data/global_invest/
air_filtration/ (pulled from the consortium drive; also shipped in reference/ as the anchor).
The InMAP-based deaths columns are upstream science taken as given; the VSL build behind the
workbook is an open ask (the folder's vsl.R is a different vintage).
"""
import hazelbean as hb

from global_invest.air_filtration import air_filtration_initialize


def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    air_filtration_initialize.build_gep_service_task_tree(p)


def run_project(p):

    # Every task publishes its own inputs (publish_inputs in the tasks module): no setup call.
    build_task_tree(p)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

    return p


if __name__ == '__main__':

    # Create ProjectFlow object
    p = hb.ProjectFlow(project_name='gep_air_filtration', run_mode='check')

    # Run the project
    run_project(p)

    result = 'Done!'

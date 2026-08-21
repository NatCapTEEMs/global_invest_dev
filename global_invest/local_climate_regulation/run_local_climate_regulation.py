"""Full local-climate-regulation GEP run: the committed urban-cooling valuation -> report.

Thin runner: builds ONE tree and executes it. Inputs are published by each task itself
(publish_inputs in the tasks module); base_data_dir is resolved by ProjectFlow (default /
machine.env), never hardcoded here.

Requires (es_parameters row): local_climate_regulation_gep.csv under base_data/global_invest/
local_climate_regulation/ (the drive's committed final; also shipped in reference/ as the
anchor, with the v02 and December 2025 lineage versions and a city-month sample). The v04
correction's own inputs -- the IEA-CMCC CDD file and the v2_nkd energy workbook -- are the
open asks; with them the correction becomes reproducible from raw.
"""
import hazelbean as hb

from global_invest.local_climate_regulation import local_climate_regulation_initialize


def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    local_climate_regulation_initialize.build_gep_service_task_tree(p)


def run_project(p):

    # Every task publishes its own inputs (publish_inputs in the tasks module): no setup call.
    build_task_tree(p)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

    return p


if __name__ == '__main__':

    # Create ProjectFlow object
    p = hb.ProjectFlow(project_name='gep_local_climate_regulation', run_mode='check')

    # Run the project
    run_project(p)

    result = 'Done!'

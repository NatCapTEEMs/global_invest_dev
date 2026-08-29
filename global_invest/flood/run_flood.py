"""Full flood GEP run: inputs -> SDA -> service flow -> valuation -> GEP -> results report.

Thin runner: builds ONE tree and executes it. Inputs are published by each task itself
(publish_inputs in the tasks module), which hydrates es_config and es_parameters and then calls
configure_paths to resolve every location under `flood_root_dir`. Nothing is hardcoded here.

⚠ This file used to carry `set_flood_paths(p)`, 200 lines setting 55 `p.flood_*` attributes, which
the MSI runners imported to learn the project layout. A grep across flood_tasks, flood_functions,
flood_utils, flood_initialize and the cluster scripts on 2026-08-29 found only six of those
attributes ever read anywhere: `flood_root_dir` and `flood_gep_for_merge_path`, both already in
es_parameters, plus `flood_iso3_list`, `flood_iso3_start`, `flood_iso3_n` and
`flood_skip_depth_download`, which are now there too. The other 51 were dead: `configure_paths`
builds every path from `flood_root_dir` on its own.

That dead configuration did real damage rather than merely sitting there. Because the paths were
hardcoded, the valuation ran correctly on a machine where `input_template/` was missing and neither
es_config nor es_parameters hydrated at all -- so nothing revealed the absence until the GEP chain,
which does read config, returned $0 for all 250 countries while reporting ok=250/250 at every stage.
Two configuration mechanisms, one of them silently masking that the other was absent.
"""
import hazelbean as hb

from global_invest.flood import flood_initialize


def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    flood_initialize.build_flood_task_tree(p)


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

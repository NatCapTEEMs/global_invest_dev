"""Full coastal-carbon GEP run: per-ecosystem chains (mangrove + salt marsh + seagrass) ->
combine -> gep_calculation -> results report.

Thin runner: it builds ONE tree and executes it. All input paths are get_path REFERENCE paths
resolved in coastal_carbon_initialize.initialize_paths (one source of truth for every runner);
base_data_dir is resolved by ProjectFlow (its default, overridable per machine via
~/.config/hazelbean/machine.env), never hardcoded here.
"""
import hazelbean as hb

from global_invest.coastal_carbon import coastal_carbon_initialize

def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    coastal_carbon_initialize.build_gep_service_task_tree(p, include_seagrass=True)


def run_project(p):

    build_task_tree(p)

    p.results = {}
    coastal_carbon_initialize.initialize_paths(p)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

    return p


if __name__ == '__main__':

    # Create ProjectFlow object
    p = hb.ProjectFlow(project_name='gep_coastal_carbon', run_mode='check')

    # Run the project
    run_project(p)

    result = 'Done!'

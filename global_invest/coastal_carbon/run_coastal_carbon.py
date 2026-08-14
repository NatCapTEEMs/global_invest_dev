"""Full coastal-carbon GEP run: per-ecosystem chains (mangrove + salt marsh + seagrass) ->
combine -> gep_calculation -> results report.

Thin runner: it builds ONE tree and executes it. All input paths are get_path REFERENCE paths
resolved in coastal_carbon_initialize.initialize_paths (one source of truth for every runner);
base_data_dir is resolved by ProjectFlow (its default, overridable per machine via
~/.config/hazelbean/machine.env), never hardcoded here.
"""
import os
import hazelbean as hb

from global_invest.coastal_carbon import coastal_carbon_initialize

if __name__ == '__main__':

    p = hb.ProjectFlow()
    p.project_name = 'gep_coastal_carbon'
    p.project_dir = os.path.join(os.path.expanduser('~'), 'Files', 'global_invest', 'projects', p.project_name)
    p.set_project_dir(p.project_dir)  # Sets p.base_data_dir (default / machine.env), p.input_dir, p.intermediate_dir, p.output_dir.

    coastal_carbon_initialize.build_gep_service_task_tree(p, include_seagrass=True)

    p.results = {}
    coastal_carbon_initialize.initialize_paths(p)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script
           + '\n    with base_data set at ' + p.base_data_dir)
    p.execute()

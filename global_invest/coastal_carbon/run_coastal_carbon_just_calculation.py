"""Calculation-only coastal-carbon GEP run: the per-ecosystem chains + combine + gep_calculation,
without the results report. Same thin-runner shape as run_coastal_carbon.py: one tree, all inputs
as get_path reference paths via initialize_paths, base_data_dir from ProjectFlow / machine.env.
"""
import os
import hazelbean as hb

from global_invest.coastal_carbon import coastal_carbon_initialize

if __name__ == '__main__':

    p = hb.ProjectFlow()
    p.project_name = 'gep_coastal_carbon'
    p.project_dir = os.path.join(os.path.expanduser('~'), 'Files', 'global_invest', 'projects', p.project_name)
    p.set_project_dir(p.project_dir)

    coastal_carbon_initialize.build_gep_service_calculation_task_tree(p, include_seagrass=True)

    p.results = {}
    coastal_carbon_initialize.initialize_paths(p)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script
           + '\n    with base_data set at ' + p.base_data_dir)
    p.execute()

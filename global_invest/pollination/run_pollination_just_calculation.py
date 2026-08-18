"""Calculation-only pollination GEP run: summarize + valuation, without the results report.
Same thin-runner shape as run_pollination.py."""
import os
import hazelbean as hb

from global_invest.pollination import pollination_initialize

if __name__ == '__main__':

    p = hb.ProjectFlow()
    p.project_name = 'gep_pollination'
    p.project_dir = os.path.join(os.path.expanduser('~'), 'Files', 'global_invest', 'projects', p.project_name)
    p.set_project_dir(p.project_dir)

    pollination_initialize.build_gep_service_calculation_task_tree(p)

    p.results = {}
    pollination_initialize.initialize_paths(p)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script
           + '\n    with base_data set at ' + p.base_data_dir)
    p.execute()

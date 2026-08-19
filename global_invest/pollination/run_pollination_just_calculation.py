"""Calculation-only pollination GEP run: summarize + valuation, without the results report.
Same thin-runner shape as run_pollination.py."""
import hazelbean as hb

from global_invest.pollination import pollination_initialize

def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    pollination_initialize.build_gep_service_calculation_task_tree(p)


if __name__ == '__main__':

    p = hb.ProjectFlow(project_name='gep_pollination', run_mode='check')

    build_task_tree(p)

    p.results = {}
    pollination_initialize.initialize_paths(p)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

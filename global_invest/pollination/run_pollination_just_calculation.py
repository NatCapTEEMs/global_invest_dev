"""Calculation-only pollination GEP run: summarize + valuation, without the results report.
Same thin-runner shape as run_pollination.py."""
import hazelbean as hb

from global_invest.pollination import pollination_initialize

def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    pollination_initialize.build_gep_service_calculation_task_tree(p)


def run_project(p):

    build_task_tree(p)

    p.results = {}
    pollination_initialize.initialize_paths(p)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

    return p


if __name__ == '__main__':

    # Create ProjectFlow object
    p = hb.ProjectFlow(project_name='gep_pollination', run_mode='check')

    # Run the project
    run_project(p)

    result = 'Done!'

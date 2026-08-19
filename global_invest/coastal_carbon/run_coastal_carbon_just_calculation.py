"""Calculation-only coastal-carbon GEP run: the per-ecosystem chains + combine + gep_calculation,
without the results report. Same thin-runner shape as run_coastal_carbon.py: one tree, all inputs
as get_path reference paths via initialize_paths, base_data_dir from ProjectFlow / machine.env.
"""
import hazelbean as hb

from global_invest.coastal_carbon import coastal_carbon_initialize

def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    coastal_carbon_initialize.build_gep_service_calculation_task_tree(p, include_seagrass=True)


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

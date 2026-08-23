"""Full erosion GEP run: InVEST SDR -> prevention-share valuation -> maps/figures.

Thin runner: one tree, executed. The calculation is cluster-scale (global InVEST SDR); the SDR input/
output locations and table paths are set on p here or by the submitting script -- the configure_*
functions in erosion_functions.py read them off p at run time (their defaults point at the source
repo's cluster layout). The ES-shock runner is run_erosion_shock.py -- a separate tree, a separate
thin runner, no mode switch.
"""
import hazelbean as hb

from global_invest.erosion import erosion_initialize

def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    erosion_initialize.build_gep_service_task_tree(p)


def run_project(p):

    # Every task publishes its own inputs (publish_inputs in the tasks module): no setup call.
    build_task_tree(p)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

    return p


if __name__ == '__main__':

    # Create ProjectFlow object
    p = hb.ProjectFlow(project_name='gep_erosion', run_mode='check')

    # Run the project
    run_project(p)

    result = 'Done!'

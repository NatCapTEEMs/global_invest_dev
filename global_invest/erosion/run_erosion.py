"""Full erosion GEP run: InVEST SDR -> prevention-share valuation -> results report.

Thin runner: one tree, executed. The calculation is cluster-scale (global InVEST SDR); the SDR input/
output locations and table paths are es_parameters rows that each task hydrates through
publish_inputs, and a submitting script may set them on p first. The ES-shock runner is
run_erosion_shock.py -- a separate tree, a separate thin runner, no mode switch.
"""
import hazelbean as hb

from global_invest.erosion import erosion_initialize

def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    erosion_initialize.build_gep_service_task_tree(p)


def run_project(p):

    # Every task publishes its own inputs (publish_inputs in the tasks module): no setup call.
    build_task_tree(p)
    # Skipping is run configuration, so the CHOICE belongs to the caller and applying it belongs
    # here. The tree is always built in full, so its structure is identical across variants and
    # only the run flags differ. A run that sets nothing skips nothing.
    p.skip_tasks(getattr(p, 'tasks_to_skip', None))

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

    return p


if __name__ == '__main__':

    # Create ProjectFlow object
    p = hb.ProjectFlow(project_name='gep_erosion', run_mode='check')

    # Run the project
    run_project(p)

    result = 'Done!'

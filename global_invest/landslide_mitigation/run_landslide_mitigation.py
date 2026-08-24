"""Full landslide-mitigation GEP run. Thin runner: one tree, executed. Folded from
m-braaksma/landslide_mitigation v0.2.0."""
import hazelbean as hb

from global_invest.landslide_mitigation import landslide_mitigation_initialize

def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    landslide_mitigation_initialize.build_gep_service_task_tree(p)


def run_project(p):

    # Every task publishes its own inputs (publish_inputs in the tasks module): no setup call.
    build_task_tree(p)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

    return p


if __name__ == '__main__':

    # Create ProjectFlow object
    p = hb.ProjectFlow(project_name='gep_landslide_mitigation', run_mode='check')

    # Run the project
    run_project(p)

    result = 'Done!'

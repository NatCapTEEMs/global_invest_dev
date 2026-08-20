import pandas as pd
import hazelbean as hb

from global_invest.livestock_provision import livestock_provision_initialize

# TODO Note massive violation of DRY (Dont repeat yourself) here. This is a copy of the crop_provision run file, but with livestock_provision instead of crop_provision. I took this shortcut cause I couldn't think of the right way of combining the multiple different provisioning services.

def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    livestock_provision_initialize.build_gep_service_task_tree(p)


def run_project(p):

    # Task tree
    build_task_tree(p)

    # Project level attributes

    # Run the model
    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)    
    p.execute()

    return p


if __name__ == '__main__':

    # Create ProjectFlow object
    p = hb.ProjectFlow(project_name='gep_livestock_provision', run_mode='check')

    # Run the project
    run_project(p)

    result = 'Done!'

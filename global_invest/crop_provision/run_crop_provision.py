import pandas as pd
import hazelbean as hb

from global_invest.crop_provision import crop_provision_initialize

def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    crop_provision_initialize.build_gep_service_task_tree(p)


def run_project(p):

    # Task tree
    build_task_tree(p)


    # Run the model
    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)    
    p.execute()

    return p


if __name__ == '__main__':

    # Create ProjectFlow object
    p = hb.ProjectFlow(project_name='gep_crop_provision', run_mode='check')

    # Run the project
    run_project(p)

    result = 'Done!'

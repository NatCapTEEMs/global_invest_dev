import pandas as pd
import hazelbean as hb

from global_invest.livestock_provision import livestock_provision_initialize

# TODO Note massive violation of DRY (Dont repeat yourself) here. This is a copy of the crop_provision run file, but with livestock_provision instead of crop_provision. I took this shortcut cause I couldn't think of the right way of combining the multiple different provisioning services.

def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    livestock_provision_initialize.build_gep_service_task_tree(p)


if __name__ == '__main__':
    
    """Simplified run file that assumes the user has already run the project and just wants to rerender the results."""
    
    # ProjectFlow object
    p = hb.ProjectFlow(project_name='gep_livestock_provision', run_mode='check')
    
    # Task tree
    build_task_tree(p)

    # Project level attributes
    p.results = {}  # All results will be stored here by each child task.
    livestock_provision_initialize.initialize_paths(p)

    # Run the model
    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)    
    p.execute()
    
    result = 'Done!'
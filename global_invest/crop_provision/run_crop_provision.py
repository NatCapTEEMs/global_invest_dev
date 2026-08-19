import pandas as pd
import hazelbean as hb

from global_invest.crop_provision import crop_provision_initialize

def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    crop_provision_initialize.build_gep_service_task_tree(p)


if __name__ == '__main__':
    
    """Simplified run file that assumes the user has already run the project and just wants to rerender the results."""
    
    # ProjectFlow object
    p = hb.ProjectFlow(project_name='gep_crop_provision', run_mode='check')
    
    # Task tree
    build_task_tree(p)

    # Inputs resolve in initialize_paths (one source of truth; shared country block).
    p.results = {}  # All results will be stored here by each child task.
    crop_provision_initialize.initialize_paths(p)

    # Run the model
    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)    
    p.execute()
    
    result = 'Done!'
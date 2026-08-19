import pandas as pd
import hazelbean as hb

from global_invest.extractive_materials_provision import extractive_materials_provision_initialize

def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    extractive_materials_provision_initialize.build_gep_service_calculation_task_tree(p)


if __name__ == '__main__':
    
    # ProjectFlow object
    p = hb.ProjectFlow(project_name='gep_extractive_materials_provision', run_mode='check')
    
    # Task tree
    build_task_tree(p)

    # Inputs resolve in initialize_paths (one source of truth; shared country block).
    p.results = {}  # All results will be stored here by each child task.
    extractive_materials_provision_initialize.initialize_paths(p)

    # Run the model
    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)    
    p.execute()
    
    result = 'Done!'
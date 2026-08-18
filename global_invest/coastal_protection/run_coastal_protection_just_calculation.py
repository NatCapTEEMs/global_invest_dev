# just calculation
import os
import pandas as pd
import hazelbean as hb

from global_invest.coastal_protection import coastal_protection_initialize

if __name__ == '__main__':
    
    # ProjectFlow object
    p = hb.ProjectFlow() # Create a ProjectFlow Object to organize directories and enable parallel processing.
    p.project_name = 'gep_coastal_protection'  # Determines the folder created to store intermediate and final results.
    p.project_dir = os.path.join(os.path.expanduser('~'), 'Files', 'global_invest', 'projects', p.project_name) # Put it in the right location relative to the user's home directory.
    # base_data_dir resolves via ProjectFlow default / machine.env (never hardcoded).
    p.set_project_dir(p.project_dir) # Set the project directory in the ProjectFlow object. Also defines p.input_dir, p.intermediate_dir, and p.output_dir based on the project_dir.

    # Task tree
    coastal_protection_initialize.build_gep_service_calculation_task_tree(p) # Defines the actual logic of the model. Navigate into here to see what the model does.

    # Inputs resolve in initialize_paths (one source of truth; shared country block).
    p.results = {}  # All results will be stored here by each child task.
    coastal_protection_initialize.initialize_paths(p)

    # Run the model
    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script + '\n    with base_data set at ' + p.base_data_dir)    
    p.execute()
    
    result = 'Done!'
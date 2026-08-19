"""Standalone runner for the terrestrial-carbon GEP CALCULATION only (no results report).

Runs the compute chain and writes the per-country GEP, without rendering the report. For the full GEP
run (calculation + report) use run_terrestrial_carbon.py; for the shock, run_terrestrial_carbon_shock.py.
base_data_dir is resolved by ProjectFlow (default / machine.env); inputs are resolved in initialize_paths.
"""
import os
import pandas as pd
import hazelbean as hb

from global_invest.terrestrial_carbon import terrestrial_carbon_initialize

if __name__ == '__main__':

    # ProjectFlow object
    p = hb.ProjectFlow() # Create a ProjectFlow Object to organize directories and enable parallel processing.
    p.project_name = 'gep_terrestrial_carbon'  # Determines the folder created to store intermediate and final results.
    p.project_dir = os.path.join(os.path.expanduser('~'), 'Files', 'global_invest', 'projects', p.project_name) # Put it in the right location relative to the user's home directory.
    p.set_project_dir(p.project_dir) # Sets p.base_data_dir (default / machine.env), p.input_dir, p.intermediate_dir, p.output_dir.

    # Task tree
    terrestrial_carbon_initialize.build_gep_service_calculation_task_tree(p) # Defines the actual logic of the model. Navigate into here to see what the model does.

    # Project-level attributes resolved in initialize_paths (one source of truth, all via get_path).
    p.results = {}  # All results will be stored here by each child task.
    terrestrial_carbon_initialize.initialize_paths(p)

    # Run the model
    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script + '\n    with base_data set at ' + p.base_data_dir)
    p.execute()

    result = 'Done!'








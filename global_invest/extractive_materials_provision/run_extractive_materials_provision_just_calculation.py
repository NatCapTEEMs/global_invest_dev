import os
import pandas as pd
import hazelbean as hb

from global_invest.extractive_materials_provision import extractive_materials_provision_initialization

if __name__ == '__main__':
    
    # ProjectFlow object
    p = hb.ProjectFlow() # Create a ProjectFlow Object to organize directories and enable parallel processing.
    p.project_name = 'gep_extractive_materials_provision'  # Determines the folder created to store intermediate and final results.
    p.project_dir = os.path.join(os.path.expanduser('~'), 'Files', 'global_invest', 'projects', p.project_name) # Put it in the right location relative to the user's home directory.
    p.base_data_dir = "G:/Shared drives/NatCapTEEMs/Files/base_data" # Set where data outside the project will be stored. CAUTION: For GEP we are using the shared Google Drive, but best practice is to use a local directory that you can control (also it's faster)
    p.set_project_dir(p.project_dir) # Set the project directory in the ProjectFlow object. Also defines p.input_dir, p.intermediate_dir, and p.output_dir based on the project_dir.
    
    # Task tree
    extractive_materials_provision_initialization.build_gep_service_calculation_task_tree(p) # Defines the actual logic of the model. Navigate into here to see what the model does.

    # Project level attributes
    p.df_countries_csv_path = p.get_path('cartographic', 'ee', 'ee_r264_correspondence.csv') # ProjectFlow downloads all files automatically via the p.get_path() function. 
    p.gdf_countries_vector_path = p.get_path('cartographic', 'ee', 'ee_r264_correspondence.gpkg') 
    p.results = {}  # All results will be stored here by each child task.
    extractive_materials_provision_initialization.initialize_paths(p)

    # Run the model
    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script + '\n    with base_data set at ' + p.base_data_dir)    
    p.execute()
    
    result = 'Done!'
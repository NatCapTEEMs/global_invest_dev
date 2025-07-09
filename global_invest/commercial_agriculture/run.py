import os
import pandas as pd
import hazelbean as hb

from global_invest.commercial_agriculture import commercial_agriculture_initialization, commercial_agriculture_tasks


if __name__ == '__main__':
    
    # ProjectFlow object
    p = hb.ProjectFlow() # Create a ProjectFlow Object to organize directories and enable parallel processing.
    p.project_name = 'gep_commercial_agriculture'  # Determines the folder created to store intermediate and final results.
    p.project_dir = os.path.join(os.path.expanduser('~'), 'Files', 'global_invest', 'projects', p.project_name) # Put it in the right location relative to the user's home directory.
    p.base_data_dir = "G:/Shared drives/NatCapTEEMs/Files/base_data" # Set where data outside the project will be stored. CAUTION: For GEP we are using the shared Google Drive, but best practice is to use a local directory that you can control (also it's faster)
    p.data_credentials_path = None # If you want to use a different bucket than the default, provide the credentials here. Otherwise uses default public data 'gtap_invest_seals_2023_04_21'.
    p.input_bucket_name = None # If you want to use a different bucket than the default, provide the name here. Otherwise uses default public data 'gtap_invest_seals_2023_04_21'.
    p.set_project_dir(p.project_dir) # Set the project directory in the ProjectFlow object. Also defines p.input_dir, p.intermediate_dir, and p.output_dir based on the project_dir.
    
    # Task tree
    commercial_agriculture_initialization.build_standard_task_tree(p) # Defines the actual logic of the model. Navigate into here to see what the model does.

    # Project level attributes
    p.countries_csv_path = p.get_path('cartographic', 'ee', 'ee_r264_correspondence.csv') # ProjectFlow downloads all files automatically via the p.get_path() function. 
    p.countries_vector_path = p.get_path('cartographic', 'ee', 'ee_r264_correspondence.gpkg') 
    p.results = {}  # All results will be stored here by each child task.
    commercial_agriculture_initialization.initialize_paths(p)


    
    # START HERE: Finish the minimal run.py for commecial agriculture. Then make more detailed run files for the input_repo stype and the projectflow style.
    

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script + '\n    with base_data set at ' + p.base_data_dir)
    
    p.execute()
    
    result = 'Done!'


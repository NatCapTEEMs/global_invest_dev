import os, sys
import hazelbean as hb
import pandas as pd

from global_invest.example_service import example_service_tasks

# Create the project flow object
p = hb.ProjectFlow()

# Set project-directories
p.user_dir = os.path.expanduser('~')        
p.extra_dirs = ['Files', 'global_invest', 'projects']
p.project_name = 'test_global_invest'
p.project_name = p.project_name + '_' + hb.pretty_time() # Comment this line out if you want it to use an existing project. Will skip recreation of files that already exist.
p.project_dir = os.path.join(p.user_dir, os.sep.join(p.extra_dirs), p.project_name)
p.set_project_dir(p.project_dir) 

# Set basa_data_dir. Will download required files here.
p.base_data_dir = os.path.join(p.user_dir, 'Files', 'base_data')
    
# Set model-paths
p.aoi = 'global'
p.base_year_lulc_path = p.get_path('lulc/esa/lulc_esa_2019.tif') # Defines the fine_resolution
p.region_ids_coarse_path = p.get_path('cartographic/ee/id_rasters/eemarine_r566_ids_900sec.tif') # Defines the coarse_resolution
p.global_regions_vector_path = p.get_path('cartographic/ee/eemarine_r566_correspondence.gpkg') # Will be used to create the aoi vector

def build_task_tree(p):
    p.example_parent_task = p.add_task(example_service_tasks.example_parent_task)
    p.example_service_task = p.add_task(example_service_tasks.example_task, parent=p.example_parent_task)

# Build the task tree and excute it!
build_task_tree(p)
p.execute()
    

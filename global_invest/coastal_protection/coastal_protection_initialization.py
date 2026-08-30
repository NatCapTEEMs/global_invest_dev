import os

import pandas as pd
import hazelbean as hb

from global_invest.coastal_protection import coastal_protection_tasks


PROJECT_NAME = 'gep_coastal_protection'
DEFAULT_BASE_DATA_DIR = (
    '/Users/long/Library/CloudStorage/GoogleDrive-yxlong@umn.edu/'
    'Shared drives/NatCapTEEMs/Files/base_data/submissions'
)
DEFAULT_SIMPLIFIED_COUNTRIES = 'ee_r264_simplified30sec.gpkg'


def create_projectflow(project_dir=None, base_data_dir=None):
    """Create standard coastal-protection ProjectFlow configuration."""
    p = hb.ProjectFlow()
    p.project_name = PROJECT_NAME
    p.project_dir = project_dir or os.path.join(
        os.path.expanduser('~'), 'Files', 'global_invest', 'projects', PROJECT_NAME
    )
    p.base_data_dir = base_data_dir or DEFAULT_BASE_DATA_DIR
    p.set_project_dir(p.project_dir)
    return p


def initialize_paths(p):
    """Resolve shared country inputs and keep lightweight country-table metadata."""
    p.df_countries = pd.read_csv(p.df_countries_csv_path)
    p.gdf_countries = p.gdf_countries_vector_path
    p.gdf_countries_simplified = p.gdf_countries_vector_simplified_path


def initialize_project_inputs(p, simplified_countries=DEFAULT_SIMPLIFIED_COUNTRIES):
    """Resolve standard country inputs and initialize result paths."""
    p.df_countries_csv_path = p.get_path(
        'cartographic', 'ee', 'ee_r264_correspondence.csv'
    )
    p.gdf_countries_vector_path = p.get_path(
        'cartographic', 'ee', 'ee_r264_correspondence.gpkg'
    )
    p.gdf_countries_vector_simplified_path = p.get_path(
        'cartographic', 'ee', simplified_countries
    )
    p.results = {}
    initialize_paths(p)
    return p


def build_gep_service_calculation_task_tree(p):
    """Build calculation-only task tree."""
    p.coastal_protection_task = p.add_task(
        coastal_protection_tasks.coastal_protection
    )
    p.coastal_protection_gep_calculation_task = p.add_task(
        coastal_protection_tasks.gep_calculation,
        parent=p.coastal_protection_task,
    )
    return p


def build_gep_service_task_tree(p):
    """Build calculation and report-rendering task tree."""
    build_gep_service_calculation_task_tree(p)
    p.coastal_protection_gep_result_task = p.add_task(
        coastal_protection_tasks.gep_result,
        parent=p.coastal_protection_task,
    )
    return p


def build_gep_task_tree(p):
    """Build full task tree, including result distribution."""
    build_gep_service_task_tree(p)
    p.coastal_protection_gep_results_distribution_task = p.add_task(
        coastal_protection_tasks.gep_results_distribution,
        parent=p.coastal_protection_task,
    )
    return p

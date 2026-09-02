import pandas as pd
import hazelbean as hb

from global_invest.coastal_carbon import coastal_carbon_tasks


def initialize_paths(p):
    """Initialize path references."""
    marine_zones = pd.read_csv(p.df_eez_csv_path)
    p.df_eez = marine_zones.loc[
        marine_zones['eemarine_r566_label'].astype(str).str.endswith('_EEZ')
    ].copy()

    # Notice optimization here: the GDFs are still just path_strings
    p.gdf_eez = p.gdf_eez_vector_path
    p.gdf_eez_simplified = p.gdf_eez_vector_path


# ============================================================================
# Per-ecosystem task trees
#
# Each ecosystem flow follows the same three-step pattern:
#   1. Area within EEZs  (rasterize extent, intersect with EEZ, sum ha)
#   2. Carbon stock           (per-pixel density x ha, sum to EEZ)
#   3. Storage value          (stock x rental SCC for the base year)
# ============================================================================

def build_mangrove_carbon_calculation_task_tree(p):
    """Add mangrove area, stock, and storage-value tasks to the task tree."""
    p.task_calculate_mangrove_area = p.add_task(
        coastal_carbon_tasks.task_calculate_mangrove_area_within_eez
    )
    p.task_calculate_mangrove_carbon_stock = p.add_task(
        coastal_carbon_tasks.task_calculate_mangrove_carbon_stock
    )
    p.task_calculate_mangrove_storage_value = p.add_task(
        coastal_carbon_tasks.task_calculate_mangrove_storage_value
    )
    return p


def build_marsh_carbon_calculation_task_tree(p):
    """Add salt marsh area, stock, and storage-value tasks to the task tree."""
    p.task_calculate_salt_marsh_area = p.add_task(
        coastal_carbon_tasks.task_calculate_salt_marsh_area_within_eez
    )
    p.task_calculate_salt_marsh_carbon_stock = p.add_task(
        coastal_carbon_tasks.task_calculate_salt_marsh_carbon_stock
    )
    p.task_calculate_salt_marsh_storage_value = p.add_task(
        coastal_carbon_tasks.task_calculate_salt_marsh_storage_value
    )
    return p


def build_seagrass_carbon_calculation_task_tree(p):
    """Add GlobalSeagrass area, stock, and storage-value tasks."""
    p.task_calculate_seagrass_area = p.add_task(
        coastal_carbon_tasks.task_calculate_seagrass_area_within_eez
    )
    p.task_calculate_seagrass_carbon_stock = p.add_task(
        coastal_carbon_tasks.task_calculate_seagrass_carbon_stock
    )
    p.task_calculate_seagrass_storage_value = p.add_task(
        coastal_carbon_tasks.task_calculate_seagrass_storage_value
    )
    return p


# ============================================================================
# Composite trees
# ============================================================================

def build_gep_service_calculation_task_tree(p):
    """Build all ecosystem calculations, combine outputs, and calculate GEP."""
    build_mangrove_carbon_calculation_task_tree(p)
    build_marsh_carbon_calculation_task_tree(p)
    build_seagrass_carbon_calculation_task_tree(p)

    # Cross-ecosystem aggregation
    p.task_combine_ecosystem_areas = p.add_task(
        coastal_carbon_tasks.task_combine_ecosystem_areas
    )
    p.task_gep_calculation = p.add_task(
        coastal_carbon_tasks.gep_calculation
    )

    return p


def build_gep_service_task_tree(p):
    """Full calculation tree plus the Quarto results task."""
    p = build_gep_service_calculation_task_tree(p)
    p.coastal_carbon_gep_result_task = p.add_task(
        coastal_carbon_tasks.gep_result
    )
    return p

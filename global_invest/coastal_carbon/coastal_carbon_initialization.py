import pandas as pd
import hazelbean as hb

from global_invest.coastal_carbon import coastal_carbon_tasks


def initialize_paths(p):
    """Initialize path references."""
    p.df_countries = pd.read_csv(p.df_countries_marine_csv_path)

    # Notice optimization here: the GDFs are still just path_strings
    p.gdf_countries = p.gdf_countries_marine_vector_path
    p.gdf_countries_simplified = p.gdf_countries_marine_vector_path


# ============================================================================
# Per-ecosystem task trees
#
# Each ecosystem flow follows the same three-step pattern:
#   1. Area within countries  (rasterize extent, intersect with EEZ, sum ha)
#   2. Carbon stock           (per-pixel density x ha, sum to country)
#   3. Storage value          (stock x rental SCC for the base year)
# ============================================================================

def build_mangrove_carbon_calculation_task_tree(p):
    """Add mangrove area, stock, and storage-value tasks to the task tree."""
    p.task_calculate_mangrove_area = p.add_task(
        coastal_carbon_tasks.task_calculate_mangrove_area_within_countries
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
        coastal_carbon_tasks.task_calculate_salt_marsh_area_within_countries
    )
    p.task_calculate_salt_marsh_carbon_stock = p.add_task(
        coastal_carbon_tasks.task_calculate_salt_marsh_carbon_stock
    )
    p.task_calculate_salt_marsh_storage_value = p.add_task(
        coastal_carbon_tasks.task_calculate_salt_marsh_storage_value
    )
    return p


def build_seagrass_carbon_calculation_task_tree(p):
    """
    Add seagrass area, stock, and storage-value tasks to the task tree.

    Seagrass tasks are currently stubs that print a NotImplemented notice and
    return early. To activate, replace the stubs in coastal_carbon_tasks.py
    with real implementations following the mangrove or salt marsh pattern.
    """
    p.task_calculate_seagrass_area = p.add_task(
        coastal_carbon_tasks.task_calculate_seagrass_area_within_countries
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

def build_gep_service_calculation_task_tree(p, include_seagrass=False):
    """
    Build the full coastal carbon GEP calculation task tree.

    Composes the per-ecosystem trees (mangrove + salt marsh, with seagrass
    optional) and appends the cross-ecosystem combine + GEP tasks.

    Parameters
    ----------
    p : hb.ProjectFlow
    include_seagrass : bool
        If True, append the seagrass stub tree. Defaults to False since the
        seagrass tasks are not yet implemented.
    """
    build_mangrove_carbon_calculation_task_tree(p)
    build_marsh_carbon_calculation_task_tree(p)
    if include_seagrass:
        build_seagrass_carbon_calculation_task_tree(p)

    # Cross-ecosystem aggregation
    p.task_combine_ecosystem_areas = p.add_task(
        coastal_carbon_tasks.task_combine_ecosystem_areas
    )
    p.task_gep_calculation = p.add_task(
        coastal_carbon_tasks.gep_calculation
    )

    return p


def build_gep_service_task_tree(p, include_seagrass=False):
    """Full calculation tree plus the Quarto results task."""
    p = build_gep_service_calculation_task_tree(p, include_seagrass=include_seagrass)
    p.coastal_carbon_gep_result_task = p.add_task(
        coastal_carbon_tasks.gep_result
    )
    return p

"""coastal_carbon wiring.

The develop_yanxu rework is MERGED onto this branch (2026-08-15): per-ecosystem task trees
(mangrove + salt marsh + seagrass) composed into the GEP calculation tree, real
area->stock->storage-value chains, results.qmd + references.bib, and a gep_calculation that
enforces the r250-only rule via the canonical `ee_r264_label == iso3_r250_label` filter
(see global_invest/utilities.py). This file was renamed from coastal_carbon_initialization.py
to match the other services.

Conform state relative to the terrestrial_carbon template:
- [x] configuration is data: the aggregation surface + id column are es_config cells, the
      science inputs are es_parameters rows, and every config-consuming task opens with
      publish_inputs (coastal_carbon_tasks) -- no initialize_paths, no setup call anywhere.
- [x] ProjectFlow-native skip: data tasks registered with skip_existing=1; each publishes its
      output paths then `if not p.run_this: return`. _task_outputs_exist survives only as an
      internal sub-step cache; the pipeline-wide _final_result_exists short-circuit is gone.
- [ ] hazelbean-first pass over the raster ops (rasterio/rasterstats -> hb equivalents),
      each swap verified against the cached output.
- [ ] number-verify the chain against the coastal source data (NatCapTEEMs Drive).
"""
from global_invest import utilities

from global_invest.coastal_carbon import coastal_carbon_tasks


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
    p.mangrove_area_within_countries = p.add_task(
        coastal_carbon_tasks.mangrove_area_within_countries, skip_existing=1
    )
    p.mangrove_carbon_stock = p.add_task(
        coastal_carbon_tasks.mangrove_carbon_stock, skip_existing=1
    )
    p.mangrove_storage_value = p.add_task(
        coastal_carbon_tasks.mangrove_storage_value, skip_existing=1
    )
    return p


def build_marsh_carbon_calculation_task_tree(p):
    """Add salt marsh area, stock, and storage-value tasks to the task tree."""
    p.salt_marsh_area_within_countries = p.add_task(
        coastal_carbon_tasks.salt_marsh_area_within_countries, skip_existing=1
    )
    p.salt_marsh_carbon_stock = p.add_task(
        coastal_carbon_tasks.salt_marsh_carbon_stock, skip_existing=1
    )
    p.salt_marsh_storage_value = p.add_task(
        coastal_carbon_tasks.salt_marsh_storage_value, skip_existing=1
    )
    return p


def build_seagrass_carbon_calculation_task_tree(p):
    """Add seagrass area, stock, and storage-value tasks to the task tree.

    Implemented: WCMC v7.1 extent -> per-country intersection (GENUS kept) ->
    genus-aware Gomis 2025 pool densities -> storage value. A missing extent
    RAISES in the area task; exclude seagrass by not building this tree
    (include_seagrass=False), never via a silent data-gap skip.
    """
    p.seagrass_area_within_countries = p.add_task(
        coastal_carbon_tasks.seagrass_area_within_countries, skip_existing=1
    )
    p.seagrass_carbon_stock = p.add_task(
        coastal_carbon_tasks.seagrass_carbon_stock, skip_existing=1
    )
    p.seagrass_storage_value = p.add_task(
        coastal_carbon_tasks.seagrass_storage_value, skip_existing=1
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
        If True, append the seagrass tree (implemented; needs the WCMC v7.1
        extent staged -- the run files pass True).
    """
    build_mangrove_carbon_calculation_task_tree(p)
    build_marsh_carbon_calculation_task_tree(p)
    if include_seagrass:
        build_seagrass_carbon_calculation_task_tree(p)

    # Cross-ecosystem aggregation
    p.combined_ecosystem_areas = p.add_task(
        coastal_carbon_tasks.combined_ecosystem_areas, skip_existing=1
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

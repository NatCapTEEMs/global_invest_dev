"""coastal_carbon wiring.

The develop_yanxu rework is MERGED onto this branch (2026-08-15): per-ecosystem task trees
(mangrove + salt marsh + seagrass) composed into the GEP calculation tree, real
area->stock->storage-value chains, results.qmd + references.bib, and a gep_calculation that
enforces the r250-only rule via the canonical `ee_r264_label == iso3_r250_label` filter
(see global_invest/utilities.py). This file was renamed from coastal_carbon_initialization.py
to match the other services.

Conform state relative to the terrestrial_carbon template:
- [x] input paths live HERE as get_path REFERENCE paths (no absolute paths, no per-run-file
      duplication) -- one source of truth for every runner and the results report.
- [x] ProjectFlow-native skip: data tasks registered with skip_existing=1; each publishes its
      output paths then `if not p.run_this: return`. _task_outputs_exist survives only as an
      internal sub-step cache; the pipeline-wide _final_result_exists short-circuit is gone.
- [ ] hazelbean-first pass over the raster ops (rasterio/rasterstats -> hb equivalents),
      each swap verified against the cached output.
- [ ] number-verify the chain against the coastal source data (NatCapTEEMs Drive).
"""
import pandas as pd
import hazelbean as hb

from global_invest import utilities

from global_invest.coastal_carbon import coastal_carbon_tasks


def initialize_paths(p):
    """Resolve every coastal-carbon input on p via get_path (machine-agnostic reference paths;
    base_data_dir itself comes from ProjectFlow default / machine.env). One source of truth for
    all run files and the results report. p.gep_price_convention may be overridden before calling.
    """
    # Country boundaries: marine EEZ (r566) for the per-region chains, terrestrial r264 for the
    # final iso3_r250 aggregation in gep_calculation.
    p.df_countries_marine_csv_path = p.get_path('cartographic', 'ee', 'eemarine_r566_correspondence.csv')
    p.gdf_countries_marine_vector_path = p.get_path('cartographic', 'ee', 'eemarine_r566_correspondence.gpkg')
    utilities.initialize_country_paths(p, simplified='30sec')   # shared r264 block

    # Ecosystem extents: Global Mangrove Watch v3 (2019), salt marsh processed from GWL_FCS30D,
    # seagrass UNEP-WCMC013-014 v7.1 (GENUS attribute consumed by the seagrass stock task).
    p.mangrove_vector_path = p.get_path('global_invest', 'coastal_carbon', 'gmw_v3_2019_vec', 'gmw_v3_2019_vec.shp')
    p.salt_marsh_vector_path = p.get_path('global_invest', 'coastal_carbon', 'global_salt_marsh2019.gpkg')
    p.seagrass_vector_path = p.get_path('global_invest', 'coastal_carbon', '014_001_WCMC013-014_SeagrassPtPy2021_v7_1',
                                        '01_Data', 'WCMC013_014_Seagrasses_Py_v7_1.shp')

    # Rasters: Sanderman et al. 2018 mangrove SOC (top 1 m, Mg C/ha; Zenodo 7727569, median
    # typology 2019-2020, EPSG:4326) and optional mean annual precipitation (mm/yr) for the
    # tropical wet/dry split in the IPCC BGB:AGB ratio (absent -> all tropics wet, 0.49).
    p.mangrove_soc_path = p.get_path('global_invest', 'coastal_carbon',
                                     'soc.tha_tnc.mangroves.typology_m_30m_b0..100cm_2019_2020_go_epsg.4326_v1.2.tif')
    # Maxwell et al. 2024 tidal-marsh SOC (top 1 m, Mg C/ha; Zenodo 10940066, pred0+pred30 summed
    # to one 0-100 cm GeoTIFF). OPTIONAL inputs resolve with raise_error_if_fail=False: the stock
    # tasks carry documented fallbacks when these files are absent (latitude step fn for marsh SOC;
    # all-tropics-wet BGB ratio for precipitation), so a missing file is a configuration, not an error.
    p.salt_marsh_soc_path = p.get_path('global_invest', 'coastal_carbon', 'maxwell_2024_marsoc_0_100cm.tif',
                                       raise_error_if_fail=False)
    p.precipitation_path = p.get_path('global_invest', 'coastal_carbon', 'mean_annual_precipitation_mm.tif',
                                      raise_error_if_fail=False)
    p.ha_per_cell_10sec_path = p.get_path('pyramids', 'ha_per_cell_10sec.tif')

    # Valuation configuration (price input, price convention, base year) comes from
    # es_config.csv as a defaults layer: anything the caller already set on p wins.
    utilities.hydrate_es_config(p, 'coastal_carbon')

    # Coastal aggregates on the MARINE surface: override the shared aliases with the r566 vector
    # (initialize_country_paths set the terrestrial r264 ones; the r264 csv still feeds the final
    # iso3_r250 aggregation in gep_calculation).
    p.df_countries = pd.read_csv(p.df_countries_marine_csv_path)
    p.gdf_countries = p.gdf_countries_marine_vector_path
    p.gdf_countries_simplified = p.gdf_countries_marine_vector_path
    return p


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
        coastal_carbon_tasks.task_calculate_mangrove_area_within_countries, skip_existing=1
    )
    p.task_calculate_mangrove_carbon_stock = p.add_task(
        coastal_carbon_tasks.task_calculate_mangrove_carbon_stock, skip_existing=1
    )
    p.task_calculate_mangrove_storage_value = p.add_task(
        coastal_carbon_tasks.task_calculate_mangrove_storage_value, skip_existing=1
    )
    return p


def build_marsh_carbon_calculation_task_tree(p):
    """Add salt marsh area, stock, and storage-value tasks to the task tree."""
    p.task_calculate_salt_marsh_area = p.add_task(
        coastal_carbon_tasks.task_calculate_salt_marsh_area_within_countries, skip_existing=1
    )
    p.task_calculate_salt_marsh_carbon_stock = p.add_task(
        coastal_carbon_tasks.task_calculate_salt_marsh_carbon_stock, skip_existing=1
    )
    p.task_calculate_salt_marsh_storage_value = p.add_task(
        coastal_carbon_tasks.task_calculate_salt_marsh_storage_value, skip_existing=1
    )
    return p


def build_seagrass_carbon_calculation_task_tree(p):
    """Add seagrass area, stock, and storage-value tasks to the task tree.

    Implemented: WCMC v7.1 extent -> per-country intersection (GENUS kept) ->
    genus-aware Gomis 2025 pool densities -> storage value. A missing extent
    RAISES in the area task; exclude seagrass by not building this tree
    (include_seagrass=False), never via a silent data-gap skip.
    """
    p.task_calculate_seagrass_area = p.add_task(
        coastal_carbon_tasks.task_calculate_seagrass_area_within_countries, skip_existing=1
    )
    p.task_calculate_seagrass_carbon_stock = p.add_task(
        coastal_carbon_tasks.task_calculate_seagrass_carbon_stock, skip_existing=1
    )
    p.task_calculate_seagrass_storage_value = p.add_task(
        coastal_carbon_tasks.task_calculate_seagrass_storage_value, skip_existing=1
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
    p.task_combine_ecosystem_areas = p.add_task(
        coastal_carbon_tasks.task_combine_ecosystem_areas, skip_existing=1
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

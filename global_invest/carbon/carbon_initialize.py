"""Task-tree builders for the global_invest carbon (storage & sequestration) model.

`add_carbon_tasks(p, parent)` registers STAGE-1: it builds the SEALS7 carbon-density lookup
(LULC class x carbon zone -> tC/ha). It is the reproducible BUILDER of the lookup, used by
run_carbon.py (standalone) and for regeneration:

    from global_invest.carbon import carbon_initialize
    carbon_initialize.initialize_paths(p)
    carbon_initialize.add_carbon_tasks(p)                     # or parent=p.es_task to nest it

Consumers do NOT need this hook: nff_global / ngfs_pnas read the prebuilt lookup at
p.carbon_density_lookup_table_path and call carbon_functions.generate_carbon_density_raster to
apply it per scenario map, then do their own downstream step (nff keeps the density raster;
ngfs does r50 -> frs shock). Follows the per-service structure
(run_<svc> / <svc>_tasks / <svc>_functions / <svc>_initialize).
"""
from global_invest.carbon import carbon_tasks


def initialize_paths(p):
    """Set the carbon input paths on p, resolved against p.base_data_dir via
    p.get_path (path components relative to base_data; downloads on demand). The
    caller sets p.base_data_dir before calling."""
    # SEALS7 base map (NOT ESA): the lookup must be keyed on SEALS7 classes so it matches
    # the downscaled maps consumers apply it to. Building from lulc_esa_2019 would key the
    # table on ESA classes, which the SEALS7 consumer maps cannot index.
    p.base_year_lulc_path = p.get_path('carbon_storage', 'lulc_seals7_2020_from_esa.tif')
    p.carbon_zones_path = p.get_path('carbon_storage', 'carbon_zones_rasterized.tif')
    # Canonical lookup: Stage-1 builds it; every consumer (nff/ngfs) reads it from here.
    p.carbon_density_lookup_table_path = p.get_path('carbon_storage', 'carbon_density_lookup_seals7_spawn.csv')
    p.region_boundary_path = p.get_path('cartographic', 'ee', 'ee_r264_correspondence.gpkg')
    return p


def add_carbon_tasks(p, parent=None):
    """Attach the carbon STAGE-1 chain to p's task tree (under `parent` if given), return p.

    Stage 1 (build the lookup, once): convert Spawn dtype -> combine AGB+BGB -> reproject to
    the LULC grid -> compute the (SEALS7 class x carbon zone) -> tC/ha lookup table.

    Stage 2 (apply the lookup to a map) is deliberately NOT here — each consumer owns it, so
    global_invest.carbon stays free of consumer-specific logic (e.g. GTAP r50/shock). Consumers
    read p.carbon_density_lookup_table_path and call the public
    carbon_functions.generate_carbon_density_raster(lulc, zones, lookup, out); run_carbon.py adds
    the base-year Stage-2 for standalone use.
    """
    p.task_convert_carbon_density_maps_dtype = p.add_task(
        carbon_tasks.task_convert_carbon_density_maps_dtype, parent=parent)
    p.task_combine_two_carbon_density_maps = p.add_task(
        carbon_tasks.task_combine_two_carbon_density_maps, parent=parent)
    p.task_reproject_total_carbon_density = p.add_task(
        carbon_tasks.task_reproject_total_carbon_density, parent=parent)
    p.task_compute_carbon_density_table = p.add_task(
        carbon_tasks.task_compute_carbon_density_table, parent=parent)               # -> the SEALS7 lookup CSV
    return p

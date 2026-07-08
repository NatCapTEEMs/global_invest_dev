"""Task-tree builders for the global_invest carbon (storage & sequestration) model.

Import this to attach the carbon tasks to any ProjectFlow tree:

    from global_invest.carbon import carbon_initialize
    carbon_initialize.add_carbon_tasks(p)                     # stand-alone (run_carbon.py)
    carbon_initialize.add_carbon_tasks(p, parent=p.es_task)   # embedded (run_nff_global.py / run_ngfs_pnas.py)

This lifts the task tree out of run_carbon.py so the carbon model is reusable as a
sub-tree in other projects, following the per-service structure
(run_<svc> / <svc>_tasks / <svc>_functions / <svc>_initialize).
"""
from global_invest.carbon import carbon_tasks


def initialize_paths(p):
    """Set the carbon input paths on p, resolved against p.base_data_dir via
    p.get_path (path components relative to base_data; downloads on demand). The
    caller sets p.base_data_dir (and p.aoi) before calling."""
    p.base_year_lulc_path = p.get_path('lulc', 'esa', 'lulc_esa_2019.tif')
    p.all_lulcs_path = p.get_path('lulc', 'esa')
    p.carbon_zones_path = p.get_path('carbon_storage', 'carbon_zones_rasterized.tif')
    p.region_boundary_path = p.get_path('cartographic', 'ee', 'ee_r264_correspondence.gpkg')
    return p


def add_carbon_tasks(p, parent=None):
    """Attach the carbon task chain to p's task tree (under `parent` if given), return p.

    Stage 1 (build once): convert -> combine (AGB + BGB) -> reproject -> density table.
    Stage 2 (apply per map): generate the carbon-density raster, then summarize by region.
    """
    p.task_convert_carbon_density_maps_dtype = p.add_task(
        carbon_tasks.task_convert_carbon_density_maps_dtype, parent=parent)
    p.task_combine_two_carbon_density_maps = p.add_task(
        carbon_tasks.task_combine_two_carbon_density_maps, parent=parent)
    p.task_reproject_total_carbon_density = p.add_task(
        carbon_tasks.task_reproject_total_carbon_density, parent=parent)
    p.task_compute_carbon_density_table = p.add_task(
        carbon_tasks.task_compute_carbon_density_table, parent=parent)               # Stage 1: LULC x zone -> tC/ha lookup
    p.task_generate_carbon_density_raster_base_year = p.add_task(
        carbon_tasks.task_generate_carbon_density_raster_base_year, parent=parent)   # Stage 2: apply lookup to a map
    p.task_summarize_carbon_density_by_region = p.add_task(
        carbon_tasks.task_summarize_carbon_density_by_region, parent=parent)
    return p

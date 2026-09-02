import pandas as pd
import hazelbean as hb


def initialize_paths(p):
    """Initialize shared tabular data, vector paths, and variant defaults."""
    required_attrs = [
        "project_dir",
        "base_data_dir",
        "carbon_zones_path",
        "lulc_folder_path",
        "df_countries_csv_path",
        "gdf_countries_vector_path",
    ]
    missing = [name for name in required_attrs if not getattr(p, name, None)]
    if missing:
        raise ValueError(f"Missing required ProjectFlow attributes: {missing}")

    p.n_lat_bands = getattr(p, "n_lat_bands", 0)
    p.n_workers = getattr(p, "n_workers", None)
    p.use_parallel = getattr(p, "use_parallel", True)
    p.lookup_table_type = "3key" if p.n_lat_bands else "2key"
    p.variant_name = getattr(
        p,
        "variant_name",
        f"3key_{p.n_lat_bands}bands" if p.n_lat_bands else "2key",
    )

    p.df_countries = pd.read_csv(p.df_countries_csv_path)

    # Keep vectors as paths. Tasks read vectors only when geometry is needed.
    p.gdf_countries = p.gdf_countries_vector_path
    p.gdf_countries_simplified = getattr(
        p,
        "gdf_countries_vector_simplified_path",
        p.get_path("cartographic", "ee", "ee_r264_simplified300sec.gpkg"),
    )
    p.results = {}

    hb.log("=" * 60)
    hb.log(f"Configuration: variant={p.variant_name}, n_lat_bands={p.n_lat_bands}")
    hb.log(f"  project_dir: {p.project_dir}")
    hb.log(f"  intermediate_dir: {p.intermediate_dir}")
    hb.log("=" * 60)


def build_preprocessing_task_tree(p):
    """Build shared Spawn preparation and country-zone rasterization tasks."""
    from . import terrestrial_carbon_tasks

    p.task_prepare_spawn_rasters = p.add_task(
        terrestrial_carbon_tasks.task_prepare_spawn_rasters,
    )
    p.task_rasterize_regions = p.add_task(
        terrestrial_carbon_tasks.task_rasterize_regions,
    )
    return p


def build_gep_service_calculation_task_tree(p):
    """Build one variant carbon-density, regional-summary, and GEP DAG."""
    from . import terrestrial_carbon_tasks

    p.task_compute_carbon_density_table = p.add_task(
        terrestrial_carbon_tasks.task_compute_carbon_density_table,
    )
    p.task_generate_carbon_density_raster = p.add_task(
        terrestrial_carbon_tasks.task_generate_carbon_density_raster,
        dependencies=[p.task_compute_carbon_density_table],
    )
    p.task_summarize_carbon_by_region = p.add_task(
        terrestrial_carbon_tasks.task_summarize_carbon_by_region,
        dependencies=[p.task_generate_carbon_density_raster],
    )
    p.task_combine_region_summaries = p.add_task(
        terrestrial_carbon_tasks.task_combine_region_summaries,
        dependencies=[p.task_summarize_carbon_by_region],
    )
    p.task_gep_by_country = p.add_task(
        terrestrial_carbon_tasks.task_gep_by_country,
        dependencies=[p.task_combine_region_summaries],
    )
    return p


def build_gep_service_results_task_tree(p):
    """Build GEP output from existing regional-summary CSV files only."""
    from . import terrestrial_carbon_tasks

    p.task_gep_by_country = p.add_task(
        terrestrial_carbon_tasks.task_gep_by_country,
    )
    return p


def build_gep_service_task_tree(p):
    """Build default terrestrial-carbon GEP service task tree."""
    return build_gep_service_calculation_task_tree(p)


"""landslide_mitigation wiring (folded from m-braaksma/landslide_mitigation v0.2.0, 2026-08-16).

The cleanest fold so far: the source repo already matched the stack conventions (stage-split task
modules + functions + utils, hazelbean/ProjectFlow throughout, ZERO import-time side effects in the
audit). Files arrive intact with imports rewritten to this package; the full task tree is lifted
verbatim from the source run script and exposed under the template name. Data staged at
base_data/landslide_mitigation/input_data_raw (50.5 GiB, see the base_data changelog).

STATUS: folded + import-clean; NOT yet run here and NOT number-verified -- the current reference is
the release-day appendix tables (the older gep xlsx predates the v0.2.0 method change); exact
replication waits on a machine-readable v0.2.0 output from the author. Numbers provisional.
"""
import hazelbean as hb

from global_invest import utilities

from global_invest.landslide_mitigation import landslide_mitigation_tasks


def initialize_paths(p):
    """Landslide inputs via get_path reference paths (base_data/landslide_mitigation/input_data_raw,
    staged from the author's TEEMs-drive migration), plus the source run script's full configuration
    block. The first-run crashes came from porting that block one attribute at a time; it lives here
    whole now, source lines 84-114."""
    # Configuration comes from the two shared CSVs as a defaults layer: es_config carries the
    # raw-data reference (gep_quantity_input_path -> landslide_mitigation/input_data_raw,
    # staged from the author's TEEMs-drive migration); es_parameters carries the run knobs and
    # method constants that used to live here as literals.
    utilities.hydrate_es_config(p, 'landslide_mitigation')
    utilities.hydrate_es_parameters(p, 'landslide_mitigation')
    p.landslide_input_data_dir = p.gep_quantity_input_path   # the module's descriptive alias
    p.L = hb.get_logger('landslide_mitigation_workflow')
    return p


def build_gep_service_task_tree(p):
    """Full landslide chain (source: the v0.2.0 run script tree, verbatim): input data ->
    preprocessing -> stability model -> valuation -> tables/figures."""
    # ---------------------------------------------------------------- #
    # INPUT DATA (landslide_mitigation_tasks.py)
    # ---------------------------------------------------------------- #
    p.input_data_task = p.add_task(landslide_mitigation_tasks.input_data, creates_dir=True)
    p.build_ease_grid_reference_task = p.add_task(landslide_mitigation_tasks.build_ease_grid_reference, creates_dir=False)
    p.dem_task = p.add_task(landslide_mitigation_tasks.reproject_dem, creates_dir=False)
    p.gaez_task = p.add_task(landslide_mitigation_tasks.reproject_gaez, creates_dir=False)
    p.esacci_forest_share_task = p.add_task(landslide_mitigation_tasks.reproject_esacci_forest_share, creates_dir=False)
    p.uglc_events_task = p.add_task(landslide_mitigation_tasks.reproject_uglc_events, creates_dir=False)
    p.landscan_task = p.add_task(landslide_mitigation_tasks.reproject_landscan_population, creates_dir=False)
    p.soilgrids_properties_task = p.add_task(landslide_mitigation_tasks.reproject_soilgrids_properties, creates_dir=False)
    p.worldclim_bio12_task = p.add_task(landslide_mitigation_tasks.reproject_worldclim_bio12, creates_dir=False)
    p.hihydrosoil_ksat_task = p.add_task(landslide_mitigation_tasks.reproject_hihydrosoil_ksat, creates_dir=False)
    p.soil_depth_task = p.add_task(landslide_mitigation_tasks.reproject_soil_depth, creates_dir=False)
    p.grip_roads_task = p.add_task(landslide_mitigation_tasks.reproject_grip_roads, creates_dir=False)
    p.rain_daily_task = p.add_task(landslide_mitigation_tasks.reproject_rain_daily, creates_dir=False)
    p.validate_input_rasters_task = p.add_task(landslide_mitigation_tasks.validate_input_rasters, creates_dir=False)

    # ---------------------------------------------------------------- #
    # PREPROCESSING (landslide_mitigation_tasks.py)
    # ---------------------------------------------------------------- #
    p.preprocessing_task = p.add_task(landslide_mitigation_tasks.preprocessing, creates_dir=True)
    p.build_uglc_annual_panels_task = p.add_task(landslide_mitigation_tasks.build_uglc_annual_panels,creates_dir=False)
    p.fill_pits_task = p.add_task(landslide_mitigation_tasks.fill_pits, creates_dir=False)
    p.flow_dir_d8_task = p.add_task(landslide_mitigation_tasks.compute_flow_dir_d8, creates_dir=False)
    p.upslope_area_task = p.add_task(landslide_mitigation_tasks.compute_upslope_area, creates_dir=False)
    p.slope_task = p.add_task(landslide_mitigation_tasks.compute_slope, creates_dir=False)
    p.soil_hydraulic_properties_task = p.add_task(landslide_mitigation_tasks.compute_soil_hydraulic_properties, creates_dir=False)
    p.static_q_task = p.add_task(landslide_mitigation_tasks.compute_static_q, creates_dir=False)
    p.si_scenarios_task = p.add_task(landslide_mitigation_tasks.compute_si_scenarios, creates_dir=False)
    p.build_estimation_table_task = p.add_task(landslide_mitigation_tasks.build_estimation_table, creates_dir=False)

    # ---------------------------------------------------------------- #
    # MODELING (landslide_mitigation_tasks.py)
    # ---------------------------------------------------------------- #
    p.modeling_task = p.add_task(landslide_mitigation_tasks.modeling, creates_dir=True)
    p.calibrate_si_to_probability_task = p.add_task(landslide_mitigation_tasks.calibrate_si_to_probability, creates_dir=False)
    p.estimate_severity_model_task = p.add_task(landslide_mitigation_tasks.estimate_severity_model, creates_dir=False)
    p.estimate_severity_model_si_sensitivity_task = p.add_task(landslide_mitigation_tasks.estimate_severity_model_si_sensitivity, creates_dir=False)

    # ---------------------------------------------------------------- #
    # VALUATION / PREDICTION (landslide_mitigation_tasks.py)
    # ---------------------------------------------------------------- #
    # A. Tile-level
    p.tile_zones_task = p.add_iterator(landslide_mitigation_tasks.tile_zones, run_in_parallel=p.run_in_parallel)
    p.predict_landslides_scenarios_task = p.add_task(landslide_mitigation_tasks.predict_landslides_scenarios, parent=p.tile_zones_task)
    p.predict_mortality_scenarios_task = p.add_task(landslide_mitigation_tasks.predict_mortality_scenarios, parent=p.tile_zones_task)
    p.stitch_tiles_task = p.add_task(landslide_mitigation_tasks.stitch_tiles, creates_dir=True)
    # B. Global-level
    p.valuation_task = p.add_task(landslide_mitigation_tasks.valuation, creates_dir=True)
    p.build_vsl_raster_task = p.add_task(landslide_mitigation_tasks.build_vsl_raster, creates_dir=False)
    p.avoided_mortality_task = p.add_task(landslide_mitigation_tasks.compute_avoided_mortality, creates_dir=False)

    # ---------------------------------------------------------------- #
    # Tables & Figures (landslide_mitigation_tasks.py)
    # ---------------------------------------------------------------- #
    p.tables_figures_task = p.add_task(landslide_mitigation_tasks.tables_figures, creates_dir=True)
    p.zonal_statistics_task = p.add_task(landslide_mitigation_tasks.compute_zonal_statistics, creates_dir=False)
    p.export_regression_tables_task = p.add_task(landslide_mitigation_tasks.export_regression_tables, creates_dir=False)
    p.plot_global_rasters_png_task = p.add_task(landslide_mitigation_tasks.plot_global_rasters_png, creates_dir=False)
    p.plot_country_choropleth_maps_task = p.add_task(landslide_mitigation_tasks.plot_country_choropleth_maps, creates_dir=False)
    p.plot_uglc_fatality_bins_task = p.add_task(landslide_mitigation_tasks.plot_uglc_from_vector, creates_dir=False)
    p.export_results_tables_task = p.add_task(landslide_mitigation_tasks.export_results_tables, creates_dir=False)
    p.export_si_severity_sensitivity_table_task = p.add_task(landslide_mitigation_tasks.export_si_severity_sensitivity_table, creates_dir=False)
    p.export_pi_audit_table_task = p.add_task(landslide_mitigation_tasks.export_pi_audit_table, creates_dir=False)

    return p



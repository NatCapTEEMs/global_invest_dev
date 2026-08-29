import glob
import os
from tqdm import tqdm
from osgeo import gdal

import pandas as pd
import hazelbean as hb
import numpy as np


from global_invest import utilities
from global_invest.terrestrial_carbon import terrestrial_carbon_functions as tcf

SPAWN_INTEGER_SCALE = 0.1  # raw Spawn tiles store carbon as integers x10; x0.1 recovers Mg C/ha



def stack_layers_summary(group_layer1_path, group_layer2_path, value_layer_path,
                         group1_name="group1", group2_name="group2", value_name="value"):
    """A value raster summarized over the pairs of two category rasters.

    Args:
        group_layer1_path (str): first grouping raster.
        group_layer2_path (str): second grouping raster, on the same grid.
        value_layer_path (str): the raster being summarized, on the same grid.
        group1_name (str): column the first grouping raster is written under.
        group2_name (str): column the second grouping raster is written under.
        value_name (str): stem of the four summary columns.

    Returns:
        pd.DataFrame: one row per (group1, group2) pair, carrying the mean, min, max and count
        of the value raster over the cells where all three rasters are valid.
    """
    # hb.iterblocks streams the value raster block by block; the two category rasters share its grid, so
    # read them at the same window (raw gdal -- hb has no aligned multi-raster block reader). A cell is
    # kept only where all three are valid. Groupby per block, then
    # combine: only ~thousands of (g1, g2) pairs, so the accumulator stays tiny and nothing 33 GB is
    # written (unlike composite-key + zonal).
    ds1 = gdal.Open(group_layer1_path); g1b = ds1.GetRasterBand(1)   # hold the datasets, else the band handle dies
    ds2 = gdal.Open(group_layer2_path); g2b = ds2.GetRasterBand(1)
    ndv1, ndv2 = g1b.GetNoDataValue(), g2b.GetNoDataValue()
    dsv = gdal.Open(value_layer_path); ndv_val = dsv.GetRasterBand(1).GetNoDataValue()

    parts = []
    for offset, value_block in tqdm(hb.iterblocks((value_layer_path, 1)), desc="Summarizing blocks"):
        w = (offset['xoff'], offset['yoff'], offset['win_xsize'], offset['win_ysize'])
        v = value_block.astype('float32').ravel()
        g1 = g1b.ReadAsArray(*w).ravel()
        g2 = g2b.ReadAsArray(*w).ravel()

        keep = ~np.isnan(v)
        if ndv_val is not None: keep &= v != ndv_val
        if ndv1 is not None: keep &= g1 != ndv1
        if ndv2 is not None: keep &= g2 != ndv2
        if not keep.any():   # block is all-nodata -- expected at raster edges
            continue

        block = pd.DataFrame({group1_name: g1[keep], group2_name: g2[keep], value_name: v[keep]})
        parts.append(block.groupby([group1_name, group2_name], as_index=False)[value_name]
                     .agg(_sum='sum', _min='min', _max='max', _count='count'))

    summary = pd.concat(parts).groupby([group1_name, group2_name], as_index=False).agg(
        _sum=('_sum', 'sum'), _min=('_min', 'min'), _max=('_max', 'max'), _count=('_count', 'sum'))
    summary[f'{value_name}_mean'] = summary['_sum'] / summary['_count']
    summary = summary.rename(columns={'_min': f'{value_name}_min', '_max': f'{value_name}_max',
                                      '_count': f'{value_name}_count'})
    return summary[[group1_name, group2_name, f'{value_name}_mean', f'{value_name}_min',
                    f'{value_name}_max', f'{value_name}_count']]


def publish_inputs(p):
    """Every config-consuming task's first line: this service's es_config row (defaults layer --
    anything the caller set wins) plus the shared country references. Cheap and idempotent, so a
    task stays a working piece on its own: graft it anywhere, no setup call to remember. The
    carbon-zones reference (gep_quantity_input_path) is the SAME raster the shock task uses, so
    the GEP valuation and the shock can never diverge."""
    utilities.hydrate_es_config(p, 'terrestrial_carbon', log=hb.log)
    utilities.hydrate_es_parameters(p, 'terrestrial_carbon', log=hb.log)
    utilities.initialize_country_paths(p)
    if not hasattr(p, 'results'):
        p.results = {}   # every gep task registers its outputs here; the report renders from it
    return p


def _raster_shape(raster_path):
    return tuple(hb.get_raster_info_hb(raster_path)['raster_size'])


def total_carbon_density(p):
    """
    Task to reproject the total carbon density raster to the project's coordinate reference system (CRS).
    """
    publish_inputs(p)

    # Input: the total biomass-carbon density raster, a one-off base-data product; its reference
    # is the total_carbon_density_path row in es_parameters (published by publish_inputs).
    p.reprojected_total_carbon_density_path = os.path.join(p.cur_dir, "total_biomass_carbon_2010_float_reprojected.tif")
    if not p.run_this:
        return True

    # Align the density raster to the base-year LULC grid. Float32 keeps the carbon values (the
    # LULC grid is uint8 and would round them); 'near' + NaN nodata preserve them exactly.
    hb.resample_to_match(p.total_carbon_density_path, p.gep_lulc_input_path,
                         p.reprojected_total_carbon_density_path,
                         resample_method='near', output_data_type=tcf.GDAL_FLOAT32,
                         src_ndv=np.nan, ndv=np.nan)
    return True


def carbon_density_table(p):
    publish_inputs(p)
    p.carbon_density_lookup_table_path = os.path.join(p.cur_dir, "carbon_density_lookup_table.csv")
    if not p.run_this:
        return True

    summary = stack_layers_summary(
        group_layer1_path=p.gep_lulc_input_path,
        group_layer2_path=p.gep_quantity_input_path,
        value_layer_path=p.reprojected_total_carbon_density_path,
        group1_name="lulc_id",
        group2_name="carbon_zone_id",
        value_name="carbon_density")
    summary.to_csv(p.carbon_density_lookup_table_path, index=False)
    return True


def carbon_density_raster_base_year(p):
    publish_inputs(p)
    p.carbon_density_raster_base_year_path = os.path.join(p.cur_dir, f"projected_carbon_density_{p.gep_base_year}.tif")
    if not p.run_this:
        return True
    tcf.generate_carbon_density_raster(
        lulc_path=p.gep_lulc_input_path,
        cz_path=p.gep_quantity_input_path,
        density_lookup=tcf.carbon_density_lookup(pd.read_csv(p.carbon_density_lookup_table_path, index_col=False)),
        out_path=p.carbon_density_raster_base_year_path)
    return True


def carbon_density_raster_per_cell_base_year(p):
    publish_inputs(p)
    utilities.initialize_pyramid_paths(p)
    p.carbon_density_per_cell_base_year_path = os.path.join(p.cur_dir, f'projected_carbon_density_{p.gep_base_year}_per_cell.tif')
    if not p.run_this:
        return True
    hb.multiply(p.carbon_density_raster_base_year_path, p.ha_per_cell_10sec_path, p.carbon_density_per_cell_base_year_path)
    return True


def carbon_by_region(p):
    publish_inputs(p)
    p.carbon_by_region_base_year_path = os.path.join(p.cur_dir, "gep_by_country_base_year.csv")
    if not p.run_this:
        return True
    result = tcf.summarize_raster_by_region(
        value_raster_path=p.carbon_density_per_cell_base_year_path,
        region_boundary_path=p.gep_regions_input_path,
        out_path=p.carbon_by_region_base_year_path,
        year=p.gep_base_year, id_column=p.gep_regions_id_col)
    return result


def gep_preprocess(p):
    """Rebuild the base-data carbon-density raster (global_invest/terrestrial_carbon/
    spawn_total_biomass_carbon_2010.tif) that both the GEP valuation and the shock consume. A
    one-off base-data job: registered only in build_gep_service_preprocess_task_tree, NOT in the
    default run; its product is promoted to that base_data ref and read from there per run.

    total = aboveground + belowground (Mg C/ha). When the raw Spawn tiles (uint, carbon stored x10) are
    present under base_data/terrestrial_carbon/spawn_2020, first scale them to Mg C/ha and reproject onto
    the base-year LULC grid to (re)make the *_projected rasters; otherwise start from the projected
    rasters already in carbon_storage. All raster ops go through hazelbean.
    """
    publish_inputs(p)
    # Generator and consumers share ONE ref: the total_carbon_density_path row (canonical home
    # global_invest/terrestrial_carbon/). Resolved to the existing copy where one exists, else
    # formed under cur_dir for generation; promotion = copying to the same ref in base_data.
    p.spawn_total_carbon_density_path = p.total_carbon_density_path
    if not p.run_this:
        return True

    product_dir = os.path.dirname(p.spawn_total_carbon_density_path)
    spawn_raw_dir = p.spawn_raw_input_path
    projected = {}
    for band in ('aboveground', 'belowground'):
        projected[band] = os.path.join(product_dir, 'spawn_%s_biomass_carbon_2010_projected.tif' % band)
        raw = os.path.join(spawn_raw_dir, 'spawn_%s_biomass_carbon_2010.tif' % band)
        if hb.path_exists(raw):   # raw Spawn present -> (re)build the projected raster from it
            scaled = os.path.join(p.cur_dir, 'spawn_%s_biomass_carbon_2010_scaled.tif' % band)
            hb.raster_calculator_flex(raw, lambda a: a * SPAWN_INTEGER_SCALE, scaled)
            hb.reproject_dataset_to_match(scaled, p.gep_lulc_input_path, projected[band], 'near')

    hb.raster_calculator_flex([projected['aboveground'], projected['belowground']],
                              lambda a, b: a + b, p.spawn_total_carbon_density_path)
    return True


def gep_calculation(p):
    """GEP valuation for terrestrial carbon: the r264 carbon quantity priced at the base-year
    carbon price and collapsed to one row per country."""
    publish_inputs(p)
    # Register what this task writes; the report renders from p.results. Only results this task
    # actually writes: per-year results belong to a multi-year run and are registered there.
    service_results, already_done = utilities.begin_gep_calculation(p, 'terrestrial_carbon')
    if already_done:
        return

    df_regions = hb.df_read(p.carbon_by_region_base_year_path)
    df_price = pd.read_excel(p.gep_price_input_path)[[p.gep_price_convention, 'year']]
    df_gep = tcf.collapse_regions_to_countries(df_regions, df_price, p.gep_price_convention)
    hb.df_write(df_gep, service_results['gep_by_country_base_year'])

    # Map only: r264-expanded, each sub-region carries its country's value, never summed.
    gdf = hb.df_merge(p.gdf_countries_simplified,
                      tcf.expand_country_values_to_regions(df_regions, df_gep),
                      how='outer', left_on='ee_r264_id', right_on='ee_r264_id')
    gdf.to_file(service_results['gep_by_country_base_year'].replace('.csv', '.gpkg'), driver='GPKG')

    value_gep_base_year = df_gep['terrestrial_carbon_gep'].sum()
    hb.log(f"Total GEP value for base year {p.gep_base_year}: {value_gep_base_year}")
    return value_gep_base_year


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)

def gep_load_results(p):
    """Load the GEP results computed by a PRIOR calculation run, so the report can render without
    recomputing. Fails loudly if they are not present -- run the calculation (run_terrestrial_carbon.py)
    or promote the results into base_data first. This is the 'results-only' entry point.
    """
    publish_inputs(p)
    result_path = os.path.join(p.intermediate_dir, 'gep_calculation', 'gep_by_country_base_year.csv')
    if not hb.path_exists(result_path):
        raise FileNotFoundError(
            f"terrestrial_carbon GEP results not found at {result_path}. "
            f"Run the calculation first (run_terrestrial_carbon.py), then re-run results.")
    p.results.setdefault('terrestrial_carbon', {})
    p.results['terrestrial_carbon']['gep_by_country_base_year'] = result_path

# =============================================================================
# ES-shock tasks. These feed the GTAP shock; the GEP valuation above is a separate
# consumer of the same carbon-density front-end. Neither depends on the other.
# =============================================================================

def _zone_mean(p, scenario, year, density_lookup):
    """Mean carbon density per boundary polygon for one scenario map year.

    Both stages are cached in the task dir so a partial re-run picks up where it stopped. The
    polygon geometry is identical across scenarios, so the mean is sufficient and area cancels.
    """
    density_path = os.path.join(p.cur_dir, 'carbon_density_%s_%d.tif' % (scenario, year))
    if not hb.path_exists(density_path):
        tcf.generate_carbon_density_raster(
            lulc_path=p.scenario_lulc_paths[scenario][year],
            cz_path=p.terrestrial_quantity_input_path,
            density_lookup=density_lookup,
            out_path=density_path)
    summary_path = os.path.join(p.cur_dir, 'carbon_by_zone_%s_%d.csv' % (scenario, year))
    if not hb.path_exists(summary_path):
        tcf.summarize_raster_by_region(density_path, p.region_boundary_path, summary_path,
                                       year=year, id_column=p.terrestrial_carbon_shock_id_col)
    return hb.df_read(summary_path).set_index('region_id')[
        getattr(p, 'terrestrial_carbon_shock_value_col', 'mean')]


def _align_zones_to_lulc_grid(p, reference_lulc_path):
    """Point p.terrestrial_quantity_input_path at a carbon-zones raster on the reference map's grid.

    generate_carbon_density_raster asserts LULC and carbon zones share a grid. The zones raster is
    global 300 m and a SEALS LULC map may be a sub-window (single-country or short-horizon test
    AOI). Same resolution and an aligned grid, so nearest is a lossless clip and the task works at
    any extent. The clip is cached: the zones raster does not change between runs.
    """
    if _raster_shape(p.terrestrial_quantity_input_path) == _raster_shape(reference_lulc_path):
        return
    aligned_path = os.path.join(p.cur_dir, 'carbon_zones_aligned.tif')
    if not hb.path_exists(aligned_path):
        hb.resample_to_match(p.terrestrial_quantity_input_path, reference_lulc_path,
                             aligned_path, resample_method='near')
    p.terrestrial_quantity_input_path = aligned_path


def _inject_base_year_map(p, base_scenario, base_year, reference_lulc_path):
    """Register the ES-shared SEALS7 base-year map as the baseline's base-year map, so the
    fixed-base denominator can be measured on it.

    NEVER p.base_year_lulc_path: SEALS OWNS that attribute and overwrites it at runtime with its
    raw-ESA source, and the density lookup is keyed on SEALS7 classes, so a raw-ESA base map yields
    all-NaN densities. A carbon-specific terrestrial_carbon_base_year_lulc_path overrides if a
    caller sets one. The aligned copy is rebuilt every run because the map it comes from can change
    while the aligned filename does not.
    """
    base_map = (getattr(p, 'terrestrial_carbon_base_year_lulc_path', None)
                or getattr(p, 'es_base_year_lulc_path', None))
    if not base_map or base_year in p.scenario_lulc_paths.get(base_scenario, {}):
        return
    if not os.path.isabs(base_map):
        base_map = p.get_path(base_map)
    if _raster_shape(base_map) != _raster_shape(reference_lulc_path):
        aligned_path = os.path.join(p.cur_dir, 'lulc_base_year_aligned.tif')
        hb.resample_to_match(base_map, reference_lulc_path, aligned_path, resample_method='near')
        base_map = aligned_path
    p.scenario_lulc_paths.setdefault(base_scenario, {})[base_year] = base_map


def terrestrial_carbon_shock(p):
    """Turn per-scenario 300 m LULC into a carbon ES-productivity shock -- region-agnostic.

    At each SEALS anchor year in es_shock_years (5-year MAgPIE steps), measure the mean carbon
    density per polygon of p.region_boundary_path for the baseline scenario and each scenario, and
    hand those means to terrestrial_carbon_functions.dynamic_shock_rows, which reports both the
    contemporaneous measure the GTAP shock reads (`shock_pct`) and the fixed-base measure
    (`shock_pct_fixedbase`). The only region-specific knowledge is the boundary's column names,
    supplied by the caller via p, so nothing GTAP-specific is hardcoded.

    Caller sets on p: es_shock_years (SEALS anchor years, from seals_years),
    scenario_lulc_paths {scenario: {year: path}} or es_lulc_path_template, es_shock_scenarios,
    region_boundary_path, terrestrial_quantity_input_path, terrestrial_carbon_density_lookup_table_path,
    terrestrial_carbon_shock_output_path. Optional: terrestrial_carbon_shock_{base_scenario, base_year,
    endw_col, reg_col, value_col, acts}.
    """
    # Default into the es_shocks parent dir. Runtime, not build time: p.es_shock_dir is
    # published by that task, which ProjectFlow runs before this one.
    if not getattr(p, 'terrestrial_carbon_shock_output_path', None):
        p.terrestrial_carbon_shock_output_path = os.path.join(getattr(p, 'es_shock_dir', None) or p.project_dir, 'terrestrial_carbon_interpolated.csv')
    if not p.run_this:
        return
    import geopandas as gpd

    # The export keys and boundary are SHIPPED DEFAULTS in es_parameters (GTAP r50xAEZ18 --
    # today's consumer family), hydrated as a defaults layer: a consumer's own tables win.
    utilities.hydrate_es_parameters(p, 'terrestrial_carbon', log=hb.log)
    base_scenario      = utilities.required_base_scenario(p, 'terrestrial_carbon')   # validated vs the caller's naming
    es_shock_base_year = int(p.es_shock_base_year)                # interp 0-anchor
    anchor_years = sorted(y for y in map(int, p.es_shock_years) if y > es_shock_base_year)  # SEALS anchors

    # Resolve the LULC map per scenario by globbing es_lulc_path_template ({scenario}/{year}
    # placeholders) when the caller didn't pre-build scenario_lulc_paths, so a project passes only
    # a template string rather than a path-building task.
    scenarios = list(getattr(p, 'es_shock_scenarios', []))
    if not getattr(p, 'scenario_lulc_paths', None):
        tmpl = p.es_lulc_path_template
        p.scenario_lulc_paths = {
            scen: {y: glob.glob(tmpl.format(scenario=scen, year=y))[0]
                   for y in anchor_years if glob.glob(tmpl.format(scenario=scen, year=y))}
            for scen in [base_scenario] + scenarios}
    if not scenarios:
        scenarios = [s for s in p.scenario_lulc_paths if s != base_scenario]

    reference_lulc_path = p.scenario_lulc_paths[base_scenario][anchor_years[-1]]
    _align_zones_to_lulc_grid(p, reference_lulc_path)
    _inject_base_year_map(p, base_scenario, es_shock_base_year, reference_lulc_path)

    density_lookup = tcf.carbon_density_lookup(
        pd.read_csv(p.terrestrial_carbon_density_lookup_table_path, index_col=False))
    zone_labels = tcf.zone_labels_from_boundary(
        gpd.read_file(p.region_boundary_path, engine='pyogrio'),
        p.terrestrial_carbon_shock_id_col, p.terrestrial_carbon_shock_endw_col,
        p.terrestrial_carbon_shock_reg_col, p.terrestrial_carbon_shock_endw_format)

    baseline_by_year = {y: _zone_mean(p, base_scenario, y, density_lookup) for y in anchor_years}
    # Only measure the fixed-base level if the base year is actually available; without it
    # shock_pct_fixedbase degrades to NaN and shock_pct is untouched.
    baseline_at_base_year = (_zone_mean(p, base_scenario, es_shock_base_year, density_lookup)
                             if es_shock_base_year in p.scenario_lulc_paths.get(base_scenario, {}) else None)

    rows = []
    for scenario in scenarios:
        rows += tcf.dynamic_shock_rows(
            {y: _zone_mean(p, scenario, y, density_lookup) for y in anchor_years},
            baseline_by_year, baseline_at_base_year, zone_labels, es_shock_base_year,
            p.terrestrial_carbon_shock_acts, scenario)

    out = pd.DataFrame(rows)
    utilities.assert_shock_table_sound(out, scenarios, 'terrestrial_carbon')
    out.to_csv(p.terrestrial_carbon_shock_output_path, index=False)
    print('  carbon shock: %d rows, %d scenarios (shock_pct=shock_pct_contemp=/base_Y, shock_pct_fixedbase=/base_%d) -> %s'
          % (len(out), out['scenario'].nunique() if rows else 0, es_shock_base_year, p.terrestrial_carbon_shock_output_path))
    return True


def terrestrial_carbon_shock_static(p):
    """Static per-scenario carbon shock -> FRS, linear ramp 0->end_year, from the frozen dependency table.

    add_terrestrial_carbon_tasks grafts this (instead of the dynamic recompute) when 'terrestrial_carbon' is NOT
    in p.dynamic_es. READS input_dir/raw_dependencies/carbon_storage_dependency.csv
    (override p.terrestrial_carbon_dependency_path) and subtracts the p.es_shock_base_scenario row at the end year
    (percentage_change x100), ramping that difference linearly from 0 at base_year. NEVER writes back to
    raw_dependencies -- the output goes to p.terrestrial_carbon_shock_output_path (terrestrial_carbon_interpolated.csv),
    the same file the dynamic task writes, so build_combined_afeall_cc_es is agnostic to which one ran.
    Caller sets: es_shock_base_year, es_shock_end_year, es_shock_scenarios,
    terrestrial_carbon_shock_output_path; scenario->raw name via p.terrestrial_carbon_scenario_map
    (default: identity -- each scenario maps to its own name; a scenario the table labels differently is
    warned about loudly and skipped rather than silently zeroed, so set the map for those);
    sector via p.terrestrial_carbon_shock_acts (default 'FRS', matching the dynamic task).
    """
    # Default into the es_shocks parent dir. Runtime, not build time: p.es_shock_dir is
    # published by that task, which ProjectFlow runs before this one.
    if not getattr(p, 'terrestrial_carbon_shock_output_path', None):
        p.terrestrial_carbon_shock_output_path = os.path.join(getattr(p, 'es_shock_dir', None) or p.project_dir, 'terrestrial_carbon_interpolated.csv')
    if not p.run_this:
        return
    utilities.hydrate_es_parameters(p, 'terrestrial_carbon', log=hb.log)   # shipped defaults; caller wins
    es_shock_base_year = int(p.es_shock_base_year)
    es_shock_end_year = int(p.es_shock_end_year)
    terrestrial_carbon_scenario_map = getattr(p, 'terrestrial_carbon_scenario_map', {})
    es_shock_scenarios = list(p.es_shock_scenarios)
    base_scenario = utilities.required_base_scenario(p, 'terrestrial_carbon')  # validated vs the caller's naming

    carb_path = getattr(p, 'terrestrial_carbon_dependency_path', None) or os.path.join(
        p.input_dir, 'raw_dependencies', 'carbon_storage_dependency.csv')
    if not hb.path_exists(carb_path):
        print('  carbon shock: dependency csv not found (%s) -- skipping' % carb_path)
        return

    df = hb.df_read(carb_path)
    # The base resolves through the same candidate mechanism as the data scenarios (and FATALLY if it
    # can't): the frozen tables spell the nature-off baseline two ways across services, and an
    # exact-match miss here gave an empty base -> empty output -> silent GTAP zero.
    raw_base = utilities.resolve_base_scenario(df['scenario'].values, terrestrial_carbon_scenario_map, base_scenario, 'terrestrial_carbon', log=hb.log)
    base = df[(df['scenario'] == raw_base) & (df['year'] == es_shock_end_year)]
    base_vals = base.set_index(['ENDW', 'REG'])['percentage_change'].astype(float) * tcf.PERCENT

    rows = []
    for our_scn in es_shock_scenarios:
        raw_scn = utilities.resolve_raw_scenario(df['scenario'].values, terrestrial_carbon_scenario_map, our_scn, 'terrestrial_carbon', log=hb.log)
        if raw_scn is None:
            continue
        scn = df[(df['scenario'] == raw_scn) & (df['year'] == es_shock_end_year)]
        scn_vals = scn.set_index(['ENDW', 'REG'])['percentage_change'].astype(float) * tcf.PERCENT
        rows += tcf.static_shock_rows(base_vals, scn_vals, our_scn, p.terrestrial_carbon_shock_acts,
                                      es_shock_base_year, es_shock_end_year)

    out = pd.DataFrame(rows)
    utilities.assert_shock_table_sound(out, es_shock_scenarios, 'terrestrial_carbon')
    out.to_csv(p.terrestrial_carbon_shock_output_path, index=False)
    nz = out[(out['year'] == es_shock_end_year) & (out['shock_pct'] != 0)] if len(out) else out
    print('  carbon shock: %d rows, %d scenarios, %d nonzero @%d (static, uncapped) -> %s'
          % (len(out), out['scenario'].nunique() if len(out) else 0, len(nz), es_shock_end_year,
             p.terrestrial_carbon_shock_output_path))
    return True

import os

import pandas as pd
import rasterio
import hazelbean as hb
import numpy as np


from global_invest import utilities
from global_invest.terrestrial_carbon import terrestrial_carbon_functions as tcf

SPAWN_INTEGER_SCALE = 0.1  # raw Spawn tiles store carbon as integers x10; x0.1 recovers Mg C/ha
# Carbon zone 0 is the zone map's fill, the cells outside every carbon zone (an id is >= 1, 0 is no data).
# The authors' code leaves it out of the lookup and the map (c0c86ba and ef89219, functions lines 594, 684).
CARBON_ZONE_NDV = 0



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
    """The density lookup: mean Spawn biomass carbon per land-cover class and carbon zone.

    The classes come from the 2010 land cover (carbon_density_lookup_lulc_path), the year of the
    Spawn map, so biomass and class describe the same year. carbon_density_raster_base_year then
    applies the lookup to the base-year land cover (gep_lulc_input_path).

    The parameter carries no terrestrial_carbon_ prefix, like total_carbon_density_path, so it
    stays out of the carbon-shock reuse signature: the shock reads its own lookup table, not this one.

    Carbon zone 0 has no key. The lookup is applied by reindex, so a zone-0 cell gets no density
    (missing, never 0), and every other key is untouched, since zone-0 cells feed only zone-0 keys.

    Raises:
        ValueError: from utilities.assert_same_grid, if the lookup land cover and the base-year land
            cover the Spawn raster was aligned to differ in CRS (compared in full, parameters
            included), shape or transform, since the cells would then not pair.
    """
    publish_inputs(p)
    p.carbon_density_lookup_table_path = os.path.join(p.cur_dir, "carbon_density_lookup_table.csv")
    if not p.run_this:
        return True

    with rasterio.open(p.carbon_density_lookup_lulc_path) as lookup_lulc, rasterio.open(p.gep_lulc_input_path) as base_lulc:
        # rtol=0: a relative tolerance scales with the coordinates, and at an origin of -180 the
        # default 1e-6 lets a 0.036-cell shift of the 300 m grid through. The 1e-9 absolute allowance stays.
        utilities.assert_same_grid(lookup_lulc, base_lulc, label_a='lookup land cover ' + p.carbon_density_lookup_lulc_path,
                                   label_b='base-year land cover ' + p.gep_lulc_input_path, rtol=0)
    summary = utilities.stack_layers_summary(
        group_layer1_path=p.carbon_density_lookup_lulc_path,
        group_layer2_path=p.gep_quantity_input_path,
        value_layer_path=p.reprojected_total_carbon_density_path,
        group1_name="lulc_id",
        group2_name="carbon_zone_id",
        value_name="carbon_density")
    summary = summary[summary['carbon_zone_id'] != CARBON_ZONE_NDV]
    summary.to_csv(p.carbon_density_lookup_table_path, index=False)
    return True


def carbon_density_raster_base_year(p):
    publish_inputs(p)
    p.carbon_density_raster_base_year_path = os.path.join(p.cur_dir, f"projected_carbon_density_{p.gep_base_year}.tif")
    # Checked before the skip: skip_existing keys on the folder alone, so a project built before zone 0
    # was left out would otherwise reuse its lookup and the four outputs built from it without a word.
    if hb.path_exists(p.carbon_density_lookup_table_path):
        lookup_zones = pd.read_csv(p.carbon_density_lookup_table_path, usecols=['carbon_zone_id'])['carbon_zone_id']
        if (lookup_zones == CARBON_ZONE_NDV).any():
            raise ValueError(
                f"{p.carbon_density_lookup_table_path} holds carbon zone {CARBON_ZONE_NDV}, which the lookup now "
                f"leaves out: it predates that rule. Move carbon_density_table, carbon_density_raster_base_year, "
                f"carbon_density_raster_per_cell_base_year, carbon_by_region and gep_calculation aside together, "
                f"then rerun.")
    if not p.run_this:
        return True
    tcf.generate_carbon_density_raster(
        lulc_path=p.gep_lulc_input_path,
        cz_path=p.gep_quantity_input_path,
        density_lookup=tcf.carbon_density_lookup(pd.read_csv(p.carbon_density_lookup_table_path, index_col=False)),
        out_path=p.carbon_density_raster_base_year_path)
    return True


def carbon_density_raster_per_cell_base_year(p):
    """Carbon per cell in the base year: the density map times the cell's own hectares.

    This is the service's MAP, and it is registered as a result here, where it is written, rather
    than in gep_calculation, which does not write it. Registering is what makes condition 16 reach
    it: `distribute_results` converts every raster it copies into a POG, and it copies what the
    results registry names and nothing else. An unregistered map never leaves `intermediate/`, so
    nobody outside a run can open it and the condition has nothing to check.
    """
    publish_inputs(p)
    utilities.initialize_pyramid_paths(p)
    p.carbon_density_per_cell_base_year_path = utilities.register_result(
        p, 'terrestrial_carbon', f'terrestrial_carbon_per_cell_{p.gep_base_year}.tif',
        os.path.join(p.cur_dir, f'projected_carbon_density_{p.gep_base_year}_per_cell.tif'))
    if not p.run_this:
        return True
    hb.multiply(p.carbon_density_raster_base_year_path, p.ha_per_cell_10sec_path, p.carbon_density_per_cell_base_year_path)
    return True


def carbon_by_region(p):
    publish_inputs(p)
    p.carbon_by_region_base_year_path = os.path.join(p.cur_dir, "gep_by_country_base_year.csv")
    if not p.run_this:
        return True
    result = utilities.summarize_raster_by_region(
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

    if not p.run_this:
        return p
    df_regions = hb.df_read(p.carbon_by_region_base_year_path)
    df_price = pd.read_excel(p.gep_price_input_path)[[p.gep_price_convention, 'year']]
    df_gep = tcf.collapse_regions_to_countries(df_regions, df_price, p.gep_price_convention)
    # Every country keeps a row. One whose regions hold no valid cell (all of it in carbon zone 0, say)
    # has no region row, so it is added here with quantity and value blank: missing, never 0. A
    # measured zero is a region row with total 0 and stays 0. The valued rows are not touched.
    countries = utilities.collapse_countries_to_r250(p.df_countries)[tcf.CARBON_ATTR_COLS]
    absent = countries[~countries['iso3_r250_id'].isin(df_gep['iso3_r250_id'])].assign(
        year=df_gep['year'].iloc[0], **{p.gep_price_convention: df_gep[p.gep_price_convention].iloc[0]})
    df_gep = (pd.concat([df_gep, absent], ignore_index=True)[df_gep.columns]
              .sort_values('iso3_r250_id', kind='stable').reset_index(drop=True))
    hb.df_write(df_gep, service_results['gep_by_country_base_year'])

    # Map only: r264-expanded, each sub-region carries its country's value, never summed.
    gdf = hb.df_merge(p.gdf_countries_simplified,
                      utilities.expand_country_values_to_regions(
                          df_regions, df_gep, 'terrestrial_carbon_gep'),
                      how='outer', left_on='ee_r264_id', right_on='ee_r264_id')
    gdf.to_file(service_results['gep_by_country_base_year'].replace('.csv', '.gpkg'), driver='GPKG')

    value_gep_base_year = df_gep['terrestrial_carbon_gep'].sum()
    hb.log(f"Total GEP value for base year {p.gep_base_year}: {value_gep_base_year}")
    return value_gep_base_year


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    if not p.run_this:
        return p
    utilities.render_service_results(p)

def gep_load_results(p):
    """Load the GEP results computed by a PRIOR calculation run, so the report can render without
    recomputing. Fails loudly if they are not present -- run the calculation (run_terrestrial_carbon.py)
    or promote the results into base_data first. This is the 'results-only' entry point.
    """
    publish_inputs(p)
    result_path = os.path.join(p.intermediate_dir, 'gep_calculation', 'gep_by_country_base_year.csv')
    if not p.run_this:
        return p
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
        utilities.summarize_raster_by_region(density_path, p.region_boundary_path, summary_path,
                                       year=year, id_column=p.terrestrial_carbon_shock_id_col)
    return hb.df_read(summary_path).set_index('region_id')[
        p.terrestrial_carbon_shock_value_col]



def terrestrial_carbon_shock(p):
    """Turn per-scenario 300 m LULC into a carbon ES-productivity shock -- region-agnostic.

    At each SEALS anchor year in es_shock_years (5-year MAgPIE steps), measure the mean carbon
    density per polygon of p.region_boundary_path for the baseline scenario and each scenario, and
    hand those means to utilities.dynamic_shock_rows, which reports both the
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
    scenarios = utilities.resolve_scenario_lulc_paths(p, base_scenario, anchor_years)

    reference_lulc_path = p.scenario_lulc_paths[base_scenario][anchor_years[-1]]
    p.terrestrial_quantity_input_path = utilities.align_zones_to_lulc_grid(
        p.terrestrial_quantity_input_path, reference_lulc_path, os.path.join(p.cur_dir, 'carbon_zones_aligned.tif'))
    utilities.inject_base_year_map(p, base_scenario, es_shock_base_year, reference_lulc_path,
                                   override_path=getattr(p, 'terrestrial_carbon_base_year_lulc_path', None))

    # Reuse the table when it was made from these very maps and settings. The zonal means below read
    # every scenario-year map, so a pass that finds the table must return in seconds -- and must not
    # rewrite it, which is work repeated on resume and defeats an unchanged-resume check.
    #
    # This sits AFTER the two calls above, and must stay there. inject_base_year_map publishes
    # scenario_lulc_paths[base][base_year], which erosion_damage_pipeline reads later; returning
    # before it leaves that entry unset and the erosion damage seam fails on a missing configured
    # input. A reuse shortcut must skip the work, never the path publication. Both calls are
    # themselves cached, so this costs nothing.
    shock_maps = sorted({m for by_year in p.scenario_lulc_paths.values() for m in by_year.values()})
    reason, signature, signature_path = _reuse_check(
        p, [p.terrestrial_carbon_shock_output_path], TERRESTRIAL_CARBON_SHOCK_SIGNATURE,
        extra_settings=dict(scenarios=list(scenarios), anchors=list(anchor_years),
                            base_year=es_shock_base_year, base_scenario=base_scenario,
                            mode='dynamic'),
        extra_inputs=[p.region_boundary_path], light_inputs=shock_maps)
    if reason is None:
        hb.log('  carbon shock: reusing %s (same maps, same settings)' % p.terrestrial_carbon_shock_output_path)
        return True
    hb.log('  carbon shock: computing because %s' % reason)

    density_lookup = tcf.carbon_density_lookup(
        pd.read_csv(p.terrestrial_carbon_density_lookup_table_path, index_col=False))
    zone_labels = utilities.zone_labels_from_boundary(
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
        rows += utilities.dynamic_shock_rows(
            {y: _zone_mean(p, scenario, y, density_lookup) for y in anchor_years},
            baseline_by_year, baseline_at_base_year, zone_labels, es_shock_base_year,
            p.terrestrial_carbon_shock_acts, scenario, log=hb.log)

    out = pd.DataFrame(rows)
    out = utilities.filter_to_model_domain(out, p.terrestrial_carbon_shock_output_path, 'terrestrial_carbon', log=hb.log)
    utilities.assert_shock_table_sound(out, scenarios, 'terrestrial_carbon')
    out.to_csv(p.terrestrial_carbon_shock_output_path, index=False)
    utilities.write_outputs_signature(signature, signature_path)
    hb.log('  carbon shock: %d rows, %d scenarios (shock_pct=shock_pct_contemp=/base_Y, shock_pct_fixedbase=/base_%d) -> %s'
          % (len(out), out['scenario'].nunique() if rows else 0, es_shock_base_year, p.terrestrial_carbon_shock_output_path))
    return True


def _aligned_ha_per_cell(p, density_path):
    """The hectares-per-cell raster that ALIGNS with this carbon raster.

    Density x hectares is a per-pixel product, so the two rasters must share extent, resolution
    and geotransform exactly. The global ha_per_cell_10sec is right for a global run and wrong for
    an AOI subset, whose density rasters cover only the AOI window, so raster_calculator_hb refuses
    the pair as not the same dimensions.

    An AOI run already publishes an aligned raster: p.aoi_ha_per_cell_fine_path is the same
    10-arcsec grid clipped to the AOI, identical in size, pixel and origin to the density rasters.
    So the aligned input is CHOSEN, and neither raster is resampled: resampling hectares-per-cell
    would change the areas the carbon stock is computed from, and resampling density would change
    the stock itself.

    Raises rather than guessing, because a silently misaligned product would still produce a
    plausible-looking carbon total.
    """
    want = hb.get_raster_info_hb(density_path)
    candidates = [getattr(p, name, None) for name in
                  ('aoi_ha_per_cell_fine_path', 'ha_per_cell_10sec_path', 'ha_per_cell_fine_path')]
    tried = []
    for candidate in candidates:
        if not candidate or not hb.path_exists(candidate):
            continue
        have = hb.get_raster_info_hb(candidate)
        tried.append((candidate, tuple(have['raster_size'])))
        same_size = tuple(have['raster_size']) == tuple(want['raster_size'])
        same_gt = all(abs(a - b) <= 1e-12 for a, b in
                      zip(have['geotransform'], want['geotransform']))
        if same_size and same_gt:
            return candidate
    raise NameError(
        'no hectares-per-cell raster aligns with %s, which is %s at geotransform %s. Tried: %s. '
        'Density x hectares is a per-pixel product, so the grids must match exactly; resampling '
        'either one would change the areas or the stock rather than fixing the alignment.'
        % (density_path, tuple(want['raster_size']), tuple(want['geotransform']), tried))


def _stable(value):
    """A JSON-stable form of a signature value.

    json.dumps writes a tuple as a list, so a signature holding tuples never compares equal to the
    one read back and the outputs are rebuilt on every pass.
    """
    if isinstance(value, (list, tuple, set)):
        # str() on each element, so a list mixing strings with nested containers still sorts.
        return sorted(str(_stable(v)) for v in value)
    if isinstance(value, dict):
        return sorted('%s=%s' % (k, _stable(v)) for k, v in value.items())
    return value if isinstance(value, (str, int, float, bool, type(None))) else str(value)


TERRESTRIAL_CARBON_SHOCK_SIGNATURE = 'terrestrial_carbon_shock_signature.json'   # beside the table, in the shared es_shocks dir


def _reuse_check(p, outputs, signature_name, extra_settings=(), extra_inputs=(), light_inputs=()):
    """The reuse signature for one carbon task, and where the last run recorded it.

    The service's own es_parameters (settings by value, `_path` values by content) plus what
    `_service_settings_and_inputs` cannot see: the es_shock_* settings, which are not prefixed, and
    inputs named without the prefix (the region boundary, the GEP tables). The task's own output
    paths are dropped -- a signature that records where it was written changes whenever the project
    directory moves, which forces a needless rebuild.

    Returns:
        tuple: (reason to recompute or None, signature, signature_path).
    """
    settings, inputs = utilities._service_settings_and_inputs(p, 'terrestrial_carbon')
    # Drop the task's own outputs by VALUE, not by attribute name. Name-matching misses
    # terrestrial_carbon_scc_valuation_path and _scc_figure_path, which end in _path rather than
    # _output_path, so the task would fingerprint the files it is about to write: the signature
    # would change every time it ran and the outputs could never be reused.
    own = {str(o) for o in outputs}
    settings = {k: v for k, v in settings.items() if str(v) not in own}
    inputs = {k: v for k, v in inputs.items()
              if str(getattr(p, k, None)) not in own and not k.endswith('_output_path')}
    settings.update(extra_settings)
    inputs.update({str(path): utilities.file_fingerprint(str(path)) for path in extra_inputs})
    # JSON turns tuples into lists, so a tuple in settings never equals what was recorded. Every
    # value is normalised to a stable string form instead.
    settings = {k: _stable(v) for k, v in settings.items()}
    signature = utilities._signature(settings, inputs, light_inputs)
    signature_path = os.path.join(os.path.dirname(outputs[0]), signature_name)
    return utilities.outputs_reuse_reason(outputs, signature, signature_path), signature, signature_path


def terrestrial_carbon_scc_valuation(p):
    """Every scenario map's carbon stock, valued the way the GEP valuation values the base year.

    The shock task above measures carbon as a mean density per zone, because a shock is a ratio and
    area cancels. This task measures it as a STOCK -- density x hectares per cell, summed by
    country -- and prices it at the GEP price convention (`gep_price_convention`, read from
    `gep_price_input_path` at `gep_base_year`), exactly as gep_calculation does for the base year.
    The result is the change from the base year, per country, scenario and year, and a figure.

    Reads the density rasters the shock task cached (carbon_density_<scenario>_<year>.tif in its
    task dir), so it never re-derives carbon from the maps. Writes:
      terrestrial_carbon_scc_valuation.csv   iso3_r250_id, scenario, year, stock_mgc,
                                             delta_stock_mgc, delta_value_usd (annual, interpolated
                                             between anchors as the shock is)
      terrestrial_carbon_scc_valuation.png
    """
    publish_inputs(p)
    utilities.initialize_pyramid_paths(p)
    p.terrestrial_carbon_scc_valuation_path = os.path.join(p.cur_dir, 'terrestrial_carbon_scc_valuation.csv')
    p.terrestrial_carbon_scc_figure_path = os.path.join(p.cur_dir, 'terrestrial_carbon_scc_valuation.png')
    if not p.run_this:
        return True
    import geopandas as gpd

    base_scenario = utilities.required_base_scenario(p, 'terrestrial_carbon')
    base_year = int(p.es_shock_base_year)
    anchor_years = sorted(y for y in map(int, p.es_shock_years) if y > base_year)
    excluded = set(getattr(p, 'es_shock_excluded_scenarios', ()) or ())
    scenarios = [s for s in p.es_shock_scenarios if s not in excluded]
    density_dir = p.terrestrial_carbon_shock_dir   # published by ProjectFlow when the shock task ran

    def stock_by_country(scenario, year):
        """Mg C per country for one scenario map: density x ha per cell, summed by country."""
        density_path = os.path.join(density_dir, 'carbon_density_%s_%d.tif' % (scenario, year))
        if not hb.path_exists(density_path):
            raise NameError('the shock task left no density raster for %s %d: %s' % (scenario, year, density_path))
        per_cell_path = os.path.join(p.cur_dir, 'carbon_per_cell_%s_%d.tif' % (scenario, year))
        if not hb.path_exists(per_cell_path):
            hb.multiply(density_path, _aligned_ha_per_cell(p, density_path), per_cell_path)
        summary_path = os.path.join(p.cur_dir, 'carbon_by_country_region_%s_%d.csv' % (scenario, year))
        if not hb.path_exists(summary_path):
            utilities.summarize_raster_by_region(per_cell_path, p.gep_regions_input_path, summary_path,
                                           year=year, id_column=p.gep_regions_id_col)
        regions = hb.df_read(summary_path)
        # summed on the country id, so a country split across r264 sub-regions counts once
        return regions.groupby('iso3_r250_id')['total'].sum()

    reason, signature, signature_path = _reuse_check(
        p, [p.terrestrial_carbon_scc_valuation_path, p.terrestrial_carbon_scc_figure_path],
        'terrestrial_carbon_scc_valuation_signature.json',
        extra_settings=dict(scenarios=list(scenarios), anchors=list(anchor_years),
                            base_year=base_year, base_scenario=base_scenario,
                            price_convention=p.gep_price_convention,
                            gep_base_year=p.gep_base_year,
                            retained=getattr(p, 'es_provision_retained', None),
                            from_year=getattr(p, 'es_provision_from_year', None),
                            cut_source=getattr(p, 'es_provision_source_scenario', None),
                            cut_label=getattr(p, 'es_provision_label', None)),
        extra_inputs=[p.gep_regions_input_path, p.gep_price_input_path],
        light_inputs=sorted(hb.list_filtered_paths_nonrecursively(density_dir, include_extensions='.tif')
                            if hb.path_exists(density_dir) else []))
    if reason is None:
        hb.log('  carbon SCC valuation: reusing %s (same density rasters, same settings)'
               % p.terrestrial_carbon_scc_valuation_path)
        return True
    hb.log('  carbon SCC valuation: computing because %s' % reason)

    base_stock = stock_by_country(base_scenario, base_year)
    baseline_by_year = {y: stock_by_country(base_scenario, y) for y in anchor_years}
    stocks = {s: {y: stock_by_country(s, y) for y in anchor_years} for s in scenarios}

    df_price = pd.read_excel(p.gep_price_input_path)[[p.gep_price_convention, 'year']]
    price = float(df_price.loc[df_price['year'] == int(p.gep_base_year), p.gep_price_convention].iloc[0])

    out = tcf.scc_valuation_rows(
        stocks, base_stock, price, base_year,
        retained=getattr(p, 'es_provision_retained', None),
        from_year=getattr(p, 'es_provision_from_year', None),
        cut_source=getattr(p, 'es_provision_source_scenario', None),
        cut_label=getattr(p, 'es_provision_label', None),
        baseline_by_year=baseline_by_year)
    out.to_csv(p.terrestrial_carbon_scc_valuation_path, index=False, encoding='utf-8-sig')

    attributes = gpd.read_file(p.gep_regions_input_path, engine='pyogrio')[
        ['iso3_r250_id', 'continent']].drop_duplicates('iso3_r250_id')
    tcf.plot_scc_valuation(out, attributes, price, p.gep_price_convention, base_year,
                           p.terrestrial_carbon_scc_figure_path,
                           aside=getattr(p, 'es_provision_label', None))
    utilities.write_outputs_signature(signature, signature_path)
    world = out.groupby(['scenario', 'year'])['delta_value_usd'].sum()
    last = int(out['year'].max())
    hb.log('  carbon SCC valuation: %d rows, price %.2f USD/Mg C (%s at %s); world change at %d: %s -> %s'
           % (len(out), price, p.gep_price_convention, p.gep_base_year, last,
              {s: '%.1fbn' % (world[(s, last)] / 1e9) for s in sorted(out['scenario'].unique())},
              p.terrestrial_carbon_scc_valuation_path))
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
        hb.log('  carbon shock: dependency csv not found (%s) -- skipping' % carb_path)
        return

    reason, signature, signature_path = _reuse_check(
        p, [p.terrestrial_carbon_shock_output_path], TERRESTRIAL_CARBON_SHOCK_SIGNATURE,
        extra_settings=dict(scenarios=es_shock_scenarios, base_year=es_shock_base_year,
                            end_year=es_shock_end_year, base_scenario=base_scenario,
                            scenario_map=sorted(terrestrial_carbon_scenario_map.items()),
                            mode='static'),
        extra_inputs=[carb_path])
    if reason is None:
        hb.log('  carbon shock: reusing %s (same dependency table, same settings)' % p.terrestrial_carbon_shock_output_path)
        return True
    hb.log('  carbon shock: computing because %s' % reason)

    df = hb.df_read(carb_path)
    # The base resolves through the same candidate mechanism as the data scenarios (and FATALLY if it
    # can't): the frozen tables spell the nature-off baseline two ways across services, and an
    # exact-match miss here would give an empty base -> empty output -> silent GTAP zero.
    raw_base = utilities.resolve_base_scenario(df['scenario'].values, terrestrial_carbon_scenario_map, base_scenario, 'terrestrial_carbon', log=hb.log)
    base = df[(df['scenario'] == raw_base) & (df['year'] == es_shock_end_year)]
    base_vals = base.set_index(['ENDW', 'REG'])['percentage_change'].astype(float) * 100.0

    rows = []
    for our_scn in es_shock_scenarios:
        raw_scn = utilities.resolve_raw_scenario(df['scenario'].values, terrestrial_carbon_scenario_map, our_scn, 'terrestrial_carbon', log=hb.log)
        if raw_scn is None:
            continue
        scn = df[(df['scenario'] == raw_scn) & (df['year'] == es_shock_end_year)]
        scn_vals = scn.set_index(['ENDW', 'REG'])['percentage_change'].astype(float) * 100.0
        rows += utilities.static_shock_rows(base_vals, scn_vals, our_scn, [p.terrestrial_carbon_shock_acts],
                                            es_shock_base_year, es_shock_end_year)

    out = pd.DataFrame(rows)
    out = utilities.filter_to_model_domain(out, p.terrestrial_carbon_shock_output_path, 'terrestrial_carbon', log=hb.log)
    utilities.assert_shock_table_sound(out, es_shock_scenarios, 'terrestrial_carbon')
    out.to_csv(p.terrestrial_carbon_shock_output_path, index=False)
    utilities.write_outputs_signature(signature, signature_path)
    nz = out[(out['year'] == es_shock_end_year) & (out['shock_pct'] != 0)] if len(out) else out
    hb.log('  carbon shock: %d rows, %d scenarios, %d nonzero @%d (static, uncapped) -> %s'
          % (len(out), out['scenario'].nunique() if len(out) else 0, len(nz), es_shock_end_year,
             p.terrestrial_carbon_shock_output_path))
    return True

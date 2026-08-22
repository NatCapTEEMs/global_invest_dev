"""Dynamic pollination ES-shock task (V_F, OSD).

Runs William Sidemo-Holm's crop_benefits pollination chain on our SEALS 300 m maps at EACH SEALS
anchor year (seals_years), then piecewise-linearly interpolates the shock to annual values. Writes
pollination_interpolated.csv -- the file build_combined_afeall_cc_es reads -- into the shared ES-shock
directory (p.es_shock_dir). Grafted by consumers via add_pollination_tasks (dispatch on p.dynamic_es).
"""
import os
import glob
import numpy as np
import pandas as pd
import hazelbean as hb
from global_invest import utilities
from global_invest.pollination import pollination_functions as pf


def publish_inputs(p):
    """Every GEP task's first line: the pollination es_config row (defaults layer -- a caller-set
    value wins) plus the shared country references and the results registry. gep_base_year and
    the value raster (gep_quantity_input_path) are a PAIR defaulting to 2023 -- the crop_benefits
    rasters on hand; the GEP manuscript's base year is 2019, so regenerate
    poll_value_global_2019usd.tif with the crop_benefits recipe and update BOTH cells (the row
    names the year twice) before quoting a manuscript-aligned number."""
    utilities.hydrate_es_config(p, 'pollination', log=hb.log)
    utilities.hydrate_es_parameters(p, 'pollination', log=hb.log)
    utilities.initialize_country_paths(p)
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def _read_masked(raster_path):
    """A single-band raster as float64 with its nodata read as NaN, and its metadata beside it."""
    import rasterio
    with rasterio.open(raster_path) as src:
        arr = src.read(1).astype(np.float64)
        arr[arr == src.nodata] = np.nan
        return arr, src.meta.copy()


def _zonal_context(denominator_path, correspondence_gpkg):
    """The fixed side of every zonal percent change, read once.

    The baseline value raster defines the grid, so the pixel-area raster and the burned zone ids
    are built on it. All three plus the zone labels are handed to pf.zonal_pct_change with each
    scenario's difference array.
    """
    import geopandas as gpd
    from rasterio.features import rasterize
    from crop_benefits.raster.spatial import build_area_km2_raster

    gdf = gpd.read_file(correspondence_gpkg, engine='pyogrio')
    if gdf.crs is None or gdf.crs.to_epsg() != pf.LATLON_EPSG:
        gdf = gdf.to_crs(pf.LATLON_EPSG)
    gdf[pf.REGION_ID_FIELD] = gdf[pf.REGION_ID_FIELD].astype(int)

    baseline, meta = _read_masked(denominator_path)
    zones = rasterize(
        ((g, int(r)) for g, r in zip(gdf.geometry, gdf[pf.REGION_ID_FIELD]) if g is not None and not g.is_empty),
        out_shape=(meta['height'], meta['width']), transform=meta['transform'],
        fill=pf.NO_ZONE_ID, dtype=np.int32)
    return baseline, build_area_km2_raster(meta), zones, pf.zone_labels_from_boundary(gdf)


def pollination_shock(p):
    """Per-scenario 300 m LULC at each SEALS anchor year -> V_F/OSD shock, piecewise-interp to annual.

    Caller sets on p: es_shock_years (SEALS anchor years, from seals_years),
    es_shock_base_year, es_shock_scenarios, es_lulc_path_template
    ({scenario}/{year}) or scenario_lulc_paths, pollination_base_year_lulc_path (or the shared es_base_year_lulc_path),
    pollination_shock_output_path. Optional: es_shock_base_scenario, pollination_shock_acts,
    region_boundary_path.
    """
    # Default into the es_shocks parent dir. Runtime, not build time: p.es_shock_dir is
    # published by that task, which ProjectFlow runs before this one.
    if not getattr(p, 'pollination_shock_output_path', None):
        p.pollination_shock_output_path = os.path.join(getattr(p, 'es_shock_dir', None) or p.project_dir, 'pollination_interpolated.csv')
    if not p.run_this:
        return
    # Export keys and boundary are es_parameters shipped defaults; a consumer's own tables win.
    utilities.hydrate_es_parameters(p, 'pollination', log=hb.log)
    base_scenario      = utilities.required_base_scenario(p, 'pollination')
    es_shock_base_year = int(p.es_shock_base_year)
    es_shock_scenarios = list(p.es_shock_scenarios)      # was read unbound: this task raised NameError
    anchor_years = sorted(y for y in map(int, p.es_shock_years) if y > es_shock_base_year)

    cfg = pf.configure_crop_benefits(p, es_shock_base_year)            # crop_benefits Config -> our base_data + task dir

    if not getattr(p, 'scenario_lulc_paths', None):
        tmpl = p.es_lulc_path_template
        p.scenario_lulc_paths = {s: {y: glob.glob(tmpl.format(scenario=s, year=y))[0]
                                     for y in anchor_years if glob.glob(tmpl.format(scenario=s, year=y))}
                                 for s in [base_scenario] + es_shock_scenarios}

    # base-year SEALS7 map: per-ES override, else the ES-shared attr. NEVER p.base_year_lulc_path, which
    # SEALS OWNS and overwrites at runtime with its raw-ESA source (a raw-ESA base map would make the
    # SEALS7-keyed sufficiency lookup produce garbage). This base-year value IS pollination's primary
    # denominator (it feeds both fixedbase and contemp), so a missing one is fatal.
    base_map = getattr(p, 'pollination_base_year_lulc_path', None) or getattr(p, 'es_base_year_lulc_path', None)
    if not base_map:
        raise ValueError('pollination base-year LULC not set: point p.es_base_year_lulc_path (or '
                         'p.pollination_base_year_lulc_path) at the SEALS7 base-year map.')
    # The denominator (unpaired 2023 value) is year- and scenario-independent, so the fixed side of
    # the zonal step is built once.
    denominator_path = pf.baseline_denominator(cfg, base_map, es_shock_base_year)
    baseline_arr, area_arr, zones_arr, zone_labels = _zonal_context(denominator_path, p.region_boundary_path)

    # value[scenario][year] = per-zone % change of that scenario's year-map vs the 2023 baseline (stable
    # ag). level_usd = the denominator of that % change, the per-zone absolute baseline value in base-year
    # USD, emitted so the GEP chain can consume this task instead of rerunning the same rasters.
    value, level_usd = {}, None
    for year in anchor_years:
        for scen in [base_scenario] + es_shock_scenarios:
            diff_arr, _ = _read_masked(pf.scenario_diff_raster(
                cfg, scenario=f'{scen}_{year}', lulc_path=p.scenario_lulc_paths[scen][year],
                baseline_lulc_path=base_map, target_year=es_shock_base_year))
            pct, level = pf.zonal_pct_change(diff_arr, baseline_arr, area_arr, zones_arr, zone_labels)
            value.setdefault(scen, {})[year] = pct
            if level_usd is None:
                level_usd = level

    # The shock numerator is scenario minus nature-off baseline at each anchor. anchor_shock_tables
    # puts it over the two denominators and dynamic_shock_rows expands those to annual rows.
    rows = []
    for scen in es_shock_scenarios:
        anchor_shock, anchor_contemp = pf.anchor_shock_tables(
            {y: value[scen][y] for y in anchor_years},
            {y: value[base_scenario][y] for y in anchor_years})
        rows += pf.dynamic_shock_rows(anchor_shock, anchor_contemp, level_usd, scen,
                                      p.pollination_shock_acts, es_shock_base_year)

    out = pd.DataFrame(rows)
    utilities.assert_shock_table_sound(out, es_shock_scenarios, 'pollination')
    out.to_csv(p.pollination_shock_output_path, index=False)
    print('  pollination shock: %d rows, %d scenarios (shock_pct=shock_pct_contemp=/baseline-year value, shock_pct_fixedbase=/2023 value) -> %s'
          % (len(out), out['scenario'].nunique() if rows else 0, p.pollination_shock_output_path))
    # value_usd_base is the GEP hand-off, not read by GTAP (build_combined_afeall takes shock_pct only).
    if level_usd is not None and len(level_usd):
        print('  pollination value (GEP): %d zones, total %.4g base-year USD -> column value_usd_base'
              % (len(level_usd), float(level_usd.sum())))
    return True


def pollination_shock_static(p):
    """Static per-scenario pollination shock -> V_F/OSD, linear ramp 0->es_shock_end_year, from the frozen table.

    The fallback add_pollination_tasks selects when <2 SEALS map years exist. READS
    input_dir/raw_dependencies/pollination_dependency.csv (override p.pollination_dependency_path) and
    subtracts the baseline_ignore_damages row (the frozen table's nature-off baseline; == ignore
    dependencies, just the old label kept in this CSV), ramping that difference linearly from 0 at
    es_shock_base_year. NEVER writes back to raw_dependencies -- output goes to p.pollination_shock_output_path
    (pollination_interpolated.csv), the same file the dynamic task writes. Caller sets:
    es_shock_base_year, es_shock_end_year, es_shock_scenarios,
    pollination_shock_output_path; scenario->raw via p.pollination_scenario_map (default: identity --
    each scenario maps to its own name; a scenario the table labels differently is warned about loudly
    and skipped rather than silently zeroed, so set the map for those);
    sectors via p.pollination_shock_acts (default ('V_F', 'OSD'), matching the dynamic task).
    """
    # Default into the es_shocks parent dir. Runtime, not build time: p.es_shock_dir is
    # published by that task, which ProjectFlow runs before this one.
    if not getattr(p, 'pollination_shock_output_path', None):
        p.pollination_shock_output_path = os.path.join(getattr(p, 'es_shock_dir', None) or p.project_dir, 'pollination_interpolated.csv')
    if not p.run_this:
        return

    utilities.hydrate_es_parameters(p, 'pollination', log=hb.log)   # shipped defaults; caller wins
    es_shock_base_year = int(p.es_shock_base_year)
    es_shock_end_year = int(p.es_shock_end_year)
    pollination_scenario_map = getattr(p, 'pollination_scenario_map', {})
    es_shock_scenarios = list(p.es_shock_scenarios)

    poll_path = getattr(p, 'pollination_dependency_path', None) or os.path.join(
        p.input_dir, 'raw_dependencies', 'pollination_dependency.csv')
    if not os.path.exists(poll_path):
        print('  pollination shock: dependency csv not found (%s) -- skipping' % poll_path)
        return

    df = pd.read_csv(poll_path)
    # Normalise the one year-suffixed label so pollination's table presents the same scenario names as
    # the other services (carbon is plain everywhere, erosion strips '_2050' on read). net_zero_2050 is
    # the only alias here, so target it explicitly rather than a blunt '_2050' strip that would silently
    # mangle any future label carrying that suffix. Keeps the consumer's pollination_scenario_map at identity.
    df['scenario'] = df['scenario'].replace({'net_zero_2050': 'net_zero'})
    df = df[df['ENDW'] != 'AEZ0']  # AEZ0 not valid in GTAP
    # The base resolves through the candidate mechanism (fatal if absent): the table's spelling
    # may differ from the configured name (nature-off aliasing in utilities).
    base_scenario = utilities.required_base_scenario(p, 'pollination')
    raw_base = utilities.resolve_base_scenario(df['scenario'].values, pollination_scenario_map, base_scenario, 'pollination')
    base = df[df['scenario'] == raw_base].set_index(['ENDW', 'REG'])['value'].astype(float)

    rows = []
    for our_scn in es_shock_scenarios:
        raw_scn = utilities.resolve_raw_scenario(df['scenario'].values, pollination_scenario_map, our_scn, 'pollination')
        if raw_scn is None:
            continue
        scn_vals = df[df['scenario'] == raw_scn].set_index(['ENDW', 'REG'])['value'].astype(float)
        rows += pf.static_shock_rows(base, scn_vals, our_scn, p.pollination_shock_acts,
                                     es_shock_base_year, es_shock_end_year)

    out = pd.DataFrame(rows)
    utilities.assert_shock_table_sound(out, es_shock_scenarios, 'pollination')
    out.to_csv(p.pollination_shock_output_path, index=False)
    nz = out[(out['year'] == es_shock_end_year) & (out['shock_pct'] != 0)] if len(out) else out
    print('  pollination shock: %d rows, %d scenarios, %d nonzero @%d (static, uncapped) -> %s'
          % (len(out), out['scenario'].nunique() if len(out) else 0, len(nz), es_shock_end_year,
             p.pollination_shock_output_path))
    return True


# =============================================================================
# GEP valuation tasks. The ES shock above and the valuation below are separate
# consumers of the same crop_benefits science; neither depends on the other.
# The value raster (poll_value_global_<year>usd.tif, a crop_benefits output) is
# USD per cell with crop prices and pollination dependence already embedded
# upstream, so lambda = 1 and no price join happens here: quantity IS value.
# =============================================================================

def pollination_value_by_region(p):
    """GEP quantity stage: per-region sum of the pollination value raster (USD per cell) on the
    service's aggregation surface (the gep_regions cells)."""
    publish_inputs(p)
    p.pollination_value_by_region_path = os.path.join(p.cur_dir, "pollination_value_by_region.csv")
    if not p.run_this:
        return
    utilities.summarize_raster_by_region(
        value_raster_path=p.gep_quantity_input_path,
        region_boundary_path=p.gep_regions_input_path,
        out_path=p.pollination_value_by_region_path,
        year=p.gep_base_year, id_column=p.gep_regions_id_col)
    # Full-scale conservation: the region sums must add up to the raster's own total
    # (verified at 100.0000% on the real raster when this check was added).
    df_regions = pd.read_csv(p.pollination_value_by_region_path)
    utilities.assert_zonal_conservation(df_regions['total'].sum(),
                                        p.gep_quantity_input_path, 'pollination')
    return True


def gep_calculation(p):
    """GEP valuation for pollination: r264 region values -> ONE row per country (r250)."""
    publish_inputs(p)
    service_results = {}
    p.results['pollination'] = service_results
    p.results['pollination']['gep_by_country_base_year'] = os.path.join(p.cur_dir, "gep_by_country_base_year.csv")
    # Only register results this task actually writes (per-year results belong to a multi-year run).

    if hb.path_all_exist(list(service_results.values())):
        hb.log("All results already exist. Skipping GEP calculation for pollination.")
        return
    hb.log("Starting GEP calculation for pollination.")

    # 1. Per-region (r264) USD value -> one row per COUNTRY (r250), written as the per-country CSV
    #    that is the source of truth for every sum. Summing the r264-expanded table instead would
    #    double-count split countries (see utilities docstring).
    df_q264 = pd.read_csv(p.pollination_value_by_region_path)
    df_gep = pf.collapse_regions_to_countries(df_q264)
    hb.df_write(df_gep, service_results['gep_by_country_base_year'])

    # 2. Map only: r264-expanded, each sub-region carries its country's value, never summed
    #    (carbon template; the repo-wide map convention is the open flag-3 decision).
    df_regions = pf.expand_country_values_to_regions(df_q264, df_gep)
    gdf = hb.df_merge(p.gdf_countries_simplified, df_regions, how='outer',
                      left_on='ee_r264_id', right_on='ee_r264_id')
    gdf.to_file(service_results['gep_by_country_base_year'].replace('.csv', '.gpkg'), driver='GPKG')

    # 3. National total = sum over the one-row-per-country table.
    value_gep_base_year = df_gep['pollination_gep'].sum()
    hb.log(f"Total pollination GEP for base year {p.gep_base_year}: {value_gep_base_year}")
    return value_gep_base_year


def gep_load_results(p):
    """Load GEP results from a PRIOR calculation run so the report renders without recomputing.
    Fails loudly if absent (run run_pollination.py first). The results-only entry point."""
    publish_inputs(p)
    result_path = os.path.join(p.intermediate_dir, 'gep_calculation', 'gep_by_country_base_year.csv')
    if not hb.path_exists(result_path):
        raise FileNotFoundError(
            f"pollination GEP results not found at {result_path}. "
            f"Run the calculation first (run_pollination.py), then re-run results.")
    p.results.setdefault('pollination', {})
    p.results['pollination']['gep_by_country_base_year'] = result_path


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)

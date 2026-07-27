"""Erosion-control ES-shock task (STATIC, per-scenario). Mirrors carbon/pollination/fisheries on the
add_<es>_tasks seam.

Ported verbatim from the old prepare_es_shocks erosion block: read erosion_prevention_dependency.csv,
subtract the baseline_ignore_damages reference, linearly ramp 0 -> the scenario value over the horizon,
apply to the 8 erosion-affected crop sectors, write erosion_prevention_interpolated.csv. UNCAPPED here --
the cap is applied later on the COMBINED value in build_combined_afeall_cc_es (matches the old block).
The paper wants this DYNAMIC (InVEST SDR on each SEALS map -- the erosion GEP model), the heavy
upgrade tracked in #26; this module is the static seam so the dynamic swap is contained later.
"""
import os
import pandas as pd

from global_invest.erosion import erosion_functions as ef

# 8 crop sectors whose productivity depends on erosion control (sediment retention).
EROSION_SECTORS = ('PDR', 'WHT', 'GRO', 'V_F', 'OSD', 'C_B', 'PFB', 'OCR')
# our scenario -> raw_dependencies scenario name(s), with fallbacks (stress_test reuses current_policies).
EROSION_SCENARIO_MAP = {
    'below_2c': ['below_2c'], 'current_policies': ['current_policies'],
    'delayed_transition': ['delayed_transition'], 'fragmented_world': ['fragmented_world'],
    'low_demand': ['low_demand'], 'ndcs': ['ndcs'],
    'net_zero': ['net_zero', 'net_zero_2050'], 'stress_test': ['current_policies'],
}


def task_compute_erosion_shock(p):
    """Static per-scenario erosion shock -> 8 crop sectors, linear ramp 0->end_year.

    Caller sets on p before calling: erosion_shock_scenarios, erosion_shock_base_year,
    erosion_shock_end_year, erosion_shock_output_path. Dependency csv defaults to
    input_dir/raw_dependencies/erosion_prevention_dependency.csv (override p.erosion_dependency_path);
    scenario->raw name via p.erosion_scenario_map (default EROSION_SCENARIO_MAP).
    """
    if not p.run_this:
        return

    base_year = int(p.erosion_shock_base_year)
    end_year = int(p.erosion_shock_end_year)
    n_years = end_year - base_year
    scenario_map = getattr(p, 'erosion_scenario_map', EROSION_SCENARIO_MAP)
    scenarios = list(p.erosion_shock_scenarios)

    ero_path = getattr(p, 'erosion_dependency_path', None) or os.path.join(
        p.input_dir, 'raw_dependencies', 'erosion_prevention_dependency.csv')
    if not os.path.exists(ero_path):
        print('  erosion shock: dependency csv not found (%s) -- skipping' % ero_path)
        return

    df, base_vals = ef.read_erosion_dependency(ero_path)

    rows = []
    for our_scn in scenarios:
        candidates = scenario_map.get(our_scn)
        raw_scn = ef.find_scenario(df, candidates) if candidates else None
        if not raw_scn:
            continue
        scn_vals = df[df['scenario'] == raw_scn].set_index(
            ['aez18_id', 'gtapv7_r50_label'])['value'].astype(float).fillna(0)
        common = scn_vals.index.intersection(base_vals.index)
        shock = scn_vals.loc[common] - base_vals.loc[common]
        for year in range(base_year, end_year + 1):
            frac = (year - base_year) / n_years
            for (aez_id, reg), val in shock.items():
                endw = 'AEZ%d' % int(aez_id)
                for sector in EROSION_SECTORS:
                    rows.append({'ENDW': endw, 'ACTS': sector, 'REG': reg,
                                 'scenario': our_scn, 'year': year, 'shock_pct': val * frac})

    out = pd.DataFrame(rows)
    out.to_csv(p.erosion_shock_output_path, index=False)
    nz = out[(out['year'] == end_year) & (out['shock_pct'] != 0)] if len(out) else out
    print('  erosion shock: %d rows, %d scenarios, %d nonzero @%d (static, uncapped) -> %s'
          % (len(out), out['scenario'].nunique() if len(out) else 0, len(nz), end_year,
             p.erosion_shock_output_path))
    return True


# ---------------------------------------------------------------------------
# DYNAMIC path (#26): recompute the shock from our SEALS maps instead of the frozen table.
# Reached only when add_erosion_tasks dispatches to it (>=2 SEALS land-cover years). Bodies are
# filled incrementally; each mirrors carbon/pollination. Heavy imports (natcap.invest.sdr,
# pygeoprocessing.routing) go inside the functions so module import stays light.
# ---------------------------------------------------------------------------

def task_erosion_sdr(p):
    """DYNAMIC step 1: per (scenario, anchor year), resample the SEALS map to the erosion analysis
    grid (p.modality: local -> 6.45 km reference grid, sc/msi -> native SEALS resolution) and run
    InVEST SDR. Outputs per map, in p.cur_dir/<scn>_<yr>/: usle_<scn>_<yr>.tif (actual erosion) and
    rkls_<scn>_<yr>.tif (potential/bare soil); avoided = rkls - usle is formed downstream.

    Caller sets on p: scenario_lulc_paths {scenario: {year: seals_lulc_path}}; erosion_dem_path,
    erosion_erosivity_path, erosion_erodibility_path, erosion_watersheds_path,
    erosion_biophysical_table_path (SEALS7 lucode -> usle_c/usle_p); erosion_analysis_grid_path
    (6.45 km reference raster; local only). SDR knobs via p.erosion_sdr_params (defaults below).
    """
    if not p.run_this:
        return
    import hazelbean as hb
    from natcap.invest.sdr import sdr

    # analysis grid: downsample to a 6.45 km reference for local; run at native SEALS res on the cluster
    native = getattr(p, 'modality', 'local') in ('sc', 'msi')
    grid_ref = None if native else p.get_path(p.erosion_analysis_grid_path)
    dem         = p.get_path(p.erosion_dem_path)
    erosivity   = p.get_path(p.erosion_erosivity_path)
    erodibility = p.get_path(p.erosion_erodibility_path)
    watersheds  = p.get_path(p.erosion_watersheds_path)
    biophysical = p.get_path(p.erosion_biophysical_table_path)
    sdr_params = dict(threshold_flow_accumulation=1000, k_param=2, sdr_max=0.8,
                      ic_0_param=0.5, l_max=122, flow_dir_algorithm='D8', n_workers=-1)
    sdr_params.update(getattr(p, 'erosion_sdr_params', {}))

    n = 0
    for scenario, by_year in p.scenario_lulc_paths.items():
        for year, lulc in by_year.items():
            lulc = p.get_path(lulc)
            if native:
                lulc_grid = lulc
            else:
                lulc_grid = os.path.join(p.cur_dir, 'lulc_%s_%d_grid.tif' % (scenario, year))
                if not os.path.exists(lulc_grid):  # categorical LULC -> mode
                    hb.resample_to_match(lulc, grid_ref, lulc_grid, resample_method='mode')
            suffix = '%s_%d' % (scenario, year)
            sdr.execute(dict(workspace_dir=os.path.join(p.cur_dir, suffix), results_suffix=suffix,
                             dem_path=dem, erosivity_path=erosivity, erodibility_path=erodibility,
                             lulc_path=lulc_grid, watersheds_path=watersheds,
                             biophysical_table_path=biophysical, **sdr_params))
            n += 1
    p.erosion_sdr_dir = p.cur_dir      # downstream tasks read usle_/rkls_/avoided_erosion_ from here
    print('  erosion SDR: %d scenario x year maps (%s grid) -> usle_/rkls_ in %s'
          % (n, 'native SEALS' if native else '6.45 km', p.cur_dir))
    return True


def task_erosion_upstream(p):
    """DYNAMIC step 2: per (scenario, year), upstream prevention share = acc(avoided) / acc(rkls),
    D8 flow-accumulation of avoided-mass over potential-mass (the pixel-area weight cancels in the
    ratio). Recomputed per scenario because the upslope land cover changes. Reads the SDR outputs
    from p.erosion_sdr_dir; writes upstream_<scn>_<yr>.tif to p.cur_dir. Uses pygeoprocessing.routing
    (fill_pits -> flow_dir_d8 -> flow_accumulation_d8), verified equal to a hand-written D8.

    Caller sets on p: erosion_dem_path (aligned to the SDR grid here) + the step-1 outputs.
    """
    if not p.run_this:
        return
    import numpy as np
    import hazelbean as hb
    import pygeoprocessing as pgp
    import pygeoprocessing.routing as routing

    dem = p.get_path(p.erosion_dem_path)

    def _clean_weight(src, dst, ps, gt, wkt):        # nodata/negatives -> 0 for weighted accumulation
        info = pgp.get_raster_info(src); nd = info['nodata'][0]
        a = hb.as_array(src).astype('float64')
        a = np.where(np.isfinite(a) & (a != nd) & (np.abs(a) < 1e30), np.maximum(a, 0.0), 0.0)
        pgp.numpy_array_to_raster(a.astype('float32'), -1.0, ps, (gt[0], gt[3]), wkt, dst)

    n = 0
    for scenario, by_year in p.scenario_lulc_paths.items():
        for year in by_year:
            suffix = '%s_%d' % (scenario, year)
            sdr_dir = os.path.join(p.erosion_sdr_dir, suffix)
            avoided = os.path.join(sdr_dir, 'avoided_erosion_%s.tif' % suffix)
            rkls    = os.path.join(sdr_dir, 'rkls_%s.tif' % suffix)
            work = os.path.join(p.cur_dir, suffix); os.makedirs(work, exist_ok=True)
            info = pgp.get_raster_info(avoided)
            ps, gt, wkt = info['pixel_size'], info['geotransform'], info['projection_wkt']

            dem_g = os.path.join(work, 'dem_grid.tif')
            hb.resample_to_match(dem, avoided, dem_g, resample_method='bilinear')
            _clean_weight(avoided, os.path.join(work, 'avoided_w.tif'), ps, gt, wkt)
            _clean_weight(rkls,    os.path.join(work, 'rkls_w.tif'),    ps, gt, wkt)

            routing.fill_pits((dem_g, 1), os.path.join(work, 'filled.tif'))
            routing.flow_dir_d8((os.path.join(work, 'filled.tif'), 1), os.path.join(work, 'fdir.tif'))
            routing.flow_accumulation_d8((os.path.join(work, 'fdir.tif'), 1), os.path.join(work, 'acc_avoided.tif'),
                                         weight_raster_path_band=(os.path.join(work, 'avoided_w.tif'), 1))
            routing.flow_accumulation_d8((os.path.join(work, 'fdir.tif'), 1), os.path.join(work, 'acc_rkls.tif'),
                                         weight_raster_path_band=(os.path.join(work, 'rkls_w.tif'), 1))

            acc_av = hb.as_array(os.path.join(work, 'acc_avoided.tif')).astype('float64')
            acc_rk = hb.as_array(os.path.join(work, 'acc_rkls.tif')).astype('float64')
            with np.errstate(invalid='ignore', divide='ignore'):
                ups = np.where(acc_rk > 0, np.clip(acc_av / acc_rk, 0.0, 1.0), -9999.0).astype('float32')
            pgp.numpy_array_to_raster(ups, -9999.0, ps, (gt[0], gt[3]), wkt,
                                      os.path.join(p.cur_dir, 'upstream_%s.tif' % suffix))
            n += 1
    p.erosion_upstream_dir = p.cur_dir
    print('  erosion upstream: %d maps -> upstream_<scn>_<yr>.tif in %s' % (n, p.cur_dir))
    return True


def task_erosion_prevention(p):
    """DYNAMIC step 3: per (scenario, year), on-farm PS = avoided/(avoided+usle) on severe pixels
    (usle > threshold); combined = 1 - (1-onfarm)(1-upstream), zeroed off severe pixels. Reads
    usle/avoided (p.erosion_sdr_dir) and upstream (p.erosion_upstream_dir); writes
    ps_onfarm_<scn>_<yr>.tif and ps_combined_<scn>_<yr>.tif.

    NB: the cropland restriction is NOT applied here -- it comes from SPAM production being zero
    off-cropland in task_erosion_valuation (matches the erosion valuation code, which puts PS on all
    severe pixels). The PS is a direct, validated formula; the SPAM/elasticity step is deferred to
    valuation. Caller may set p.erosion_severe_threshold_t_ha (default 11; the per-country 11/2
    policy is a later refinement).
    """
    if not p.run_this:
        return
    import numpy as np
    import hazelbean as hb
    import pygeoprocessing as pgp

    threshold = float(getattr(p, 'erosion_severe_threshold_t_ha', 11.0))

    def _rd(path):
        info = pgp.get_raster_info(path); nd = info['nodata'][0]
        a = hb.as_array(path).astype('float64')
        return np.where(np.isfinite(a) & (a != nd) & (np.abs(a) < 1e30), a, np.nan)

    n = 0
    for scenario, by_year in p.scenario_lulc_paths.items():
        for year in by_year:
            suffix = '%s_%d' % (scenario, year)
            sdr_dir = os.path.join(p.erosion_sdr_dir, suffix)
            usle_p    = os.path.join(sdr_dir, 'usle_%s.tif' % suffix)
            avoided_p = os.path.join(sdr_dir, 'avoided_erosion_%s.tif' % suffix)
            ups_p     = os.path.join(p.erosion_upstream_dir, 'upstream_%s.tif' % suffix)

            info = pgp.get_raster_info(usle_p)
            ps, gt, wkt = info['pixel_size'], info['geotransform'], info['projection_wkt']
            usle = np.nan_to_num(_rd(usle_p)); avoided = np.nan_to_num(np.maximum(_rd(avoided_p), 0))
            ups = np.clip(np.nan_to_num(_rd(ups_p)), 0.0, 1.0)

            mask = usle > threshold          # severe pixels; cropland comes from SPAM in valuation
            with np.errstate(invalid='ignore', divide='ignore'):
                onfarm = np.where(mask & (avoided + usle > 0), avoided / (avoided + usle), 0.0)
            combined = np.where(mask, 1.0 - (1.0 - onfarm) * (1.0 - ups), 0.0)

            pgp.numpy_array_to_raster(onfarm.astype('float32'), -9999.0, ps, (gt[0], gt[3]), wkt,
                                      os.path.join(p.cur_dir, 'ps_onfarm_%s.tif' % suffix))
            pgp.numpy_array_to_raster(combined.astype('float32'), -9999.0, ps, (gt[0], gt[3]), wkt,
                                      os.path.join(p.cur_dir, 'ps_combined_%s.tif' % suffix))
            n += 1
    p.erosion_prevention_dir = p.cur_dir
    print('  erosion prevention: %d maps -> ps_onfarm_/ps_combined_ (severe cropland, threshold=%.1f)'
          % (n, threshold))
    return True


def task_erosion_valuation(p):
    """DYNAMIC step 4: per scenario, per-pixel crop-productivity value (combined PS x SPAM crop
    production x supply elasticity) -> zonal means over ee_r50_aez18 (summarize_raster_by_region)
    -> the shock as ABSOLUTE differences of the productivity-share level (comparable to
    carbon/pollination because that level is already a fraction of output): contemporaneous
    (scn_Y - base_Y) and fixed-base (scn_Y - base_0). Writes the 8-sector per-zone shock CSV at
    p.erosion_shock_output_path (same format as the static path / carbon / pollination).
    """
    if not p.run_this:
        return
    raise NotImplementedError('dynamic erosion valuation -- build tracked in #26')

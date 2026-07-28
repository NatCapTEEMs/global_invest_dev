"""Erosion-control ES-shock tasks (static + dynamic), on the add_<es>_tasks seam like carbon/pollination.

STATIC (task_compute_erosion_shock): read raw_dependencies/erosion_prevention_dependency.csv, subtract
the baseline reference, linearly ramp 0 -> the scenario value over the horizon, apply to the 8
erosion-affected crop sectors -> erosion_prevention_interpolated.csv. UNCAPPED here -- the cap is applied
later on the COMBINED value in build_combined_afeall_cc_es.

DYNAMIC (#26; task_erosion_sdr -> upstream -> prevention -> valuation): recompute the shock from our
SEALS maps via InVEST SDR -> D8 upstream -> prevention shares -> per-zone valuation. add_erosion_tasks
(erosion_initialize) dispatches static vs dynamic on the number of SEALS map years.
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
    sectors = getattr(p, 'erosion_shock_acts', EROSION_SECTORS)   # GTAP sectors, standardized name (was erosion_sectors)

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
                for sector in sectors:
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
    import glob
    import hazelbean as hb
    from natcap.invest.sdr import sdr

    # Build scenario_lulc_paths from a template if the caller didn't (mirrors carbon/pollination).
    # p.erosion_lulc_path_template uses {scenario} and {year}; include the base scenario for differencing.
    if not getattr(p, 'scenario_lulc_paths', None) and getattr(p, 'erosion_lulc_path_template', None):
        tmpl = p.erosion_lulc_path_template
        years = [int(y) for y in getattr(p, 'erosion_shock_years', [])]
        scens = list(getattr(p, 'erosion_shock_scenarios', []))
        base = getattr(p, 'erosion_shock_base_scenario', 'baseline_ignore_damages')
        if base not in scens:
            scens = scens + [base]
        p.scenario_lulc_paths = {}
        for scn in scens:
            yr_map = {y: sorted(glob.glob(tmpl.format(scenario=scn, year=y)))[0]
                      for y in years if glob.glob(tmpl.format(scenario=scn, year=y))}
            if yr_map:
                p.scenario_lulc_paths[scn] = yr_map

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
    usle/avoided (p.erosion_sdr_dir) and upstream (p.erosion_upstream_dir); writes ps_combined_<scn>_<yr>.tif
    (on-farm is an intermediate for the combined union -- only combined is valued downstream).

    NB: the cropland restriction is NOT applied here -- it comes from SPAM production being zero
    off-cropland in task_erosion_valuation (matches the erosion valuation code, which puts PS on all
    severe pixels). The PS is a direct, validated formula; the SPAM/elasticity step is deferred to
    valuation. The severe threshold follows the per-country SES-11 policy (T=2 for small-area
    <50,000 km2 or low-elevation <250 m countries, else 11) when p.erosion_country_boundary_path
    (+ p.erosion_dem_path for the elevation rule) is set; otherwise flat p.erosion_severe_threshold_t_ha.
    """
    if not p.run_this:
        return
    import numpy as np
    import rioxarray as rxr
    import pygeoprocessing as pgp
    from rasterio.crs import CRS as rioCRS
    from rasterio.enums import Resampling
    from global_invest.erosion import erosion_functions as ef

    thresh_high = float(getattr(p, 'erosion_severe_threshold_t_ha', 11.0))
    analysis_crs = rioCRS.from_epsg(int(getattr(p, 'erosion_analysis_epsg', 8857)))

    def _to_grid(path, template=None):     # reproject to the equal-area analysis grid (average)
        da = rxr.open_rasterio(path, masked=True).squeeze().rio.reproject(analysis_crs, resampling=Resampling.average)
        return da if template is None else da.rio.reproject_match(template, resampling=Resampling.average)

    n = 0
    for scenario, by_year in p.scenario_lulc_paths.items():
        for year in by_year:
            suffix = '%s_%d' % (scenario, year)
            sdr_dir = os.path.join(p.erosion_sdr_dir, suffix)
            # PS is computed ON the analysis grid (reproject usle/avoided/ups FIRST, then PS) -- the
            # order matters because PS is nonlinear; computing it on the native grid then reprojecting
            # biases the shock high.
            usle = _to_grid(os.path.join(sdr_dir, 'usle_%s.tif' % suffix))
            avoided = _to_grid(os.path.join(sdr_dir, 'avoided_erosion_%s.tif' % suffix), usle)
            ups = _to_grid(os.path.join(p.erosion_upstream_dir, 'upstream_%s.tif' % suffix), usle)
            usle_v = np.nan_to_num(np.maximum(usle.values, 0.0))
            avoided_v = np.nan_to_num(np.maximum(avoided.values, 0.0))
            ups_v = np.clip(np.nan_to_num(ups.values), 0.0, 1.0)

            # per-country severe threshold (SES-11 policy: T=2 for small-area/low-elevation countries,
            # else thresh_high). Computed once on the analysis grid (same across scenarios), cached.
            thr = getattr(p, '_erosion_threshold_raster', None)
            if thr is None:
                cb = getattr(p, 'erosion_country_boundary_path', None)
                if cb:
                    thr = ef.build_severe_threshold_raster(
                        usle, p.get_path(cb),
                        p.get_path(p.erosion_dem_path) if getattr(p, 'erosion_dem_path', None) else None,
                        thresh_high=thresh_high,
                        thresh_low=float(getattr(p, 'erosion_threshold_low_t_ha', 2.0)),
                        small_area_km2=float(getattr(p, 'erosion_small_country_area_km2', 50_000)),
                        low_elevation_mean_m=float(getattr(p, 'erosion_low_elevation_mean_m', 250)))
                else:
                    thr = thresh_high        # flat fallback when no country boundary is provided
                p._erosion_threshold_raster = thr
            mask = usle_v > thr              # severe pixels (per-country T); cropland from SPAM in valuation
            with np.errstate(invalid='ignore', divide='ignore'):
                onfarm = np.where(mask & (avoided_v + usle_v > 0), avoided_v / (avoided_v + usle_v), 0.0)
            combined = np.where(mask, 1.0 - (1.0 - onfarm) * (1.0 - ups_v), 0.0)

            tr = usle.rio.transform(); px = usle.rio.resolution()
            pgp.numpy_array_to_raster(combined.astype('float32'), -9999.0, (px[0], px[1]),
                                      (tr.c, tr.f), usle.rio.crs.to_wkt(),
                                      os.path.join(p.cur_dir, 'ps_combined_%s.tif' % suffix))
            n += 1
    p.erosion_prevention_dir = p.cur_dir
    per_country = getattr(p, 'erosion_country_boundary_path', None) is not None
    print('  erosion prevention: %d maps -> ps_combined_ on EPSG:%d (severe T=%s)'
          % (n, analysis_crs.to_epsg(), 'per-country 11/2' if per_country else '%.1f flat' % thresh_high))
    return True


def task_erosion_valuation(p):
    """DYNAMIC step 4: per (scenario, year) compute the per-ee_r50_aez18 erosion protection LEVEL
    (the erosion-model formula, validated to ~1.02x the reference country account): for each SPAM crop, protected =
    combined_PS * yield * area, total = yield * area, summed per zone; level =
    sum(protected*elasticity)/sum(total). Then the shock as ABSOLUTE two-reference differences (x100):
    contemporaneous (scn_Y - base_Y) and fixed-base (scn_Y - base_0), interpolated to annual (0 at
    base_year). Writes the 8-sector per-zone shock CSV at p.erosion_shock_output_path (same columns
    as the static path / carbon / pollination: ENDW, ACTS, REG, scenario, year, shock_pct,
    shock_pct_contemp, shock_pct_fixedbase).

    Caller sets on p: scenario_lulc_paths (incl. the base scenario), seals_years (anchor years),
    erosion_shock_base_year, erosion_shock_end_year, erosion_shock_output_path; erosion_prevention_dir
    (set by step 3); region_boundary_path (ee_r50_aez18 correspondence gpkg with ee_r50_aez18_id,
    aez18_id, gtapv7_r50_label); erosion_yield_stack_path, erosion_area_stack_path,
    erosion_bandmap_csv_path, erosion_elasticity_csv_path; base scenario via
    erosion_shock_base_scenario (default baseline_ignore_damages).
    """
    if not p.run_this:
        return
    import numpy as np, pandas as pd, rioxarray as rxr, geopandas as gpd
    from rasterio.enums import Resampling
    from rasterio.features import rasterize as rio_rasterize
    from global_invest.erosion import erosion_functions as ef

    base_year = int(p.erosion_shock_base_year); end_year = int(p.erosion_shock_end_year)
    base_scn = getattr(p, 'erosion_shock_base_scenario', 'baseline_ignore_damages')
    fallback_elast = float(getattr(p, 'erosion_elasticity_fallback', 0.08))
    sectors = getattr(p, 'erosion_shock_acts', EROSION_SECTORS)

    yield_stack = p.get_path(p.erosion_yield_stack_path)
    area_stack = p.get_path(p.erosion_area_stack_path)
    bandmap = pd.read_csv(p.get_path(p.erosion_bandmap_csv_path))
    bcol = next(c for c in bandmap.columns if 'band' in c.lower())
    crcol = next(c for c in bandmap.columns if c.lower() in ('crop', 'crop_name', 'name'))
    elast_map = ef.load_erosion_elasticity_map(p.get_path(p.erosion_elasticity_csv_path))
    if not getattr(p, 'region_boundary_path', None):
        p.region_boundary_path = p.get_path('gtap_invest/region_boundaries/ee_r50_aez18_correspondence.gpkg')
    zones = gpd.read_file(p.get_path(p.region_boundary_path))
    zid_col = next(c for c in zones.columns if c.lower() == 'ee_r50_aez18_id')
    aez_col = next(c for c in zones.columns if c.lower() == 'aez18_id')
    reg_col = next(c for c in zones.columns if c.lower() == 'gtapv7_r50_label')
    labels = {int(r[zid_col]): ('AEZ%d' % int(r[aez_col]), r[reg_col]) for _, r in zones.iterrows()}

    anchor_years = sorted(int(y) for y in getattr(p, 'seals_years', []) if int(y) > base_year) or [end_year]
    scenarios = [s for s in p.scenario_lulc_paths if s != base_scn]

    # Precompute ONCE (all ps_combined rasters share the analysis grid): rasterize the zones and reproject
    # each SPAM crop's production to that grid, plus its zone totals (ps-independent, so constant across
    # scenarios). zone_level then only reads ps and does the ps-weighted bincount -- no per-(scenario,year)
    # SPAM reproject or zone re-rasterize.
    def _ps_path(scn, yr): return os.path.join(p.erosion_prevention_dir, 'ps_combined_%s_%d.tif' % (scn, yr))
    _ref = rxr.open_rasterio(_ps_path(base_scn, anchor_years[0]), masked=True).squeeze()   # any ps: shared grid
    zr = zones.to_crs(_ref.rio.crs)
    zone_id = rio_rasterize([(g, int(z)) for g, z in zip(zr.geometry, zr[zid_col])],
                            out_shape=_ref.shape, transform=_ref.rio.transform(), fill=0, dtype='int32')
    max_id = int(zone_id.max())
    dy = rxr.open_rasterio(yield_stack, masked=True); da = rxr.open_rasterio(area_stack, masked=True)
    nb = dy.sizes.get('band', 1)
    crop_prod = []                     # [(production_array float64, elasticity)] per SPAM crop, on the grid
    tot = np.zeros(max_id + 1)         # total production per zone (ps-independent -> constant)
    for _, r in bandmap.iterrows():
        b = int(r[bcol])
        if b < 1 or b > nb:
            continue
        elast = ef.get_erosion_elasticity(str(r[crcol]).strip().lower(), elast_map, fallback_elast)
        y = dy.sel(band=b).squeeze().rio.reproject_match(_ref, resampling=Resampling.average).fillna(0.0)
        ha = da.sel(band=b).squeeze().clip(min=0).fillna(0).rio.reproject_match(_ref, resampling=Resampling.sum).fillna(0.0)
        prod = (y * ha).values.astype('float64')
        crop_prod.append((prod, elast))
        m = np.isfinite(prod) & (zone_id > 0)
        tot += np.bincount(zone_id[m], weights=prod[m], minlength=max_id + 1)

    def zone_level(scn, yr):
        """per-ee_r50_aez18 erosion protection level for one scenario x year (pd.Series keyed by zid)."""
        ps_arr = np.clip(np.nan_to_num(rxr.open_rasterio(_ps_path(scn, yr), masked=True).squeeze().values), 0.0, 1.0)
        protel = np.zeros(max_id + 1)
        for prod, elast in crop_prod:
            prot = ps_arr * prod
            mp = np.isfinite(prot) & (zone_id > 0)
            protel += np.bincount(zone_id[mp], weights=prot[mp] * elast, minlength=max_id + 1)
        with np.errstate(invalid='ignore', divide='ignore'):
            lvl = np.where(tot > 0, np.clip(protel / tot, 0.0, 1.0), np.nan)
        return pd.Series({int(i): lvl[i] for i in range(1, max_id + 1) if tot[i] > 0})

    base_by_year = {y: zone_level(base_scn, y) for y in anchor_years}
    base_map = p.scenario_lulc_paths.get(base_scn, {})
    base_at_base = zone_level(base_scn, base_year) if base_year in base_map else None
    all_years = list(range(base_year, end_year + 1))

    rows = []
    for scn in scenarios:
        scn_by_year = {y: zone_level(scn, y) for y in anchor_years}
        zids = sorted(set().union(*[set(s.index) for s in scn_by_year.values()]))
        for zid in zids:
            if zid not in labels:
                continue
            endw, reg = labels[zid]
            # ABSOLUTE difference of the productivity-share level, x100 (percentage points)
            c_anchor = [100.0 * (scn_by_year[y].get(zid, np.nan) - base_by_year[y].get(zid, np.nan)) for y in anchor_years]
            annual_c = np.interp(all_years, [base_year] + anchor_years, [0.0] + c_anchor)
            if base_at_base is not None:
                f_anchor = [100.0 * (scn_by_year[y].get(zid, np.nan) - base_at_base.get(zid, np.nan)) for y in anchor_years]
                annual_f = np.interp(all_years, [base_year] + anchor_years, [0.0] + f_anchor)
            else:
                annual_f = [np.nan] * len(all_years)
            for yr, vc, vf in zip(all_years, annual_c, annual_f):
                for sector in sectors:
                    rows.append({'ENDW': endw, 'ACTS': sector, 'REG': reg, 'scenario': scn, 'year': yr,
                                 'shock_pct': vc, 'shock_pct_contemp': vc, 'shock_pct_fixedbase': vf})

    out = pd.DataFrame(rows)
    out.to_csv(p.erosion_shock_output_path, index=False)
    print('  erosion valuation (dynamic): %d rows, %d scenarios, %d anchor years -> %s'
          % (len(out), len(scenarios), len(anchor_years), p.erosion_shock_output_path))
    return True

"""Erosion-control ES-shock tasks (static + dynamic), on the add_<es>_tasks seam like carbon/pollination.

METHOD SOURCE: the prevention-share erosion valuation in `global_erosion_gep`. That is where the
science comes from -- the on-farm share AE/(AE+USLE), the upstream share, the union
1 - (1-onfarm)(1-upstream) that avoids double-counting, the restriction to cropland AND
severely-eroding pixels, the per-country SES-11 soil-loss-tolerance policy, and the aggregation
weighted by crop production with per-crop erosion-yield elasticities. The biophysics below that is
InVEST SDR plus pygeoprocessing's D8 routing.

What differs here: that version runs one static 2019 map, so this recomputes the whole chain PER
SCENARIO AND YEAR on the SEALS maps and aggregates to r50xAEZ rather than to countries -- and, because
scenarios exist here and not there, holds the severe-pixel set FIXED to the base scenario (a set that
moves between scenarios would make part of the shock a change in WHICH pixels are averaged rather than
a change in protection; see level_service_threshold).

STATIC (erosion_shock_static): read raw_dependencies/erosion_prevention_dependency.csv, subtract
the baseline reference, linearly ramp 0 -> the scenario value over the horizon, apply to the 8
erosion-affected crop sectors -> erosion_interpolated.csv. UNCAPPED here -- the cap is applied
later on the COMBINED value in build_combined_afeall_cc_es.

DYNAMIC (#26; erosion_sdr -> upstream -> exposure -> shock): recompute the shock from our SEALS
maps via InVEST SDR -> D8 upstream -> prevention shares -> per-zone crop-productivity shock, by THREE
methods reported side by side (A = 'damage', thresholded/area; B = 'service', threshold-free and
magnitude-weighted with a per-crop coefficient; B-thresholded = 'service_threshold', B restricted to a
FIXED severe-pixel set and the DEFAULT; see erosion_shock). add_erosion_tasks (erosion_initialize) dispatches static vs dynamic on p.dynamic_es.
"""
import os
import pandas as pd
import hazelbean as hb

from global_invest import utilities
from global_invest.erosion import erosion_functions as ef

# 8 crop sectors whose productivity depends on erosion control (sediment retention).
# SPAM2020 crop code -> GTAP crop sector. Lets method B report a DIFFERENT shock per sector instead of
# one zone number copied across all eight: each sector aggregates only its own crops, so a zone that
# erodes mainly under cereals sends that signal to GRO rather than to V_F.
#
# Built from the GTAP sector definitions in base_data/gtappy/aggregation_mappings/: PDR "Paddy rice",
# WHT "Wheat", GRO "Cereal grains nec", V_F "Vegetables, fruit, nuts", OSD "Oil seeds", C_B "Sugar cane,
# sugar beet", PFB "Plant-based fibers", OCR "Crops nec". These eight are IDENTICAL across the s65, c26
# and a24 aggregations (verified in gtapv7_s65_correspondence.csv, which carries all three label
# columns), so this map holds for our v12_s26_r50 runs and would survive a change of aggregation.
# CPC separates starchy roots and tubers (015) and pulses (017) from vegetables (012) and fruit (013),
# while GTAP's V_F names only the latter two, so those groups need placing: roots and tubers
# (cass/pota/swpo/yams/orts) go to V_F, pulses (bean/lent/cowp/pige/chic/opul) to OCR. This shifts
# shock between V_F and OCR only; every crop lands in some sector either way, so no total changes.
# Crops absent fall back to OCR. The map itself is the erosion_crop_to_sector row in
# es_parameters (shipped default; a consumer overrides p.erosion_crop_to_sector).
# SEALS7 cropland class, the cropland definition method A weights by.
CROPLAND_SEALS7_CLASS = 2
# The erosion -> yield bridge: the fraction of yield lost per unit of erosion exposure. Converting
# biophysical erosion into an economic productivity shock requires such a coefficient, so all three
# methods rest on one. Method A applies this flat value to its thresholded area share. The service
# methods read a
# per-crop coefficient from elasticity_crops_fao_revised.csv (see alpha_for in erosion_shock) and
# falls back here only when neither the crop nor its sector has a value.
EROSION_ALPHA = 0.08
# The SES-11 threshold policy and analysis frame (SES-11 = the erosion author's run-series tag;
# the 11 is the 11 t/ha/yr severe threshold, a standard tolerable-soil-loss benchmark -- the
# expansion of 'SES' is the author's naming, to confirm at submission): METHOD CONSTANTS defining
# the published science
# (provisional, the erosion author's to bless) -- in code so a change costs a reviewed commit, not
# an input/-copy edit. getattr hooks below allow a deliberate consumer override.
SES11_SEVERE_THRESHOLD_T_HA = 11.0
SES11_THRESHOLD_LOW_T_HA = 2.0
SES11_SMALL_COUNTRY_AREA_KM2 = 50_000
SES11_LOW_ELEVATION_MEAN_M = 250
EROSION_ANALYSIS_EPSG = 8857          # Equal Earth: area-true math for the exposure shares
EROSION_YIELD_COEFFICIENT_FALLBACK = 0.08   # the SAME erosion->yield bridge as EROSION_ALPHA


# ---------------------------------------------------------------------------
# DYNAMIC path (#26): recompute the shock from our SEALS maps instead of the frozen table. Reached only
# when 'erosion' is in p.dynamic_es. Heavy imports (natcap.invest.sdr, pygeoprocessing.routing) go
# inside the functions so module import stays light.
# ---------------------------------------------------------------------------


def publish_inputs(p):
    """Every task's first line: erosion's es_config row plus its es_parameters block (the SDR
    data references, the SES-11 threshold policy, the crop-sector export map, the blank
    erosion_gep_root machine key the configure_* functions read) and the results registry.
    Defaults layer throughout: anything the caller set wins."""
    utilities.hydrate_es_config(p, 'erosion', log=hb.log)
    utilities.hydrate_es_parameters(p, 'erosion', log=hb.log)
    # Derived from the DEM row (caller wins): configure_* in erosion_functions reads these, and
    # their own fallbacks point at the author's cluster layout -- without these lines a local
    # Section-A run would silently reach for MSI paths.
    if getattr(p, 'erosion_sdr_input_dir', None) is None:
        p.erosion_sdr_input_dir = os.path.dirname(p.erosion_dem_path)
    if getattr(p, 'erosion_elevation_path', None) is None:
        p.erosion_elevation_path = p.erosion_dem_path
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def erosion_sdr(p):
    """DYNAMIC step 1: per (scenario, anchor year), resample the SEALS map to the erosion analysis
    grid (p.modality: local -> 6.45 km reference grid, sc/msi -> native SEALS resolution) and run
    InVEST SDR. Outputs per map, in p.cur_dir/<scn>_<yr>/: usle_<scn>_<yr>.tif (actual erosion) and
    rkls_<scn>_<yr>.tif (potential/bare soil); avoided = rkls - usle is formed downstream.

    Caller sets on p: scenario_lulc_paths {scenario: {year: seals_lulc_path}}; erosion_dem_path,
    erosion_erosivity_path, erosion_erodibility_path, erosion_watersheds_path,
    erosion_biophysical_table_path (SEALS7 lucode -> usle_c/usle_p); erosion_analysis_grid_path
    (6.45 km reference raster; local only). SDR knobs via p.erosion_sdr_params (defaults below).
    """
    publish_inputs(p)
    # Published before the run_this guard: a skipped task (skip_existing with the dir already there)
    # still needs downstream to find these outputs, and a task body that returns early would leave
    # the attribute unset and fail the next task with an AttributeError.
    p.erosion_sdr_dir = p.cur_dir      # downstream tasks read usle_/rkls_/avoided_erosion_ from here
    if not p.run_this:
        return
    import glob
    import hazelbean as hb
    from natcap.invest.sdr import sdr

    # Build scenario_lulc_paths from a template if the caller didn't (mirrors carbon/pollination).
    # p.es_lulc_path_template uses {scenario} and {year}; include the base scenario for differencing.
    if not getattr(p, 'scenario_lulc_paths', None) and getattr(p, 'es_lulc_path_template', None):
        tmpl = p.es_lulc_path_template
        years = [int(y) for y in getattr(p, 'es_shock_years', [])]
        scens = list(getattr(p, 'es_shock_scenarios', []))
        base = utilities.required_base_scenario(p, 'erosion')
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
    # Repaired once per run and cached: SDR's report step unions these and GEOS raises on an invalid
    # ring, so a bad geometry kills the run AFTER the rasters are already computed.
    watersheds = os.path.join(p.cur_dir, 'watersheds_valid.gpkg')
    if not os.path.exists(watersheds):
        ef.repair_watersheds(p.get_path(p.erosion_watersheds_path), watersheds)
    # SDR matches the biophysical table's lucode against the LULC values, and our maps are SEALS7 while
    # the shipped table is keyed on ESA codes -- so re-key it (once) rather than matching nothing.
    biophysical = ef.build_seals7_biophysical_table(
        p.get_path(p.erosion_biophysical_table_path),
        os.path.join(p.cur_dir, 'biophysical_table_seals7.csv'))
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
    print('  erosion SDR: %d scenario x year maps (%s grid) -> usle_/rkls_ in %s'
          % (n, 'native SEALS' if native else '6.45 km', p.cur_dir))
    return True


def erosion_upstream(p):
    """DYNAMIC step 2: per (scenario, year), upstream prevention share = acc(avoided) / acc(rkls),
    D8 flow-accumulation of avoided-mass over potential-mass (the pixel-area weight cancels in the
    ratio). Recomputed per scenario because the upslope land cover changes. Reads the SDR outputs
    from p.erosion_sdr_dir; writes upstream_<scn>_<yr>.tif to p.cur_dir. Uses pygeoprocessing.routing
    (fill_pits -> flow_dir_d8 -> flow_accumulation_d8), verified equal to a hand-written D8.

    Caller sets on p: erosion_dem_path (aligned to the SDR grid here) + the step-1 outputs.
    """
    publish_inputs(p)
    # Published before the run_this guard, as in steps 1 and 3: a skipped task must still tell the
    # exposure task where its rasters are, or a resumed run dies on an AttributeError.
    p.erosion_upstream_dir = p.cur_dir
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
    print('  erosion upstream: %d maps -> upstream_<scn>_<yr>.tif in %s' % (n, p.cur_dir))
    return True


def erosion_exposure(p):
    """DYNAMIC step 3: per (scenario, year), turn the SDR outputs into the pixel fields the level
    functions consume, on the equal-area analysis grid.

    Reads usle/avoided (p.erosion_sdr_dir) and upstream (p.erosion_upstream_dir). On-farm PS =
    avoided/(avoided+usle), which is identically 1 - USLE/RKLS; combined = 1 - (1-onfarm)(1-upstream),
    the serial-filter union (a tonne must escape both on-site and downslope retention to be lost).
    Writes six rasters:
      ps_gated              combined, zeroed off severe pixels    -> B-thresholded (the default)
      ps_continuous         combined across all land              -> B
      rkls_grid             potential (bare-soil) erosion         -> the service methods' weight
      cropland_frac         SEALS cropland fraction               -> method A denominator
      severe_cropland_frac  the same, zeroed off severe pixels    -> method A numerator
      severe_mask           the severe gate itself, so a level function can restrict BOTH halves of
                            a ratio to it. Gating only the numerator measures how much severe erosion
                            a zone HAS rather than how well it is protected.

    No cropland restriction is applied here. It is deferred to the shock task, where SPAM production
    is zero off cropland and multiplies through every term, so non-cropland drops out on its own and
    a binary mask would add nothing.

    The severe threshold follows the per-country SES-11 policy (T=2 for small-area <50,000 km2 or
    low-elevation <250 m countries, else 11) when p.erosion_country_boundary_path (+ p.erosion_dem_path
    for the elevation rule) is set; otherwise flat p.erosion_severe_threshold_t_ha.
    """
    publish_inputs(p)
    # Published before the run_this guard, for the same reason as step 1: a skipped task must still
    # tell the shock task where its rasters are.
    p.erosion_exposure_dir = p.cur_dir
    if not p.run_this:
        return
    import numpy as np
    import rioxarray as rxr
    import pygeoprocessing as pgp
    from osgeo import gdal
    from rasterio.crs import CRS as rioCRS
    from rasterio.enums import Resampling
    from global_invest.erosion import erosion_functions as ef

    thresh_high = float(getattr(p, 'erosion_severe_threshold_t_ha', SES11_SEVERE_THRESHOLD_T_HA))
    analysis_crs = rioCRS.from_epsg(int(getattr(p, 'erosion_analysis_epsg', EROSION_ANALYSIS_EPSG)))

    def _to_grid_da(da, template=None):    # reproject an open DataArray to the equal-area analysis grid
        da = da.rio.reproject(analysis_crs, resampling=Resampling.average)
        return da if template is None else da.rio.reproject_match(template, resampling=Resampling.average)

    def _to_grid(path, template=None):
        return _to_grid_da(rxr.open_rasterio(path, masked=True).squeeze(), template)

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
                        thresh_low=float(getattr(p, 'erosion_threshold_low_t_ha', SES11_THRESHOLD_LOW_T_HA)),
                        small_area_km2=float(getattr(p, 'erosion_small_country_area_km2', SES11_SMALL_COUNTRY_AREA_KM2)),
                        low_elevation_mean_m=float(getattr(p, 'erosion_low_elevation_mean_m', SES11_LOW_ELEVATION_MEAN_M)))
                else:
                    thr = thresh_high        # flat fallback when no country boundary is provided
                p._erosion_threshold_raster = thr
            mask = usle_v > thr              # severe pixels (per-country T); cropland from SPAM in the shock task
            with np.errstate(invalid='ignore', divide='ignore'):
                onfarm = np.where(mask & (avoided_v + usle_v > 0), avoided_v / (avoided_v + usle_v), 0.0)
            combined = np.where(mask, 1.0 - (1.0 - onfarm) * (1.0 - ups_v), 0.0)

            # Method B needs the same prevention WITHOUT the severe gate. avoided/(avoided+usle) is
            # identically avoided/rkls = 1 - USLE/RKLS, so this is the continuous prevention fraction the
            # A/B framing calls for; rkls is kept as B's magnitude weight, so that preventing most of a
            # negligible erosion rate earns almost no credit.
            with np.errstate(invalid='ignore', divide='ignore'):
                onfarm_cont = np.where(avoided_v + usle_v > 0, avoided_v / (avoided_v + usle_v), 0.0)
            continuous = 1.0 - (1.0 - onfarm_cont) * (1.0 - ups_v)
            rkls_v = avoided_v + usle_v      # potential (bare-soil) erosion

            # Method A weights by CROPLAND AREA (SEALS7 class 2), not SPAM production: p_crop is the
            # severe share of a zone's cropland. Averaging a 0/1 cropland mask onto the analysis grid
            # gives the cropland fraction per cell, and the equal-area grid makes cell area cancel.
            #
            # Done BLOCK-WISE, never in memory: a global 300 m SEALS map is ~8.4e9 pixels, so building
            # the mask as a float32 array would need ~34 GB and gets the run OOM-killed. raster_calculator
            # streams it by block to a compressed byte raster, then the average-resample coarsens it.
            lulc_native = p.get_path(by_year[year])
            crop_mask = os.path.join(p.cur_dir, 'cropland_mask_%s.tif' % suffix)
            if not os.path.exists(crop_mask):
                _nodata = pgp.get_raster_info(lulc_native)['nodata'][0]
                pgp.raster_calculator(
                    [(lulc_native, 1)],
                    lambda a: (a == CROPLAND_SEALS7_CLASS).astype('uint8'),
                    crop_mask, gdal.GDT_Byte, 255,
                    raster_driver_creation_tuple=('GTIFF', (
                        'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=DEFLATE', 'PREDICTOR=2')))
            cropfrac = np.nan_to_num(_to_grid(crop_mask, usle).values)

            tr = usle.rio.transform(); px = usle.rio.resolution()

            def _write(arr, name):
                pgp.numpy_array_to_raster(arr.astype('float32'), -9999.0, (px[0], px[1]),
                                          (tr.c, tr.f), usle.rio.crs.to_wkt(),
                                          os.path.join(p.cur_dir, '%s_%s.tif' % (name, suffix)))

            _write(combined, 'ps_gated')            # threshold-gated (original candidate)
            _write(continuous, 'ps_continuous')        # threshold-free (method B)
            _write(rkls_v, 'rkls_grid')                # method B magnitude weight
            _write(cropfrac, 'cropland_frac')          # method A denominator
            _write(np.where(mask, cropfrac, 0.0), 'severe_cropland_frac')   # method A numerator

            # The severe gate itself, so a level function can restrict BOTH halves of a ratio to it.
            # A severe pixel can legitimately have zero protection, so this cannot be recovered by
            # testing ps_gated > 0.
            _write(mask.astype('float32'), 'severe_mask')
            n += 1
    per_country = getattr(p, 'erosion_country_boundary_path', None) is not None
    print('  erosion prevention: %d maps -> ps_gated_ on EPSG:%d (severe T=%s)'
          % (n, analysis_crs.to_epsg(), 'per-country 11/2' if per_country else '%.1f flat' % thresh_high))
    return True


def erosion_shock(p):
    """DYNAMIC step 4: per-ee_r50_aez18 crop-productivity LEVELS by three methods, reported side by side.

    All share the SDR front-end (USLE, RKLS, avoided) and all bridge erosion to yield with the same
    coefficient, so they differ only in how erosion exposure is measured:
      A ("damage")                    level = -100 * alpha * p_crop, where p_crop is the severe share
        of the zone's cropland AREA and severe = USLE > the per-country T (2/11). Thresholded, binary,
        on-farm only, one flat alpha, and necessarily uniform across erosion_shock_acts.
      B ("service", threshold-free)   level = +100 * mean over crops of alpha_crop * the prevention
        share, prevention = prevented tonnes / potential tonnes including the upstream D8 term.
        Continuous and per-crop, but composed across ALL land, which saturates it: the union
        1-(1-onfarm)(1-upstream) sits near 1 over the ~98% of land that is not severely eroding, so
        the level reaches a median of 0.9988 with about half of pixels pinned at the ceiling, where
        no improvement in land cover can register.
      B-thresholded ("service_threshold", the DEFAULT)   B confined to SEVERE pixels, with that set
        taken from the base scenario and held FIXED, so the shock measures protection change and not a
        change of population. Verified in-task on the full ZAF run: min exactly 0.0, max +1.30%, no
        negative outliers, 63 zone-erosion_shock_acts responding at 2050 with a mean of +0.157% -- roughly 2.5x
        unthresholded B, which tops out at +0.52% on the same run. A scenario-VARYING set put -18%
        into a paddy-rice zone; fixing it removed that entirely. Matches how the published account of
        this method builds it.
    A is signed negative (damage borne) and B positive (protection delivered), but BOTH increase with
    better land condition, so they are positively correlated by construction and neither is a sign
    flip of the other. They are differently shaped functions of the erosion field, not offsets of one
    another, so their difference does not cancel.
    A fourth PRESERVED level (a prevention share behind A's severe gate, weighted by production alone)
    is emitted for comparison and never fed to GTAP. Its numerator is gated while its denominator is
    not, which makes it track erosion PREVALENCE rather than protection, and inverts its orientation.
    p.erosion_method ('damage'|'service'|'service_threshold', default 'service_threshold') selects
    which becomes shock_pct, the column GTAP consumes.

    Each level is differenced ABSOLUTELY against the contemporaneous baseline (the level is already a %
    of crop productivity) and ramped 0 at es_shock_base_year through the anchors. Writes the 8-sector per-zone
    CSV at p.erosion_shock_output_path: the shared ENDW, ACTS, REG, scenario, year, shock_pct,
    shock_pct_contemp, shock_pct_fixedbase plus shock_pct_damage, shock_pct_service and
    shock_pct_service_threshold.

    Caller sets on p: scenario_lulc_paths (incl. the base scenario), es_shock_years (anchors),
    es_shock_base_year, es_shock_end_year; erosion_exposure_dir (set by step 3);
    region_boundary_path (ee_r50_aez18 correspondence gpkg with ee_r50_aez18_id, aez18_id,
    gtapv7_r50_label); erosion_yield_stack_path, erosion_area_stack_path, erosion_bandmap_csv_path,
    erosion_elasticity_csv_path; base scenario via es_shock_base_scenario. Optional: erosion_alpha,
    erosion_method.
    """
    publish_inputs(p)
    # Default into the es_shocks parent dir. Runtime, not build time: p.es_shock_dir is
    # published by that task, which ProjectFlow runs before this one.
    if not getattr(p, 'erosion_shock_output_path', None):
        p.erosion_shock_output_path = os.path.join(getattr(p, 'es_shock_dir', None) or p.project_dir, 'erosion_interpolated.csv')
    if not p.run_this:
        return
    import numpy as np, pandas as pd, rioxarray as rxr, geopandas as gpd
    from rasterio.enums import Resampling
    from rasterio.features import rasterize as rio_rasterize
    from global_invest.erosion import erosion_functions as ef

    es_shock_base_year = int(p.es_shock_base_year); es_shock_end_year = int(p.es_shock_end_year)
    base_scenario = utilities.required_base_scenario(p, 'erosion')
    fallback_coef = float(getattr(p, 'erosion_yield_coefficient_fallback', EROSION_YIELD_COEFFICIENT_FALLBACK))
    alpha = float(getattr(p, 'erosion_alpha', EROSION_ALPHA))
    # Method B lets the erosion->yield coefficient vary by crop; A applies the flat alpha to all.
    # SOURCE = elasticity_crops_fao_revised.csv (already loaded above as coef_map). Despite the column
    # name, that table holds erosion-to-yield sensitivities, not price responses: its references are all
    # soil-erosion-and-yield literature (Lal 1998, Borrelli 2020, Panagos 2018, Oldfield 2019) and its
    # categories are a qualitative ranking rather than estimated coefficients. It is the crop-specific
    # version of alpha: yield lost per unit of erosion exposure (greens 0.5, cereals 0.3, default 0.08).
    # p.erosion_alpha_by_crop overrides per SPAM code.
    alpha_by_crop = dict(getattr(p, 'erosion_alpha_by_crop', {}) or {})

    # Six SPAM codes (grou, ocer, orts, pige, rest, vege) have no counterpart in the table -- they are
    # n.e.c. aggregates FAO does not carry. Falling back to the flat alpha would give "other cereals"
    # 0.08 when every named cereal in the table is 0.30, and the table-wide mean would give it 0.163;
    # its own SECTOR's mean (0.30, from barley/maize/millet) is the better estimate. So an unmatched
    # crop inherits the mean of the crops that DID match in its GTAP sector, and only a sector with no
    # matches at all falls back to the flat alpha.
    _MISS = object()

    def _table_alpha(crop):
        for key in [crop] + list(ef.SPAM_ALIAS_MAP.get(crop, [])):
            if str(key).strip().lower() in coef_map:
                return ef.get_erosion_yield_coefficient(crop, coef_map, alpha)
        return _MISS

    def alpha_for(crop):
        if crop in alpha_by_crop:
            return float(alpha_by_crop[crop])
        v = _matched.get(crop, _table_alpha(crop))
        return _sector_mean[crop_sector(crop)] if v is _MISS else v

    crop_to_sector = dict(p.erosion_crop_to_sector)

    def crop_sector(crop):
        return crop_to_sector.get(crop, 'OCR')     # unmapped SPAM codes fall to other crops

    erosion_shock_acts = tuple(p.erosion_shock_acts)

    yield_stack = p.get_path(p.erosion_yield_stack_path)
    area_stack = p.get_path(p.erosion_area_stack_path)
    bandmap = pd.read_csv(p.get_path(p.erosion_bandmap_csv_path))
    bcol = next(c for c in bandmap.columns if 'band' in c.lower())
    crcol = next(c for c in bandmap.columns if c.lower() in ('crop', 'crop_name', 'name'))
    coef_map = ef.load_erosion_yield_coefficients(p.get_path(p.erosion_elasticity_csv_path))
    zones = gpd.read_file(p.get_path(p.region_boundary_path), engine='pyogrio')
    zid_col = next(c for c in zones.columns if c.lower() == 'ee_r50_aez18_id')
    aez_col = next(c for c in zones.columns if c.lower() == 'aez18_id')
    reg_col = next(c for c in zones.columns if c.lower() == 'gtapv7_r50_label')
    labels = {int(r[zid_col]): ('AEZ%d' % int(r[aez_col]), r[reg_col]) for _, r in zones.iterrows()}

    anchor_years = sorted(int(y) for y in getattr(p, 'es_shock_years', []) if int(y) > es_shock_base_year) or [es_shock_end_year]
    # The base scenario is EMITTED too, not just used as the reference. Its rows are the self-difference
    # (base - base), so they are identically 0 -- which is what carbon and pollination already write. GTAP
    # is indifferent (a missing row and an explicit zero both mean "no shock"), but writing it keeps the
    # four services on one shape and makes "B_y == 0 for the ignore-dependencies baseline" a check that can
    # actually be run against the CSV rather than inferred from an absence.
    scenarios = list(p.scenario_lulc_paths)

    # Precompute ONCE (all ps_gated rasters share the analysis grid): rasterize the zones and reproject
    # each SPAM crop's production to that grid, plus its zone totals (ps-independent, so constant across
    # scenarios). zone_level then only reads ps and does the ps-weighted bincount -- no per-(scenario,year)
    # SPAM reproject or zone re-rasterize.
    def _ps_path(scn, yr): return os.path.join(p.erosion_exposure_dir, 'ps_gated_%s_%d.tif' % (scn, yr))
    _ref = rxr.open_rasterio(_ps_path(base_scenario, anchor_years[0]), masked=True).squeeze()   # any ps: shared grid
    zr = zones.to_crs(_ref.rio.crs)
    zone_id = rio_rasterize([(g, int(z)) for g, z in zip(zr.geometry, zr[zid_col])],
                            out_shape=_ref.shape, transform=_ref.rio.transform(), fill=0, dtype='int32')
    max_id = int(zone_id.max())
    dy = rxr.open_rasterio(yield_stack, masked=True); da = rxr.open_rasterio(area_stack, masked=True)
    nb = dy.sizes.get('band', 1)
    crop_prod = []                     # [(spam_crop, production_array float64, coefficient)] per SPAM crop, on the grid
    tot = np.zeros(max_id + 1)         # total production per zone (ps-independent -> constant)
    for _, r in bandmap.iterrows():
        b = int(r[bcol])
        if b < 1 or b > nb:
            continue
        elast = ef.get_erosion_yield_coefficient(str(r[crcol]).strip().lower(), coef_map, fallback_coef)
        y = dy.sel(band=b).squeeze().rio.reproject_match(_ref, resampling=Resampling.average).fillna(0.0)
        ha = da.sel(band=b).squeeze().clip(min=0).fillna(0).rio.reproject_match(_ref, resampling=Resampling.sum).fillna(0.0)
        prod = (y * ha).values.astype('float64')
        crop_prod.append((str(r[crcol]).strip().lower(), prod, elast))
        m = np.isfinite(prod) & (zone_id > 0)
        tot += np.bincount(zone_id[m], weights=prod[m], minlength=max_id + 1)

    _matched = {c: _table_alpha(c) for c in {cr for cr, _, _ in crop_prod}}
    _sector_mean = {}
    for s in erosion_shock_acts:
        vals = [v for c, v in _matched.items() if v is not _MISS and crop_sector(c) == s]
        _sector_mean[s] = sum(vals) / len(vals) if vals else alpha


    def _grid(name, scn, yr):
        path = os.path.join(p.erosion_exposure_dir, '%s_%s_%d.tif' % (name, scn, yr))
        return np.nan_to_num(rxr.open_rasterio(path, masked=True).squeeze().values)

    def _zonal(weights):
        """sum a per-pixel weight into zones -> array indexed by zone id."""
        m = np.isfinite(weights) & (zone_id > 0)
        return np.bincount(zone_id[m], weights=weights[m], minlength=max_id + 1)

    def _series(num, den):
        with np.errstate(invalid='ignore', divide='ignore'):
            lvl = np.where(den > 0, num / den, np.nan)
        return pd.Series({int(i): lvl[i] for i in range(1, max_id + 1) if den[i] > 0})

    def level_damage(scn, yr):
        """METHOD A ("damage") -- the documented GTAP method behind the paper's frozen
        numbers. p_crop = severe share of the zone's cropland AREA (USLE > the per-country T of 2/11,
        cropland = SEALS7 class 2); level = -100*alpha*p_crop. Binary threshold, flat alpha, no off-site
        routing. UNIFORM across the GTAP crop sectors by construction: it is measured from LAND COVER, which
        carries no crop detail, so A cannot distinguish wheat land from vegetable land."""
        p_crop = _series(_zonal(_grid('severe_cropland_frac', scn, yr)),
                         _zonal(_grid('cropland_frac', scn, yr)))
        lvl = -100.0 * alpha * p_crop
        return {s: lvl for s in erosion_shock_acts}

    def _service_level(ps, rkls):
        """Production-weighted prevention level per GTAP sector, shared by B and B-thresholded.

        Per crop, the prevention share is prevented tonnes over potential tonnes, so a pixel with
        negligible erosion cannot earn credit for preventing almost nothing. Each crop's share is
        bridged to yield by its OWN alpha and averaged across crops by production. The two callers
        differ only in the ps field and in whether rkls is restricted to severe pixels."""
        per_sector = {s: [np.zeros(max_id + 1), np.zeros(max_id + 1)] for s in erosion_shock_acts}
        all_num = np.zeros(max_id + 1)
        for crop, prod, _elast in crop_prod:
            potential = _zonal(rkls * prod)
            prevented = _zonal(ps * rkls * prod)
            at_stake = potential > 0
            share = np.where(at_stake, prevented / np.where(at_stake, potential, 1.0), 0.0)
            # A zone-crop with nothing at stake carries NO WEIGHT, rather than scoring a share of 0.
            # Zero would read as "no protection delivered" when it means "no erosion to protect
            # against", so under a severe threshold a zone that stops eroding between baseline and
            # scenario would look like its protection collapsed. That produced a -28% shock for paddy
            # rice in South Africa, a zone with almost no rice and no severe pixels in the scenario.
            # Dropping it from both sides of the ratio lets it fall back to the all-crop level below.
            weight = _zonal(prod) * at_stake
            contribution = weight * alpha_for(crop) * share
            all_num += contribution
            sector = crop_sector(crop)
            if sector in per_sector:
                per_sector[sector][0] += contribution
                per_sector[sector][1] += weight
        # A zone growing none of a sector's crops has no sector-specific signal, so fall back to the
        # all-crop level there rather than emitting NaN into the GTAP shock.
        all_crop = 100.0 * _series(all_num, tot)
        out = {}
        for s, (num, den) in per_sector.items():
            lvl = 100.0 * _series(num, den)
            out[s] = lvl.reindex(all_crop.index).fillna(all_crop)
        return out

    def level_service(scn, yr):
        """METHOD B ("service") -- threshold-free. Credits the continuous prevention share across ALL
        land, which saturates it (median 0.9988, about half of pixels pinned at full protection), so
        it is reported for comparison and no longer feeds GTAP. Signed positive as a service
        delivered, but it still INCREASES with better land condition exactly as A does."""
        return _service_level(np.clip(_grid('ps_continuous', scn, yr), 0.0, 1.0),
                              _grid('rkls_grid', scn, yr))

    def level_service_threshold(scn, yr):
        """METHOD B THRESHOLDED -- B confined to severely eroding pixels, with the severe set taken
        from the BASE scenario and held FIXED. THE DEFAULT (see p.erosion_method).

        Verified inside this task by the full ZAF pipeline run: shock_pct, the column
        build_combined_afeall consumes, comes out identical to shock_pct_service_threshold with a
        minimum of exactly 0.0 and a maximum of +1.30%, i.e. no negative outliers at all. At 2050,
        63 zone-erosion_shock_acts respond with a mean of +0.157%. Unthresholded B on the same run tops out at
        +0.52%, so the threshold carries roughly 2.5x the signal.

        Restricting to severe pixels is what stops B saturating, since the union sits near 1 across
        the ~98% of land that is not eroding. But a scenario-VARYING severe set makes the shock partly
        a change of population rather than of protection: the two levels then average over different
        pixels, and in a zone with only a handful of severe pixels one entering or leaving swings the
        average. That put -18% into a paddy-rice zone in South Africa. Holding the set fixed makes the
        difference measure protection alone, and also makes `potential` identical across scenarios,
        because RKLS carries no cover factor and so does not vary with land use.

        ps_continuous is read here rather than ps_gated: ps_gated is masked to each scenario's OWN
        severe set, which is exactly what is being held fixed. rkls carries the same fixed mask, so
        numerator and denominator are restricted together and this stays a prevention share where
        higher means better. Gating only the numerator would instead measure how much severe erosion a
        zone HAS. No cropland term is needed: every sum carries prod as a factor, so a pixel with no
        production contributes nothing regardless."""
        keep = _grid('severe_mask', base_scenario, yr) > 0.5
        return _service_level(np.where(keep, np.clip(_grid('ps_continuous', scn, yr), 0.0, 1.0), 0.0),
                              np.where(keep, _grid('rkls_grid', scn, yr), 0.0))

    LEVELS = {'damage': level_damage, 'service': level_service,
              'service_threshold': level_service_threshold}
    primary = str(p.erosion_method).lower()
    if primary not in ('damage', 'service', 'service_threshold'):
        raise ValueError("p.erosion_method must be 'damage' (the deck's Method A), 'service' "
                         "(Method B) or 'service_threshold' (B restricted to severely eroding "
                         "pixels before compositing), got %r. All three are computed and "
                         "reported side by side; this only selects which becomes shock_pct." % primary)

    base_map = p.scenario_lulc_paths.get(base_scenario, {})
    all_years = list(range(es_shock_base_year, es_shock_end_year + 1))
    anchors_x = [es_shock_base_year] + anchor_years

    def annual(scn_by_year, base_by_year, sector, zid):
        """ABSOLUTE difference of the % levels for one sector (the level IS a % of crop productivity, so
        differencing it gives the % productivity change), ramped 0 at es_shock_base_year through the anchors."""
        a = [scn_by_year[y][sector].get(zid, np.nan) - base_by_year[y][sector].get(zid, np.nan)
             for y in anchor_years]
        return np.interp(all_years, anchors_x, [0.0] + a)

    # one pass per method: {sector: level Series} at each anchor, for the baseline and every scenario
    by_method = {}
    for name, fn in LEVELS.items():
        base_by_year = {y: fn(base_scenario, y) for y in anchor_years}
        base_at_base = fn(base_scenario, es_shock_base_year) if es_shock_base_year in base_map else None
        by_method[name] = (base_by_year, base_at_base,
                           {scn: {y: fn(scn, y) for y in anchor_years} for scn in scenarios})

    rows = []
    for scn in scenarios:
        anchor_levels = by_method[primary][2][scn]
        zids = sorted(set().union(*[set(lv[erosion_shock_acts[0]].index) for lv in anchor_levels.values()]))
        for zid in zids:
            if zid not in labels:
                continue
            endw, reg = labels[zid]
            for sector in erosion_shock_acts:
                series = {name: annual(base_and_scn[2][scn], base_and_scn[0], sector, zid)
                          for name, base_and_scn in by_method.items()}
                base_by_year, base_at_base, scn_levels = by_method[primary]
                if base_at_base is not None:
                    f = [scn_levels[scn][y][sector].get(zid, np.nan) - base_at_base[sector].get(zid, np.nan)
                         for y in anchor_years]
                    annual_f = np.interp(all_years, anchors_x, [0.0] + f)
                else:
                    annual_f = [np.nan] * len(all_years)
                for i, yr in enumerate(all_years):
                    rows.append({'ENDW': endw, 'ACTS': sector, 'REG': reg, 'scenario': scn, 'year': yr,
                                 'shock_pct': series[primary][i],
                                 # Equal to shock_pct by construction here: erosion's level is already
                                 # a % of crop productivity, so the shock is an absolute difference with
                                 # no denominator to vary. Emitted anyway because carbon and pollination
                                 # carry the contemp/fixedbase pair and the viz gates a figure on both.
                                 'shock_pct_contemp': series[primary][i],
                                 'shock_pct_fixedbase': annual_f[i],
                                 'shock_pct_damage': series['damage'][i],
                                 'shock_pct_service': series['service'][i],
                                 'shock_pct_service_threshold': series['service_threshold'][i]})

    out = pd.DataFrame(rows)
    utilities.assert_shock_table_sound(out, scenarios, 'erosion')
    out.to_csv(p.erosion_shock_output_path, index=False)
    end = out[out['year'] == es_shock_end_year]
    print('  erosion shock (dynamic): %d rows, %d scenarios, %d anchors, alpha=%.3f, primary=%s'
          % (len(out), len(scenarios), len(anchor_years), alpha, primary.upper()))
    print('     mean shock @%d   A: %+.4f%%   B: %+.4f%%   B-thresholded: %+.4f%%'
          % (es_shock_end_year, end['shock_pct_damage'].mean(), end['shock_pct_service'].mean(),
             end['shock_pct_service_threshold'].mean()))
    return True

def erosion_shock_static(p):
    """Static per-scenario erosion shock -> 8 crop erosion_shock_acts, linear ramp 0->es_shock_end_year.

    Caller sets on p before calling: es_shock_scenarios, es_shock_base_year,
    es_shock_end_year, erosion_shock_output_path. Dependency csv defaults to
    input_dir/raw_dependencies/erosion_prevention_dependency.csv (override p.erosion_dependency_path);
    scenario->raw name via p.erosion_scenario_map (default: identity -- each scenario maps to its own
    name; a scenario the table labels differently is warned about loudly and skipped rather than
    silently zeroed, so set the map for those).
    """
    publish_inputs(p)
    # Default into the es_shocks parent dir. Runtime, not build time: p.es_shock_dir is
    # published by that task, which ProjectFlow runs before this one.
    if not getattr(p, 'erosion_shock_output_path', None):
        p.erosion_shock_output_path = os.path.join(getattr(p, 'es_shock_dir', None) or p.project_dir, 'erosion_interpolated.csv')
    if not p.run_this:
        return

    es_shock_base_year = int(p.es_shock_base_year)
    es_shock_end_year = int(p.es_shock_end_year)
    n_years = es_shock_end_year - es_shock_base_year
    erosion_scenario_map = getattr(p, 'erosion_scenario_map', {})
    es_shock_scenarios = list(p.es_shock_scenarios)
    erosion_shock_acts = tuple(p.erosion_shock_acts)   # GTAP crop sectors

    ero_path = getattr(p, 'erosion_dependency_path', None) or os.path.join(
        p.input_dir, 'raw_dependencies', 'erosion_prevention_dependency.csv')
    if not os.path.exists(ero_path):
        print('  erosion shock: dependency csv not found (%s) -- skipping' % ero_path)
        return

    df = ef.read_erosion_dependency(ero_path)
    # Resolve the configured base through the candidate mechanism (fatal if absent) -- the erosion
    # table spells the nature-off baseline 'baseline_ignore_damages' while the shared config may say
    # 'baseline_ignore_dependencies'; the two spellings are mutual aliases by default
    # (utilities.NATURE_OFF_SPELLINGS), so no consumer map is needed for this.
    base_scenario = utilities.required_base_scenario(p, 'erosion')
    raw_base = utilities.resolve_base_scenario(df['scenario'].values, erosion_scenario_map, base_scenario, 'erosion')
    base_vals = df[df['scenario'] == raw_base].set_index(
        ['aez18_id', 'gtapv7_r50_label'])['value'].astype(float).fillna(0)

    rows = []
    for our_scn in es_shock_scenarios:
        raw_scn = utilities.resolve_raw_scenario(df['scenario'].values, erosion_scenario_map, our_scn, 'erosion')
        if raw_scn is None:
            continue
        scn_vals = df[df['scenario'] == raw_scn].set_index(
            ['aez18_id', 'gtapv7_r50_label'])['value'].astype(float).fillna(0)
        common = scn_vals.index.intersection(base_vals.index)
        shock = scn_vals.loc[common] - base_vals.loc[common]
        for year in range(es_shock_base_year, es_shock_end_year + 1):
            frac = (year - es_shock_base_year) / n_years
            for (aez_id, reg), val in shock.items():
                endw = 'AEZ%d' % int(aez_id)
                for sector in erosion_shock_acts:
                    rows.append({'ENDW': endw, 'ACTS': sector, 'REG': reg,
                                 'scenario': our_scn, 'year': year, 'shock_pct': val * frac})

    out = pd.DataFrame(rows)
    utilities.assert_shock_table_sound(out, es_shock_scenarios, 'erosion')
    out.to_csv(p.erosion_shock_output_path, index=False)
    nz = out[(out['year'] == es_shock_end_year) & (out['shock_pct'] != 0)] if len(out) else out
    print('  erosion shock: %d rows, %d scenarios, %d nonzero @%d (static, uncapped) -> %s'
          % (len(out), out['scenario'].nunique() if len(out) else 0, len(nz), es_shock_end_year,
             p.erosion_shock_output_path))
    return True


# =============================================================================
# GEP valuation tasks (folded from global_erosion_gep): InVEST SDR -> prevention
# shares -> per-country GEP -> maps/figures. The ES-shock tasks above and this
# valuation are separate consumers of the same erosion science.
# =============================================================================


def invest_sdr(p):
    """Section A: run InVEST SDR to produce the erosion rasters (USLE, avoided erosion) that
    Section B consumes. ProjectFlow-idiomatic: outputs default into THIS task's dir (caller may
    override erosion_sdr_output_dir), the paths Section B reads are PUBLISHED on p before the
    run_this guard (so a skipped rerun still feeds downstream), and the builders register this
    with skip_existing=1 -- delete the task dir to force a rerun.
    """
    publish_inputs(p)
    if not getattr(p, 'erosion_sdr_output_dir', None):
        p.erosion_sdr_output_dir = p.cur_dir
    if not getattr(p, 'erosion_watersheds_sanitized_path', None):
        p.erosion_watersheds_sanitized_path = os.path.join(p.erosion_sdr_output_dir, 'wshed_sanitized.gpkg')
    # Publish Section A's outputs under the names configure_prevention_shares reads off p --
    # explicit task chaining instead of the source repo's copy-between-stage-dirs convention
    # (whose input defaults even carry a different date suffix than the outputs).
    _sfx = getattr(p, 'erosion_sdr_results_suffix', '2019_revised_dec_14')
    p.erosion_usle_path = os.path.join(p.erosion_sdr_output_dir, f'usle_{_sfx}.tif')
    p.erosion_avoided_erosion_path = os.path.join(p.erosion_sdr_output_dir, f'avoided_erosion_{_sfx}.tif')
    if not p.run_this:
        return
    ef.configure_sdr(p)
    p.erosion_sdr_args, p.erosion_sdr_file_registry = ef.run_invest_sdr()
    return True


def prevention_shares(p):
    """Section B: combine on-farm (AE/(AE+USLE)) and upstream prevention shares into the
    union-of-protection PS_combined, then country-crop protected production and the GEP valuation
    (onfarm / upstream / combined) -> integrated_country_gep.csv + the PS rasters the maps task
    reads. ProjectFlow-idiomatic: outputs default into THIS task's dir via erosion_gep_output_dir
    (the same attr configure_maps chains on), USLE/avoided arrive from invest_sdr's
    published attrs, and the registered result is the skip check (like every gep_calculation).
    """
    publish_inputs(p)
    if not getattr(p, 'erosion_gep_output_dir', None):
        p.erosion_gep_output_dir = p.cur_dir
    service_results = p.results.setdefault('erosion', {})
    service_results['integrated_country_gep'] = os.path.join(
        p.erosion_gep_output_dir, "integrated_country_gep.csv")
    if not p.run_this:
        return
    if hb.path_all_exist(list(service_results.values())):
        hb.log("integrated_country_gep.csv already exists. Skipping prevention-share calculation for erosion.")
        return True
    ef.configure_prevention_shares(p)
    ef.integrate_and_write()
    return True


def maps_and_figures(p):
    """Section C: publication-ready choropleths, raster previews and charts from Section B's
    outputs (found via the shared erosion_gep_output_dir attr). Figures default into THIS task's
    dir; skip_existing at registration."""
    publish_inputs(p)
    if not getattr(p, 'erosion_figures_dir', None):
        p.erosion_figures_dir = p.cur_dir
    if not p.run_this:
        return
    ef.configure_maps(p)
    ef.generate_all_maps_and_figures()
    return True

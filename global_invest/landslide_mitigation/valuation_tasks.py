"""
valuation_tasks.py
"""
import os
import json
import numpy as np
import pandas as pd
import glob
import geopandas as gpd
from osgeo import gdal
import pygeoprocessing as pygeo


def tile_zones(p):
    """Generate tile zones for parallel prediction, filtered to
    land-containing tiles only. Tiles are defined on the EASE-Grid
    reference's exact dimensions/transform via p.gaez_path 
    """
    if not p.run_this:
        return p

    blocks_list_path = os.path.join(p.cur_dir, 'blocks_list.csv')

    if os.path.exists(blocks_list_path):
        p.L.info('Blocks list already exists, loading from file...')
        blocks_df = pd.read_csv(blocks_list_path, header=None)
        blocks_df.columns = ['col_offset', 'row_offset', 'n_cols', 'n_rows']
        blocks_list = blocks_df.values.tolist()
        p.L.info(f'Loaded {len(blocks_list)} tiles from existing blocks_list.csv')
    else:
        p.L.info('Creating tile list from GAEZ zones raster (real land data, '
               'on the exact EASE-Grid reference grid)...')

        ds = gdal.Open(p.gaez_path)
        n_cols = ds.RasterXSize
        n_rows = ds.RasterYSize
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()

        p.tile_size = getattr(p, 'processing_resolution', 2000)

        blocks_list = []
        for row_offset in range(0, n_rows, p.tile_size):
            for col_offset in range(0, n_cols, p.tile_size):
                actual_n_cols = min(p.tile_size, n_cols - col_offset)
                actual_n_rows = min(p.tile_size, n_rows - row_offset)

                tile = band.ReadAsArray(col_offset, row_offset, actual_n_cols, actual_n_rows)
                land_mask = np.isfinite(tile)
                if nodata is not None:
                    land_mask &= (tile != nodata)

                if land_mask.sum() > 0:
                    blocks_list.append([col_offset, row_offset, actual_n_cols, actual_n_rows])

        ds = None

        blocks_df = pd.DataFrame(blocks_list, columns=['col_offset', 'row_offset', 'n_cols', 'n_rows'])
        blocks_df.to_csv(blocks_list_path, index=False, header=False)
        p.L.info(f'Created {len(blocks_list)} land tiles (filtered ocean tiles)')
        p.L.info(f'Blocks list saved to: {blocks_list_path}')

    p.iterator_replacements = {
        'tile_col_offset': [block[0] for block in blocks_list],
        'tile_row_offset': [block[1] for block in blocks_list],
        'tile_n_cols': [block[2] for block in blocks_list],
        'tile_n_rows': [block[3] for block in blocks_list],
        'cur_dir_parent_dir': [
            os.path.join(p.cur_dir, f'{block[1]}_{block[0]}')
            for block in blocks_list
        ]
    }
    p.L.info(f'Set up iterator replacements for {len(blocks_list)} tiles.')
    return p


def predict_landslides_scenarios(p):
    """Per tile, per scenario: apply the calibrated logistic
    (alpha_corrected + beta_si*SI + beta_rain*rain) to produce a hazard
    probability GeoTIFF. Written to disk per tile x scenario.
    """
    if not p.run_this:
        return p

    hazard_model_coefficients_path = os.path.join(p.modeling_dir, 'hazard_model_coefficients.json')
    with open(hazard_model_coefficients_path) as f:
        coef = json.load(f)
    alpha = coef['alpha_corrected']
    beta_si = coef['beta_si']
    beta_rain = coef['beta_rain']

    ref_info = pygeo.get_raster_info(p.ease_grid_reference_path)
    gt = ref_info['geotransform']
    proj = ref_info['projection_wkt']

    col_off, row_off = p.tile_col_offset, p.tile_row_offset
    n_cols, n_rows = p.tile_n_cols, p.tile_n_rows
    tile_gt = (
        gt[0] + col_off * gt[1], gt[1], 0,
        gt[3] + row_off * gt[5], 0, gt[5],
    )

    # Rain: NOT scenario-varying, same prediction-year raster for both.
    prediction_year = p.prediction_years[0]  # NOTE: assumes single prediction year
    rain_path = os.path.join(
        p.input_data_dir, 'era5_land', f'era5_max_daily_mm_{prediction_year}.tif'
    )
    rain_ds = gdal.Open(rain_path)
    rain_band = rain_ds.GetRasterBand(1)
    rain_nodata = rain_band.GetNoDataValue()
    rain_tile = rain_band.ReadAsArray(col_off, row_off, n_cols, n_rows)
    rain_ds = None

    for scenario_name, si_paths_by_year in p.si_paths.items():
        si_path = si_paths_by_year.get(prediction_year)
        if si_path is None:
            p.L.warning(f'{scenario_name}: no SI for prediction year {prediction_year}, skipping tile.')
            continue

        si_ds = gdal.Open(si_path)
        si_band = si_ds.GetRasterBand(1)
        si_nodata = si_band.GetNoDataValue()
        si_tile = si_band.ReadAsArray(col_off, row_off, n_cols, n_rows)
        si_ds = None

        valid = np.ones(si_tile.shape, dtype=bool)
        if si_nodata is not None:
            valid &= (si_tile != si_nodata)
        if rain_nodata is not None:
            valid &= (rain_tile != rain_nodata)

        logodds = alpha + beta_si * si_tile + beta_rain * rain_tile
        prob = 1 / (1 + np.exp(-logodds))
        prob_out = np.where(valid, prob, -9999.0).astype(np.float32)

        out_path = os.path.join(p.cur_dir, f'hazard_prob_{scenario_name}_{prediction_year}.tif')
        driver = gdal.GetDriverByName('GTiff')
        ds_out = driver.Create(
            out_path, n_cols, n_rows, 1, gdal.GDT_Float32,
            options=['TILED=YES', 'COMPRESS=LZW'],
        )
        ds_out.SetGeoTransform(tile_gt)
        ds_out.SetProjection(proj)
        band_out = ds_out.GetRasterBand(1)
        band_out.WriteArray(prob_out)
        band_out.SetNoDataValue(-9999.0)
        ds_out = None

        p.L.info(f'Tile ({row_off},{col_off}) {scenario_name}: {out_path}')

    return p



def predict_mortality_scenarios(p):
    """
    Per tile, per scenario: predict expected deaths per pixel by combining:
 
        P(landslide | hazard model)
        *
        P(fatality > 0 | landslide, severity covariates)
        *
        E(fatalities | fatal, severity covariates)
 
    Part B expectation uses Duan's smearing correction.
    Writes mortality GeoTIFFs per tile x scenario.
    """
    if not p.run_this:
        return p
 
    severity_path = os.path.join(p.modeling_dir, 'severity_model_coefficients.json')
    with open(severity_path) as f:
        severity = json.load(f)
 
    part_a = severity['part_a_params']
    part_b = severity['part_b_params']
    smearing = severity['smearing_factor']
 
    ref_info = pygeo.get_raster_info(p.ease_grid_reference_path)
    gt = ref_info['geotransform']
    proj = ref_info['projection_wkt']
 
    col_off = p.tile_col_offset
    row_off = p.tile_row_offset
    n_cols = p.tile_n_cols
    n_rows = p.tile_n_rows
 
    tile_gt = (
        gt[0] + col_off * gt[1], gt[1], 0,
        gt[3] + row_off * gt[5], 0, gt[5],
    )
 
    def read_tile(path, already_tiled=False):
        ds = gdal.Open(path)
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        xsize, ysize = band.XSize, band.YSize
 
        if already_tiled or (xsize == n_cols and ysize == n_rows):
            arr = band.ReadAsArray()
        else:
            arr = band.ReadAsArray(col_off, row_off, n_cols, n_rows)
 
        ds = None
        return arr, nodata
 
    prediction_year = p.prediction_years[0]
 
    population_path = os.path.join(
        p.input_data_dir, 'landscan_1km', f'landscan_{prediction_year}_1km.tif'
    )
    rain_path = os.path.join(
        p.input_data_dir, 'era5_land', f'era5_max_daily_mm_{prediction_year}.tif'
    )
    slope_path = p.slope_path
    road_path = p.road_density_path

    population, pop_nd = read_tile(population_path)
    rain, rain_nd = read_tile(rain_path)
    slope, slope_nd = read_tile(slope_path)
    road, road_nd = read_tile(road_path)
 
    population_log1p = np.log1p(np.maximum(population, 0))
 
    valid = np.ones(population.shape, dtype=bool)
    for arr, nd in [(population, pop_nd), (rain, rain_nd), (slope, slope_nd), (road, road_nd)]:
        if nd is not None:
            valid &= (arr != nd)
 
    logit_a = (
        part_a['Intercept']
        + part_a['population_log1p'] * population_log1p
        + part_a['rain_max_daily'] * rain
        + part_a['slope_degrees'] * slope
        + part_a['road_density'] * road
    )
    p_fatal = 1 / (1 + np.exp(-logit_a))
 
    log_fatalities = (
        part_b['Intercept']
        + part_b['population_log1p'] * population_log1p
        + part_b['rain_max_daily'] * rain
        + part_b['slope_degrees'] * slope
        + part_b['road_density'] * road
    )
    expected_fatalities_if_fatal = np.exp(log_fatalities) * smearing
 
    severity_expectation = p_fatal * expected_fatalities_if_fatal
 
    for scenario_name in p.si_paths.keys():
        hazard_dir = os.path.join(os.path.dirname(p.cur_dir), 'predict_landslides_scenarios')
        hazard_path = os.path.join(hazard_dir, f'hazard_prob_{scenario_name}_{prediction_year}.tif')
 
        if not os.path.exists(hazard_path):
            p.L.warning(f'Missing hazard raster: {hazard_path}')
            continue
 
        hazard, hazard_nd = read_tile(hazard_path, already_tiled=True)
 
        valid_hazard = valid.copy()
        if hazard_nd is not None:
            valid_hazard &= (hazard != hazard_nd)
 
        deaths = hazard * severity_expectation
        deaths_out = np.where(valid_hazard, deaths, -9999.0).astype(np.float32)
 
        out_path = os.path.join(p.cur_dir, f'expected_deaths_{scenario_name}_{prediction_year}.tif')
        driver = gdal.GetDriverByName('GTiff')
        ds_out = driver.Create(
            out_path, n_cols, n_rows, 1, gdal.GDT_Float32,
            options=['TILED=YES', 'COMPRESS=LZW'],
        )
        ds_out.SetGeoTransform(tile_gt)
        ds_out.SetProjection(proj)
        band = ds_out.GetRasterBand(1)
        band.WriteArray(deaths_out)
        band.SetNoDataValue(-9999.0)
        ds_out = None
 
        p.L.info(f'Tile ({row_off},{col_off}) {scenario_name}: {out_path}')
 
    return p
 
 
def stitch_tiles(p):
    """
    Stitch tile-level hazard and mortality predictions into global rasters.
    """
    if not p.run_this:
        return p
 
    blocks_list_path = os.path.join(p.tile_zones_dir, 'blocks_list.csv')
    blocks_df = pd.read_csv(blocks_list_path, header=None)
    blocks_df.columns = ['col_offset', 'row_offset', 'n_cols', 'n_rows']
    blocks_list = blocks_df.values.tolist()
 
    ref_ds = gdal.Open(p.gaez_path)
    n_cols_full = ref_ds.RasterXSize
    n_rows_full = ref_ds.RasterYSize
    gt = ref_ds.GetGeoTransform()
    proj = ref_ds.GetProjection()
    ref_ds = None
 
    driver = gdal.GetDriverByName('GTiff')
    NODATA = -9999.0
 
    outputs = []
    for scenario_name in p.si_paths.keys():
        outputs.extend([
            {
                'name': f'hazard_{scenario_name}',
                'tile_subdir': 'predict_landslides_scenarios',
                'tile_filename': f'hazard_prob_{scenario_name}_{{year}}.tif',
                'global_filename': f'hazard_prob_{scenario_name}_{{year}}.tif',
            },
            {
                'name': f'mortality_{scenario_name}',
                'tile_subdir': 'predict_mortality_scenarios',
                'tile_filename': f'expected_deaths_{scenario_name}_{{year}}.tif',
                'global_filename': f'expected_deaths_{scenario_name}_{{year}}.tif',
            },
        ])
 
    for year in p.prediction_years:
        for spec in outputs:
            out_path = os.path.join(p.cur_dir, spec['global_filename'].format(year=year))
 
            if os.path.exists(out_path) and not getattr(p, 'force_run', False):
                p.L.info(f'Skipping existing: {out_path}')
                continue
            if os.path.exists(out_path):
                os.remove(out_path)
 
            p.L.info(f'Stitching {spec["name"]} {year}')
 
            ds_out = driver.Create(
                out_path, n_cols_full, n_rows_full, 1, gdal.GDT_Float32,
                options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'],
            )
            ds_out.SetGeoTransform(gt)
            ds_out.SetProjection(proj)
            band_out = ds_out.GetRasterBand(1)
            band_out.SetNoDataValue(NODATA)
            band_out.Fill(NODATA)
 
            written, missing = 0, 0
            for block in blocks_list:
                col_off, row_off, n_c, n_r = [int(x) for x in block]
                tile_dir = os.path.join(p.tile_zones_dir, f'{row_off}_{col_off}')
                tile_path = os.path.join(
                    tile_dir, spec['tile_subdir'], spec['tile_filename'].format(year=year)
                )
 
                if not os.path.exists(tile_path):
                    missing += 1
                    continue
 
                ds_tile = gdal.Open(tile_path)
                arr = ds_tile.GetRasterBand(1).ReadAsArray().astype(np.float32)
                ds_tile = None
 
                arr = np.where(np.isnan(arr), NODATA, arr)
                band_out.WriteArray(arr, col_off, row_off)
                written += 1
 
            band_out.FlushCache()
            ds_out = None
 
            p.L.info(f'  Wrote {written} tiles')
            if missing:
                p.L.info(f'  Missing {missing} tiles')
 
    p.L.info('Stitching complete.')
    return p


def valuation(p):
    """
    Creates directory for valuation outputs. 
    """
    if p.run_this:
        return p

# Table 6.1, OECD (2025) Mortality Risk Valuation in Policy Assessment.
# USD millions, 2022 base year (same as the individual-country CSV).
GROUP_BASE_VSL_2022 = {
    'Global': 2.7,
    'OECD': 7.1,
    'EU': 8.4,
    'United States': 8.5,
    'High-income': 7.9,
    'Low-and-middle-income': 1.1,
}
 
INCOME_GRP_TO_FALLBACK_GROUP = {
    '1. High income: OECD': 'OECD',
    '2. High income: nonOECD': 'High-income',
    '3. Upper middle income': 'Low-and-middle-income',
    '4. Lower middle income': 'Low-and-middle-income',
    '5. Low income': 'Low-and-middle-income',
}
 
# PMPRB CPI-based price-adjustment factors (US CPI-derived): benchmark
# year 2019 -> 2022 cumulative price-adjustment factor = 1.050
DEFLATOR_2022_TO_2019 = 1 / 1.050
 
def build_vsl_raster(p):
    if p.run_this:
        out_path = os.path.join(p.valuation_dir, 'vsl_usd_2019_1km.tif')
        if os.path.exists(out_path) and not p.force_run:
            p.vsl_raster_path = out_path
            return p
 
        # ---- 1. Parse OECD VSL CSV ----
        # NOTE: has commas/special characters in
        # the name as downloaded, glob to avoid a brittle hardcoded match.
        oecd_candidates = glob.glob(os.path.join(p.base_data_dir, 'oecd_vsl', '*.csv'))
        if not oecd_candidates:
            raise FileNotFoundError(f'No CSV found in {p.base_data_dir}/oecd_vsl/')
        oecd_path = oecd_candidates[0]
 
        oecd = pd.read_csv(oecd_path)
 
        if 'MEASURE_VSL' in oecd.columns:
            oecd = oecd[oecd['MEASURE_VSL'] == 'VSL'].copy()
 
        oecd['unit_mult_factor'] = 10 ** oecd['UNIT_MULT'].astype(float)
        oecd['vsl_usd_2022'] = oecd['OBS_VALUE'].astype(float) * oecd['unit_mult_factor']
        oecd['vsl_usd_2019'] = oecd['vsl_usd_2022'] * DEFLATOR_2022_TO_2019
 
        vsl_by_iso3 = oecd.set_index('REF_AREA')['vsl_usd_2019'].to_dict()
        p.L.info(f'OECD VSL: {len(vsl_by_iso3)} countries with direct estimates '
                  f'(deflated to 2019 constant USD, factor={DEFLATOR_2022_TO_2019:.4f}).')
 
        # ---- 2. Join to correspondence GPKG ----
        correspondence_path = os.path.join(p.shared_base_data_dir, 'cartographic', 'ee_r264_correspondence.gpkg')
        corr = gpd.read_file(correspondence_path)
        iso3_field = 'iso3'
        if iso3_field not in corr.columns:
            raise KeyError(
                f'{iso3_field} not found in correspondence GPKG -- check '
                f'actual column names: {list(corr.columns)}'
            )
 
        corr['vsl_usd'] = corr[iso3_field].map(vsl_by_iso3)
 
        n_direct = corr['vsl_usd'].notna().sum()
 
        # ---- Group-level fallback for countries without a direct estimate ----
        needs_fallback = corr['vsl_usd'].isna()
        fallback_group_vsl_millions = corr.loc[needs_fallback, 'income_grp'].map(
            lambda ig: GROUP_BASE_VSL_2022.get(INCOME_GRP_TO_FALLBACK_GROUP.get(ig), np.nan)
        )
        # Anything still missing (unclassified income_grp) -> Global catch-all
        fallback_group_vsl_millions = fallback_group_vsl_millions.fillna(GROUP_BASE_VSL_2022['Global'])
 
        fallback_vsl_usd_2022 = fallback_group_vsl_millions * 1e6
        fallback_vsl_usd_2019 = fallback_vsl_usd_2022 * DEFLATOR_2022_TO_2019
        corr.loc[needs_fallback, 'vsl_usd'] = fallback_vsl_usd_2019
 
        n_group_fallback = (needs_fallback & corr['vsl_usd'].notna()).sum()
        p.L.info(f'VSL coverage: {n_direct} direct OECD estimates, '
                  f'{n_group_fallback} via income-group fallback (Table 6.1), '
                  f'{corr["vsl_usd"].isna().sum()} still unmatched.')
 
        # ---- 3. Reproject to EASE-Grid, rasterize ----
        corr_ease = corr.to_crs('EPSG:6933')
        work_dir = os.path.join(p.valuation_dir, 'vsl_work')
        os.makedirs(work_dir, exist_ok=True)
        temp_gpkg = os.path.join(work_dir, 'corr_ease_vsl.gpkg')
        corr_ease.to_file(temp_gpkg, driver='GPKG')
 
        ref_info = pygeo.get_raster_info(p.ease_grid_reference_path)
        gt = ref_info['geotransform']
        n_cols, n_rows = ref_info['raster_size']
 
        driver = gdal.GetDriverByName('GTiff')
        ds_out = driver.Create(
            out_path, n_cols, n_rows, 1, gdal.GDT_Float32,
            options=['TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                     'BLOCKXSIZE=256', 'BLOCKYSIZE=256'],
        )
        ds_out.SetGeoTransform(gt)
        ds_out.SetProjection(ref_info['projection_wkt'])
        band_out = ds_out.GetRasterBand(1)
        band_out.SetNoDataValue(-9999.0)
        band_out.Fill(-9999.0)
        ds_out = None
 
        pygeo.rasterize(
            temp_gpkg, out_path,
            option_list=['ATTRIBUTE=vsl_usd'],
        )
 
        p.L.info(f'VSL raster (2019 constant USD): {out_path}')
        p.vsl_raster_path = out_path
    return p



def compute_avoided_mortality(p):
    if p.run_this:
        for year in p.prediction_years:
            deaths_observed_path = os.path.join(
                p.stitch_tiles_dir, f'expected_deaths_observed_{year}.tif'
            )
            deaths_full_impacts_path = os.path.join(
                p.stitch_tiles_dir, f'expected_deaths_full_impacts_{year}.tif'
            )
            vsl_path = os.path.join(p.valuation_dir, 'vsl_usd_2019_1km.tif')
 
            avoided_mortality_path = os.path.join(p.valuation_dir, f'avoided_mortality_{year}.tif')
            avoided_mortality_value_path = os.path.join(
                p.valuation_dir, f'avoided_mortality_value_{year}.tif'
            )
 
            if (os.path.exists(avoided_mortality_path)
                    and os.path.exists(avoided_mortality_value_path)
                    and not p.force_run):
                p.L.info(f'{year}: avoided mortality outputs already exist, skipping.')
                continue
 
            for path, label in [(deaths_observed_path, 'expected_deaths_observed'),
                                 (deaths_full_impacts_path, 'expected_deaths_full_impacts'),
                                 (vsl_path, 'vsl_usd')]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f'{label} missing for {year}: {path}')
 
            # ---- avoided_mortality = full_impacts - observed ----
            deaths_obs_nd = pygeo.get_raster_info(deaths_observed_path)['nodata'][0]
            deaths_fi_nd = pygeo.get_raster_info(deaths_full_impacts_path)['nodata'][0]
 
            negative_count = [0]  # mutable closure for the sanity check below
 
            def avoided_op(deaths_obs, deaths_fi):
                valid = np.ones(deaths_obs.shape, dtype=bool)
                if deaths_obs_nd is not None:
                    valid &= (deaths_obs != deaths_obs_nd)
                if deaths_fi_nd is not None:
                    valid &= (deaths_fi != deaths_fi_nd)
 
                avoided = deaths_fi - deaths_obs
 
                negative_count[0] += int(((avoided < 0) & valid).sum())
 
                return np.where(valid, avoided, -9999.0).astype(np.float32)
 
            pygeo.raster_calculator(
                [(deaths_observed_path, 1), (deaths_full_impacts_path, 1)],
                avoided_op, avoided_mortality_path, gdal.GDT_Float32, -9999.0,
                calc_raster_stats=True,
            )
 
            if negative_count[0] > 0:
                p.L.warning(
                    f'{year}: {negative_count[0]} pixels have NEGATIVE avoided '
                    f'mortality (full_impacts predicts FEWER deaths than observed '
                    f'forest cover) -- this should not happen physically. Worth '
                    f'investigating (e.g. severity-model covariates that happen to '
                    f'differ between scenarios, though they should not; or SI clip '
                    f'edge cases) before trusting results near these locations.'
                )
            p.L.info(f'Avoided mortality {year}: {avoided_mortality_path}')
 
            # ---- avoided_mortality_value = avoided_mortality x VSL ----
            avoided_nd = -9999.0
            vsl_nd = pygeo.get_raster_info(vsl_path)['nodata'][0]
 
            def value_op(avoided, vsl):
                valid = np.ones(avoided.shape, dtype=bool)
                if avoided_nd is not None:
                    valid &= (avoided != avoided_nd)
                if vsl_nd is not None:
                    valid &= (vsl != vsl_nd)
                value = avoided * vsl
                return np.where(valid, value, -9999.0).astype(np.float32)
 
            pygeo.raster_calculator(
                [(avoided_mortality_path, 1), (vsl_path, 1)],
                value_op, avoided_mortality_value_path, gdal.GDT_Float32, -9999.0,
                calc_raster_stats=True,
            )
            p.L.info(f'Avoided mortality value {year}: {avoided_mortality_value_path}')
    return p



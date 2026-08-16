"""
preprocessing_tasks.py
"""
import os
import numpy as np
import geopandas as gpd
from osgeo import gdal
import pygeoprocessing as pygeo
import pygeoprocessing.routing as routing
import pandas as pd

from global_invest.landslide_mitigation.landslide_mitigation_functions import compute_si_global
from global_invest.landslide_mitigation.landslide_mitigation_utils import (
    warp_to_reference, 
    sample_raster_at_points,
    write_raster_from_array,
)

C_ROOT_MAX_KPA = 5.0 
GRAVITY_M_S2 = 9.81

# ==================================================================== #
# 0. Parent dir-creator
# ==================================================================== #

def preprocessing(p):
    """Creates p.preprocessing_dir. All other tasks below write into it."""
    if p.run_this:
        p.L.info(f'preprocessing_dir ready: {p.preprocessing_dir}')
        return p

# ==================================================================== #
# 1. Build UGLC annual panels
# ==================================================================== #

def build_uglc_annual_panels(p):
    """For each year in p.data_processing_range, builds:
      - uglc_binary_{year}.tif: 1 where any event's accuracy-buffer
        touches the pixel, 0 elsewhere.
      - uglc_mortality_{year}.tif: fatality count spread across each
        event's accuracy buffer via linear distance decay (1 - dist/accuracy_m),
        summed across events, for events in that year only.
    """
    if p.run_this:
        out_dir = os.path.join(p.preprocessing_dir, 'uglc_annual_panels')
        os.makedirs(out_dir, exist_ok=True)
        output_count = len(os.listdir(out_dir))
        if output_count == 2*len(p.data_processing_range) and not p.force_run:
            return p

        gdf = gpd.read_file(p.uglc_path)  # already EPSG:6933, has ease_x/ease_y

        # Remove events with non-finite coordinates
        n_before = len(gdf)
        gdf = gdf[np.isfinite(gdf['ease_x']) & np.isfinite(gdf['ease_y'])]
        n_removed = n_before - len(gdf)
        if n_removed > 0:
            p.L.warning(f'Removed {n_removed} events with non-finite coordinates (no data loss: '
                 f'all had 0 fatalities and invalid geometries)')

        ref_info = pygeo.get_raster_info(p.ease_grid_reference_path)
        gt = ref_info['geotransform']
        x_size, y_size = ref_info['raster_size']
        pixel_size = gt[1]

        for year in p.data_processing_range:
            binary_out_path = os.path.join(out_dir, f'uglc_binary_{year}.tif')
            mortality_out_path = os.path.join(out_dir, f'uglc_mortality_{year}.tif')

            if (os.path.exists(binary_out_path) and os.path.exists(mortality_out_path)
                    and not p.force_run):
                continue

            yearly_gdf = gdf[gdf['event_year'] == year]
            binary_arr = np.zeros((y_size, x_size), dtype=np.uint8)
            mortality_arr = np.zeros((y_size, x_size), dtype=np.float32)

            if yearly_gdf.empty:
                p.L.warning(f'No UGLC events for {year}, writing empty panels.')
            else:
                for _, row in yearly_gdf.iterrows():
                    cx, cy = row['ease_x'], row['ease_y']
                    radius_m = row['accuracy_m']
                    fatalities = row['fatality_count']
                    if radius_m <= 0:
                        continue

                    # Bounding box in pixel space, small (radius_m /
                    # pixel_size pixels each direction)
                    col_center = int((cx - gt[0]) / gt[1])
                    row_center = int((cy - gt[3]) / gt[5])
                    pad = int(np.ceil(radius_m / pixel_size)) + 1

                    row_start = max(row_center - pad, 0)
                    row_stop = min(row_center + pad, y_size - 1)
                    col_start = max(col_center - pad, 0)
                    col_stop = min(col_center + pad, x_size - 1)
                    if row_start > row_stop or col_start > col_stop:
                        continue  # event bbox entirely off-grid

                    rows = np.arange(row_start, row_stop + 1)
                    cols = np.arange(col_start, col_stop + 1)
                    col_grid, row_grid = np.meshgrid(cols, rows)

                    # Pixel-center coords in EASE-Grid meters
                    # no CRS transform needed (grid IS already in meters).
                    px = gt[0] + (col_grid + 0.5) * gt[1]
                    py = gt[3] + (row_grid + 0.5) * gt[5]
                    dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)

                    within = dist <= radius_m
                    if not within.any():
                        continue

                    # binary: touched by this event's buffer
                    sub_binary = binary_arr[row_start:row_stop + 1, col_start:col_stop + 1]
                    sub_binary[within] = 1

                    # mortality: linear distance-decay weight, only if fatalities > 0
                    if fatalities > 0:
                        weight = np.where(within, 1 - (dist / radius_m), 0)
                        sub_mortality = mortality_arr[row_start:row_stop + 1, col_start:col_stop + 1]
                        sub_mortality += fatalities * weight

            write_raster_from_array(binary_arr, gt, ref_info['projection_wkt'], binary_out_path,
                          nodata=255, dtype=gdal.GDT_Byte)
            write_raster_from_array(mortality_arr, gt, ref_info['projection_wkt'], mortality_out_path,
                          nodata=-9999.0, dtype=gdal.GDT_Float32)

            n_events = len(yearly_gdf)
            n_fatal_events = int((yearly_gdf['fatality_count'] > 0).sum()) if n_events else 0
            p.L.info(f'{year}: {n_events} events ({n_fatal_events} with fatalities) '
                      f'-> {binary_out_path}, {mortality_out_path}')

        p.uglc_binary_dir = out_dir
        p.uglc_mortality_dir = out_dir
    return p


# ==================================================================== #
# 2. DEM preprocessing: pit-filling, flow direction, upslope area, slope
# ==================================================================== #

def fill_pits(p):
    """Re-fills any new depressions introduced by resampling the trusted
    300m pit-filled DEM down to 1km EASE-Grid. Averaging can create small 
    new local minima that weren't in the original 300m DEM.
    """
    if p.run_this:
        out_path = os.path.join(p.preprocessing_dir, 'pit_filled_dem_1km.tif')
        if os.path.exists(out_path) and not p.force_run:
            p.pit_filled_dem_path = out_path
            return p
 
        working_dir = os.path.join(p.preprocessing_dir, 'fill_pits_work')
        os.makedirs(working_dir, exist_ok=True)
 
        routing.fill_pits(
            (p.dem_path, 1), out_path, working_dir=working_dir,
        )
        p.L.info(f'Pit-filled 1km DEM: {out_path}')
        p.pit_filled_dem_path = out_path
    return p
 
 
def compute_flow_dir_d8(p):
    """
    Fresh D8 flow direction computation at 1km.
    """
    if p.run_this:
        out_path = os.path.join(p.preprocessing_dir, 'flow_dir_1km.tif')
        if os.path.exists(out_path) and not p.force_run:
            p.flow_dir_path = out_path
            return p
 
        working_dir = os.path.join(p.preprocessing_dir, 'flow_dir_work')
        os.makedirs(working_dir, exist_ok=True)
 
        routing.flow_dir_d8(
            (p.pit_filled_dem_path, 1), out_path, working_dir=working_dir,
        )
        p.L.info(f'D8 flow direction (1km, freshly computed): {out_path}')
        p.flow_dir_path = out_path
    return p
 
 
def compute_upslope_area(p):
    """
    Flow accumulation (raw pixel count draining through each cell),
    then scaled to real area in m^2. EASE-Grid is equal-area, so this is
    just a constant multiply.
    """
    if p.run_this:
        accum_path = os.path.join(p.preprocessing_dir, 'flow_accum_pixel_count_1km.tif')
        out_path = os.path.join(p.preprocessing_dir, 'upslope_area_m2_1km.tif')
        if os.path.exists(out_path) and not p.force_run:
            p.upslope_area_path = out_path
            return p
 
        routing.flow_accumulation_d8((p.flow_dir_path, 1), accum_path)
        p.L.info(f'Flow accumulation (pixel count, 1km): {accum_path}')
 
        ref_info = pygeo.get_raster_info(p.ease_grid_reference_path)
        pixel_size_m = ref_info['pixel_size'][0]  # square pixels, EASE-Grid
        pixel_area_m2 = pixel_size_m ** 2
 
        accum_info = pygeo.get_raster_info(accum_path)
        accum_nodata = accum_info['nodata'][0]
 
        def scale_to_area(accum_array):
            valid = accum_array != accum_nodata
            result = np.where(valid, accum_array * pixel_area_m2, accum_nodata)
            return result.astype(np.float32)
 
        pygeo.raster_calculator(
            [(accum_path, 1)], scale_to_area, out_path,
            gdal.GDT_Float32, accum_nodata,
        )
        p.L.info(f'Upslope area (m^2, 1km): {out_path}')
        p.upslope_area_path = out_path
    return p


def compute_slope(p):
    """
    gdal.DEMProcessing computes slope in degrees from the pit-filled DEM, 
    using Horn (1981) algorithm. EASE-Grid is already in meters, so no 
    lat/lon degree-to-meter scale factor needed.
    """
    if p.run_this: 
        out_path = os.path.join(p.preprocessing_dir, 'slope_degrees_1km.tif')
        if os.path.exists(out_path) and not p.force_run:
            p.slope_path = out_path
            return p
 
        work_dir = os.path.join(p.preprocessing_dir, 'slope_work')
        os.makedirs(work_dir, exist_ok=True)
 
        # ---- 1. Build a fine (~250m, exact 4x subdivision) EASE-Grid raster ----
        ref_info = pygeo.get_raster_info(p.ease_grid_reference_path)
        ref_gt = ref_info['geotransform']
        ref_cols, ref_rows = ref_info['raster_size']
 
        FINE_FACTOR = 4
        fine_pixel_size = ref_gt[1] / FINE_FACTOR
        fine_cols = ref_cols * FINE_FACTOR
        fine_rows = ref_rows * FINE_FACTOR
        fine_gt = (ref_gt[0], fine_pixel_size, 0, ref_gt[3], 0, -fine_pixel_size)
 
        fine_ref_path = os.path.join(work_dir, 'ease_grid_reference_fine.tif')
        if not os.path.exists(fine_ref_path) or p.force_run:
            driver = gdal.GetDriverByName('GTiff')
            ds = driver.Create(
                fine_ref_path, fine_cols, fine_rows, 1, gdal.GDT_Byte,
                options=['TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                         'BLOCKXSIZE=256', 'BLOCKYSIZE=256'],
            )
            ds.SetGeoTransform(fine_gt)
            ds.SetProjection(ref_info['projection_wkt'])
            ds.GetRasterBand(1).SetNoDataValue(0)
            ds = None
 
        # ---- 2. Warp RAW (unfilled) elevation to the fine grid ----
        raw_dem_src = os.path.join(p.shared_base_data_dir, 'seals', 'static_regressors', 'alt_m.tif')
        dem_fine_path = os.path.join(work_dir, 'dem_fine.tif')
        if not os.path.exists(dem_fine_path) or p.force_run:
            warp_to_reference(
                raw_dem_src, dem_fine_path, fine_ref_path,
                resample_method='average',  # ~300m native -> ~250m, still continuous
                src_nodata=-9999, dst_nodata=-9999.0,
                output_type=gdal.GDT_Float32,
            )
 
        # ---- 3. Compute slope at fine resolution (Horn 1981) ----
        p.L.info('Computing slope at fine resolution (gdal.DEMProcessing)...')
        slope_fine_path = os.path.join(work_dir, 'slope_fine.tif')
        if not os.path.exists(slope_fine_path) or p.force_run:
            dem_options = gdal.DEMProcessingOptions(
                slopeFormat='degree',
                creationOptions=['TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                                  'BLOCKXSIZE=256', 'BLOCKYSIZE=256'],
            )
            result_ds = gdal.DEMProcessing(
                slope_fine_path, dem_fine_path, 'slope', options=dem_options,
            )
            if result_ds is None or not os.path.exists(slope_fine_path):
                raise RuntimeError(
                    f'gdal.DEMProcessing failed to produce {slope_fine_path} '
                    f'(check disk space and BIGTIFF support in this GDAL build).'
                )
            result_ds = None
            p.L.info(f'Slope (fine resolution) computed: {slope_fine_path}')

 
        # ---- 4. Aggregate fine slope down to 1km via average ----
        if not os.path.exists(out_path) or p.force_run:
            warp_to_reference(
                slope_fine_path, out_path, p.ease_grid_reference_path,
                resample_method='average',
                src_nodata=-9999.0, dst_nodata=-9999.0,
                output_type=gdal.GDT_Float32,
            )
        p.L.info(f'Slope (fine-computed, aggregated to 1km): {out_path}')
        p.slope_path = out_path
    return p

 
 
# ==================================================================== #
# 4. Soil hydraulic properties: friction angle, cohesion, unit weight,
# transmissivity. 
# ==================================================================== #
 

def compute_soil_hydraulic_properties(p):
    if p.run_this:
        p.friction_angle_path = os.path.join(p.preprocessing_dir, 'friction_angle_1km.tif')
        p.cohesion_soil_path = os.path.join(p.preprocessing_dir, 'cohesion_soil_1km.tif')
        p.unit_weight_path = os.path.join(p.preprocessing_dir, 'unit_weight_1km.tif')
        p.transmissivity_path = os.path.join(p.preprocessing_dir, 'transmissivity_1km.tif')
 
        out_paths = [p.friction_angle_path, p.cohesion_soil_path,
                     p.unit_weight_path, p.transmissivity_path]
        if all(os.path.exists(op) for op in out_paths) and not p.force_run:
            p.L.info('Soil hydraulic properties already computed.')
            return p
 
        sand = p.soilgrids_paths['sand_pct']
        clay = p.soilgrids_paths['clay_pct']
        org_carbon = p.soilgrids_paths['org_carbon_pct']
        bulk_density = p.soilgrids_paths['bulk_density']
        ksat = p.ksat_path
        soil_depth = p.soil_depth_path
 
        NODATA = -9999.0
 
        def get_nodata(path):
            return pygeo.get_raster_info(path)['nodata'][0]
 
        # ---- Friction angle ----
        sand_nd, clay_nd = get_nodata(sand), get_nodata(clay)
 
        def compute_friction_angle(sand_arr, clay_arr):
            valid = np.ones(sand_arr.shape, dtype=bool)
            if sand_nd is not None:
                valid &= (sand_arr != sand_nd)
            if clay_nd is not None:
                valid &= (clay_arr != clay_nd)
            phi = 25 + (sand_arr - clay_arr) / 20.0
            phi = np.clip(phi, 15, 40)
            return np.where(valid, phi, NODATA).astype(np.float32)
 
        pygeo.raster_calculator(
            [(sand, 1), (clay, 1)], compute_friction_angle,
            p.friction_angle_path, gdal.GDT_Float32, NODATA,
            calc_raster_stats=True,
        )
        p.L.info(f'Friction angle: {p.friction_angle_path}')
 
        # ---- Soil cohesion ----
        clay_nd2, org_nd = get_nodata(clay), get_nodata(org_carbon)
 
        def compute_cohesion(clay_arr, org_c_arr):
            valid = np.ones(clay_arr.shape, dtype=bool)
            if clay_nd2 is not None:
                valid &= (clay_arr != clay_nd2)
            if org_nd is not None:
                valid &= (org_c_arr != org_nd)
            c_soil = 2.0 + 0.03 * clay_arr + 0.1 * org_c_arr
            c_soil = np.clip(c_soil, 0, 50)
            return np.where(valid, c_soil, NODATA).astype(np.float32)
 
        pygeo.raster_calculator(
            [(clay, 1), (org_carbon, 1)], compute_cohesion,
            p.cohesion_soil_path, gdal.GDT_Float32, NODATA,
            calc_raster_stats=True,
        )
        p.L.info(f'Soil cohesion: {p.cohesion_soil_path}')
 
        # ---- Unit weight: DERIVED from bulk_density (gamma = rho * g),
        # SoilGrids is in conventional units kg/dm3 = Mg/m3 
        # (per earlier conv_factor=100 applied during acquisition). 
        # gamma_kN_m3 = bulk_density_Mg_m3 * g
        bulk_nd = get_nodata(bulk_density)
 
        def compute_unit_weight(bulk_arr):
            valid = np.ones(bulk_arr.shape, dtype=bool)
            if bulk_nd is not None:
                valid &= (bulk_arr != bulk_nd)
            gamma_kn_m3 = bulk_arr * GRAVITY_M_S2  # Mg/m3 * m/s2 = kN/m3
            return np.where(valid, gamma_kn_m3, NODATA).astype(np.float32)
 
        pygeo.raster_calculator(
            [(bulk_density, 1)], compute_unit_weight,
            p.unit_weight_path, gdal.GDT_Float32, NODATA,
            calc_raster_stats=True,
        )
        p.L.info(f'Unit weight (derived from bulk density): {p.unit_weight_path}')
 
        # ---- Transmissivity = K_sat x soil_depth ----
        # HiHydroSoil scaling: raw Int32 = float_value x 10,000,
        ksat_nd, depth_nd = get_nodata(ksat), get_nodata(soil_depth)
 
        def compute_transmissivity(ksat_arr, depth_arr):
            valid = np.ones(ksat_arr.shape, dtype=bool)
            if ksat_nd is not None:
                valid &= (ksat_arr != ksat_nd)
            if depth_nd is not None:
                valid &= (depth_arr != depth_nd)
            ksat_cm_day = ksat_arr * 0.0001       # HiHydroSoil scale factor
            ksat_m_day = ksat_cm_day / 100.0      # cm/day -> m/day
            T = ksat_m_day * depth_arr            # m/day x m = m^2/day
            return np.where(valid, T, NODATA).astype(np.float32)
 
        pygeo.raster_calculator(
            [(ksat, 1), (soil_depth, 1)], compute_transmissivity,
            p.transmissivity_path, gdal.GDT_Float32, NODATA,
            calc_raster_stats=True,
        )
        p.L.info(f'Transmissivity (UNITS UNVERIFIED, see note): {p.transmissivity_path}')
    return p

 
 
# ==================================================================== #
# 5. Static specific discharge q = rain x upslope_area / cell_width
# ==================================================================== #
 

def compute_static_q(p):
    if p.run_this:
        out_path = os.path.join(p.preprocessing_dir, 'static_q_1km.tif')
        if os.path.exists(out_path) and not p.force_run:
            p.static_q_path = out_path
            return p
 
        rain_path = p.climatological_rain_path
        upslope_path = p.upslope_area_path  # ALREADY in m^2, not pixel count
 
        ref_info = pygeo.get_raster_info(p.ease_grid_reference_path)
        cell_width_m = ref_info['pixel_size'][0]
 
        NODATA = -9999.0
        rain_nd = pygeo.get_raster_info(rain_path)['nodata'][0]
        upslope_nd = pygeo.get_raster_info(upslope_path)['nodata'][0]
 
        def compute_q(rain_mm_yr, upslope_area_m2):
            valid = np.ones(rain_mm_yr.shape, dtype=bool)
            if rain_nd is not None:
                valid &= (rain_mm_yr != rain_nd)
            if upslope_nd is not None:
                valid &= (upslope_area_m2 != upslope_nd)
            rain_m_day = (rain_mm_yr / 365.25) / 1000.0
            q = (rain_m_day * upslope_area_m2) / cell_width_m  # m^2/day
            return np.where(valid, q, NODATA).astype(np.float32)
 
        pygeo.raster_calculator(
            [(rain_path, 1), (upslope_path, 1)], compute_q,
            out_path, gdal.GDT_Float32, NODATA,
            calc_raster_stats=True,
        )
        p.L.info(f'Static q: {out_path}')
        p.static_q_path = out_path
    return p

 
 
# ==================================================================== #
# 6. SI scenarios
#   'observed': per-year, across modeling_range UNION prediction_years
#   all other scenarios: prediction_years ONLY (counterfactuals have no
#   meaning during calibration)
# ==================================================================== #
 
def compute_si_scenarios(p):
    if p.run_this:
 
        p.si_paths = {} 
 
        observed_years = sorted(set(p.modeling_range) | set(p.prediction_years))
 
        for scenario_name in p.c_root_scenarios:
            years_needed = observed_years if scenario_name == 'observed' else p.prediction_years
            p.si_paths[scenario_name] = {}
 
            for year in years_needed:
                out_path = os.path.join(
                    p.preprocessing_dir, 'si_scenarios', f'si_{scenario_name}_{year}_1km.tif'
                )
                if os.path.exists(out_path) and not p.force_run:
                    p.si_paths[scenario_name][year] = out_path
                    continue
 
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
 
                forest_share_path = os.path.join(
                    p.input_data_dir, 'forest_share_1km', f'forest_share_{year}_1km.tif'
                )
                c_root_max = 0.0 if p.c_root_scenarios[scenario_name] == 0 else C_ROOT_MAX_KPA
 
                compute_si_global(
                    friction_angle_path=p.friction_angle_path,
                    cohesion_soil_path=p.cohesion_soil_path,
                    forest_share_path=forest_share_path,
                    c_root_max=c_root_max,
                    unit_weight_path=p.unit_weight_path,
                    transmissivity_path=p.transmissivity_path,
                    static_q_path=p.static_q_path,
                    slope_path=p.slope_path,
                    soil_depth_path=p.soil_depth_path,
                    output_si_path=out_path,
                )
                p.si_paths[scenario_name][year] = out_path
                p.L.info(f'SI ({scenario_name}, {year}): {out_path}')
    return p


# ==================================================================== #
# 7. Build estimation panel for modeling
# ==================================================================== #

def build_estimation_table(p):
    if p.run_this:
 
        out_path = os.path.join(p.preprocessing_dir, 'estimation_table.csv')
        if os.path.exists(out_path) and not p.force_run:
            p.estimation_table_path = out_path
            return p
 
        gdf = gpd.read_file(p.uglc_path)  # ease_x, ease_y, event_year, fatality_count
 
        observed_years = sorted(set(p.modeling_range) | set(p.prediction_years))
 
        # ---- Land-pixel bounds for uniform control sampling ----
        ref_info = pygeo.get_raster_info(p.ease_grid_reference_path)
        ref_gt = ref_info['geotransform']
        x_size, y_size = ref_info['raster_size']
 
        gaez_ds = gdal.Open(p.gaez_path)
        gaez_band = gaez_ds.GetRasterBand(1)
        gaez_nodata = gaez_band.GetNoDataValue()
 
        rng = np.random.default_rng(seed=42)  # reproducible control draws
 
        def draw_land_controls(n_needed, max_attempts_factor=20):
            """Uniform-random pixel draws over the full grid, kept only if
            they land on valid land (per GAEZ nodata)
            """
            accepted_x, accepted_y = [], []
            attempts = 0
            max_attempts = n_needed * max_attempts_factor
            while len(accepted_x) < n_needed and attempts < max_attempts:
                batch_size = min(n_needed * 2, 5000)
                cols = rng.integers(0, x_size, size=batch_size)
                rows = rng.integers(0, y_size, size=batch_size)
                attempts += batch_size
 
                # Single-pixel windowed reads via GAEZ band
                for r, c in zip(rows, cols):
                    val = gaez_band.ReadAsArray(int(c), int(r), 1, 1)[0, 0]
                    if gaez_nodata is None or val != gaez_nodata:
                        x = ref_gt[0] + (c + 0.5) * ref_gt[1]
                        y = ref_gt[3] + (r + 0.5) * ref_gt[5]
                        accepted_x.append(x)
                        accepted_y.append(y)
                        if len(accepted_x) >= n_needed:
                            break
 
            if len(accepted_x) < n_needed:
                p.L.warning(
                    f'Only drew {len(accepted_x)}/{n_needed} valid land '
                    f'controls after {attempts} attempts -- land-valid '
                    f'fraction may be lower than expected, or nodata '
                    f'check is wrong.'
                )
            return np.array(accepted_x), np.array(accepted_y)
 
        # ---- Build per-year case + control rows ----
        ref_gt_for_dedup = pygeo.get_raster_info(p.ease_grid_reference_path)['geotransform']
        pixel_size_dedup = ref_gt_for_dedup[1]
 
        all_rows = []
        for year in observed_years:
            cases_year_raw = gdf[gdf['event_year'] == year].copy()
 
            # DEDUPLICATE to one row per (year, pixel) before fitting.
            cases_year_raw['pixel_col'] = (
                (cases_year_raw['ease_x'] - ref_gt_for_dedup[0]) // pixel_size_dedup
            ).astype(int)
            cases_year_raw['pixel_row'] = (
                (cases_year_raw['ease_y'] - ref_gt_for_dedup[3]) // ref_gt_for_dedup[5]
            ).astype(int)
 
            before_dedup = len(cases_year_raw)
            cases_year = (
                cases_year_raw
                .groupby(['pixel_row', 'pixel_col'], as_index=False)
                .agg({'ease_x': 'first', 'ease_y': 'first', 'fatality_count': 'max'})
            )
            if before_dedup != len(cases_year):
                p.L.info(f'{year}: deduplicated {before_dedup} case points -> '
                          f'{len(cases_year)} unique pixels.')
 
            n_cases = len(cases_year)
            if n_cases == 0:
                p.L.warning(f'{year}: no UGLC events, skipping controls for this year too.')
                continue
 
            n_controls = int(round(p.control_ratio * n_cases))
            control_x, control_y = draw_land_controls(n_controls)
 
            case_df = pd.DataFrame({
                'ease_x': cases_year['ease_x'].values,
                'ease_y': cases_year['ease_y'].values,
                'year': year,
                'case': 1,
                'fatality_count': cases_year['fatality_count'].values,
            })
            control_df = pd.DataFrame({
                'ease_x': control_x,
                'ease_y': control_y,
                'year': year,
                'case': 0,
                'fatality_count': 0.0,
            })
            all_rows.append(pd.concat([case_df, control_df], ignore_index=True))
            p.L.info(f'{year}: {n_cases} cases + {len(control_x)} controls')
 
        panel = pd.concat(all_rows, ignore_index=True)
 
        # ---- Sample static covariates (same for every row regardless of year) ----
        panel['gaez_zone'] = sample_raster_at_points(
            p.gaez_path, panel['ease_x'], panel['ease_y']
        )
        panel['road_density'] = sample_raster_at_points(
            p.road_density_path, panel['ease_x'], panel['ease_y']
        )
        panel['slope_degrees'] = sample_raster_at_points(
            p.slope_path, panel['ease_x'], panel['ease_y']
        )
 
        # ---- Sample per-year covariates, one year at a time (avoids
        # opening/sampling every year's raster for every row) ----
        panel['si_observed'] = np.nan
        panel['rain_max_daily'] = np.nan
        panel['population'] = np.nan
 
        for year in observed_years:
            mask = panel['year'] == year
            if not mask.any():
                continue
 
            si_path = p.si_paths['observed'].get(year)
            if si_path:
                panel.loc[mask, 'si_observed'] = sample_raster_at_points(
                    si_path, panel.loc[mask, 'ease_x'], panel.loc[mask, 'ease_y']
                )
 
            rain_path = os.path.join(
                p.input_data_dir, 'era5_land', f'era5_max_daily_mm_{year}.tif'
            )
            if os.path.exists(rain_path):
                panel.loc[mask, 'rain_max_daily'] = sample_raster_at_points(
                    rain_path, panel.loc[mask, 'ease_x'], panel.loc[mask, 'ease_y']
                )
            else:
                p.L.warning(f'{year}: rain_max_daily raster not found at {rain_path}')
 
            pop_path = os.path.join(
                p.input_data_dir, 'landscan_1km', f'landscan_{year}_1km.tif'
            )
            if os.path.exists(pop_path):
                panel.loc[mask, 'population'] = sample_raster_at_points(
                    pop_path, panel.loc[mask, 'ease_x'], panel.loc[mask, 'ease_y']
                )
            else:
                p.L.warning(f'{year}: population raster not found at {pop_path}')
 
        # ---- Drop rows with missing critical covariates (e.g. min_slope_deg
        # exclusion in compute_si_global, or ocean/nodata GAEZ edge cases) ----
        before = len(panel)
        panel = panel.dropna(subset=['si_observed', 'rain_max_daily', 'gaez_zone'])
        p.L.info(f'Dropped {before - len(panel)} rows with missing critical '
                  f'covariates (likely min_slope_deg exclusion or nodata edges).')
 
        panel.to_csv(out_path, index=False)
        p.L.info(f'Estimation table built: {out_path} ({len(panel)} rows)')
        p.estimation_table_path = out_path
    return p

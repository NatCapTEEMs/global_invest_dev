"""
input_data_tasks.py

Raw sources read from p.base_data_dir (input_data_raw/), except ESA-CCI
which reads from p.shared_base_data_dir (shared base_data, used across
projects). All reprojected outputs write into p.input_data_dir.
"""
import os
import numpy as np
import pandas as pd
from osgeo import gdal, osr
import geopandas as gpd
import pygeoprocessing as pygeo

from global_invest.landslide_mitigation.landslide_mitigation_utils import (
    parse_gpd_grid_definition,
    warp_to_reference,
)
from global_invest.landslide_mitigation.landslide_mitigation_functions import (
    thickness_weighted_combine,
    DEPTH_WEIGHTS_0_30CM,
)


# ==================================================================== #
# 0. Parent dir-creator
# ==================================================================== #

def input_data(p):
    """Creates p.input_data_dir. All other tasks below write into it."""
    if p.run_this:
        p.L.info(f'input_data_dir ready: {p.input_data_dir}')
    return p


# ==================================================================== #
# 1. EASE-Grid 2.0 reference raster
# ==================================================================== #

def build_ease_grid_reference(p):
    """Parses EASE2_M01km.gpd (a SOURCE file, read from base_data_dir
    and builds an empty reference raster on that exact grid.
    """
    if p.run_this:
        gpd_path = os.path.join(p.base_data_dir, 'nsidc_proj', 'EASE2_M01km.gpd')
        grid = parse_gpd_grid_definition(gpd_path)

        out_path = os.path.join(p.input_data_dir, 'ease_grid_reference.tif')
        if os.path.exists(out_path) and not p.force_run:
            p.L.info('EASE-Grid reference raster already exists, skipping.')
            p.ease_grid_reference_path = out_path
            return p

        gt = (grid['origin_x'], grid['pixel_size'], 0,
              grid['origin_y'], 0, -grid['pixel_size'])
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(
            out_path, grid['n_cols'], grid['n_rows'], 1, gdal.GDT_Byte,
            options=['TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                     'BLOCKXSIZE=256', 'BLOCKYSIZE=256'],
        )
        ds.SetGeoTransform(gt)
        ds.SetProjection(grid['srs_wkt'])
        ds.GetRasterBand(1).SetNoDataValue(0)
        ds = None

        p.L.info(f'Built EASE-Grid reference raster from {gpd_path}: {out_path}')
        p.ease_grid_reference_path = out_path
    return p


# ==================================================================== #
# 2. DEM -- SEALS alt_m.tif (shared base_data)
# ==================================================================== #

def reproject_dem(p):
    if p.run_this:
        src_path = os.path.join(p.shared_base_data_dir, 'seals', 'static_regressors', 'alt_m.tif')
        out_path = os.path.join(p.input_data_dir, 'dem_1km.tif')
        if os.path.exists(out_path) and not p.force_run:
            p.dem_path = out_path
            return p

        warp_to_reference(
            src_path, out_path, p.ease_grid_reference_path,
            resample_method='average',
            src_nodata=-9999,
            dst_nodata=-9999.0,
            output_type=gdal.GDT_Float32,
        )
        p.L.info(f'DEM reprojected to EASE-Grid 1km: {out_path}')
        p.dem_path = out_path
    return p


# ==================================================================== #
# 3. GAEZ zones -- categorical
# ==================================================================== #

def reproject_gaez(p):
    if p.run_this:
        src_path = os.path.join(p.base_data_dir, 'fao_gaez', 'GAEZ-V5.AEZ57.tif')
        out_path = os.path.join(p.input_data_dir, 'gaez_zones_1km.tif')
        if os.path.exists(out_path) and not p.force_run:
            p.gaez_path = out_path
            return p

        warp_to_reference(
            src_path, out_path, p.ease_grid_reference_path,
            resample_method='mode',  # categorical -- majority zone per cell
            output_type=gdal.GDT_Byte,  # 57 classes fit comfortably
        )
        p.L.info(f'GAEZ zones reprojected: {out_path}')
        p.gaez_path = out_path
    return p


# ==================================================================== #
# 4. ESA-CCI forest_share -- class weights, medium weight for mosaics
# ==================================================================== #

FOREST_WEIGHT = {
    50: 1.0, 60: 1.0, 61: 1.0, 62: 1.0,
    70: 1.0, 71: 1.0, 72: 1.0, 80: 1.0, 81: 1.0, 82: 1.0,
    90: 1.0,             # tree_mixed_type
    160: 1.0, 170: 1.0,  # flooded tree cover
    100: 0.5, 110: 0.5,  # mosaic tree/shrub-herbaceous 50/50 -- medium weight (chosen)
    151: 0.0,            # sparse_tree_15 -- below meaningful root reinforcement
    # all other classes default to 0.0
}

def reproject_esacci_forest_share(p):
    if p.run_this:
        work_dir = os.path.join(p.input_data_dir, 'esacci_work')
        os.makedirs(work_dir, exist_ok=True)

        for year in p.data_processing_range:
            src_path = os.path.join(
                p.shared_base_data_dir, 'lulc', 'esa', f'lulc_esa_{year}.tif'
            )
            if not os.path.exists(src_path):
                p.L.warning(f'ESA-CCI {year} not found, skipping year.')
                continue

            out_path = os.path.join(
                p.input_data_dir, 'forest_share_1km', f'forest_share_{year}_1km.tif'
            )
            if os.path.exists(out_path) and not p.force_run:
                continue

            # ---- classify raw class codes -> forest weight (0-1), BLOCK-WISE ----
            src_ds = gdal.Open(src_path)
            src_band = src_ds.GetRasterBand(1)
            src_nodata = src_band.GetNoDataValue()
            gt = src_ds.GetGeoTransform()
            proj = src_ds.GetProjection()
            x_size = src_ds.RasterXSize
            y_size = src_ds.RasterYSize

            # LUT sized to 256 (ESA-CCI class codes are 8-bit)
            lut = np.zeros(256, dtype=np.float32)
            for class_code, weight in FOREST_WEIGHT.items():
                lut[class_code] = weight

            native_path = os.path.join(work_dir, f'forest_weight_{year}_native.tif')
            driver = gdal.GetDriverByName('GTiff')
            ds_out = driver.Create(
                native_path, x_size, y_size, 1, gdal.GDT_Float32,
                options=['TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                         'BLOCKXSIZE=256', 'BLOCKYSIZE=256'],
            )
            ds_out.SetGeoTransform(gt)
            ds_out.SetProjection(proj)
            band_out = ds_out.GetRasterBand(1)
            band_out.SetNoDataValue(-9999.0)

            block_rows = 2048  # small per-chunk memory footprint
            for row_start in range(0, y_size, block_rows):
                rows_here = min(block_rows, y_size - row_start)
                chunk = src_band.ReadAsArray(0, row_start, x_size, rows_here)
                safe_idx = np.clip(chunk, 0, 255)
                weight_chunk = lut[safe_idx]
                if src_nodata is not None:
                    weight_chunk = np.where(chunk == src_nodata, -9999.0, weight_chunk)
                band_out.WriteArray(weight_chunk, 0, row_start)

            src_ds = None
            ds_out = None
            p.L.info(f'{year}: classified forest weight block-wise -> {native_path}')

            warp_to_reference(
                native_path, out_path, p.ease_grid_reference_path,
                resample_method='average',  # 0-1 weight field -> fractional share
                src_nodata=-9999.0, dst_nodata=-9999.0,
                output_type=gdal.GDT_Float32,
            )
            p.L.info(f'ESA-CCI forest_share {year} reprojected: {out_path}')
    return p


# ==================================================================== #
# 5. UGLC -- point coordinate transform, not raster warp
# ==================================================================== #

def reproject_uglc_events(p):
    """Load UGLC, clean/filter, transform points to EASE-Grid x/y (meters).
    Real schema confirmed from the prior pipeline's preprocess_uglc:
    pipe-delimited, WKT_GEOM geometry column, ACCURACY/START DATE/END DATE/
    FATALITIES fields. This task ONLY produces a clean point table --
    the annual binary/mortality raster panels are built separately by
    preprocessing_tasks.build_uglc_annual_panels(), which consumes
    p.uglc_path (this task's output).
    """
    if p.run_this:
 
        src_path = os.path.join(p.base_data_dir, 'uglc', 'UGLC_point.csv')
        out_path = os.path.join(p.input_data_dir, 'uglc_points_ease.gpkg')
        if os.path.exists(out_path) and not p.force_run:
            p.uglc_path = out_path
            return p
 
        df = pd.read_csv(src_path, sep='|', low_memory=False)
        rename_map = {
            'WKT_GEOM': 'geometry_wkt', 'ID': 'uglc_id', 'ACCURACY': 'accuracy_m',
            'START DATE': 'start_date', 'END DATE': 'end_date',
            'TYPE': 'landslide_type', 'PHYSICAL FACTORS': 'physical_factors',
            'RECORD TYPE': 'record_type', 'FATALITIES': 'fatality_count',
            'INJURIES': 'injury_count',
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
 
        if 'geometry_wkt' not in df.columns:
            raise KeyError('UGLC CSV does not contain a WKT_GEOM column.')
 
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.GeoSeries.from_wkt(df['geometry_wkt']), crs='EPSG:4326',
        )
 
        for date_col in ['start_date', 'end_date']:
            if date_col in gdf.columns:
                gdf[date_col] = pd.to_datetime(gdf[date_col], errors='coerce')
 
        year_source = getattr(p, 'uglc_year_source', 'start_date')
        gdf['event_year'] = gdf[year_source].dt.year
        if 'end_date' in gdf.columns:
            gdf['event_year'] = gdf['event_year'].fillna(gdf['end_date'].dt.year)
 
        gdf['fatality_count'] = (
            pd.to_numeric(gdf.get('fatality_count', 0), errors='coerce').fillna(0).clip(lower=0)
        )
        if 'injury_count' in gdf.columns:
            gdf['injury_count'] = pd.to_numeric(gdf['injury_count'], errors='coerce').fillna(0).clip(lower=0)
 
        if 'accuracy_m' not in gdf.columns:
            raise KeyError('UGLC CSV does not contain an ACCURACY column.')
        gdf['accuracy_m'] = pd.to_numeric(gdf['accuracy_m'], errors='coerce')
        gdf = gdf[gdf['accuracy_m'].notna()]
        before = len(gdf)
        # Exclude known NDV codes
        NDV_CODES = {-99999, -9999, -999, 0, np.nan}
        gdf = gdf[~gdf['accuracy_m'].isin(NDV_CODES)].copy()
        gdf = gdf[gdf['accuracy_m'] > 0].copy()
        gdf = gdf[gdf['accuracy_m'] <= p.max_location_accuracy_m].copy()
        p.L.info(f'UGLC: {before} -> {len(gdf)} events after accuracy filter '
                  f'(<= {p.max_location_accuracy_m}m)')
        
        # Reproject directly to EASE-Grid (EPSG:6933)
        gdf_ease = gdf.to_crs('EPSG:6933')
        gdf_ease['ease_x'] = gdf_ease.geometry.x
        gdf_ease['ease_y'] = gdf_ease.geometry.y
 
        # BUG FIX: non-finite coordinates (invalid/empty source geometries)
        before_finite = len(gdf_ease)
        gdf_ease = gdf_ease[
            np.isfinite(gdf_ease['ease_x']) & np.isfinite(gdf_ease['ease_y'])
        ].copy()
        if before_finite != len(gdf_ease):
            p.L.info(f'Removed {before_finite - len(gdf_ease)} events with '
                      f'non-finite coordinates (invalid/empty geometries).')
 
        gdf_ease.to_file(out_path, driver='GPKG')
        p.L.info(f'UGLC events reprojected to EASE-Grid: {out_path} '
                 f'({len(gdf_ease)} events)')
        p.uglc_path = out_path
    return p
 

# ==================================================================== #
# 6. LandScan population -- SUM-conserving warp, per-year
# ==================================================================== #

def reproject_landscan_population(p):
    """
    Reproject LandScan population data to EASE-Grid (EPSG:6933).
    LandScan population, per year. Native ~1km, 
    but not aligned with EASE-Grid 1km, so warp to EASE-Grid 1km. 
    SUM-conserving warp (resample_method='sum') to preserve total population counts.
    """
    if p.run_this:
        for year in p.data_processing_range:
            src_path = os.path.join(
                p.base_data_dir, 'landscan', f'landscan-global-{year}.tif'
            )
            if not os.path.exists(src_path):
                p.L.warning(f'LandScan {year} not found, skipping year.')
                continue
 
            out_path = os.path.join(
                p.input_data_dir, 'landscan_1km', f'landscan_{year}_1km.tif'
            )
            if os.path.exists(out_path) and not p.force_run:
                continue
 
            warp_to_reference(
                src_path, out_path, p.ease_grid_reference_path,
                resample_method='sum',        # conserve total population
                src_nodata=-2147483647,
                dst_nodata=-9999.0,
                output_type=gdal.GDT_Float32,
            )
            p.L.info(f'LandScan {year} reprojected (sum-conserving): {out_path}')
    return p


# ==================================================================== #
# 7. SoilGrids --- combine 0-30cm depth layers, unit conversion, warp
# ==================================================================== #

SOILGRIDS_PROPERTIES = {
    'sand_pct': ('sand', 10),
    'clay_pct': ('clay', 10),
    'org_carbon_pct': ('soc', 10),
    'bulk_density': ('bdod', 100),
}
def reproject_soilgrids_properties(p):
    if p.run_this:
        work_dir = os.path.join(p.input_data_dir, 'soilgrids_work')
        os.makedirs(work_dir, exist_ok=True)
        p.soilgrids_paths = {}

        for out_name, (prop_code, conv_factor) in SOILGRIDS_PROPERTIES.items():
            out_path = os.path.join(p.input_data_dir, f'soilgrids_{out_name}_1km.tif')
            if os.path.exists(out_path) and not p.force_run:
                p.soilgrids_paths[out_name] = out_path
                continue

            depth_paths_local = {
                depth: os.path.join(
                    p.base_data_dir, 'soilgrids', f'{prop_code}_{depth}_mean.tif'
                )
                for depth in DEPTH_WEIGHTS_0_30CM
            }
            for depth, path in depth_paths_local.items():
                if not os.path.exists(path):
                    raise FileNotFoundError(f'Missing SoilGrids TIF: {path}')

            native_combined_path = os.path.join(work_dir, f'{out_name}_native.tif')
            thickness_weighted_combine(
                depth_paths_local, native_combined_path, conv_factor=conv_factor
            )
            p.L.info(f'{out_name}: combined 0-30cm + unit-converted -> {native_combined_path}')

            warp_to_reference(
                native_combined_path, out_path, p.ease_grid_reference_path,
                resample_method='average',
                src_nodata=-9999.0, dst_nodata=-9999.0,
                output_type=gdal.GDT_Float32,
            )
            p.soilgrids_paths[out_name] = out_path
            p.L.info(f'{out_name} warped to EASE-Grid: {out_path}')
    return p



# ==================================================================== #
# 8. WorldClim BIO12 -- climatological rain (q input)
# ==================================================================== #

def reproject_worldclim_bio12(p):
    """Mean annual precipitation (mm), 1970-2000 climate normal."""
    if p.run_this:
        src_path = os.path.join(p.base_data_dir, 'worldclim', 'wc2.1_30s_bio_12.tif')
        out_path = os.path.join(p.input_data_dir, 'worldclim_bio12_1km.tif')
        
        if os.path.exists(out_path) and not p.force_run:
            p.climatological_rain_path = out_path
            return p
        
        # Intermediate step: replace negatives with -9999.0 using raster_calculator
        temp_path = out_path.replace('.tif', '_temp_cleaned.tif')
        
        def replace_negatives(data):
            """Replace all negative values with -9999.0"""
            data[data < 0] = -9999.0
            return data
        
        pygeo.raster_calculator(
            base_raster_path_band_const_list=[(src_path, 1)],
            local_op=replace_negatives,
            target_raster_path=temp_path,
            datatype_target=gdal.GDT_Float32,
            nodata_target=-9999.0,
            calc_raster_stats=True,
            raster_driver_creation_tuple=('GTIFF', ('TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))
        )
        
        # Warp the cleaned file
        warp_to_reference(
            temp_path, out_path, p.ease_grid_reference_path,
            resample_method='average',
            src_nodata=-9999.0,
            dst_nodata=-9999.0,
            output_type=gdal.GDT_Float32,
        )
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        p.L.info(f'WorldClim BIO12 reprojected: {out_path}')
        p.climatological_rain_path = out_path
    
    return p


# ==================================================================== #
# 9. HiHydroSoil K_sat -- combine + warp
# ==================================================================== #

def reproject_hihydrosoil_ksat(p):
    if p.run_this:
        work_dir = os.path.join(p.input_data_dir, 'hihydrosoil_work')
        os.makedirs(work_dir, exist_ok=True)

        out_path = os.path.join(p.input_data_dir, 'ksat_1km.tif')
        if os.path.exists(out_path) and not p.force_run:
            p.ksat_path = out_path
            return p

        depth_paths_local = {
            depth: os.path.join(p.base_data_dir, 'hihydrosoil', f'Ksat_{depth}_M_250m.tif')
            for depth in DEPTH_WEIGHTS_0_30CM
        }
        for depth, path in depth_paths_local.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f'Missing HiHydroSoil file: {path}')

        native_combined_path = os.path.join(work_dir, 'ksat_native.tif')
        thickness_weighted_combine(depth_paths_local, native_combined_path)
        p.L.info(f'K_sat combined 0-30cm: {native_combined_path}')

        warp_to_reference(
            native_combined_path, out_path, p.ease_grid_reference_path,
            resample_method='average',
            src_nodata=-9999.0, dst_nodata=-9999.0,
            output_type=gdal.GDT_Float32,
        )
        p.L.info(f'K_sat warped to EASE-Grid: {out_path}')
        p.ksat_path = out_path
    return p


# ==================================================================== #
# 10. Soil depth (b) -- Pelletier/ORNL DAAC, single file
# ==================================================================== #

def reproject_soil_depth(p):
    if p.run_this:
        src_path = os.path.join(
            p.base_data_dir, 'Global_Soil_Regolith_Sediment_1304', 'data',
            'average_soil_and_sedimentary-deposit_thickness.tif'
        )
        out_path = os.path.join(p.input_data_dir, 'soil_depth_1km.tif')
        if os.path.exists(out_path) and not p.force_run:
            p.soil_depth_path = out_path
            return p

        warp_to_reference(
            src_path, out_path, p.ease_grid_reference_path,
            resample_method='average',
            src_nodata=-1.0, dst_nodata=-9999.0,
            output_type=gdal.GDT_Float32,
        )
        p.L.info(f'Soil depth (Pelletier/ORNL) reprojected: {out_path}')
        p.soil_depth_path = out_path
    return p

# ==================================================================== #
# 11. GRIP4 road density -- continuous covariate, m of road per km^2
# ==================================================================== #

def reproject_grip_roads(p):
    """GRIP4 road density (m of road per km^2), the severity-model
    covariate.

    Per GRIP4 ReadMe: WGS84 lat/lon, 5 arcminute cells (~9.26km at the
    equator) -- UPSAMPLES (coarse ~9km source -> 1km target)
    """
    if p.run_this:
        src_path = os.path.join(
            p.base_data_dir, 'GRIP4_density_total', 'grip4_total_dens_m_km2.asc'
        )
        out_path = os.path.join(p.input_data_dir, 'road_density_1km.tif')
        if os.path.exists(out_path) and not p.force_run:
            p.road_density_path = out_path
            return p

        # .asc files often lack an embedded CRS; GRIP4's ReadMe confirms
        # WGS84 lat/lon, so assign it explicitly
        probe_ds = gdal.Open(src_path)
        has_crs = probe_ds.GetProjection() not in (None, '')
        probe_ds = None

        src_path_for_warp = src_path
        if not has_crs:
            vrt_path = os.path.join(p.input_data_dir, 'grip_roads_work', 'grip4_wgs84.vrt')
            os.makedirs(os.path.dirname(vrt_path), exist_ok=True)
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)
            gdal.Translate(vrt_path, src_path, outputSRS=srs.ExportToWkt())
            src_path_for_warp = vrt_path
            p.L.info('Assigned WGS84 CRS to GRIP4 .asc (confirmed via ReadMe.txt, '
                      'not embedded in the source file).')

        warp_to_reference(
            src_path_for_warp, out_path, p.ease_grid_reference_path,
            resample_method='bilinear',  # UPSAMPLING ~9km -> 1km, not downsampling
            src_nodata=-9999,
            dst_nodata=-9999.0,
            output_type=gdal.GDT_Float32,
        )
        p.L.info(f'GRIP4 road density reprojected: {out_path}')
        p.road_density_path = out_path
    return p



# ==================================================================== #
# 12. ERA5 daily-max rainfall
# ==================================================================== #

def reproject_rain_daily(p):
    """ERA5-Land annual max daily rainfall, per year. Native ~0.1deg
    (~11km at the equator) -- COARSER than the 1km target, so this
    UPSAMPLES. 'bilinear', not 'average' (same class as GRIP roads).
    """
    if p.run_this:
 
        for year in p.data_processing_range:
            src_path = os.path.join(
                p.base_data_dir, 'era5_land_precip_annual_tif',
                f'era5_max_daily_mm_{year}.tif'
            )
            if not os.path.exists(src_path):
                p.L.warning(f'ERA5 max daily rain {year} not found at {src_path}, skipping.')
                continue
 
            out_path = os.path.join(
                p.input_data_dir, 'era5_land', f'era5_max_daily_mm_{year}.tif'
            )
            if os.path.exists(out_path) and not p.force_run:
                continue
 
            warp_to_reference(
                src_path, out_path, p.ease_grid_reference_path,
                resample_method='bilinear',  # UPSAMPLING ~11km -> 1km
                src_nodata=-9999.0, dst_nodata=-9999.0,
                output_type=gdal.GDT_Float32,
            )
            p.L.info(f'ERA5 max daily rain {year} reprojected: {out_path}')
    return p


# ==================================================================== #
# 13. Final validation
# ==================================================================== #

def validate_input_rasters(p):
    """Checks every reprojected raster against the EASE-Grid reference:
    matching size, pixel resolution, projection. Run last, after every
    other input_data_tasks.py task.
    """
    if p.run_this:
        ref_info = pygeo.get_raster_info(p.ease_grid_reference_path)
        ref_size = ref_info['raster_size']
        ref_pixel = ref_info['pixel_size']

        paths_to_check = {}
        for attr, label in [
            ('dem_path', 'dem'), ('gaez_path', 'gaez'),
            ('ksat_path', 'ksat'), ('soil_depth_path', 'soil_depth'),
            ('climatological_rain_path', 'climatological_rain'),
            ('road_density_path', 'road_density'),
        ]:
            if hasattr(p, attr):
                paths_to_check[label] = getattr(p, attr)

        if hasattr(p, 'soilgrids_paths'):
            paths_to_check.update({f'soilgrids_{k}': v for k, v in p.soilgrids_paths.items()})

        for subdir, label in [('landscan_1km', 'landscan'), ('forest_share_1km', 'forest_share')]:
            year_dir = os.path.join(p.input_data_dir, subdir)
            if os.path.isdir(year_dir):
                for fname in os.listdir(year_dir):
                    if fname.endswith('.tif'):
                        paths_to_check[f'{label}_{fname}'] = os.path.join(year_dir, fname)

        errors = []
        for label, path in paths_to_check.items():
            if not os.path.exists(path):
                errors.append(f'{label}: MISSING file at {path}')
                continue
            info = pygeo.get_raster_info(path)
            if info['raster_size'] != ref_size:
                errors.append(f'{label}: size mismatch {info["raster_size"]} != {ref_size}')
            if abs(info['pixel_size'][0] - ref_pixel[0]) > 1e-6:
                errors.append(f'{label}: pixel size mismatch {info["pixel_size"]} != {ref_pixel}')
            if info['projection_wkt'] != ref_info['projection_wkt']:
                errors.append(f'{label}: projection WKT differs from reference')
                # NOTE: WKT string comparison is fragile -- if this false-
                # positives in practice, compare EPSG codes via osr instead.

        if errors:
            p.L.error(f'Raster validation found {len(errors)} issue(s):')
            for e in errors:
                p.L.error(f'  - {e}')
            raise RuntimeError(
                f'{len(errors)} input raster(s) failed validation -- see log above.'
            )
        else:
            p.L.info(f'All {len(paths_to_check)} input rasters validated against EASE-Grid reference.')
    return p
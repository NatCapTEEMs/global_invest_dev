"""landslide_mitigation ProjectFlow tasks (folded from m-braaksma/landslide_mitigation v0.2.0).
One tasks module per the service template; the source repo's five stage files survive as the
section banners below (input data -> preprocessing -> stability model -> valuation ->
tables/figures), bodies unchanged.
"""
import os
import numpy as np
import pandas as pd
from osgeo import gdal, osr
import geopandas as gpd
import pygeoprocessing as pygeo
import hazelbean as hb
import pygeoprocessing.routing as routing
from global_invest import utilities
from global_invest.landslide_mitigation.landslide_mitigation_functions import (
    parse_gpd_grid_definition,
    warp_to_reference,
    sample_raster_at_points,
    write_raster_from_array,
    thickness_weighted_combine,
    compute_si_global,
    DEPTH_WEIGHTS_0_30CM,
)
import json
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score
import glob
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


# =============================================================================
# SECTION: input data tasks (was input_data_tasks.py in the source repo)
# =============================================================================
"""
input_data_tasks.py

Raw sources read from p.landslide_input_data_dir (input_data_raw/), except ESA-CCI and the
shared statics, which resolve via p.get_path (machine-agnostic; base_data on this machine).
All reprojected outputs write into p.input_data_dir.
"""


# ==================================================================== #
# 0. Parent dir-creator
# ==================================================================== #

# METHOD CONSTANTS defining the ported v0.2.0 science (the landslide author's to bless) -- in code
# so a change costs a reviewed commit; publish_inputs applies them caller-wins, so deliberate overrides survive.
LANDSLIDE_DATA_PROCESSING_YEARS = list(range(2007, 2020))
LANDSLIDE_MODELING_YEARS = list(range(2007, 2019))
LANDSLIDE_PREDICTION_YEARS = [2019]
LANDSLIDE_MAX_LOCATION_ACCURACY_M = 1000
LANDSLIDE_CONTROL_RATIO = 25
LANDSLIDE_C_ROOT_SCENARIOS = {'observed': 'observed', 'full_impacts': 0}


def publish_inputs(p):
    """Every landslide task's first line: the es_config row and run knobs (defaults layer -- a
    caller-set value wins), the method constants above, the module's input alias, the logger,
    and the results registry. Idempotent, so the tree builder may also call it to read the
    iterator knobs at build time."""
    utilities.hydrate_es_config(p, 'landslide_mitigation', log=hb.log)
    utilities.hydrate_es_parameters(p, 'landslide_mitigation', log=hb.log)
    if getattr(p, 'landslide_input_data_dir', None) is None:
        p.landslide_input_data_dir = p.gep_quantity_input_path   # the module's descriptive alias
    for attribute, value in (('data_processing_range', LANDSLIDE_DATA_PROCESSING_YEARS),
                             ('modeling_range', LANDSLIDE_MODELING_YEARS),
                             ('prediction_years', LANDSLIDE_PREDICTION_YEARS),
                             ('max_location_accuracy_m', LANDSLIDE_MAX_LOCATION_ACCURACY_M),
                             ('control_ratio', LANDSLIDE_CONTROL_RATIO),
                             ('c_root_scenarios', LANDSLIDE_C_ROOT_SCENARIOS)):
        if getattr(p, attribute, None) is None:
            setattr(p, attribute, value)
    if getattr(p, 'L', None) is None:
        p.L = hb.get_logger('landslide_mitigation_workflow')
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def input_data(p):
    """Creates p.input_data_dir. All other tasks below write into it."""
    publish_inputs(p)
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
    publish_inputs(p)
    if p.run_this:
        gpd_path = os.path.join(p.landslide_input_data_dir, 'nsidc_proj', 'EASE2_M01km.gpd')
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
    publish_inputs(p)
    if p.run_this:
        src_path = p.get_path('seals', 'static_regressors', 'alt_m.tif')
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
    publish_inputs(p)
    if p.run_this:
        src_path = os.path.join(p.landslide_input_data_dir, 'fao_gaez', 'GAEZ-V5.AEZ57.tif')
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
    publish_inputs(p)
    if p.run_this:
        work_dir = os.path.join(p.input_data_dir, 'esacci_work')
        os.makedirs(work_dir, exist_ok=True)

        for year in p.data_processing_range:
            src_path = p.get_path('lulc', 'esa', f'lulc_esa_{year}.tif',
                                  raise_error_if_fail=False)
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
    publish_inputs(p)
    if p.run_this:
 
        src_path = os.path.join(p.landslide_input_data_dir, 'uglc', 'UGLC_point.csv')
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
    publish_inputs(p)
    if p.run_this:
        for year in p.data_processing_range:
            src_path = os.path.join(
                p.landslide_input_data_dir, 'landscan', f'landscan-global-{year}.tif'
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
    publish_inputs(p)
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
                    p.landslide_input_data_dir, 'soilgrids', f'{prop_code}_{depth}_mean.tif'
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
    publish_inputs(p)
    if p.run_this:
        src_path = os.path.join(p.landslide_input_data_dir, 'worldclim', 'wc2.1_30s_bio_12.tif')
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
    publish_inputs(p)
    if p.run_this:
        work_dir = os.path.join(p.input_data_dir, 'hihydrosoil_work')
        os.makedirs(work_dir, exist_ok=True)

        out_path = os.path.join(p.input_data_dir, 'ksat_1km.tif')
        if os.path.exists(out_path) and not p.force_run:
            p.ksat_path = out_path
            return p

        depth_paths_local = {
            depth: os.path.join(p.landslide_input_data_dir, 'hihydrosoil', f'Ksat_{depth}_M_250m.tif')
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
    publish_inputs(p)
    if p.run_this:
        src_path = os.path.join(
            p.landslide_input_data_dir, 'Global_Soil_Regolith_Sediment_1304', 'data',
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
    publish_inputs(p)
    if p.run_this:
        src_path = os.path.join(
            p.landslide_input_data_dir, 'GRIP4_density_total', 'grip4_total_dens_m_km2.asc'
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
    publish_inputs(p)
    if p.run_this:
 
        for year in p.data_processing_range:
            src_path = os.path.join(
                p.landslide_input_data_dir, 'era5_land_precip_annual_tif',
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
    publish_inputs(p)
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

# =============================================================================
# SECTION: preprocessing tasks (was preprocessing_tasks.py in the source repo)
# =============================================================================
"""
preprocessing_tasks.py
"""


C_ROOT_MAX_KPA = 5.0 
GRAVITY_M_S2 = 9.81

# ==================================================================== #
# 0. Parent dir-creator
# ==================================================================== #


def preprocessing(p):
    """Creates p.preprocessing_dir. All other tasks below write into it."""
    publish_inputs(p)
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
    publish_inputs(p)
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
    publish_inputs(p)
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
    publish_inputs(p)
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
    publish_inputs(p)
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
    publish_inputs(p)
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
        raw_dem_src = p.get_path('seals', 'static_regressors', 'alt_m.tif')
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
    publish_inputs(p)
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
    publish_inputs(p)
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
    publish_inputs(p)
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
    publish_inputs(p)
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


# =============================================================================
# SECTION: model tasks (was model_tasks.py in the source repo)
# =============================================================================
"""
model_tasks.py
"""


def modeling(p):
    """
    Task to create a modeling_dir for the downstream tasks.
    """
    publish_inputs(p)
    if p.run_this:
        os.makedirs(p.modeling_dir, exist_ok=True)
        return p
    return p


def calibrate_si_to_probability(p):
    publish_inputs(p)
    if p.run_this:
        out_coef_path = os.path.join(p.modeling_dir, 'hazard_model_coefficients.json')
        if os.path.exists(out_coef_path) and not p.force_run:
            with open(out_coef_path) as f:
                p.hazard_model_coefficients = json.load(f)
            return p
 
        panel = pd.read_csv(p.estimation_table_path)
 
        train = panel[panel['year'].isin(set(p.modeling_range))].copy()
        holdout = panel[panel['year'].isin(set(p.prediction_years))].copy()
 
        # ---- Fit raw logistic on the case-control sample ----
        X_train = sm.add_constant(train[['si_observed', 'rain_max_daily']])
        y_train = train['case']
        raw_model = sm.Logit(y_train, X_train).fit(disp=False)
        p.L.info(f'Raw case-control logistic fit:\n{raw_model.summary()}')
 
        # ---- Case-control intercept correction (Prentice & Pyke, 1979) ----
        # Slope coefficients from ordinary logistic on case-control data
        # are consistent estimates of the true population slopes -- but
        # the intercept is biased because the sample over-represents
        # cases relative to the real population. Correct via the offset:
        #   alpha_corrected = alpha_raw - log((tau/(1-tau)) / (pi/(1-pi)))
        # where tau = fraction of cases in the SAMPLE, pi = fraction of
        # cases in the true POPULATION (estimated below from the UGLC
        # binary panels' actual hit-rate across all land pixel-years).
        tau = y_train.mean()
 
        # Estimate population prevalence pi from the UGLC annual binary
        # panels: (total event pixel-years) / (total land pixel-years)
        # across the SAME years used for training.
        total_event_pixels = 0
        total_land_pixels = 0
        for year in p.modeling_range:
            binary_path = os.path.join(
                p.preprocessing_dir, 'uglc_annual_panels', f'uglc_binary_{year}.tif'
            )
            si_path = p.si_paths.get('observed', {}).get(year)
            if not os.path.exists(binary_path) or not si_path or not os.path.exists(si_path):
                continue
 
            ds = gdal.Open(binary_path)
            band = ds.GetRasterBand(1)
            arr = band.ReadAsArray()
            nodata = band.GetNoDataValue()
            ds = None
 
            si_ds = gdal.Open(si_path)
            si_band = si_ds.GetRasterBand(1)
            si_arr = si_band.ReadAsArray()
            si_nodata = si_band.GetNoDataValue()
            si_ds = None
 
            valid = (arr != nodata) if nodata is not None else np.ones_like(arr, dtype=bool)
            si_valid = (si_arr != si_nodata) if si_nodata is not None else np.ones_like(si_arr, dtype=bool)
            valid &= si_valid  # restrict to the SI-eligible (slope-passing) population
 
            total_event_pixels += int((arr[valid] == 1).sum())
            total_land_pixels += int(valid.sum())
 
        pi = total_event_pixels / total_land_pixels if total_land_pixels > 0 else tau
        p.L.info(f'Sample case fraction (tau) = {tau:.6f}, '
                  f'estimated population prevalence (pi) = {pi:.8f}')
 
        offset = np.log((tau / (1 - tau)) / (pi / (1 - pi)))
        alpha_corrected = raw_model.params['const'] - offset
 
        coefficients = {
            'alpha_raw': float(raw_model.params['const']),
            'alpha_corrected': float(alpha_corrected),
            'beta_si': float(raw_model.params['si_observed']),
            'beta_rain': float(raw_model.params['rain_max_daily']),
            'tau_sample_case_fraction': float(tau),
            'pi_population_prevalence': float(pi),
            'n_train': int(len(train)),
            'pseudo_r_squared': float(raw_model.prsquared),
            'std_err': {k: float(v) for k, v in raw_model.bse.items()},
            'z_stat': {k: float(v) for k, v in raw_model.tvalues.items()},
            'p_value': {k: float(v) for k, v in raw_model.pvalues.items()},
        }
 
        # ---- Training-set AUC, for comparison against the held-out AUC.
        train_pred_raw = raw_model.predict(X_train)
        train_auc = roc_auc_score(y_train, train_pred_raw)
        coefficients['train_auc'] = float(train_auc)
        p.L.info(f'Training AUC: {train_auc:.4f}')

        # ---- Validate on held-out prediction_years 
        if len(holdout) > 0:
            X_holdout = sm.add_constant(
                holdout[['si_observed', 'rain_max_daily']], has_constant='add'
            )
            holdout_pred_raw = raw_model.predict(X_holdout)
            auc = roc_auc_score(holdout['case'], holdout_pred_raw)
            coefficients['holdout_auc'] = float(auc)
            p.L.info(f'Held-out ({sorted(set(p.prediction_years))}) AUC: {auc:.4f}')

        with open(out_coef_path, 'w') as f:
            json.dump(coefficients, f, indent=2)
        p.L.info(f'Hazard model coefficients saved: {out_coef_path}')
        p.hazard_model_coefficients = coefficients
    return p
 
 


def estimate_severity_model(p):
    publish_inputs(p)
    if p.run_this:
        out_coef_path = os.path.join(p.modeling_dir, 'severity_model_coefficients.json')
        if os.path.exists(out_coef_path) and not p.force_run:
            with open(out_coef_path) as f:
                p.severity_model_coefficients = json.load(f)
            return p
 
        panel = pd.read_csv(p.estimation_table_path)
        train = panel[
            (panel['year'].isin(set(p.modeling_range))) & (panel['case'] == 1)
        ].copy()
 
        train['population_log1p'] = np.log1p(train['population'].clip(lower=0))
        train['fatality_occurred'] = (train['fatality_count'] > 0).astype(int)
 
        base_formula = 'population_log1p + rain_max_daily + slope_degrees + road_density'
 
        # ---- Part A: P(fatality > 0 | landslide occurred) ----
        logit_formula = f'fatality_occurred ~ {base_formula}'
        try:
            part_a_model = smf.logit(logit_formula, data=train).fit(disp=False)
        except np.linalg.LinAlgError:
            p.L.warning(
                'Standard MLE logit fit failed (likely separation even '
                'without GAEZ FE) -- falling back to L2-regularized fit. '
                'Coefficients will be shrunk toward 0 and standard errors '
                'from the regularized fit are not directly comparable to '
                'an unpenalized fit -- treat this as a point-estimate '
                'fallback, not full inference.'
            )
            part_a_model = smf.logit(logit_formula, data=train).fit_regularized(
                alpha=1.0, disp=False
            )
        p.L.info(f'Severity part A (fatality occurrence | landslide):\n{part_a_model.summary()}')
 
        # ---- Part B: E[log(fatalities) | fatality > 0] ----
        positive = train[train['fatality_count'] > 0].copy()
        if len(positive) < 10:
            p.L.warning(
                f'Only {len(positive)} fatal-landslide rows available for '
                f'part B -- coefficients will be unstable. Consider a '
                f'coarser covariate set or pooling more years.'
            )
        positive['log_fatalities'] = np.log(positive['fatality_count'])
        ols_formula = f'log_fatalities ~ {base_formula}'
        part_b_model = smf.ols(ols_formula, data=positive).fit()
        p.L.info(f'Severity part B (log fatalities | fatal):\n{part_b_model.summary()}')
 
        # ---- Duan's smearing correction ----
        # exp(predicted log fatalities) is a biased (too-low) estimate of
        # E[fatalities] due to Jensen's inequality on the log transform.
        # Duan (1983) smearing: multiply by the mean of exp(residuals)
        # from the log-scale fit, rather than assuming log-normal errors.
        residuals = part_b_model.resid
        smearing_factor = float(np.mean(np.exp(residuals)))
        p.L.info(f"Duan's smearing factor: {smearing_factor:.4f}")
 
        coefficients = {
            'part_a_params': {k: float(v) for k, v in part_a_model.params.items()},
            'part_a_std_err': {k: float(v) for k, v in part_a_model.bse.items()},
            'part_a_z_stat': {k: float(v) for k, v in part_a_model.tvalues.items()},
            'part_a_p_value': {k: float(v) for k, v in part_a_model.pvalues.items()},
            'part_a_pseudo_r_squared': float(part_a_model.prsquared),
            'part_b_params': {k: float(v) for k, v in part_b_model.params.items()},
            'part_b_std_err': {k: float(v) for k, v in part_b_model.bse.items()},
            'part_b_t_stat': {k: float(v) for k, v in part_b_model.tvalues.items()},
            'part_b_p_value': {k: float(v) for k, v in part_b_model.pvalues.items()},
            'part_b_r_squared': float(part_b_model.rsquared),
            'smearing_factor': smearing_factor,
            'n_train_landslides': int(len(train)),
            'n_train_fatal': int(len(positive)),
        }
 
        with open(out_coef_path, 'w') as f:
            json.dump(coefficients, f, indent=2)
        p.L.info(f'Severity model coefficients saved: {out_coef_path}')
        p.severity_model_coefficients = coefficients
    return p


def estimate_severity_model_si_sensitivity(p):
    """
    Sensitivity check: does SI subsume slope's information in severity?
 
    Diagnostic only -- writes to a separate output, does NOT overwrite
    the main severity_model_coefficients.json used by prediction.
    """
    publish_inputs(p)
    if p.run_this:
        out_coef_path = os.path.join(p.modeling_dir, 'severity_model_si_sensitivity.json')
        if os.path.exists(out_coef_path) and not p.force_run:
            with open(out_coef_path) as f:
                p.severity_model_si_sensitivity = json.load(f)
            return p
 
        panel = pd.read_csv(p.estimation_table_path)
        train = panel[
            (panel['year'].isin(set(p.modeling_range))) & (panel['case'] == 1)
        ].copy()
 
        train['population_log1p'] = np.log1p(train['population'].clip(lower=0))
        train['fatality_occurred'] = (train['fatality_count'] > 0).astype(int)
 
        # slope_degrees -> si_observed, road_density kept
        si_formula = 'population_log1p + rain_max_daily + road_density + si_observed'
 
        logit_formula = f'fatality_occurred ~ {si_formula}'
        part_a_model = smf.logit(logit_formula, data=train).fit(disp=False)
        p.L.info(f'[SI sensitivity] Severity part A:\n{part_a_model.summary()}')
 
        positive = train[train['fatality_count'] > 0].copy()
        positive['log_fatalities'] = np.log(positive['fatality_count'])
        ols_formula = f'log_fatalities ~ {si_formula}'
        part_b_model = smf.ols(ols_formula, data=positive).fit()
        p.L.info(f'[SI sensitivity] Severity part B:\n{part_b_model.summary()}')
 
        residuals = part_b_model.resid
        smearing_factor = float(np.mean(np.exp(residuals)))
 
        result = {
            'formula': si_formula,
            'part_a_pseudo_r_squared': float(part_a_model.prsquared),
            'part_a_params': {k: float(v) for k, v in part_a_model.params.items()},
            'part_a_p_value': {k: float(v) for k, v in part_a_model.pvalues.items()},
            'part_b_r_squared': float(part_b_model.rsquared),
            'part_b_params': {k: float(v) for k, v in part_b_model.params.items()},
            'part_b_p_value': {k: float(v) for k, v in part_b_model.pvalues.items()},
            'smearing_factor': smearing_factor,
            'n_train_landslides': int(len(train)),
            'n_train_fatal': int(len(positive)),
        }
 
        # Quick side-by-side vs. the baseline (slope, no SI)
        baseline_path = os.path.join(p.modeling_dir, 'severity_model_coefficients.json')
        if os.path.exists(baseline_path):
            with open(baseline_path) as f:
                baseline = json.load(f)
            p.L.info(
                f'\nCOMPARISON (baseline slope-based vs. SI-based):\n'
                f'  Part A pseudo R2: {baseline.get("part_a_pseudo_r_squared"):.4f} '
                f'(baseline) vs. {result["part_a_pseudo_r_squared"]:.4f} (SI)\n'
                f'  Part B R2:        {baseline.get("part_b_r_squared"):.4f} '
                f'(baseline) vs. {result["part_b_r_squared"]:.4f} (SI)'
            )
 
        with open(out_coef_path, 'w') as f:
            json.dump(result, f, indent=2)
        p.L.info(f'SI sensitivity results saved: {out_coef_path}')
        p.severity_model_si_sensitivity = result
    return p


# =============================================================================
# SECTION: valuation tasks (was valuation_tasks.py in the source repo)
# =============================================================================
"""
valuation_tasks.py
"""


def tile_zones(p):
    """Generate tile zones for parallel prediction, filtered to
    land-containing tiles only. Tiles are defined on the EASE-Grid
    reference's exact dimensions/transform via p.gaez_path 
    """
    publish_inputs(p)
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
    publish_inputs(p)
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
    publish_inputs(p)
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
    publish_inputs(p)
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
    publish_inputs(p)
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
    publish_inputs(p)
    if p.run_this:
        out_path = os.path.join(p.valuation_dir, 'vsl_usd_2019_1km.tif')
        if os.path.exists(out_path) and not p.force_run:
            p.vsl_raster_path = out_path
            return p
 
        # ---- 1. Parse OECD VSL CSV ----
        # NOTE: has commas/special characters in
        # the name as downloaded, glob to avoid a brittle hardcoded match.
        oecd_candidates = glob.glob(os.path.join(p.landslide_input_data_dir, 'oecd_vsl', '*.csv'))
        if not oecd_candidates:
            raise FileNotFoundError(f'No CSV found in {p.landslide_input_data_dir}/oecd_vsl/')
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
        correspondence_path = p.get_path('cartographic', 'ee', 'ee_r264_correspondence.gpkg')
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
    publish_inputs(p)
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


# =============================================================================
# SECTION: tables figures tasks (was tables_figures_tasks.py in the source repo)
# =============================================================================
"""
tables_figures_tasks.py
"""


# optional, map still renders without a basemap
try:
    import contextily as ctx
except ImportError:
    ctx = None


def tables_figures(p):
    """Creates a directory for tables and figures."""
    publish_inputs(p)
    if p.run_this:
        return p
    return p    


def compute_zonal_statistics(p):
    publish_inputs(p)
    if p.run_this:
        for year in p.prediction_years:
            csv_out_path = os.path.join(p.tables_figures_dir, f'zonal_statistics_{year}.csv')
            gpkg_out_path = os.path.join(p.tables_figures_dir, f'zonal_statistics_{year}.gpkg')
 
            if os.path.exists(csv_out_path) and os.path.exists(gpkg_out_path) and not p.force_run:
                p.L.info(f'{year}: zonal statistics already exist, skipping.')
                continue
 
            avoided_mortality_path = os.path.join(
                p.valuation_dir, f'avoided_mortality_{year}.tif'
            )
            avoided_mortality_value_path = os.path.join(
                p.valuation_dir, f'avoided_mortality_value_{year}.tif'
            )
 
            # Reuse the same EASE-Grid-reprojected correspondence vector
            # built for build_vsl_raster, rather than re-reprojecting
            corr_ease_path = os.path.join(p.valuation_dir, 'vsl_work', 'corr_ease_vsl.gpkg')
            if not os.path.exists(corr_ease_path):
                raise FileNotFoundError(
                    f'{corr_ease_path} not found -- run build_vsl_raster first '
                    f'(it produces this reprojected correspondence vector).'
                )
 
            corr = gpd.read_file(corr_ease_path, fid_as_index=True)
 
            id_field = 'ee_r264_id'
 
            deaths_stats = pygeo.zonal_statistics(
                (avoided_mortality_path, 1), corr_ease_path,
            )
            value_stats = pygeo.zonal_statistics(
                (avoided_mortality_value_path, 1), corr_ease_path,
            )
 
            corr['avoided_deaths_sum'] = [
                deaths_stats.get(fid, {}).get('sum', None) for fid in corr.index
            ]
            corr['avoided_value_sum_usd'] = [
                value_stats.get(fid, {}).get('sum', None) for fid in corr.index
            ]
            corr['pixel_count'] = [
                deaths_stats.get(fid, {}).get('count', None) for fid in corr.index
            ]
 
            # ---- CSV ----
            csv_cols = [id_field, 'iso3_r250_label', 'name_long', 'income_grp',
                        'region_wb', 'avoided_deaths_sum', 'avoided_value_sum_usd',
                        'pixel_count']
            csv_cols = [c for c in csv_cols if c in corr.columns]
            corr[csv_cols].to_csv(csv_out_path, index=False)
            p.L.info(f'Zonal statistics CSV: {csv_out_path}')
 
            # ---- GPKG (stats + geometry, for mapping) ----
            corr.to_file(gpkg_out_path, driver='GPKG')
            p.L.info(f'Zonal statistics GPKG: {gpkg_out_path}')
 
            total_deaths = corr['avoided_deaths_sum'].sum()
            total_value = corr['avoided_value_sum_usd'].sum()
            p.L.info(f'{year} global totals (zonal): '
                      f'{total_deaths:.4f} avoided deaths, '
                      f'${total_value:,.2f} avoided value')
    return p


def _regression_stars(pval):
    if pd.isna(pval):
        return ''
    if pval < 0.001:
        return '***'
    if pval < 0.01:
        return '**'
    if pval < 0.05:
        return '*'
    return ''
 
 


def _build_publication_table(coef_rows, bottom_rows, dep_label):
    """coef_rows: list of (label, coefficient, std_error, p_value).
    bottom_rows: list of (label, value_string) appended after the
    coefficient block (N, R^2, fixed effects, etc).
    Returns a two-row-per-variable DataFrame: coefficient+stars on one
    row, blank label + '(SE)' on the row beneath - standard econometrics
    table layout.
    """
    rows = []
    for label, coef, se, pval in coef_rows:
        stars = _regression_stars(pval)
        rows.append({' ': label, dep_label: f'{coef:.4f}{stars}'})
        se_text = f'({se:.4f})' if pd.notna(se) else ''
        rows.append({' ': '', dep_label: se_text})
 
    rows.append({' ': '', dep_label: ''})
    for label, value_str in bottom_rows:
        rows.append({' ': label, dep_label: value_str})
 
    return pd.DataFrame(rows)
 
 


def _save_table_multiple_formats(table_df, base_path, title, notes_text):
    """
    Saves {base_path}.csv, .tex, .md 
    """
    csv_path = f'{base_path}.csv'
    tex_path = f'{base_path}.tex'
    md_path = f'{base_path}.md'
 
    table_df.to_csv(csv_path, index=False)
 
    tex_lines = [
        '\\centering',
        table_df.to_latex(index=False, escape=True),
        '\\vspace{2pt}',
        '\\begin{minipage}{0.92\\linewidth}',
        f'\\footnotesize Notes: {notes_text}',
        '\\end{minipage}',
        '',
    ]
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex_lines))
 
    md_lines = [
        table_df.to_markdown(index=False),
        '',
        f'Notes: {notes_text}',
        '',
    ]
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
 
    return csv_path, tex_path, md_path
 
 
VARIABLE_NOTES = {
    'si_observed': 'infinite-slope stability index (higher = more stable)',
    'rain_max_daily': 'annual maximum daily precipitation from ERA5-Land, mm',
    'population_log1p': 'log(1+population) from LandScan',
    'slope_degrees': 'terrain slope in degrees',
    'road_density': 'road length per square kilometer, GRIP4',
}
 
TERM_LABELS = {
    'const': 'Intercept',
    'Intercept': 'Intercept',
    'si_observed': 'Stability Index',
    'rain_max_daily': 'Extreme Precipitation',
    'population_log1p': 'Population (log scale)',
    'slope_degrees': 'Slope',
    'road_density': 'Road Density',
}
 
 


def export_regression_tables(p):
    publish_inputs(p)
    if p.run_this:
        out_dir = p.tables_figures_dir
 
        # ---- Hazard model ----
        hazard_model_coefficients_path = os.path.join(p.modeling_dir, 'hazard_model_coefficients.json')
        with open(hazard_model_coefficients_path) as f:
            hazard = json.load(f)
 
        hazard_coef_rows = [
            ('Intercept', hazard['alpha_raw'], hazard['std_err']['const'], hazard['p_value']['const']),
            (TERM_LABELS['si_observed'], hazard['beta_si'],
             hazard['std_err']['si_observed'], hazard['p_value']['si_observed']),
            (TERM_LABELS['rain_max_daily'], hazard['beta_rain'],
             hazard['std_err']['rain_max_daily'], hazard['p_value']['rain_max_daily']),
        ]
        hazard_bottom_rows = [
            ('Observations', f"{hazard.get('n_train'):,}"),
            ('Pseudo R-squared (McFadden)', f"{hazard.get('pseudo_r_squared'):.4f}"),
            ('Case-control corrected intercept', f"{hazard.get('alpha_corrected'):.4f}"),
            ('Training AUC', f"{hazard.get('train_auc'):.4f}"),
            ('Held-out AUC', f"{hazard.get('holdout_auc'):.4f}"),
            ('Fixed effects', 'No'),
        ]
        hazard_table = _build_publication_table(hazard_coef_rows, hazard_bottom_rows, 'Pr(Landslide)')
        hazard_notes = (
            'Logit model estimated on a stratified case-control sample; absolute probabilities '
            'corrected for case-control oversampling. Standard errors in '
            'parentheses. * p < 0.05, ** p < 0.01, *** p < 0.001. Variable definitions: '
            f"Stability Index = {VARIABLE_NOTES['si_observed']}; "
            f"Extreme Precipitation = {VARIABLE_NOTES['rain_max_daily']}."
        )
        _save_table_multiple_formats(
            hazard_table, os.path.join(out_dir, 'hazard_model_table'),
            'Hazard Model of Landslide Occurrence', hazard_notes,
        )
 
        # ---- Severity model: single table, hurdle and severity stages as columns ----
        severity_model_coefficients_path = os.path.join(p.modeling_dir, 'severity_model_coefficients.json')
        with open(severity_model_coefficients_path) as f:
            severity = json.load(f)

        def _severity_coef_by_term(params_key, se_key, pval_key):
            return {
                term: (coef, severity[se_key].get(term, np.nan), severity[pval_key].get(term, np.nan))
                for term, coef in severity[params_key].items()
            }

        part_a = _severity_coef_by_term('part_a_params', 'part_a_std_err', 'part_a_p_value')
        part_b = _severity_coef_by_term('part_b_params', 'part_b_std_err', 'part_b_p_value')
        term_order = list(part_a.keys()) + [t for t in part_b if t not in part_a]

        def _severity_stars(pval):
            if pd.isna(pval):
                return ''
            return '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''

        rows = []
        for term in term_order:
            label = TERM_LABELS.get(term, term)
            a_coef, a_se, a_p = part_a.get(term, (np.nan, np.nan, np.nan))
            b_coef, b_se, b_p = part_b.get(term, (np.nan, np.nan, np.nan))
            coef_row = {
                ' ': label,
                'Pr(Mortality > 0)': f'{a_coef:.4f}{_severity_stars(a_p)}' if pd.notna(a_coef) else '',
                'log(Fatalities)': f'{b_coef:.4f}{_severity_stars(b_p)}' if pd.notna(b_coef) else '',
            }
            se_row = {
                ' ': '',
                'Pr(Mortality > 0)': f'({a_se:.4f})' if pd.notna(a_se) else '',
                'log(Fatalities)': f'({b_se:.4f})' if pd.notna(b_se) else '',
            }
            rows.extend([coef_row, se_row])

        rows.extend([
            {' ': 'Observations',
             'Pr(Mortality > 0)': f"{severity.get('n_train_landslides'):,}",
             'log(Fatalities)': f"{severity.get('n_train_fatal'):,}"},
            {' ': 'Pseudo R-squared (McFadden)',
             'Pr(Mortality > 0)': f"{severity.get('part_a_pseudo_r_squared'):.4f}",
             'log(Fatalities)': ''},
            {' ': 'R-squared',
             'Pr(Mortality > 0)': '',
             'log(Fatalities)': f"{severity.get('part_b_r_squared'):.4f}"},
            {' ': "Duan's smearing factor",
             'Pr(Mortality > 0)': '',
             'log(Fatalities)': f"{severity.get('smearing_factor'):.4f}"},
            {' ': 'Fixed effects', 'Pr(Mortality > 0)': 'No', 'log(Fatalities)': 'No'},
        ])

        severity_table = pd.DataFrame(rows)
        severity_notes = (
            'Two-part hurdle model of landslide fatalities. Column (1) is the logistic hurdle stage, '
            'estimated on realized landslide occurrences only. Column (2) is the log-linear severity '
            "stage, estimated on the fatal-event subsample and back-transformed using Duan's (1983) "
            'smearing correction. Standard errors in parentheses. * p < 0.05, ** p < 0.01, *** p < 0.001. '
            'Variable definitions: '
            f"Population (log scale) = {VARIABLE_NOTES['population_log1p']}; "
            f"Extreme Precipitation = {VARIABLE_NOTES['rain_max_daily']}; "
            f"Slope = {VARIABLE_NOTES['slope_degrees']}; "
            f"Road Density = {VARIABLE_NOTES['road_density']}."
        )
        _save_table_multiple_formats(
            severity_table, os.path.join(out_dir, 'severity_combined_table'),
            'Two-Part Hurdle Model of Landslide Fatalities', severity_notes,
        )
 
        p.L.info(f'Publication-style regression tables (csv/tex/md) exported to {out_dir}')
    return p


def _save_table_safe(df, base_path, title, notes_text=None):
    csv_path = f'{base_path}.csv'
    tex_path = f'{base_path}.tex'
    md_path = f'{base_path}.md'
 
    df.to_csv(csv_path, index=False)
 
    tex_lines = ['\\centering', df.to_latex(index=False, escape=True)]
    if notes_text:
        tex_lines += [
            '\\vspace{2pt}',
            '\\begin{minipage}{0.92\\linewidth}',
            f'\\footnotesize Notes: {notes_text}',
            '\\end{minipage}',
        ]
    tex_lines.append('')
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex_lines))
 
    md_lines = [df.to_markdown(index=False), '']
    if notes_text:
        md_lines += [f'Notes: {notes_text}', '']
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
 
    return csv_path, tex_path, md_path
 
 


def export_results_tables(p):
    publish_inputs(p)
    if p.run_this:
        for year in p.prediction_years:
            zonal_csv_path = os.path.join(p.tables_figures_dir, f'zonal_statistics_{year}.csv')
            if not os.path.exists(zonal_csv_path):
                p.L.warning(f'{year}: {zonal_csv_path} not found, skipping.')
                continue
 
            df = pd.read_csv(zonal_csv_path)
 
            # ---- Global summary ----
            global_df = pd.DataFrame([{
                'Avoided mortality': df['avoided_deaths_sum'].sum(),
                'Value (US$ millions)': df['avoided_value_sum_usd'].sum() / 1_000_000,
            }])
            global_df['Avoided mortality'] = global_df['Avoided mortality'].map(lambda v: f'{v:,.2f}')
            global_df['Value (US$ millions)'] = global_df['Value (US$ millions)'].map(lambda v: f'${v:,.2f}')
            _save_table_safe(
                global_df, os.path.join(p.tables_figures_dir, f'global_summary_full_impacts_{year}'),
                f'Global Avoided Mortality and Value, {year}',
            )
 
            # ---- Region summary ----
            if 'region_wb' in df.columns:
                region_df = (
                    df[df['region_wb'] != 'Antarctica']
                    .groupby('region_wb', as_index=False)
                    .agg(
                        avoided_mortality_sum=('avoided_deaths_sum', 'sum'),
                        value_sum=('avoided_value_sum_usd', 'sum'),
                    )
                    .sort_values('value_sum', ascending=False)
                )
                region_df['value_sum'] = region_df['value_sum'] / 1_000_000
                region_out = region_df.rename(columns={
                    'region_wb': 'Geography',
                    'avoided_mortality_sum': 'Avoided mortality',
                    'value_sum': 'Value (US$ millions)',
                })
                region_out['Avoided mortality'] = region_out['Avoided mortality'].map(lambda v: f'{v:,.2f}')
                region_out['Value (US$ millions)'] = region_out['Value (US$ millions)'].map(lambda v: f'${v:,.2f}')
                _save_table_safe(
                    region_out, os.path.join(p.tables_figures_dir, f'region_wb_summary_full_impacts_{year}'),
                    f'Aggregate Benefits by World Bank Region, {year}',
                )
 
            # ---- Top 15 countries ----
            top = df.sort_values('avoided_deaths_sum', ascending=False).head(15).copy()
            top['avoided_value_sum_usd'] = top['avoided_value_sum_usd'] / 1_000_000
            keep_cols = [c for c in ['name_long', 'region_wb', 'avoided_deaths_sum', 'avoided_value_sum_usd']
                         if c in top.columns]
            top_out = top[keep_cols].rename(columns={
                'name_long': 'Country',
                'region_wb': 'World Bank region',
                'avoided_deaths_sum': 'Avoided mortality',
                'avoided_value_sum_usd': 'Value (US$ millions)',
            })
            top_out['Avoided mortality'] = top_out['Avoided mortality'].map(lambda v: f'{v:,.2f}')
            top_out['Value (US$ millions)'] = top_out['Value (US$ millions)'].map(lambda v: f'${v:,.2f}')
            _save_table_safe(
                top_out, os.path.join(p.tables_figures_dir, f'top_countries_mortality_full_impacts_{year}'),
                f'Top 15 Countries by Avoided Landslide Mortality, {year}',
            )
 
            p.L.info(f'{year}: results tables (csv/tex/md) exported to {p.tables_figures_dir}')
    return p


def export_si_severity_sensitivity_table(p):
    """Side-by-side comparison table for the Appendix: main (slope-based)
    vs. sensitivity-check (SI-based) severity specifications.
    """
    publish_inputs(p)
    if p.run_this:
        severity_model_coefficients_path = os.path.join(p.modeling_dir, 'severity_model_coefficients.json')
        with open(severity_model_coefficients_path) as f:
            main_spec = json.load(f)  # slope-based, current production
 
        si_path = os.path.join(p.modeling_dir, 'severity_model_si_sensitivity.json')
        if not os.path.exists(si_path):
            p.L.warning(f'{si_path} not found -- run estimate_severity_model_si_sensitivity first.')
            return p
        with open(si_path) as f:
            si_spec = json.load(f)
 
        rows = []
        rows.append({'Model': 'Part A Pseudo R-sq.',
                      'Slope-based (main)': f"{main_spec.get('part_a_pseudo_r_squared'):.4f}",
                      'SI-based (sensitivity)': f"{si_spec.get('part_a_pseudo_r_squared'):.4f}"})
        rows.append({'Model': 'Part B R-sq.',
                      'Slope-based (main)': f"{main_spec.get('part_b_r_squared'):.4f}",
                      'SI-based (sensitivity)': f"{si_spec.get('part_b_r_squared'):.4f}"})
        rows.append({'Model': 'Terrain covariate coefficient (Part A)',
                      'Slope-based (main)': f"{main_spec['part_a_params'].get('slope_degrees', float('nan')):.4f}",
                      'SI-based (sensitivity)': f"{si_spec['part_a_params'].get('si_observed', float('nan')):.4f}"})
        rows.append({'Model': 'Terrain covariate p-value (Part A)',
                      'Slope-based (main)': f"{main_spec.get('part_a_p_value', {}).get('slope_degrees', float('nan')):.4g}",
                      'SI-based (sensitivity)': f"{si_spec.get('part_a_p_value', {}).get('si_observed', float('nan')):.4g}"})
        rows.append({'Model': 'Terrain covariate coefficient (Part B)',
                      'Slope-based (main)': f"{main_spec['part_b_params'].get('slope_degrees', float('nan')):.4f}",
                      'SI-based (sensitivity)': f"{si_spec['part_b_params'].get('si_observed', float('nan')):.4f}"})
        rows.append({'Model': 'Terrain covariate p-value (Part B)',
                      'Slope-based (main)': f"{main_spec.get('part_b_p_value', {}).get('slope_degrees', float('nan')):.4g}",
                      'SI-based (sensitivity)': f"{si_spec.get('part_b_p_value', {}).get('si_observed', float('nan')):.4g}"})
        rows.append({'Model': 'N (fatal events)',
                      'Slope-based (main)': f"{main_spec.get('n_train_fatal'):,}",
                      'SI-based (sensitivity)': f"{si_spec.get('n_train_fatal'):,}"})
 
        table_df = pd.DataFrame(rows)
        notes = (
            'Terrain covariate row reports $slope_degrees$ for the main specification and '
            '$si_observed$ for the sensitivity check, both occupy the same role (a terrain-based '
            'severity predictor) in their respective specifications, but are not the same variable.'
        )
        _save_table_safe(
            table_df, os.path.join(p.tables_figures_dir, 'si_severity_sensitivity_table'),
            'Severity Model: Slope-Based vs. Stability-Index-Based', notes,
        )
        p.L.info('SI-severity sensitivity table exported.')
    return p
 
 


def export_pi_audit_table(p):
    """Per-year event/land pixel counts underlying the population
    prevalence estimate, for Appendix transparency.
    """
    publish_inputs(p)
    if p.run_this:
        out_path = os.path.join(p.tables_figures_dir, 'pi_audit_table')
        if not os.path.exists(f'{out_path}.csv') and not p.force_run:
            rows = []
            total_event_pixels = 0
            total_land_pixels = 0
    
            for year in p.modeling_range:
                binary_path = os.path.join(
                    p.preprocessing_dir, 'uglc_annual_panels', f'uglc_binary_{year}.tif'
                )
                si_path = p.si_paths.get('observed', {}).get(year)
                if not os.path.exists(binary_path) or not si_path or not os.path.exists(si_path):
                    continue
                
                ds = gdal.Open(binary_path)
                band = ds.GetRasterBand(1)
                arr = band.ReadAsArray()
                nodata = band.GetNoDataValue()
                ds = None
    
                si_ds = gdal.Open(si_path)
                si_band = si_ds.GetRasterBand(1)
                si_arr = si_band.ReadAsArray()
                si_nodata = si_band.GetNoDataValue()
                si_ds = None
    
                valid = (arr != nodata) if nodata is not None else np.ones_like(arr, dtype=bool)
                si_valid = (si_arr != si_nodata) if si_nodata is not None else np.ones_like(si_arr, dtype=bool)
                valid &= si_valid
    
                event_pixels = int((arr[valid] == 1).sum())
                land_pixels = int(valid.sum())
                prevalence = event_pixels / land_pixels if land_pixels > 0 else float('nan')
    
                rows.append({
                    'Year': year,
                    'Event pixels': f'{event_pixels:,}',
                    'Land pixels (SI-eligible)': f'{land_pixels:,}',
                    'Prevalence': f'{prevalence:.8f}',
                })
                total_event_pixels += event_pixels
                total_land_pixels += land_pixels
    
            overall_pi = total_event_pixels / total_land_pixels if total_land_pixels > 0 else float('nan')
            rows.append({
                'Year': 'Total',
                'Event pixels': f'{total_event_pixels:,}',
                'Land pixels (SI-eligible)': f'{total_land_pixels:,}',
                'Prevalence': f'{overall_pi:.8f}',
            })
    
            table_df = pd.DataFrame(rows)
            _save_table_safe(
                table_df, out_path,
                'Population Prevalence Audit',
            )
            p.L.info('Pi audit table exported.')
    return p


def plot_global_rasters_png(p):
    publish_inputs(p)
    failures = []
 
    def _plot_raster(raster_path, out_png, title, cmap, q_low, q_high, cbar_format):
        if os.path.exists(out_png):
            p.L.info(f'✓ Plot already exists: {out_png}')
            return True
        if not os.path.exists(raster_path):
            msg = f'Raster not found: {raster_path}'
            p.L.info(f'WARNING: {msg}')
            failures.append(msg)
            return False
 
        try:
            with rasterio.open(raster_path) as src:
                max_dim = int(getattr(p, 'plot_raster_max_dim', 4096))
                scale = max(src.height / max_dim, src.width / max_dim, 1.0)
                out_h = max(1, int(np.ceil(src.height / scale)))
                out_w = max(1, int(np.ceil(src.width / scale)))
                arr = src.read(
                    1,
                    out_shape=(out_h, out_w),
                    resampling=Resampling.nearest,
                ).astype(np.float32)
                ndv = src.nodata
 
            if ndv is not None:
                arr = np.where(arr == ndv, np.nan, arr)
 
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                msg = f'Raster has no finite data: {raster_path}'
                p.L.info(f'WARNING: {msg}')
                failures.append(msg)
                return False
 
            vmin, vmax = np.nanpercentile(finite, [q_low, q_high])
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                msg = f'Invalid plotting range: {raster_path}'
                p.L.info(f'WARNING: {msg}')
                failures.append(msg)
                return False
 
            if vmin == vmax:
                eps = abs(vmin) * 1e-6 if vmin != 0 else 1e-6
                vmin -= eps
                vmax += eps
 
            fig, ax = plt.subplots(figsize=(9, 4.8), dpi=220)
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_axis_off()
            cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, shrink=0.4)
            cbar.ax.tick_params(labelsize=8)
            cbar.ax.yaxis.set_major_formatter(mtick.StrMethodFormatter(cbar_format))
            plt.tight_layout()
            plt.savefig(out_png, dpi=300, bbox_inches='tight')
            plt.close(fig)
            p.L.info(f'✓ Saved figure: {out_png}')
            return True
        except Exception as e:
            msg = f'Failed plotting {os.path.basename(raster_path)}: {e}'
            p.L.info(f'WARNING: {msg}')
            failures.append(msg)
            plt.close('all')
            return False
 
    for year in p.prediction_years:
        # ---- Stability Index (SI), observed scenario ----
        si_path = p.si_paths.get('observed', {}).get(year)
        if si_path:
            p.L.info(f'\nPlotting Stability Index for {year}...')
            _plot_raster(
                si_path,
                os.path.join(p.tables_figures_dir, f'si_observed_{year}.png'),
                f'Stability Index (Observed Forest Cover), {year}',
                'RdYlGn',  # red=less stable, green=more stable -- intuitive
                2,
                98,
                '{x:.2f}',
            )
 
        # ---- Hazard probability, per scenario ----
        for scenario_name in p.si_paths.keys():
            p.L.info(f'\nPlotting hazard probability for {year} / {scenario_name}...')
            _plot_raster(
                os.path.join(p.stitch_tiles_dir, f'hazard_prob_{scenario_name}_{year}.tif'),
                os.path.join(p.tables_figures_dir, f'hazard_prob_{scenario_name}_{year}.png'),
                f'Landslide Hazard Probability ({scenario_name}), {year}',
                'Reds',
                2,
                98,
                '{x:.5f}',  # hazard probabilities are small (pixel-year scale)
                            # 3 decimals would show mostly zeros
            )
 
        # ---- Expected deaths, per scenario ----
        for scenario_name in p.si_paths.keys():
            p.L.info(f'\nPlotting expected deaths for {year} / {scenario_name}...')
            _plot_raster(
                os.path.join(p.stitch_tiles_dir, f'expected_deaths_{scenario_name}_{year}.tif'),
                os.path.join(p.tables_figures_dir, f'expected_deaths_{scenario_name}_{year}.png'),
                f'Expected Deaths ({scenario_name}), {year}',
                'Reds',
                2,
                98,
                '{x:.5f}',
            )
 
        # ---- Avoided mortality ----
        p.L.info(f'\nPlotting avoided mortality for {year}...')
        _plot_raster(
            os.path.join(p.valuation_dir, f'avoided_mortality_{year}.tif'),
            os.path.join(p.tables_figures_dir, f'avoided_mortality_{year}.png'),
            f'Avoided Landslide Mortality, {year}',
            'Purples',
            2,
            98,
            '{x:.5f}',
        )
        _plot_raster(
            os.path.join(p.valuation_dir, f'avoided_mortality_value_{year}.tif'),
            os.path.join(p.tables_figures_dir, f'avoided_mortality_value_{year}.png'),
            f'Economic Value of Avoided Mortality, {year}',
            'Greens',
            2,
            98,
            '${x:,.0f}',
        )
 
    if failures:
        p.L.info(f'plot_global_rasters_png completed with {len(failures)} warnings.')
 
    return p


def plot_country_choropleth_maps(p):
    publish_inputs(p)
    if p.run_this: 
        for year in p.prediction_years:
            gpkg_path = os.path.join(p.tables_figures_dir, f'zonal_statistics_{year}.gpkg')
            if not os.path.exists(gpkg_path):
                p.L.warning(f'{gpkg_path} not found, skipping.')
                continue
 
            gdf = gpd.read_file(gpkg_path)
 
            specs = [
                ('avoided_deaths_sum', None, 'Purples',
                 f'Avoided Landslide Mortality, {year}', 'Avoided deaths',
                 '{:.2f}', f'avoided_mortality_choropleth_{year}.png',
                 [0, 1, 5, 15, 50, 100, float('inf')]),
                ('avoided_value_sum_usd', 1e6, 'Greens',
                 f'Economic Value of Avoided Mortality, {year}', 'Value (US$ millions)',
                 '${:,.2f}M', f'avoided_mortality_value_choropleth_{year}.png',
                 [0, 1, 5, 15, 50, 100, float('inf')]),
            ]
 
            for column, divisor, cmap, title, legend_label, tick_fmt, out_name, bucket_edges in specs:
                out_path = os.path.join(p.tables_figures_dir, out_name)
                if os.path.exists(out_path) and not p.force_run:
                    p.L.info(f'Choropleth already exists: {out_path}')
                    continue
 
                # Prepare and normalize data
                gdf_plot = gdf[gdf[column].notna()].copy()
                gdf_plot[column] = pd.to_numeric(gdf_plot[column], errors='coerce')
                values = gdf_plot[column] / divisor if divisor else gdf_plot[column]
                
                # Assign to buckets
                gdf_plot['bucket'] = pd.cut(values, bins=bucket_edges, labels=False, 
                                             include_lowest=True)
                
                fig, ax = plt.subplots(figsize=(12, 6), dpi=220)
 
                # Plot all countries with white base
                gdf.plot(ax=ax, color='white', edgecolor='#cccccc', linewidth=0.3)
                
                # Color by bucket
                cmap_obj = plt.get_cmap(cmap)
                num_buckets = len(bucket_edges) - 1
                colors = [cmap_obj(i / (num_buckets - 1)) for i in range(num_buckets)]
                
                for bucket_idx in range(num_buckets):
                    bucket_data = gdf_plot[gdf_plot['bucket'] == bucket_idx]
                    if not bucket_data.empty:
                        bucket_data.plot(ax=ax, color=colors[bucket_idx], 
                                        edgecolor='#666666', linewidth=0.3)
                
                # Create legend
                legend_elements = []
                for i in range(num_buckets):
                    lower = bucket_edges[i]
                    upper = bucket_edges[i + 1]
                    if upper == float('inf'):
                        label = f'{tick_fmt.format(lower)}+'
                    else:
                        label = f'{tick_fmt.format(lower)} – {tick_fmt.format(upper)}'
                    legend_elements.append(mpatches.Patch(color=colors[i], label=label))
                
                ax.legend(handles=legend_elements, loc='lower left', frameon=False, 
                         fontsize=8, title=legend_label, title_fontsize=9)
                
                ax.set_axis_off()
                plt.tight_layout()
                plt.savefig(out_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                p.L.info(f'Saved choropleth: {out_path}')
    return p


def plot_uglc_from_vector(p):
    """
    Plot UGLC event geometry as a point map with fatality bins.
    """
    publish_inputs(p)
    if not p.run_this:
        return p
 
    out_png = os.path.join(p.tables_figures_dir, 'uglc_events_fatality_bins.png')
    if os.path.exists(out_png) and not p.force_run:
        p.L.info(f'✓ UGLC events plot already exists: {out_png}')
        return p
 
    p.L.info(f'Plotting UGLC events from: {p.uglc_path}')
    gdf = gpd.read_file(p.uglc_path)
    if gdf.empty:
        raise ValueError('WARNING: UGLC vector is empty, skipping.')
 
    # Points already, not buffered polygons
    points = gdf.to_crs('EPSG:3857').copy()
 
    fatalities = points['fatality_count'].fillna(0).clip(lower=0)
    nonfatal = fatalities <= 0
    bins = {
        '1-5': (fatalities >= 1) & (fatalities < 5),
        '5-25': (fatalities >= 5) & (fatalities < 25),
        '25-100': (fatalities >= 25) & (fatalities < 100),
        '100+': fatalities >= 100,
    }
 
    size_map = {'1-5': 12, '5-25': 24, '25-100': 44, '100+': 80}
    color_map = {
        '1-5': '#f28e2b',
        '5-25': '#e15759',
        '25-100': '#b51d39',
        '100+': '#5b0f1f',
    }
 
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=220)
    if nonfatal.any():
        points.loc[nonfatal].plot(
            ax=ax, color='#6c757d', markersize=6, alpha=0.32, linewidth=0, zorder=2,
        )
 
    for label, mask in bins.items():
        if mask.any():
            points.loc[mask].plot(
                ax=ax, color=color_map[label], markersize=size_map[label],
                alpha=0.82, linewidth=0, zorder=3,
            )
 
    legend_handles = [
        Line2D([0], [0], marker='o', color='none', label='Nonfatal landslide',
               markerfacecolor='#6c757d', markeredgecolor='none', markersize=6, alpha=0.4),
        Line2D([0], [0], marker='o', color='none', label='1-5 deaths',
               markerfacecolor=color_map['1-5'], markeredgecolor='none', markersize=5, alpha=0.9),
        Line2D([0], [0], marker='o', color='none', label='5-25 deaths',
               markerfacecolor=color_map['5-25'], markeredgecolor='none', markersize=6, alpha=0.9),
        Line2D([0], [0], marker='o', color='none', label='25-100 deaths',
               markerfacecolor=color_map['25-100'], markeredgecolor='none', markersize=7, alpha=0.9),
        Line2D([0], [0], marker='o', color='none', label='100+ deaths',
               markerfacecolor=color_map['100+'], markeredgecolor='none', markersize=9, alpha=0.9),
    ]
 
    ax.legend(
        handles=legend_handles, title='Fatality count', loc='lower left',
        frameon=True, framealpha=0.9, facecolor='white', edgecolor='none',
        title_fontproperties={'family': 'serif', 'size': 9},
        prop={'family': 'serif', 'size': 8},
    )
 
    if ctx is not None:
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels,
                             crs=points.crs, attribution=False)
        except Exception as e:
            p.L.info(f'WARNING: basemap fetch failed ({e}), continuing without it.')
 
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    p.L.info(f'✓ Saved figure: {out_png}')
    return p


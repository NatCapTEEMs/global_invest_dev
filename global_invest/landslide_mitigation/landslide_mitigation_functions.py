"""
landslide_mitigation_functions.py

Science and raster helpers for the landslide chain. The former landslide_mitigation_utils.py
(EASE-grid parsing, reference warping, point sampling, array writing) is folded in here per the
five-file service template.
"""
import os

import numpy as np
import rasterio
from osgeo import gdal, osr
import pygeoprocessing as pygeo

# ==================================================================== #
# Infinite-slope stability index (SI) computation
# ==================================================================== #

def compute_si_global(
    friction_angle_path,
    cohesion_soil_path,
    forest_share_path,
    c_root_max,
    unit_weight_path,
    transmissivity_path,
    static_q_path,
    slope_path,
    soil_depth_path,
    output_si_path,
    nodata=-9999.0,
    min_slope_deg=2.0,
):
    """Computes the infinite-slope stability index globally, block-wise.

    SI = tan(phi)/tan(beta) + c_total/(gamma*h*sin(beta)*cos(beta)) - q/(T*sin(beta))
      phi = friction angle (deg); beta = slope (deg)
      c_total = cohesion_soil + c_root, c_root = forest_share * c_root_max
      gamma = unit weight (kN/m^3); h = soil depth (m)
      q = specific discharge (m^2/day); T = transmissivity (m^2/day)

    c_root_max: plain scalar closed over by si_op (not passed through
    raster_calculator). 0.0 for the 'full_impacts' bound, C_ROOT_MAX_KPA
    otherwise (see preprocessing_tasks.compute_si_scenarios).

    min_slope_deg: excludes near-flat terrain entirely (infinite-slope
    theory doesn't apply there; standard in SHALSTAB/SINMAP/TRIGRS), not
    just a div/0 guard. Without it, both the friction and hydrological
    terms blow up near beta=0 and swamp the real forest-cover signal
    (median observed-vs-full_impacts diff was exactly 0 before this fix).

    Clipping: slope clipped to [0.5, 89.5] deg before trig (div/0 at flat,
    blowup near vertical); final SI clipped to [-10, 10].
    """
    paths = [
        friction_angle_path, cohesion_soil_path, forest_share_path,
        unit_weight_path, transmissivity_path, static_q_path,
        slope_path, soil_depth_path,
    ]
    nodatas = [pygeo.get_raster_info(path)['nodata'][0] for path in paths]

    def si_op(phi_deg, c_soil, forest_share, gamma, T, q, slope_deg, h):
        arrays = [phi_deg, c_soil, forest_share, gamma, T, q, slope_deg, h]
        valid = np.ones(phi_deg.shape, dtype=bool)
        for arr, nd in zip(arrays, nodatas):
            if nd is not None:
                valid &= (arr != nd)

        # Physical exclusion, not a div/0 guard
        valid &= (slope_deg >= min_slope_deg)

        slope_deg_c = np.clip(slope_deg, 0.5, 89.5)
        slope_rad = np.radians(slope_deg_c)
        phi_rad = np.radians(np.clip(phi_deg, 15, 40))  # re-clip defensively

        c_root = np.clip(forest_share, 0, 1) * c_root_max
        c_total = c_soil + c_root

        # Term 1: friction / geometry.
        term1 = np.tan(phi_rad) / np.tan(slope_rad)

        # Term 2: cohesion (stabilizing). errstate suppresses a spurious
        # div/0 warning at denom2==0 (bare rock / zero soil depth)
        denom2 = gamma * h * np.sin(slope_rad) * np.cos(slope_rad)
        with np.errstate(divide='ignore', invalid='ignore'):
            term2 = np.where(denom2 > 0, c_total / denom2, 0)

        # Term 3: hydrological (destabilizing), physically a saturation
        # ratio -> capped at 1 (standard in this model family). Uncapped,
        # ~90% of pixels clipped to the -10 SI floor even after the
        # slope filter, since upslope_area is heavily right-skewed.
        denom3 = T * np.sin(slope_rad)
        with np.errstate(divide='ignore', invalid='ignore'):
            saturation_ratio = np.where(denom3 > 0, q / denom3, 0)
        term3 = np.clip(saturation_ratio, 0, 1)

        si = np.clip(term1 + term2 - term3, -10, 10)
        return np.where(valid, si, nodata).astype(np.float32)

    pygeo.raster_calculator(
        [(path, 1) for path in paths],
        si_op, output_si_path, gdal.GDT_Float32, nodata,
        calc_raster_stats=True,
    )
    return output_si_path


# ==================================================================== #
# Thickness-weighted 0-30cm combine (SoilGrids + HiHydroSoil share this)
# ==================================================================== #

DEPTH_WEIGHTS_0_30CM = {'0-5cm': 5, '5-15cm': 10, '15-30cm': 15}  # total 30cm

def thickness_weighted_combine(depth_raster_paths, out_path, nodata=-9999.0,
                                conv_factor=None):
    """Combine 3 depth-interval rasters (0-5, 5-15, 15-30cm) into one
    thickness-weighted 0-30cm 'topsoil' raster. Inputs must share the same
    native grid (true for SoilGrids/HiHydroSoil).
    """
    keys = list(DEPTH_WEIGHTS_0_30CM.keys())
    paths = [depth_raster_paths[k] for k in keys]
    weights = [DEPTH_WEIGHTS_0_30CM[k] for k in keys]

    # Size/alignment check up front.
    infos = [pygeo.get_raster_info(p) for p in paths]
    first_size = infos[0]['raster_size']
    for path, info in zip(paths, infos):
        if info['raster_size'] != first_size:
            raise ValueError(
                f'{path} size {info["raster_size"]} does not match first '
                f'input {first_size} -- inputs must share the same native '
                f'grid before combining.'
            )
    src_nodatas = [info['nodata'][0] for info in infos]

    def combine_op(*arrays):
        weighted_sum = np.zeros(arrays[0].shape, dtype=np.float64)
        weight_present = np.zeros(arrays[0].shape, dtype=np.float64)
        for arr, w, nd in zip(arrays, weights, src_nodatas):
            arr64 = arr.astype(np.float64)
            valid = np.ones(arr.shape, dtype=bool) if nd is None else (arr != nd)
            if conv_factor is not None:
                arr64 = arr64 / conv_factor
            weighted_sum[valid] += arr64[valid] * w
            weight_present[valid] += w

        with np.errstate(invalid='ignore', divide='ignore'):
            combined = np.where(weight_present > 0, weighted_sum / weight_present, np.nan)
        return np.where(np.isnan(combined), nodata, combined).astype(np.float32)

    pygeo.raster_calculator(
        [(path, 1) for path in paths], combine_op, out_path,
        gdal.GDT_Float32, nodata,
    )
    return out_path

# ==================================================================== #
# EASE-Grid 2.0 reference grid: parse from the authoritative .gpd file
# ==================================================================== #

def parse_gpd_grid_definition(gpd_path):
    """Parse an NSIDC .gpd grid parameter definition file into a dict.

    .gpd format is `Key: value ; comment` lines. Only pulls what this
    pipeline needs.
    """
    params = {}
    with open(gpd_path, 'r') as f:
        for line in f:
            line = line.split(';')[0].strip()
            if not line or ':' not in line:
                continue
            key, value = line.split(':', 1)
            params[key.strip()] = value.strip()

    origin_x = float(params['Map Origin X'])
    origin_y = float(params['Map Origin Y'])
    pixel_size = float(params['Grid Map Units per Cell'])
    n_cols = int(params['Grid Width'])
    n_rows = int(params['Grid Height'])

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(6933)  # WGS 84 / NSIDC EASE-Grid 2.0 Global
    # NOTE: assumes EPSG:6933 regardless of the .gpd's own projection line --
    # true for all EASE2_M-family (global) grids; N/S polar grids differ.

    return {
        'origin_x': origin_x, 'origin_y': origin_y,
        'pixel_size': pixel_size, 'n_cols': n_cols, 'n_rows': n_rows,
        'srs_wkt': srs.ExportToWkt(),
    }


# ==================================================================== #
# Warp a raster onto a reference grid
# ==================================================================== #

def warp_to_reference(
    src_path, dst_path, reference_raster_path,
    resample_method='bilinear',
    src_nodata=None, dst_nodata=None, output_type=None,
    n_threads=4,
    creation_options=('TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                       'BLOCKXSIZE=256', 'BLOCKYSIZE=256'),
    show_progress=True,
):
    """Warp src_path onto the exact grid of reference_raster_path.

    resample_method by variable type:
      - 'average': continuous fields being downsampled (DEM, forest weight,
        soil texture/bulk density/K_sat/soil depth)
      - 'bilinear': continuous fields at similar resolution
      - 'near' or 'mode': categorical fields (GAEZ zones)
      - 'sum': count fields needing conservation (LandScan population)
    """
    ref_info = pygeo.get_raster_info(reference_raster_path)
    ref_gt = ref_info['geotransform']
    ref_x_size, ref_y_size = ref_info['raster_size']

    x_min = ref_gt[0]
    y_max = ref_gt[3]
    x_max = x_min + ref_gt[1] * ref_x_size
    y_min = y_max + ref_gt[5] * ref_y_size

    resample_alg_map = {
        'bilinear': gdal.GRA_Bilinear,
        'average': gdal.GRA_Average,
        'near': gdal.GRA_NearestNeighbour,
        'mode': gdal.GRA_Mode,
        'cubic': gdal.GRA_Cubic,
        'sum': gdal.GRA_Sum,  # GDAL >= 3.1
    }
    if resample_method not in resample_alg_map:
        raise ValueError(f"Unknown resample_method '{resample_method}', "
                          f"choose from {list(resample_alg_map)}")

    warp_options = gdal.WarpOptions(
        format='GTiff',
        outputBounds=(x_min, y_min, x_max, y_max),
        xRes=ref_gt[1],
        yRes=abs(ref_gt[5]),
        dstSRS=ref_info['projection_wkt'],
        resampleAlg=resample_alg_map[resample_method],
        srcNodata=src_nodata,
        dstNodata=dst_nodata if dst_nodata is not None else src_nodata,
        outputType=output_type if output_type is not None else gdal.GDT_Unknown,
        multithread=True,
        warpOptions=[f'NUM_THREADS={n_threads}', 'UNIFIED_SRC_NODATA=YES'],
        # UNIFIED_SRC_NODATA=YES avoids including nodata pixels as if they were valid data
        # confirmed on the DEM's coastlines.
        creationOptions=list(creation_options),
        callback=gdal.TermProgress_nocb if show_progress else None,
    )

    if show_progress:
        print(f'Warping: {os.path.basename(str(src_path))} -> {os.path.basename(dst_path)}')

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    # str() both: this GDAL build's binding rejects pathlib.Path for src (wrong GDALDatasetShadow).
    result_ds = gdal.Warp(str(dst_path), str(src_path), options=warp_options)
    if result_ds is None:
        raise RuntimeError(f'gdal.Warp failed: {src_path} -> {dst_path}')
    result_ds = None
    return dst_path


# ==================================================================== #
# Sample a raster at point coordinates
# ==================================================================== #
 
def sample_raster_at_points(raster_path, x_coords, y_coords, band=1):
    """Sample a raster at a list of (x, y) coordinates (in the raster's
    own CRS -- EASE-Grid meters throughout this pipeline). Returns a
    numpy array of values, with the raster's nodata replaced by np.nan.
    """
    with rasterio.open(raster_path) as src:
        nodata = src.nodatavals[band - 1]
        coords = list(zip(x_coords, y_coords))
        values = np.array(
            [v[band - 1] for v in src.sample(coords)], dtype=np.float64
        )
    if nodata is not None:
        values = np.where(values == nodata, np.nan, values)
    return values


# ==================================================================== #
# Write a numpy array to GeoTIFF
# ==================================================================== #

def write_raster_from_array(arr, gt, proj_wkt, out_path, nodata, dtype):
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(
        out_path, arr.shape[1], arr.shape[0], 1, dtype,
        options=['TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                 'BLOCKXSIZE=256', 'BLOCKYSIZE=256'],
    )
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj_wkt)
    band = ds.GetRasterBand(1)
    band.WriteArray(arr)
    band.SetNoDataValue(nodata)
    ds = None

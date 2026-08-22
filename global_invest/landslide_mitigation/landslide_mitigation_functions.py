"""Raster and grid drivers for the landslide chain: the file-touching layer.

Everything here reads or writes a raster. The arithmetic each driver applies lives in
landslide_mitigation_chain.py as a pure function over arrays, so the science is testable
without staged inputs and these wrappers stay thin.

The former landslide_mitigation_utils.py (EASE-grid parsing, reference warping, point
sampling, array writing) is folded in here per the service template.
"""
import os

import numpy as np
import rasterio
from osgeo import gdal, osr
import pygeoprocessing as pygeo

from global_invest.landslide_mitigation import landslide_mitigation_chain as chain

DEPTH_WEIGHTS_0_30CM = chain.DEPTH_WEIGHTS_0_30CM


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
    nodata=chain.NODATA,
    min_slope_deg=chain.MIN_SLOPE_DEG,
):
    """Writes the stability index globally, block-wise, from the nine input rasters.

    The formula and its clipping are chain.stability_index; this function supplies the
    nodata mask and the near-flat exclusion, which belong with the rasters rather than with
    the arithmetic.

    c_root_max is a plain scalar closed over by si_op (not passed through raster_calculator):
    0.0 for the 'full_impacts' bound, chain.C_ROOT_MAX_KPA otherwise (see
    compute_si_scenarios in the tasks module).

    min_slope_deg excludes near-flat terrain entirely -- infinite-slope theory does not apply
    there, and the exclusion is standard in SHALSTAB/SINMAP/TRIGRS. It is not just a div/0
    guard: without it both the friction and hydrological terms blow up near beta = 0 and swamp
    the real forest-cover signal (the median observed-vs-full_impacts difference was exactly
    zero before this exclusion was added).
    """
    paths = [
        friction_angle_path, cohesion_soil_path, forest_share_path,
        unit_weight_path, transmissivity_path, static_q_path,
        slope_path, soil_depth_path,
    ]
    nodatas = [pygeo.get_raster_info(path)['nodata'][0] for path in paths]

    def si_op(phi_deg, c_soil, forest_share, gamma, transmissivity, q, slope_deg, soil_depth):
        arrays = [phi_deg, c_soil, forest_share, gamma, transmissivity, q, slope_deg, soil_depth]
        valid = np.ones(phi_deg.shape, dtype=bool)
        for array, source_nodata in zip(arrays, nodatas):
            if source_nodata is not None:
                valid &= (array != source_nodata)
        valid &= (slope_deg >= min_slope_deg)

        si = chain.stability_index(phi_deg, c_soil, forest_share, gamma, transmissivity, q,
                                   slope_deg, soil_depth, c_root_max)
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

def thickness_weighted_combine(depth_raster_paths, out_path, nodata=chain.NODATA,
                               conv_factor=None):
    """Combine the 0-5, 5-15 and 15-30cm rasters into one 0-30cm topsoil raster.

    Inputs must share the same native grid (true for SoilGrids and HiHydroSoil), so the size
    check is what stops a silently misaligned combine.
    """
    keys = list(chain.DEPTH_WEIGHTS_0_30CM.keys())
    paths = [depth_raster_paths[key] for key in keys]
    weights = [chain.DEPTH_WEIGHTS_0_30CM[key] for key in keys]

    infos = [pygeo.get_raster_info(path) for path in paths]
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
        combined = chain.thickness_weighted_mean(arrays, weights, src_nodatas, conv_factor)
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

    .gpd format is `Key: value ; comment` lines. Only pulls what this pipeline needs.
    """
    params = {}
    with open(gpd_path, 'r') as f:
        for line in f:
            line = line.split(';')[0].strip()
            if not line or ':' not in line:
                continue
            key, value = line.split(':', 1)
            params[key.strip()] = value.strip()

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(6933)  # WGS 84 / NSIDC EASE-Grid 2.0 Global
    # NOTE: assumes EPSG:6933 regardless of the .gpd's own projection line --
    # true for all EASE2_M-family (global) grids; N/S polar grids differ.

    return {
        'origin_x': float(params['Map Origin X']),
        'origin_y': float(params['Map Origin Y']),
        'pixel_size': float(params['Grid Map Units per Cell']),
        'n_cols': int(params['Grid Width']),
        'n_rows': int(params['Grid Height']),
        'srs_wkt': srs.ExportToWkt(),
    }


def create_reference_raster(out_path, geotransform, n_cols, n_rows, srs_wkt):
    """An empty single-band Byte raster on the given grid, used as a warp target."""
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(out_path, n_cols, n_rows, 1, gdal.GDT_Byte,
                       options=list(chain.GTIFF_CREATION_OPTIONS))
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(srs_wkt)
    ds.GetRasterBand(1).SetNoDataValue(0)
    ds = None
    return out_path


# ==================================================================== #
# Warp a raster onto a reference grid
# ==================================================================== #

def warp_to_reference(
    src_path, dst_path, reference_raster_path,
    resample_method='bilinear',
    src_nodata=None, dst_nodata=None, output_type=None,
    n_threads=4,
    creation_options=chain.GTIFF_CREATION_OPTIONS,
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

def write_raster_from_array(arr, gt, proj_wkt, out_path, nodata, dtype,
                            creation_options=chain.GTIFF_CREATION_OPTIONS):
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(out_path, arr.shape[1], arr.shape[0], 1, dtype,
                       options=list(creation_options))
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj_wkt)
    band = ds.GetRasterBand(1)
    band.WriteArray(arr)
    band.SetNoDataValue(nodata)
    ds = None
    return out_path

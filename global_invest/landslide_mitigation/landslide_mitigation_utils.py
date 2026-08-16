"""
landslide_mitigation_utils.py
"""
import os
import numpy as np
from osgeo import gdal, osr
import pygeoprocessing as pygeo
import rasterio


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
    result_ds = gdal.Warp(dst_path, src_path, options=warp_options)
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
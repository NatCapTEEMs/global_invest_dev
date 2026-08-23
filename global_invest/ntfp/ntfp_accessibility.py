# -*- coding: utf-8 -*-
"""The accessibility stage: forest hectares within reach of a road or a river, per country.

The source scripts call forest accessible when it lies within 10 km of a road or a river. Doing
that on the native lat-lon grids would mean a buffer whose width in kilometres changes with
latitude, so everything here happens on an equal-area grid at 1 km, where a 10 km buffer is
exactly ten cells in every direction and a cell is exactly 100 hectares. The warps are left to
GDAL, which streams them; only the buffer itself needs the whole mask in memory, and as a
boolean array that is under a gigabyte.

The arithmetic this file drives lives in ntfp_functions as array functions with their own tests.
This module is the part that touches files.
"""
import os

import numpy as np
from osgeo import gdal, ogr

from global_invest.ntfp import ntfp_functions as nf

# NSIDC EASE-Grid 2.0 global, the equal-area grid the rest of the library uses for area work.
EQUAL_AREA_EPSG = 6933
ACCESSIBILITY_CELL_SIZE_M = 1000.0
# One 1 km cell on an equal-area grid is exactly 100 hectares, so no per-cell area raster is needed.
HECTARES_PER_CELL = (ACCESSIBILITY_CELL_SIZE_M / 100.0) ** 2
GTIFF_CREATION_OPTIONS = ('TILED=YES', 'COMPRESS=DEFLATE', 'BIGTIFF=YES')


def rasterize_rivers(rivers_vector_path, reference_raster_path, out_path):
    """Burn the river centrelines onto the reference grid as a 0/1 mask."""
    reference = gdal.Open(reference_raster_path)
    driver = gdal.GetDriverByName('GTiff')
    target = driver.Create(out_path, reference.RasterXSize, reference.RasterYSize, 1,
                           gdal.GDT_Byte, options=list(GTIFF_CREATION_OPTIONS))
    target.SetGeoTransform(reference.GetGeoTransform())
    target.SetProjection(reference.GetProjection())
    target.GetRasterBand(1).SetNoDataValue(0)
    source = ogr.Open(rivers_vector_path)
    gdal.RasterizeLayer(target, [1], source.GetLayer(0), burn_values=[1])
    target = None
    return out_path


def warp_to_equal_area(src_path, out_path, resample_algorithm, output_type=gdal.GDT_Float32,
                       src_nodata=None, dst_nodata=None):
    """One raster on the equal-area 1 km grid."""
    gdal.Warp(out_path, src_path,
              dstSRS=f'EPSG:{EQUAL_AREA_EPSG}',
              xRes=ACCESSIBILITY_CELL_SIZE_M, yRes=ACCESSIBILITY_CELL_SIZE_M,
              resampleAlg=resample_algorithm, outputType=output_type,
              srcNodata=src_nodata, dstNodata=dst_nodata,
              multithread=True, creationOptions=list(GTIFF_CREATION_OPTIONS))
    return out_path


def buffer_mask_by_cells(source_mask, radius_cells):
    """The source mask grown by a disk of the given radius, which on this grid is kilometres.

    A disk rather than a square: a square would call a cell 14 km away accessible on the
    diagonal, which is not what a 10 km buffer means.
    """
    from scipy import ndimage
    span = np.arange(-radius_cells, radius_cells + 1)
    y_offsets, x_offsets = np.meshgrid(span, span, indexing='ij')
    disk = (y_offsets ** 2 + x_offsets ** 2) <= radius_cells ** 2
    return ndimage.binary_dilation(source_mask, structure=disk)


def accessible_forest_hectares_by_country(lulc_equal_area_path, access_mask, country_id_path,
                                          n_countries):
    """Accessible forest hectares summed per country id, on the equal-area grid.

    Forest is read from the land-cover raster warped to this grid by dominant class, so a 1 km
    cell counts as forest when most of it is forest. Hectares are exact here, one cell being
    100 of them, which is why the whole stage moved onto this grid.
    """
    lulc = gdal.Open(lulc_equal_area_path).ReadAsArray()
    countries = gdal.Open(country_id_path).ReadAsArray()
    forest = nf.forest_mask(lulc)
    per_cell_hectares = nf.accessible_forest_hectares(
        forest, access_mask, np.full(forest.shape, HECTARES_PER_CELL, dtype=np.float32))
    return nf.hectares_by_zone(per_cell_hectares, countries, n_countries)


def build_access_mask(road_length_path, river_mask_path, out_path):
    """Roads and rivers combined into one 0/1 source mask on their shared grid."""
    # The datasets are held in their own names: a band taken off gdal.Open(...) directly outlives
    # the dataset only until the next collection, and then reading it raises on the C pointer.
    roads_dataset = gdal.Open(road_length_path)
    rivers_dataset = gdal.Open(river_mask_path)
    roads_band = roads_dataset.GetRasterBand(1)
    rivers_band = rivers_dataset.GetRasterBand(1)
    roads = roads_band.ReadAsArray().astype(np.float32)
    rivers = rivers_band.ReadAsArray().astype(np.float32)
    mask = nf.access_source_mask(roads, rivers, road_ndv=roads_band.GetNoDataValue(),
                                 river_ndv=None)

    reference = roads_dataset
    target = gdal.GetDriverByName('GTiff').Create(
        out_path, reference.RasterXSize, reference.RasterYSize, 1, gdal.GDT_Byte,
        options=list(GTIFF_CREATION_OPTIONS))
    target.SetGeoTransform(reference.GetGeoTransform())
    target.SetProjection(reference.GetProjection())
    target.GetRasterBand(1).WriteArray(mask.astype(np.uint8))
    target.GetRasterBand(1).SetNoDataValue(255)
    target = None
    return out_path

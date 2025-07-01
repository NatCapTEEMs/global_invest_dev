"""
Pollination sufficiency analysis. This is based off the IPBES-Pollination
project so that we can run on any new LULC scenarios with ESA classification.
Used to be called dasgupta_agriculture.py but then we did it for more than just Dasgupta
"""
import argparse
import collections
import glob
import itertools
import logging
import multiprocessing
import os
import re
import sys
import time
import zipfile

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
# import ecoshard
import numpy
import pandas
import pygeoprocessing
import rtree
import scipy.ndimage.morphology
import shapely.prepared
import shapely.wkb
import taskgraph

# format of the key pairs is [data suffix]: [landcover raster]
# these must be ESA landcover map type
# LANDCOVER_DATA_MAP = {
#     'data_suffix': 'landcover raster.tif',
# }

LANDCOVER_DATA_MAP = {
    'data_suffix': 'landcover_seals_raster.tif',  # Update with the new SEALS raster.
}


# set a limit for the cache
gdal.SetCacheMax(2**28)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(pathname)s.%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger('pollination')
logging.getLogger('taskgraph').setLevel(logging.INFO)

_MULT_NODATA = -1
_MASK_NODATA = 2
# the following are the globio landcover codes. A tuple (x, y) indicates the
# inclusive range from x to y. Pollinator habitat was defined as any natural
# land covers, as defined (GLOBIO land-cover classes 6, secondary vegetation,
# and  50-180, various types of primary vegetation). To test sensitivity to
# this definition we included "semi-natural" habitats (GLOBIO land-cover
# classes 3, 4, and 5; pasture, rangeland and forestry, respectively) in
# addition to "natural", and repeated all analyses with semi-natural  plus
# natural habitats, but this did not substantially alter the results  so we do
# not include it in our final analysis or code base.

#ESA
# GLOBIO_AG_CODES = [2, (10, 40), (230, 232)]
# GLOBIO_NATURAL_CODES = [6, (50, 180)]
# BMP_LULC_CODES = [300]
#SEALS
GLOBIO_AG_CODES = [2]
GLOBIO_NATURAL_CODES = [(3, 5)]
BMP_LULC_CODES = [300]

WORKING_DIR = './workspace_poll_suff'
ECOSHARD_DIR = os.path.join(WORKING_DIR, 'ecoshard_dir')
CHURN_DIR = os.path.join(WORKING_DIR, 'churn')

NODATA = -9999
N_WORKERS = max(1, multiprocessing.cpu_count())


def calculate_for_landcover(task_graph, landcover_path):
    """Calculate values for a given landcover.
    Parameters:
        task_graph (taskgraph.TaskGraph): taskgraph object used to schedule
            work.
        landcover_path (str): path to a landcover map with globio style
            landcover codes.

    Returns:
        None.
    """
    landcover_key = os.path.splitext(os.path.basename(landcover_path))[0]
    output_dir = os.path.join(WORKING_DIR, landcover_key)
    for dir_path in [output_dir, ECOSHARD_DIR, CHURN_DIR]:
        try:
            os.makedirs(dir_path)
        except OSError:
            pass


        #NEEDED
    # The proportional area of natural within 2 km was calculated for every
    #  pixel of agricultural land (GLOBIO land-cover classes 2, 230, 231, and
    #  232) at 10 arc seconds (~300 m) resolution. This 2 km scale represents
    #  the distance most commonly found to be predictive of pollination
    #  services (Kennedy et al. 2013).
    kernel_raster_path = os.path.join(CHURN_DIR, 'radial_kernel.tif')
    kernel_task = task_graph.add_task(
        func=create_radial_convolution_mask,
        args=(0.00277778, 2000., kernel_raster_path),
        target_path_list=[kernel_raster_path],
        task_name='make convolution kernel')

    # This loop is so we don't duplicate code for each mask type with the
    # only difference being the lulc codes and prefix
    mask_task_path_map = {}
    for mask_prefix, globio_codes in [
            ('ag', GLOBIO_AG_CODES), ('hab', GLOBIO_NATURAL_CODES),
            ('bmp', BMP_LULC_CODES)]:
        mask_key = f'{landcover_key}_{mask_prefix}_mask'
        mask_target_path = os.path.join(
            CHURN_DIR, f'{mask_prefix}_mask',
            f'{mask_key}.tif')
        mask_task = task_graph.add_task(
            func=mask_raster,
            args=(landcover_path, globio_codes, mask_target_path),
            target_path_list=[mask_target_path],
            task_name=f'mask {mask_key}',)

        mask_task_path_map[mask_prefix] = (mask_task, mask_target_path)


    pollhab_2km_prop_path = os.path.join(
        CHURN_DIR, 'pollhab_2km_prop',
        f'pollhab_2km_prop_{landcover_key}.tif')
    pollhab_2km_prop_task = task_graph.add_task(
        func=pygeoprocessing.convolve_2d,
        args=[
            (mask_task_path_map['hab'][1], 1), (kernel_raster_path, 1),
            pollhab_2km_prop_path],
        kwargs={
            'working_dir': CHURN_DIR,
            'ignore_nodata_and_edges': True},
        dependent_task_list=[mask_task_path_map['hab'][0], kernel_task],
        target_path_list=[pollhab_2km_prop_path],
        task_name=(
            'calculate proportional'
            f' {os.path.basename(pollhab_2km_prop_path)}'))

    # calculate pollhab_2km_prop_on_ag_10s by multiplying pollhab_2km_prop
    # by the ag mask
    pollhab_2km_prop_on_ag_path = os.path.join(
        output_dir, f'''pollhab_2km_prop_on_ag_10s_{
            landcover_key}.tif''')
    pollhab_2km_prop_on_ag_task = task_graph.add_task(
        func=mult_rasters,
        args=(
            mask_task_path_map['ag'][1], pollhab_2km_prop_path,
            pollhab_2km_prop_on_ag_path),
        target_path_list=[pollhab_2km_prop_on_ag_path],
        dependent_task_list=[
            pollhab_2km_prop_task, mask_task_path_map['ag'][0]],
        task_name=(
            f'''pollhab 2km prop on ag {
                os.path.basename(pollhab_2km_prop_on_ag_path)}'''))

    #  1.1.4.  Sufficiency threshold A threshold of 0.3 was set to
    #  evaluate whether there was sufficient pollinator habitat in the 2
    #  km around farmland to provide pollination services, based on
    #  Kremen et al.'s (2005)  estimate of the area requirements for
    #  achieving full pollination. This produced a map of wild
    #  pollination sufficiency where every agricultural pixel was
    #  designated in a binary fashion: 0 if proportional area of habitat
    #  was less than 0.3; 1 if greater than 0.3. Maps of pollination
    #  sufficiency can be found at (permanent link to output), outputs
    #  "poll_suff_..." below.

    threshold_val = 0.3
    pollinator_suff_hab_path = os.path.join(
        CHURN_DIR, 'poll_suff_hab_ag_coverage_rasters',
        f'poll_suff_ag_coverage_prop_10s_{landcover_key}.tif')
    poll_suff_task = task_graph.add_task(
        func=threshold_select_raster,
        args=(
            pollhab_2km_prop_path,
            mask_task_path_map['ag'][1], threshold_val,
            pollinator_suff_hab_path),
        target_path_list=[pollinator_suff_hab_path],
        dependent_task_list=[
            pollhab_2km_prop_task, mask_task_path_map['ag'][0]],
        task_name=f"""poll_suff_ag_coverage_prop {
            os.path.basename(pollinator_suff_hab_path)}""")


def build_spatial_index(vector_path):
    """Build an rtree/geom list tuple from ``vector_path``."""
    vector = gdal.OpenEx(vector_path)
    layer = vector.GetLayer()
    geom_index = rtree.index.Index()
    geom_list = []
    for index in range(layer.GetFeatureCount()):
        feature = layer.GetFeature(index)
        geom = feature.GetGeometryRef()
        shapely_geom = shapely.wkb.loads(geom.ExportToWkb())
        shapely_prep_geom = shapely.prepared.prep(shapely_geom)
        geom_list.append(shapely_prep_geom)
        geom_index.insert(index, shapely_geom.bounds)

    return geom_index, geom_list


def calculate_total_requirements(
        pop_path_list, nut_need_list, target_path):
    """Calculate total nutrient requirements.
    Create a new raster by summing all rasters in `pop_path_list` multiplied
    by their corresponding scalar in `nut_need_list`.
    Parameters:
        pop_path_list (list of str): list of paths to population counts.
        nut_need_list (list): list of scalars that correspond in order to
            the per-count nutrient needs of `pop_path_list`.
        target_path (str): path to target file.
    Return:
        None.
    """
    nodata = -1
    pop_nodata = pygeoprocessing.get_raster_info(
        pop_path_list[0])['nodata'][0]

    def mult_and_sum(*arg_list):
        """Arg list is an (array0, scalar0, array1, scalar1,...) list.
        Returns:
            array0*scalar0 + array1*scalar1 + .... but ignore nodata.
        """
        result = numpy.empty(arg_list[0].shape, dtype=numpy.float32)
        result[:] = nodata
        array_stack = numpy.array(arg_list[0::2])
        scalar_list = numpy.array(arg_list[1::2])
        # make a valid mask as big as a single array
        valid_mask = numpy.logical_and.reduce(
            array_stack != pop_nodata, axis=0)

        # mask out all invalid elements but reshape so there's still the same
        # number of arrays
        valid_array_elements = (
            array_stack[numpy.broadcast_to(valid_mask, array_stack.shape)])
        array_stack = None

        # sometimes this array is empty, check first before reshaping
        if valid_array_elements.size != 0:
            valid_array_elements = valid_array_elements.reshape(
                -1, numpy.count_nonzero(valid_mask))
            # multiply each element of the scalar with each row of the valid
            # array stack, then sum along the 0 axis to get the result
            result[valid_mask] = numpy.sum(
                (valid_array_elements.T * scalar_list).T, axis=0)
        scalar_list = None
        valid_mask = None
        valid_array_elements = None
        return result

    pygeoprocessing.raster_calculator(list(itertools.chain(*[
        ((path, 1), (scalar, 'raw')) for path, scalar in zip(
            pop_path_list, nut_need_list)])), mult_and_sum, target_path,
        gdal.GDT_Float32, nodata)


def sub_two_op(a_array, b_array, a_nodata, b_nodata, target_nodata):
    """Subtract a from b and ignore nodata."""
    result = numpy.empty_like(a_array)
    result[:] = target_nodata
    valid_mask = (a_array != a_nodata) & (b_array != b_nodata)
    result[valid_mask] = a_array[valid_mask] - b_array[valid_mask]
    return result


def average_rasters(*raster_list, clamp=None):
    """Average rasters in raster list except write to the last one.
    Parameters:
        raster_list (list of string): list of rasters to average over.
        clamp (float): value to clamp the individual raster to before the
            average.
    Returns:
        None.
    """
    nodata_list = [
        pygeoprocessing.get_raster_info(path)['nodata'][0]
        for path in raster_list[:-1]]
    target_nodata = -1.

    def average_op(*array_list):
        result = numpy.empty_like(array_list[0])
        result[:] = target_nodata
        valid_mask = numpy.ones(result.shape, dtype=numpy.bool)
        clamped_list = []
        for array, nodata in zip(array_list, nodata_list):
            valid_mask &= array != nodata
            if clamp:
                clamped_list.append(
                    numpy.where(array > clamp, clamp, array))
            else:
                clamped_list.append(array)

        if valid_mask.any():
            array_stack = numpy.stack(clamped_list)
            result[valid_mask] = numpy.average(
                array_stack[numpy.broadcast_to(
                    valid_mask, array_stack.shape)].reshape(
                        len(array_list), -1), axis=0)
        return result

    pygeoprocessing.raster_calculator(
        [(path, 1) for path in raster_list[:-1]], average_op,
        raster_list[-1], gdal.GDT_Float32, target_nodata)


def subtract_2_rasters(
        raster_path_a, raster_path_b, target_path):
    """Calculate target = a-b and ignore nodata."""
    a_nodata = pygeoprocessing.get_raster_info(raster_path_a)['nodata'][0]
    b_nodata = pygeoprocessing.get_raster_info(raster_path_b)['nodata'][0]
    target_nodata = -9999

    def sub_op(a_array, b_array):
        """Sub a-b-c as arrays."""
        result = numpy.empty(a_array.shape, dtype=numpy.float32)
        result[:] = target_nodata
        valid_mask = (
            ~numpy.isclose(a_array, a_nodata) &
            ~numpy.isclose(b_array, b_nodata))
        result[valid_mask] = (
            a_array[valid_mask] - b_array[valid_mask])
        return result

    pygeoprocessing.raster_calculator(
        [(raster_path_a, 1), (raster_path_b, 1)],
        sub_op, target_path, gdal.GDT_Float32, target_nodata)


def subtract_3_rasters(
        raster_path_a, raster_path_b, raster_path_c, target_path):
    """Calculate target = a-b-c and ignore nodata."""
    a_nodata = pygeoprocessing.get_raster_info(raster_path_a)['nodata'][0]
    b_nodata = pygeoprocessing.get_raster_info(raster_path_b)['nodata'][0]
    c_nodata = pygeoprocessing.get_raster_info(raster_path_c)['nodata'][0]
    target_nodata = -9999

    def sub_op(a_array, b_array, c_array):
        """Sub a-b-c as arrays."""
        result = numpy.empty(a_array.shape, dtype=numpy.float32)
        result[:] = target_nodata
        valid_mask = (
            (a_array != a_nodata) &
            (b_array != b_nodata) &
            (c_array != c_nodata))
        result[valid_mask] = (
            a_array[valid_mask] - b_array[valid_mask] - c_array[valid_mask])
        return result

    pygeoprocessing.raster_calculator(
        [(raster_path_a, 1), (raster_path_b, 1), (raster_path_c, 1)],
        sub_op, target_path, gdal.GDT_Float32, target_nodata)


def create_radial_convolution_mask(
        pixel_size_degree, radius_meters, kernel_filepath):
    """Create a radial mask to sample pixels in convolution filter.
    Parameters:
        pixel_size_degree (float): size of pixel in degrees.
        radius_meters (float): desired size of radial mask in meters.
    Returns:
        A 2D numpy array that can be used in a convolution to aggregate a
        raster while accounting for partial coverage of the circle on the
        edges of the pixel.
    """
    degree_len_0 = 110574  # length at 0 degrees
    degree_len_60 = 111412  # length at 60 degrees
    pixel_size_m = pixel_size_degree * (degree_len_0 + degree_len_60) / 2.0
    pixel_radius = numpy.ceil(radius_meters / pixel_size_m)
    n_pixels = (int(pixel_radius) * 2 + 1)
    sample_pixels = 200
    mask = numpy.ones((sample_pixels * n_pixels, sample_pixels * n_pixels))
    mask[mask.shape[0]//2, mask.shape[0]//2] = 0
    distance_transform = scipy.ndimage.morphology.distance_transform_edt(mask)
    mask = None
    stratified_distance = distance_transform * pixel_size_m / sample_pixels
    distance_transform = None
    in_circle = numpy.where(stratified_distance <= 2000.0, 1.0, 0.0)
    stratified_distance = None
    reshaped = in_circle.reshape(
        in_circle.shape[0] // sample_pixels, sample_pixels,
        in_circle.shape[1] // sample_pixels, sample_pixels)
    kernel_array = numpy.sum(reshaped, axis=(1, 3)) / sample_pixels**2
    normalized_kernel_array = kernel_array / numpy.sum(kernel_array)
    reshaped = None

    driver = gdal.GetDriverByName('GTiff')
    kernel_raster = driver.Create(
        kernel_filepath.encode('utf-8'), n_pixels, n_pixels, 1,
        gdal.GDT_Float32, options=[
            'BIGTIFF=IF_SAFER', 'TILED=YES', 'BLOCKXSIZE=256',
            'BLOCKYSIZE=256'])

    # Make some kind of geotransform, it doesn't matter what but
    # will make GIS libraries behave better if it's all defined
    kernel_raster.SetGeoTransform([-180, 1, 0, 90, 0, -1])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    kernel_raster.SetProjection(srs.ExportToWkt())
    kernel_band = kernel_raster.GetRasterBand(1)
    kernel_band.SetNoDataValue(NODATA)
    kernel_band.WriteArray(normalized_kernel_array)


def threshold_select_raster(
        base_raster_path, select_raster_path, threshold_val, target_path):
    """Select `select` if `base` >= `threshold_val`.
    Parameters:
        base_raster_path (string): path to single band raster that will be
            used to determine the threshold mask to select from
            `select_raster_path`.
        select_raster_path (string): path to single band raster to pass
            through to target if aligned `base` pixel is >= `threshold_val`
            0 otherwise, or nodata if base == nodata. Must be the same
            shape as `base_raster_path`.
        threshold_val (numeric): value to use as threshold cutoff
        target_path (string): path to desired output raster, raster is a
            byte type with same dimensions and projection as
            `base_raster_path`. A pixel in this raster will be `select` if
            the corresponding pixel in `base_raster_path` is >=
            `threshold_val`, 0 otherwise or nodata if `base` == nodata.
    Returns:
        None.
    """
    base_nodata = pygeoprocessing.get_raster_info(
        base_raster_path)['nodata'][0]
    target_nodata = -9999.

    def threshold_select_op(
            base_array, select_array, threshold_val, base_nodata,
            target_nodata):
        result = numpy.empty(select_array.shape, dtype=numpy.float32)
        result[:] = target_nodata
        valid_mask = (base_array != base_nodata) & (
            select_array >= 0) & (select_array <= 1)
        result[valid_mask] = select_array[valid_mask] * numpy.interp(
            base_array[valid_mask], [0, threshold_val], [0.0, 1.0], 0, 1)
        return result

    pygeoprocessing.raster_calculator(
        [(base_raster_path, 1), (select_raster_path, 1),
         (threshold_val, 'raw'), (base_nodata, 'raw'),
         (target_nodata, 'raw')], threshold_select_op,
        target_path, gdal.GDT_Float32, target_nodata)


def mask_raster(base_path, codes, target_path):
    """Mask `base_path` to 1 where values are in codes. 0 otherwise.
    Parameters:
        base_path (string): path to single band integer raster.
        codes (list): list of integer or tuple integer pairs. Membership in
            `codes` or within the inclusive range of a tuple in `codes`
            is sufficient to mask the corresponding raster integer value
            in `base_path` to 1 for `target_path`.
        target_path (string): path to desired mask raster. Any corresponding
            pixels in `base_path` that also match a value or range in
            `codes` will be masked to 1 in `target_path`. All other values
            are 0.
    Returns:
        None.
    """
    code_list = numpy.array([
        item for sublist in [
            range(x[0], x[1]+1) if isinstance(x, tuple) else [x]
            for x in codes] for item in sublist])
    LOGGER.debug(f'expanded code array {code_list}')

    base_nodata = pygeoprocessing.get_raster_info(base_path)['nodata'][0]

    def mask_codes_op(base_array, codes_array):
        """Return a bool raster if value in base_array is in codes_array."""
        result = numpy.empty(base_array.shape, dtype=numpy.int8)
        result[:] = _MASK_NODATA
        valid_mask = base_array != base_nodata
        result[valid_mask] = numpy.isin(
            base_array[valid_mask], codes_array)
        return result

    pygeoprocessing.raster_calculator(
        [(base_path, 1), (code_list, 'raw')], mask_codes_op, target_path,
        gdal.GDT_Byte, 2)


def unzip_file(zipfile_path, target_dir, touchfile_path):
    """Unzip contents of `zipfile_path`.
    Parameters:
        zipfile_path (string): path to a zipped file.
        target_dir (string): path to extract zip file to.
        touchfile_path (string): path to a file to create if unzipping is
            successful.
    Returns:
        None.
    """
    with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

    with open(touchfile_path, 'w') as touchfile:
        touchfile.write(f'unzipped {zipfile_path}')


def _make_logger_callback(message):
    """Build a timed logger callback that prints `message` replaced.
    Parameters:
        message (string): a string that expects a %f placement variable,
            for % complete.
    Returns:
        Function with signature:
            logger_callback(df_complete, psz_message, p_progress_arg)
    """
    def logger_callback(df_complete, psz_message, p_progress_arg):
        """Log updates using GDAL API for callbacks."""
        try:
            current_time = time.time()
            if ((current_time - logger_callback.last_time) > 5.0 or
                    (df_complete == 1.0 and
                     logger_callback.total_time >= 5.0)):
                LOGGER.info(message, df_complete * 100)
                logger_callback.last_time = current_time
                logger_callback.total_time += current_time
        except AttributeError:
            logger_callback.last_time = time.time()
            logger_callback.total_time = 0.0

    return logger_callback






def area_of_pixel(pixel_size, center_lat):
    """Calculate m^2 area of a wgs84 square pixel.
    Adapted from: https://gis.stackexchange.com/a/127327/2397
    Parameters:
        pixel_size (float): length of side of pixel in degrees.
        center_lat (float): latitude of the center of the pixel. Note this
            value +/- half the `pixel-size` must not exceed 90/-90 degrees
            latitude or an invalid area will be calculated.
    Returns:
        Area of square pixel of side length `pixel_size` centered at
        `center_lat` in m^2.
    """
    a = 6378137  # meters
    b = 6356752.3142  # meters
    e = numpy.sqrt(1-(b/a)**2)
    area_list = []
    for f in [center_lat+pixel_size/2, center_lat-pixel_size/2]:
        zm = 1 - e*numpy.sin(numpy.radians(f))
        zp = 1 + e*numpy.sin(numpy.radians(f))
        area_list.append(
            numpy.pi * b**2 * (
                numpy.log(zp/zm) / (2*e) +
                numpy.sin(numpy.radians(f)) / (zp*zm)))
    return pixel_size / 360. * (area_list[0]-area_list[1])


def _mult_raster_op(array_a, array_b, nodata_a, nodata_b, target_nodata):
    """Multiply a by b and skip nodata."""
    result = numpy.empty(array_a.shape, dtype=numpy.float32)
    result[:] = target_nodata
    valid_mask = (array_a != nodata_a) & (array_b != nodata_b)
    result[valid_mask] = array_a[valid_mask] * array_b[valid_mask]
    return result


def mult_rasters(raster_a_path, raster_b_path, target_path):
    """Multiply a by b and skip nodata."""
    raster_info_a = pygeoprocessing.get_raster_info(raster_a_path)
    raster_info_b = pygeoprocessing.get_raster_info(raster_b_path)

    nodata_a = raster_info_a['nodata'][0]
    nodata_b = raster_info_b['nodata'][0]

    if raster_info_a['raster_size'] != raster_info_b['raster_size']:
        aligned_raster_a_path = (
            target_path + os.path.basename(raster_a_path) + '_aligned.tif')
        aligned_raster_b_path = (
            target_path + os.path.basename(raster_b_path) + '_aligned.tif')
        pygeoprocessing.align_and_resize_raster_stack(
            [raster_a_path, raster_b_path],
            [aligned_raster_a_path, aligned_raster_b_path],
            ['near'] * 2, raster_info_a['pixel_size'], 'intersection')
        raster_a_path = aligned_raster_a_path
        raster_b_path = aligned_raster_b_path

    pygeoprocessing.raster_calculator(
        [(raster_a_path, 1), (raster_b_path, 1), (nodata_a, 'raw'),
         (nodata_b, 'raw'), (_MULT_NODATA, 'raw')], _mult_raster_op,
        target_path, gdal.GDT_Float32, _MULT_NODATA)


def add_op(target_nodata, *array_list):
    """Add & return arrays in ``array_list`` but ignore ``target_nodata``."""
    result = numpy.zeros(array_list[0].shape, dtype=numpy.float32)
    valid_mask = numpy.zeros(result.shape, dtype=numpy.bool)
    for array in array_list:
        # nodata values will be < 0
        local_valid_mask = array >= 0
        valid_mask |= local_valid_mask
        result[local_valid_mask] += array[local_valid_mask]
    result[~valid_mask] = target_nodata
    return result


def sum_num_sum_denom(
        num1_array, num2_array, denom1_array, denom2_array, nodata):
    """Calculate sum of num divided by sum of denom."""
    result = numpy.empty_like(num1_array)
    result[:] = nodata
    valid_mask = (
        ~numpy.isclose(num1_array, nodata) &
        ~numpy.isclose(num2_array, nodata) &
        ~numpy.isclose(denom1_array, nodata) &
        ~numpy.isclose(denom2_array, nodata))
    result[valid_mask] = (
        num1_array[valid_mask] + num2_array[valid_mask]) / (
        denom1_array[valid_mask] + denom2_array[valid_mask] + 1e-9)
    return result


def avg_3_op(array_1, array_2, array_3, nodata):
    """Average 3 arrays. Skip nodata."""
    result = numpy.empty_like(array_1)
    result[:] = nodata
    valid_mask = (
        ~numpy.isclose(array_1, nodata) &
        ~numpy.isclose(array_2, nodata) &
        ~numpy.isclose(array_3, nodata))
    result[valid_mask] = (
        array_1[valid_mask] +
        array_2[valid_mask] +
        array_3[valid_mask]) / 3.
    return result


def weighted_avg_3_op(
        array_1, array_2, array_3,
        scalar_1, scalar_2, scalar_3,
        nodata):
    """Weighted average 3 arrays. Skip nodata."""
    result = numpy.empty_like(array_1)
    result[:] = nodata
    valid_mask = (
        ~numpy.isclose(array_1, nodata) &
        ~numpy.isclose(array_2, nodata) &
        ~numpy.isclose(array_3, nodata))
    result[valid_mask] = (
        array_1[valid_mask]/scalar_1 +
        array_2[valid_mask]/scalar_2 +
        array_3[valid_mask]/scalar_3) / 3.
    return result


def count_ge_one(array):
    """Return count of elements >= 1."""
    return numpy.count_nonzero(array >= 1)


def prop_diff_op(array_a, array_b, nodata):
    """Calculate prop change from a to b."""
    result = numpy.empty_like(array_a)
    result[:] = nodata
    valid_mask = (
        ~numpy.isclose(array_a, nodata) &
        ~numpy.isclose(array_b, nodata))
    # the 1e-12 is to prevent a divide by 0 error
    result[valid_mask] = (
        array_b[valid_mask] - array_a[valid_mask]) / (
            array_a[valid_mask] + 1e-12)
    return result


def build_lookup_from_csv(
        table_path, key_field, to_lower=True, warn_if_missing=True):
    """Read a CSV table into a dictionary indexed by `key_field`.
    Creates a dictionary from a CSV whose keys are unique entries in the CSV
    table under the column named by `key_field` and values are dictionaries
    indexed by the other columns in `table_path` including `key_field` whose
    values are the values on that row of the CSV table.
    Parameters:
        table_path (string): path to a CSV file containing at
            least the header key_field
        key_field: (string): a column in the CSV file at `table_path` that
            can uniquely identify each row in the table.
        to_lower (bool): if True, converts all unicode in the CSV,
            including headers and values to lowercase, otherwise uses raw
            string values.
        warn_if_missing (bool): If True, warnings are logged if there are
            empty headers or value rows.
    Returns:
        lookup_dict (dict): a dictionary of the form {
                key_field_0: {csv_header_0: value0, csv_header_1: value1...},
                key_field_1: {csv_header_0: valuea, csv_header_1: valueb...}
            }
        if `to_lower` all strings including key_fields and values are
        converted to lowercase unicode.
    """
    # Check if the file encoding is UTF-8 BOM first, related to issue
    # https://bitbucket.org/natcap/invest/issues/3832/invest-table-parsing-does-not-support-utf
    encoding = None
    with open(table_path) as file_obj:
        first_line = file_obj.readline()
        if first_line.startswith('\xef\xbb\xbf'):
            encoding = 'utf-8-sig'
    table = pandas.read_csv(
        table_path, sep=None, engine='python', encoding=encoding)
    header_row = list(table)
    try:  # no unicode() in python 3
        key_field = unicode(key_field)
    except NameError:
        pass
    if to_lower:
        key_field = key_field.lower()
        header_row = [
            x if not isinstance(x, str) else x.lower()
            for x in header_row]

    if key_field not in header_row:
        raise ValueError(
            '%s expected in %s for the CSV file at %s' % (
                key_field, header_row, table_path))
    if warn_if_missing and '' in header_row:
        LOGGER.warn(
            "There are empty strings in the header row at %s", table_path)

    key_index = header_row.index(key_field)
    lookup_dict = {}
    for index, row in table.iterrows():
        if to_lower:
            row = pandas.Series([
                x if not isinstance(x, str) else x.lower()
                for x in row])
        # check if every single element in the row is null
        if row.isnull().values.all():
            LOGGER.warn(
                "Encountered an entirely blank row on line %d", index+2)
            continue
        if row.isnull().values.any():
            row = row.fillna('')
        lookup_dict[row[key_index]] = dict(zip(header_row, row))
    return lookup_dict





def sum_rasters_op(*raster_nodata_list):
    """Sum all non-nodata values.
    Parameters:
        raster_nodata_list (list): list of 2n+1 length where the first n
            elements are raster array values and the second n elements are the
            nodata values for that array. The last element is the target
            nodata.
    Returns:
        sum(raster_nodata_list[0:n]) -- while accounting for nodata.
    """
    result = numpy.zeros(raster_nodata_list[0].shape, dtype=numpy.float32)
    nodata_mask = numpy.zeros(raster_nodata_list[0].shape, dtype=numpy.bool)
    n = len(raster_nodata_list) // 2
    for index in range(n):
        valid_mask = ~numpy.isclose(
            raster_nodata_list[index], raster_nodata_list[index+n])
        nodata_mask |= ~valid_mask
        result[valid_mask] += raster_nodata_list[index][valid_mask]
    result[nodata_mask] = raster_nodata_list[-1]
    return result


def dot_prod_op(scalar, *raster_nodata_list):
    """Do a dot product of vectors A*B.
    Parameters:
        scalar (float): value to multiply each pair by.
        raster_nodata_list (list): list of 4*n+1 length where the first n
            elements are from vector A, the next n are B, and last 2n are
            nodata values for those elements in order.
    Returns:
        A*B and nodata where it overlaps.
    """
    n_elements = (len(raster_nodata_list)-1) // 4
    result = numpy.zeros(raster_nodata_list[0].shape, dtype=numpy.float32)
    nodata_mask = numpy.zeros(result.shape, dtype=numpy.bool)
    for index in range(n_elements*2):
        nodata_mask |= numpy.isclose(
            raster_nodata_list[index],
            raster_nodata_list[index+n_elements*2]) | numpy.isnan(
            raster_nodata_list[index])
    for index in range(n_elements):
        result[~nodata_mask] += (
            scalar * raster_nodata_list[index][~nodata_mask] *
            raster_nodata_list[index+n_elements][~nodata_mask])
    result[nodata_mask] = raster_nodata_list[-1]
    return result

def execute(landcover_path, working_dir):
    global WORKING_DIR, ECOSHARD_DIR, CHURN_DIR
    WORKING_DIR = working_dir
    ECOSHARD_DIR = os.path.join(WORKING_DIR, 'ecoshard_dir')
    CHURN_DIR = os.path.join(WORKING_DIR, 'churn')

    print('WORKING_DIR', WORKING_DIR)
    print('ECOSHARD_DIR', ECOSHARD_DIR)
    print('CHURN_DIR', CHURN_DIR)

    landcover_raster_list = []
    landcover_raster_list.append(landcover_path)

    task_graph = taskgraph.TaskGraph(
        WORKING_DIR, N_WORKERS, reporting_interval=5.0)
    for dir_path in [
            ECOSHARD_DIR, CHURN_DIR]:
        try:
            os.makedirs(dir_path)
        except OSError:
            pass



    time.sleep(1.0)

    for landcover_path in landcover_raster_list:
        LOGGER.info("process landcover map: %s", landcover_path)
        calculate_for_landcover(task_graph, landcover_path)

    task_graph.join()
    task_graph.close()


if __name__ == '__main__':
    # Set the landcover raster path and working directory
    #kernel size, nodata values, and spatial reference 
    #landcover_path = r"D:\Shared drives\NatCapTEEMs\Projects\WB_MANAGE_Project\ES\LC_SEALS\withFullBoundary\lulc_esa_seals7_2017_full_IND_linear.tif"
    #landcover_path = r"D:\Shared drives\NatCapTEEMs\Projects\WB_MANAGE_Project\ES\LC_SEALS\withFullBoundary\lulc_esa_seals7_2017.tif"
    landcover_path = r"D:\Shared drives\NatCapTEEMs\Projects\WB_MANAGE_Project\ES\LC_SEALS\withFullBoundary\lulc_esa_seals7_ssp2_rcp45_luh2-message_bau_shift_2022_2022.tif"
    working_dir = r'D:\Shared drives\NatCapTEEMs\Projects\WB_MANAGE_Project\Shocks\pollination\workspace_poll_suff/seals_ori'

    # Call execute with the required arguments
    execute(landcover_path, working_dir)
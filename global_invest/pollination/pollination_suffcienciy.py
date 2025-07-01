import os
import sys
import taskgraph
import logging
import pygeoprocessing
import numpy
import scipy.ndimage.morphology
import shapely.wkb
import rtree
import pandas as pd
from osgeo import gdal, ogr, osr
import errno

WORKING_DIR = r'D:\Shared drives\NatCapTEEMs\Projects\WB_MANAGE_Project\Shocks\pollination\workspace_poll_suff'
ECOSHARD_DIR = os.path.join(WORKING_DIR, 'ecoshard_dir')
CHURN_DIR = os.path.join(WORKING_DIR, 'churn')

N_WORKERS = 4  # Modify based on your system's capacity

# File Paths
lulc_2017_path = r"D:\Shared drives\NatCapTEEMs\Projects\WB_MANAGE_Project\ES\LC_SEALS\withFullBoundary\lulc_esa_seals7_2017_full_IND_linear.tif"
lulc_2022_path = r"D:\Shared drives\NatCapTEEMs\Projects\WB_MANAGE_Project\ES\LC_SEALS\withFullBoundary\lulc_esa_seals7_ssp2_rcp45_luh2-message_bau_shift_2022_2022_full_IND_linear.tif"

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(pathname)s.%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger('pollination_analysis')
logging.getLogger('taskgraph').setLevel(logging.INFO)


def create_radial_convolution_mask(pixel_size_degree, radius_meters, kernel_filepath):
    """
    Create a radial convolution mask raster.
    
    Args:
        pixel_size_degree (float): Size of each pixel in degrees.
        radius_meters (float): Radius of the convolution mask in meters.
        kernel_filepath (str): Path to save the convolution mask raster.
    """
    n_pixels = int(radius_meters / (111320 * pixel_size_degree))
    LOGGER.debug(f"Number of pixels for radius: {n_pixels}")
    kernel_matrix = numpy.zeros((n_pixels*2+1, n_pixels*2+1))
    y, x = numpy.ogrid[-n_pixels:n_pixels+1, -n_pixels:n_pixels+1]
    mask = x**2 + y**2 <= n_pixels**2
    kernel_matrix[mask] = 1
    pygeoprocessing.numpy_array_to_raster(
        kernel_matrix, -1, (pixel_size_degree, -pixel_size_degree),
        (0, 0), None, kernel_filepath)


def mask_raster(base_path, codes, target_path):
    """
    Mask the raster to include only the specified landcover codes.
    
    Args:
        base_path (str): Path to the base raster.
        codes (list): List of integer codes to mask.
        target_path (str): Path to save the masked raster.
    """
    def mask_op(base_array):
        result = numpy.isin(base_array, codes).astype(numpy.uint8)
        return result

    pygeoprocessing.raster_calculator(
        [(base_path, 1)], mask_op, target_path, gdal.GDT_Byte, -1)


def threshold_select_raster(base_raster_path, select_raster_path, threshold_val, target_path):
    """
    Apply a threshold to select raster values.
    
    Args:
        base_raster_path (str): Path to the base raster.
        select_raster_path (str): Path to the raster used for threshold selection.
        threshold_val (float): Threshold value for selection.
        target_path (str): Path to save the result raster.
    """
    def threshold_select_op(base_array, select_array):
        result = numpy.where(select_array > threshold_val, base_array, 0)
        return result

    pygeoprocessing.raster_calculator(
        [(base_raster_path, 1), (select_raster_path, 1)],
        threshold_select_op, target_path, gdal.GDT_Float32, -9999)


def calculate_for_landcover(task_graph, landcover_path):
    landcover_key = os.path.splitext(os.path.basename(landcover_path))[0]
    output_dir = os.path.join(WORKING_DIR, landcover_key)
    for dir_path in [output_dir, ECOSHARD_DIR, CHURN_DIR]:
        try:
            os.makedirs(dir_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                LOGGER.error(f"Failed to create directory {dir_path}: {e}")
            else:
                LOGGER.info(f"Directory {dir_path} already exists.")

    # Create radial convolution mask
    kernel_raster_path = os.path.join(CHURN_DIR, 'radial_kernel.tif')
    kernel_task = task_graph.add_task(
        func=create_radial_convolution_mask,
        args=(0.00277778, 2000., kernel_raster_path),
        target_path_list=[kernel_raster_path],
        task_name='make convolution kernel')

    # Mask agricultural and natural areas
    mask_task_path_map = {}
    for mask_prefix, original_codes in [
            ('ag', [2]), ('hab', [3, 4, 5])]:
        mask_key = f'{landcover_key}_{mask_prefix}_mask'
        mask_target_path = os.path.join(
            CHURN_DIR, f'{mask_prefix}_mask',
            f'{mask_key}.tif')
        mask_task = task_graph.add_task(
            func=mask_raster,
            args=(landcover_path, original_codes, mask_target_path),
            target_path_list=[mask_target_path],
            task_name=f'mask {mask_key}',)

        mask_task_path_map[mask_prefix] = (mask_task, mask_target_path)

    # Calculate proportional area of habitat
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
        task_name=(f'calculate proportional {os.path.basename(pollhab_2km_prop_path)}'))

    # Calculate pollhab_2km_prop_on_ag_10s by multiplying pollhab_2km_prop by the ag mask
    pollhab_2km_prop_on_ag_path = os.path.join(
        output_dir, f'pollhab_2km_prop_on_ag_10s_{landcover_key}.tif')
    pollhab_2km_prop_on_ag_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            [(mask_task_path_map['ag'][1], 1), (pollhab_2km_prop_path, 1)],
            lambda a, b: a * b,
            pollhab_2km_prop_on_ag_path,
            gdal.GDT_Float32,
            -9999),
        dependent_task_list=[pollhab_2km_prop_task, mask_task_path_map['ag'][0]],
        target_path_list=[pollhab_2km_prop_on_ag_path],
        task_name=f'calculate {os.path.basename(pollhab_2km_prop_on_ag_path)}')


if __name__ == '__main__':
    task_graph = taskgraph.TaskGraph(WORKING_DIR, N_WORKERS, reporting_interval=5.0)
    calculate_for_landcover(task_graph, lulc_2017_path)
    calculate_for_landcover(task_graph, lulc_2022_path)
    task_graph.join()
    task_graph.close()

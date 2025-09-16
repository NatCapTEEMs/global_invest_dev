import os
import sys, ast
import numpy as np
import pandas as pd
import yaml
import geopandas as gpd
import pygeoprocessing.geoprocessing as geo
from osgeo import gdal
gdal.UseExceptions()

import natcap.invest.ndr.ndr
import time

import multiprocessing as mp
import shutil

import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box


config  = sys.argv[1]           # config  = "/users/5/salmamun/CASC/ESMidwestCASC/casc_sm_msi.yaml"
with open(config) as yaml_data_file:
    args = yaml.load(yaml_data_file, Loader=yaml.FullLoader)

common_inputs = args['common_input_folder']
country_folders = args["result_folder"]
projected_crs = args["projected_crs"]
country_vector = gpd.read_file(args["country_vector"])
es_dict = args["es_dict"]
if args["parallelized"]:
    args["n_workers"] = -1
    n_cores = args["n_cores"]
else:
    args["n_workers"] = args["n_cores"]

def slice_inputs(base_raster, source_raster_dict, aoi_vector, target_folder):
    """
    Assumes that `source_raster_dict` will be keyed as new_name: orig_path
    """
    base_lulc_info = geo.get_raster_info(base_raster)
    
    src_paths = []
    dst_paths = []
    dst_dict = {}
    for k, v in source_raster_dict.items():
        if os.path.splitext(v)[1] in [".tif", ".bil", ".img"]:
            src_paths.append(v)
            dst_paths.append(os.path.join(target_folder, f'{k}.tif'))
            dst_dict[k] = os.path.join(target_folder, f'{k}.tif')
    
    try:
        raster_align_index = int([i for i, x in enumerate(src_paths) if x == source_raster_dict[f'dem_7755']][0])
    except:
        raster_align_index = None

    geo.align_and_resize_raster_stack(
        base_raster_path_list=src_paths,
        target_raster_path_list=dst_paths,
        resample_method_list=['near' for _ in src_paths],
        target_pixel_size=base_lulc_info['pixel_size'],
        bounding_box_mode="intersection",
        base_vector_path_list=[aoi_vector],
        raster_align_index=raster_align_index,
        vector_mask_options={'mask_vector_path': aoi_vector},
    )
    return dst_dict

def reproject_raster(raster_path, target_raster_path, dst_crs='EPSG:3857'):
    '''
    This function reprojects the raster to EPSG:7755 or any given crs.
    This helper function is used in data_preparation function.
    '''
    with rio.open(raster_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            "compress": "LZW"
        })
        if src.crs != dst_crs:
            with rio.open(target_raster_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rio.band(src, i),
                        destination=rio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest)
        else:
            shutil.copy(raster_path, target_raster_path)

def delete_intermediate_files(path_to_dir):
    shutil.rmtree(path_to_dir)

def create_aoi_buffer(aoi_path, aoi_buffer_path, projected_crs, aoi_buffer_distance=5000):
    '''
    This function creates a buffer around the aoi shapefile.
    The buffer distance is defined in the parameters file.
    '''    
    aoi_gdf = gpd.read_file(aoi_path)
    aoi_projected_gdf = aoi_gdf.to_crs(f"EPSG: {projected_crs}")
    aoi_projected_path = aoi_path.replace('.gpkg', f'_{projected_crs}.gpkg')
    aoi_projected_gdf.to_file(aoi_projected_path, driver='GPKG')
    aoi_projected_gdf['geometry'] = aoi_projected_gdf.buffer(aoi_buffer_distance)
    aoi_projected_gdf.to_file(aoi_path.replace('.gpkg', f'_buffer_{projected_crs}.gpkg'), driver='GPKG')
    aoi_buffer_gdf = aoi_projected_gdf.to_crs(4326)      # converting back to 4326 for saving the file.
    aoi_buffer_gdf.to_file(aoi_buffer_path, driver='GPKG')
    return aoi_buffer_path

def write_status(workspace_dir, huc4, satus='done', error=None):
    completed_list = os.path.join(workspace_dir, "processing_done.txt")
    err_list = os.path.join(workspace_dir, "processing_error.txt")
    try:
        os.remove(completed_list)
        os.remove(err_list)
    except OSError:
        pass
    if satus=='done':
        with open(completed_list, "w") as f:
            f.write(f"{huc4}")
    else:
        print(f"Error processing HUC4 {huc4}: {error}")
        with open(err_list, "w") as f:
            f.write(f"{huc4}")

def keep_mv_files(es_dict, model_args):
    int_folder = os.path.join(model_args['workspace_dir'], 'intermediate_outputs')
    file_list = []
    if es_dict['ndr']:
        file_list += [
            'load_n.tif', 'load_p.tif', 'effective_retention_n.tif', 'effective_retention_p.tif', 'ic_factor.tif', 'runoff_proxy_index.tif',
        ]
    if es_dict['sdr']:
        file_list += [
            'cp.tif', 'sdr_factor.tif'
        ]
    if es_dict['polination']:
        file_list += [
            'n_export.tif', 'n_retention.tif', 'n_retention_index.tif', # does not have a list yet
        ]
    if es_dict['swy']:
        file_list += [
            'aet.tif', 'Si.tif'
        ] + [f"qf_{i}.tif" for i in range(1, 13)]
    
    file_list = [f"{str(file).split('.')[0]}_{model_args['results_suffix']}.{str(file).split('.')[1]}" for file in file_list]
    all_files = [file for file in os.listdir(int_folder) if os.path.isfile(os.path.join(int_folder, file))]
    for file in all_files:
        if file not in file_list:
            try:
                os.remove(os.path.join(int_folder, file))
            except:
                pass
    
def delete_input_files(watershed_dir):
    all_files = [file for file in os.listdir(watershed_dir) if os.path.isfile(os.path.join(watershed_dir, file))]
    for file in all_files:
        try:
            os.remove(os.path.join(watershed_dir, file))
        except:
            pass

def fix_watersheds(watersheds_path, aoi_path, aoi_label, unique_id_field, projected_crs, output_folder=None):
    '''
    This function fixes the watersheds shapefile by:
     1) clipping the watershed file to aoi and then
     2) exploding multipart polygons and keeping the largest area polygon for each unique id.
    TODO: The fix might delete relative large polygons if they are not the largest in the multipart polygon. Need to discuss this.
    Do we implement a filter based on area?
    '''
    # watersheds_path = "C:/Users/salmamun/Files/base_data/seals/es_common_inputs/watersheds_level_7_valid.gpkg"
    # aoi_path = "C:/Users/salmamun/Files/seals/projects/Tanzania/intermediate/project_aoi_gtap_r251/aoi_buffer/aoi_TZA_32736.gpkg"
    # aoi_path = "C:/Users/salmamun/Files/seals/projects/Tanzania/intermediate/project_aoi_gtap_r251/aoi_TZA.gpkg"
    # aoi_label = "TZA"
    # unique_id_field = "SORT"
    # projected_crs = 32736
    # output_folder = 'C:\\Users\\salmamun\\Files\\seals\\projects\\Tanzania\\intermediate\\es_models\\sdr_results\\es_clipped_inputs'
    
    if projected_crs==54030:
        crs_string = "ESRI"
    else:
        crs_string = "EPSG"
    watersheds_gdf = gpd.read_file(watersheds_path)
        
    if output_folder==None:
        output_folder = os.path.dirname(watersheds_path)
    clipped_watershed = os.path.join(output_folder, os.path.basename(watersheds_path).replace(".gpkg", f"_{aoi_label}_clipped.gpkg"))
    if not os.path.exists(clipped_watershed):
        aoi_gdf = gpd.read_file(aoi_path)
        clipped_gdf = watersheds_gdf.clip(aoi_gdf, keep_geom_type=False)
        # Change clipped_gdf crs to projected_crs
        if clipped_gdf.crs==None:
            clipped_gdf.crs = "EPSG:4326"
        if clipped_gdf.crs != f"{crs_string}:{projected_crs}":
            clipped_gdf = clipped_gdf.to_crs(f"{crs_string}:{projected_crs}")
        clipped_gdf.to_file(clipped_watershed, driver='GPKG')
    else:
        clipped_gdf = gpd.read_file(clipped_watershed)

    multipart_watershed = os.path.join(output_folder, os.path.basename(watersheds_path).replace(".gpkg", f"_{aoi_label}_multipart.gpkg"))
    if not os.path.exists(multipart_watershed):
        multipart_gdf = clipped_gdf.explode(index_parts=True)
        multipart_gdf = multipart_gdf.loc[multipart_gdf.geometry.geometry.type=='Polygon']       # Only keep Polygons
        multipart_gdf["area"] = multipart_gdf['geometry'].area
        multipart_gdf.to_file(multipart_watershed, driver='GPKG')
    else:
        multipart_gdf = gpd.read_file(multipart_watershed)

    fixed_watersheds = os.path.join(output_folder, os.path.basename(watersheds_path).replace(".gpkg", f"_{aoi_label}_fixed.gpkg"))
    if not os.path.exists(fixed_watersheds):
        fixed_gdf = multipart_gdf.merge(multipart_gdf.groupby([unique_id_field])['area'].max().reset_index(), on=[unique_id_field, 'area'], how='right')
        fixed_gdf.to_file(fixed_watersheds, driver='GPKG')

def process_country(iso_year_tuple):
    try:
        aoi_label = iso_year_tuple[0]
        year = iso_year_tuple[1]
        print(f"Processing for {aoi_label} for the year {year}")
        country_dir = os.path.join(country_folders, f"{aoi_label}")
        es_clipped_input_folder = os.path.join(country_dir, "InputData", "es_clipped_input_folder")
        clipped_projected_raster_folder = os.path.join(country_dir, "InputData", "clipped_projected_raster_folder")
        vector_folder = os.path.join(country_dir, "InputData", "vectors")
        for fol in [country_dir, es_clipped_input_folder, clipped_projected_raster_folder, vector_folder]:
            if not os.path.exists(fol):
                os.makedirs(fol)

        # Create AOI and buffer
        aoi_path = os.path.join(vector_folder, f"aoi_{aoi_label}.gpkg")
        if not os.path.exists(aoi_path):  
            country_gpd = country_vector[country_vector['ee_r264_label'] == aoi_label]
            country_gpd.to_file(aoi_path)
        aoi_buffer_path = aoi_path.replace('.gpkg', f'_buffer.gpkg')
        if not os.path.exists(aoi_buffer_path):
            aoi_buffer_path = create_aoi_buffer(aoi_path, aoi_buffer_path, projected_crs, aoi_buffer_distance=5000)
        
        # Fix watersheds
        watersheds_path = args["watersheds_huc7"]
        fixed_watersheds_path = os.path.join(vector_folder, os.path.basename(watersheds_path).replace(".gpkg", f"_{aoi_label}_fixed.gpkg"))
        if not os.path.exists(fixed_watersheds_path):
            if os.path.exists(watersheds_path):
                fix_watersheds(watersheds_path=watersheds_path, aoi_path=aoi_path, aoi_label=aoi_label, unique_id_field="SORT", 
                            projected_crs = projected_crs, output_folder=vector_folder)

        # Slice raster inputs to the country boundary
        base_raster = args["dem"]
        source_raster_dict = {}
        source_raster_dict["dem"] = args["dem"]
        source_raster_dict[f"lulc_{year}"]= os.path.join(args["lulc_folder"], f"lulc_esa_{year}.tif")
        source_raster_dict[f"precip_{year}"] = os.path.join(args["precip_folder"], f"precip_annual_{year}.tif")
        
        force_es_data_preparation = False
        for k, v in source_raster_dict.items():
            single_dict = {k: v}
            base_raster = v
            unprojected_raster = os.path.join(es_clipped_input_folder, f'{k}.tif')
            if not os.path.exists(unprojected_raster) or force_es_data_preparation:
                if os.path.exists(base_raster):
                    slice_inputs(base_raster, single_dict, aoi_buffer_path, es_clipped_input_folder)
            
            projected_raster = os.path.join(clipped_projected_raster_folder, f'{k}_{projected_crs}.tif')
            if not os.path.exists(projected_raster) or force_es_data_preparation:
                if os.path.exists(unprojected_raster):
                    reproject_raster(unprojected_raster, projected_raster, dst_crs=f"EPSG:{projected_crs}")
        
        
        # Create NDR model args
        ndr_args = {
            # The following args are re-defined globally and linked to model-specific args here
            "workspace_dir": str(os.path.join(country_dir, "ndr", str(year))),
            "results_suffix": str(args["country_output_folder"]),
            "n_workers": args["n_workers"],
            "lulc_path": str(os.path.join(clipped_projected_raster_folder, f"lulc_{year}_{projected_crs}.tif")),
            "biophysical_table_path": str(args["ndr_" + "lulc_parameter_table_path"]),
            "dem_path": str(os.path.join(clipped_projected_raster_folder, f"dem_{projected_crs}.tif")),
            "watersheds_path": str(fixed_watersheds_path),
            "runoff_proxy_path": str(os.path.join(clipped_projected_raster_folder, f"precip_{year}_{projected_crs}.tif")),
            "threshold_flow_accumulation": args["threshold_flow_accumulation"],
            "k_param": args["k_param"],
            # The following args are model-specific
            "calc_n": args["calc_n"],
            "calc_p": args["calc_p"],
            "subsurface_critical_length_n": args["subsurface_critical_length_n"],
            "subsurface_eff_n": args["subsurface_eff_n"],
            }
        # print(ndr_args)
        natcap.invest.ndr.ndr.execute(ndr_args)
        # keep_mv_files(es_dict, ndr_args)
        print(f"=============================================={aoi_label} for {year}  NDR finished======================================")
        # DELETE WORKSPACE FILES
        # delete_input_files(watershed_dir)
        # WRITE A FILE TO INDICATE THAT THE PROCESSING IS DONE
        # write_status(watershed_args[WORKSPACE_FOLDER], huc4, satus='done', error=None)
    except Exception as e:
        print("did not worked")
        # write_status(watershed_args[WORKSPACE_FOLDER], huc4, satus='error', error=e)
        

def estimate_es(aoi_year_list, parallelized=True):
    if parallelized:
        with mp.Pool(n_cores) as pool:
            # execute tasks in order
            pool.map(process_country, aoi_year_list)
    else:
        for iso_year_tuple in aoi_year_list:
            process_country(iso_year_tuple)


if __name__ == '__main__':
    start = time.time()
    
    # if len(sys.argv) != 3:
    #     raise Exception("Usage: python preprocess.py [config file path]")
    
    # if not os.path.isfile(sys.argv[1]):
    #     raise Exception(f"Error: config file {sys.argv[1]} does not exist")
    
    iso_list = ['BGD', 'GMB']
    year_list = [2015, 2016]
    from itertools import product
    aoi_year_list = list(product(iso_list, year_list))
    parallelized = args["parallelized"]

    # listed = [str(i) for i in sys.argv[2][1:len(sys.argv[2])-1].split(',')]
    estimate_es(
        aoi_year_list=aoi_year_list, # ast.literal_eval(sys.argv[2]),
        parallelized=parallelized
    )
    print( f"Total time: {time.time() - start}")

    # srun -N 1 --ntasks-per-node=4  --mem-per-cpu=1gb -t 1:00:00 -p interactive --pty bash 
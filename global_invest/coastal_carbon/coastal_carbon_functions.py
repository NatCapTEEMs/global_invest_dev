# =============================================================================
# imports
# =============================================================================
import os
import re
import rasterio
from rasterio.mask import mask
from rasterio.enums import Resampling
from rasterio.merge import merge
import rasterio.warp
import rasterio.features
import rasterio.vrt
import numpy as np
from tqdm import tqdm
import gc
import rioxarray as rxr
import dask.array as da
import dask.dataframe as dd
import pandas as pd
import numpy as np
from tqdm import tqdm
import geopandas as gpd
import pyogrio
import glob
import os
os.environ["CHECK_DISK_FREE_SPACE"] = "FALSE"
# =============================================================================
# define functions
# =============================================================================




import os
import glob
import rasterio
from rasterio.merge import merge
import re
from tqdm import tqdm


# Function to extract longitude from filename
def extract_longitude(filename):
    """
    Extract longitude coordinate from filename.
    Handles patterns like:
    - GWL_FCS30D_2019Maps_E0N10.tif -> E0
    - GWL_FCS30D_2019Maps_E10N20.tif -> E10
    - GWL_FCS30D_2019Maps_W5N15.tif -> W5
    """
    # Pattern to match longitude coordinates (E or W followed by digits)
    pattern = r'([EW]\d+)'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)  # Returns E0, E10, W5, etc.
    return None

def group_tif_files_by_longitude(input_folder):
    """
    Group TIFF files by their longitude coordinate.
    """
    tif_files = glob.glob(os.path.join(input_folder, "*.tif"))
    tif_files.extend(glob.glob(os.path.join(input_folder, "*.tiff")))

    if not tif_files:
        print("No TIFF files found!")
        return {}

    # Group files by longitude
    file_groups = {}
    for file_path in tif_files:
        filename = os.path.basename(file_path)
        lon_group = extract_longitude(filename)
        
        if lon_group is None:
            print(f"Warning: Could not extract longitude from {filename}, skipping")
            continue
            
        if lon_group not in file_groups:
            file_groups[lon_group] = []
        file_groups[lon_group].append(file_path)
    
    # Sort files within each group by latitude for consistent merging
    for lon_group in file_groups:
        file_groups[lon_group].sort()
    
    return file_groups

def merge_tif_files_by_longitude(input_folder, output_folder):
    """
    Group TIFF files by longitude and merge each group separately.
    For example, files with E0N10 and E0N15 will be merged into one file for longitude E0.
    """
    # Group files by longitude
    file_groups = group_tif_files_by_longitude(input_folder)
    
    if not file_groups:
        print("No valid TIFF files with coordinate patterns found!")
        return
    
    print(f"Found {len(file_groups)} longitude groups: {list(file_groups.keys())}")
    
    # Process each group separately
    for lon_group, file_list in file_groups.items():
        print(f"\nProcessing longitude {lon_group} with {len(file_list)} files:")
        for file in file_list:
            print(f"  - {os.path.basename(file)}")
        
        # Generate output filename
        first_filename = os.path.basename(file_list[0])
        base_pattern = re.sub(r'_[EW]\d+[NS]\d+', '', first_filename)  # Remove coordinate part
        base_pattern = re.sub(r'\.tiff?$', '', base_pattern)  # Remove extension
        
        output_filename = f"{base_pattern}_{lon_group}.tif"
        output_path = os.path.join(output_folder, output_filename)
        
        if os.path.exists(output_path):
            print(f"{output_filename} already exists")
            continue

        # Merge files in this longitude group
        merge_longitude_group(file_list, output_path, lon_group)

def merge_longitude_group(file_list, output_path, longitude):
    """
    Merge a group of TIFF files that share the same longitude.
    This will create north-south strips for each longitude band.
    """
    if len(file_list) == 1:
        print(f"Only one file for longitude {longitude}, copying...")
        try:
            with rasterio.open(file_list[0]) as src:
                data = src.read()
                meta = src.meta.copy()
            
            with rasterio.open(output_path, "w", **meta) as dest:
                dest.write(data)
            print(f"Copied single file to: {output_path}")
        except Exception as e:
            print(f"Error copying single file: {e}")
        return

    print(f"Merging {len(file_list)} files for longitude {longitude}...")

    # Open all datasets for this longitude
    datasets = []
    try:
        for file_path in file_list:
            src = rasterio.open(file_path)
            datasets.append(src)
        
        # Get metadata from first dataset
        first_meta = datasets[0].meta.copy()
        nodata = first_meta.get('nodata', None)
        
        # Merge all datasets for this longitude
        mosaic, out_transform = merge(
            datasets,
            method='first',
            nodata=nodata
        )
        
        # Update metadata
        first_meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
            "compress": "lzw"
        })
        
        # Write output
        print(f"Writing output for longitude {longitude}...")
        with rasterio.open(output_path, "w", **first_meta) as dest:
            if nodata is not None:
                dest.fill(nodata)
            
            for i in tqdm(range(mosaic.shape[0]), desc="Writing bands"):
                dest.write(mosaic[i], i + 1)
        
        print(f"Successfully merged {len(file_list)} files to: {output_path}")
        print(f"Final output size: {mosaic.shape}")
        
    except Exception as e:
        print(f"Error processing longitude {longitude}: {e}")
    
    finally:
        # Ensure all datasets are closed
        for src in datasets:
            if not src.closed:
                src.close()




if __name__ == "__main__":
    input_folder = "/Users/long/Library/CloudStorage/GoogleDrive-yxlong@umn.edu/Shared drives/NatCapTEEMs/Files/base_data/submissions/coastal_carbon/GWL_FCS30D_2019"
    output_folder = "/Users/long/Library/CloudStorage/GoogleDrive-yxlong@umn.edu/Shared drives/NatCapTEEMs/Files/base_data/submissions/coastal_carbon/GWL_2019"
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Merge files by longitude
    merge_tif_files_by_longitude(input_folder, output_folder)

#%%


import rasterio
from rasterio.transform import from_bounds
import numpy as np

def create_blank_global_canvas(output_path, resolution=0.008333333333333333, dtype='uint16', nodata_val=255):
    """Creates a blank raster file covering the full globe."""
    
    # Standard WGS 84 Bounds (Full Extent)
    bounds = [-180.0, -90.0, 180.0, 90.0] 
    
    # Calculate transform (top-left corner coordinates)
    transform = from_bounds(*bounds, width=360 / resolution, height=180 / resolution)
    
    # Calculate Height and Width (number of rows and columns)
    height = int(180 / resolution)
    width = int(360 / resolution)

    # Define metadata
    profile = {
        'driver': 'GTiff',
        'dtype': dtype,
        'count': 1,  # Single band for LULC
        'crs': 'EPSG:4326', # WGS 84 standard
        'transform': transform,
        'width': width,
        'height': height,
        'nodata': nodata_val,
        'compress': 'lzw'
    }
    
    # Initialize a blank NumPy array filled with the nodata value
    blank_array = np.full((1, height, width), nodata_val, dtype=dtype)
    
    # Write the file
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(blank_array)
        
    print(f"✅ Created blank global map: {output_path} with shape ({height}, {width})")

# Example: create_blank_global_canvas("blank_global_lulc_canvas.tif", resolution=0.008333)

output_file_path = "/Users/long/Library/CloudStorage/GoogleDrive-yxlong@umn.edu/Shared drives/NatCapTEEMs/Files/base_data/submissions/coastal_carbon/global_map.tif"

create_blank_global_canvas(output_file_path, resolution=0.008333333333333333, dtype='uint16', nodata_val=255)





import rasterio
import numpy as np
import os
import glob
from tqdm import tqdm
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds

def insert_multiple_lulc_tiles(global_map_path, lulc_input_folder, output_path):
    """
    Inserts pixel values from all LULC tiles in a folder into the corresponding 
    area of a global map array (the base mosaic), handling necessary reprojection 
    and ensuring integer data type compatibility for categorical LULC data.

    Args:
        global_map_path (str): Path to the base global raster file (the canvas).
        lulc_input_folder (str): Folder containing all the LULC tile files.
        output_path (str): Path to save the updated global raster.
    """
    
    # 1. Get the list of LULC files
    lulc_files = glob.glob(os.path.join(lulc_input_folder, "*.tif"))
    lulc_files.extend(glob.glob(os.path.join(lulc_input_folder, "*.tiff")))
    lulc_files.sort()

    if not lulc_files:
        print("❌ No LULC TIFF files found in the specified folder!")
        return
        
    print(f"Found {len(lulc_files)} LULC tiles to process.")

    # 2. Open the global map and initialize the mutable mosaic
    try:
        with rasterio.open(global_map_path) as global_src:
            global_profile = global_src.profile
            
            # Read the entire global data into a mutable array for in-place updates
            mosaic_array = global_src.read() 
            
            # Check for multi-band, assuming insertion happens in Band 1 (index 0)
            if mosaic_array.shape[0] > 1:
                 print("⚠️ Global map has multiple bands. Insertion will only happen in the first band (index 0).")
            
    except Exception as e:
        print(f"❌ Error during global raster initialization: {e}")
        return

    # 3. Loop through all LULC tiles and insert
    for i, lulc_file in enumerate(tqdm(lulc_files, desc="Inserting LULC Tiles")):
        
        try:
            with rasterio.open(lulc_file) as lulc_src:
                lulc_data = lulc_src.read()
                
                if lulc_data.shape[0] != 1:
                    print(f"\n❌ Skipping {os.path.basename(lulc_file)}: Must be a single band.")
                    continue
                
                lulc_array = lulc_data[0] # The 2D array of values
                
                # Get the necessary metadata
                lulc_bounds = lulc_src.bounds
                lulc_transform = lulc_src.transform
                lulc_crs = lulc_src.crs
                
                # Get LULC nodata and explicitly cast to an integer (safe default of 0)
                lulc_nodata = int(lulc_src.nodata) if lulc_src.nodata is not None else 0

            # Calculate the window (pixel indices) on the global grid
            window = from_bounds(
                lulc_bounds.left, lulc_bounds.bottom, lulc_bounds.right, lulc_bounds.top, 
                global_profile['transform']
            )
            
            # Extract array indices and target dimensions
            row_start, row_end = window.row_off, window.row_off + window.height
            col_start, col_end = window.col_off, window.col_off + window.width
            target_height = row_end - row_start
            target_width = col_end - col_start
            
            # 4. Resample LULC array if shapes don't match (fixing misalignment/resolution)
            if lulc_array.shape != (target_height, target_width):
                
                # CRITICAL FIX: Explicitly set the destination dtype to a safe integer type (np.int16)
                resampled_lulc = np.empty((target_height, target_width), dtype=np.int16) 
                
                out_transform = global_src.window_transform(window)
                
                reproject(
                    source=lulc_data, 
                    destination=resampled_lulc,
                    src_transform=lulc_transform,
                    src_crs=lulc_crs,
                    dst_transform=out_transform,
                    dst_crs=global_profile['crs'],
                    resampling=Resampling.nearest, 
                    
                    # Ensure nodata values are integers
                    src_nodata=lulc_nodata, 
                    dst_nodata=lulc_nodata
                )
                lulc_array = resampled_lulc
            
            # 5. Insert the LULC data into the global map array (Overwriting Band 1)
            # This works because lulc_array (resampled or not) is now the correct shape
            # and an integer type, ready to be assigned to the mosaic array.
            mosaic_array[0, row_start:row_end, col_start:col_end] = lulc_array

        except Exception as e:
            print(f"\n❌ An error occurred processing {os.path.basename(lulc_file)}: {e}")
            continue

    # 6. Write the final output
    out_meta = global_profile.copy()
    out_meta.update({"compress": "lzw"})
    
    print("\nWriting final output mosaic...")
    try:
        # Re-open global_src for the profile, but write the mosaic_array
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic_array)

        print(f"✅ Successfully created final mosaic and saved to: {output_path}")

    except Exception as e:
        print(f"❌ Error writing output file: {e}")




global_map_path = "/Users/long/Library/CloudStorage/GoogleDrive-yxlong@umn.edu/Shared drives/NatCapTEEMs/Files/base_data/submissions/coastal_carbon/global_map.tif"
lulc_input_folder =  "/Users/long/Library/CloudStorage/GoogleDrive-yxlong@umn.edu/Shared drives/NatCapTEEMs/Files/base_data/submissions/coastal_carbon/GWL_2019_latitude"
#lulc_input_folder =  "/Users/long/Library/CloudStorage/GoogleDrive-yxlong@umn.edu/Shared drives/NatCapTEEMs/Files/base_data/submissions/coastal_carbon/GWL_FCS30D_2019"
output_path = "/Users/long/Library/CloudStorage/GoogleDrive-yxlong@umn.edu/Shared drives/NatCapTEEMs/Files/base_data/submissions/coastal_carbon/GWL_2019.tif"
insert_multiple_lulc_tiles(global_map_path, lulc_input_folder, output_path)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import os
import glob
import rasterio
from rasterio.merge import merge
import re
from tqdm import tqdm

def extract_latitude(filename):
    """
    Extract latitude coordinate from filename.
    Handles patterns like:
    - GWL_FCS30D_2019Maps_E0N10.tif -> N10
    - GWL_FCS30D_2019Maps_E10S20.tif -> S20
    - GWL_FCS30D_2019Maps_W5N15.tif -> N15
    """
    # Pattern to match latitude coordinates (N or S followed by digits)
    pattern = r'[EW]\d+([NS]\d+)'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)  # Returns N10, S20, N15, etc.
    return None

def group_tif_files_by_latitude(input_folder):
    """
    Group TIFF files by their latitude coordinate.
    """
    tif_files = glob.glob(os.path.join(input_folder, "*.tif"))
    tif_files.extend(glob.glob(os.path.join(input_folder, "*.tiff")))

    if not tif_files:
        print("No TIFF files found!")
        return {}

    # Group files by latitude
    file_groups = {}
    for file_path in tif_files:
        filename = os.path.basename(file_path)
        lat_group = extract_latitude(filename)
        
        if lat_group is None:
            print(f"Warning: Could not extract latitude from {filename}, skipping")
            continue
            
        if lat_group not in file_groups:
            file_groups[lat_group] = []
        file_groups[lat_group].append(file_path)
    
    # Sort files within each group by longitude for consistent merging
    for lat_group in file_groups:
        file_groups[lat_group].sort()
    
    return file_groups

def merge_tif_files_by_latitude(input_folder, output_folder):
    """
    Group TIFF files by latitude and merge each group separately.
    For example, files with E0N10 and W10N10 will be merged into one file for latitude N10.
    """
    # Group files by latitude
    file_groups = group_tif_files_by_latitude(input_folder)
    
    if not file_groups:
        print("No valid TIFF files with coordinate patterns found!")
        return
    
    print(f"Found {len(file_groups)} latitude groups: {list(file_groups.keys())}")
    
    # Process each group separately
    for lat_group, file_list in file_groups.items():
        print(f"\nProcessing latitude {lat_group} with {len(file_list)} files:")
        for file in file_list:
            print(f"  - {os.path.basename(file)}")
        
        # Generate output filename
        first_filename = os.path.basename(file_list[0])
        base_pattern = re.sub(r'_[EW]\d+[NS]\d+', '', first_filename)  # Remove coordinate part
        base_pattern = re.sub(r'\.tiff?$', '', base_pattern)  # Remove extension
        
        output_filename = f"{base_pattern}_{lat_group}.tif"
        output_path = os.path.join(output_folder, output_filename)
        
        if os.path.exists(output_path):
            print(f"{output_filename} already exists")
            continue

        # Merge files in this latitude group
        merge_latitude_group(file_list, output_path, lat_group)

def merge_latitude_group(file_list, output_path, latitude):
    """
    Merge a group of TIFF files that share the same latitude.
    This will create east-west strips for each latitude band.
    """
    if len(file_list) == 1:
        print(f"Only one file for latitude {latitude}, copying...")
        try:
            with rasterio.open(file_list[0]) as src:
                data = src.read()
                meta = src.meta.copy()
            
            with rasterio.open(output_path, "w", **meta) as dest:
                dest.write(data)
            print(f"Copied single file to: {output_path}")
        except Exception as e:
            print(f"Error copying single file: {e}")
        return

    print(f"Merging {len(file_list)} files for latitude {latitude}...")

    # Open all datasets for this latitude
    datasets = []
    try:
        for file_path in file_list:
            src = rasterio.open(file_path)
            datasets.append(src)
        
        # Get metadata from first dataset
        first_meta = datasets[0].meta.copy()
        nodata = first_meta.get('nodata', None)
        
        # Merge all datasets for this latitude
        mosaic, out_transform = merge(
            datasets,
            method='first',
            nodata=nodata
        )
        
        # Update metadata
        first_meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
            "compress": "lzw"
        })
        
        # Write output
        print(f"Writing output for latitude {latitude}...")
        with rasterio.open(output_path, "w", **first_meta) as dest:
            if nodata is not None:
                dest.fill(nodata)
            
            for i in tqdm(range(mosaic.shape[0]), desc="Writing bands"):
                dest.write(mosaic[i], i + 1)
        
        print(f"Successfully merged {len(file_list)} files to: {output_path}")
        print(f"Final output size: {mosaic.shape}")
        
    except Exception as e:
        print(f"Error processing latitude {latitude}: {e}")
    
    finally:
        # Ensure all datasets are closed
        for src in datasets:
            if not src.closed:
                src.close()

# Usage
if __name__ == "__main__":
    input_folder = "/Users/long/Library/CloudStorage/GoogleDrive-yxlong@umn.edu/Shared drives/NatCapTEEMs/Files/base_data/submissions/coastal_carbon/GWL_FCS30D_2019"
    output_folder = "/Users/long/Library/CloudStorage/GoogleDrive-yxlong@umn.edu/Shared drives/NatCapTEEMs/Files/base_data/submissions/coastal_carbon/GWL_2019_latitude"
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Merge files by latitude
    merge_tif_files_by_latitude(input_folder, output_folder)



#%%


import rasterio
from rasterio.merge import merge
import os
import glob
from tqdm import tqdm
import gc

def merge_all_tifs_into_one(input_folder, output_file_path):
    """
    Simply merge all TIFF files in input folder into a single file.
    """
    # Get all TIFF files
    tif_files = glob.glob(os.path.join(input_folder, "*.tif"))
    tif_files.extend(glob.glob(os.path.join(input_folder, "*.tiff")))
    
    if not tif_files:
        print("No TIFF files found!")
        return
    
    print(f"Found {len(tif_files)} TIFF files to merge into one")
    
    # Check if output already exists
    if os.path.exists(output_file_path):
        print(f"✓ Merged file already exists: {output_file_path}")
        return
    
    print("Merging all files into one...")
    
    # Use rasterio environment context manager
    with rasterio.Env() as env:
        datasets = []
        try:
            # Open all datasets
            for file_path in tqdm(tif_files, desc="Opening files"):
                ds = rasterio.open(file_path)
                datasets.append(ds)
            
            # Get metadata from first dataset
            meta = datasets[0].meta.copy()
            nodata = meta.get('nodata', None)
            
            # Merge all files
            mosaic, transform = merge(datasets, method='first', nodata=nodata)
            
            # Update metadata
            meta.update({
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "compress": "lzw"
            })
            
            # Write output
            with rasterio.open(output_file_path, "w", **meta) as dest:
                if nodata is not None:
                    dest.fill(nodata)
                
                for i in tqdm(range(mosaic.shape[0]), desc="Writing bands"):
                    dest.write(mosaic[i], i + 1)
            
            print(f"✓ Successfully created merged file: {output_file_path}")
            print(f"Final size: {mosaic.shape}")
            
        except Exception as e:
            print(f"✗ Error merging files: {e}")
            # Remove partial output file if error occurred
            if os.path.exists(output_file_path):
                try:
                    os.remove(output_file_path)
                    print("Removed partial output file")
                except:
                    pass
        
        finally:
            # Close all datasets explicitly
            print("Closing datasets...")
            for i, ds in enumerate(datasets):
                try:
                    if not ds.closed:
                        ds.close()
                except Exception as e:
                    print(f"Warning: Could not close dataset {i}: {e}")
            
            # Clear the list to help garbage collection
            datasets.clear()
            
            # Force garbage collection
            gc.collect()

input_folder = "/Users/long/Library/CloudStorage/GoogleDrive-yxlong@umn.edu/Shared drives/NatCapTEEMs/Files/base_data/submissions/coastal_carbon/GWL_2019_latitude"
    
output_file_path = "/Users/long/Library/CloudStorage/GoogleDrive-yxlong@umn.edu/Shared drives/NatCapTEEMs/Files/base_data/submissions/coastal_carbon/GWL_2019.tif"
    
merge_all_tifs_into_one(input_folder, output_file_path)



def merge_all_tifs_into_one_safe(input_folder, output_file_path):
    """
    Simply merge all TIFF files in input folder into a single file.
    Uses context managers for all file operations.
    """
    # Get all TIFF files
    tif_files = glob.glob(os.path.join(input_folder, "*.tif"))
    tif_files.extend(glob.glob(os.path.join(input_folder, "*.tiff")))
    
    if not tif_files:
        print("No TIFF files found!")
        return
    
    print(f"Found {len(tif_files)} TIFF files to merge into one")
    
    # Check if output already exists
    if os.path.exists(output_file_path):
        print(f"✓ Merged file already exists: {output_file_path}")
        return
    
    print("Merging all files into one...")
    
    # Use context managers for all dataset operations
    try:
        # Open all datasets using context managers in a list comprehension
        with contextlib.ExitStack() as stack:
            datasets = [stack.enter_context(rasterio.open(file_path)) 
                       for file_path in tif_files]
            
            # Get metadata from first dataset
            meta = datasets[0].meta.copy()
            nodata = meta.get('nodata', None)
            
            # Merge all files
            mosaic, transform = merge(datasets, method='first', nodata=nodata)
            
            # Update metadata
            meta.update({
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "compress": "lzw"
            })
            
            # Write output
            with rasterio.open(output_file_path, "w", **meta) as dest:
                if nodata is not None:
                    dest.fill(nodata)
                
                for i in tqdm(range(mosaic.shape[0]), desc="Writing bands"):
                    dest.write(mosaic[i], i + 1)
        
        print(f"✓ Successfully created merged file: {output_file_path}")
        print(f"Final size: {mosaic.shape}")
        
    except Exception as e:
        print(f"✗ Error merging files: {e}")
        # Remove partial output file if error occurred
        if os.path.exists(output_file_path):
            try:
                os.remove(output_file_path)
                print("Removed partial output file")
            except:
                pass


#%%
import rasterio
import numpy as np
import glob
import os
from tqdm import tqdm
from rasterio.warp import reproject, Resampling

def filter_global_raster_with_multiple_lulc(global_raster_path, lulc_input_folder, output_path, target_code=186):
    """
    Filters a global raster using multiple LULC maps (e.g., split by latitude).
    It explicitly warps each LULC tile to the global raster's grid before masking, 
    fixing the shape and dtype mismatch errors.

    Args:
        global_raster_path (str): Path to the primary global raster map.
        lulc_input_folder (str): Folder containing all the LULC raster files.
        output_path (str): Path to save the resulting filtered raster file.
        target_code (int): The LULC code to filter by (default is 186).
    """
    
    # 1. Setup and Initialization
    lulc_files = glob.glob(os.path.join(lulc_input_folder, "*.tif"))
    lulc_files.extend(glob.glob(os.path.join(lulc_input_folder, "*.tiff")))
    lulc_files.sort()

    if not lulc_files:
        print("❌ No LULC TIFF files found in the specified folder!")
        return
        
    print(f"Found {len(lulc_files)} LULC files to process.")
    
    # Open global raster ONCE and keep the handler (src) open
    try:
        global_src = rasterio.open(global_raster_path)
        global_profile = global_src.profile
        
        # Get nodata value and set a sensible default if missing
        nodata_val = global_src.nodata if global_src.nodata is not None else 0
        global_profile.update({"nodata": nodata_val, "compress": "lzw"})
        
        # Initialize the output array with nodata values (memory efficient)
        filtered_data = np.full(
            (global_profile['count'], global_profile['height'], global_profile['width']), 
            nodata_val,
            dtype=global_profile['dtype']
        )
        
    except Exception as e:
        print(f"❌ Error during global raster initialization: {e}")
        if 'global_src' in locals() and not global_src.closed:
            global_src.close()
        return
    
    # 2. Iteratively Apply Masks
    for i, lulc_file in enumerate(tqdm(lulc_files[:2], desc="Applying LULC Masks")):
        
        try:
            with rasterio.open(lulc_file) as lulc_src:
                lulc_bounds = lulc_src.bounds
                lulc_transform = lulc_src.transform
                lulc_crs = lulc_src.crs
                
                # FIX: Access the dtype using lulc_src.profile['dtype'] for the destination array
                lulc_dtype = lulc_src.profile['dtype'] 
                
                # Check for misalignment
                if global_src.crs != lulc_src.crs:
                    print(f"\n⚠️ Warning: CRS mismatch for {os.path.basename(lulc_file)}. Reprojection is necessary.")
                
                # Calculate the target geometry based on the global grid
                lulc_window = global_src.window(*lulc_bounds)
                out_transform = global_src.window_transform(lulc_window)
                
                out_height = lulc_window.height
                out_width = lulc_window.width

                # Initialize a temporary array for the warped LULC data
                # Using the correctly retrieved lulc_dtype
                warped_lulc_data = np.empty((1, out_height, out_width), dtype=lulc_dtype)
                
                # WARP/REPROJECT the LULC data to match the global slice's grid
                reproject(
                    source=rasterio.band(lulc_src, 1), 
                    destination=warped_lulc_data,
                    src_transform=lulc_transform,
                    src_crs=lulc_crs,
                    dst_transform=out_transform,
                    dst_crs=global_src.crs,
                    resampling=Resampling.nearest, # Use nearest for classification data
                    num_threads=4 
                )
            
            # 3. Create the Mask and Read the Global Slice
            current_mask = (warped_lulc_data[0] == target_code)
            global_slice = global_src.read(window=lulc_window)
            
            # 4. Update the Filtered Data in place
            row_start, row_end = lulc_window.row_off, lulc_window.row_off + lulc_window.height
            col_start, col_end = lulc_window.col_off, lulc_window.col_off + lulc_window.width

            for band_idx in range(global_slice.shape[0]):
                global_band_slice = global_slice[band_idx]
                
                # Apply the mask: True (186) keeps global data, False sets nodata
                filtered_slice = np.where(current_mask, global_band_slice, nodata_val)
                
                # Write the result back to the correct location in the master array
                filtered_data[band_idx, row_start:row_end, col_start:col_end] = filtered_slice
        
        except Exception as e:
            # Catch the error and continue
            print(f"\n❌ An error occurred during processing {os.path.basename(lulc_file)}: {e}")
            continue
            
    # Close the global source file
    global_src.close()
    
    # 3. Finalize and Write Output
    print("\nWriting final filtered mosaic...")
    try:
        with rasterio.open(output_path, "w", **global_profile) as dest:
            dest.write(filtered_data)
        
        print(f"✅ Successfully created filtered global map for code {target_code} at: {output_path}")

    except Exception as e:
        print(f"❌ Error writing output file: {e}")

# --- Example Usage ---
# NOTE: Replace these placeholder paths with your actual file paths
global_map_file = "/Users/long/Library/CloudStorage/GoogleDrive-yxlong@umn.edu/Shared drives/NatCapTEEMs/Files/base_data/pyramids/ha_per_cell_10sec.tif"
lulc_tiles_folder = "/Users/long/Library/CloudStorage/GoogleDrive-yxlong@umn.edu/Shared drives/NatCapTEEMs/Files/base_data/submissions/coastal_carbon/GWL_2019_latitude"
output_result_file = "/Users/long/Library/CloudStorage/GoogleDrive-yxlong@umn.edu/Shared drives/NatCapTEEMs/Files/base_data/submissions/coastal_carbon/filtered_global_map_code_186.tif"

# Execute the function
filter_global_raster_with_multiple_lulc(
    global_map_file, 
    lulc_tiles_folder, 
    output_result_file, 
    target_code=186
)


#%%

import os
import glob
import rasterio
from rasterio.merge import merge
import re
from tqdm import tqdm
import numpy as np
import gc

def extract_latitude_value(filename):
    """
    Extract numeric latitude value from filename for proper sorting.
    """
    n_pattern = r'N(\d+)'
    s_pattern = r'S(\d+)'
    
    match = re.search(n_pattern, filename)
    if match:
        return int(match.group(1))
    
    match = re.search(s_pattern, filename)
    if match:
        return -int(match.group(1))
    
    return 0

def get_latitude_files_sorted(input_folder):
    """
    Get all latitude files and sort them properly from South to North.
    """
    lat_files = glob.glob(os.path.join(input_folder, "*_N*.tif"))
    lat_files.extend(glob.glob(os.path.join(input_folder, "*_S*.tif")))
    lat_files.extend(glob.glob(os.path.join(input_folder, "*.tiff")))
    
    if not lat_files:
        return []
    
    valid_files = []
    for file_path in lat_files:
        filename = os.path.basename(file_path)
        lat_value = extract_latitude_value(filename)
        if lat_value != 0:
            valid_files.append((lat_value, file_path))
    
    valid_files.sort(key=lambda x: x[0])
    return [file_path for _, file_path in valid_files]

def merge_global_latitude_safe(input_folder, output_file_path, batch_size=2):
    """
    Safe version that merges latitude files in small batches to avoid memory issues.
    """
    lat_files = get_latitude_files_sorted(input_folder)
    
    if not lat_files:
        print("No valid latitude files found!")
        return
    
    print(f"Found {len(lat_files)} latitude files")
    
    if os.path.exists(output_file_path):
        print(f"✓ Global map already exists: {output_file_path}")
        return
    
    # Show file list
    print("Files to process (South to North):")
    for i, file_path in enumerate(lat_files):
        filename = os.path.basename(file_path)
        lat_value = extract_latitude_value(filename)
        print(f"  {i+1:2d}. {filename} (lat: {lat_value:4d}°)")
    
    # If few files, merge directly
    if len(lat_files) <= batch_size:
        print("Merging files directly...")
        return merge_files_directly(lat_files, output_file_path)
    
    print(f"Merging {len(lat_files)} files in batches of {batch_size}...")
    
    temp_files = []
    temp_dir = os.path.dirname(output_file_path)
    
    try:
        # Process in small batches
        for i in range(0, len(lat_files), batch_size):
            batch = lat_files[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(lat_files) - 1) // batch_size + 1
            
            temp_file = os.path.join(temp_dir, f"temp_batch_{batch_num:03d}.tif")
            temp_files.append(temp_file)
            
            print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch)} files)...")
            
            if os.path.exists(temp_file):
                print("  Batch file exists, skipping")
                continue
            
            success = merge_files_directly(batch, temp_file)
            if not success:
                print(f"  ✗ Failed to process batch {batch_num}")
                return False
            
            # Force garbage collection
            gc.collect()
        
        # Now merge all temporary batch files
        print(f"\nMerging {len(temp_files)} batch files into final global map...")
        success = merge_files_directly(temp_files, output_file_path)
        
        if success:
            print(f"✓ Successfully created global map: {output_file_path}")
            
            # Clean up temporary files
            print("Cleaning up temporary files...")
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"  Removed: {os.path.basename(temp_file)}")
        
        return success
        
    except Exception as e:
        print(f"✗ Error during batch processing: {e}")
        return False

def merge_files_directly(file_list, output_path):
    """
    Merge a list of files directly with memory management.
    """
    if len(file_list) == 1:
        # Just copy the single file
        try:
            print(f"  Copying single file...")
            with rasterio.open(file_list[0]) as src:
                data = src.read()
                meta = src.meta.copy()
            
            with rasterio.open(output_path, "w", **meta) as dest:
                dest.write(data)
            print(f"  ✓ Copied: {os.path.basename(file_list[0])}")
            return True
        except Exception as e:
            print(f"  ✗ Error copying file: {e}")
            return False
    
    datasets = []
    try:
        # Open all datasets
        print(f"  Opening {len(file_list)} files...")
        for file_path in file_list:
            try:
                datasets.append(rasterio.open(file_path))
            except Exception as e:
                print(f"  ✗ Error opening {os.path.basename(file_path)}: {e}")
                continue
        
        if not datasets:
            print("  ✗ No valid datasets to merge")
            return False
        
        # Get metadata
        meta = datasets[0].meta.copy()
        nodata = meta.get('nodata', None)
        
        # Merge files
        print("  Merging files...")
        mosaic, transform = merge(
            datasets,
            method='first',
            nodata=nodata
        )
        
        # Update metadata
        meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform,
            "compress": "lzw",
            "nodata": nodata
        })
        
        # Write output
        print("  Writing output...")
        with rasterio.open(output_path, "w", **meta) as dest:
            if nodata is not None:
                dest.fill(nodata)
            
            for i in tqdm(range(mosaic.shape[0]), desc="    Writing bands", leave=False):
                dest.write(mosaic[i], i + 1)
        
        print(f"  ✓ Created: {os.path.basename(output_path)}")
        print(f"    Size: {mosaic.shape}")
        
        # Calculate coverage
        if nodata is not None:
            total_pixels = mosaic[0].size
            valid_pixels = np.sum(mosaic[0] != nodata)
            coverage = (valid_pixels / total_pixels) * 100
            print(f"    Coverage: {coverage:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error merging files: {e}")
        return False
    
    finally:
        # Close all datasets and clean up memory
        for ds in datasets:
            if not ds.closed:
                ds.close()
        datasets.clear()
        gc.collect()

def create_global_map_step_by_step(input_folder, output_file_path):
    """
    Alternative approach: Build global map step by step, merging one file at a time.
    This is the most memory-efficient method.
    """
    lat_files = get_latitude_files_sorted(input_folder)
    
    if not lat_files:
        print("No valid latitude files found!")
        return
    
    print(f"Found {len(lat_files)} latitude files")
    
    if os.path.exists(output_file_path):
        print(f"✓ Global map already exists: {output_file_path}")
        return
    
    # Start with first file
    print("Building global map step by step...")
    current_temp = os.path.join(os.path.dirname(output_file_path), "temp_current.tif")
    
    try:
        # Copy first file as starting point
        print(f"Step 1/{(len(lat_files)*2)-1}: Starting with {os.path.basename(lat_files[0])}")
        with rasterio.open(lat_files[0]) as src:
            data = src.read()
            meta = src.meta.copy()
        
        with rasterio.open(current_temp, "w", **meta) as dest:
            dest.write(data)
        
        # Merge remaining files one by one
        for i, file_path in enumerate(lat_files[1:], 2):
            print(f"Step {i}/{(len(lat_files)*2)-1}: Merging {os.path.basename(file_path)}")
            
            next_temp = os.path.join(os.path.dirname(output_file_path), f"temp_next_{i}.tif")
            
            datasets = []
            try:
                datasets.append(rasterio.open(current_temp))
                datasets.append(rasterio.open(file_path))
                
                mosaic, transform = merge(datasets, method='first')
                
                meta = datasets[0].meta.copy()
                nodata = meta.get('nodata')
                meta.update({
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": transform,
                    "compress": "lzw",
                    "nodata": nodata
                })
                
                with rasterio.open(next_temp, "w", **meta) as dest:
                    if nodata is not None:
                        dest.fill(nodata)
                    for band_idx in range(mosaic.shape[0]):
                        dest.write(mosaic[band_idx], band_idx + 1)
                
                # Update current temp file
                if os.path.exists(current_temp):
                    os.remove(current_temp)
                current_temp = next_temp
                
                print(f"  ✓ Progress: {i-1}/{len(lat_files)} files merged")
                
            finally:
                for ds in datasets:
                    if not ds.closed:
                        ds.close()
                gc.collect()
        
        # Rename final temp file to output
        os.rename(current_temp, output_file_path)
        print(f"✓ Successfully created global map: {output_file_path}")
        
    except Exception as e:
        print(f"✗ Error during step-by-step merge: {e}")
        # Clean up temporary files
        for temp_file in glob.glob(os.path.join(os.path.dirname(output_file_path), "temp_*.tif")):
            if os.path.exists(temp_file):
                os.remove(temp_file)


# Usage
if __name__ == "__main__":
    input_folder = "/Users/long/Library/CloudStorage/GoogleDrive-yxlong@umn.edu/Shared drives/NatCapTEEMs/Files/base_data/submissions/coastal_carbon/GWL_2019_latitude"
    lulc_tiles_folder = "/Users/long/Library/CloudStorage/GoogleDrive-yxlong@umn.edu/Shared drives/NatCapTEEMs/Files/base_data/submissions/coastal_carbon/GWL_2019_latitude"
    output_file = "/Users/long/Library/CloudStorage/GoogleDrive-yxlong@umn.edu/Shared drives/NatCapTEEMs/Files/base_data/submissions/coastal_carbon/GWL_2019.tif"
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Method 1: Direct merge (for smaller datasets)
    merge_global_latitude_safe(input_folder, output_file)
    
    # Method 2: Safe batch merge (for larger datasets or if you get memory errors)
    # merge_global_latitude_safe(input_folder, output_file, batch_size=3)















#%%


from osgeo import gdal
import glob
import os

def merge_rasters_direct(input_folder, output_path):
    # Find all TIFF files
    tif_files = glob.glob(os.path.join(input_folder, "*.tif"))
    
    # Use gdal_merge.py functionality
    gdal.Warp(output_path, tif_files[:10], format="GTiff")
    
    print(f"Merged raster saved to: {output_path}")

merge_rasters_direct(input_folder, output_path)

#%%

import os
import glob
import rasterio
from rasterio.merge import merge
from tqdm import tqdm
import gc
import numpy as np # Needed for nodata fill

def merge_tif_files(input_folder, output_path):
    """
    Merge TIFF files sequentially using rasterio.merge for minimal memory usage,
    and correctly updating the mosaic and metadata in each step.
    This also aims to minimize the final output file size by ensuring the merge
    correctly calculates the minimal bounding box.
    """
    tif_files = glob.glob(os.path.join(input_folder, "*.tif"))
    tif_files.extend(glob.glob(os.path.join(input_folder, "*.tiff")))

    if not tif_files:
        print("No TIFF files found!")
        return

    # Sort files to ensure deterministic merging order
    tif_files.sort()

    print(f"Found {len(tif_files)} TIFF files")
    print("Processing files sequentially...")

    # --- Initialization ---
    first_file = tif_files[0]
    print(f"Starting with: {os.path.basename(first_file)}")

    # Store the list of datasets to be merged, starting with the first one
    # Note: We open the file here and keep the dataset open for the initial meta/transform
    datasets_to_merge = []
    try:
        # Open and keep the first dataset
        first_src = rasterio.open(first_file)
        datasets_to_merge.append(first_src)
        current_meta = first_src.meta.copy()
        current_nodata = current_meta.get('nodata', None)
    except Exception as e:
        print(f"Error opening initial file {first_file}: {e}")
        return

    # --- Sequential Merging ---
    # Process remaining files one by one
    for i, file in enumerate(tqdm(tif_files[1:9], desc="Merging files")): # Loop over ALL remaining files
        try:
            # Open the new dataset and add it to the list
            new_src = rasterio.open(file)
            datasets_to_merge.append(new_src)

            # --- Perform Merge (The critical step) ---
            # Using merge() on the list of datasets correctly calculates the union
            # of all extents up to this point and creates the combined array.
            # This is the most robust way to manage the extent growth.
            mosaic, out_transform = merge(
                datasets_to_merge,
                method='first', # Or 'max', 'min', 'mean', 'last' - 'first' is common for simple mosaics
                nodata=current_nodata
            )

            # Important: Close the dataset that was just merged, to free memory
            # The list 'datasets_to_merge' keeps the *references* to the datasets
            # which can hold memory, but the largest memory consumer is the 'mosaic' array itself.
            # We only keep the list of open sources for the next merge iteration.
            # To be truly memory efficient, we must use a different approach (e.g., VRT and then gdal_translate)
            # but sticking to your current rasterio.merge approach:

            # Note: The *most* memory efficient way would be to write the current mosaic to a temporary file
            # and use that temporary file as the base for the next merge.
            # Sticking to the in-memory array for simplicity:

            current_mosaic = mosaic
            current_transform = out_transform

            # Force cleanup (Crucial for memory-efficient iteration)
            del mosaic
            gc.collect()

        except Exception as e:
            print(f"Error merging {os.path.basename(file)}: {e}")
            # Ensure the problematic file is closed if it was opened
            if 'new_src' in locals() and not new_src.closed:
                new_src.close()
            continue

        # Print progress every 50 files
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 2}/{len(tif_files)} files")

    # Ensure all source files are closed before final write
    for src in datasets_to_merge:
        src.close()
    
    # --- Finalize and Write Output ---
    if 'current_mosaic' not in locals():
        print("Error: Merging failed or only one file was found.")
        return

    # Update final metadata
    current_meta.update({
        "height": current_mosaic.shape[1],
        "width": current_mosaic.shape[2],
        "transform": current_transform,
        # Set compression to reduce file size (LZW is lossless and standard)
        "compress": "lzw"
    })
    
    # Write final output
    print("Writing final output...")
    try:
        with rasterio.open(output_path, "w", **current_meta) as dest:
            # Write 'nodata' value to areas not covered by any input data.
            # This is essential for a clean mosaic.
            if current_nodata is not None:
                 # Fill any unwritten areas with the nodata value
                 # In many cases, merge() does this, but it's a good safety step
                 # if the initial array wasn't fully filled.
                 dest.fill(current_nodata) 

            for i in tqdm(range(current_mosaic.shape[0]), desc="Writing bands"):
                # Use current_mosaic.astype(current_meta['dtype']) if data type issues arise
                dest.write(current_mosaic[i], i + 1)
        
        print(f"Successfully merged {len(tif_files)} files to: {output_path}")
        print(f"Final output size: {current_mosaic.shape}")

    except Exception as e:
        print(f"Error writing output file: {e}")




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

















#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def convert_uint_to_float_raster(
        input_path,
        output_path,
        scale_factor=0.1,
        compress="lzw"
):
    """
    Read unsigned integer GeoTIFF, scale values, and write as float32 GeoTIFF.

    Parameters
    ----------
    input_path : str
        Path to the input uint raster.
    output_path : str
        Path to save the output float32 raster.
    scale_factor : float
        The multiplier to scale the values (e.g., 0.1).
    compress : str
        Compression method for output (default: 'lzw').
    """
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        dtype = np.dtype(src.dtypes[0])

        if not np.issubdtype(dtype, np.unsignedinteger):
            raise ValueError("Input raster must have unsigned integer dtype.")

        # Determine nodata value
        nodata = src.nodata if src.nodata is not None else np.iinfo(dtype).max

        # Update output profile
        profile.update({
            "dtype": "float32",
            "nodata": np.nan,
            "compress": compress
        })

        with rasterio.open(output_path, "w", **profile) as dst:
            windows = list(src.block_windows(1))

            for idx, window in tqdm(windows, desc="Scaling raster blocks"):
                block = src.read(1, window=window)
                block = np.where(block == nodata, np.nan, block)
                block_scaled = block * scale_factor
                dst.write(block_scaled.astype("float32"), 1, window=window)

    gc.collect()
    print(f"Saved scaled float32 raster to: {output_path}")


def combine_two_float_rasters(
    raster1_path,
    raster2_path,
    out_path,
    operation=lambda a, b: a + b,  # Default operation: addition
    fill_value=np.nan,
    compress="lzw"
):
    """
    Combine two float32 raster maps using a specified operation (e.g., addition),
    handle fill values, and write the result to a new raster file.

    Parameters
    ----------
    raster1_path : str
        Path to the first input raster.
    raster2_path : str
        Path to the second input raster.
    out_path : str
        Path to the output raster file.
    operation : callable
        A function that takes two arrays and returns their combination (default: addition).
    fill_value : float
        NoData value to ignore in computations (default: np.nan).
    compress : str
        Compression method for the output raster (default: 'lzw').
    """
    with rasterio.open(raster1_path) as src1, rasterio.open(raster2_path) as src2:
        if src1.shape != src2.shape:
            raise ValueError("Input rasters must have the same dimensions.")

        profile = src1.profile.copy()
        profile.update({
            "dtype": "float32",
            "nodata": fill_value,
            "compress": compress
        })

        with rasterio.open(out_path, "w", **profile) as dst:
            windows = list(src1.block_windows(1))

            for _, window in tqdm(windows, desc="Combining raster blocks"):
                b1 = src1.read(1, window=window)
                b2 = src2.read(1, window=window)

                mask1 = np.isnan(b1) if np.isnan(fill_value) else (b1 == fill_value)
                mask2 = np.isnan(b2) if np.isnan(fill_value) else (b2 == fill_value)

                b1_clean = np.where(mask1, 0, b1)
                b2_clean = np.where(mask2, 0, b2)

                del b1, b2
                gc.collect()

                combined = operation(b1_clean, b2_clean)
                combined[mask1 & mask2] = fill_value

                del mask1, mask2, b1_clean, b2_clean
                gc.collect()

                dst.write(combined.astype("float32"), 1, window=window)

                # Explicit memory cleanup
                gc.collect()

    gc.collect()
    print(f"Combined raster saved to: {out_path}")



def reproject_raster(
    input_path,
    reference_path,
    output_path,
    compress="lzw",
    chunks={"x": 1024, "y": 1024},
    overwrite=False
    ):
    """
    Reproject a raster to match the CRS, resolution, and extent of a reference raster.

    Parameters
    ----------
    input_path : str
        Path to the raster to reproject.
    reference_path : str
        Path to the reference raster.
    output_path : str
        Path to save the reprojected raster.
    compress : str
        Compression method for output (default: 'lzw').
    chunks : dict
        Chunk size for Dask loading (default: {"x": 1024, "y": 1024}).
    overwrite : bool
        Whether to overwrite an existing file.
    """
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"{output_path} exists. Use overwrite=True to replace it.")

    ref = rxr.open_rasterio(reference_path, masked=True, chunks=chunks).squeeze("band", drop=True)
    target = rxr.open_rasterio(input_path, masked=True, chunks=chunks).squeeze("band", drop=True)

    reprojected = target.rio.reproject_match(ref)

    reprojected.rio.to_raster(
        output_path,
        compress=compress,
        tiled=True,
        blockxsize=256,
        blockysize=256
    )

    print(f"Reprojected raster saved to: {output_path}")
    del ref, target, reprojected
    gc.collect()


def stack_layers_to_csv(
    group_layer1_path,
    group_layer2_path,
    value_layer_path,
    output_path="stacked_summary.csv",
    num_slices=100,
    group1_name="group1",
    group2_name="group2",
    value_name="value"
):
    """
    Stack three raster layers, summarize the third by grouping over the first two, and write to CSV.

    Parameters
    ----------
    group_layer1_path : str
        Path to the first grouping raster layer.
    group_layer2_path : str
        Path to the second grouping raster layer.
    value_layer_path : str
        Path to the value raster layer to be summarized.
    output_path : str
        Output CSV file path.
    num_slices : int
        Number of vertical slices to process in chunks.
    group1_name : str
        Column name for the first group layer.
    group2_name : str
        Column name for the second group layer.
    value_name : str
        Column name for the value layer.
    """
    print("Loading raster layers...")
    layer1 = rxr.open_rasterio(group_layer1_path, masked=True, chunks={"x": 1024, "y": 1024}).squeeze("band")
    layer2 = rxr.open_rasterio(group_layer2_path, masked=True, chunks={"x": 1024, "y": 1024}).squeeze("band")
    layer3 = rxr.open_rasterio(value_layer_path, masked=True, chunks={"x": 1024, "y": 1024}).squeeze("band")
    gc.collect()

    layers = [layer1, layer2, layer3]
    layer_names = [group1_name, group2_name, value_name]
    group_cols = layer_names[:-1]
    value_col = layer_names[-1]

    total_width = layer1.sizes["x"]
    step = total_width // num_slices
    dfs = []

    print("Processing raster slices...")
    for i in tqdm(range(num_slices), desc="Slicing and summarizing"):
        x_start = i * step
        x_end = (i + 1) * step if i < (num_slices - 1) else total_width

        try:
            sliced_layers = [layer.isel(x=slice(x_start, x_end)) for layer in layers]
            flattened = [sl.values.reshape(-1).astype("float32") for sl in sliced_layers]

            if len(set(arr.shape[0] for arr in flattened)) != 1:
                print(f"Skipping slice {i + 1} due to shape mismatch.")
                continue

            stacked = da.stack(flattened, axis=1)
            df = dd.from_dask_array(stacked, columns=layer_names)
            df_pd = df.compute().dropna(subset=group_cols + [value_col])

            if df_pd.empty:
                continue

            summary = df_pd.groupby(group_cols)[value_col].agg(
                mean="mean",
                min="min",
                max="max",
                count="count"
            ).reset_index()

            dfs.append(summary)
            del df_pd, summary, df, stacked, flattened
            gc.collect()

        except Exception as e:
            print(f"Slice {i + 1} failed: {e}")
            continue

    if dfs:
        final = pd.concat(dfs)
        final["weighted_sum"] = final["mean"] * final["count"]
        final_summary = (
            final.groupby(group_cols, as_index=False)
            .agg({
                "weighted_sum": "sum",
                "count": "sum",
                "min": "min",
                "max": "max"
            })
        )

        # Calculate final weighted mean
        final_summary["mean"] = final_summary["weighted_sum"] / final_summary["count"]

        # Clean up and reorder
        final_summary = final_summary[[group1_name, group2_name, "mean", "min", "max", "count"]]
        final_summary = final_summary.rename(columns={
            "mean": f"{value_col}_mean",
            "min": f"{value_col}_min",
            "max": f"{value_col}_max",
            "count": f"{value_col}_count"
        })
        final_summary.to_csv(output_path, index=False)
        print(f"Summary written to: {output_path}")


def generate_carbon_density_raster(lulc_path, cz_path, carbon_density_lookup_table_path, out_path):
    """
    Generate a carbon density raster by mapping carbon zone and LULC combinations
    to values from a carbon lookup table.

    Parameters
    ----------
    lulc_path : str
        Path to the land use land cover (LULC) raster.
    cz_path : str
        Path to the carbon zone raster.
    carbon_density_lookup_table_path : str
        Path to CSV file containing the carbon density lookup table,
        indexed by carbon_zone_id with columns for LULC types.
    out_path : str
        Output path for the resulting carbon density raster.
    """
    # Read lookup table from CSV

    carbon_table = pd.read_csv(carbon_density_lookup_table_path, index_col=False)

    lulc_filename = os.path.basename(lulc_path)

    with rasterio.open(lulc_path) as lulc_src, rasterio.open(cz_path) as cz_src:
        assert lulc_src.shape == cz_src.shape, f"Shape mismatch between rasters: {lulc_filename}"

        profile = lulc_src.profile
        profile.update(dtype=rasterio.float32, nodata=np.nan)

        block_indices = list(lulc_src.block_windows(1))

        with rasterio.open(out_path, "w", **profile) as out_dst:
            for ji, window in tqdm(block_indices, desc=f"Processing {lulc_filename}"):
                try:
                    lulc_block = lulc_src.read(1, window=window)
                    carbon_zone_block = cz_src.read(1, window=window)
                except rasterio.errors.RasterioIOError as e:
                    print(f"Skipping corrupt tile in {lulc_filename} at {window}: {e}")
                    continue

                carbon_block = np.full_like(lulc_block, np.nan, dtype=np.float32)

                for carbon_zone_id in np.unique(carbon_zone_block):
                    for lulc_id in np.unique(lulc_block):
                        # Filter the lookup table for the matching carbon zone and lulc
                        match = carbon_table[
                            (carbon_table["carbon_zone_id"] == carbon_zone_id) &
                            (carbon_table["lulc_id"] == lulc_id)
                            ]

                        # Skip if no match found
                        if match.empty:
                            continue

                        value = match["carbon_density_mean"].values[0]

                        mask = (carbon_zone_block == carbon_zone_id) & (lulc_block == lulc_id)
                        carbon_block[mask] = value

                out_dst.write(carbon_block, 1, window=window)

    gc.collect()
    print(f"Saved: {out_path}")

def summarize_raster_by_region(value_raster_path, region_boundary_path, out_path):
    """
    Summarize value raster by polygon regions from a vector file (e.g., GPKG).
    Includes mean, min, max, count, and area (m², ha, km²) for each region.

    Parameters
    ----------
    value_raster_path : str
        Path to the value raster (e.g., carbon density).
    region_boundary_path : str
        Path to the vector file (GeoPackage) containing polygon regions.
    out_csv_path : str
        Output path for the CSV summary.
    """
    # Load vector data
    regions = gpd.read_file(region_boundary_path)

    # Open the raster once
    with rasterio.open(value_raster_path) as src:
        raster_crs = src.crs
        if regions.crs != raster_crs:
            print(f"Reprojecting vector data from {regions.crs} to match raster CRS {raster_crs}")
            regions = regions.to_crs(raster_crs)

        results = []
        id_list = []
        for idx, row in tqdm(regions.iterrows(), total=len(regions), desc="Summarizing polygons"):
            geom = [row.geometry]
            id_list.append(row.get("id", idx))
            try:
                masked, _ = mask(src, geom, crop=True, nodata=np.nan, all_touched=True)
                values = masked[0]
                values = values[~np.isnan(values)]

                area_m2 = values.size

                stats = {
                    "index_id": row.get("id", idx),
                    "mean": values.mean(),
                    "min": values.min(),
                    "max": values.max(),
                    "count": values.size,
                    "total": values.sum()
                }
                results.append(stats)

            except Exception as e:
                print(f"Error processing region {idx}: {e}")
                continue
    regions["index_id"] =id_list
    df = pd.DataFrame(results)
    df = regions.merge(df, on="index_id", how="right")
    df = df.drop(columns=["index_id","geometry"])
    df['year'] = '2019'
    df.to_csv(out_path, index=False)
    print(f"Summary written to: {out_path}")

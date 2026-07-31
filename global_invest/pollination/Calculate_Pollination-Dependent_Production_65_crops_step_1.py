
import os
import pygeoprocessing
from osgeo import gdal
import pandas as pd

# Load the Excel file with pollination-dependent crops data
file_path = r"D:\Shared drives\NatCapTEEMs\Projects\Global GEP\Ecosystem Services SubFolders\Pollination\poll_dep_greater_0.xlsx"
df = pd.read_excel(file_path)

# Define base directories
base_crop_data_dir = r"D:\Shared drives\NatCapTEEMs\Projects\Global GEP\Ecosystem Services SubFolders\Pollination\HarvestedAreaYield175Crops_Geotiff\HarvestedAreaYield175Crops_Geotiff\GeoTiff"
pollination_suff_path = r"D:\Shared drives\NatCapTEEMs\Projects\Global GEP\Ecosystem Services SubFolders\Pollination\aligned_poll_suff_to_apple.tif"
output_base_dir = r"D:\Shared drives\NatCapTEEMs\Projects\Global GEP\Ecosystem Services SubFolders\Pollination\pollination_dependent_crops"

# Ensure output directory exists
os.makedirs(output_base_dir, exist_ok=True)

# Function to calculate pollination-dependent production
def calculate_pollination_dependent_production(poll_suff, crop_production, dependence_ratio):
    return poll_suff * crop_production * dependence_ratio

# Process each row in the DataFrame
for _, row in df.iterrows():
    crop_filenm = row['filenm']
    dependence_ratio = row['poll.dep']
    
    # Locate the crop production raster file
    crop_production_dir = os.path.join(base_crop_data_dir, crop_filenm)
    crop_production_raster = os.path.join(crop_production_dir, f"{crop_filenm}_Production.tif")
    print(crop_filenm)
    print(dependence_ratio)
    print(crop_production_raster)
    
    if os.path.exists(crop_production_raster):
        # Define output raster path
        output_raster = os.path.join(output_base_dir, f"pollination_dependent_{crop_filenm}.tif")
        
        # Execute the raster operation
        pygeoprocessing.raster_calculator(
            [(pollination_suff_path, 1), (crop_production_raster, 1), (dependence_ratio, 'raw')],
            calculate_pollination_dependent_production,
            output_raster,
            gdal.GDT_Float32,
            None
        )
        
        print(f"Pollination-dependent production raster generated for {crop_filenm} and saved to {output_raster}")
    else:
        print(f"Production raster not found for {crop_filenm}. Skipping.")

print("Processing complete.")




# import pygeoprocessing
# import os
# from osgeo import gdal 

# # Define file paths
# resampled_poll_suff = r"D:\Shared drives\NatCapTEEMs\Projects\Global GEP\Ecosystem Services SubFolders\Pollination\aligned_poll_suff_to_apple.tif"
# apple_production_raster = r"D:\Shared drives\NatCapTEEMs\Projects\Global GEP\Ecosystem Services SubFolders\Pollination\HarvestedAreaYield175Crops_Geotiff\HarvestedAreaYield175Crops_Geotiff\GeoTiff\apple\apple_Production.tif"
# output_raster = r"D:\Shared drives\NatCapTEEMs\Projects\Global GEP\Ecosystem Services SubFolders\Pollination\pollination_dependent_apple_production.tif"

# # Pollination dependence ratio for apples
# apple_dependence_ratio = 0.65

# # Define the operation to multiply the rasters and the dependence ratio
# def calculate_pollination_dependent_production(poll_suff, apple_production):
#     return poll_suff * apple_production * apple_dependence_ratio

# # Execute the raster operation
# pygeoprocessing.raster_calculator(
#     [(resampled_poll_suff, 1), (apple_production_raster, 1)],
#     calculate_pollination_dependent_production,
#     output_raster,
#     gdal.GDT_Float32,
#     None
# )

# print("Pollination-dependent apple production raster generated successfully.")

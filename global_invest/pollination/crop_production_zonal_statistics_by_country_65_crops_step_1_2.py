import os
import glob
import pygeoprocessing
from osgeo import ogr
import time
from pathlib import Path

import os
import glob
import pygeoprocessing
from osgeo import ogr, gdal
from pathlib import Path
import pandas as pd

# Load the Excel file with pollination-dependent crops data
file_path = r"D:\Shared drives\NatCapTEEMs\Projects\Global GEP\Ecosystem Services SubFolders\Pollination\poll_dep_greater_0.xlsx"
df = pd.read_excel(file_path)

# Define base directories
base_crop_data_dir = r"D:\Shared drives\NatCapTEEMs\Projects\Global GEP\Ecosystem Services SubFolders\Pollination\HarvestedAreaYield175Crops_Geotiff\HarvestedAreaYield175Crops_Geotiff\GeoTiff"
pollination_suff_path = r"D:\Shared drives\NatCapTEEMs\Projects\Global GEP\Ecosystem Services SubFolders\Pollination\aligned_poll_suff_to_apple.tif"
output_raster_dir = r"D:\Shared drives\NatCapTEEMs\Projects\Global GEP\Ecosystem Services SubFolders\Pollination\pollination_dependent_crops"
zonal_output_dir = r"D:\Shared drives\NatCapTEEMs\Projects\Global GEP\Ecosystem Services SubFolders\Pollination\crop_production_by_country"

# Ensure output directories exist
os.makedirs(output_raster_dir, exist_ok=True)
os.makedirs(zonal_output_dir, exist_ok=True)

# Define the path to the zone shapefile (e.g., EEZ or country boundaries)
zone_input_path = r"D:\Shared drives\NatCapTEEMs\Projects\Global GEP\Ecosystem Services SubFolders\Pollination\ee_r264_correspondence.shp"

# Define the field name in the shapefile that represents the ID for each zone (e.g., country, EEZ)
ID_field = 'gtapv7_r_7'  # Adjust this field based on your shapefile

# Function to calculate pollination-dependent production
def calculate_pollination_dependent_production(poll_suff, crop_production, dependence_ratio):
    return poll_suff * crop_production * dependence_ratio

# Function to perform zonal statistics and save results
def calculate_zonal_statistics(raster_input_path, output_table):
    # Open the output file and write the headers
    with open(output_table, 'w', newline='') as results_file:
        results_file.write('Zone_ID,Count,Sum\n')

        # Run zonal statistics to calculate the sum within each zone (e.g., EEZ or country)
        zonal_stats = pygeoprocessing.zonal_statistics((raster_input_path, 1), zone_input_path, polygons_might_overlap=False)

        # Open the shapefile and access the layer
        driver = ogr.GetDriverByName('ESRI Shapefile')
        target_vector = driver.Open(zone_input_path, 0)
        target_layer = target_vector.GetLayer()

        # Iterate through each feature (e.g., EEZ or country) in the shapefile
        for feature in target_layer:
            feature_FID = feature.GetFID()
            feature_ID = feature.GetField(ID_field)

            # Retrieve zonal statistics for the current feature
            feature_stats = zonal_stats.get(feature_FID, {})
            feature_count = float(feature_stats.get('count', 0))
            feature_sum = float(feature_stats.get('sum', 0))

            # Write the results to the output file
            results_file.write(f'"{feature_ID}",{feature_count},{feature_sum}\n')

        # Close the shapefile to ensure the data is saved and released
        target_vector = None
        target_layer = None

    print(f"Finished writing zonal stats for raster input: {raster_input_path}")

# Process each row in the DataFrame
for _, row in df.iterrows():
    crop_filenm = row['filenm']
    dependence_ratio = row['poll.dep']
    
    # Locate the crop production raster file
    crop_production_dir = os.path.join(base_crop_data_dir, crop_filenm)
    crop_production_raster = os.path.join(crop_production_dir, f"{crop_filenm}_Production.tif")
    
    if os.path.exists(crop_production_raster):
        # Define output raster path
        output_raster = os.path.join(output_raster_dir, f"pollination_dependent_{crop_filenm}.tif")
        
        # Execute the raster operation to calculate pollination-dependent production
        pygeoprocessing.raster_calculator(
            [(pollination_suff_path, 1), (crop_production_raster, 1), (dependence_ratio, 'raw')],
            calculate_pollination_dependent_production,
            output_raster,
            gdal.GDT_Float32,
            None
        )
        
        print(f"Pollination-dependent production raster generated for {crop_filenm} and saved to {output_raster}")
        
        # Define the output CSV file path based on the raster file name
        output_table = os.path.join(zonal_output_dir, Path(output_raster).stem + "_crop_production_by_country.csv")
        
        # Perform zonal statistics and save the results
        calculate_zonal_statistics(output_raster, output_table)
    
    else:
        print(f"Production raster not found for {crop_filenm}. Skipping.")

print("Processing complete for all crops.")






# # Define the folder containing the pollination-dependent crop production raster data
# raster_folder = r"D:\Shared drives\NatCapTEEMs\Projects\Global GEP\Ecosystem Services SubFolders\Pollination\pollination_dependent_crop_production"

# # Define the path to the zone shapefile (e.g., EEZ or country boundaries)
# zone_input_path = r"D:\Shared drives\NatCapTEEMs\Projects\Global GEP\Ecosystem Services SubFolders\Pollination\ee_r264_correspondence.shp"

# # Define the field name in the shapefile that represents the ID for each zone (e.g., country, EEZ)
# ID_field = 'gtapv7_r_7'  # Adjust this field based on your shapefile


# # Get a list of all raster files related to pollination-dependent crop production in the specified folder
# raster_inputs = glob.glob(os.path.join(raster_folder, '*.tif'))  # Update with the correct file extension if needed
# print(raster_inputs)

# # Define the output directory for the results
# output_dir = r"D:\Shared drives\NatCapTEEMs\Projects\Global GEP\Ecosystem Services SubFolders\Pollination\crop_production_by_country"

# # Ensure the output directory exists
# os.makedirs(output_dir, exist_ok=True)

# # Loop through all raster inputs and calculate zonal statistics
# for rastr in raster_inputs:

#     # Define the output CSV file path based on the raster file name
#     output_table = os.path.join(output_dir, Path(rastr).stem + "_crop_production_by_country.csv")
#     print(f"Output will be saved to: {output_table}")

#     # Open the output file and write the headers
#     with open(output_table, 'w', newline='') as results_file:
#         results_file.write('Zone_ID,Count,Sum\n')

#         raster_input_path = rastr
#         print(f"Processing raster: {raster_input_path}")

#         # Run zonal statistics to calculate the sum within each zone (e.g., EEZ or country)
#         start_time = time.time()
#         zonal_stats = pygeoprocessing.zonal_statistics((raster_input_path, 1), zone_input_path, polygons_might_overlap=False)
#         end_time = time.time()

#         # Calculate and print the elapsed time
#         elapsed_time = (end_time - start_time) / 60.0
#         print(f"Elapsed time: {elapsed_time:.4f} minutes")

#         # Open the shapefile and access the layer
#         driver = ogr.GetDriverByName('ESRI Shapefile')
#         target_vector = driver.Open(zone_input_path, 0)
#         target_layer = target_vector.GetLayer()

#         # Iterate through each feature (e.g., EEZ or country) in the shapefile
#         for feature in target_layer:
#             feature_FID = feature.GetFID()
#             feature_ID = feature.GetField(ID_field)

#             # Retrieve zonal statistics for the current feature
#             feature_stats = zonal_stats.get(feature_FID, {})
#             feature_count = float(feature_stats.get('count', 0))
#             feature_sum = float(feature_stats.get('sum', 0))

#             # Write the results to the output file
#             results_file.write(f'"{feature_ID}",{feature_count},{feature_sum}\n')

#         print(f"Finished writing zonal stats for raster input: {raster_input_path}")

#     # Close the shapefile to ensure the data is saved and released
#     target_vector = None
#     target_layer = None

# print("Finished aggregating results for all rasters.")

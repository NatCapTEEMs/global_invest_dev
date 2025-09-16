import os
import glob
import rasterio
import numpy as np
import re


def sum_yearly_rasters(input_folder, output_folder, file_pattern="*.tif"):
    """
    Sums raster files for each year found in the input folder and saves the
    summed raster for each year to the output folder.

    Args:
        input_folder (str): Path to the folder containing the TIFF files.
        output_folder (str): Path to the folder where the yearly summed rasters will be saved.
        file_pattern (str): Pattern to match TIFF files (e.g., "*.tif", "data_*.tiff").
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Group files by year
    yearly_files = {}
    filepaths = glob.glob(os.path.join(input_folder, file_pattern))

    if not filepaths:
        print(f"No files found matching pattern '{file_pattern}' in '{input_folder}'.")
        return

    for filepath in filepaths:
        filename = os.path.basename(filepath)
        # Try to extract year from filename. This assumes a 4-digit year.
        # You might need to adjust this regex based on your actual filenames.
        match = re.search(r'(\d{4})', filename)
        if match:
            year = match.group(1)
            if year not in yearly_files:
                yearly_files[year] = []
            yearly_files[year].append(filepath)
        else:
            print(f"Could not extract year from filename: {filename}. Skipping.")

    if not yearly_files:
        print("No years could be identified from the filenames. Please check your file naming convention.")
        return

    # Process each year
    for year, files_for_year in yearly_files.items():
        print(f"Processing year: {year} with {len(files_for_year)} files...")
        summed_data = None
        profile = None

        for i, filepath in enumerate(files_for_year):
            try:
                with rasterio.open(filepath) as src:
                    if i == 0:
                        # Initialize summed_data with the first raster's data and get its profile
                        summed_data = src.read(1).astype(np.float32) # Ensure float for summation
                        profile = src.profile
                        profile.update(dtype=rasterio.float32) # Update profile to reflect float output
                    else:
                        summed_data += src.read(1)
            except Exception as e:
                print(f"Error reading {filepath}: {e}. Skipping this file.")

        if summed_data is not None and profile is not None:
            output_filepath = os.path.join(output_folder, f"precip_annual_{year}.tif")
            try:
                with rasterio.open(output_filepath, 'w', **profile) as dst:
                    dst.write(summed_data, 1)
                print(f"Successfully created: {output_filepath}")
            except Exception as e:
                print(f"Error writing {output_filepath}: {e}")
        else:
            print(f"No data to sum for year {year}. This might happen if all files for the year were skipped.")


if __name__ == '__main__':
    precip_folder = "E:/MonthlyPrecipCHIRPS"
    output_folder = "E/PrecipAnnualCHIRPS"
    sum_yearly_rasters(precip_folder, output_folder, file_pattern="*.tif")

    test = os.path.join(precip_folder, "chirps-v2.0.2000.01_precip.tif")
    os.path.exists(test)

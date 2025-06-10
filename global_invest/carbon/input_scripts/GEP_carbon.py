import os
from tqdm import tqdm
import numpy as np
import geopandas as gpd
from collections import defaultdict
import pandas as pd
import rioxarray
import dask.dataframe as dd
import dask.array as da
import gc
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.enums import Resampling as ResamplingEnum
from rasterio.mask import mask
import xarray as xr
from glob import glob
import time
import matplotlib.pyplot as plt
import folium
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  # Correct import
matplotlib.use('Agg')
from branca.colormap import LinearColormap
import re

# Set the working directory and the input/output directory
working_dir = "/Users/long/Library/CloudStorage/GoogleDrive-yxlong@umn.edu/Shared drives/NatCapTEEMs/Projects/Global GEP/Ecosystem Services SubFolders/Carbon"
os.chdir(working_dir)
print("Current working directory:", os.getcwd())
input_dir = os.path.join(working_dir, "input")
output_dir = os.path.join(working_dir, "output")


#%% Generate new carbon density lookup table

# Generate total carbon density map by combining aboveground and belowground carbon density layers.
# The input carbon density maps are obtained from:
# NASA ORNL DAAC: https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1763

ag_carbon_density_path = os.path.join(input_dir, "aboveground_biomass_carbon_2010.tif")
bg_carbon_density_path = os.path.join(input_dir, "belowground_biomass_carbon_2010.tif")
ag_carbon_density = rioxarray.open_rasterio(ag_carbon_density_path, chunks="auto")
bg_carbon_density = rioxarray.open_rasterio(bg_carbon_density_path, chunks="auto").rio.reproject_match(ag_carbon_density)
fill_value = 65535
ag_carbon_density_filled = ag_carbon_density.where(ag_carbon_density != fill_value, 0)
bg_carbon_density_filled = bg_carbon_density.where(bg_carbon_density != fill_value, 0)
total_carbon_density = ag_carbon_density_filled + bg_carbon_density_filled


del ag_carbon_density_filled, bg_carbon_density_filled

# Mask where both ag and bg have nodata values
both_nan = (ag_carbon_density == fill_value) & (bg_carbon_density == fill_value)

del ag_carbon_density, bg_carbon_density
# Where both were NaN, set results back to NaN
total_carbon_density = total_carbon_density.where(~both_nan, fill_value)
total_carbon_density.rio.to_raster(
    os.path.join(input_dir, "total_biomass_carbon_2010.tif"),
    compress="lzw",
    windowed=True,
    dtype="uint16"
)

del both_nan


#%%
############ Start here

# import carbon zone and land use land cover data

lulc2010_path = os.path.join(input_dir, "esa/lulc_esa_2010.tif")
cz_path = os.path.join(input_dir, "carbon_zones_rasterized.tif")
total_carbon_density_path = os.path.join(input_dir, "total_biomass_carbon_2010.tif")

lulc2010 = rioxarray.open_rasterio(lulc2010_path, chunks="auto").squeeze()
cz = rioxarray.open_rasterio(cz_path, chunks="auto").squeeze()
total_carbon_density = rioxarray.open_rasterio(total_carbon_density_path, chunks="auto").squeeze().rio.reproject_match(lulc2010)


def compute_carbon_density_table(cz, lulc, carbon, num_slices=100):
    total_width = cz.sizes['x']
    step = total_width // num_slices
    dfs = []

    for i in range(num_slices):
        print(f"Processing longitudinal slice {i+1}/{num_slices}")
        x_start = i * step
        x_end = (i + 1) * step if i < (num_slices - 1) else total_width

        cz_slice = cz.isel(x=slice(x_start, x_end))
        lulc_slice = lulc.isel(x=slice(x_start, x_end))
        carbon_slice = carbon.isel(x=slice(x_start, x_end))

        # Flatten
        cz_vals = cz_slice.values.reshape(-1)
        lulc_vals = lulc_slice.values.reshape(-1)
        carbon_vals = carbon_slice.values.reshape(-1)
        carbon_vals = carbon_vals.astype("float32")  # Convert from UInt16 to Float32
        carbon_vals[carbon_vals == 65535] = np.nan  # Replace invalid 65535 with NaN
        carbon_vals = carbon_vals * 0.1


        # Mask valid data
        mask = (~da.isnan(carbon_vals)) & (cz_vals >= 0) & (lulc_vals >= 0)
        cz_id = cz_vals[mask]
        lulc_id = lulc_vals[mask]
        carbon_masked = carbon_vals[mask]

        # Stack and group
        stacked = da.stack([cz_id, lulc_id, carbon_masked], axis=1)
        df = dd.from_dask_array(stacked, columns=["cz_id", "lulc_id", "carbon"])
        df_mean = df.groupby(["cz_id", "lulc_id"]).carbon.mean().compute().reset_index()

        dfs.append(df_mean)

    # Merge all slices and finalize
    final_df = pd.concat(dfs)
    return final_df.groupby(["cz_id", "lulc_id"]).mean().reset_index()

# === Compute tables ===
# ag_carbon_density_table = compute_carbon_density_table(cz, lulc2010, ag_carbon_density)
# del ag_carbon_density
# bg_carbon_density_table = compute_carbon_density_table(cz, lulc2010, bg_carbon_density)
# del bg_carbon_density

total_carbon_density_table = compute_carbon_density_table(cz, lulc2010, total_carbon_density)
del total_carbon_density

total_carbon_density_table = total_carbon_density_table[total_carbon_density_table['cz_id'] != 0]

total_carbon_density_table = total_carbon_density_table.sort_index().sort_index(axis=1)
total_carbon_density_table_path = os.path.join(output_dir, "carbon_table_SG2020_long.xlsx")
total_carbon_density_table.to_excel(total_carbon_density_table_path, index=False)


# Pivot the table
pivot_df = total_carbon_density_table.pivot_table(
    index="cz_id",
    columns="lulc_id",
    values="carbon"
)

# Optional: sort for readability
pivot_df = pivot_df.sort_index().sort_index(axis=1)
pivot_carbon_out_path = os.path.join(output_dir, "carbon_table_SG2020.xlsx")
pivot_df.to_excel(pivot_carbon_out_path)
print("final lookup table saved")

#%% time series application

carbon_table_path = os.path.join(output_dir, "carbon_table_SG2020.xlsx")
carbon_table = pd.read_excel(carbon_table_path, index_col=0)
carbon_table.columns = carbon_table.columns.astype(int)  # Ensure LULC codes are integers
lulc_folder = os.path.join(input_dir, "esa")
cz_path = os.path.join(input_dir, "carbon_zones_rasterized.tif")

# Loop over all LULC tif files
for lulc_path in glob(os.path.join(lulc_folder, "*.tif")):
    lulc_filename = os.path.basename(lulc_path)
    suffix = lulc_filename[-8:-4]  # Example: "2020"

    out_path = os.path.join(output_dir, f"carbon_density_results/carbon_density_{suffix}.tif")

    if os.path.exists(out_path):
        print(f"⏭️ Skipping {suffix}, output already exists.")
        continue

        with rasterio.open(lulc_path) as lulc_src, rasterio.open(cz_path) as cz_src:
            assert lulc_src.shape == cz_src.shape, f"Shape mismatch for {lulc_filename}"

            profile = lulc_src.profile
            profile.update(dtype=rasterio.float32, nodata=np.nan)

            block_indices = list(lulc_src.block_windows(1))

            with rasterio.open(out_path, "w", **profile) as out_dst:
                for ji, window in tqdm(block_indices, desc=f"Processing {lulc_filename}"):
                    try:
                        lulc_block = lulc_src.read(1, window=window)
                        cz_block = cz_src.read(1, window=window)
                    except rasterio.errors.RasterioIOError as e:
                        print(f"Skipping corrupt tile in {lulc_filename} at {window}: {e}")
                        continue

                    carbon_block = np.full_like(lulc_block, np.nan, dtype=np.float32)

                    for cz_id in np.unique(cz_block):
                        if cz_id not in carbon_table.index:
                            continue
                        for lulc_id in np.unique(lulc_block):
                            if lulc_id not in carbon_table.columns:
                                continue
                            value = carbon_table.at[cz_id, lulc_id]
                            if value == 0:
                                continue
                            mask = (cz_block == cz_id) & (lulc_block == lulc_id)
                            carbon_block[mask] = value

                    out_dst.write(carbon_block, 1, window=window)

        print(f"✅ Saved: {out_path}")



#%% calculate carbon storage

def compute_total_carbon(carbon_density_path, boundary_data):
    """Compute total carbon (Mg) per country from a TIFF and add as new column.
    Args:
        carbon_density_path: Path to carbon density TIFF (MgC/ha)
        boundary_data: GeoDataFrame with country and region polygons (will be modified in-place)
    Returns:
        Modified GeoDataFrame with new column for carbon totals
    """
    start_time = time.time()

    with rasterio.open(carbon_density_path) as src:
        # Read CRS and resolution once (outside country loop)
        crs = src.crs
        is_geographic = crs.is_geographic if crs else False
        x_res, y_res = src.res

        # Extract year from filename (e.g., "carbon_2000.tif" -> "2000")
        filename = os.path.basename(carbon_density_path)
        year = filename.split('_')[-1].split('.')[0]  # More robust extraction

        # Initialize results list
        results = []
        country_count = len(boundary_data)
        for _, country in boundary_data.iterrows():
            geom = [country.geometry]

            try:
                # Calculate pixel area in hectares
                if is_geographic:
                    lat = country.geometry.centroid.y
                    # WGS84 ellipsoid approximation
                    lon_scale = 111320 * np.cos(np.radians(lat))
                    pixel_width_m = lon_scale * abs(x_res)
                    pixel_height_m = 111000 * abs(y_res)
                    pixel_area_m2 = pixel_width_m * pixel_height_m
                else:
                    # Projected CRS (units in meters)
                    pixel_area_m2 = abs(x_res * y_res)

                pixel_area_ha = pixel_area_m2 / 10000  # Convert to hectares

                # Mask data to country boundary
                masked, _ = mask(src, geom, crop=True, nodata=np.nan, all_touched=True)
                masked_data = masked[0]  # Values in MgC/ha

                # Calculate total carbon (Mg) = sum(MgC/ha * area_ha)
                total_carbon_Mg = np.nansum(masked_data * pixel_area_ha)
                total_carbon_Mt = total_carbon_Mg / 1e6

                print(f"Carbon Storage in {country['id']} {country['nev_name']}: {total_carbon_Mt:.2f} Mt")

            except Exception as e:
                print(f"Error processing {country['nev_name']}: {str(e)}")
                total_carbon_Mt = np.nan

            results.append(total_carbon_Mt)

        # Add new column to GeoDataFrame
        boundary_data[f'carbon_{year}'] = results

        # Calculate processing time
        processing_time = time.time() - start_time
        time_per_country = processing_time / country_count

        print(f"\nCompleted {country_count} countries in {processing_time:.2f} seconds")
        print(f"Average time per country: {time_per_country:.4f} seconds")

        return boundary_data, processing_time

    return boundary_data


carbon_density_folder = os.path.join(output_dir, "carbon_density_results")

boundary_data_path = os.path.join(input_dir, "ee_r264_correspondence.gpkg")

if __name__ == "__main__":
    print("Starting carbon calculation pipeline...")
    pipeline_start_time = time.time()

    # Load vector boundary data
    boundary_data = gpd.read_file(boundary_data_path)

    carbon_density_files = [
        f for f in os.listdir(carbon_density_folder)
        if f.lower().endswith(('.tif', '.tiff'))
    ]

    if not carbon_density_files:
        print("No carbon density raster files found in the specified folder.")
    else:
        print(f"Found {len(carbon_density_files)} carbon density rasters to process...")

        # Process each carbon density raster
        for file_num, carbon_file in enumerate(carbon_density_files, 1):
            print(f"\nProcessing raster {file_num} of {len(carbon_density_files)}: {carbon_file}")
            file_start_time = time.time()

            full_raster_path = os.path.join(carbon_density_folder, carbon_file)
            boundary_data, process_time = compute_total_carbon(full_raster_path, boundary_data)

            file_processing_time = time.time() - file_start_time
            print(f"Completed in {file_processing_time:.2f} seconds")


    total_processing_time = time.time() - pipeline_start_time
    print(f"\nPipeline summary:")
    print(f"- Total execution time: {total_processing_time:.2f} seconds")
    print(f"- Processed {len(carbon_density_files)} carbon density rasters")
    print(f"- Analyzed {len(boundary_data)} boundary features")

    # Save final carbon calculation results
    carbon_storage_results = boundary_data
    carbon_storage_results_path = os.path.join(output_dir, "carbon_storage_results.gpkg")
    carbon_storage_results.to_file(carbon_storage_results_path, driver="GPKG")
    carbon_storage_results.to_csv(os.path.join(output_dir, "carbon_storage_results.csv"))
    print(f"\nFinal carbon calculation results saved to {carbon_results_output_path}")

#%% plotting

carbon_storage_results =  pd.read_csv(os.path.join(output_dir, "carbon_storage_results.csv"))

def plot_carbon_map(gdf, year, output_file="carbon_decile_map.png", min_lat=-60):
    """
    Create a carbon storage heat map with decile-based bins, excluding extreme southern latitudes.

    Args:
        gdf (GeoDataFrame): Input geographic data
        year (int/str): Year column to visualize
        output_file (str): Output file path
        min_lat (float): Minimum latitude to include (-60 by default to exclude Antarctica)
    """
    try:
        # Filter out southern polar regions
        #gdf = gdf[gdf.geometry.centroid.y > min_lat].copy()
        gdf = gdf[gdf.geometry.apply(lambda geom: geom.centroid.y) > min_lat].copy()
        # Set up figure with Robinson projection for better global view
        fig, ax = plt.subplots(1, 1, figsize=(18, 12))
        ax.set_aspect('equal')

        # Calculate deciles (using your custom bins)
        carbon_data = gdf[f'carbon_{year}']
        custom_bins = np.array([-np.inf, 0, 100, 500, 1000, 2500, 5000,
                                7500, 10000, 25000, 50000, np.inf])
        bin_labels = [
            "0", "0–100", "100–500", "500–1,000", "1,000–2,500",
            "2,500–5,000", "5,000–7,500", "7,500–10,000", "10,000–25,000", "25,000–50,000", "50,000+"
        ]

        # Create categorical column
        categorized = pd.cut(
            carbon_data,
            bins=custom_bins,
            labels=bin_labels,
            include_lowest=True
        )

        # Create colormap - using Viridis for better color distinction
        colors = ['lightgrey'] + list(plt.cm.viridis_r(np.linspace(0, 1, len(bin_labels)-1)))
        cmap = ListedColormap(colors)

        # Plot with improved styling
        plot = gdf.plot(
            column=categorized,
            cmap=cmap,
            categorical=True,
            legend=True,
            ax=ax,
            legend_kwds={
                'title': "Carbon Storage (MtC)",
                'loc': 'lower center',
                'bbox_to_anchor': (0.5, -0.1),  # Centered below plot
                'ncol': 11,
                'frameon': False,
                'fontsize': 10,
                'title_fontsize': 12
            },
            edgecolor='white',
            linewidth=0.1,
            missing_kwds={
                'color': 'lightgrey',
                'label': 'No data'
            }
        )

        # Style improvements
        title = f"Global Carbon Storage ({year})"
        ax.set_title(title, fontsize=16, pad=20, fontweight='bold')
        ax.set_axis_off()

        # Set map bounds to exclude polar regions
        ax.set_ylim(min_lat, 90)

        # Add subtle grid lines
        ax.grid(True, linestyle=':', alpha=0.3)

        # Add ocean background
        ax.set_facecolor('#e6f3f7')

        plt.tight_layout()
        plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Map saved to {output_file}°)")

    except Exception as e:
        print(f"Error creating map: {str(e)}")

for col in carbon_storage_results.columns:
    if re.match(r"^carbon_\d{4}$", col):
        year = col.split("_")[1]  # Extract the year
        output_file = os.path.join(output_dir, f"carbon_storage_results/carbon_storage_{year}.png")
        print(f"Generating map for {year}...")
        try:
            plot_carbon_map(carbon_storage_results, year=year, output_file=output_file)
        except Exception as e:
            print(f"Failed for {year}: {str(e)}")

#%% quantification



for col in carbon_storage_results.columns:
    if re.match(r"^carbon_\d{4}$", col):
        year = col.split("_")[1]  # Extract the year


carbon_storage_results_country = (
    carbon_storage_results
    .groupby("gtapv7_r251_id", as_index=False)
    .agg(lambda x: x.sum() if pd.api.types.is_numeric_dtype(x) else x.iloc[0])
)

carbon_cols = [col for col in carbon_storage_results_country.columns if re.match(r"^carbon_\d{4}$", col)]
carbon_cols = sorted(carbon_cols, key=lambda x: int(x.replace("carbon_", "")))
other_cols = [col for col in carbon_storage_results_country.columns if col not in carbon_cols]

# Create a row with the sum of all carbon columns
total_row = {col: carbon_storage_results_country[col].sum() for col in carbon_cols}

for col in carbon_storage_results_country.columns:
    if col not in carbon_cols:
        total_row[col] = "World"

# Append to DataFrame
carbon_storage_results_country = pd.concat(
    [carbon_storage_results_country, pd.DataFrame([total_row])],
    ignore_index=True
)

# Reorder the DataFrame columns
carbon_storage_results_country = carbon_storage_results_country[other_cols + carbon_cols]

carbon_storage_results_country.to_excel(os.path.join(output_dir, "carbon_storage_results_country.xlsx"))

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})

# Create figure
plt.figure(figsize=(12, 7))

# Main plot line
main_line = plt.plot(
    years,
    world_values,
    color='#006400',  # Dark green
    marker='o',
    markersize=8,
    linewidth=2.5,
    label='Global Carbon Storage',
    zorder=1
)

# Highlight 2019 value
year_idx = years.index(2019)  # Find index of 2019
highlight = plt.scatter(
    [years[year_idx]],
    [world_values[year_idx]],
    color='#006400',
    s=120,
    zorder=20,
    label='2019'
)

# Add value annotation
plt.annotate(
    f'2019: {world_values[year_idx]} GtC',
    xy=(years[year_idx], world_values[year_idx]),
    xytext=(-100, 50),
    textcoords='offset points',
    fontsize=14,
    bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.9),
    arrowprops=dict(arrowstyle='-')
)

# Styling
plt.xlabel("Year")
plt.ylabel("Carbon Storage (GtC)")
plt.title("Global Carbon Storage Over Time (1992-2020)", pad=20)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')

# Adjust layout and save
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, 'carbon_storage_trend.png'),
    dpi=300,
    bbox_inches='tight',
    facecolor='white'  # Optional: set background color
)
plt.close()  # Prevents double-display in notebooks

#%%


def save_interactive_map(gpkg_data, year, output_html="carbon_map.html", min_lat=-60):
    """Generates an interactive Leaflet map with tooltip, excluding extreme southern latitudes."""

    # Filter out features below min_lat (safely using representative points)
    filtered_gdf = gpkg_data.copy()
    mask = filtered_gdf.geometry.representative_point().y > min_lat
    filtered_gdf = filtered_gdf[mask]

    # Create map without online tiles to avoid timeout
    m = folium.Map(location=[20, 0], zoom_start=2, tiles='OpenStreetMap')

    # Add GeoJson layer with tooltip
    geojson_layer = folium.GeoJson(
        filtered_gdf,
        name='Carbon Storage',
        tooltip=folium.features.GeoJsonTooltip(
            fields=['ee_r264_name', f'carbon_{year}'],
            aliases=['Country:', 'Carbon (MtC):'],
            localize=True
        )
    )
    geojson_layer.add_to(m)

    # Add choropleth
    folium.Choropleth(
        geo_data=filtered_gdf,
        name='Carbon Storage Choropleth',
        data=filtered_gdf,
        columns=['id', f'carbon_{year}'],
        key_on='feature.properties.id',
        fill_color='YlGnBu',
        nan_fill_color='lightgray',
        legend_name=f'Carbon Storage (MtC) {year}'
    ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(output_html)
    print(f"✅ Interactive map saved to {output_html}")

# Usage
save_interactive_map(carbon_storage_results, year="2019")




#%% calculate GEP

#scc
SCC = pd.read_excel(os.path.join(input_dir, "SCC.xlsx"))

SCC.head()

# Set figure size
plt.figure(figsize=(12, 8))

# Extract x-axis (year)
x = SCC['year']

# Plot each SCC column
for col in SCC.columns[1:]:  # Skip the 'year' column
    plt.plot(x, SCC[col], label=col)

# Labels and title
plt.xlabel("Year")
plt.ylabel("Social Cost of Carbon (SCC)")
plt.title("Social Cost of Carbon Over Time")
plt.legend(loc="upper left", fontsize=8)
plt.grid(True)
plt.tight_layout()

plt.savefig("scc_trends.png", dpi=300, bbox_inches='tight')


# Define column groupings
year = SCC.iloc[:, 0]
line_df = SCC.iloc[:, 3:8]  # Columns 4–8 (lines)
bar_df = SCC.iloc[:, 8:10]  # Columns 9–10 (bars)

fig, ax1 = plt.subplots(figsize=(14, 6))

# Plot lines on the left y-axis
for col in line_df.columns:
    ax1.plot(year, line_df[col], label=col, linewidth=2)
ax1.set_xlabel("Year")
ax1.set_ylabel("Line Series (SCC)")
ax1.tick_params(axis='y')

# Plot bars on the right y-axis
ax2 = ax1.twinx()
width = 0.35
x = range(len(year))

# Bar offsets
bar1 = bar_df.columns[0]
bar2 = bar_df.columns[1]

ax2.bar([i - width/2 for i in x], bar_df[bar1], width=width, label=bar1, alpha=0.6)
ax2.bar([i + width/2 for i in x], bar_df[bar2], width=width, label=bar2, alpha=0.6)
ax2.set_ylabel("Bar Series (SCC)")
ax2.tick_params(axis='y')

# X-axis ticks
ax1.set_xticks(x)
ax1.set_xticklabels(year, rotation=45)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

plt.title("SCC Trends: Lines (Cols 4–8) & Bars (Cols 9–10)")
plt.tight_layout()

plt.savefig("scc_metics.png", dpi=300, bbox_inches='tight')



#%%

import matplotlib.pyplot as plt
import pandas as pd

# Assuming SCC is your DataFrame
year = SCC.iloc[:, 0]
line_df = SCC.iloc[:, 3:8]  # Columns 4-8 (lines)
bar_df = SCC.iloc[:, 8:10]  # Columns 9-10 (bars)

# Create figure and primary axis
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot line charts on primary axis
for column in line_df.columns:
    ax1.plot(year, line_df[column], marker='o', label=column)

# Set labels for primary axis
ax1.set_xlabel('Year')
ax1.set_ylabel('Line Values')
ax1.set_title('Line and Bar Chart Comparison')
ax1.grid(True)

# Create secondary axis for bar charts
ax2 = ax1.twinx()

# Plot bar charts on secondary axis
width = 0.35  # width of the bars
bar_positions = range(len(year))
bar1 = ax2.bar([p - width/2 for p in bar_positions], bar_df.iloc[:, 0], width, label=bar_df.columns[0])
bar2 = ax2.bar([p + width/2 for p in bar_positions], bar_df.iloc[:, 1], width, label=bar_df.columns[1])

# Set labels for secondary axis
ax2.set_ylabel('Bar Values')

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# Set x-axis ticks to show all years
plt.xticks(bar_positions, year)

# Adjust layout to prevent clipping
plt.tight_layout()
plt.savefig("scc_metics.png", dpi=300, bbox_inches='tight')



























#%% comparisions


def compute_raster_gap(a_path, b_path, out_path=None):
    # === Generate output path if not provided ===
    if out_path is None:
        a_name = os.path.splitext(os.path.basename(a_path))[0]
        b_name = os.path.splitext(os.path.basename(b_path))[0]
        out_name = f"gap_{a_name}_minus_{b_name}.tif"
        out_path = os.path.join(output_dir, out_name)

    with rasterio.open(a_path) as a_src, rasterio.open(b_path) as b_src:
        # Safety checks
        assert a_src.shape == b_src.shape, "Rasters must have same shape"
        assert a_src.transform == b_src.transform, "Rasters must be spatially aligned"
        assert a_src.crs == b_src.crs, "CRS mismatch"

        # Get nodata or fallback
        a_nodata = a_src.nodata
        b_nodata = b_src.nodata

        # Prepare output metadata
        profile = a_src.profile.copy()
        profile.update(dtype="float32", nodata=np.nan)

        with rasterio.open(out_path, "w", **profile) as out_dst:
            for ji, window in tqdm(list(a_src.block_windows()), desc="Computing gap"):
                a_block = a_src.read(1, window=window).astype("float32")
                b_block = b_src.read(1, window=window).astype("float32")

                # Handle nodata
                a_block[a_block == a_nodata] = np.nan
                b_block[b_block == b_nodata] = np.nan

                # Compute gap
                gap_block = a_block - b_block
                gap_block[np.isnan(a_block) | np.isnan(b_block)] = np.nan

                out_dst.write(gap_block, 1, window=window)

    print(f"✅ Gap raster saved to: {out_path}")
    return out_path


a_path= os.path.join(output_dir, "carbon_storage_SG_direct_2020.tif")

b_path= os.path.join(output_dir, "carbon_storage_ipcc_2020.tif")

# === Example usage ===
compute_raster_gap(
    a_path, b_path
)


a_path= os.path.join(output_dir, "carbon_storage_SG_direct_2010.tif")

b_path= os.path.join(output_dir, "carbon_storage_ipcc_2010.tif")

# === Example usage ===
compute_raster_gap(
    a_path, b_path
)


a_path= os.path.join(output_dir, "carbon_storage_SG_direct_2010.tif")

b_path= os.path.join(output_dir, "total_carbon_density_SG2010_aligned.tif")

# === Example usage ===
compute_raster_gap(
    a_path, b_path
)


a_path= os.path.join(output_dir, "carbon_storage_SG_direct_2020.tif")

b_path= os.path.join(output_dir, "carbon_storage_SG_direct_2010.tif")

# === Example usage ===
compute_raster_gap(
    a_path, b_path
)


a_path= os.path.join(output_dir, "carbon_storage_ipcc_2020.tif")

b_path= os.path.join(output_dir, "carbon_storage_ipcc_2010.tif")

# === Example usage ===
compute_raster_gap(
    a_path, b_path
)



a_path= os.path.join(output_dir, "gap_carbon_storage_SG_direct_2020_minus_carbon_storage_SG_direct_2010.tif")

b_path= os.path.join(output_dir, "gap_carbon_storage_ipcc_2020_minus_carbon_storage_ipcc_2010.tif")

# === Example usage ===
compute_raster_gap(
    a_path, b_path
)




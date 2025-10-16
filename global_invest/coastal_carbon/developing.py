import rasterio
import numpy as np
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterio import features
from shapely.geometry import shape
import os
from rasterstats import zonal_stats
from tqdm import tqdm

def lulc_raster_to_gdf(input_dir, target_code=186, output_gdf_path=None):
    """
    Convert all LULC rasters to a single GeoDataFrame, keeping only target code pixels
    """
    gdf_list = []
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.tif'):
            input_path = os.path.join(input_dir, filename)
            print(f"Processing: {filename}")
            
            with rasterio.open(input_path) as src:
                # Read the data
                data = src.read(1)
                
                # Create mask for target code
                mask = data == target_code
                
                if np.any(mask):  # Only process if target code exists
                    # Get transform and CRS
                    transform = src.transform
                    crs = src.crs
                    
                    # Extract geometries and values where mask is True
                    shapes = features.shapes(data, mask=mask, transform=transform)
                    
                    # Create GeoDataFrame for this tile
                    geometries = []
                    values = []
                    
                    for geom, value in shapes:
                        if value == target_code:  # Double check
                            geometries.append(shape(geom))
                            values.append(value)
                    
                    if geometries:  # If we have any features
                        tile_gdf = gpd.GeoDataFrame({
                            'lulc_code': values,
                            'tile_name': filename,
                            'geometry': geometries
                        }, crs=crs)
                        
                        gdf_list.append(tile_gdf)
    
    # Combine all tiles into one GeoDataFrame
    if gdf_list:
        combined_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))
        
        # Save if path provided
        if output_gdf_path:
            combined_gdf.to_file(output_gdf_path, driver='GPKG')
            print(f"Saved combined GDF to: {output_gdf_path}")
        
        return combined_gdf
    else:
        print("No target code found in any tiles")
        return gpd.GeoDataFrame()

input_dir = "/Users/long/Library/CloudStorage/GoogleDrive-yxlong@umn.edu/Shared drives/NatCapTEEMs/Files/base_data/submissions/coastal_carbon/test"
output_gdf_path = ("/Users/long/Library/CloudStorage/GoogleDrive-yxlong@umn.edu/Shared "
                   "drives/NatCapTEEMs/Files/base_data/submissions/coastal_carbon/test.gpkg")
lulc_raster_to_gdf(input_dir, target_code=186, output_gdf_path=output_gdf_path)




global_salt_marsh2019_path = "/Users/long/Library/CloudStorage/GoogleDrive-yxlong@umn.edu/Shared drives/NatCapTEEMs/Files/base_data/submissions/coastal_carbon/global_salt_marsh2019.gpkg"

# Read the first (default) layer
gdf = gpd.read_file(global_salt_marsh2019_path)

# View basic info
print(gdf.head())
print(gdf.crs)


ha_per_cell_path = ("/Users/long/Library/CloudStorage/GoogleDrive-yxlong@umn.edu/Shared "
                    "drives/NatCapTEEMs/Files/base_data/pyramids/ha_per_cell_1sec.tif")

with rasterio.open(ha_per_cell_path) as src:
    gdf = gdf.to_crs(src.crs)


stats_list = []
for i, geom in enumerate(tqdm(gdf.geometry, desc="Computing zonal stats")):
    stats = zonal_stats(
        vectors=[geom],
        raster=ha_per_cell_path,
        stats=['sum'],
        geojson_out=True,
        nodata=-9999.0,
        all_touched=True
    )
    stats_list.extend(stats)

gdf_zonal = gpd.GeoDataFrame.from_features(stats_list)
gdf_zonal.set_crs("EPSG:4326", inplace=True)
gdf_zonal = gdf_zonal.to_crs(src.crs)
output_path = ("/Users/long/Library/CloudStorage/GoogleDrive-yxlong@umn.edu/Shared "
               "drives/NatCapTEEMs/Files/base_data/submissions/coastal_carbon/global_salt_marsh2019.gpkg")
gdf_zonal.to_file(output_path, driver="GPKG")

print(f"✅ Saved zonal stats to: {output_path}")

gdf_zonal.rename(columns={'sum': 'salt_marsh_area_ha'}, inplace=True)


global_salt_marsh2019 = gpd.GeoDataFrame.from_features(stats)



global_salt_marsh_ha2019 = gdf_zonal

gdf_zonal = gdf_zonal.rename(columns={'sum': 'area_ha'})


global_salt_marsh2019 = gpd.read_file(global_salt_marsh2019_path)
gdf_countries_marine_vector = gpd.read_file(p.gdf_countries_marine_vector_path)  # e.g., Natural Earth admin boundaries
output_path = os.path.join("/Users/long/Library/CloudStorage/GoogleDrive-yxlong@umn.edu/Shared "
                           "drives/NatCapTEEMs/Files/base_data/submissions/coastal_carbon","global_salt_marsh_within_countries.gpkg")

intersect_list = []

# --- Step 3: Loop through each country polygon with progress bar ---
for idx, country in tqdm(
        gdf_countries_marine_vector.iterrows(),
        total=len(gdf_countries_marine_vector),
        desc="Intersecting countries"
):
    # Create single-row GeoDataFrame for this country
    country_gdf = gpd.GeoDataFrame([country], crs=gdf_countries_marine_vector.crs)

    # Perform intersection (keep only overlapping parts)
    clipped = gpd.overlay(global_salt_marsh2019, country_gdf, how="intersection")

    if clipped.empty:
        continue  # skip if no overlap

    if 'eemarine_r566_id' in gdf_countries_marine_vector.columns:
        clipped['eemarine_r566_id'] = country['eemarine_r566_id']

    # Store this batch
    intersect_list.append(clipped)

# --- Step 4: Combine all intersections into one GeoDataFrame ---
if len(intersect_list) > 0:
    intersected = pd.concat(intersect_list, ignore_index=True)
    intersected = gpd.GeoDataFrame(intersected, crs=gdf_countries_marine_vector.crs)
else:
    raise ValueError("No intersections found — check layer overlap or CRS.")

# --- Step 5: Save final result as GeoPackage ---
intersected.to_file(output_path, driver="GPKG")

print(f"✅ Done! Saved intersected salt marsh polygons to:\n{output_path}")

output_path = os.path.join("/Users/long/Library/CloudStorage/GoogleDrive-yxlong@umn.edu/Shared "
                           "drives/NatCapTEEMs/Files/base_data/submissions/coastal_carbon",
                           "global_salt_marsh_within_countries.csv")


intersected.to_csv(output_path)



global_salt_marsh_ha2019 = global_salt_marsh_ha2019.to_crs(gdf_countries_marine_vector.crs)
global_salt_marsh_ha2019 = global_salt_marsh_ha2019.rename(columns={'sum': 'area_ha'})

intersected = gpd.sjoin(
    global_salt_marsh_ha2019,
    gdf_countries_marine_vector,
    how='inner',
    predicate='intersects'
)

intersected = gpd.overlay(gdf_zonal, gdf_countries_marine_vector, how='intersection')

country_col = 'SOVEREIGN1'  # change this to your identifier

area_by_country = (
    intersected.groupby(country_col, as_index=False)['area_ha']
    .sum()
    .rename(columns={'area_ha': 'salt_marsh_area_ha'})
)


gdf_countries_marine_vector_ex_hs = gdf_countries_marine_vector[gdf_countries_marine_vector]
# --- Step 4: Aggregate by country ---
country_sum = intersected.dissolve(by='NAME', aggfunc={'weighted_sum': 'sum'})  # or 'ADMIN', 'CNTRY_NAME', etc.

# --- Step 5: Clean up results ---
country_sum.rename(columns={'weighted_sum': 'total_raster_sum'}, inplace=True)
country_sum = country_sum.reset_index()

print(country_sum[['NAME', 'total_raster_sum']].head())


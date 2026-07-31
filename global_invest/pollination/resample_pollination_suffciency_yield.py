import os
import pygeoprocessing

"D:\Shared drives\NatCapTEEMs\Projects\Global GEP\Ecosystem Services SubFolders\Pollination\poll_dep_greater_0.xlsx"


# File paths
apple_production_raster = r"D:\Shared drives\NatCapTEEMs\Projects\Global GEP\Ecosystem Services SubFolders\Pollination\HarvestedAreaYield175Crops_Geotiff\HarvestedAreaYield175Crops_Geotiff\GeoTiff\apple\apple_Production.tif"
poll_suff_raster = r"D:\Shared drives\NatCapTEEMs\Projects\Global GEP\Ecosystem Services SubFolders\Pollination\poll_suff_ag_coverage_prop_lulc_esa_gtap1_baseline_2014.tif"

# Aligned output paths
aligned_poll_suff_raster = r"D:\Shared drives\NatCapTEEMs\Projects\Global GEP\Ecosystem Services SubFolders\Pollination\aligned_poll_suff_to_apple.tif"

# Get the target pixel size and bounding box from the apple production raster
target_pixel_size = pygeoprocessing.get_raster_info(apple_production_raster)['pixel_size']
target_bb = pygeoprocessing.get_raster_info(apple_production_raster)['bounding_box']

# Align and resize the pollination sufficiency raster to match the apple production raster
pygeoprocessing.warp_raster(
    poll_suff_raster,  # Input raster to resample
    target_pixel_size,  # Target pixel size (from the apple production raster)
    aligned_poll_suff_raster,  # Output aligned raster path
    'average',  # Resampling method (mean)
    target_bb  # Target bounding box (from the apple production raster)
)

print("Resampling and alignment completed.")

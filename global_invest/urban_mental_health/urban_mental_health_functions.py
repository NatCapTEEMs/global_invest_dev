import os
import rasterio
from rasterstats import zonal_stats
from rasterio.enums import Resampling
import rasterio.warp
import numpy as np
from tqdm import tqdm
import gc
import pandas as pd
import geopandas as gpd
from global_invest.carbon import carbon_functions


def resample_raster_to_reference(
        input_path,
        reference_path,
        out_path,
        resampling_method=Resampling.nearest,
        compress="lzw"
        ):
    """
    Resample input raster to match the resolution and grid of reference raster.

    The function handles the downsampling of LULC data to ensure consistent pixel
    grids for overlay analysis.
    """
    with rasterio.open(reference_path) as ref:
        ref_profile = ref.profile.copy()
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_width = ref.width
        ref_height = ref.height

    with rasterio.open(input_path) as src:
        # Resample data to match reference grid
        data = src.read(
            out_shape=(src.count, ref_height, ref_width),
            resampling=resampling_method
        )

        # Update profile for output
        profile = src.profile.copy()
        profile.update({
            'height': ref_height,
            'width': ref_width,
            'transform': ref_transform,
            'crs': ref_crs,
            'compress': compress
        })

        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(data)

    print(f"Resampled raster saved to: {out_path}")
    print(f"Original resolution: {src.res}, Target resolution: {ref.res}")


def map_lulc_to_ndvi(lulc_path, attr_table_path, ndvi_col, out_path, compress="lzw"):
    """
    Map LULC raster codes to NDVI values using attribute table.
    """
    # Read attribute table
    df = pd.read_csv(attr_table_path)

    # Create lookup dictionary with NaN handling for unmapped codes
    lc_to_ndvi = dict(zip(df['lc_code'], df[ndvi_col]))

    with rasterio.open(lulc_path) as src:
        profile = src.profile.copy()
        profile.update(dtype='float32', nodata=np.nan, compress=compress)

        with rasterio.open(out_path, 'w', **profile) as dst:
            windows = list(src.block_windows(1))
            for idx, window in tqdm(windows, desc="Converting LULC to NDVI"):
            #for idx, window in windows:
                arr = src.read(1, window=window)

                # Map LULC codes to NDVI values, handling unmapped codes as NaN
                ndvi_arr = np.vectorize(lambda x: lc_to_ndvi.get(x, np.nan), otypes=[float])(arr)  # np.vectorize for better performance with large arrays thus avoiding the need for a loop over each pixel

                dst.write(ndvi_arr.astype('float32'), 1, window=window)

    #print(f"NDVI raster saved to: {out_path}")


def calculate_delta_raster(
        raster1_path,
        raster2_path,
        out_path,
        operation,
        fill_value=np.nan,
        compress="lzw"
        ):
    """
    Calculate difference between two rasters (e.g., NDVI_2019 - NDVI_2010).
    """
    with rasterio.open(raster1_path) as src1, rasterio.open(raster2_path) as src2:
        if src1.shape != src2.shape:
            raise ValueError("Input rasters must have the same dimensions.")

        profile = src1.profile.copy()
        profile.update(dtype='float32', nodata=fill_value, compress=compress)

        with rasterio.open(out_path, "w", **profile) as dst:
            windows = list(src1.block_windows(1))
            #for _, window in windows:
            for _, window in tqdm(windows, desc="Calculating delta raster"):
                arr1 = src1.read(1, window=window)
                arr2 = src2.read(1, window=window)

                # Handle NaN values properly
                result = operation(arr1, arr2)
                result = np.where(np.isnan(arr1) | np.isnan(arr2), fill_value, result)

                dst.write(result.astype("float32"), 1, window=window)

    #print(f"Delta raster saved to: {out_path}")


def calculate_preventable_cases(
        delta_ne_path,
        pop_path,
        effect_size_table_path,
        prevalence,
        out_path,
        compress="lzw"
        ):
    """
    Calculate preventable cases per pixel using delta nature exposure and population.
    """
    # Read effect size table
    df = pd.read_excel(effect_size_table_path)

    # Get relative risk per unit NDVI
    rr = float(df.loc[df['health_indicator'] == 'depression', 'effect_size'].iloc[0])
    #rr = float(df[df['health_indicator'] == 'depression']['effect_size'].iloc[0])

    with rasterio.open(delta_ne_path) as dsrc, rasterio.open(pop_path) as psrc:
        if dsrc.shape != psrc.shape:
            raise ValueError("Delta NE and population rasters must have the same dimensions.")

        profile = dsrc.profile.copy()
        profile.update(dtype='float32', nodata=np.nan, compress=compress)

        with rasterio.open(out_path, 'w', **profile) as dst:
            windows = list(dsrc.block_windows(1))
            for _, window in tqdm(windows, desc="Calculating preventable cases"):
                delta_ne = dsrc.read(1, window=window)
                pop = psrc.read(1, window=window)

                # Calculate preventable fraction
                pf = 1 - np.power(rr, 10*delta_ne)

                # Calculate baseline cases and preventable cases
                bc = prevalence * pop  # Baseline cases per pixel
                preventable_cases = bc * pf  # Preventable cases per pixel

                # Handle NaN values
                preventable_cases = np.where(np.isnan(delta_ne) | np.isnan(pop), np.nan, preventable_cases)

                dst.write(preventable_cases.astype('float32'), 1, window=window)

    #print(f"Preventable cases raster saved to: {out_path}")


def aggregate_preventable_cases_by_region(
        preventable_cases_raster_path,
        urban_region_boundary_path,
        out_csv_path
        ):
    """
    Aggregate preventable cases by urban regions.
    """
    # Load urban boundaries
    urban_boundaries = gpd.read_file(urban_region_boundary_path)

    with rasterio.open(preventable_cases_raster_path) as src:
        raster_crs = src.crs
        print(f"Raster CRS: {raster_crs}")

        # Reproject vector data if needed
        if urban_boundaries.crs != raster_crs:
            print(f"Reprojecting vector data from {urban_boundaries.crs} to match raster CRS {raster_crs}")
            urban_boundaries = urban_boundaries.to_crs(raster_crs)

        # Calculate pixel area for area calculations
        x_res, y_res = src.res
        is_geographic = raster_crs.is_geographic if raster_crs else False  # geographic vs projected

        if is_geographic:
            pixel_area_m2 = (111_320 ** 2) * abs(x_res * y_res)  # circumference of Earth/360 = ~111.32 km per degree
        else:
            pixel_area_m2 = abs(x_res * y_res)

        # Calculate zonal statistics for all polygons at once
        print("Calculating zonal statistics...")
        zs = zonal_stats(
        urban_boundaries,
        preventable_cases_raster_path,
        stats=['count', 'sum', 'mean', 'min', 'max'],
        nodata=np.nan,
        all_touched=True
        )

        results = []

        # Process results with progress bar
        #for idx, (row, stats) in enumerate(zip(urban_boundaries.iterrows(), zs)):
        for idx, (row, stats) in enumerate(tqdm(zip(urban_boundaries.iterrows(), zs), 
                                          total=len(urban_boundaries), 
                                          desc="Processing zonal statistics")):
            _, row = row  # unpack the iterrows tuple

            # Skip regions with no valid pixels
            if stats['count'] == 0 or stats['count'] is None:
                continue

            # Calculate area from pixel count
            area_m2 = stats['count'] * pixel_area_m2

            # Compile statistics
            region_stats = {
            "region_id": row.get("id", idx),
            "country": row.get("country", None),
            "total_preventable_cases": stats['sum'] if stats['sum'] is not None else 0,
            "mean_cases_per_pixel": stats['mean'] if stats['mean'] is not None else 0,
            "min_cases_per_pixel": stats['min'] if stats['min'] is not None else 0,
            "max_cases_per_pixel": stats['max'] if stats['max'] is not None else 0,
            "count": stats['count'],
            "area_m2": area_m2,
            "area_ha": area_m2 / 10_000,
            "area_km2": area_m2 / 1_000_000
            }

            results.append(region_stats)

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(out_csv_path, index=False)
    print(f"Regional preventable cases summary written to: {out_csv_path}")

    # Return CSV path
    return out_csv_path


def apply_country_costs(
        regional_cases_csv_path,
        health_cost_rate_path,
        out_country_csv_path
        ):
    """
    Apply country-specific cost rates to aggregated preventable cases.
    """
    # Read input data
    regional_df = pd.read_csv(regional_cases_csv_path)
    cost_df = pd.read_excel(health_cost_rate_path)

    # Filter regions by countries
    filtered = regional_df[regional_df['country'].isin(cost_df['country'])]

    # Sum preventable cases for each country
    country_sums = filtered[['country', 'total_preventable_cases']].groupby('country').sum()

    # Merge summed cases with per-patient costs
    merged = pd.merge(country_sums, cost_df, on='country')

    # Calculate cost savings
    merged['cost_savings'] = merged['total_preventable_cases'] * merged['cost_per_case']

    # Write output CSV with desired columns
    merged[['country', 'total_preventable_cases', 'cost_savings']].to_csv('country_cost_savings.csv', index=False)
    print(f"Country-level costs saved to: {out_country_csv_path}")

    return out_country_csv_path
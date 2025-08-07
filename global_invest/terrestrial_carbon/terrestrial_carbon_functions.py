# =============================================================================
# imports
# =============================================================================
import os
import re
import rasterio
from rasterio.mask import mask
from rasterio.enums import Resampling
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


# =============================================================================
# define functions
# =============================================================================

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

        x_res, y_res = src.res
        is_geographic = raster_crs.is_geographic if raster_crs else False

        # Calculate pixel area
        if is_geographic:
            pixel_area_m2 = (111_320 ** 2) * abs(x_res * y_res)
        else:
            pixel_area_m2 = abs(x_res * y_res)

        results = []

        for idx, row in tqdm(regions.iterrows(), total=len(regions), desc="Summarizing polygons"):
            geom = [row.geometry]

            try:
                masked, _ = mask(src, geom, crop=True, nodata=np.nan, all_touched=True)
                values = masked[0]
                values = values[~np.isnan(values)]

                if values.size == 0:
                    continue

                area_m2 = values.size * pixel_area_m2

                stats = {
                    "region_id": row.get("id", idx),
                    "mean": values.mean(),
                    "min": values.min(),
                    "max": values.max(),
                    "count": values.size,
                    "area_m2": area_m2,
                    "area_ha": area_m2 / 10_000,
                    "area_km2": area_m2 / 1_000_000,
                }
                results.append(stats)

            except Exception as e:
                print(f"Error processing region {idx}: {e}")
                continue

    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"Summary written to: {out_path}")

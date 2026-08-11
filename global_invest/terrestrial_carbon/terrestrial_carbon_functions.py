# =============================================================================
# imports
# =============================================================================
import os
import gc
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import rioxarray as rxr
import dask.array as da
import dask.dataframe as dd
from tqdm import tqdm

# =============================================================================
# define functions
# =============================================================================
# The raw-Spawn density build (uint -> float32 scaling, aboveground+belowground add) is a one-off
# base-data job, not part of the per-run tree; it lives in howto/rebuild_spawn_total_carbon_density.md
# and the run consumes its finished product (spawn_total_biomass_carbon_2010.tif) from base_data.

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

        sliced_layers = [layer.isel(x=slice(x_start, x_end)) for layer in layers]
        flattened = [sl.values.reshape(-1).astype("float32") for sl in sliced_layers]

        if len(set(arr.shape[0] for arr in flattened)) != 1:
            raise ValueError("Slice %d: the grouping and value layers have mismatched pixel counts %s -- "
                             "the three rasters are not on the same grid." % (i + 1, [arr.shape[0] for arr in flattened]))

        stacked = da.stack(flattened, axis=1)
        df = dd.from_dask_array(stacked, columns=layer_names)
        df_pd = df.compute().dropna(subset=group_cols + [value_col])

        if df_pd.empty:   # slice is all-nodata -- nothing to summarize, expected at raster edges
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
        Path to a LONG/TIDY CSV with one row per (carbon_zone_id, lulc_id) and a single
        carbon_density_mean value column. NOT a wide table indexed by carbon_zone_id with
        one column per LULC type: the lookup below filters on a `lulc_id` COLUMN and reads
        `carbon_density_mean`, so a wide table matches nothing and, because an empty match
        is skipped rather than raised, yields an all-NoData raster instead of an error.
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
                lulc_block = lulc_src.read(1, window=window)
                carbon_zone_block = cz_src.read(1, window=window)

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

def summarize_raster_by_region(value_raster_path, region_boundary_path, out_path, year):
    """
    Summarize a value raster by the polygon regions of a vector file (e.g. GPKG): per-region
    mean, min, max, pixel count, and sum of the raster values.

    Parameters
    ----------
    value_raster_path : str
        Path to the value raster (e.g. carbon density).
    region_boundary_path : str
        Path to the vector file (GeoPackage) containing the polygon regions.
    out_path : str
        Output path for the CSV summary.
    year : int
        The year the value raster represents; written to the `year` column (base year for a GEP run).
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
            id_list.append(row.get("id", idx))
            masked, _ = mask(src, [row.geometry], crop=True, nodata=np.nan, all_touched=True)
            values = masked[0][~np.isnan(masked[0])]
            if values.size == 0:   # zone's raster window is all-nodata -> drop it (see key note below)
                continue
            results.append({
                "index_id": row.get("id", idx),
                "mean": values.mean(),
                "min": values.min(),
                "max": values.max(),
                "count": values.size,
                "total": values.sum(),
            })
    regions["index_id"] = id_list
    df = pd.DataFrame(results)
    df = regions.merge(df, on="index_id", how="right")
    df = df.drop(columns=["index_id", "geometry"])
    df['year'] = year
    # Stable key. Zones whose raster window is empty are dropped above, so row POSITION in this CSV
    # does not correspond to row position in the boundary file, and anything aligning on position
    # silently pairs the wrong zones. Emit the boundary's own id so consumers can join on a value.
    if 'ee_r50_aez18_id' in df.columns:
        df['region_id'] = df['ee_r50_aez18_id'].astype(int)
    df.to_csv(out_path, index=False)
    print(f"Summary written to: {out_path}")

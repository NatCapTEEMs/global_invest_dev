# =============================================================================
# imports
# =============================================================================
import os
import gc
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import dask.array as da
import dask.dataframe as dd
from tqdm import tqdm
import hazelbean as hb

# =============================================================================
# define functions
# =============================================================================
# The raw-Spawn density build (uint -> float32 scaling, aboveground+belowground add) is a one-off
# base-data job, not part of the per-run tree; it lives in howto/rebuild_spawn_total_carbon_density.md
# and the run consumes its finished product (spawn_total_biomass_carbon_2010.tif) from base_data.

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
    # (carbon_zone_id, lulc_id) -> carbon_density_mean. The ids are whole numbers (stored as float in the
    # table, uint in the rasters), so key on int64 to match by value; a pair absent from the table
    # reindexes to NaN, exactly the old "no match -> leave NoData" behaviour.
    lut = pd.read_csv(carbon_density_lookup_table_path, index_col=False)
    lut = lut.astype({'carbon_zone_id': 'int64', 'lulc_id': 'int64'}).set_index(
        ['carbon_zone_id', 'lulc_id'])['carbon_density_mean'].astype('float32')

    def carbon_density(lulc_block, cz_block):
        idx = pd.MultiIndex.from_arrays([cz_block.astype('int64').ravel(), lulc_block.astype('int64').ravel()])
        return lut.reindex(idx).to_numpy('float32').reshape(lulc_block.shape)

    # datatype=6 (Float32) + ndv=NaN: without them raster_calculator_flex would inherit the LULC uint8.
    hb.raster_calculator_flex([lulc_path, cz_path], carbon_density, out_path, datatype=6, ndv=np.nan)
    print(f"Saved: {out_path}")

def summarize_raster_by_region(value_raster_path, region_boundary_path, out_path, year, id_column):
    """
    Per-polygon total / mean / pixel count of a value raster, via hb.zonal_statistics_flex.

    Parameters
    ----------
    value_raster_path : str
        Path to the value raster (per-cell carbon stock for the GEP total, carbon density for the shock).
    region_boundary_path : str
        Path to the vector (GeoPackage) of polygon regions.
    out_path : str
        Output path for the CSV summary.
    year : int
        The year the value raster represents; written to the `year` column.
    id_column : str
        The vector column holding a unique integer id per polygon to key the zonal statistics on
        ('ee_r264_id' for the country valuation, 'ee_r50_aez18_id' for the shock zones).
    """
    regions = gpd.read_file(region_boundary_path)
    # zones_raster_data_type=5 (Int32) so ids past 255 don't saturate (r264 runs to 264, r50xAEZ higher);
    # all_touched=True matches the old per-polygon masking. Returns a frame indexed by zone id, with a
    # zone 0 = everything outside every polygon.
    zone_ids_raster = os.path.splitext(out_path)[0] + '_zone_ids.tif'
    stats = hb.zonal_statistics_flex(
        value_raster_path, region_boundary_path, zone_ids_raster_path=zone_ids_raster,
        id_column_label=id_column, zones_raster_data_type=5, all_touched=True,
        stats_to_retrieve='sums_counts', assert_projections_same=False, verbose=False)
    stats = stats[(stats.index != 0) & (stats['counts'] > 0)]   # drop background + empty zones

    df = regions.assign(_zid=regions[id_column].astype('int64')).merge(
        stats, left_on='_zid', right_index=True, how='right').drop(columns=['_zid', 'geometry'])
    df = df.rename(columns={'sums': 'total', 'counts': 'count'})
    df['mean'] = df['total'] / df['count']
    df['year'] = year
    # Emit the boundary's own id so consumers join on a value, never on row position.
    if 'ee_r50_aez18_id' in df.columns:
        df['region_id'] = df['ee_r50_aez18_id'].astype(int)
    df.to_csv(out_path, index=False)
    print(f"Summary written to: {out_path}")

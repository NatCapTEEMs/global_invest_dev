# =============================================================================
# imports
# =============================================================================
import os
import numpy as np
import pandas as pd
import geopandas as gpd
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
    group1_name="group1",
    group2_name="group2",
    value_name="value"
):
    """
    Summarize a value raster by grouping over two category rasters, and write to CSV.

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
    group1_name : str
        Column name for the first group layer.
    group2_name : str
        Column name for the second group layer.
    value_name : str
        Column name for the value layer.
    """
    from osgeo import gdal
    # hb.iterblocks streams the value raster block by block; the two category rasters share its grid, so
    # read them at the same window (raw gdal -- hb has no aligned multi-raster block reader). A cell is
    # kept only where all three are valid, matching the old masked=True + dropna. Groupby per block, then
    # combine: only ~thousands of (g1, g2) pairs, so the accumulator stays tiny and nothing 33 GB is
    # written (unlike composite-key + zonal).
    ds1 = gdal.Open(group_layer1_path); g1b = ds1.GetRasterBand(1)   # hold the datasets, else the band handle dies
    ds2 = gdal.Open(group_layer2_path); g2b = ds2.GetRasterBand(1)
    ndv1, ndv2 = g1b.GetNoDataValue(), g2b.GetNoDataValue()
    dsv = gdal.Open(value_layer_path); ndv_val = dsv.GetRasterBand(1).GetNoDataValue()

    parts = []
    for offset, value_block in tqdm(hb.iterblocks((value_layer_path, 1)), desc="Summarizing blocks"):
        w = (offset['xoff'], offset['yoff'], offset['win_xsize'], offset['win_ysize'])
        v = value_block.astype('float32').ravel()
        g1 = g1b.ReadAsArray(*w).ravel()
        g2 = g2b.ReadAsArray(*w).ravel()

        keep = ~np.isnan(v)
        if ndv_val is not None: keep &= v != ndv_val
        if ndv1 is not None: keep &= g1 != ndv1
        if ndv2 is not None: keep &= g2 != ndv2
        if not keep.any():   # block is all-nodata -- expected at raster edges
            continue

        block = pd.DataFrame({group1_name: g1[keep], group2_name: g2[keep], value_name: v[keep]})
        parts.append(block.groupby([group1_name, group2_name], as_index=False)[value_name]
                     .agg(_sum='sum', _min='min', _max='max', _count='count'))

    summary = pd.concat(parts).groupby([group1_name, group2_name], as_index=False).agg(
        _sum=('_sum', 'sum'), _min=('_min', 'min'), _max=('_max', 'max'), _count=('_count', 'sum'))
    summary[f'{value_name}_mean'] = summary['_sum'] / summary['_count']
    summary = summary.rename(columns={'_min': f'{value_name}_min', '_max': f'{value_name}_max',
                                      '_count': f'{value_name}_count'})
    summary[[group1_name, group2_name, f'{value_name}_mean', f'{value_name}_min',
             f'{value_name}_max', f'{value_name}_count']].to_csv(output_path, index=False)
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

    # Tally what the lookup actually matched, so a wholesale mismatch raises instead of flowing on as an
    # all-NoData raster. The concrete trap: an ESA-classed map (ids 10..220) against the SEALS7-keyed
    # lookup (ids 1..7) matches nothing, the density raster is all NaN, the zonal means drop every zone,
    # and downstream emits an empty shock CSV -- GTAP then runs a silent zero. Per-pair misses stay NaN
    # (a legitimately absent combination is not an error); only zero matches over valid cells raises.
    lulc_ndv = hb.get_ndv_from_path(lulc_path)
    seen_lulc_ids = set()
    counts = {'valid': 0, 'matched': 0}

    def carbon_density(lulc_block, cz_block):
        idx = pd.MultiIndex.from_arrays([cz_block.astype('int64').ravel(), lulc_block.astype('int64').ravel()])
        out = lut.reindex(idx).to_numpy('float32').reshape(lulc_block.shape)
        valid = lulc_block != lulc_ndv if lulc_ndv is not None else np.ones(lulc_block.shape, dtype=bool)
        seen_lulc_ids.update(np.unique(lulc_block[valid]).tolist())
        counts['valid'] += int(valid.sum())
        counts['matched'] += int(np.isfinite(out).sum())
        return out

    # datatype=6 (Float32) + ndv=NaN: without them raster_calculator_flex would inherit the LULC uint8.
    hb.raster_calculator_flex([lulc_path, cz_path], carbon_density, out_path, datatype=6, ndv=np.nan)
    if counts['valid'] > 0 and counts['matched'] == 0:
        raise ValueError(
            f"carbon density lookup matched ZERO of {counts['valid']} valid cells: the LULC raster "
            f"{lulc_path} carries classes {sorted(seen_lulc_ids)[:20]} but the lookup "
            f"{carbon_density_lookup_table_path} is keyed on lulc_id {sorted(set(lut.index.get_level_values('lulc_id')))} "
            f"(likely an ESA-classed map fed to the SEALS7-keyed lookup, or a wrong carbon-zones raster). "
            f"Refusing to emit an all-NoData density raster.")
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

# =============================================================================
# imports
# =============================================================================
import os
import re
import time
import logging
import gc
from datetime import datetime
from typing import List, Callable, Optional, Tuple, Any, Dict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from functools import partial

import rasterio
from rasterio.mask import mask
from rasterio.enums import Resampling
import rasterio.warp
import rasterio.features
import rasterio.vrt
import numpy as np
from tqdm import tqdm
import rioxarray as rxr
import pandas as pd
import geopandas as gpd
import pyogrio
import hazelbean as hb

from global_invest.terrestrial_carbon import terrestrial_carbon_initialization

logger = logging.getLogger('carbon_pipeline')

# =============================================================================
# define functions
# =============================================================================


def setup_logging(project_dir):
    """Configure logging to both console and a timestamped log file."""
    log_dir = os.path.join(project_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'run_terrestrial_carbon_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    logging.info(f'Log file: {log_path}')
    return log_path


def _reset_task_tree(p):
    """Reset p's task tree so a fresh task set can be added without stacking."""
    from hazelbean.project_flow import Task
    p.task_tree = Task(p.execute, 'Task tree')
    p.task_names_defined = []


def _apply_dry_run(p):
    """If p.dry_run is truthy, replace each task function with a no-op so
    p.execute() walks the DAG (resolving paths, creating cur_dirs) without
    running any task body.

    Note: hazelbean's `task.run = 0` flag does NOT actually skip execution
    (it only sets p.run_this, which task functions are supposed to check
    but don't). Monkey-patching task.function is the reliable way.
    """
    if not getattr(p, 'dry_run', False):
        return
    import anytree
    n = 0
    for task in anytree.PreOrderIter(p.task_tree):
        if task is p.task_tree:
            continue
        name = task.name
        task.function = lambda p, _name=name: hb.log(f'DRY RUN: skipped {_name}')
        n += 1
    hb.log(f'DRY RUN: replaced {n} task function(s) with no-ops.')


def run_shared_preprocessing(p):
    """Run Spawn data preparation + region rasterization once. Shared across variants.

    Mutates the passed-in `p`: resets its task tree, sets variant-specific dirs
    (intermediate_dir, output_dir, variant_name, n_lat_bands), builds the
    preprocessing task tree, and executes. Paths read from `p` are not redefined.
    """
    shared_dir = os.path.join(p.project_dir, 'intermediate', 'shared')
    os.makedirs(shared_dir, exist_ok=True)

    _reset_task_tree(p)
    p.intermediate_dir = shared_dir
    p.output_dir = shared_dir
    p.n_lat_bands = 0
    p.variant_name = 'shared'
    p.results = {}

    terrestrial_carbon_initialization.build_preprocessing_task_tree(p)
    terrestrial_carbon_initialization.initialize_paths(p)

    hb.log('Running shared preprocessing (convert dtype + reproject)')
    _apply_dry_run(p)
    p.execute()


def run_variant(p, n_lat_bands, task_tree_builder, variants=None):
    """
    Run a single variant with the given task tree builder.

    Mutates the passed-in `p`: resets its task tree, sets variant-specific dirs
    and config, builds the task tree, and executes. Paths read from `p` are
    not redefined.

    Parameters
    ----------
    p : hb.ProjectFlow
        Top-level ProjectFlow with all paths/dirs already set.
    n_lat_bands : int
        0 = 2key, 18 = 3key_18bands, 36 = 3key_36bands.
    task_tree_builder : callable
        Function that takes a ProjectFlow and builds the task tree.
    variants : list[int], optional
        Full list of variants being run (for cross-variant comparison tasks).
    """
    variant_name = f'3key_{n_lat_bands}bands' if n_lat_bands else '2key'

    _reset_task_tree(p)
    p.intermediate_dir = os.path.join(p.project_dir, 'intermediate', variant_name)
    p.output_dir = p.intermediate_dir
    os.makedirs(p.intermediate_dir, exist_ok=True)

    p.n_lat_bands = n_lat_bands
    p.variant_name = variant_name
    if variants is not None:
        p.variants = list(variants)

    p.shared_intermediate_dir = os.path.join(p.project_dir, 'intermediate', 'shared')
    p.results = {}

    task_tree_builder(p)
    terrestrial_carbon_initialization.initialize_paths(p)

    hb.log(f'Running variant: {variant_name} (n_lat_bands={n_lat_bands})')
    hb.log(f'  intermediate_dir: {p.intermediate_dir}')
    hb.log(f'  shared preprocessing: {p.shared_intermediate_dir}')
    _apply_dry_run(p)
    p.execute()

    return p.results


# =============================================================================

def convert_and_reproject_raster(
        input_path,
        reference_path,
        output_path,
        scale_factor=0.1,
        compress="zstd",
):
    """
    Convert a uint raster to float32 (with scaling) and reproject to match a
    reference raster, using a temporary file and block-by-block I/O to keep
    peak memory low.

    Parameters
    ----------
    input_path : str
        Path to the input unsigned integer raster.
    reference_path : str
        Path to the reference raster whose CRS, resolution, and extent to match.
    output_path : str
        Path to write the final float32 reprojected raster.
    scale_factor : float
        Multiplier applied after uint to float conversion (e.g., 0.1).
    compress : str
        Compression for the output GeoTIFF.
    """
    import tempfile

    if os.path.exists(output_path):
        print(f"  Skipping (exists): {output_path}")
        return

    tmp_dir = os.path.dirname(output_path)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tif", dir=tmp_dir)
    os.close(tmp_fd)

    try:
        # Step 1: block-by-block uint to float32 conversion into a temp file
        print("  Step 1/2: Converting to float32...")
        with rasterio.open(input_path) as src:
            profile = src.profile.copy()
            dtype = np.dtype(src.dtypes[0])
            nodata = src.nodata if src.nodata is not None else np.iinfo(dtype).max

            profile.update(
                dtype="float32", nodata=np.nan, compress=compress,
                tiled=True, blockxsize=256, blockysize=256,
            )

            with rasterio.open(tmp_path, "w", **profile) as dst:
                for _, window in tqdm(src.block_windows(1), desc="  Converting"):
                    block = src.read(1, window=window)
                    block = np.where(block == nodata, np.nan, block.astype("float32") * scale_factor)
                    dst.write(block, 1, window=window)

        gc.collect()

        # Step 2: align spatial extent to match reference raster.
        # If CRS and resolution already match, do a fast row-level copy with
        # nodata padding (no warp needed). Otherwise fall back to WarpedVRT.
        print("  Step 2/2: Aligning to reference extent...")
        with rasterio.open(reference_path) as ref:
            dst_crs = ref.crs
            dst_transform = ref.transform
            dst_width = ref.width
            dst_height = ref.height
            dst_res = ref.res

        with rasterio.open(tmp_path) as src:
            src_res = src.res
            same_crs = src.crs == dst_crs
            same_res = (abs(src_res[0] - dst_res[0]) < 1e-10 and
                        abs(src_res[1] - dst_res[1]) < 1e-10)

            dst_profile = {
                "driver": "GTiff",
                "crs": dst_crs,
                "transform": dst_transform,
                "width": dst_width,
                "height": dst_height,
                "count": 1,
                "dtype": "float32",
                "nodata": np.nan,
                "compress": compress,
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
            }

            if same_crs and same_res:
                # Fast path: same grid, just different extents.
                # Compute row/col offset of source within destination grid.
                src_origin_col = int(round((src.transform.c - dst_transform.c) / dst_transform.a))
                src_origin_row = int(round((src.transform.f - dst_transform.f) / dst_transform.e))

                with rasterio.open(output_path, "w", **dst_profile) as dst:
                    # Process destination row-by-row in chunks
                    chunk_height = 256
                    n_chunks = (dst_height + chunk_height - 1) // chunk_height
                    for i in tqdm(range(n_chunks), desc="  Aligning extent"):
                        dst_row_start = i * chunk_height
                        dst_row_end = min(dst_row_start + chunk_height, dst_height)
                        dst_rows = dst_row_end - dst_row_start

                        # Corresponding source rows
                        src_row_start = dst_row_start - src_origin_row
                        src_row_end = src_row_start + dst_rows

                        out_block = np.full((1, dst_rows, dst_width), np.nan, dtype="float32")

                        # Clamp to valid source range
                        read_row_start = max(src_row_start, 0)
                        read_row_end = min(src_row_end, src.height)
                        if read_row_start < read_row_end:
                            read_col_start = max(-src_origin_col, 0)
                            read_col_end = min(dst_width - src_origin_col, src.width)
                            if read_col_start < read_col_end:
                                src_window = rasterio.windows.Window(
                                    read_col_start, read_row_start,
                                    read_col_end - read_col_start,
                                    read_row_end - read_row_start,
                                )
                                data = src.read(1, window=src_window)

                                # Place into output block at correct offset
                                out_r = read_row_start - src_row_start
                                out_c = read_col_start + src_origin_col
                                out_block[0,
                                          out_r:out_r + data.shape[0],
                                          out_c:out_c + data.shape[1]] = data

                        dst_window = rasterio.windows.Window(0, dst_row_start, dst_width, dst_rows)
                        dst.write(out_block, window=dst_window)
            else:
                # Slow path: actual reprojection needed via WarpedVRT
                vrt_options = {
                    "crs": dst_crs,
                    "transform": dst_transform,
                    "width": dst_width,
                    "height": dst_height,
                    "resampling": Resampling.nearest,
                    "src_nodata": np.nan,
                    "nodata": np.nan,
                }

                with rasterio.Env(GDAL_CACHEMAX=512):
                    with rasterio.vrt.WarpedVRT(src, **vrt_options) as vrt:
                        with rasterio.open(output_path, "w", **dst_profile) as dst:
                            windows = list(dst.block_windows(1))
                            for _, window in tqdm(windows, desc="  Reprojecting"):
                                block = vrt.read(1, window=window)
                                dst.write(block, 1, window=window)

        print(f"  Saved: {output_path}")
        gc.collect()

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


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

    This implementation is optimized to:
    - avoid materializing all windows in memory
    - reduce explicit GC calls
    - let GDAL/Rasterio use multiple threads where possible
    """
    # Allow GDAL to use multiple CPU cores for I/O
    with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS"):
        with rasterio.open(raster1_path) as src1, rasterio.open(raster2_path) as src2:
            if src1.shape != src2.shape:
                raise ValueError("Input rasters must have the same dimensions.")

            profile = src1.profile.copy()
            profile.update({
                "dtype": "float32",
                "nodata": fill_value,
                "compress": compress,
            })

            with rasterio.open(out_path, "w", **profile) as dst:
                # Iterate windows directly instead of building a full list
                for _, window in tqdm(src1.block_windows(1), desc="Combining raster blocks"):
                    b1 = src1.read(1, window=window)
                    b2 = src2.read(1, window=window)

                    mask1 = np.isnan(b1) if np.isnan(fill_value) else (b1 == fill_value)
                    mask2 = np.isnan(b2) if np.isnan(fill_value) else (b2 == fill_value)

                    b1_clean = np.where(mask1, 0, b1)
                    b2_clean = np.where(mask2, 0, b2)

                    combined = operation(b1_clean, b2_clean)
                    combined[mask1 & mask2] = fill_value

                    dst.write(combined.astype("float32"), 1, window=window)

    print(f"Combined raster saved to: {out_path}")


def multiply_two_rasters(value_path, multiplier_path, out_path, fill_value=np.nan, compress="lzw"):
    """
    Multiply two rasters band-by-band (value * multiplier), block-by-block.
    Uses GDAL_NUM_THREADS for I/O and avoids per-block GC for speed.
    Output: float32; pixels where either input is NaN become fill_value.
    """
    with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS"):
        with rasterio.open(value_path) as src1, rasterio.open(multiplier_path) as src2:
            if src1.shape != src2.shape:
                raise ValueError("Input rasters must have the same dimensions.")

            profile = src1.profile.copy()
            profile.update(dtype="float32", nodata=fill_value, compress=compress)

            with rasterio.open(out_path, "w", **profile) as dst:
                for _, window in tqdm(src1.block_windows(1), desc="Multiply raster blocks"):
                    a = src1.read(1, window=window)
                    b = src2.read(1, window=window)
                    out = np.float32(a) * np.float32(b)
                    out[np.isnan(a) | np.isnan(b)] = fill_value
                    dst.write(out, 1, window=window)
    print(f"Saved: {out_path}")


def reproject_raster(
    input_path,
    reference_path,
    output_path,
    compress="lzw",
    chunks={"x": 1024, "y": 1024},
    overwrite=False
    ):
    # ... docstring unchanged ...

    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"{output_path} exists. Use overwrite=True to replace it.")

    with tqdm(total=4, desc="Reprojecting", unit="step") as pbar:
        pbar.set_description("Loading reference raster")
        ref = rxr.open_rasterio(reference_path, masked=True, chunks=chunks).squeeze("band", drop=True)
        pbar.update(1)

        pbar.set_description("Loading input raster")
        target = rxr.open_rasterio(input_path, masked=True, chunks=chunks).squeeze("band", drop=True)
        pbar.update(1)

        pbar.set_description("Reprojecting to match reference")
        reprojected = target.rio.reproject_match(ref)
        pbar.update(1)

        pbar.set_description("Writing output")
        reprojected.rio.to_raster(
            output_path,
            compress=compress,
            tiled=True,
            blockxsize=256,
            blockysize=256
        )
        pbar.update(1)

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
    value_name="value",
    group_layer3_path=None,
    group3_name="lat_band_id",
    n_lat_bands=None,
):
    """
    Stack raster layers, summarize values by grouping over 2 or 3 keys, and write to CSV.

    Parameters
    ----------
    group_layer1_path : str
        Path to the first grouping raster layer (LULC).
    group_layer2_path : str
        Path to the second grouping raster layer (Carbon Zone).
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
    group_layer3_path : str, optional
        Path to a third grouping raster. If provided, used as 3rd key directly.
    group3_name : str
        Column name for the third key (default: "lat_band_id").
    n_lat_bands : int or None
        If set, automatically compute latitude bands as the 3rd key using this
        many bands. None (default) produces a 2-key table.
        Ignored when group_layer3_path is provided.
    """
    print("Loading raster layers...")
    layer1 = rxr.open_rasterio(group_layer1_path, masked=True, chunks={"x": 1024, "y": 1024}).squeeze("band")
    layer2 = rxr.open_rasterio(group_layer2_path, masked=True, chunks={"x": 1024, "y": 1024}).squeeze("band")
    layer3 = rxr.open_rasterio(value_layer_path, masked=True, chunks={"x": 1024, "y": 1024}).squeeze("band")

    # Determine mode and build layer list
    if group_layer3_path is not None:
        layer4 = rxr.open_rasterio(group_layer3_path, masked=True, chunks={"x": 1024, "y": 1024}).squeeze("band")
        layers = [layer1, layer2, layer4, layer3]
        col_names = [group1_name, group2_name, group3_name, value_name]
        is_3key = True
        lat_band_rows = None  # not needed
    elif n_lat_bands is not None:
        # Pre-compute lat band for every row once — same for all slices
        with rasterio.open(group_layer1_path) as src:
            transform = src.transform
            n_rows = src.height
        row_indices = np.arange(n_rows)
        lats = transform.f + (row_indices + 0.5) * transform.e
        band_width = 180.0 / n_lat_bands
        lat_band_rows = np.clip(
            np.floor((lats + 90.0) / band_width).astype(np.int32),
            0, n_lat_bands - 1
        )
        layers = [layer1, layer2, layer3]
        col_names = [group1_name, group2_name, group3_name, value_name]
        is_3key = True
    else:
        layers = [layer1, layer2, layer3]
        col_names = [group1_name, group2_name, value_name]
        is_3key = False
        lat_band_rows = None

    gc.collect()

    group_cols = col_names[:-1]
    value_col = col_names[-1]

    total_width = layer1.sizes["x"]
    step = total_width // num_slices
    dfs = []

    print(f"Processing raster slices ({'3-key' if is_3key else '2-key'} lookup)...")
    for i in tqdm(range(num_slices), desc="Slicing and summarizing"):
        x_start = i * step
        x_end = (i + 1) * step if i < (num_slices - 1) else total_width

        try:
            sliced_layers = [layer.isel(x=slice(x_start, x_end)) for layer in layers]
            flattened = [sl.values.reshape(-1).astype("float32") for sl in sliced_layers]

            if len(set(arr.shape[0] for arr in flattened)) != 1:
                print(f"Skipping slice {i + 1} due to shape mismatch.")
                continue

            if lat_band_rows is not None:
                # Tile pre-computed row lat bands across columns for this slice
                n_cols_slice = sliced_layers[0].sizes["x"]
                lat_band_flat = np.repeat(lat_band_rows, n_cols_slice).astype("float32")
                flattened.insert(2, lat_band_flat)

            df_pd = pd.DataFrame(np.stack(flattened, axis=1), columns=col_names).dropna(subset=col_names)

            if df_pd.empty:
                continue

            # Exclude carbon_zone_id = 0 (fill/water/no-data)
            df_pd = df_pd[df_pd['carbon_zone_id'] != 0]

            if df_pd.empty:
                continue

            summary = df_pd.groupby(group_cols)[value_col].agg(
                mean="mean",
                min="min",
                max="max",
                count="count"
            ).reset_index()

            dfs.append(summary)
            del df_pd, summary, flattened
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
        final_summary["mean"] = final_summary["weighted_sum"] / final_summary["count"]
        final_summary = final_summary[group_cols + ["mean", "min", "max", "count"]]
        final_summary = final_summary.rename(columns={
            "mean": f"{value_col}_mean",
            "min": f"{value_col}_min",
            "max": f"{value_col}_max",
            "count": f"{value_col}_count"
        })
        final_summary.to_csv(output_path, index=False)
        print(f"Summary written to: {output_path}")


def generate_carbon_density_raster(lulc_path, cz_path, carbon_density_lookup_table_path, out_path,
                                   n_lat_bands=18, **_):
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
        Path to CSV file containing the carbon density lookup table.
    out_path : str
        Output path for the resulting carbon density raster.
    n_lat_bands : int
        Number of latitude bands for 3-key lookup (auto-detected from table if 0).
    """
    # Vectorized sequential lookup via LookupTableArrayBuilder
    carbon_table = pd.read_csv(carbon_density_lookup_table_path, index_col=False)
    lookup_builder = LookupTableArrayBuilder(carbon_table)
    is_3key = lookup_builder.is_3key

    lulc_filename = os.path.basename(lulc_path)

    with rasterio.open(lulc_path) as lulc_src, rasterio.open(cz_path) as cz_src:
        assert lulc_src.shape == cz_src.shape, f"Shape mismatch between rasters: {lulc_filename}"

        profile = lulc_src.profile
        profile.update(dtype=rasterio.float32, nodata=np.nan)

        lat_band_ids_global = None
        if is_3key:
            lat_band_ids_global = compute_latitude_bands(lulc_src.transform, lulc_src.height, n_lat_bands)

        with rasterio.open(out_path, "w", **profile) as out_dst:
            for ji, window in tqdm(list(lulc_src.block_windows(1)), desc=f"Processing {lulc_filename}"):
                try:
                    lulc_block = lulc_src.read(1, window=window)
                    cz_block = cz_src.read(1, window=window)
                except rasterio.errors.RasterioIOError as e:
                    print(f"Skipping corrupt tile in {lulc_filename} at {window}: {e}")
                    continue

                carbon_block = np.full_like(lulc_block, np.nan, dtype=np.float32)
                valid_mask = (lulc_block > 0) & (cz_block > 0)

                if np.any(valid_mask):
                    if is_3key:
                        row_start = window.row_off
                        lat_band_ids = lat_band_ids_global[row_start:row_start + window.height]
                        lat_band_2d = np.broadcast_to(lat_band_ids[:, np.newaxis], lulc_block.shape)
                        carbon_block[valid_mask] = lookup_builder.lookup(
                            lulc_block[valid_mask].astype(int),
                            cz_block[valid_mask].astype(int),
                            lat_band_2d[valid_mask].astype(int),
                        )
                    else:
                        carbon_block[valid_mask] = lookup_builder.lookup(
                            lulc_block[valid_mask].astype(int),
                            cz_block[valid_mask].astype(int),
                        )

                out_dst.write(carbon_block, 1, window=window)

    gc.collect()
    print(f"Saved: {out_path}")


def compute_carbon_by_region_all_years(
    lulc_folder_path,
    cz_path,
    ab_lookup_path,
    be_lookup_path,
    ha_per_cell_path,
    countries_path,
    out_gpkg_path,
    n_lat_bands=0,
    reference_lulc_path=None,
):
    """
    For every LULC year, compute carbon totals and mean density by country region
    and write one layer per year into a GeoPackage.

    Replicates the old raster pipeline per region:
      1. Lookup ab/be density per pixel (MgC/ha).
      2. Multiply density × ha_per_cell → MgC/cell.
      3. Total uses ab_or_0 + be_or_0 (NaN only if BOTH pools missing),
         matching combine_two_float_rasters behaviour.
      4. Aggregate with np.bincount per country; convert to GgC.

    Output layers (one per year, layer name = str(year)) have columns:
      <original country fields>
      ab_per_ha_MgC       area-weighted mean aboveground density   (MgC/ha)
      be_per_ha_MgC       area-weighted mean belowground density   (MgC/ha)
      total_per_ha_MgC    area-weighted mean total density         (MgC/ha)
      ab_per_cell_GgC     country total aboveground carbon         (GgC)
      be_per_cell_GgC     country total belowground carbon         (GgC)
      total_per_cell_GgC  country total carbon                     (GgC)
    """
    import re as _re

    ab_builder = LookupTableArrayBuilder(pd.read_csv(ab_lookup_path, index_col=False))
    be_builder = LookupTableArrayBuilder(pd.read_csv(be_lookup_path, index_col=False))
    is_3key = ab_builder.is_3key or be_builder.is_3key

    # ------------------------------------------------------------------ #
    # Pre-compute static arrays (once, reused across all years)           #
    # ------------------------------------------------------------------ #
    # Reference grid from a single LULC file
    lulc_files = sorted([
        f for f in os.listdir(lulc_folder_path)
        if f.endswith('.tif') and not f.startswith('._')
    ])
    if not lulc_files:
        raise FileNotFoundError(f"No LULC .tif files found in {lulc_folder_path}")

    ref_lulc = reference_lulc_path or os.path.join(lulc_folder_path, lulc_files[0])
    with rasterio.open(ref_lulc) as ref:
        grid_shape     = (ref.height, ref.width)
        grid_transform = ref.transform
        grid_crs       = ref.crs

    # ha_per_cell array
    with rasterio.open(ha_per_cell_path) as ha_src:
        assert ha_src.shape == grid_shape, (
            f"Shape mismatch: LULC {grid_shape} vs ha_per_cell {ha_src.shape}."
        )
        ha_array = ha_src.read(1).astype(np.float32)
        ha_nd = ha_src.nodata
        if ha_nd is not None:
            ha_array[ha_array == ha_nd] = np.nan

    # Latitude band array (rows only, broadcast per window)
    lat_band_ids_global = None
    if is_3key:
        lat_band_ids_global = compute_latitude_bands(grid_transform, grid_shape[0], n_lat_bands)

    # Rasterize country polygons → integer ID array (1-indexed; 0 = outside)
    gdf = gpd.read_file(countries_path)
    if gdf.crs != grid_crs:
        gdf = gdf.to_crs(grid_crs)
    gdf = gdf.reset_index(drop=True)
    gdf['_rid'] = np.arange(1, len(gdf) + 1, dtype=np.int32)

    print("Rasterizing country boundaries...")
    country_raster = rasterio.features.rasterize(
        ((geom, rid) for geom, rid in zip(gdf.geometry, gdf['_rid'])),
        out_shape=grid_shape,
        transform=grid_transform,
        fill=0,
        dtype=np.int32,
    )
    n_regions = int(len(gdf)) + 1   # index 0 = outside/nodata

    # ------------------------------------------------------------------ #
    # Loop over years                                                      #
    # ------------------------------------------------------------------ #
    year_re = _re.compile(r'\d{4}')

    for fname in tqdm(lulc_files, desc="Years"):
        m = year_re.search(fname)
        if not m:
            continue
        year = m.group(0)
        lulc_path = os.path.join(lulc_folder_path, fname)

        # Per-country accumulators (indexed by _rid, 1-based)
        ab_wsum  = np.zeros(n_regions, dtype=np.float64)  # sum(ab_density * ha)
        be_wsum  = np.zeros(n_regions, dtype=np.float64)
        tot_wsum = np.zeros(n_regions, dtype=np.float64)
        ab_ha    = np.zeros(n_regions, dtype=np.float64)
        be_ha    = np.zeros(n_regions, dtype=np.float64)
        tot_ha   = np.zeros(n_regions, dtype=np.float64)

        try:
            lulc_test = rasterio.open(lulc_path)
            lulc_test.close()
        except Exception as e:
            print(f"  Skipping {fname}: cannot open ({e})")
            continue

        with rasterio.open(lulc_path) as lulc_src, rasterio.open(cz_path) as cz_src:
            for ji, window in lulc_src.block_windows(1):
                lulc_b = lulc_src.read(1, window=window)
                cz_b   = cz_src.read(1, window=window)

                r0, c0 = window.row_off, window.col_off
                r1, c1 = r0 + window.height, c0 + window.width

                ha_b  = ha_array[r0:r1, c0:c1]
                cid_b = country_raster[r0:r1, c0:c1]

                base = (
                    (lulc_b > 0) & (cz_b > 0) &
                    np.isfinite(ha_b) & (ha_b > 0) &
                    (cid_b > 0)
                )
                if not np.any(base):
                    continue

                lulc_v = lulc_b[base].astype(int)
                cz_v   = cz_b[base].astype(int)
                ha_v   = ha_b[base]
                cid_v  = cid_b[base].astype(int)

                if is_3key:
                    lat_ids = lat_band_ids_global[r0:r1]
                    lat_2d  = np.broadcast_to(lat_ids[:, np.newaxis], (window.height, window.width))
                    lat_v   = lat_2d[base].astype(int)
                    ab_d = ab_builder.lookup(lulc_v, cz_v, lat_v)
                    be_d = be_builder.lookup(lulc_v, cz_v, lat_v)
                else:
                    ab_d = ab_builder.lookup(lulc_v, cz_v)
                    be_d = be_builder.lookup(lulc_v, cz_v)

                ab_ok  = np.isfinite(ab_d)
                be_ok  = np.isfinite(be_d)
                any_ok = ab_ok | be_ok

                # ab pool
                if np.any(ab_ok):
                    ab_wsum += np.bincount(cid_v[ab_ok], weights=ab_d[ab_ok] * ha_v[ab_ok], minlength=n_regions)
                    ab_ha   += np.bincount(cid_v[ab_ok], weights=ha_v[ab_ok],               minlength=n_regions)

                # be pool
                if np.any(be_ok):
                    be_wsum += np.bincount(cid_v[be_ok], weights=be_d[be_ok] * ha_v[be_ok], minlength=n_regions)
                    be_ha   += np.bincount(cid_v[be_ok], weights=ha_v[be_ok],               minlength=n_regions)

                # total: ab_or_0 + be_or_0 (mirrors combine_two_float_rasters)
                if np.any(any_ok):
                    tot_d = np.where(ab_ok, ab_d, 0.0) + np.where(be_ok, be_d, 0.0)
                    tot_wsum += np.bincount(cid_v[any_ok], weights=tot_d[any_ok] * ha_v[any_ok], minlength=n_regions)
                    tot_ha   += np.bincount(cid_v[any_ok], weights=ha_v[any_ok],                 minlength=n_regions)

        # Compute per-country metrics; index by _rid
        rids = gdf['_rid'].values

        layer = gdf.drop(columns=['_rid']).copy()
        layer['ab_per_ha_MgC']      = np.where(ab_ha[rids]  > 0, ab_wsum[rids]  / ab_ha[rids],  np.nan)
        layer['be_per_ha_MgC']      = np.where(be_ha[rids]  > 0, be_wsum[rids]  / be_ha[rids],  np.nan)
        layer['total_per_ha_MgC']   = np.where(tot_ha[rids] > 0, tot_wsum[rids] / tot_ha[rids], np.nan)
        layer['ab_per_cell_GgC']    = ab_wsum[rids]  / 1_000.0
        layer['be_per_cell_GgC']    = be_wsum[rids]  / 1_000.0
        layer['total_per_cell_GgC'] = tot_wsum[rids] / 1_000.0

        layer.to_file(out_gpkg_path, driver='GPKG', layer=year)
        print(f"  Written layer: {year}")

    gc.collect()
    print(f"Saved: {out_gpkg_path}")


def summarize_raster_by_region(value_raster_path, region_boundary_path, out_path, year=None, carbon_type=None):
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

        results = []
        id_list = []
        for idx, row in tqdm(regions.iterrows(), total=len(regions), desc="Summarizing polygons"):
            geom = [row.geometry]
            id_list.append(row.get("id", idx))
            try:
                masked, _ = mask(src, geom, crop=True, nodata=np.nan, all_touched=True)
                values = masked[0]
                values = values[~np.isnan(values)]

                area_m2 = values.size

                stats = {
                    "index_id": row.get("id", idx),
                    "mean": values.mean(),
                    "min": values.min(),
                    "max": values.max(),
                    "count": values.size,
                    "total": values.sum()
                }
                results.append(stats)

            except Exception as e:
                print(f"Error processing region {idx}: {e}")
                continue
    regions["index_id"] =id_list
    df = pd.DataFrame(results)
    df = regions.merge(df, on="index_id", how="right")
    df = df.drop(columns=["index_id","geometry"])
    # If year is not provided, try to infer from the raster filename; otherwise default to 'unknown'
    if year is None:
        basename = os.path.basename(value_raster_path)
        m = re.search(r"(\d{4})", basename)
        if m:
            year = m.group(1)
        else:
            year = 'unknown'

    df['year'] = year

    # Optionally record which carbon pool this raster represents
    if carbon_type is not None:
        df['carbon_type'] = carbon_type

    df.to_csv(out_path, index=False,float_format="%.6f")
    print(f"Summary written to: {out_path}")


def compare_two_carbon_rasters(reference_raster_path, source_raster_path, out_diff_path=None, out_stats_path=None):
    """
    Compare two carbon density rasters: reproject source to reference grid, compute difference and summary stats.

    Parameters
    ----------
    reference_raster_path : str
        Reference raster (grid and CRS used for output).
    source_raster_path : str
        Raster to compare (will be reprojected to reference grid).
    out_diff_path : str, optional
        If set, write difference raster (source - reference) here.
    out_stats_path : str, optional
        If set, write summary statistics CSV here.

    Returns
    -------
    dict
        Summary stats (mean_diff, rmse, correlation, etc.).
    """
    with rasterio.open(reference_raster_path) as ref, rasterio.open(source_raster_path) as src:
        ref_profile = ref.profile.copy()
        ref_height, ref_width = ref.shape
        block_h, block_w = ref.block_shapes[0]
        total_blocks = ((ref_height + block_h - 1) // block_h) * ((ref_width + block_w - 1) // block_w)

        vrt_options = {
            "crs": ref.crs,
            "transform": ref.transform,
            "width": ref_width,
            "height": ref_height,
            "resampling": Resampling.bilinear,
        }
        if src.nodata is not None:
            vrt_options["src_nodata"] = src.nodata
            vrt_options["nodata"] = src.nodata

        # Streaming aggregates avoid keeping full rasters in memory.
        n_valid = 0
        sum_ref = 0.0
        sum_src = 0.0
        sum_diff = 0.0
        sum_diff2 = 0.0
        sum_ref2 = 0.0
        sum_src2 = 0.0
        sum_ref_src = 0.0
        sum_abs_diff = 0.0
        ref_min = np.inf
        ref_max = -np.inf
        n_positive_diff = 0
        n_negative_diff = 0
        n_zero_diff = 0

        diff_dst = None
        if out_diff_path:
            ref_profile.update(
                dtype=rasterio.float32,
                nodata=np.nan,
                compress="zstd",
                zstd_level=22,
                predictor=3,
                tiled=True,
                blockxsize=512,
                blockysize=512,
                num_threads="all_cpus",
                BIGTIFF="YES",
            )
            diff_dst = rasterio.open(out_diff_path, "w", **ref_profile)

        try:
            with rasterio.vrt.WarpedVRT(src, **vrt_options) as src_vrt:
                for _, window in tqdm(ref.block_windows(1), total=total_blocks, desc="Comparing raster blocks"):
                    ref_block = ref.read(1, window=window, masked=True, out_dtype="float32")
                    src_block = src_vrt.read(1, window=window, masked=True, out_dtype="float32")

                    valid = (~ref_block.mask) & (~src_block.mask)
                    if np.any(valid):
                        ref_vals = ref_block.data[valid].astype(np.float64)
                        src_vals = src_block.data[valid].astype(np.float64)
                        diff_vals = src_vals - ref_vals

                        count = ref_vals.size
                        n_valid += count
                        sum_ref += np.sum(ref_vals)
                        sum_src += np.sum(src_vals)
                        sum_diff += np.sum(diff_vals)
                        sum_diff2 += np.sum(diff_vals * diff_vals)
                        sum_ref2 += np.sum(ref_vals * ref_vals)
                        sum_src2 += np.sum(src_vals * src_vals)
                        sum_ref_src += np.sum(ref_vals * src_vals)
                        sum_abs_diff += np.sum(np.abs(diff_vals))
                        ref_min = min(ref_min, float(np.min(ref_vals)))
                        ref_max = max(ref_max, float(np.max(ref_vals)))
                        n_positive_diff += int(np.sum(diff_vals > 0))
                        n_negative_diff += int(np.sum(diff_vals < 0))
                        n_zero_diff += int(np.sum(diff_vals == 0))

                    if diff_dst is not None:
                        diff_block = np.full(ref_block.shape, np.nan, dtype=np.float32)
                        if np.any(valid):
                            diff_block[valid] = (src_block.data[valid] - ref_block.data[valid]).astype(np.float32)
                        diff_dst.write(diff_block, 1, window=window)
        finally:
            if diff_dst is not None:
                diff_dst.close()

    if n_valid == 0:
        stats = {
            "mean_reference": np.nan,
            "mean_source": np.nan,
            "mean_difference": np.nan,
            "std_difference": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "reference_min": np.nan,
            "reference_max": np.nan,
            "reference_range": np.nan,
            "nrmse_range": np.nan,
            "nrmse_mean": np.nan,
            "bias_pct_of_mean_reference": np.nan,
            "positive_diff_fraction": np.nan,
            "negative_diff_fraction": np.nan,
            "zero_diff_fraction": np.nan,
            "n_valid": 0,
            "correlation": np.nan,
        }
    else:
        mean_ref = sum_ref / n_valid
        mean_src = sum_src / n_valid
        mean_diff = sum_diff / n_valid
        var_diff = max((sum_diff2 / n_valid) - (mean_diff * mean_diff), 0.0)
        rmse = np.sqrt(sum_diff2 / n_valid)
        mae = sum_abs_diff / n_valid
        ref_range = ref_max - ref_min if np.isfinite(ref_max) and np.isfinite(ref_min) else np.nan
        nrmse_range = (rmse / ref_range) if np.isfinite(ref_range) and ref_range > 0 else np.nan
        nrmse_mean = (rmse / abs(mean_ref)) if np.isfinite(mean_ref) and abs(mean_ref) > 0 else np.nan
        bias_pct = (mean_diff / mean_ref * 100.0) if np.isfinite(mean_ref) and abs(mean_ref) > 0 else np.nan

        ref_var_term = (n_valid * sum_ref2) - (sum_ref * sum_ref)
        src_var_term = (n_valid * sum_src2) - (sum_src * sum_src)
        cov_term = (n_valid * sum_ref_src) - (sum_ref * sum_src)
        if n_valid > 1 and ref_var_term > 0 and src_var_term > 0:
            correlation = cov_term / np.sqrt(ref_var_term * src_var_term)
        else:
            correlation = np.nan

        stats = {
            "mean_reference": float(mean_ref),
            "mean_source": float(mean_src),
            "mean_difference": float(mean_diff),
            "std_difference": float(np.sqrt(var_diff)),
            "rmse": float(rmse),
            "mae": float(mae),
            "reference_min": float(ref_min),
            "reference_max": float(ref_max),
            "reference_range": float(ref_range),
            "nrmse_range": float(nrmse_range) if np.isfinite(nrmse_range) else np.nan,
            "nrmse_mean": float(nrmse_mean) if np.isfinite(nrmse_mean) else np.nan,
            "bias_pct_of_mean_reference": float(bias_pct) if np.isfinite(bias_pct) else np.nan,
            "positive_diff_fraction": float(n_positive_diff / n_valid),
            "negative_diff_fraction": float(n_negative_diff / n_valid),
            "zero_diff_fraction": float(n_zero_diff / n_valid),
            "n_valid": int(n_valid),
            "correlation": float(correlation) if np.isfinite(correlation) else np.nan,
        }

    if out_diff_path:
        print(f"Difference raster written to: {out_diff_path}")

    if out_stats_path:
        pd.DataFrame([stats]).to_csv(out_stats_path, index=False, float_format="%.6f")
        print(f"Comparison stats written to: {out_stats_path}")

    return stats


def plot_comparison(diff_raster_path, output_prefix):
    """
    Create quicklook PNGs and profile CSV/PNG from a diff raster.
    Produces:
      - {output_prefix}_diff_map.png
      - {output_prefix}_latitude_profile.csv
      - {output_prefix}_longitude_profile.csv
      - {output_prefix}_profiles.png
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except Exception as e:
        print(f"Skipping PNG/profile plots because matplotlib is unavailable: {e}")
        return

    with rasterio.open(diff_raster_path) as src:
        height, width = src.height, src.width
        transform = src.transform

        row_sum = np.zeros(height, dtype=np.float64)
        row_count = np.zeros(height, dtype=np.float64)
        col_sum = np.zeros(width, dtype=np.float64)
        col_count = np.zeros(width, dtype=np.float64)

        for _, window in src.block_windows(1):
            block = src.read(1, window=window, masked=True, out_dtype="float32")
            valid = ~block.mask
            if not np.any(valid):
                continue

            r0, c0 = window.row_off, window.col_off
            r1, c1 = r0 + window.height, c0 + window.width

            filled = np.where(valid, block.data, 0.0).astype(np.float64)
            row_sum[r0:r1] += np.sum(filled, axis=1)
            row_count[r0:r1] += np.sum(valid, axis=1)
            col_sum[c0:c1] += np.sum(filled, axis=0)
            col_count[c0:c1] += np.sum(valid, axis=0)

        lat_profile = np.full(height, np.nan, dtype=np.float64)
        lon_profile = np.full(width, np.nan, dtype=np.float64)
        np.divide(row_sum, row_count, out=lat_profile, where=row_count > 0)
        np.divide(col_sum, col_count, out=lon_profile, where=col_count > 0)

        lat_vals = transform.f + (np.arange(height) + 0.5) * transform.e
        lon_vals = transform.c + (np.arange(width) + 0.5) * transform.a

        lat_df = pd.DataFrame({"latitude": lat_vals, "mean_difference": lat_profile})
        lon_df = pd.DataFrame({"longitude": lon_vals, "mean_difference": lon_profile})
        lat_csv = f"{output_prefix}_latitude_profile.csv"
        lon_csv = f"{output_prefix}_longitude_profile.csv"
        lat_df.to_csv(lat_csv, index=False, float_format="%.6f")
        lon_df.to_csv(lon_csv, index=False, float_format="%.6f")

        # Downsampled map quicklook
        max_dim = 1600
        scale = max(height / max_dim, width / max_dim, 1.0)
        out_h = max(1, int(np.ceil(height / scale)))
        out_w = max(1, int(np.ceil(width / scale)))
        arr = src.read(
            1,
            out_shape=(out_h, out_w),
            resampling=Resampling.bilinear,
            masked=True,
            out_dtype="float32",
        )
        arr_plot = np.ma.filled(arr, np.nan).astype(np.float32)

    # NaN values are explicitly masked so non-overlap/ocean cells are not compared nor drawn.
    arr_masked = np.ma.masked_invalid(arr_plot)

    # Use requested red-yellow-green diverging colormap.
    base_cmap = plt.get_cmap("RdYlGn")
    cmap = base_cmap.copy() if hasattr(base_cmap, "copy") else base_cmap
    # Show NaN/non-overlap cells as white.
    cmap.set_bad("#ffffff")

    # Use the full valid difference range (no percentile clipping).
    valid_vals = arr_masked.compressed()
    if valid_vals.size > 0:
        vmax = float(np.nanmax(np.abs(valid_vals)))
    else:
        vmax = 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0

    # Difference map with geographic axes.
    lon_min = transform.c
    lon_max = transform.c + transform.a * width
    lat_max = transform.f
    lat_min = transform.f + transform.e * height

    def _apply_rounded_rectangle_frame(ax, corner_radius=0.04, pad=0.01):
        """
        Clip map artists to a rounded rectangle in axes coordinates.
        """
        clip_patch = FancyBboxPatch(
            (pad, pad),
            1.0 - 2.0 * pad,
            1.0 - 2.0 * pad,
            boxstyle=f"round,pad=0,rounding_size={corner_radius}",
            transform=ax.transAxes,
            facecolor="none",
            edgecolor="none",
        )
        ax.add_patch(clip_patch)

        for artist in ax.get_children():
            if artist is clip_patch:
                continue
            if hasattr(artist, "set_clip_path"):
                try:
                    artist.set_clip_path(clip_patch)
                except Exception:
                    pass

        border_patch = FancyBboxPatch(
            (pad, pad),
            1.0 - 2.0 * pad,
            1.0 - 2.0 * pad,
            boxstyle=f"round,pad=0,rounding_size={corner_radius}",
            transform=ax.transAxes,
            facecolor="none",
            edgecolor="#f0f0f0",
            linewidth=0.8,
            zorder=100,
        )
        ax.add_patch(border_patch)

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        fig = plt.figure(figsize=(12, 6), dpi=300)
        ax = plt.axes(projection=ccrs.Robinson())
        ax.set_global()
        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN, facecolor="#a8a9b3", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="#d9d9d9", edgecolor="none", zorder=1)
        ax.coastlines(color="#efefef", linewidth=0.3, zorder=2)
        ax.gridlines(color="#c4c6ce", linewidth=0.4, alpha=0.4)

        im = ax.imshow(
            arr_masked,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            extent=[lon_min, lon_max, lat_min, lat_max],
            origin="upper",
            interpolation="nearest",
            zorder=3,
        )
        _apply_rounded_rectangle_frame(ax)
    except Exception:
        # Fallback without cartopy.
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        ax.set_facecolor("#a8a9b3")
        im = ax.imshow(
            arr_masked,
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            extent=[lon_min, lon_max, lat_min, lat_max],
            origin="upper",
            aspect="auto",
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.2, color="#c4c6ce")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        _apply_rounded_rectangle_frame(ax)

    ax.set_title("Difference Map (Full Value Range)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, extend="both")
    cbar.set_ticks(np.linspace(-vmax, vmax, 5))
    cbar.set_label("Carbon Density Difference (ton/ha)")
    fig.tight_layout()
    diff_png = f"{output_prefix}_diff_map.png"
    fig.savefig(diff_png, dpi=160)
    plt.close(fig)

    # Profiles plot
    def _downsample(x, y, max_points=2000):
        n = len(x)
        if n <= max_points:
            return x, y
        idx = np.linspace(0, n - 1, max_points, dtype=int)
        return x[idx], y[idx]

    lat_x, lat_y = _downsample(lat_vals, lat_profile)
    lon_x, lon_y = _downsample(lon_vals, lon_profile)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=160, sharey=True)
    ax1.plot(lat_x, lat_y, linewidth=1.0, color="#2c7fb8")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_title("Mean Difference by Latitude")
    ax1.set_xlabel("Latitude")
    ax1.set_ylabel("Mean Difference")
    ax1.grid(alpha=0.2)

    ax2.plot(lon_x, lon_y, linewidth=1.0, color="#d95f0e")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Mean Difference by Longitude")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Mean Difference")
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    profiles_png = f"{output_prefix}_profiles.png"
    fig.savefig(profiles_png, dpi=160)
    plt.close(fig)

    print(f"Saved diff PNG: {diff_png}")
    print(f"Saved profile PNG: {profiles_png}")
    print(f"Saved latitude profile CSV: {lat_csv}")
    print(f"Saved longitude profile CSV: {lon_csv}")


def plot_model_reference_diff_map(diff_raster_path, output_path):
    """
    Create a model-reference difference map in Robinson projection using RdYlGn at 600 dpi.

    Positive values (green): Model overestimates compared to reference
    Negative values (red): Model underestimates compared to reference

    Parameters
    ----------
    diff_raster_path : str
        Path to difference raster (model - reference).
    output_path : str
        Path to save output PNG.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import cartopy.crs as ccrs

    with rasterio.open(diff_raster_path) as src:
        height, width = src.height, src.width
        transform = src.transform

        # Downsample large rasters for visualization
        max_dim = 4000
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            out_h = int(height * scale)
            out_w = int(width * scale)
            arr = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.average)
        else:
            out_h, out_w = height, width
            arr = src.read(1)

        # Compute geographic extent from the raster transform
        left = transform.c
        top = transform.f
        right = left + transform.a * width
        bottom = top + transform.e * height
        extent = [left, right, bottom, top]

        # Mask nodata (NaN safe)
        nodata = src.nodata
        if nodata is not None and np.isfinite(nodata):
            arr_masked = np.ma.masked_where(arr == nodata, arr)
        else:
            arr_masked = np.ma.masked_invalid(arr)

    valid_vals = arr_masked.compressed()
    if valid_vals.size == 0:
        print(f"No valid data in {diff_raster_path}")
        return

    # Use actual data min/max for color limits
    vmin = float(valid_vals.min())
    vmax = float(valid_vals.max())

    cmap = cm.get_cmap('RdYlGn')
    cmap.set_bad("white")

    # Robinson projection
    fig, ax = plt.subplots(figsize=(16, 10),
                           subplot_kw={'projection': ccrs.Robinson()})
    ax.set_global()

    im = ax.imshow(
        arr_masked,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        origin="upper",
        extent=extent,
        transform=ccrs.PlateCarree(),
    )

    # Horizontal colorbar under the map
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
    cbar.set_label("Difference (MgC/ha)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax.spines['geo'].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"Saved model-reference diff map: {output_path}")


def generate_all_variant_diff_maps(project_dir, output_dir=None):
    """
    Generate model-reference difference maps for all variants (2key, 3key-18, 3key-36).

    Creates green-red difference maps showing where model over/under-estimates.

    Parameters
    ----------
    project_dir : str
        Project directory containing intermediate outputs.
    output_dir : str, optional
        Directory to save maps. If None, uses project_dir/output/validation_maps.
    """
    import os

    if output_dir is None:
        output_dir = os.path.join(project_dir, 'output', 'validation_maps')
    os.makedirs(output_dir, exist_ok=True)

    variants = [
        ('2key', 'LULC + Carbon Zone'),
        ('3key_18bands', 'LULC + Carbon Zone + 10° latitude'),
        ('3key_36bands', 'LULC + Carbon Zone + 5° latitude'),
    ]

    pools = [
        ('aboveground', 'Aboveground'),
        ('belowground', 'Belowground'),
    ]

    for var_code, var_desc in variants:
        for pool_code, pool_desc in pools:
            diff_path = os.path.join(
                project_dir, 'intermediate', var_code,
                'task_validation_all_variants_vs_spawn', 'all_variants_vs_spawn',
                f'{var_code}_{pool_code}_diff_vs_spawn.tif'
            )

            if not os.path.exists(diff_path):
                print(f"Skipping {var_code} {pool_code}: diff raster not found")
                continue

            output_path = os.path.join(
                output_dir,
                f'{var_code}_{pool_code}_diff_map.png'
            )

            try:
                plot_model_reference_diff_map(diff_path, output_path)
            except Exception as e:
                print(f"Error processing {var_code} {pool_code}: {e}")

    print(f"\nAll difference maps saved to: {output_dir}")


def adjust_carbon_raster_with_lat_lon_bias(
    model_raster_path,
    latitude_profile_csv_path,
    longitude_profile_csv_path,
    out_path,
    stats_csv_path=None,
    clamp_min=0.0,
    compress="lzw",
):
    """
    Apply additive bias correction to a carbon raster using latitude/longitude
    mean-difference profiles.

    Bias field:
      bias(lat, lon) = b_lat(lat) + b_lon(lon) - mu
    where:
      b_lat and b_lon are interpolated from profile CSVs,
      mu is global mean_difference from stats_csv_path if available, else 0.

    Corrected raster:
      corrected = model + bias
    """
    lat_df = pd.read_csv(latitude_profile_csv_path)
    lon_df = pd.read_csv(longitude_profile_csv_path)

    if not {"latitude", "mean_difference"}.issubset(lat_df.columns):
        raise ValueError(
            f"Latitude profile CSV missing required columns: {latitude_profile_csv_path}"
        )
    if not {"longitude", "mean_difference"}.issubset(lon_df.columns):
        raise ValueError(
            f"Longitude profile CSV missing required columns: {longitude_profile_csv_path}"
        )

    lat_df = lat_df[["latitude", "mean_difference"]].dropna().copy()
    lon_df = lon_df[["longitude", "mean_difference"]].dropna().copy()
    if lat_df.empty or lon_df.empty:
        raise ValueError("Latitude/longitude profile CSV has no valid rows after dropping NaNs.")

    lat_df = lat_df.sort_values("latitude")
    lon_df = lon_df.sort_values("longitude")
    lat_coords = lat_df["latitude"].to_numpy(dtype=np.float64)
    lat_bias_vals = lat_df["mean_difference"].to_numpy(dtype=np.float64)
    lon_coords = lon_df["longitude"].to_numpy(dtype=np.float64)
    lon_bias_vals = lon_df["mean_difference"].to_numpy(dtype=np.float64)

    mu = 0.0
    if stats_csv_path and os.path.isfile(stats_csv_path):
        try:
            stats_df = pd.read_csv(stats_csv_path)
            if "mean_difference" in stats_df.columns and not stats_df.empty:
                mu_candidate = stats_df["mean_difference"].iloc[0]
                if np.isfinite(mu_candidate):
                    mu = float(mu_candidate)
        except Exception:
            pass

    with rasterio.open(model_raster_path) as src:
        profile = src.profile.copy()
        profile.update(
            dtype="float32",
            nodata=np.nan,
            compress=compress,
            BIGTIFF="YES",
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                block = src.read(1, window=window, masked=True, out_dtype="float32")
                valid = ~block.mask
                out_block = np.full(block.shape, np.nan, dtype=np.float32)
                if np.any(valid):
                    row_idx = window.row_off + np.arange(window.height)
                    col_idx = window.col_off + np.arange(window.width)
                    lat_vals = src.transform.f + (row_idx + 0.5) * src.transform.e
                    lon_vals = src.transform.c + (col_idx + 0.5) * src.transform.a

                    lat_bias = np.interp(lat_vals, lat_coords, lat_bias_vals).astype(np.float32)
                    lon_bias = np.interp(lon_vals, lon_coords, lon_bias_vals).astype(np.float32)
                    bias_surface = lat_bias[:, None] + lon_bias[None, :] - float(mu)

                    corrected = block.data + bias_surface
                    if clamp_min is not None:
                        corrected = np.maximum(corrected, float(clamp_min))
                    out_block[valid] = corrected[valid].astype(np.float32)

                dst.write(out_block, 1, window=window)

    print(f"Wrote bias-adjusted raster: {out_path}")


def get_zonal_stats_fast(raster_path, gdf, region_id_col='ee_r264_id', num_slices=20):
    """
    Calculate zonal statistics using rasterization for efficiency.
    Processes raster in slices to manage memory.

    Parameters
    ----------
    raster_path : str
        Path to input raster.
    gdf : GeoDataFrame
        Regions with geometry.
    region_id_col : str, default 'ee_r264_id'
        Column name for region IDs.
    num_slices : int, default 20
        Number of horizontal slices for processing.

    Returns
    -------
    pd.DataFrame
        DataFrame with zonal statistics per region.
    """
    if not os.path.exists(raster_path):
        print(f"  Warning: Raster not found: {raster_path}")
        return None

    from rasterio import features
    import rasterio.windows

    region_ids = gdf[region_id_col].values.astype(int)

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

        if gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)

        transform = src.transform
        height, width = src.height, src.width

        print(f"  Rasterizing {len(gdf)} regions to {height}x{width}...")
        shapes = ((geom, int(rid)) for geom, rid in zip(gdf.geometry, region_ids))
        zones = features.rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.int32,
            all_touched=True
        )

        unique_ids = np.unique(region_ids)
        stats = {rid: {'sum': 0.0, 'sum_sq': 0.0, 'count': 0, 'min': np.inf, 'max': -np.inf}
                 for rid in unique_ids}

        total_width = width
        step = total_width // num_slices

        print(f"  Processing {num_slices} slices...")
        for i in tqdm(range(num_slices), desc="Processing slices", leave=False):
            x_start = i * step
            x_end = (i + 1) * step if i < (num_slices - 1) else total_width

            window = rasterio.windows.Window(x_start, 0, x_end - x_start, height)
            data_slice = src.read(1, window=window)
            zone_slice = zones[:, x_start:x_end]

            valid_mask = (~np.isnan(data_slice)) & (data_slice != src.nodata if src.nodata else True)
            valid_data = data_slice[valid_mask]
            valid_zones = zone_slice[valid_mask]

            for rid in unique_ids:
                mask_rid = valid_zones == rid
                if np.any(mask_rid):
                    values = valid_data[mask_rid]
                    stats[rid]['sum'] += np.sum(values)
                    stats[rid]['sum_sq'] += np.sum(values ** 2)
                    stats[rid]['count'] += len(values)
                    stats[rid]['min'] = min(stats[rid]['min'], float(np.min(values)))
                    stats[rid]['max'] = max(stats[rid]['max'], float(np.max(values)))

    results = []
    for rid in unique_ids:
        s = stats[rid]
        if s['count'] > 0:
            mean_val = s['sum'] / s['count']
            std_val = np.sqrt(s['sum_sq'] / s['count'] - mean_val ** 2) if s['count'] > 1 else 0.0
            results.append({
                'region_id': rid,
                'mean': mean_val,
                'sum': s['sum'],
                'count': s['count'],
                'std': std_val,
                'min': s['min'] if np.isfinite(s['min']) else np.nan,
                'max': s['max'] if np.isfinite(s['max']) else np.nan,
            })
        else:
            results.append({
                'region_id': rid,
                'mean': np.nan, 'sum': 0.0, 'count': 0,
                'std': np.nan, 'min': np.nan, 'max': np.nan,
            })

    return pd.DataFrame(results)


def plot_country_timeseries(countries_data, country_name, output_dir):
    """
    Create time series plot for a single country.

    Parameters
    ----------
    countries_data : dict
        Dictionary mapping country names to DataFrames with time series.
    country_name : str
        Name of country to plot.
    output_dir : str
        Directory to save output PNG.
    """
    if country_name not in countries_data:
        print(f"  Warning: {country_name} not found in data")
        return

    df = countries_data[country_name]
    if len(df) < 2:
        print(f"  Warning: insufficient data for {country_name}")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot 1: Total carbon stock
    ax = axes[0]
    ax.plot(df['year'], df['total_gt'], marker='o', linewidth=2, markersize=6, color='forestgreen')
    ax.fill_between(df['year'], df['total_gt'], alpha=0.3, color='forestgreen')
    ax.set_ylabel('Total Carbon (Gt C)', fontsize=12)
    ax.set_title(f'{country_name} - Terrestrial Carbon Time Series (1992-2020)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add trend annotation
    if len(df) > 1:
        z = np.polyfit(df['year'], df['total_gt'], 1)
        trend = z[0]
        total_change = df['total_gt'].iloc[-1] - df['total_gt'].iloc[0]
        pct_change = (total_change / df['total_gt'].iloc[0]) * 100

        ax.text(0.02, 0.95, f'Total change: {total_change:+.2f} Gt ({pct_change:+.1f}%)\n'
                           f'Trend: {trend*10:+.2f} Gt/decade',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Mean carbon density
    ax = axes[1]
    ax.plot(df['year'], df['mean_density'], marker='s', linewidth=2, markersize=6, color='steelblue')
    ax.fill_between(df['year'], df['mean_density'], alpha=0.3, color='steelblue')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Mean Density (t C/ha)', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    safe_name = country_name.replace(' ', '_').replace('/', '_')[:50]
    output_path = os.path.join(output_dir, f'{safe_name}_timeseries.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def plot_global_carbon_trend_standalone(data_dir, output_path, years=range(1992, 2021)):
    """
    Create global carbon trend plot (standalone version).

    Parameters
    ----------
    data_dir : str
        Directory containing {year}_total_by_region.csv files.
    output_path : str
        Path to save output PNG.
    years : range
        Years to include.
    """
    global_data = []

    for year in years:
        csv_path = os.path.join(data_dir, f'{year}_total_by_region.csv')
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        df_grouped = df.groupby('ee_r264_name')['total'].sum().reset_index()
        year_total = df_grouped['total'].sum() / 1e9
        global_data.append({'year': year, 'total_gt': year_total})

    df = pd.DataFrame(global_data)
    if len(df) == 0:
        print("No data found")
        return

    trend_coef = np.polyfit(df['year'], df['total_gt'], 1)

    # Calculate stats
    start_val = df['total_gt'].iloc[0]
    end_val = df['total_gt'].iloc[-1]
    change = end_val - start_val
    pct_change = (change / start_val) * 100
    trend_per_decade = trend_coef[0] * 10

    # Plot settings
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(11.2, 8))

    ax.plot(df['year'], df['total_gt'], 'o-', color='darkgreen',
            linewidth=2.5, markersize=6, label='Global Carbon Stock')
    ax.plot(df['year'], np.polyval(trend_coef, df['year']),
            '--', color='red', linewidth=2, label='Linear Trend')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='darkgreen', linewidth=2.5, linestyle='-', marker='o',
               markersize=6, label='Global Carbon Stock'),
        Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Linear Trend'),
        Line2D([0], [0], color='none', label=f'1992: {start_val:.2f} Gt C'),
        Line2D([0], [0], color='none', label=f'2020: {end_val:.2f} Gt C'),
        Line2D([0], [0], color='none', label=f'Δ: {change:.2f} Gt C ({pct_change:.2f}%)'),
        Line2D([0], [0], color='none', label=f'Trend: {trend_per_decade:.2f} Gt C/decade')
    ]

    ax.legend(handles=legend_elements, loc='upper right',
              frameon=True, fancybox=True, shadow=True, fontsize=10)

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Total Carbon (Gt C)', fontsize=12)
    ax.set_title('Global Terrestrial Carbon Stock (1992-2020)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    ymin, ymax = df['total_gt'].min(), df['total_gt'].max()
    ax.set_ylim(ymin - 0.5, ymax + 0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")



# =============================================================================
# PARALLEL PROCESSING UTILITIES
# =============================================================================

class ParallelConfig:
    """Configuration for parallel processing."""

    def __init__(
        self,
        n_workers: Optional[int] = None,
        max_workers: Optional[int] = None,
        use_processes: bool = True,
        chunks_per_worker: int = 4,
        show_progress: bool = True
    ):
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        if max_workers:
            self.n_workers = min(self.n_workers, max_workers)

        self.use_processes = use_processes
        self.chunks_per_worker = chunks_per_worker
        self.show_progress = show_progress

    def get_executor_class(self):
        """Get the appropriate executor class."""
        return ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor


def split_windows_into_chunks(
    windows: List[Tuple],
    n_chunks: int
) -> List[List[Tuple]]:
    """Split list of windows into chunks for parallel processing."""
    if n_chunks <= 1:
        return [windows]

    windows_sorted = sorted(windows, key=lambda x: (x[1].row_off, x[1].col_off))
    chunk_size = max(1, len(windows_sorted) // n_chunks)
    chunks = []
    for i in range(0, len(windows_sorted), chunk_size):
        chunk = windows_sorted[i:i + chunk_size]
        if chunk:
            chunks.append(chunk)

    return chunks


def _worker_process_chunk(
    chunk_windows: List[Tuple],
    input_paths: Dict[str, str],
    output_path: str,
    process_func: Callable,
    process_kwargs: Dict
) -> Dict[str, int]:
    """Worker function to process a chunk of windows."""
    chunk_stats = {'processed': 0, 'failed': 0}

    srcs = {}
    try:
        for key, path in input_paths.items():
            srcs[key] = rasterio.open(path)

        dst = rasterio.open(output_path, 'r+')

        for ji, window in chunk_windows:
            try:
                blocks = {}
                for key, src in srcs.items():
                    blocks[key] = src.read(1, window=window)

                result_block = process_func(blocks, window, **process_kwargs)
                dst.write(result_block, 1, window=window)
                chunk_stats['processed'] += 1

            except Exception as e:
                logger.error(f"Failed processing window {window}: {e}")
                chunk_stats['failed'] += 1

        dst.close()

    finally:
        for src in srcs.values():
            src.close()

    return chunk_stats


def process_blocks_parallel(
    input_paths: Dict[str, str],
    output_path: str,
    process_func: Callable,
    config: ParallelConfig = None,
    **process_kwargs
) -> Dict[str, Any]:
    """Process raster blocks in parallel."""
    if config is None:
        config = ParallelConfig()

    first_path = list(input_paths.values())[0]

    with rasterio.open(first_path) as src:
        profile = src.profile.copy()
        block_windows = list(src.block_windows(1))

    if 'dtype' in process_kwargs:
        profile.update(dtype=process_kwargs['dtype'])
    if 'nodata' in process_kwargs:
        profile.update(nodata=process_kwargs['nodata'])

    n_chunks = config.n_workers * config.chunks_per_worker
    window_chunks = split_windows_into_chunks(block_windows, n_chunks)

    logger.info(f"Processing {len(block_windows)} blocks using {config.n_workers} workers")

    with rasterio.open(output_path, 'w', **profile) as dst:
        pass

    worker_func = partial(
        _worker_process_chunk,
        input_paths=input_paths,
        output_path=output_path,
        process_func=process_func,
        process_kwargs=process_kwargs
    )

    stats = {
        'blocks_processed': 0,
        'blocks_failed': 0,
        'start_time': time.time()
    }

    ExecutorClass = config.get_executor_class()

    with ExecutorClass(max_workers=config.n_workers) as executor:
        futures = {
            executor.submit(worker_func, chunk): chunk
            for chunk in window_chunks
        }

        iterator = as_completed(futures)
        if config.show_progress:
            iterator = tqdm(iterator, total=len(futures), desc="Processing blocks")

        for future in iterator:
            try:
                chunk_stats = future.result()
                stats['blocks_processed'] += chunk_stats['processed']
                stats['blocks_failed'] += chunk_stats['failed']
            except Exception as e:
                logger.error(f"Chunk failed: {e}")

    stats["elapsed_time"] = time.time() - stats["start_time"]
    if stats["blocks_processed"] == 0:
        raise RuntimeError(f"All {stats['blocks_failed']} chunks failed — no blocks were written.")
    logger.info(f"Completed in {stats['elapsed_time']:.2f}s")

    return stats


class LookupTableArrayBuilder:
    """Build numpy arrays from lookup tables for fast raster generation."""

    def __init__(self, df_lookup: pd.DataFrame):
        self.df = df_lookup
        self.is_3key = 'lat_band_id' in df_lookup.columns

        self.max_lulc = int(df_lookup['lulc_id'].max()) + 1
        self.max_cz = int(df_lookup['carbon_zone_id'].max()) + 1

        if self.is_3key:
            self.max_lat = int(df_lookup['lat_band_id'].max()) + 1
            self.array = self._build_3key_array()
        else:
            self.max_lat = None
            self.array = self._build_2key_array()

    def _build_2key_array(self) -> np.ndarray:
        """Build 2D numpy array for 2-key lookup."""
        arr = np.full((self.max_lulc, self.max_cz), np.nan, dtype=np.float32)
        lulc_ids = self.df['lulc_id'].astype(int).values
        cz_ids = self.df['carbon_zone_id'].astype(int).values
        values = self.df['carbon_density_mean'].values

        valid_mask = (
            (lulc_ids >= 0) & (lulc_ids < self.max_lulc) &
            (cz_ids >= 0) & (cz_ids < self.max_cz)
        )

        arr[lulc_ids[valid_mask], cz_ids[valid_mask]] = values[valid_mask]
        return arr

    def _build_3key_array(self) -> np.ndarray:
        """Build 3D numpy array for 3-key lookup."""
        arr = np.full((self.max_lulc, self.max_cz, self.max_lat), np.nan, dtype=np.float32)
        lulc_ids = self.df['lulc_id'].astype(int).values
        cz_ids = self.df['carbon_zone_id'].astype(int).values
        lat_ids = self.df['lat_band_id'].astype(int).values
        values = self.df['carbon_density_mean'].values

        valid_mask = (
            (lulc_ids >= 0) & (lulc_ids < self.max_lulc) &
            (cz_ids >= 0) & (cz_ids < self.max_cz) &
            (lat_ids >= 0) & (lat_ids < self.max_lat)
        )

        arr[lulc_ids[valid_mask], cz_ids[valid_mask], lat_ids[valid_mask]] = values[valid_mask]
        return arr

    def lookup(self, lulc_ids: np.ndarray, cz_ids: np.ndarray,
               lat_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Vectorized lookup."""
        if self.is_3key and lat_ids is not None:
            lulc_clip = np.clip(lulc_ids, 0, self.max_lulc - 1)
            cz_clip = np.clip(cz_ids, 0, self.max_cz - 1)
            lat_clip = np.clip(lat_ids, 0, self.max_lat - 1)
            return self.array[lulc_clip, cz_clip, lat_clip]
        else:
            lulc_clip = np.clip(lulc_ids, 0, self.max_lulc - 1)
            cz_clip = np.clip(cz_ids, 0, self.max_cz - 1)
            return self.array[lulc_clip, cz_clip]


def compute_latitude_bands(
    transform: rasterio.Affine,
    n_rows: int,
    n_lat_bands: int = 18
) -> np.ndarray:
    """Compute latitude band IDs for each row."""
    row_indices = np.arange(n_rows)
    lats = transform.f + (row_indices + 0.5) * transform.e
    band_width = 180.0 / n_lat_bands
    lat_band_ids = np.floor((lats + 90.0) / band_width).astype(np.int32)
    return np.clip(lat_band_ids, 0, n_lat_bands - 1)


def _process_block_parallel(blocks, window, is_3key=False, lat_bands=None, lookup_builder=None, **_):
    """Module-level worker for parallel raster generation (must be picklable)."""
    lulc_block = blocks['lulc']
    cz_block = blocks['cz']
    carbon_block = np.full_like(lulc_block, np.nan, dtype=np.float32)
    valid_mask = (lulc_block > 0) & (cz_block > 0)

    if not np.any(valid_mask):
        return carbon_block

    if is_3key and lat_bands is not None:
        row_start = window.row_off
        lat_band_ids = lat_bands[row_start:row_start + window.height]
        lat_band_2d = np.broadcast_to(lat_band_ids[:, np.newaxis], lulc_block.shape)
        carbon_block[valid_mask] = lookup_builder.lookup(
            lulc_block[valid_mask].astype(int),
            cz_block[valid_mask].astype(int),
            lat_band_2d[valid_mask].astype(int),
        )
    else:
        carbon_block[valid_mask] = lookup_builder.lookup(
            lulc_block[valid_mask].astype(int),
            cz_block[valid_mask].astype(int),
        )

    return carbon_block


def generate_carbon_density_raster_parallel(
    lulc_path: str,
    cz_path: str,
    lookup_table_path: str,
    out_path: str,
    n_lat_bands: int = 18,
    n_workers: Optional[int] = None
) -> Dict[str, Any]:
    """Generate carbon density raster using parallel processing."""
    df_lookup = pd.read_csv(lookup_table_path, index_col=False)
    lookup_builder = LookupTableArrayBuilder(df_lookup)
    is_3key = lookup_builder.is_3key

    input_paths = {'lulc': lulc_path, 'cz': cz_path}

    lat_band_cache = None
    if is_3key:
        with rasterio.open(lulc_path) as src:
            lat_band_cache = compute_latitude_bands(src.transform, src.height, n_lat_bands)

    process_kwargs = {
        'dtype': rasterio.float32,
        'nodata': np.nan,
        'is_3key': is_3key,
        'lat_bands': lat_band_cache,
        'lookup_builder': lookup_builder,
    }

    config = ParallelConfig(
        n_workers=n_workers,
        use_processes=True,
        show_progress=True
    )

    logger.info(f"Generating raster: {out_path}")
    logger.info(f"Lookup type: {'3-key' if is_3key else '2-key'}")
    logger.info(f"Workers: {config.n_workers}")

    stats = process_blocks_parallel(
        input_paths=input_paths,
        output_path=out_path,
        process_func=_process_block_parallel,
        config=config,
        **process_kwargs
    )

    return stats

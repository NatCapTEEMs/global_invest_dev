import os
import re
import gc
from concurrent.futures import ProcessPoolExecutor
import rasterio
import pandas as pd
import hazelbean as hb
import numpy as np
from tqdm import tqdm

from global_invest.terrestrial_carbon import terrestrial_carbon_functions as terrestrial_carbon_functions


def task_prepare_spawn_rasters(p):
    """
    Convert uint Spawn rasters to float32 (scaled by 0.1) and reproject to
    match LULC CRS/resolution in a single pass, with no intermediate file.

    Output: {name}_projected.tif files in p.cur_dir
    """
    input_folder = p.get_path('terrestrial_carbon', 'spawn_2020')
    output_folder = p.cur_dir

    raw_carbon_density_maps = [
        f for f in os.listdir(input_folder)
        if f.endswith("biomass_carbon_2010.tif") and not f.startswith("._")
    ]

    print(f"Found {len(raw_carbon_density_maps)} Spawn raster(s) to prepare")
    for i, file in enumerate(raw_carbon_density_maps, 1):
        input_path = os.path.join(input_folder, file)
        name_root, ext = os.path.splitext(file)
        output_name = f"{name_root}_projected{ext}"
        output_path = os.path.join(output_folder, output_name)

        print(f"[{i}/{len(raw_carbon_density_maps)}] {file}")
        terrestrial_carbon_functions.convert_and_reproject_raster(
            input_path=input_path,
            reference_path=p.base_year_lulc_path,
            output_path=output_path,
            scale_factor=0.1,
            compress="zstd",
        )

    # Generate total carbon reference (aboveground + belowground)
    ag_path = os.path.join(output_folder, 'aboveground_biomass_carbon_2010_projected.tif')
    bg_path = os.path.join(output_folder, 'belowground_biomass_carbon_2010_projected.tif')
    total_path = os.path.join(output_folder, 'total_biomass_carbon_2010_projected.tif')

    if os.path.exists(ag_path) and os.path.exists(bg_path) and not os.path.exists(total_path):
        print("Generating total carbon reference (aboveground + belowground)...")
        with rasterio.open(ag_path) as ag_src:
            profile = ag_src.profile.copy()
            profile.update(dtype=rasterio.float32, nodata=np.float32(np.nan), compress='lzw', BIGTIFF='YES')
            block_h, block_w = ag_src.block_shapes[0]
            total_blocks = ((ag_src.height + block_h - 1) // block_h) * ((ag_src.width + block_w - 1) // block_w)

            with rasterio.open(bg_path) as bg_src, \
                 rasterio.open(total_path, 'w', **profile) as dst:
                for _, window in tqdm(ag_src.block_windows(1), total=total_blocks, desc="Summing AG+BG"):
                    ag_block = ag_src.read(1, window=window)
                    bg_block = bg_src.read(1, window=window)
                    ag_valid = np.isfinite(ag_block)
                    bg_valid = np.isfinite(bg_block)
                    total_block = np.full_like(ag_block, np.nan, dtype=np.float32)
                    both_valid = ag_valid & bg_valid
                    total_block[both_valid] = ag_block[both_valid] + bg_block[both_valid]
                    dst.write(total_block, 1, window=window)
        print(f"Saved total carbon reference: {total_path}")
    elif os.path.exists(total_path):
        print("Skipping total carbon reference (exists)")

    print("Finished preparing Spawn rasters.")
    gc.collect()
    return True


def task_rasterize_regions(p):
    """
    Rasterize the region vector to the LULC grid once. Shared across variants.

    Output: zone_{zone_col}.tif in p.cur_dir (int32, 0 = no zone).
    """
    import geopandas as gpd
    from rasterio.features import rasterize

    zone_col = getattr(p, 'zone_col', 'ee_r264_id')
    out_path = os.path.join(p.cur_dir, f"zone_{zone_col}.tif")
    if os.path.exists(out_path):
        print(f"  Skipping zone raster (exists): {out_path}")
        return True

    with rasterio.open(p.base_year_lulc_path) as tmpl:
        tmpl_profile = tmpl.profile.copy()
        tmpl_transform = tmpl.transform
        tmpl_crs = tmpl.crs
        tmpl_shape = (tmpl.height, tmpl.width)

    regions = gpd.read_file(p.gdf_countries_vector_path)
    if zone_col not in regions.columns:
        raise KeyError(f"Column {zone_col} not found in {p.gdf_countries_vector_path}")

    print(f"  Rasterizing regions to zone raster ({tmpl_shape[0]}x{tmpl_shape[1]})...")
    regions_proj = regions.to_crs(tmpl_crs) if regions.crs != tmpl_crs else regions
    ids = regions_proj[zone_col].astype('int32').to_numpy()
    shapes = ((geom, int(zid)) for geom, zid in zip(regions_proj.geometry, ids))
    zone_arr = rasterize(
        shapes=shapes,
        out_shape=tmpl_shape,
        transform=tmpl_transform,
        fill=0,
        dtype='int32',
        all_touched=False,
    )
    zone_profile = tmpl_profile.copy()
    zone_profile.update(dtype='int32', count=1, nodata=0, compress='LZW')
    with rasterio.open(out_path, 'w', **zone_profile) as dst:
        dst.write(zone_arr, 1)

    del zone_arr, regions, regions_proj, ids
    gc.collect()
    print(f"  Saved: {out_path}")
    return True


def task_convert_dtype_to_float(p):
    """Deprecated: use task_prepare_spawn_rasters instead."""
    return task_prepare_spawn_rasters(p)


# LEGACY: Deprecated wrapper retained for compatibility.
def task_reproject_to_lulc_crs(p):
    """Deprecated: use task_prepare_spawn_rasters instead."""
    return task_prepare_spawn_rasters(p)


def task_compute_carbon_density_table(p):
    """
    Compute lookup table(s) of mean carbon density by LULC and carbon zone.

    n_lat_bands=0: generates 2-key table only.
    n_lat_bands>0: generates both 2-key and 3-key (with lat bands) tables.

    Output: *_carbon_density_lookup_table.csv and/or *_3key.csv in p.cur_dir
    """
    # Look for projected rasters in shared dir first, then variant dir
    shared_dir = getattr(p, 'shared_intermediate_dir', None)
    if shared_dir:
        projected_folder = os.path.join(shared_dir, "task_prepare_spawn_rasters")
        # Fallback to old folder name for backwards compatibility
        if not os.path.isdir(projected_folder):
            projected_folder = os.path.join(shared_dir, "task_reproject_to_lulc_crs")
    else:
        projected_folder = os.path.join(p.intermediate_dir, "task_prepare_spawn_rasters")
        if not os.path.isdir(projected_folder):
            projected_folder = os.path.join(p.intermediate_dir, "task_reproject_to_lulc_crs")

    if not os.path.isdir(projected_folder):
        raise FileNotFoundError(f"Projected raster folder not found: {projected_folder}")

    projected_tifs = [
        f for f in os.listdir(projected_folder)
        if f.endswith("_projected.tif") and not f.startswith("._")
        and ("aboveground" in f or "belowground" in f)
    ]

    if not projected_tifs:
        raise FileNotFoundError(
            f"No *_projected.tif files in {projected_folder}. "
            "Run task_prepare_spawn_rasters first."
        )

    n_lat_bands = getattr(p, 'n_lat_bands', 0)
    tables_to_generate = [('2key', False)]
    if n_lat_bands:
        tables_to_generate.append(('3key', True))

    # For 3key variants, try to reuse 2key lookup tables from the 2key variant folder
    twkey_lookup_dir = None
    if n_lat_bands:
        twkey_lookup_dir = os.path.join(
            p.project_dir, 'intermediate', '2key', 'task_compute_carbon_density_table'
        )

    for file in projected_tifs:
        value_layer_path = os.path.join(projected_folder, file)

        # Determine carbon type from filename
        if "aboveground" in file:
            carbon_type = "aboveground"
        elif "belowground" in file:
            carbon_type = "belowground"
        else:
            carbon_type = os.path.splitext(file)[0].replace("_projected", "")

        for table_type, use_lat_bands in tables_to_generate:
            suffix = '_3key' if table_type == '3key' else ''
            output_basename = f"{carbon_type}_carbon_density_lookup_table{suffix}.csv"
            output_path = os.path.join(p.cur_dir, output_basename)

            if os.path.exists(output_path):
                print(f"  Skipping {output_basename} (exists)")
                continue

            # For 2key tables in a 3key run, copy from 2key variant if available
            if table_type == '2key' and twkey_lookup_dir:
                source_path = os.path.join(twkey_lookup_dir, output_basename)
                if os.path.isfile(source_path):
                    import shutil
                    print(f"  Copying {output_basename} from 2key variant folder")
                    shutil.copy2(source_path, output_path)
                    continue

            print(f"  Generating {output_basename}")
            terrestrial_carbon_functions.stack_layers_to_csv(
                group_layer1_path=p.base_year_lulc_path,
                group_layer2_path=p.carbon_zones_path,
                value_layer_path=value_layer_path,
                output_path=output_path,
                group1_name="lulc_id",
                group2_name="carbon_zone_id",
                group3_name="lat_band_id",
                value_name="carbon_density",
                num_slices=100,
                n_lat_bands=n_lat_bands if use_lat_bands else None,
            )

    gc.collect()
    return True


def task_generate_carbon_gpkg(p):
    """
    Compute carbon by country for all years and write to GeoPackage.

    Each layer (one per year) contains:
      - ab_per_ha_MgC, be_per_ha_MgC, total_per_ha_MgC (mean densities)
      - ab_per_cell_GgC, be_per_cell_GgC, total_per_cell_GgC (total stocks)

    Output: carbon_by_country_all_years.gpkg in p.cur_dir
    """
    out_path = os.path.join(p.cur_dir, "carbon_by_country_all_years.gpkg")
    if os.path.exists(out_path):
        print(f"Skipping (exists): {out_path}")
        return True

    lookup_table_dir = os.path.join(p.intermediate_dir, "task_compute_carbon_density_table")
    n_lat_bands = getattr(p, 'n_lat_bands', 0)
    lookup_suffix = '_3key' if n_lat_bands else ''

    ab_lookup = os.path.join(lookup_table_dir, f"aboveground_carbon_density_lookup_table{lookup_suffix}.csv")
    be_lookup = os.path.join(lookup_table_dir, f"belowground_carbon_density_lookup_table{lookup_suffix}.csv")

    for name, path in [("AG lookup", ab_lookup), ("BG lookup", be_lookup)]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {name}: {path}")

    ha_per_cell_path = p.get_path("pyramids", "ha_per_cell_10sec.tif")
    if not os.path.exists(ha_per_cell_path):
        raise FileNotFoundError(f"Missing ha_per_cell: {ha_per_cell_path}")

    if not os.path.isdir(p.lulc_folder_path):
        raise FileNotFoundError(f"LULC folder not found: {p.lulc_folder_path}")

    terrestrial_carbon_functions.compute_carbon_by_region_all_years(
        lulc_folder_path=p.lulc_folder_path,
        cz_path=p.carbon_zones_path,
        ab_lookup_path=ab_lookup,
        be_lookup_path=be_lookup,
        ha_per_cell_path=ha_per_cell_path,
        countries_path=p.gdf_countries,
        out_gpkg_path=out_path,
        n_lat_bands=n_lat_bands,
        reference_lulc_path=p.base_year_lulc_path,
    )

    gc.collect()
    return True


def task_generate_carbon_density_raster(p):
    """
    Generate per-year carbon density rasters from LULC + lookup table.

    Output: aboveground_{lulc_filename}.tif, belowground_{lulc_filename}.tif in p.cur_dir
    """
    if not os.path.isdir(p.lulc_folder_path):
        raise FileNotFoundError(f"LULC folder not found: {p.lulc_folder_path}")

    n_lat_bands = getattr(p, 'n_lat_bands', 0)
    lookup_suffix = '_3key' if n_lat_bands else ''

    lookup_table_dir = os.path.join(p.intermediate_dir, "task_compute_carbon_density_table")
    carbon_types = [
        ("aboveground", os.path.join(lookup_table_dir, f"aboveground_carbon_density_lookup_table{lookup_suffix}.csv")),
        ("belowground", os.path.join(lookup_table_dir, f"belowground_carbon_density_lookup_table{lookup_suffix}.csv")),
    ]
    carbon_types = [(name, path) for name, path in carbon_types if os.path.isfile(path)]

    if not carbon_types:
        raise FileNotFoundError(
            f"No lookup tables in {lookup_table_dir}. "
            f"Expected *{lookup_suffix}.csv files."
        )

    analysis_year = getattr(p, "analysis_year", None)
    lulc_files = sorted(
        f for f in os.listdir(p.lulc_folder_path)
        if f.endswith(".tif")
        and not f.startswith("._")
        and (analysis_year is None or str(analysis_year) in f)
    )
    if analysis_year is not None and not lulc_files:
        raise FileNotFoundError(
            f"No LULC TIFF found for analysis_year={analysis_year} "
            f"in {p.lulc_folder_path}"
        )

    total_rasters = len(lulc_files) * len(carbon_types)
    print(f"Generating rasters: {len(lulc_files)} LULC files x {len(carbon_types)} pools = {total_rasters} total")

    completed = 0
    skipped = 0
    for lulc_filename in lulc_files:
        lulc_path = os.path.join(p.lulc_folder_path, lulc_filename)

        for carbon_type, lookup_table_path in carbon_types:
            out_name = f"{carbon_type}_{lulc_filename}"
            out_path = os.path.join(p.cur_dir, out_name)

            if os.path.exists(out_path):
                skipped += 1
                continue

            completed += 1
            print(f"[{completed + skipped}/{total_rasters}] {out_name}")
            terrestrial_carbon_functions.generate_carbon_density_raster(
                lulc_path=lulc_path,
                cz_path=p.carbon_zones_path,
                carbon_density_lookup_table_path=lookup_table_path,
                out_path=out_path,
                n_lat_bands=n_lat_bands,
            )
            gc.collect()

    # Generate total carbon rasters (aboveground + belowground) for each LULC year
    total_generated = 0
    total_skipped = 0
    for lulc_filename in lulc_files:
        ag_path = os.path.join(p.cur_dir, f"aboveground_{lulc_filename}")
        bg_path = os.path.join(p.cur_dir, f"belowground_{lulc_filename}")
        total_path = os.path.join(p.cur_dir, f"total_{lulc_filename}")

        if os.path.exists(total_path):
            total_skipped += 1
            continue

        if not os.path.exists(ag_path) or not os.path.exists(bg_path):
            continue

        print(f"  Summing total_{lulc_filename}")
        with rasterio.open(ag_path) as ag_src:
            profile = ag_src.profile.copy()
            profile.update(dtype=rasterio.float32, nodata=np.float32(np.nan), compress='lzw', BIGTIFF='YES')
            block_h, block_w = ag_src.block_shapes[0]
            total_blocks = ((ag_src.height + block_h - 1) // block_h) * ((ag_src.width + block_w - 1) // block_w)

            with rasterio.open(bg_path) as bg_src, \
                 rasterio.open(total_path, 'w', **profile) as dst:
                for _, window in tqdm(ag_src.block_windows(1), total=total_blocks, desc="AG+BG"):
                    ag_block = ag_src.read(1, window=window)
                    bg_block = bg_src.read(1, window=window)
                    ag_valid = np.isfinite(ag_block)
                    bg_valid = np.isfinite(bg_block)
                    total_block = np.full_like(ag_block, np.nan, dtype=np.float32)
                    both_valid = ag_valid & bg_valid
                    total_block[both_valid] = ag_block[both_valid] + bg_block[both_valid]
                    dst.write(total_block, 1, window=window)
        total_generated += 1
        gc.collect()

    print(f"Finished generating carbon density rasters. Generated: {completed}, skipped: {skipped}")
    print(f"Total carbon rasters: generated {total_generated}, skipped {total_skipped}")
    gc.collect()
    return True


# LEGACY: 2010-only raster-generation path retained for compatibility.
def task_generate_2010_rasters(p):
    """
    Generate carbon density rasters for 2010 only, using the current variant's lookup table.

    Uses whichever lookup table matches p.n_lat_bands (2key or 3key).
    Output: aboveground_{lulc_2010_filename}.tif, belowground_{lulc_2010_filename}.tif in p.cur_dir
    """
    if not os.path.isdir(p.lulc_folder_path):
        raise FileNotFoundError(f"LULC folder not found: {p.lulc_folder_path}")

    n_lat_bands = getattr(p, 'n_lat_bands', 0)
    lookup_suffix = '_3key' if n_lat_bands else ''
    lookup_table_dir = os.path.join(p.intermediate_dir, "task_compute_carbon_density_table")

    carbon_types = [
        ("aboveground", os.path.join(lookup_table_dir, f"aboveground_carbon_density_lookup_table{lookup_suffix}.csv")),
        ("belowground", os.path.join(lookup_table_dir, f"belowground_carbon_density_lookup_table{lookup_suffix}.csv")),
    ]
    carbon_types = [(name, path) for name, path in carbon_types if os.path.isfile(path)]

    if not carbon_types:
        raise FileNotFoundError(
            f"No lookup tables in {lookup_table_dir}. "
            f"Expected *{lookup_suffix}.csv files."
        )

    # Find 2010 LULC file
    lulc_files = [f for f in os.listdir(p.lulc_folder_path)
                  if f.endswith(".tif") and "2010" in f and not f.startswith("._")]
    if not lulc_files:
        raise FileNotFoundError(f"Could not find 2010 LULC file in {p.lulc_folder_path}")
    lulc_filename = lulc_files[0]
    lulc_path = os.path.join(p.lulc_folder_path, lulc_filename)

    for carbon_type, lookup_table_path in carbon_types:
        out_name = f"{carbon_type}_{lulc_filename}"
        out_path = os.path.join(p.cur_dir, out_name)

        if os.path.exists(out_path):
            print(f"  Skipping {out_name} (exists)")
            continue

        print(f"  Generating {out_name} using {os.path.basename(lookup_table_path)}")
        terrestrial_carbon_functions.generate_carbon_density_raster(
            lulc_path=lulc_path,
            cz_path=p.carbon_zones_path,
            carbon_density_lookup_table_path=lookup_table_path,
            out_path=out_path,
            n_lat_bands=n_lat_bands,
        )

    print("Finished generating 2010 carbon density rasters.")
    gc.collect()
    return True


# LEGACY: Specialized 2-key 2010 raster-generation path retained for compatibility.
def task_generate_2key_2010_rasters(p):
    """
    Generate 2-key carbon density rasters (aboveground and belowground) for 2010 only.

    Uses 2-key lookup tables (lulc_id + carbon_zone_id).
    Outputs to p.cur_dir with filenames: 2010_aboveground_2key.tif, 2010_belowground_2key.tif
    """
    if not os.path.isdir(p.lulc_folder_path):
        raise FileNotFoundError(f"LULC folder not found: {p.lulc_folder_path}")

    lookup_table_dir = os.path.join(p.intermediate_dir, "task_compute_carbon_density_table")
    carbon_types = [
        ("aboveground", os.path.join(lookup_table_dir, "aboveground_carbon_density_lookup_table.csv")),
        ("belowground", os.path.join(lookup_table_dir, "belowground_carbon_density_lookup_table.csv")),
    ]
    carbon_types = [(name, path) for name, path in carbon_types if os.path.isfile(path)]

    if not carbon_types:
        raise FileNotFoundError(
            f"No 2-key lookup tables in {lookup_table_dir}. "
            f"Expected *_carbon_density_lookup_table.csv files (without _3key suffix)."
        )

    # Find 2010 LULC file
    lulc_path = os.path.join(p.lulc_folder_path, "lulc_esa_2010.tif")
    if not os.path.exists(lulc_path):
        # Try alternative naming patterns
        lulc_files = [f for f in os.listdir(p.lulc_folder_path)
                      if f.endswith(".tif") and "2010" in f and not f.startswith("._")]
        if not lulc_files:
            raise FileNotFoundError(f"Could not find 2010 LULC file in {p.lulc_folder_path}")
        lulc_path = os.path.join(p.lulc_folder_path, lulc_files[0])

    year = "2010"
    for carbon_type, lookup_table_path in carbon_types:
        out_path = os.path.join(p.cur_dir, f"{year}_{carbon_type}_2key.tif")

        if os.path.exists(out_path):
            print(f"  Skipping {year}_{carbon_type}_2key.tif (exists)")
            continue

        print(f"  Generating {year}_{carbon_type}_2key.tif")
        terrestrial_carbon_functions.generate_carbon_density_raster(
            lulc_path=lulc_path,
            cz_path=p.carbon_zones_path,
            carbon_density_lookup_table_path=lookup_table_path,
            out_path=out_path,
            n_lat_bands=0,  # 2-key: no latitude bands
        )

    print("Finished generating 2-key 2010 carbon density rasters.")
    gc.collect()
    return True


# LEGACY: Specialized 3-key 36-band 2010 path retained for compatibility.
def task_generate_3key_36band_2010_rasters(p):
    """
    Generate 3-key 36-band carbon density rasters (aboveground and belowground) for 2010 only.

    Uses 3-key lookup tables (lulc_id + carbon_zone_id + lat_band_id) with 36 latitude bands.
    Outputs to p.cur_dir with filenames: 2010_aboveground_3key36.tif, 2010_belowground_3key36.tif
    """
    if not os.path.isdir(p.lulc_folder_path):
        raise FileNotFoundError(f"LULC folder not found: {p.lulc_folder_path}")

    # For 36-band, we use the 3key tables from the 3key_36bands variant
    lookup_table_dir = os.path.join(p.project_dir, 'intermediate', '3key_36bands', 'task_compute_carbon_density_table')

    carbon_types = [
        ("aboveground", os.path.join(lookup_table_dir, "aboveground_carbon_density_lookup_table_3key.csv")),
        ("belowground", os.path.join(lookup_table_dir, "belowground_carbon_density_lookup_table_3key.csv")),
    ]
    carbon_types = [(name, path) for name, path in carbon_types if os.path.isfile(path)]

    if not carbon_types:
        raise FileNotFoundError(
            f"No 3-key 36-band lookup tables found in {lookup_table_dir}. "
            f"Run the pipeline with n_lat_bands=36 first to generate these tables."
        )

    # Find 2010 LULC file
    lulc_path = os.path.join(p.lulc_folder_path, "lulc_esa_2010.tif")
    if not os.path.exists(lulc_path):
        # Try alternative naming patterns
        lulc_files = [f for f in os.listdir(p.lulc_folder_path)
                      if f.endswith(".tif") and "2010" in f and not f.startswith("._")]
        if not lulc_files:
            raise FileNotFoundError(f"Could not find 2010 LULC file in {p.lulc_folder_path}")
        lulc_path = os.path.join(p.lulc_folder_path, lulc_files[0])

    year = "2010"
    for carbon_type, lookup_table_path in carbon_types:
        out_path = os.path.join(p.cur_dir, f"{year}_{carbon_type}_3key36.tif")

        if os.path.exists(out_path):
            print(f"  Skipping {year}_{carbon_type}_3key36.tif (exists)")
            continue

        print(f"  Generating {year}_{carbon_type}_3key36.tif")
        terrestrial_carbon_functions.generate_carbon_density_raster(
            lulc_path=lulc_path,
            cz_path=p.carbon_zones_path,
            carbon_density_lookup_table_path=lookup_table_path,
            out_path=out_path,
            n_lat_bands=36,  # 36 latitude bands
        )

    print("Finished generating 3-key 36-band 2010 carbon density rasters.")
    gc.collect()
    return True


def task_summarize_carbon_by_region(p, overwrite=False):
    """
    Zonal statistics: compute carbon stock per region from density rasters.

    Rasterizes the region vector (keyed by p.zone_col) once to the AGB grid,
    then processes each year in windowed chunks accumulating per-zone sums via
    np.bincount. Memory-bounded by chunk size, not region extent.

    Output per year: {year}_by_region.csv with columns:
      - AGB_carbon_stock, BGB_carbon_stock, total_carbon_stock (MgC)
      - total_area (ha)
      - all original vector attributes
    """
    import geopandas as gpd
    from rasterio.windows import Window
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling

    # Prefer the multi-year output from raster_generation_task_tree; fall back to
    # task_generate_2010_rasters/ which build_predict_base_year_task_tree writes
    # for the 2010-only path.
    multi_year_dir = os.path.join(p.intermediate_dir, "task_generate_carbon_density_raster")
    base_year_dir = os.path.join(p.intermediate_dir, "task_generate_2010_rasters")
    if os.path.isdir(multi_year_dir) and any(
        f.startswith("aboveground_") and f.endswith(".tif")
        for f in os.listdir(multi_year_dir)
    ):
        density_raster_dir = multi_year_dir
    else:
        density_raster_dir = base_year_dir

    ha_per_cell_path = p.get_path("pyramids", "ha_per_cell_10sec.tif")
    if not os.path.exists(ha_per_cell_path):
        raise FileNotFoundError(f"Missing ha_per_cell raster: {ha_per_cell_path}")

    zone_col = getattr(p, 'zone_col', 'ee_r264_id')
    regions = gpd.read_file(p.gdf_countries_vector_path)
    if zone_col not in regions.columns:
        raise KeyError(f"Column {zone_col} not found in {p.gdf_countries_vector_path}")

    ag_prefix = "aboveground_"
    bg_prefix = "belowground_"
    carbon_prefixes = (ag_prefix, bg_prefix, "total_")

    def strip_carbon_prefix(name):
        for pref in carbon_prefixes:
            if name.startswith(pref):
                return name[len(pref):]
        return name

    ag_files = sorted([
        f for f in os.listdir(density_raster_dir)
        if f.startswith(ag_prefix) and f.endswith(".tif") and not f.startswith("._")
    ])
    analysis_year = getattr(p, "analysis_year", None)
    if analysis_year is not None:
        ag_files = [f for f in ag_files if str(analysis_year) in f]
    if not ag_files:
        print(f"  No aboveground rasters in {density_raster_dir}")
        return True

    year_pattern = re.compile(r"\d{4}")
    raster_pairs = []
    for ag_fname in ag_files:
        lulc_base = strip_carbon_prefix(ag_fname)
        bg_fname = bg_prefix + lulc_base
        lulc_stem = os.path.splitext(lulc_base)[0]
        m = year_pattern.search(lulc_base)
        year = m.group(0) if m else lulc_stem
        raster_pairs.append((year, lulc_stem, ag_fname, bg_fname))

    # Load shared zone raster (built once by task_rasterize_regions in shared preprocessing)
    shared_dir = getattr(p, 'shared_intermediate_dir', None)
    if not shared_dir:
        raise RuntimeError("p.shared_intermediate_dir not set; run shared preprocessing first.")
    zone_raster_path = os.path.join(shared_dir, "task_rasterize_regions", f"zone_{zone_col}.tif")
    if not os.path.exists(zone_raster_path):
        raise FileNotFoundError(
            f"Shared zone raster not found: {zone_raster_path}. "
            "Run shared preprocessing (task_rasterize_regions) first."
        )
    print(f"  Using shared zone raster: {zone_raster_path}")

    # Capture AGB template grid for ha_per_cell alignment checks
    template_path = os.path.join(density_raster_dir, raster_pairs[0][2])
    with rasterio.open(template_path) as tmpl:
        tmpl_transform = tmpl.transform
        tmpl_crs = tmpl.crs
        tmpl_shape = (tmpl.height, tmpl.width)

    zone_ids = regions[zone_col].astype('int64').to_numpy()
    n_zones = int(zone_ids.max()) + 1  # bincount needs contiguous [0, max]

    total_files = 0
    for year, lulc_stem, ag_fname, bg_fname in tqdm(raster_pairs, desc="Processing years"):
        ag_path = os.path.join(density_raster_dir, ag_fname)
        bg_path = os.path.join(density_raster_dir, bg_fname)
        out_csv = os.path.join(p.cur_dir, f"carbon_summary_{lulc_stem}.csv")

        if not os.path.exists(bg_path):
            print(f"  Skipping {year}: belowground raster not found")
            continue
        if os.path.exists(out_csv) and not overwrite:
            print(f"  Skipping {year} (exists)")
            total_files += 1
            continue

        print(f"  Processing {year}...")

        agb_sum = np.zeros(n_zones, dtype=np.float64)
        bgb_sum = np.zeros(n_zones, dtype=np.float64)
        area_sum = np.zeros(n_zones, dtype=np.float64)

        with rasterio.open(ag_path) as ag_src, \
             rasterio.open(bg_path) as bg_src, \
             rasterio.open(zone_raster_path) as zn_src, \
             rasterio.open(ha_per_cell_path) as ha_raw:

            # Align ha_per_cell to AGB grid on the fly if needed
            if (ha_raw.transform == tmpl_transform and
                ha_raw.crs == tmpl_crs and
                (ha_raw.height, ha_raw.width) == tmpl_shape):
                ha_src = ha_raw
                ha_vrt = None
            else:
                ha_vrt = WarpedVRT(
                    ha_raw,
                    crs=tmpl_crs,
                    transform=tmpl_transform,
                    width=tmpl_shape[1],
                    height=tmpl_shape[0],
                    resampling=Resampling.average,
                )
                ha_src = ha_vrt

            # Iterate by internal block windows (usually row-strips)
            windows = [w for _, w in ag_src.block_windows(1)]
            for win in tqdm(windows, desc=f"  {year}", leave=False):
                ag = ag_src.read(1, window=win).astype(np.float64, copy=False)
                bg = bg_src.read(1, window=win).astype(np.float64, copy=False)
                ha = ha_src.read(1, window=win).astype(np.float64, copy=False)
                zn = zn_src.read(1, window=win)

                valid = (zn > 0) & np.isfinite(ag) & np.isfinite(bg) & np.isfinite(ha)
                if not valid.any():
                    continue

                z = zn[valid]
                a = ag[valid]
                b = bg[valid]
                h = ha[valid]

                agb_sum += np.bincount(z, weights=a * h, minlength=n_zones)
                bgb_sum += np.bincount(z, weights=b * h, minlength=n_zones)
                area_sum += np.bincount(z, weights=h, minlength=n_zones)

            if ha_vrt is not None:
                ha_vrt.close()

        # Assemble results by mapping zone_ids back onto the regions GeoDataFrame
        results = regions.drop(columns=['geometry']).copy() if 'geometry' in regions.columns else regions.copy()
        results['AGB_carbon_stock'] = agb_sum[zone_ids]
        results['BGB_carbon_stock'] = bgb_sum[zone_ids]
        results['total_carbon_stock'] = results['AGB_carbon_stock'] + results['BGB_carbon_stock']
        results['total_area'] = area_sum[zone_ids]
        results['year'] = year

        priority_cols = ['year', zone_col, 'AGB_carbon_stock', 'BGB_carbon_stock',
                         'total_carbon_stock', 'total_area']
        other_cols = [c for c in results.columns if c not in priority_cols]
        results = results[priority_cols + other_cols]

        results.to_csv(out_csv, index=False)
        print(f"    Written: {out_csv}")
        total_files += 1

        del results, agb_sum, bgb_sum, area_sum
        gc.collect()

    del regions
    gc.collect()

    print(f"\nTotal files generated: {total_files}")
    return True


def task_combine_region_summaries(p):
    """
    Combine per-LULC region summaries for this variant into one wide CSV.

    Reads every carbon_summary_lulc_esa_seals7_{suffix}.csv in
    intermediate/{variant}/task_summarize_carbon_by_region/ and merges them on
    the region vector's original attributes. For each summary file, the suffix
    after 'carbon_summary_lulc_esa_seals7_' is prefixed to the four stock
    columns, e.g.:
      2010_AGB_carbon_stock, 2010_BGB_carbon_stock,
      2010_total_carbon_stock, 2010_total_area

    Output: p.cur_dir/region_summary_combined.csv with the original ee vector
    columns (no geometry, no 'year' column) plus these per-LULC columns.
    """
    import geopandas as gpd

    summary_dir = os.path.join(p.intermediate_dir, 'task_summarize_carbon_by_region')
    if not os.path.isdir(summary_dir):
        print(f"  Skipping: {summary_dir} not found")
        return True

    prefix = 'carbon_summary_lulc_esa_seals7_'
    csvs = sorted([f for f in os.listdir(summary_dir)
                   if f.startswith(prefix) and f.endswith('.csv') and not f.startswith('._')])
    if not csvs:
        print(f"  No per-LULC summary CSVs in {summary_dir}")
        return True

    zone_col = getattr(p, 'zone_col', 'ee_r264_id')

    regions = gpd.read_file(p.gdf_countries_vector_path)
    if 'geometry' in regions.columns:
        regions = regions.drop(columns=['geometry'])
    combined = pd.DataFrame(regions)
    combined[zone_col] = pd.to_numeric(combined[zone_col], errors='coerce').astype('Int64')

    stock_cols = ['AGB_carbon_stock', 'BGB_carbon_stock', 'total_carbon_stock', 'total_area']

    scenario_strip = 'ssp2_rcp45_ngfs-remind-magpie_baseline_'
    for fname in csvs:
        suffix = fname[len(prefix):-len('.csv')]
        suffix = suffix.replace(scenario_strip, '')
        df = pd.read_csv(os.path.join(summary_dir, fname))
        keep = [zone_col] + [c for c in stock_cols if c in df.columns]
        sub = df[keep].copy()
        sub[zone_col] = pd.to_numeric(sub[zone_col], errors='coerce').astype('Int64')
        sub = sub.rename(columns={c: f'{suffix}_{c}' for c in stock_cols if c in sub.columns})
        combined = combined.merge(sub, on=zone_col, how='left')

    out_path = os.path.join(p.cur_dir, 'region_summary_combined.csv')
    combined.to_csv(out_path, index=False)
    print(f"  Saved: {out_path} ({len(combined)} rows, {len(combined.columns)} cols, "
          f"{len(csvs)} LULC files)")

    # Per-aez18 results (total carbon only) aggregated from region_summary_combined
    if 'aez18_id' in combined.columns:
        total_cols = [c for c in combined.columns if c.endswith('_total_carbon_stock')]
        if total_cols:
            aez_df = (combined[['aez18_id'] + total_cols]
                      .groupby('aez18_id', dropna=False)[total_cols]
                      .sum()
                      .reset_index())
            aez_df = aez_df.rename(columns={c: c.replace('_total_carbon_stock', '_total_MgC')
                                            for c in total_cols})
            aez_long = aez_df.melt(id_vars='aez18_id', var_name='lulc_suffix',
                                   value_name='total_MgC')
            aez_long['lulc_suffix'] = aez_long['lulc_suffix'].str.replace('_total_MgC', '', regex=False)

            aez_xlsx = os.path.join(p.cur_dir, 'region_summary_aez18.xlsx')
            with pd.ExcelWriter(aez_xlsx, engine='openpyxl') as writer:
                aez_df.to_excel(writer, sheet_name='aez18_total_wide', index=False)
                aez_long.to_excel(writer, sheet_name='aez18_total_long', index=False)
            print(f"  Saved: {aez_xlsx} ({len(aez_df)} aez18 rows, {len(total_cols)} LULC cols)")
    else:
        print("  Skipping aez18 aggregation: 'aez18_id' not in region vector columns")

    gc.collect()
    return True


def task_gep_by_country(p):
    """Calculate country-level terrestrial-carbon GEP from zonal summaries.

    Each summary may contain multiple regions for one ``iso3_r250_id``. Keep
    only its largest-area region, calculate annual rental-SCC GEP values, and
    append a global total row to each per-year output CSV.
    """
    if not getattr(p, 'gep_enabled', True):
        print('  Skipping GEP: p.gep_enabled is False')
        return True

    summary_dir = os.path.join(p.intermediate_dir, 'task_summarize_carbon_by_region')
    if not os.path.isdir(summary_dir):
        print(f"  Skipping GEP: {summary_dir} not found")
        return True

    prefix = 'carbon_summary_lulc_esa_'
    summary_files = sorted(
        f for f in os.listdir(summary_dir)
        if f.startswith(prefix) and f.endswith('.csv') and not f.startswith('._')
    )
    analysis_year = getattr(p, 'analysis_year', None)
    if analysis_year is not None:
        summary_files = [f for f in summary_files if str(analysis_year) in f]
    if not summary_files:
        print(f"  No carbon summaries found in {summary_dir}")
        return True

    iso_col = getattr(p, 'gep_iso_col', 'iso3_r250_id')
    expected_iso_count = getattr(p, 'gep_expected_iso_count', 250)
    required_cols = [iso_col, 'total_area', 'total_carbon_stock']
    if not hasattr(p, 'gep_rental_scc_2019_int_dollars_per_tc_year'):
        raise AttributeError(
            'Missing p.gep_rental_scc_2019_int_dollars_per_tc_year. '
            'Configure rental SCC rates in run_terrestrial_carbon.py.'
        )
    rental_scc_2019_int_dollars_per_tc_year = dict(
        p.gep_rental_scc_2019_int_dollars_per_tc_year
    )

    for fname in summary_files:
        input_path = os.path.join(summary_dir, fname)
        df = pd.read_csv(input_path)
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns in {input_path}: {missing_cols}")

        df[iso_col] = pd.to_numeric(df[iso_col], errors='coerce').astype('Int64')
        df['total_area'] = pd.to_numeric(df['total_area'], errors='coerce')
        df['total_carbon_stock'] = pd.to_numeric(df['total_carbon_stock'], errors='coerce')
        df = df.dropna(subset=[iso_col])
        stock_cols = [
            col for col in [
                'AGB_carbon_stock',
                'BGB_carbon_stock',
                'total_carbon_stock',
                'total_area',
            ]
            if col in df.columns
        ]
        for col in stock_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Summary initializes accumulators to zero when a region has no valid
        # model pixels. Treat that as missing coverage for GEP, not real zero.
        if getattr(p, 'gep_missing_coverage_as_na', True):
            missing_coverage = df['total_area'].eq(0) & df['total_carbon_stock'].eq(0)
            df.loc[missing_coverage, stock_cols] = np.nan

        # Retain representative metadata, including ee_r264_id, from largest
        # area row, but sum every carbon and area metric across each ISO-250 ID.
        representatives = (df.sort_values('total_area', ascending=False, kind='stable')
                              .drop_duplicates(subset=iso_col, keep='first')
                              .drop(columns=stock_cols))
        totals_by_iso = (df.groupby(iso_col, dropna=False)[stock_cols]
                           .sum(min_count=1)
                           .reset_index())
        df = (representatives.merge(totals_by_iso, on=iso_col, how='left')
                            .sort_values(iso_col, kind='stable')
                            .reset_index(drop=True))
        if expected_iso_count is not None and len(df) != expected_iso_count:
            raise ValueError(
                f"Expected {expected_iso_count} unique {iso_col} values in {input_path}; found {len(df)}"
            )

        gep_cols = []
        for suffix, rental_scc in rental_scc_2019_int_dollars_per_tc_year.items():
            gep_col = f'terrestrial_carbon_gep_{suffix}'
            df[gep_col] = df['total_carbon_stock'] * rental_scc
            gep_cols.append(gep_col)

        total_row = {col: pd.NA for col in df.columns}
        total_row['year'] = 'TOTAL' if 'year' in df.columns else pd.NA
        total_row[iso_col] = 'TOTAL'
        total_row['total_area'] = df['total_area'].sum(min_count=1)
        total_row['total_carbon_stock'] = df['total_carbon_stock'].sum(min_count=1)
        for gep_col in gep_cols:
            total_row[gep_col] = df[gep_col].sum(min_count=1)
        result = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

        numeric_cols = stock_cols + gep_cols
        for col in numeric_cols:
            result[col] = pd.to_numeric(result[col], errors='coerce')
        for col in numeric_cols:
            invalid = result[col].dropna()
            if not np.isfinite(invalid).all():
                raise ValueError(f"Non-finite value in {col}")

        value_cols = ['total_area', 'total_carbon_stock'] + gep_cols
        for col in value_cols:
            result[col] = result[col].astype(object)
            result.loc[result[col].notna() & result[col].eq(0), col] = 0

        end_cols = [
            col for col in [
                'AGB_carbon_stock',
                'BGB_carbon_stock',
                'total_carbon_stock',
            ] + gep_cols
            if col in result.columns
        ]
        result = result[[col for col in result.columns if col not in end_cols] + end_cols]

        suffix = fname[len('carbon_summary_'):-len('.csv')]
        output_prefix = getattr(p, 'gep_output_prefix', 'task_gep_by_country')
        output_path = os.path.join(p.cur_dir, f'{output_prefix}_{suffix}.csv')
        result.to_csv(output_path, index=False, na_rep='')
        print(f"  Saved: {output_path} ({len(df)} countries plus TOTAL)")

    return True


# LEGACY: Variant-comparison report and trend-plot path retained for compatibility.
def task_compare_all_variants(p):
    """
    Compare zonal-stat results across all available variants into one Excel file.

    Reads each variant's carbon_summary_{lulc_stem}.csv from
    intermediate/{variant}/task_summarize_carbon_by_region/ and writes
    intermediate/shared/task_compare_all_variants/variants_comparison.xlsx with:
      - totals_long: long-format totals (variant x year)
      - totals_pivot: wide-format totals (year x variant)
      - validation_vs_spawn_long: performance indices (RMSE, MAE, r, R^2, NRMSE,
        bias) for each variant/pool, read from each variant's
        task_validate_variant_vs_spawn/validation_summary.csv
      - validation_vs_spawn_pivot: metrics pivoted by pool x variant
      - regions_{year}: per-region comparison across variants (one sheet per year)

    Runs only when results from at least two variants are present.
    """
    import pandas as pd

    variants = getattr(p, 'variants', None)
    if variants:
        candidate_variants = [
            f'3key_{n}bands' if n else '2key' for n in variants
        ]
    else:
        candidate_variants = ['2key', '3key_18bands', '3key_36bands']
    summary_subdir = 'task_summarize_carbon_by_region'

    available = []
    for v in candidate_variants:
        vdir = os.path.join(p.project_dir, 'intermediate', v, summary_subdir)
        if os.path.isdir(vdir):
            csvs = [f for f in os.listdir(vdir)
                    if f.startswith('carbon_summary_') and f.endswith('.csv')
                    and not f.startswith('._')]
            if csvs:
                available.append((v, vdir, sorted(csvs)))

    if len(available) < 2:
        print(f"  Skipping comparison: only {len(available)} variant(s) with zonal stats found "
              f"(need >= 2). Available: {[v for v, _, _ in available]}")
        return True

    out_dir = os.path.join(p.shared_intermediate_dir, 'task_compare_all_variants')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'variants_comparison.xlsx')

    print(f"  Comparing {len(available)} variants: {[v for v, _, _ in available]}")

    zone_col = getattr(p, 'zone_col', 'ee_r264_id')

    # Load all CSVs keyed by (variant, lulc_stem)
    data = {}
    for vname, vdir, csvs in available:
        data[vname] = {}
        for f in csvs:
            stem = f[len('carbon_summary_'):-len('.csv')]
            data[vname][stem] = pd.read_csv(os.path.join(vdir, f))

    # Totals (long form)
    totals_rows = []
    for vname, years in data.items():
        for stem, df in years.items():
            year = df['year'].iloc[0] if 'year' in df.columns and len(df) else stem
            totals_rows.append({
                'variant': vname,
                'lulc_stem': stem,
                'year': year,
                'AGB_MgC': float(df['AGB_carbon_stock'].sum()),
                'BGB_MgC': float(df['BGB_carbon_stock'].sum()),
                'total_MgC': float(df['total_carbon_stock'].sum()),
                'total_area_ha': float(df['total_area'].sum()) if 'total_area' in df.columns else float('nan'),
            })
    totals_df = pd.DataFrame(totals_rows).sort_values(['lulc_stem', 'variant'])

    pivot = totals_df.pivot_table(
        index='lulc_stem', columns='variant',
        values=['AGB_MgC', 'BGB_MgC', 'total_MgC'],
    )

    # Per-year per-region sheets (union of stems across variants)
    all_stems = sorted({s for years in data.values() for s in years.keys()})

    # Collect per-variant validation-vs-Spawn performance indices, if available
    validation_frames = []
    for vname, _, _ in available:
        val_csv = os.path.join(p.project_dir, 'intermediate', vname,
                               'task_validate_variant_vs_spawn', 'validation_summary.csv')
        if not os.path.exists(val_csv):
            print(f"  No validation summary for {vname} at {val_csv}")
            continue
        vdf = pd.read_csv(val_csv)
        if 'variant' not in vdf.columns:
            vdf.insert(0, 'variant', vname)
        validation_frames.append(vdf)

    validation_long = pd.concat(validation_frames, ignore_index=True) if validation_frames else None

    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        totals_df.to_excel(writer, sheet_name='totals_long', index=False)
        pivot.to_excel(writer, sheet_name='totals_pivot')

        if validation_long is not None and not validation_long.empty:
            validation_long.to_excel(writer, sheet_name='validation_vs_spawn_long', index=False)

            # Pivot only the 2010 rows (those carry the comparison metrics)
            metric_cols = [c for c in ['rmse', 'mae', 'correlation', 'r_squared',
                                        'nrmse_range', 'nrmse_mean', 'mean_difference',
                                        'bias_pct_of_mean_reference']
                           if c in validation_long.columns]
            if metric_cols and 'pool' in validation_long.columns:
                metric_rows = validation_long.dropna(subset=metric_cols, how='all')
                if not metric_rows.empty:
                    val_pivot = metric_rows.pivot_table(
                        index='pool', columns='variant', values=metric_cols,
                    )
                    val_pivot.to_excel(writer, sheet_name='validation_vs_spawn_pivot')

            # Overall model totals per LULC year (AGB/BGB/total sums)
            if {'lulc_stem', 'pool', 'model_sum_density'}.issubset(validation_long.columns):
                totals_piv = validation_long.pivot_table(
                    index='lulc_stem', columns=['variant', 'pool'],
                    values='model_sum_density',
                )
                totals_piv.to_excel(writer, sheet_name='model_totals_by_lulc')

        for stem in all_stems:
            frames = []
            for vname in data:
                if stem not in data[vname]:
                    continue
                df = data[vname][stem]
                cols = [zone_col, 'AGB_carbon_stock', 'BGB_carbon_stock', 'total_carbon_stock']
                cols = [c for c in cols if c in df.columns]
                sub = df[cols].copy()
                sub = sub.rename(columns={
                    'AGB_carbon_stock': f'AGB_{vname}',
                    'BGB_carbon_stock': f'BGB_{vname}',
                    'total_carbon_stock': f'total_{vname}',
                })
                frames.append(sub)
            if not frames:
                continue
            merged = frames[0]
            for f2 in frames[1:]:
                merged = merged.merge(f2, on=zone_col, how='outer')
            # Excel sheet names: <=31 chars, no special chars
            sheet_name = f'regions_{stem}'[:31]
            merged.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"  Saved: {out_path}")

    # Trend plot: AGB_MgC / BGB_MgC / total_MgC per variant across scenarios
    import matplotlib.pyplot as plt

    plot_df = totals_df.dropna(subset=['lulc_stem']).copy()
    if not plot_df.empty:
        lulc_stems = sorted(plot_df['lulc_stem'].unique())
        fig, ax = plt.subplots(figsize=(12, 7))
        x = list(range(len(lulc_stems)))
        variants_sorted = sorted(plot_df['variant'].unique())

        for variant in variants_sorted:
            g = plot_df[plot_df['variant'] == variant].set_index('lulc_stem').reindex(lulc_stems)
            y = g['total_MgC'].values / 1e6  # MtC
            line, = ax.plot(x, y, marker='o', linewidth=2, markersize=6, label=variant)
            for xi, yi in zip(x, y):
                if pd.isna(yi):
                    continue
                ax.annotate(f'{yi:.2f}', (xi, yi), textcoords='offset points',
                            xytext=(0, 8), ha='center', fontsize=8,
                            color=line.get_color())

        ax.set_title('Total terrestrial carbon stock by scenario and variant')
        ax.set_xticks(x)
        ax.set_xticklabels(lulc_stems, rotation=15, ha='right')
        ax.set_ylabel('Total carbon stock (MtC)')
        ax.set_xlabel('Scenario')
        ax.grid(True, alpha=0.3)
        ax.legend(title='variant', loc='best')

        fig.tight_layout()
        plot_path = os.path.join(out_dir, 'variants_trend_total.png')
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {plot_path}")

    return True


# LEGACY: Spawn validation path retained for compatibility.
def task_validation_model_spawn(p):
    """
    Validate model outputs against Spawn et al. (2020) reference data for 2010.

    Compares the 2010 model rasters (aboveground and belowground) against the
    reference Spawn 2020 biomass data to compute per-pixel validation metrics:
      - Mean bias, RMSE, MAE
      - Pearson correlation
      - NRMSE (normalized by range and mean)
      - Difference raster (model - reference)

    Outputs:
      - {pool}_diff_2010.tif: Difference raster (model - reference)
      - {pool}_validation_stats_2010.csv: Summary statistics
    """
    # Paths
    spawn_dir = p.get_path('terrestrial_carbon', 'spawn_2020')
    raster_dir = os.path.join(p.intermediate_dir, "task_generate_carbon_density_raster")
    out_dir = p.cur_dir

    # Define pools to validate
    pools = [
        ("aboveground", "aboveground_biomass_carbon_2010_float_projected.tif"),
        ("belowground", "belowground_biomass_carbon_2010_float_projected.tif"),
    ]

    for pool_name, spawn_filename in pools:
        spawn_path = os.path.join(spawn_dir, spawn_filename)
        model_path = os.path.join(raster_dir, f"2010_{pool_name}.tif")

        # Check files exist
        if not os.path.exists(spawn_path):
            print(f"  Skipping {pool_name}: Spawn reference not found at {spawn_path}")
            continue
        if not os.path.exists(model_path):
            print(f"  Skipping {pool_name}: Model output not found at {model_path}")
            continue

        # Output paths
        diff_path = os.path.join(out_dir, f"{pool_name}_diff_2010.tif")
        stats_path = os.path.join(out_dir, f"{pool_name}_validation_stats_2010.csv")

        # Skip if already exists
        if os.path.exists(stats_path):
            print(f"  Skipping {pool_name} validation (exists)")
            continue

        print(f"  Validating {pool_name} for 2010...")

        # Run comparison
        stats = terrestrial_carbon_functions.compare_two_carbon_rasters(
            reference_raster_path=spawn_path,
            source_raster_path=model_path,
            out_diff_path=diff_path,
            out_stats_path=stats_path,
        )

        if stats:
            print(f"    RMSE: {stats.get('rmse', 'N/A'):.2f} MgC/ha")
            print(f"    MAE: {stats.get('mae', 'N/A'):.2f} MgC/ha")
            print(f"    Pearson r: {stats.get('pearson_r', 'N/A'):.3f}")
            print(f"    Bias: {stats.get('mean_diff', 'N/A'):.2f} MgC/ha")

    gc.collect()
    return True


# LEGACY: All-variant Spawn validation path retained for compatibility.
def task_validation_all_variants_vs_spawn(p):
    """
    Validate all model variants (2-key, 3key-18, 3key-36) against Spawn 2020 reference.

    Compares 2010 model rasters for all variants against Spawn reference data.
    Generates summary comparison table of all variants.

    Outputs:
      - {variant}_{pool}_diff_vs_spawn.tif: Difference rasters
      - {variant}_{pool}_stats_vs_spawn.csv: Per-variant statistics
      - validation_summary_all_variants.csv: Comparison table
    """
    spawn_dir = p.get_path('terrestrial_carbon', 'spawn_2020')
    out_dir = os.path.join(p.cur_dir, 'all_variants_vs_spawn')
    os.makedirs(out_dir, exist_ok=True)

    # Define all variants to validate
    variants = [
        ('2key', 0, '2key'),
        ('3key_18bands', 18, '3key_18bands'),
        ('3key_36bands', 36, '3key_36bands'),
    ]

    pools = ['aboveground', 'belowground']
    spawn_files = {
        'aboveground': 'aboveground_biomass_carbon_2010_float_projected.tif',
        'belowground': 'belowground_biomass_carbon_2010_float_projected.tif'
    }

    all_results = []

    for var_name, n_bands, var_dir in variants:
        raster_dir = os.path.join(p.project_dir, 'intermediate', var_dir, 'task_generate_carbon_density_raster')

        if not os.path.exists(raster_dir):
            print(f"Skipping {var_name}: rasters not found at {raster_dir}")
            continue

        for pool in pools:
            model_path = os.path.join(raster_dir, f"2010_{pool}.tif")
            spawn_path = os.path.join(spawn_dir, spawn_files[pool])

            if not os.path.exists(model_path):
                print(f"Skipping {var_name} {pool}: model raster not found")
                continue
            if not os.path.exists(spawn_path):
                print(f"Skipping {var_name} {pool}: Spawn reference not found")
                continue

            out_stats = os.path.join(out_dir, f"{var_name}_{pool}_stats_vs_spawn.csv")

            if os.path.exists(out_stats):
                print(f"Skipping {var_name} {pool} (exists)")
                df = pd.read_csv(out_stats)
                if not df.empty:
                    all_results.append({
                        'variant': var_name,
                        'n_lat_bands': n_bands,
                        'pool': pool,
                        'rmse': df['rmse'].values[0] if 'rmse' in df.columns else 0,
                        'mae': df['mae'].values[0] if 'mae' in df.columns else 0,
                        'pearson_r': df['pearson_r'].values[0] if 'pearson_r' in df.columns else 0,
                        'bias': df['mean_diff'].values[0] if 'mean_diff' in df.columns else 0,
                    })
                continue

            print(f"\nValidating {var_name} {pool} vs Spawn...")

            stats = terrestrial_carbon_functions.compare_two_carbon_rasters(
                reference_raster_path=spawn_path,
                source_raster_path=model_path,
                out_diff_path=os.path.join(out_dir, f"{var_name}_{pool}_diff_vs_spawn.tif"),
                out_stats_path=out_stats,
            )

            if stats:
                result = {
                    'variant': var_name,
                    'n_lat_bands': n_bands,
                    'pool': pool,
                    'rmse': stats.get('rmse', 0),
                    'mae': stats.get('mae', 0),
                    'pearson_r': stats.get('pearson_r', 0),
                    'bias': stats.get('mean_diff', 0),
                    'n_valid': stats.get('n_valid', 0),
                }
                all_results.append(result)

                print(f"  RMSE: {result['rmse']:.2f} MgC/ha")
                print(f"  MAE: {result['mae']:.2f} MgC/ha")
                print(f"  Pearson r: {result['pearson_r']:.3f}")
                print(f"  Bias: {result['bias']:.2f} MgC/ha")

    # Save comparison summary
    if all_results:
        df_summary = pd.DataFrame(all_results)
        summary_path = os.path.join(out_dir, 'validation_summary_all_variants.csv')
        df_summary.to_csv(summary_path, index=False)
        print(f"\n{'='*80}")
        print(f"Summary saved: {summary_path}")

        # Print comparison table
        print(f"\n{'='*80}")
        print("VALIDATION COMPARISON: ALL VARIANTS vs SPAWN 2020")
        print(f"{'='*80}")

        for pool in pools:
            print(f"\n{pool.upper()}:")
            pool_df = df_summary[df_summary['pool'] == pool].sort_values('n_lat_bands')
            print(f"{'Variant':<15} {'Bands':<6} {'RMSE':<8} {'MAE':<8} {'r':<8} {'Bias':<10}")
            print("-" * 60)
            for _, row in pool_df.iterrows():
                print(f"{row['variant']:<15} {row['n_lat_bands']:<6} "
                      f"{row['rmse']:<8.2f} {row['mae']:<8.2f} "
                      f"{row['pearson_r']:<8.3f} {row['bias']:<10.2f}")

    gc.collect()
    return True


def task_generate_validation_diff_maps(p):
    """
    Generate model-reference difference maps with green-red color scheme.

    Creates visualizations showing where model overestimates (red) or
    underestimates (green) compared to Spawn 2020 reference.

    Outputs (in task folder p.cur_dir):
      - {variant}_{pool}_diff_map.png: Green-red difference maps
    """
    # Output directly to task folder
    out_dir = p.cur_dir
    os.makedirs(out_dir, exist_ok=True)

    variants = [
        ('2key', 0, 'LULC + Carbon Zone'),
        ('3key_18bands', 18, 'LULC + Carbon Zone + 10° latitude'),
        ('3key_36bands', 36, 'LULC + Carbon Zone + 5° latitude'),
    ]

    pools = [
        ('aboveground', 'Aboveground'),
        ('belowground', 'Belowground'),
    ]

    total_generated = 0

    for var_code, n_bands, var_desc in variants:
        for pool_code, pool_desc in pools:
            # Source diff raster from validation task
            diff_path = os.path.join(
                p.project_dir, 'intermediate', var_code,
                'task_validation_all_variants_vs_spawn', 'all_variants_vs_spawn',
                f'{var_code}_{pool_code}_diff_vs_spawn.tif'
            )

            if not os.path.exists(diff_path):
                print(f"  Skipping {var_code} {pool_code}: diff raster not found")
                continue

            output_path = os.path.join(out_dir, f'{var_code}_{pool_code}_diff_map.png')

            if os.path.exists(output_path):
                print(f"  Skipping {var_code} {pool_code} (exists)")
                continue

            try:
                print(f"  Generating {var_code} {pool_code} diff map...")
                terrestrial_carbon_functions.plot_model_reference_diff_map(diff_path, output_path)
                total_generated += 1
            except Exception as e:
                print(f"    Error: {e}")

    print(f"\nGenerated {total_generated} difference maps in: {out_dir}")
    gc.collect()
    return True


def task_validate_variant_vs_spawn(p):
    """
    Validate this variant's 2010 rasters (aboveground + belowground) against Spawn 2020.

    Computes difference rasters and performance indicators (RMSE, MAE, Pearson r,
    NRMSE, bias, R squared) for the current variant only.

    Also reports overall model totals (sum and mean of density) for each LULC
    year (AGB, BGB, total) plus reference totals for 2010.

    Outputs in p.cur_dir:
      - {pool}_diff_vs_spawn.tif: difference raster (model minus reference)
      - {pool}_stats_vs_spawn.csv: per-pool statistics
      - validation_summary.csv: combined summary: per-pool 2010 vs Spawn
        comparison indices plus overall model totals for each LULC year
    """
    summary_path = os.path.join(p.cur_dir, 'validation_summary.csv')
    if os.path.exists(summary_path):
        print(f"Skipping task_validate_variant_vs_spawn (exists): {summary_path}")
        return True

    def _raster_sum_mean_count(path):
        """Return (sum, mean, count) of finite, non-nodata pixels in a raster."""
        total_sum = 0.0
        total_count = 0
        with rasterio.open(path) as src:
            nodata = src.nodata
            for _, window in src.block_windows(1):
                arr = src.read(1, window=window, masked=False).astype('float64')
                valid = np.isfinite(arr)
                if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
                    valid &= (arr != nodata)
                vals = arr[valid]
                total_sum += float(vals.sum())
                total_count += int(vals.size)
        mean = (total_sum / total_count) if total_count else float('nan')
        return total_sum, mean, total_count

    # Locate Spawn reference rasters in the shared preprocessing folder
    shared_dir = getattr(p, 'shared_intermediate_dir', None)
    if shared_dir:
        spawn_dir = os.path.join(shared_dir, 'task_prepare_spawn_rasters')
    else:
        spawn_dir = os.path.join(p.intermediate_dir, 'task_prepare_spawn_rasters')

    # Locate this variant's generated rasters. Prefer the multi-year output from
    # raster_generation_task_tree; fall back to task_generate_2010_rasters/ which
    # build_predict_base_year_task_tree writes for the 2010-only path.
    multi_year_dir = os.path.join(p.intermediate_dir, 'task_generate_carbon_density_raster')
    base_year_dir = os.path.join(p.intermediate_dir, 'task_generate_2010_rasters')
    if os.path.exists(os.path.join(multi_year_dir, 'aboveground_lulc_esa_2010.tif')):
        raster_dir = multi_year_dir
    else:
        raster_dir = base_year_dir

    pools = {
        'aboveground': {
            'model': os.path.join(raster_dir, 'aboveground_lulc_esa_2010.tif'),
            'reference': os.path.join(spawn_dir, 'aboveground_biomass_carbon_2010_projected.tif'),
        },
        'belowground': {
            'model': os.path.join(raster_dir, 'belowground_lulc_esa_2010.tif'),
            'reference': os.path.join(spawn_dir, 'belowground_biomass_carbon_2010_projected.tif'),
        },
        'total': {
            'model': os.path.join(raster_dir, 'total_lulc_esa_2010.tif'),
            'reference': os.path.join(spawn_dir, 'total_biomass_carbon_2010_projected.tif'),
        },
    }

    variant_name = getattr(p, 'variant_name', 'unknown')
    all_results = []

    for pool, paths in pools.items():
        if not os.path.exists(paths['model']):
            print(f"Skipping {pool}: model raster not found at {paths['model']}")
            continue
        if not os.path.exists(paths['reference']):
            print(f"Skipping {pool}: Spawn reference not found at {paths['reference']}")
            continue

        out_diff = os.path.join(p.cur_dir, f'{pool}_diff_vs_spawn.tif')
        out_stats = os.path.join(p.cur_dir, f'{pool}_stats_vs_spawn.csv')

        if os.path.exists(out_stats):
            print(f"Skipping {pool} (exists)")
            df = pd.read_csv(out_stats)
            if not df.empty:
                row = df.iloc[0]
                all_results.append({
                    'variant': variant_name,
                    'pool': pool,
                    **{col: row[col] for col in df.columns},
                })
            continue

        print(f"\nValidating {variant_name} {pool} vs Spawn...")
        stats = terrestrial_carbon_functions.compare_two_carbon_rasters(
            reference_raster_path=paths['reference'],
            source_raster_path=paths['model'],
            out_diff_path=out_diff,
            out_stats_path=out_stats,
        )

        if stats:
            r = stats.get('correlation', np.nan)
            r_squared = r * r if np.isfinite(r) else np.nan
            stats['r_squared'] = r_squared

            result = {'variant': variant_name, 'pool': pool}
            result.update(stats)
            all_results.append(result)

            # N = number of valid pixels, diff = model - reference
            print(f"  Mean (reference): {stats.get('mean_reference', np.nan):.4f} MgC/ha")
            print(f"  Mean (model):     {stats.get('mean_source', np.nan):.4f} MgC/ha")
            print(f"  RMSE:      {stats.get('rmse', np.nan):.4f} MgC/ha   # sqrt(sum(diff^2) / N)")
            print(f"  MAE:       {stats.get('mae', np.nan):.4f} MgC/ha   # sum(|diff|) / N")
            print(f"  Pearson r: {stats.get('correlation', np.nan):.4f}            # cov(ref, model) / (std(ref) * std(model))")
            print(f"  R squared: {r_squared:.4f}            # r^2")
            print(f"  Bias:      {stats.get('mean_difference', np.nan):.4f} MgC/ha   # sum(diff) / N")
            print(f"  NRMSE (range): {stats.get('nrmse_range', np.nan):.4f}        # RMSE / (max(ref) - min(ref))")
            print(f"  NRMSE (mean):  {stats.get('nrmse_mean', np.nan):.4f}        # RMSE / mean(ref)")
            print(f"  Bias %:    {stats.get('bias_pct_of_mean_reference', np.nan):.2f}%          # (mean(diff) / mean(ref)) * 100")

    # Annotate the 2010-vs-Spawn rows with lulc_stem so they align with the
    # per-LULC totals rows below.
    for r in all_results:
        r.setdefault('lulc_stem', 'lulc_esa_2010')
        r.setdefault('year', 2010)

    # Overall model totals for 2010 only (AGB, BGB, total).
    stem_2010 = 'lulc_esa_2010'
    lulc_filename = f'{stem_2010}.tif'
    ref_paths = {
        'aboveground': os.path.join(spawn_dir, 'aboveground_biomass_carbon_2010_projected.tif'),
        'belowground': os.path.join(spawn_dir, 'belowground_biomass_carbon_2010_projected.tif'),
        'total': os.path.join(spawn_dir, 'total_biomass_carbon_2010_projected.tif'),
    }

    for pool in ('aboveground', 'belowground', 'total'):
        model_path = os.path.join(raster_dir, f'{pool}_{lulc_filename}')
        if not os.path.exists(model_path):
            continue

        model_sum, model_mean, n_pixels = _raster_sum_mean_count(model_path)
        ref_sum = ref_mean = float('nan')
        if os.path.exists(ref_paths[pool]):
            ref_sum, ref_mean, _ = _raster_sum_mean_count(ref_paths[pool])

        match = next((r for r in all_results
                      if r.get('pool') == pool and r.get('lulc_stem') == stem_2010), None)
        totals = {
            'model_sum_density': model_sum,
            'model_mean_density': model_mean,
            'model_valid_pixels': n_pixels,
            'reference_sum_density': ref_sum,
            'reference_mean_density': ref_mean,
        }
        if match is not None:
            match.update(totals)
        else:
            row = {
                'variant': variant_name,
                'pool': pool,
                'lulc_stem': stem_2010,
                'year': 2010,
            }
            row.update(totals)
            all_results.append(row)

    if all_results:
        df_summary = pd.DataFrame(all_results)
        lead = [c for c in ['variant', 'pool', 'lulc_stem', 'year'] if c in df_summary.columns]
        others = [c for c in df_summary.columns if c not in lead]
        df_summary = df_summary[lead + others]
        df_summary = df_summary.sort_values(lead).reset_index(drop=True)
        summary_path = os.path.join(p.cur_dir, 'validation_summary.csv')
        df_summary.to_csv(summary_path, index=False, float_format='%.6f')
        print(f"\nValidation summary saved: {summary_path}")

    gc.collect()
    return True


def task_generate_variant_diff_maps(p):
    """
    Generate difference map PNGs for this variant's aboveground, belowground,
    and total pools.

    Reads difference TIFs from the preceding validation task and produces
    RdYlGn visualizations in Robinson projection.

    Outputs in p.cur_dir:
      - {pool}_diff_map.png
    """
    validation_dir = os.path.join(p.intermediate_dir, 'task_validate_variant_vs_spawn')

    pools = [
        ('aboveground', 'Aboveground'),
        ('belowground', 'Belowground'),
        ('total', 'Total'),
    ]

    total_generated = 0
    for pool_code, pool_desc in pools:
        diff_path = os.path.join(validation_dir, f'{pool_code}_diff_vs_spawn.tif')
        output_path = os.path.join(p.cur_dir, f'{pool_code}_diff_map.png')

        if not os.path.exists(diff_path):
            print(f"  Skipping {pool_code}: diff raster not found at {diff_path}")
            continue

        if os.path.exists(output_path):
            print(f"  Skipping {pool_code} (exists)")
            continue

        try:
            print(f"  Generating {pool_code} diff map...")
            terrestrial_carbon_functions.plot_model_reference_diff_map(diff_path, output_path)
            total_generated += 1
        except Exception as e:
            print(f"    Error generating {pool_code} diff map: {e}")

    print(f"\nGenerated {total_generated} difference maps in: {p.cur_dir}")
    gc.collect()
    return True

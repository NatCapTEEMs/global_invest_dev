"""Coastal-carbon GEP tasks: mapped habitat -> per-pixel carbon stock -> storage value.

This layer owns every file read and write. The science it calls lives in
coastal_carbon_functions, which never opens a file, so each step is checkable on hand-built
inputs.

Each habitat runs the same three tasks -- area within regions, carbon stock, storage value --
and the last two tasks combine them and collapse the result to one row per country.
"""
import contextlib
import functools
import glob
import os
import time
from collections import defaultdict

import geopandas as gpd
import hazelbean as hb
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.transform
import rasterio.warp
import rasterio.windows
import shapely
from rasterio.enums import Resampling
from shapely.geometry import box
from tqdm import tqdm

from global_invest import utilities
from global_invest.coastal_carbon import coastal_carbon_functions as ccf

# Region-id raster: uint16 spans the eemarine_r566 ids, and 0 means "outside every region".
# Types are GDAL codes because that is the currency hazelbean's raster writers take.
REGION_ID_GDAL_TYPE = 2   # UInt16
RASTER_NDV = 0
# Habitat extent mask: 1 where the habitat polygon covers the cell.
MASK_GDAL_TYPE = 1        # Byte
MASK_VALUE = 1
# Rows of the 10 arc-second grid read per iteration of the streaming stock pass.
STOCK_CHUNK_ROWS = 2048
# Habitat coverage is measured by burning the extent this many times finer in each direction, so
# a cell's share is resolved to 1/100th of it.
COVERAGE_SUPERSAMPLE = 10
# Coarse cells per side of a coverage tile. At that sampling the fine tile is 2560x2560 int64,
# about 52 MB, which is what keeps a global pass inside memory.
COVERAGE_TILE_COARSE = 256
# Pixels per side of a read window in the seagrass extent tiles. Most of a tile is open water, and
# a window this size is skipped before any region is burned; the densest tile ran in 5.5 s at 1024
# against 8.0 s at 4096.
SEAGRASS_WINDOW_PIXELS = 1024


def publish_inputs(p):
    """Every config-consuming task's first line. TWO SURFACES BY DESIGN: the marine r566
    correspondence (the gep_regions_input_path cell) is the per-EEZ aggregation surface with its
    gep_regions_id_col; the r264 correspondence (initialize_country_paths) is the iso3_r250-
    collapse crosswalk gep_calculation reads. Science inputs (habitat extents, SOC rasters,
    precipitation, the soil depth and the seagrass densities) hydrate from es_parameters -- the
    optional ones resolve permissively and the stock tasks keep their documented fallbacks.
    ha_per_cell is pyramid infrastructure, in code."""
    utilities.hydrate_es_config(p, 'coastal_carbon', log=hb.log)
    utilities.hydrate_es_parameters(p, 'coastal_carbon', log=hb.log)
    utilities.initialize_country_paths(p, simplified='30sec')
    utilities.initialize_pyramid_paths(p)
    if not hasattr(p, 'results'):
        p.results = {}
    return p


# ============================================================================
# Raster reads and writes
# ============================================================================

def _rasterize_to_template(vector_path, template_raster_path, out_path, field=None,
                           gdal_type=MASK_GDAL_TYPE, nodata=RASTER_NDV, all_touched=True):
    """Burn polygons from `vector_path` onto the grid of `template_raster_path`.

    Defaults produce a binary mask. Pass `field` to burn a numeric attribute (e.g. a region id).

    hazelbean owns the burn, so this module grids a vector the same way the rest of the stack
    does. It burns in place rather than reprojecting, so a vector arriving in another CRS would
    land silently in the wrong cells, which is why the CRS is checked rather than assumed.
    """
    with rasterio.open(template_raster_path) as src:
        template_crs = src.crs
    vector_crs = gpd.read_file(vector_path, rows=1).crs
    if vector_crs != template_crs:
        raise ValueError(
            f'{os.path.basename(vector_path)} is in {vector_crs} but {os.path.basename(template_raster_path)} '
            f'is in {template_crs}. Reproject the vector before burning it.')

    hb.rasterize_to_match(
        input_vector_path=vector_path,
        match_raster_path=template_raster_path,
        output_raster_path=out_path,
        burn_column_name=field,
        burn_values=None if field else [MASK_VALUE],
        datatype=gdal_type,
        ndv=nodata,
        all_touched=all_touched,
    )
    return out_path


def _align_to_template(src_raster_path, template_raster_path, out_path,
                       resampling='average', dtype=None):
    """Reproject and resample a source raster onto the grid of a template raster.

    Brings external products (Sanderman 2018 SOC, Maxwell 2024 MarSOC, precipitation) onto the
    ha_per_cell grid so they can be read in lockstep with the windowed streaming pass. An
    existing out_path is returned as-is.
    """
    if hb.path_exists(out_path):
        return out_path

    with rasterio.open(template_raster_path) as tmpl, \
         rasterio.open(src_raster_path) as src:
        out_meta = tmpl.meta.copy()
        out_meta.update({
            'count': 1,
            'dtype': dtype or src.dtypes[0],
            'nodata': src.nodata if src.nodata is not None else 0,
            'compress': 'lzw',
        })
        with rasterio.open(out_path, 'w', **out_meta) as dst:
            rasterio.warp.reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=tmpl.transform,
                dst_crs=tmpl.crs,
                resampling=getattr(Resampling, resampling),
            )
    return out_path


def _build_country_id_raster_if_needed(p):
    """Rasterize the region vector to a uint16 id grid aligned with ha_per_cell.

    Cross-task reuse: if a previous task already set p.country_id_raster_path and the file
    exists, reuse it (no rebuild and no path overwrite). Otherwise the raster is built into the
    calling task's p.cur_dir, so the first stock task to run owns the file and downstream
    callers find it via the shared p.country_id_raster_path attribute.
    """
    existing = getattr(p, 'country_id_raster_path', None)
    if existing and hb.path_exists(existing):
        return
    p.country_id_raster_path = os.path.join(p.cur_dir, "country_id_raster_10sec.tif")
    if not hb.path_exists(p.country_id_raster_path):
        _rasterize_to_template(
            vector_path=p.gep_regions_input_path,
            template_raster_path=p.ha_per_cell_10sec_path,
            out_path=p.country_id_raster_path,
            field=p.gep_regions_id_col,
            gdal_type=REGION_ID_GDAL_TYPE,
            nodata=RASTER_NDV,
            all_touched=False,
        )


def _habitat_mask(p, vector_path, mask_file_name):
    """The habitat extent burned onto the ha_per_cell grid as a 0/1 mask, built once.

    all_touched, so this is every cell the extent reaches. That makes it a superset of the true
    extent, which is what lets it pick the tiles _habitat_coverage_fraction has to visit.
    """
    mask_path = os.path.join(p.cur_dir, mask_file_name)
    if not hb.path_exists(mask_path):
        _rasterize_to_template(
            vector_path=vector_path,
            template_raster_path=p.ha_per_cell_10sec_path,
            out_path=mask_path,
            all_touched=True,
        )
    return mask_path


def _habitat_coverage_fraction(p, vector_path, mask_path, fraction_file_name):
    """The share of each cell the habitat actually covers, in [0, 1], built once.

    A 0/1 mask hands a cell its whole area whenever a polygon clips any part of it. These
    habitats are ribbons about one cell wide, so nearly every cell is a boundary cell and the
    extent roughly doubles: mangrove measures 29.08 Mha that way against 14.69 Mha of polygon.
    Burning the extent on a grid COVERAGE_SUPERSAMPLE times finer and taking the share that comes
    back set gives each cell the area the habitat really occupies.

    Tiled because that sampling over the whole 10 arc-second grid would be 840 billion cells.
    Only tiles the all_touched mask reaches are burned, since it is a superset of the extent.
    """
    fraction_path = os.path.join(p.cur_dir, fraction_file_name)
    if hb.path_exists(fraction_path):
        return fraction_path

    gdf = gpd.read_file(vector_path)
    with rasterio.open(p.ha_per_cell_10sec_path) as template:
        if gdf.crs != template.crs:
            raise ValueError(
                f'{os.path.basename(vector_path)} is in {gdf.crs} but the template is in '
                f'{template.crs}. Reproject the vector before measuring coverage.')
        meta = template.meta.copy()
        meta.update({'count': 1, 'dtype': 'float32', 'nodata': 0.0, 'compress': 'deflate',
                     'tiled': True, 'blockxsize': 512, 'blockysize': 512, 'BIGTIFF': 'YES'})
        sindex = gdf.sindex
        step = COVERAGE_TILE_COARSE
        factor = COVERAGE_SUPERSAMPLE
        tiles_burned = 0

        with rasterio.open(mask_path) as mask_src, \
             rasterio.open(fraction_path, 'w', **meta) as dst:
            for row_off in tqdm(range(0, template.height, step), desc='Coverage'):
                for col_off in range(0, template.width, step):
                    n_rows = min(step, template.height - row_off)
                    n_cols = min(step, template.width - col_off)
                    window = rasterio.windows.Window(col_off, row_off, n_cols, n_rows)
                    if not mask_src.read(1, window=window).any():
                        continue

                    bounds = rasterio.windows.bounds(window, template.transform)
                    candidates = gdf.iloc[sorted(sindex.query(box(*bounds)))]
                    if candidates.empty:
                        continue

                    # rasterio rather than hazelbean here: the fine grid exists only for this
                    # tile and never becomes a file, and hb.rasterize_to_match writes a whole
                    # raster matching a template on disk.
                    fine = rasterio.features.rasterize(
                        shapes=((geom, 1) for geom in candidates.geometry if geom is not None),
                        out_shape=(n_rows * factor, n_cols * factor),
                        transform=rasterio.transform.from_bounds(
                            *bounds, width=n_cols * factor, height=n_rows * factor),
                        fill=0, dtype='int64', all_touched=False)

                    fraction = hb.calc_proportion_of_coarse_res_with_valid_fine_res(
                        np.zeros((n_rows, n_cols), dtype=np.float64), fine)
                    dst.write(fraction.astype('float32'), 1, window=window)
                    tiles_burned += 1

    hb.log(f'{fraction_file_name}: coverage measured over {tiles_burned} tiles at '
           f'1/{factor ** 2} of a cell')
    return fraction_path


def _aligned_density_input(p, src_path, aligned_file_name, label):
    """An optional external raster on the ha_per_cell grid, or None when the run has none.

    None sends the density function to its latitude fallback, which produces a different
    number, so which source was used is logged either way.
    """
    if not (src_path and hb.path_exists(src_path)):
        hb.log(f'  {label}: not supplied, using the latitude fallback')
        return None
    hb.log(f'  {label}: {src_path}')
    return _align_to_template(src_path, p.ha_per_cell_10sec_path,
                              os.path.join(p.cur_dir, aligned_file_name),
                              resampling='average', dtype='float32')


def _stock_by_region(p, ecosystem, coverage_path, density_func, extra_raster_paths):
    """Stream the per-pixel stock pass over the 10 arc-second grid, region by region.

    This function owns the reads; the arithmetic on each block is ccf.pixel_stock_sums. A block
    with no habitat is skipped before the other rasters are read.
    """
    gdf_regions = gpd.read_file(p.gep_regions_input_path)
    region_ids = gdf_regions[p.gep_regions_id_col].dropna().astype(int).unique()
    n_region_ids = int(np.max(region_ids)) + 1
    sums = {pool: np.zeros(n_region_ids, dtype=np.float64)
            for pool in ('agb', 'bgb', 'soil', 'total')}

    with contextlib.ExitStack() as stack:
        src_ha = stack.enter_context(rasterio.open(p.ha_per_cell_10sec_path))
        src_coverage = stack.enter_context(rasterio.open(coverage_path))
        src_region = stack.enter_context(rasterio.open(p.country_id_raster_path))
        extra_srcs = {name: stack.enter_context(rasterio.open(path))
                      for name, path in extra_raster_paths.items()}

        if not src_ha.crs.is_geographic:
            raise ValueError(
                f"ha_per_cell raster must be in a geographic CRS (degrees); got {src_ha.crs}")
        height, width = src_ha.height, src_ha.width

        for row_off in tqdm(range(0, height, STOCK_CHUNK_ROWS), desc=f'Stock {ecosystem}'):
            rows = min(STOCK_CHUNK_ROWS, height - row_off)
            window = rasterio.windows.Window(0, row_off, width, rows)
            coverage_block = src_coverage.read(1, window=window).astype(np.float64)
            if not coverage_block.any():
                continue

            extras = {}
            for name, src in extra_srcs.items():
                arr = src.read(1, window=window).astype(np.float64)
                if src.nodata is not None:
                    arr = np.where(arr == src.nodata, np.nan, arr)
                extras[name] = arr

            latitudes = np.array(
                [rasterio.transform.xy(src_ha.transform, row, 0, offset='center')[1]
                 for row in range(row_off, row_off + rows)], dtype=np.float64)
            block_sums = ccf.pixel_stock_sums(
                coverage_block=coverage_block,
                region_id_block=src_region.read(1, window=window),
                ha_per_cell_block=src_ha.read(1, window=window).astype(np.float64),
                latitude_block=np.broadcast_to(latitudes[:, None], (rows, width)),
                density_func=density_func,
                n_region_ids=n_region_ids,
                extras=extras,
            )
            for pool in sums:
                sums[pool] += block_sums[pool]

    return ccf.stock_by_region_frame(region_ids, sums, ecosystem, p.gep_regions_id_col)


def _write_storage_value(p, ecosystem, stock_csv_path, out_path):
    """Write one habitat's per-region storage value: every pool's stock at the rental SCC.

    Splits the GEP calculation into two stages explicitly. The physical stage (per-pool stock in
    Mg C) is produced upstream by the habitat's stock task and read here; the GEP stage prices
    each pool at the base-year rental SCC. The written CSV holds the stock columns, the value
    columns, the year, and the price it was valued at.
    """
    if hb.path_exists(out_path):
        hb.log(f'{ecosystem} storage value: skipped (output exists at {out_path})')
        return
    if not hb.path_exists(stock_csv_path):
        raise FileNotFoundError(
            f'{ecosystem} storage value: stock CSV missing ({stock_csv_path!r}) -- the stock '
            f'task upstream in the same tree must produce it; a silent skip here would drop the '
            f'habitat from the coastal total without a trace.')

    rental_scc = ccf.rental_price_for_year(
        pd.read_excel(p.gep_price_input_path), p.gep_base_year, p.gep_price_convention)
    stock_to_value = ccf.stock_to_value_columns(ecosystem)
    df_value = ccf.apply_rental_price(hb.df_read(stock_csv_path), stock_to_value, rental_scc,
                                      p.gep_base_year, p.gep_price_convention)
    df_value.to_csv(out_path, index=False)

    pool_items = list(stock_to_value.items())
    total_stock_col, total_value_col = pool_items[-1]
    for stock_col, value_col in pool_items[:-1]:
        hb.log(f'  {ecosystem} {stock_col}: {df_value[stock_col].sum():,.0f} Mg C, '
               f'value ${df_value[value_col].sum():,.2f}')
    total_stock = df_value[total_stock_col].sum()
    hb.log(f'{ecosystem} TOTAL stock ({p.gep_base_year}): {total_stock:,.0f} Mg C '
           f'({total_stock / ccf.MG_C_PER_PG:.3f} Pg C)')
    hb.log(f'{ecosystem} TOTAL storage value ({p.gep_base_year}): '
           f'${df_value[total_value_col].sum():,.2f}')
    hb.log(f'Saved {ecosystem} storage value: {out_path}')


# ============================================================================
# Mangrove: Global Mangrove Watch extent
# ============================================================================

def mangrove_area_within_countries(p):
    """Mangrove area within each region's marine EEZ, from the Global Mangrove Watch extent."""
    publish_inputs(p)
    p.mangrove_area_by_countries_base_year_path = os.path.join(
        p.cur_dir, "mangrove_area_by_countries2019.gpkg"
    )
    csv_path = p.mangrove_area_by_countries_base_year_path.replace('.gpkg', '.csv')
    if not p.run_this:
        return

    gdf_regions = gpd.read_file(p.gep_regions_input_path)
    gdf_mangroves = gpd.read_file(p.mangrove_vector_path).to_crs(gdf_regions.crs)
    hb.log(f'Loaded {len(gdf_mangroves)} mangrove polygons, {len(gdf_regions)} marine regions')

    pieces = ccf.add_equal_area_ha(ccf.intersect_features_with_regions(
        gdf_mangroves, gdf_regions, desc='Intersecting mangroves with countries'))
    pieces.to_file(os.path.join(p.cur_dir, "mangroves_within_countries2019.gpkg"), driver="GPKG")

    # 'left': the mangrove table carries only the regions mangroves reach. The regions it omits
    # come back in as zero when combined_ecosystem_areas merges the three habitats.
    area = ccf.area_by_region(pieces, gdf_regions, p.gep_regions_id_col, how='left')
    area.to_file(p.mangrove_area_by_countries_base_year_path, driver="GPKG")
    area.drop(columns=['geometry']).to_csv(csv_path, index=False)
    hb.log(f'Saved mangrove area by countries: {p.mangrove_area_by_countries_base_year_path}')


def mangrove_carbon_stock(p):
    """Per-region mangrove carbon stock (Mg C) via per-pixel density x ha_per_cell.

    AGB: Hamilton & Friess (2018) latitude regression.
    BGB: AGB x the IPCC 2014 zone-specific ratio (uses p.precipitation_path when set).
    SOC: the Sanderman 2018 raster at p.mangrove_soc_path, else the latitude fallback, both for
         the top 1 m and taken as the share p.coastal_carbon_soil_depth_cm /
         p.coastal_carbon_soil_source_depth_cm of it.
    """
    publish_inputs(p)
    p.mangrove_carbon_stock_path = os.path.join(
        p.cur_dir, "mangrove_carbon_stock_by_countries2019.csv"
    )
    if not p.run_this:
        return

    _build_country_id_raster_if_needed(p)
    mask_path = _habitat_mask(p, p.mangrove_vector_path, "mangrove_mask_10sec.tif")
    coverage_path = _habitat_coverage_fraction(
        p, p.mangrove_vector_path, mask_path, "mangrove_coverage_10sec.tif")

    extras = {}
    for kwarg, src_path, aligned_file_name, label in (
            ('soc_arr', getattr(p, 'mangrove_soc_path', None),
             "sanderman_soc_aligned_10sec.tif", 'Mangrove SOC (Sanderman 2018)'),
            ('precipitation_arr', getattr(p, 'precipitation_path', None),
             "precipitation_aligned_10sec.tif", 'Mangrove BGB ratio precipitation')):
        aligned_path = _aligned_density_input(p, src_path, aligned_file_name, label)
        if aligned_path:
            extras[kwarg] = aligned_path

    density_func = functools.partial(
        ccf.calculate_mangrove_density_array,
        soil_depth_share=p.coastal_carbon_soil_depth_cm / p.coastal_carbon_soil_source_depth_cm)
    df_stock = _stock_by_region(p, 'mangrove', coverage_path, density_func, extras)
    df_stock.to_csv(p.mangrove_carbon_stock_path, index=False)
    hb.log(f'Saved mangrove carbon stock by countries: {p.mangrove_carbon_stock_path}')


def mangrove_storage_value(p):
    """Per-region mangrove storage value (USD), per pool: each pool's stock x the rental SCC."""
    publish_inputs(p)
    p.mangrove_storage_value_path = os.path.join(
        p.cur_dir, "mangrove_storage_value_by_countries2019.csv")
    if not p.run_this:
        return
    _write_storage_value(p, 'mangrove', p.mangrove_carbon_stock_path,
                         p.mangrove_storage_value_path)


# ============================================================================
# Salt marsh: global salt marsh extent
# ============================================================================

def salt_marsh_area_within_countries(p):
    """Salt marsh area within each region's marine EEZ.

    The area is the equal-area geometric area of each clipped piece, the same measure the other
    two habitats use.
    """
    publish_inputs(p)
    p.salt_marsh_area_by_countries_base_year_path = os.path.join(
        p.cur_dir, "salt_marsh_area_by_countries2019.gpkg"
    )
    csv_path = p.salt_marsh_area_by_countries_base_year_path.replace('.gpkg', '.csv')
    if not p.run_this:
        return

    gdf_regions = gpd.read_file(p.gep_regions_input_path)
    gdf_salt_marsh = gpd.read_file(p.salt_marsh_vector_path).to_crs(gdf_regions.crs)
    hb.log(f'Loaded {len(gdf_salt_marsh)} salt marsh polygons, {len(gdf_regions)} marine regions')

    # The area is the equal-area geometric area from add_equal_area_ha, as for the other two
    # habitats. A zonal pass over ha_per_cell would take hours and nothing reads it.
    pieces = ccf.add_equal_area_ha(ccf.intersect_features_with_regions(
        gdf_salt_marsh, gdf_regions, desc='Intersecting salt marsh with countries'))

    area = ccf.area_by_region(pieces, gdf_regions, p.gep_regions_id_col, how='right')
    area.to_file(p.salt_marsh_area_by_countries_base_year_path, driver="GPKG")
    area.drop(columns=['geometry']).to_csv(csv_path, index=False)
    hb.log(f'Saved salt marsh area by countries: '
           f'{p.salt_marsh_area_by_countries_base_year_path}')


def salt_marsh_carbon_stock(p):
    """Per-region salt marsh carbon stock (Mg C) via per-pixel density x ha_per_cell.

    AGB: Chmura et al. 2003 median with the tropical productivity boost.
    BGB: AGB x 2.5 (extensive root systems).
    SOC: the Maxwell et al. 2024 MarSOC raster at p.salt_marsh_soc_path, else the latitude
         fallback, both for the top 1 m and taken at the configured depth share as for mangrove.
    """
    publish_inputs(p)
    p.salt_marsh_carbon_stock_path = os.path.join(
        p.cur_dir, "salt_marsh_carbon_stock_by_countries2019.csv"
    )
    if not p.run_this:
        return

    _build_country_id_raster_if_needed(p)
    mask_path = _habitat_mask(p, p.salt_marsh_vector_path, "salt_marsh_mask_10sec.tif")
    coverage_path = _habitat_coverage_fraction(
        p, p.salt_marsh_vector_path, mask_path, "salt_marsh_coverage_10sec.tif")

    extras = {}
    aligned_path = _aligned_density_input(
        p, getattr(p, 'salt_marsh_soc_path', None), "maxwell_marsoc_aligned_10sec.tif",
        'Salt marsh SOC (Maxwell 2024 MarSOC)')
    if aligned_path:
        extras['soc_arr'] = aligned_path

    density_func = functools.partial(
        ccf.calculate_salt_marsh_density_array,
        soil_depth_share=p.coastal_carbon_soil_depth_cm / p.coastal_carbon_soil_source_depth_cm)
    df_stock = _stock_by_region(p, 'salt_marsh', coverage_path, density_func, extras)
    df_stock.to_csv(p.salt_marsh_carbon_stock_path, index=False)
    hb.log(f'Saved salt marsh carbon stock by countries: {p.salt_marsh_carbon_stock_path}')


def salt_marsh_storage_value(p):
    """Per-region salt marsh storage value (USD), per pool: each pool's stock x the rental SCC."""
    publish_inputs(p)
    p.salt_marsh_storage_value_path = os.path.join(
        p.cur_dir, "salt_marsh_storage_value_by_countries2019.csv")
    if not p.run_this:
        return
    _write_storage_value(p, 'salt_marsh', p.salt_marsh_carbon_stock_path,
                         p.salt_marsh_storage_value_path)


# ============================================================================
# Seagrass: GlobalSeagrass 2019-2020, 10 m presence tiles
#
# The extent is a raster of presence with no genus, so the area is measured pixel by pixel on the
# source grid and every mapped pixel takes the same configured densities. A missing extent RAISES:
# seagrass is excluded by building the tree with include_seagrass=False, never by letting a data
# gap silently understate the coastal total.
# ============================================================================

def _seagrass_tile_area(tif_path, gdf_regions, id_col):
    """One extent tile's seagrass hectares per marine region, and what the tile holds in all.

    A pixel belongs to every region that holds its centre (all_touched=False), and its area is the
    WGS84 ellipsoid area of its row. The tile is read in SEAGRASS_WINDOW_PIXELS windows, a window
    with no seagrass is skipped before anything is burned, and a region that covers a whole window
    holds every pixel in it without a burn.

    Returns (area_ha, shared_ha, diagnostic): the first two map region id -> hectares as
    ccf.presence_area_by_zone defines them, summed over the tile's windows.
    """
    start = time.monotonic()
    area_ha, shared_ha = defaultdict(float), defaultdict(float)
    diagnostic = {'tile': os.path.basename(tif_path), 'positive_pixels': 0, 'presence_ha': 0.0,
                  'unique_ha': 0.0, 'overlap_counted_ha': 0.0}
    with rasterio.open(tif_path) as src:
        cell_degrees = src.transform.a
        if src.crs.to_epsg() != 4326 or src.transform.e != -cell_degrees:
            raise ValueError(f'{os.path.basename(tif_path)}: the area per row needs square cells in '
                             f'EPSG:4326, found {src.crs} with cells {src.res}')
        tile_box = box(*src.bounds)
        regions = gdf_regions.iloc[gdf_regions.sindex.query(tile_box, predicate='intersects')]
        geometries = shapely.intersection(regions.geometry.to_numpy(), tile_box)
        region_ids = regions[id_col].astype(int).to_numpy()
        tree = shapely.STRtree(geometries)

        for row_off in range(0, src.height, SEAGRASS_WINDOW_PIXELS):
            for col_off in range(0, src.width, SEAGRASS_WINDOW_PIXELS):
                window = rasterio.windows.Window(
                    col_off, row_off, min(SEAGRASS_WINDOW_PIXELS, src.width - col_off),
                    min(SEAGRASS_WINDOW_PIXELS, src.height - row_off))
                presence = src.read(1, window=window) > 0
                if not presence.any():
                    continue
                bounds = rasterio.windows.bounds(window, src.transform)
                window_box = box(*bounds)
                masks = {}
                for i in tree.query(window_box, predicate='intersects'):
                    if geometries[i].covers(window_box):
                        masks[region_ids[i]] = np.ones(presence.shape, dtype=bool)
                        continue
                    # Only a polygonal piece can hold a pixel centre. A region that merely touches
                    # the window clips to a line or a point, which GDAL burns as a run of pixels.
                    pieces = [piece for piece in shapely.get_parts(geometries[i].intersection(window_box))
                              if piece.geom_type in ('Polygon', 'MultiPolygon')]
                    if pieces:
                        # rasterio rather than hazelbean here: the burn is one window of a source
                        # tile and never becomes a file, and hb.rasterize_to_match writes a whole
                        # raster matching a template on disk.
                        masks[region_ids[i]] = rasterio.features.rasterize(
                            [(piece, 1) for piece in pieces], out_shape=presence.shape,
                            transform=rasterio.windows.transform(window, src.transform),
                            fill=0, dtype='uint8', all_touched=False).astype(bool)

                block = ccf.presence_area_by_zone(
                    presence, masks,
                    hb.ha_per_cell_rows(bounds[3], cell_degrees, presence.shape[0])[:, None])
                for region_id, hectares in block['area_ha'].items():
                    area_ha[region_id] += hectares
                    shared_ha[region_id] += block['shared_ha'][region_id]
                diagnostic['positive_pixels'] += int(np.count_nonzero(presence))
                for key in ('presence_ha', 'unique_ha', 'overlap_counted_ha'):
                    diagnostic[key] += block[key]

    diagnostic['region_summed_ha'] = sum(area_ha.values())
    diagnostic['seconds'] = time.monotonic() - start
    return area_ha, shared_ha, diagnostic


def seagrass_area_within_countries(p):
    """Seagrass hectares within each marine region, from the GlobalSeagrass 10 m presence tiles.

    Every r566 region is measured, the same surface the other two habitats are clipped to; the
    `_EEZ` rows are the ones gep_calculation values.

    Outputs
    -------
    seagrass_area_by_countries2019.{gpkg,csv}
        Per marine region, area_ha and overlap_ha, the part of it another region also holds.
    seagrass_tiles2019.csv
        Per tile: positive pixels, hectares present, hectares some region holds (once), the
        overlap-counted hectares, the regions' sum, and seconds.

    The tables are written only after the last tile, each under a temporary name and then renamed
    into place, so an interrupted run leaves nothing a later task could read as finished.
    """
    publish_inputs(p)
    p.seagrass_area_by_countries_base_year_path = os.path.join(
        p.cur_dir, "seagrass_area_by_countries2019.gpkg"
    )
    csv_path = p.seagrass_area_by_countries_base_year_path.replace('.gpkg', '.csv')
    p.seagrass_tiles_path = os.path.join(p.cur_dir, "seagrass_tiles2019.csv")
    if not p.run_this:
        return

    extent_dir = getattr(p, 'seagrass_extent_dir_path', None)
    tile_paths = sorted(glob.glob(os.path.join(extent_dir, '*.tif'))) if hb.path_exists(extent_dir) else []
    if not tile_paths:
        raise FileNotFoundError(
            "seagrass extent tiles not found (p.seagrass_extent_dir_path=%r). A built seagrass tree "
            "with no data would silently understate the coastal GEP total; to exclude seagrass, "
            "build the tree with include_seagrass=False instead." % (extent_dir,))

    id_col = p.gep_regions_id_col
    gdf_regions = gpd.read_file(p.gep_regions_input_path)
    hb.log(f'Seagrass: {len(tile_paths)} extent tiles in {extent_dir}, {len(gdf_regions)} marine regions')

    area_ha, shared_ha, diagnostics = defaultdict(float), defaultdict(float), []
    for tif_path in tqdm(tile_paths, desc='Seagrass tiles'):
        tile_area, tile_shared, diagnostic = _seagrass_tile_area(tif_path, gdf_regions, id_col)
        for region_id, hectares in tile_area.items():
            area_ha[region_id] += hectares
            shared_ha[region_id] += tile_shared[region_id]
        diagnostics.append(diagnostic)

    df_area = pd.DataFrame({id_col: list(area_ha), 'area_ha': list(area_ha.values()),
                            'overlap_ha': [shared_ha[region_id] for region_id in area_ha]})
    area = ccf.area_by_region(df_area, gdf_regions, id_col, how='right')
    area = area.merge(df_area[[id_col, 'overlap_ha']], on=id_col, how='left')
    area['overlap_ha'] = area['overlap_ha'].fillna(0.0)
    df_tiles = pd.DataFrame(diagnostics)

    partial = {path: '%s_partial%s' % os.path.splitext(path)
               for path in (p.seagrass_area_by_countries_base_year_path, csv_path, p.seagrass_tiles_path)}
    area.to_file(partial[p.seagrass_area_by_countries_base_year_path], driver='GPKG')
    area.drop(columns=['geometry']).to_csv(partial[csv_path], index=False)
    df_tiles.to_csv(partial[p.seagrass_tiles_path], index=False)
    for path, partial_path in partial.items():
        os.replace(partial_path, path)

    totals = df_tiles[['presence_ha', 'unique_ha', 'overlap_counted_ha', 'region_summed_ha']].sum()
    hb.log(f'Seagrass: {totals["presence_ha"]:,.3f} ha mapped, {totals["unique_ha"]:,.3f} ha inside '
           f'a marine region, {totals["overlap_counted_ha"]:,.3f} ha counted again by an overlapping '
           f'region; {df_tiles["seconds"].sum():,.0f} s over {len(df_tiles)} tiles')
    hb.log(f'Saved seagrass area by countries: {p.seagrass_area_by_countries_base_year_path}')


def seagrass_carbon_stock(p):
    """Per-region seagrass carbon stock (Mg C): each region's mapped hectares times the configured
    densities, p.seagrass_biomass_c_mg_per_ha split by p.seagrass_agb_fraction, and
    p.seagrass_soil_c_mg_per_ha for the configured soil depth.
    """
    publish_inputs(p)
    p.seagrass_carbon_stock_path = os.path.join(
        p.cur_dir, "seagrass_carbon_stock_by_countries2019.csv"
    )
    if not p.run_this:
        return

    area_gpkg = getattr(p, 'seagrass_area_by_countries_base_year_path', None)
    area_csv = area_gpkg.replace('.gpkg', '.csv') if area_gpkg else None
    if not hb.path_exists(area_csv):
        raise FileNotFoundError(
            "seagrass area table missing (%r) -- seagrass_area_within_countries must run first "
            "(same tree). Skipping here would silently zero seagrass in the total." % (area_csv,))

    densities = ccf.seagrass_pool_densities(
        p.seagrass_biomass_c_mg_per_ha, p.seagrass_agb_fraction, p.seagrass_soil_c_mg_per_ha)
    hb.log('Seagrass densities (Mg C/ha): ' + ', '.join(
        f'{pool} {value:g}' for pool, value in densities.items()))
    df_stock = ccf.seagrass_stock_by_region(hb.df_read(area_csv), p.gep_regions_id_col, densities)
    df_stock.to_csv(p.seagrass_carbon_stock_path, index=False)

    total_stock = df_stock['seagrass_total_c_stock_mg'].sum()
    hb.log(f'Saved seagrass carbon stock by countries: {p.seagrass_carbon_stock_path}  '
           f'(global total {total_stock:,.0f} Mg C, {total_stock / ccf.MG_C_PER_PG:.4f} Pg C)')


def seagrass_storage_value(p):
    """Per-region seagrass storage value (USD), per pool: each pool's stock x the rental SCC."""
    publish_inputs(p)
    p.seagrass_storage_value_path = os.path.join(
        p.cur_dir, "seagrass_storage_value_by_countries2019.csv")
    if not p.run_this:
        return
    _write_storage_value(p, 'seagrass', getattr(p, 'seagrass_carbon_stock_path', None),
                         p.seagrass_storage_value_path)


# ============================================================================
# Cross-habitat combination and GEP
# ============================================================================

def combined_ecosystem_areas(p):
    """Mangrove, salt marsh and seagrass areas plus carbon stocks in one per-region table.

    Mangrove and salt marsh are required -- a missing table there is a broken run, not a
    zero. Seagrass is the optional habitat (include_seagrass on the tree builder), and its
    absence contributes zero columns.
    """
    publish_inputs(p)
    p.combined_area_path = os.path.join(p.cur_dir, "combined_ecosystem_areas.csv")
    if not p.run_this:
        return

    area_frames = {
        'mangrove': hb.df_read(
            p.mangrove_area_by_countries_base_year_path.replace('.gpkg', '.csv')),
        'salt_marsh': hb.df_read(
            p.salt_marsh_area_by_countries_base_year_path.replace('.gpkg', '.csv')),
    }
    stock_frames = {
        'mangrove': hb.df_read(p.mangrove_carbon_stock_path),
        'salt_marsh': hb.df_read(p.salt_marsh_carbon_stock_path),
    }

    seagrass_area_gpkg = getattr(p, 'seagrass_area_by_countries_base_year_path', None)
    seagrass_area_csv = (
        seagrass_area_gpkg.replace('.gpkg', '.csv') if seagrass_area_gpkg else None
    )
    if hb.path_exists(seagrass_area_csv):
        area_frames['seagrass'] = hb.df_read(seagrass_area_csv)
    else:
        hb.log('combined_ecosystem_areas: seagrass area CSV not found '
               f'({seagrass_area_csv!r}); seagrass area set to 0.')

    seagrass_stock_path = getattr(p, 'seagrass_carbon_stock_path', None)
    if hb.path_exists(seagrass_stock_path):
        stock_frames['seagrass'] = hb.df_read(seagrass_stock_path)
    elif 'seagrass' in area_frames:
        hb.log('combined_ecosystem_areas: seagrass area present but seagrass stock CSV missing '
               f'({seagrass_stock_path!r}); seagrass stock columns set to 0.')

    df_regions = gpd.read_file(p.gep_regions_input_path).drop(columns=['geometry'])
    df_combined = ccf.combine_ecosystem_areas_and_stocks(
        area_frames, stock_frames, df_regions, p.gep_regions_id_col)
    df_combined.to_csv(p.combined_area_path, index=False)
    hb.log(f'Saved combined ecosystem areas: {p.combined_area_path}')


def gep_calculation(p):
    """GEP for coastal blue carbon: storage-only, per-habitat stock (Mg C) x rental SCC ($/Mg C).

    Two outputs in p.cur_dir:
      - gep_by_country_base_year_r566.{csv,gpkg}: the full marine-surface detail (per-habitat
        area, stock pools, storage values), one row per eemarine region.
      - gep_by_country_base_year.csv: the iso3_r250 final, built by keeping the marine `_EEZ`
        rows and joining them onto the canonical r264 row per country.
    """
    publish_inputs(p)
    service_results = {}
    p.results['coastal_carbon'] = service_results
    service_results['gep_by_country_base_year_r566'] = os.path.join(
        p.cur_dir, "gep_by_country_base_year_r566.csv")
    service_results['gep_by_country_base_year'] = os.path.join(
        p.cur_dir, "gep_by_country_base_year.csv")
    # Only register results this task actually writes. Per-year results (gep_by_country_year,
    # gep_by_year) belong to a multi-year run and are registered there, not in this base-year
    # valuation.

    final_csv = service_results['gep_by_country_base_year']
    if not p.run_this:
        return p
    reason = utilities.reuse_reason(p, 'coastal_carbon', list(service_results.values()))
    if reason is None:
        hb.log('coastal_carbon reuses its gep outputs: the signature is unchanged.')
        return
    hb.log('coastal_carbon recomputes its gep, %s' % reason)

    r566_csv = service_results['gep_by_country_base_year_r566']
    r566_gpkg = r566_csv.replace('.csv', '.gpkg')

    # Stage 1: the marine surface (r566).
    if hb.path_all_exist(r566_csv, r566_gpkg):
        df_gep = hb.df_read(r566_csv)
        hb.log("gep_calculation: r566 cached, reusing.")
    else:
        hb.log("gep_calculation: computing r566 storage values "
               "(mangrove + salt marsh + seagrass)...")
        df_gep = ccf.coastal_carbon_storage_value_frame(
            df_areas=hb.df_read(p.combined_area_path),
            df_price=pd.read_excel(p.gep_price_input_path)[['year', p.gep_price_convention]],
            value_frames={ecosystem: hb.df_read(getattr(p, f'{ecosystem}_storage_value_path'))
                          for ecosystem in ccf.COASTAL_ECOSYSTEMS},
            id_col=p.gep_regions_id_col,
            base_year=p.gep_base_year,
        )

        hb.df_write(df_gep, r566_csv)
        gdf_regions = gpd.read_file(p.gep_regions_input_path)
        gdf_regions.merge(df_gep, on=p.gep_regions_id_col, how='right').to_file(
            r566_gpkg, driver='GPKG')

        for ecosystem in ccf.COASTAL_ECOSYSTEMS + ('coastal_carbon',):
            hb.log(f'r566 {ecosystem} storage value ({p.gep_base_year}): '
                   f'${df_gep[f"{ecosystem}_storage_value"].sum():,.2f}')

    # Stage 2: the iso3_r250 final.
    df_r250_final = ccf.collapse_to_iso3_r250(
        ccf.eez_storage_value_by_iso3(df_gep), hb.df_read(p.df_countries_csv_path))
    # The same shape as every other service's country table: the shared attributes, the year, and
    # a column named for the service: the upstream frame calls it `value`, which the account
    # cannot read the way it reads the rest.
    df_r250_final = df_r250_final.rename(columns={'value': 'coastal_carbon_gep'})
    df_r250_final['year'] = int(p.gep_base_year)
    attributes = [c for c in utilities.GEP_COUNTRY_ATTRIBUTE_COLUMNS if c in df_r250_final.columns]
    df_r250_final = df_r250_final[attributes + ['year', 'coastal_carbon_gep']]
    hb.df_write(df_r250_final, final_csv)
    utilities.write_reuse_signature(p, 'coastal_carbon', list(service_results.values()))
    hb.log('Total coastal_carbon GEP for base year %d: %s over %d countries'
           % (int(p.gep_base_year), format(df_r250_final['coastal_carbon_gep'].sum(), ',.2f'),
              int(df_r250_final['coastal_carbon_gep'].notna().sum())))

    return df_r250_final['coastal_carbon_gep'].sum()


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    if not p.run_this:
        return p
    utilities.render_service_results(p)

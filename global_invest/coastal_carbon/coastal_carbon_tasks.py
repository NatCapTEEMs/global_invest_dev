"""Coastal-carbon GEP tasks: mapped habitat -> per-pixel carbon stock -> storage value.

This layer owns every file read and write. The science it calls lives in
coastal_carbon_functions, which never opens a file, so each step is checkable on hand-built
inputs.

Each habitat runs the same three tasks -- area within regions, carbon stock, storage value --
and the last two tasks combine them and collapse the result to one row per country.
"""
import contextlib
import os

import geopandas as gpd
import hazelbean as hb
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.warp
import rasterio.windows
from rasterio.enums import Resampling
from rasterstats import zonal_stats
from tqdm import tqdm

from global_invest import utilities
from global_invest.coastal_carbon import coastal_carbon_functions as ccf

# The ha_per_cell pyramid every raster stage is gridded on carries this no-data value.
HA_PER_CELL_NDV = -9999.0
# Region-id raster: uint16 spans the eemarine_r566 ids, and 0 means "outside every region".
REGION_ID_RASTER_DTYPE = 'uint16'
RASTER_NDV = 0
# Habitat extent mask: 1 where the habitat polygon covers the cell.
MASK_DTYPE = 'uint8'
MASK_VALUE = 1
# Rows of the 10 arc-second grid read per iteration of the streaming stock pass.
STOCK_CHUNK_ROWS = 2048


def publish_inputs(p):
    """Every config-consuming task's first line. TWO SURFACES BY DESIGN: the marine r566
    correspondence (the gep_regions_input_path cell) is the per-EEZ aggregation surface with its
    gep_regions_id_col; the r264 correspondence (initialize_country_paths) is the iso3_r250-
    collapse crosswalk gep_calculation reads. Science inputs (habitat extents, SOC rasters,
    precipitation) hydrate from es_parameters -- the optional ones resolve permissively and the
    stock tasks keep their documented fallbacks. ha_per_cell is pyramid infrastructure, in code."""
    utilities.hydrate_es_config(p, 'coastal_carbon', log=hb.log)
    utilities.hydrate_es_parameters(p, 'coastal_carbon', log=hb.log)
    utilities.initialize_country_paths(p, simplified='30sec')
    p.ha_per_cell_10sec_path = p.get_path('pyramids', 'ha_per_cell_10sec.tif')
    if not hasattr(p, 'results'):
        p.results = {}
    return p


# ============================================================================
# Raster reads and writes
# ============================================================================

def _rasterize_to_template(vector_path, template_raster_path, out_path, field=None,
                           default_value=MASK_VALUE, dtype=MASK_DTYPE, nodata=RASTER_NDV,
                           all_touched=True):
    """Burn polygons from `vector_path` onto the grid of `template_raster_path`.

    Defaults produce a binary mask. Pass `field` to burn a numeric attribute (e.g. a region id).
    Vectors are reprojected into the template CRS as needed.
    """
    gdf = gpd.read_file(vector_path)
    with rasterio.open(template_raster_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        meta = src.meta.copy()
        meta.update({'count': 1, 'dtype': dtype, 'nodata': nodata, 'compress': 'lzw'})
        if field is None:
            shapes = ((geom, default_value) for geom in gdf.geometry if geom is not None)
        else:
            shapes = (
                (geom, val)
                for geom, val in zip(gdf.geometry, gdf[field])
                if geom is not None and pd.notna(val)
            )
        arr = rasterio.features.rasterize(
            shapes=shapes,
            out_shape=(src.height, src.width),
            fill=nodata,
            transform=src.transform,
            dtype=dtype,
            all_touched=all_touched,
        )
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(arr, 1)
    return out_path


def _align_to_template(src_raster_path, template_raster_path, out_path,
                       resampling='average', dtype=None):
    """Reproject and resample a source raster onto the grid of a template raster.

    Brings external products (Sanderman 2018 SOC, Maxwell 2024 MarSOC, precipitation) onto the
    ha_per_cell grid so they can be read in lockstep with the windowed streaming pass. An
    existing out_path is returned as-is.
    """
    if os.path.exists(out_path):
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
    if existing and os.path.exists(existing):
        return
    p.country_id_raster_path = os.path.join(p.cur_dir, "country_id_raster_10sec.tif")
    if not os.path.exists(p.country_id_raster_path):
        _rasterize_to_template(
            vector_path=p.gep_regions_input_path,
            template_raster_path=p.ha_per_cell_10sec_path,
            out_path=p.country_id_raster_path,
            field=p.gep_regions_id_col,
            dtype=REGION_ID_RASTER_DTYPE,
            nodata=RASTER_NDV,
            all_touched=False,
        )


def _habitat_mask(p, vector_path, mask_file_name):
    """The habitat extent burned onto the ha_per_cell grid as a 0/1 mask, built once."""
    mask_path = os.path.join(p.cur_dir, mask_file_name)
    if not os.path.exists(mask_path):
        _rasterize_to_template(
            vector_path=vector_path,
            template_raster_path=p.ha_per_cell_10sec_path,
            out_path=mask_path,
            all_touched=True,
        )
    return mask_path


def _aligned_density_input(p, src_path, aligned_file_name, label):
    """An optional external raster on the ha_per_cell grid, or None when the run has none.

    None sends the density function to its latitude fallback, which produces a different
    number, so which source was used is logged either way.
    """
    if not (src_path and os.path.exists(src_path)):
        hb.log(f'  {label}: not supplied, using the latitude fallback')
        return None
    hb.log(f'  {label}: {src_path}')
    return _align_to_template(src_path, p.ha_per_cell_10sec_path,
                              os.path.join(p.cur_dir, aligned_file_name),
                              resampling='average', dtype='float32')


def _stock_by_region(p, ecosystem, mask_path, density_func, extra_raster_paths):
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
        src_mask = stack.enter_context(rasterio.open(mask_path))
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
            mask_block = src_mask.read(1, window=window)
            if mask_block.sum() == 0:
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
                mask_block=mask_block,
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
    df_value = ccf.apply_rental_price(pd.read_csv(stock_csv_path), stock_to_value, rental_scc,
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
    SOC: the Sanderman 2018 raster at p.mangrove_soc_path, else the latitude fallback.
    """
    publish_inputs(p)
    p.mangrove_carbon_stock_path = os.path.join(
        p.cur_dir, "mangrove_carbon_stock_by_countries2019.csv"
    )
    if not p.run_this:
        return

    _build_country_id_raster_if_needed(p)
    mask_path = _habitat_mask(p, p.mangrove_vector_path, "mangrove_mask_10sec.tif")

    extras = {}
    for kwarg, src_path, aligned_file_name, label in (
            ('soc_arr', getattr(p, 'mangrove_soc_path', None),
             "sanderman_soc_aligned_10sec.tif", 'Mangrove SOC (Sanderman 2018)'),
            ('precipitation_arr', getattr(p, 'precipitation_path', None),
             "precipitation_aligned_10sec.tif", 'Mangrove BGB ratio precipitation')):
        aligned_path = _aligned_density_input(p, src_path, aligned_file_name, label)
        if aligned_path:
            extras[kwarg] = aligned_path

    df_stock = _stock_by_region(p, 'mangrove', mask_path,
                                ccf.calculate_mangrove_density_array, extras)
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

    The zonal pass over ha_per_cell writes a per-polygon `salt_marsh_with_area.gpkg` for
    inspection; the area that reaches the country table is the equal-area geometric area, the
    same measure the other two habitats use.
    """
    publish_inputs(p)
    p.salt_marsh_area_by_countries_base_year_path = os.path.join(
        p.cur_dir, "salt_marsh_area_by_countries2019.gpkg"
    )
    csv_path = p.salt_marsh_area_by_countries_base_year_path.replace('.gpkg', '.csv')
    if not p.run_this:
        return

    gdf_regions = gpd.read_file(p.gep_regions_input_path)
    gdf_salt_marsh = gpd.read_file(p.salt_marsh_vector_path)
    hb.log(f'Loaded {len(gdf_salt_marsh)} salt marsh polygons, {len(gdf_regions)} marine regions')

    with rasterio.open(p.ha_per_cell_10sec_path) as src:
        ha_per_cell_crs = src.crs
    gdf_salt_marsh = gdf_salt_marsh.to_crs(ha_per_cell_crs)

    stats_list = []
    for geom in tqdm(gdf_salt_marsh.geometry, desc="Computing zonal stats for salt marsh"):
        stats_list.extend(zonal_stats(
            vectors=[geom],
            raster=p.ha_per_cell_10sec_path,
            stats=['sum'],
            geojson_out=True,
            nodata=HA_PER_CELL_NDV,
            all_touched=True,
        ))

    gdf_salt_marsh_zonal = gpd.GeoDataFrame.from_features(stats_list)
    gdf_salt_marsh_zonal.set_crs(ha_per_cell_crs, inplace=True)
    gdf_salt_marsh_zonal.rename(columns={'sum': 'area_ha'}, inplace=True)
    gdf_salt_marsh_zonal = gdf_salt_marsh_zonal.to_crs(gdf_regions.crs)
    gdf_salt_marsh_zonal.to_file(os.path.join(p.cur_dir, "salt_marsh_with_area.gpkg"),
                                 driver="GPKG")

    pieces = ccf.add_equal_area_ha(ccf.intersect_features_with_regions(
        gdf_salt_marsh_zonal, gdf_regions, desc='Intersecting salt marsh with countries'))

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
         fallback.
    """
    publish_inputs(p)
    p.salt_marsh_carbon_stock_path = os.path.join(
        p.cur_dir, "salt_marsh_carbon_stock_by_countries2019.csv"
    )
    if not p.run_this:
        return

    _build_country_id_raster_if_needed(p)
    mask_path = _habitat_mask(p, p.salt_marsh_vector_path, "salt_marsh_mask_10sec.tif")

    extras = {}
    aligned_path = _aligned_density_input(
        p, getattr(p, 'salt_marsh_soc_path', None), "maxwell_marsoc_aligned_10sec.tif",
        'Salt marsh SOC (Maxwell 2024 MarSOC)')
    if aligned_path:
        extras['soc_arr'] = aligned_path

    df_stock = _stock_by_region(p, 'salt_marsh', mask_path,
                                ccf.calculate_salt_marsh_density_array, extras)
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
# Seagrass: WCMC013-014 SeagrassPtPy v7.1 extent
#
# The one habitat whose density is an attribute of the polygon (its GENUS) rather than of the
# pixel, so its stock is aggregated from the clipped polygons instead of a streamed raster
# pass. A missing extent RAISES: seagrass is excluded by building the tree with
# include_seagrass=False, never by letting a data gap silently understate the coastal total.
# ============================================================================

def seagrass_area_within_countries(p):
    """Seagrass area within each region's marine EEZ, keeping GENUS on every clipped polygon.

    Outputs
    -------
    seagrass_within_countries2019.gpkg
        Per-polygon intersection with GENUS, the region id, and area_ha -- the input the
        genus-aware stock task reads.
    seagrass_area_by_countries2019.{gpkg,csv}
        Region-level area total.
    """
    publish_inputs(p)
    p.seagrass_within_countries_path = os.path.join(
        p.cur_dir, "seagrass_within_countries2019.gpkg"
    )
    p.seagrass_area_by_countries_base_year_path = os.path.join(
        p.cur_dir, "seagrass_area_by_countries2019.gpkg"
    )
    csv_path = p.seagrass_area_by_countries_base_year_path.replace('.gpkg', '.csv')
    if not p.run_this:
        return

    if not hb.path_exists(getattr(p, 'seagrass_vector_path', None)):
        raise FileNotFoundError(
            "seagrass extent not found (p.seagrass_vector_path=%r). A built seagrass tree with no "
            "data would silently understate the coastal GEP total; to exclude seagrass, build the "
            "tree with include_seagrass=False instead." % (getattr(p, 'seagrass_vector_path', None),))

    gdf_regions = gpd.read_file(p.gep_regions_input_path)
    gdf_seagrass = gpd.read_file(
        p.seagrass_vector_path, columns=['GENUS', 'FAMILY']).to_crs(gdf_regions.crs)
    hb.log(f'Loaded {len(gdf_seagrass)} seagrass polygons, {len(gdf_regions)} marine regions')

    pieces = ccf.add_equal_area_ha(ccf.intersect_features_with_regions(
        gdf_seagrass, gdf_regions, desc='Intersecting seagrass with countries'))
    pieces.to_file(p.seagrass_within_countries_path, driver="GPKG")

    area = ccf.area_by_region(pieces, gdf_regions, p.gep_regions_id_col, how='right')
    area.to_file(p.seagrass_area_by_countries_base_year_path, driver="GPKG")
    area.drop(columns=['geometry']).to_csv(csv_path, index=False)
    hb.log(f'Saved seagrass area by countries: {p.seagrass_area_by_countries_base_year_path}')


def seagrass_carbon_stock(p):
    """Per-region seagrass carbon stock (Mg C) from Gomis et al. 2025 genus-specific biomass
    densities and the Fourqurean et al. 2012 soil constant.

    Reads the polygon-level GPKG the area task wrote (GENUS + region id + area_ha), prices each
    polygon by its genus density, and sums to the region. Non-marine genera (Trapa,
    Myriophyllum, Valisneria, Najas and the rest) carry zero density.
    """
    publish_inputs(p)
    p.seagrass_carbon_stock_path = os.path.join(
        p.cur_dir, "seagrass_carbon_stock_by_countries2019.csv"
    )
    if not p.run_this:
        return

    if not hb.path_exists(getattr(p, 'seagrass_within_countries_path', None)):
        raise FileNotFoundError(
            "seagrass polygon-level GPKG missing -- seagrass_area_within_countries "
            "must run first (same tree). Skipping here would silently zero seagrass in the total.")

    gdf_pieces = gpd.read_file(p.seagrass_within_countries_path)
    hb.log(f'Read {len(gdf_pieces)} seagrass-within-country polygons')

    df_stock = ccf.seagrass_stock_by_region(gdf_pieces, p.gep_regions_id_col)
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
        'mangrove': pd.read_csv(
            p.mangrove_area_by_countries_base_year_path.replace('.gpkg', '.csv')),
        'salt_marsh': pd.read_csv(
            p.salt_marsh_area_by_countries_base_year_path.replace('.gpkg', '.csv')),
    }
    stock_frames = {
        'mangrove': pd.read_csv(p.mangrove_carbon_stock_path),
        'salt_marsh': pd.read_csv(p.salt_marsh_carbon_stock_path),
    }

    seagrass_area_gpkg = getattr(p, 'seagrass_area_by_countries_base_year_path', None)
    seagrass_area_csv = (
        seagrass_area_gpkg.replace('.gpkg', '.csv') if seagrass_area_gpkg else None
    )
    if hb.path_exists(seagrass_area_csv):
        area_frames['seagrass'] = pd.read_csv(seagrass_area_csv)
    else:
        hb.log('combined_ecosystem_areas: seagrass area CSV not found '
               f'({seagrass_area_csv!r}); seagrass area set to 0.')

    seagrass_stock_path = getattr(p, 'seagrass_carbon_stock_path', None)
    if hb.path_exists(seagrass_stock_path):
        stock_frames['seagrass'] = pd.read_csv(seagrass_stock_path)
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
    # valuation (same contract fix as terrestrial_carbon).

    final_csv = service_results['gep_by_country_base_year']
    if hb.path_all_exist(list(service_results.values())):
        hb.log("gep_calculation: skipped (all registered results exist)")
        return

    r566_csv = service_results['gep_by_country_base_year_r566']
    r566_gpkg = r566_csv.replace('.csv', '.gpkg')

    # Stage 1: the marine surface (r566).
    if hb.path_all_exist(r566_csv, r566_gpkg):
        df_gep = pd.read_csv(r566_csv)
        hb.log("gep_calculation: r566 cached, reusing.")
    else:
        hb.log("gep_calculation: computing r566 storage values "
               "(mangrove + salt marsh + seagrass)...")
        df_gep = ccf.coastal_carbon_storage_value_frame(
            df_areas=pd.read_csv(p.combined_area_path),
            df_price=pd.read_excel(p.gep_price_input_path)[['year', p.gep_price_convention]],
            value_frames={ecosystem: pd.read_csv(getattr(p, f'{ecosystem}_storage_value_path'))
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
        ccf.eez_storage_value_by_iso3(df_gep), pd.read_csv(p.df_countries_csv_path))
    df_r250_final.to_csv(final_csv, index=False)
    hb.log(f"Final iso3_r250 GEP saved: {final_csv}  "
           f"({len(df_r250_final)} iso3 countries, "
           f"total ${df_r250_final['value'].sum():,.2f})")

    return df_r250_final['value'].sum()


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)

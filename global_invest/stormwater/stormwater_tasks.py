"""Stormwater GEP tasks: the retained volume from the InVEST run, priced per cubic metre.

The InVEST urban stormwater retention run happens outside this tree; retention_by_country
sums its raster inside each country, and gep_calculation prices the result."""
import os

import numpy as np
import pyproj
import rasterio
from osgeo import gdal
import hazelbean as hb
from global_invest import utilities
from global_invest.stormwater import stormwater_functions


def publish_inputs(p):
    """Every GEP task's first line: the stormwater es_config row and the data references
    from es_parameters (defaults layer -- a caller-set value prevails), the shared country
    references and the results registry."""
    utilities.hydrate_es_config(p, 'stormwater', log=hb.log)
    utilities.hydrate_es_parameters(p, 'stormwater', log=hb.log)
    utilities.initialize_country_paths(p)
    if not hasattr(p, 'results'):
        p.results = {}
    return p



INVEST_WORKSPACE_DIR_NAME = 'stormwater_invest'
RETENTION_VOLUME_FILE_NAME = 'retention_volume_urbanstorm_water.tif'
ZONE_ID_FIELD = 'iso3_r250_id'
ZONE_LABEL_FIELD = 'iso3_r250_label'
ZONE_ID_NDV = 0                 # the id convention (0 = NDV), so 0 is the rasterized background
RETENTION_GRID_EPSG = 3857      # the InVEST run is in pseudo-Mercator; see run_recipe.md
BLOCK_N_ROWS = 2048             # full-width row blocks; the retention grid is 74,596 px wide
LOG_EVERY_N_BLOCKS = 10
ZONE_IDS_CREATION_OPTIONS = ['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES']


def write_countries_on_retention_grid(countries_path, dst_path):
    """Reproject the r250 country polygons into the retention grid's CRS.

    Rasterizing does not reproject -- gdal's output-SRS option only labels the output -- so
    the polygons have to reach the grid's CRS first.

    Args:
        countries_path (str): the r250 country polygons (EPSG:4326).
        dst_path (str): geopackage to write, in EPSG:RETENTION_GRID_EPSG.
    """
    gdf = gpd.read_file(countries_path).to_crs(epsg=RETENTION_GRID_EPSG)
    gdf.to_file(dst_path, driver='GPKG')


def write_zone_id_raster(countries_on_grid_path, retention_path, dst_path):
    """Burn the country ids onto the retention raster's exact grid: same bounds, same shape.

    Args:
        countries_on_grid_path (str): country polygons already in the retention grid's CRS.
        retention_path (str): the retention-volume raster the grid is taken from.
        dst_path (str): country-id raster to write.
    """
    with rasterio.open(retention_path) as src:
        bounds, width, height = src.bounds, src.width, src.height
    gdal.Rasterize(dst_path, countries_on_grid_path, options=gdal.RasterizeOptions(
        attribute=ZONE_ID_FIELD, outputType=gdal.GDT_UInt16,
        noData=ZONE_ID_NDV, initValues=ZONE_ID_NDV,
        outputBounds=[bounds.left, bounds.bottom, bounds.right, bounds.top],
        width=width, height=height, outputSRS=f'EPSG:{RETENTION_GRID_EPSG}',
        creationOptions=ZONE_IDS_CREATION_OPTIONS))


def retention_m3_by_country(retention_path, zone_ids_path, countries_df):
    """One row per country: the retention-volume raster summed inside that country.

    Args:
        retention_path (str): InVEST retention-volume raster, cubic metres a year per pixel.
        zone_ids_path (str): country-id raster on the same grid (ZONE_ID_NDV = background).
        countries_df (pd.DataFrame): the r250 country table, with ZONE_ID_FIELD.

    Returns:
        pd.DataFrame: countries_df plus a retention_m3 column.

    Raises:
        ValueError: if the two rasters are not on the same grid, which would sum retention
            into the wrong countries without failing.
    """
    n_zones = int(countries_df[ZONE_ID_FIELD].max())
    sums = np.zeros(n_zones + 1, dtype='float64')
    with rasterio.open(retention_path) as src, rasterio.open(zone_ids_path) as zone_src:
        if zone_src.shape != src.shape:
            raise ValueError('country-id raster %s is %s; retention raster %s is %s' % (
                zone_ids_path, zone_src.shape, retention_path, src.shape))
        # The volumes arrive built on one constant cell area; ground_area_share puts each row
        # back on its own ground area. See stormwater_functions.ground_area_share.
        to_wgs84 = pyproj.Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True)
        for block_index, row_0 in enumerate(range(0, src.height, BLOCK_N_ROWS)):
            window = rasterio.windows.Window(0, row_0, src.width,
                                             min(BLOCK_N_ROWS, src.height - row_0))
            volume = src.read(1, window=window)
            # Nodata and non-finite pixels contribute nothing to a country's total.
            valid = np.isfinite(volume) & (volume != src.nodata)
            row_y = [src.xy(row, 0)[1] for row in range(row_0, row_0 + volume.shape[0])]
            row_lat = to_wgs84.transform(np.zeros(len(row_y)), np.asarray(row_y))[1]
            volume = volume * stormwater_functions.ground_area_share(row_lat)[:, None]
            sums += utilities.sum_by_zone(
                np.where(valid, volume, 0.0), zone_src.read(1, window=window), n_zones)
            if block_index % LOG_EVERY_N_BLOCKS == 0:
                hb.log('stormwater zonal: row %d of %d' % (row_0, src.height))

    df = countries_df.copy()
    df['retention_m3'] = sums[df[ZONE_ID_FIELD].to_numpy(dtype=int)]
    return df

def retention_by_country(p):
    """The InVEST retention raster summed inside each country, written as the per-country table.

    The InVEST 3.14 Urban Stormwater Retention run itself happens outside this tree, over the
    staged global inputs; base_data/global_invest/stormwater/run_recipe.md records its
    configuration. This task is the step after it.

    Registered with skip_existing=1 because it costs minutes over a 74,596-pixel-wide grid and
    is deterministic, the same reason erosion's SDR and routing steps skip.
    """
    publish_inputs(p)
    p.stormwater_retention_by_country_path = p.stormwater_retention_by_country_path
    if not p.run_this:
        return
    if hb.path_exists(p.stormwater_retention_by_country_path):
        hb.log('stormwater retention table already exists. Skipping the zonal sum.')
        return True
    import geopandas as gpd

    retention_path = os.path.join(p.intermediate_dir, INVEST_WORKSPACE_DIR_NAME,
                                  RETENTION_VOLUME_FILE_NAME)
    if not hb.path_exists(retention_path):
        raise NameError(
            'The InVEST retention raster is not at %s. That run happens outside this tree; see '
            'base_data/global_invest/stormwater/run_recipe.md for its configuration.'
            % retention_path)
    countries_path = p.stormwater_countries_path
    countries_on_grid_path = p.stormwater_countries_on_grid_path
    zone_ids_path = p.stormwater_zone_ids_path
    if not hb.path_exists(countries_on_grid_path):
        write_countries_on_retention_grid(countries_path, countries_on_grid_path)
    if not hb.path_exists(zone_ids_path):
        write_zone_id_raster(countries_on_grid_path, retention_path, zone_ids_path)

    countries_df = gpd.read_file(countries_path, ignore_geometry=True)[
        [ZONE_ID_FIELD, ZONE_LABEL_FIELD]]
    df = retention_m3_by_country(retention_path, zone_ids_path, countries_df)
    hb.df_write(df, p.stormwater_retention_by_country_path)
    hb.log('stormwater retention: %.4g m3/yr over %d countries with positive retention'
           % (df['retention_m3'].sum(), (df['retention_m3'] > 0).sum()))
    return True


def gep_calculation(p):
    """GEP valuation for stormwater: retained volume times the price per cubic metre.
    The volumes are read from the table retention_by_country writes, named in es_parameters
    as stormwater_retention_by_country_path."""
    publish_inputs(p)
    service_results, already_done = utilities.begin_gep_calculation(p, 'stormwater')
    if already_done:
        return

    from global_invest.stormwater import stormwater_functions as sf
    retention = hb.df_read(p.stormwater_retention_by_country_path)
    df_gep = sf.stormwater_gep_by_country(retention, sf.STORMWATER_PRICE_PER_M3_PLACEHOLDER)
    df_gep['year'] = int(p.gep_base_year)
    # Carry the country attributes every other service's table carries, so this output can be
    # read, grouped and reported the same way. Without them the report cannot even name a country.
    attribute_columns = ['iso3_r250_id', 'iso3_r250_name', 'continent', 'region_un',
                         'region_wb', 'income_grp', 'subregion']
    countries = utilities.collapse_countries_to_r250(p.df_countries)[attribute_columns]
    df_gep = countries.merge(df_gep, on='iso3_r250_id', how='right')
    hb.df_write(df_gep, service_results['gep_by_country_base_year'])
    hb.log('Total stormwater retention: %.4g m3/yr over %d countries; GEP %.4g USD at the '
           'placeholder price of %s per m3 (the open ask).' % (
               df_gep['retention_m3'].sum(), (df_gep['retention_m3'] > 0).sum(),
               df_gep['stormwater_gep'].sum(), sf.STORMWATER_PRICE_PER_M3_PLACEHOLDER))
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)

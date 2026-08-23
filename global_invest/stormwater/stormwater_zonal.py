"""Per-country retention volumes from the InVEST urban stormwater retention run.

The InVEST 3.14 Urban Stormwater Retention model runs once over the staged global inputs
(ESA 2020 land cover, WorldClim precipitation, SLGWRB soils, the v2 biophysical coefficient
table), outside this repo's task tree. `base_data/global_invest/stormwater/run_recipe.md`
records that run's configuration, its inputs and its outputs.

This script is the step after it. It reprojects the r250 country polygons onto the retention
grid, burns their ids into a country-id raster of the same shape, sums the retention-volume
raster inside each country block by block, and writes the per-country table that
`stormwater_tasks.gep_calculation` prices -- the file `es_parameters.csv` names in
`stormwater_retention_by_country_path`.

Run once by hand from the repo root, after the InVEST run finishes. No task calls it: its
input is a full-extent raster from a hours-long run that lives outside version control, so
the small table it produces is what the module consumes. It is kept here so the step is on
the record rather than in someone's shell history.
"""
import os

import geopandas as gpd
import numpy as np
import rasterio
from osgeo import gdal
import hazelbean as hb

from global_invest.timber_provision import timber_provision_functions

PROJECT_NAME = 'gep_stormwater'
# The InVEST run's workspace under the project's intermediate dir, and the retention-volume
# raster in it (the file name carries the run's results_suffix, 'urbanstorm_water').
INVEST_WORKSPACE_DIR_NAME = 'stormwater_invest'
RETENTION_VOLUME_FILE_NAME = 'retention_volume_urbanstorm_water.tif'
COUNTRIES_REF_PATH = 'cartographic/ee/ee_r250.gpkg'
COUNTRIES_ON_GRID_REF_PATH = 'cartographic/ee/ee_r250_3857.gpkg'
ZONE_IDS_REF_PATH = 'cartographic/ee/ee_r250_id_3857_537m.tif'
RETENTION_BY_COUNTRY_REF_PATH = 'global_invest/stormwater/retention_m3_by_country.csv'
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
        for block_index, row_0 in enumerate(range(0, src.height, BLOCK_N_ROWS)):
            window = rasterio.windows.Window(0, row_0, src.width,
                                             min(BLOCK_N_ROWS, src.height - row_0))
            volume = src.read(1, window=window)
            # Nodata and non-finite pixels contribute nothing to a country's total.
            valid = np.isfinite(volume) & (volume != src.nodata)
            sums += timber_provision_functions.gep_by_zone(
                np.where(valid, volume, 0.0), zone_src.read(1, window=window), n_zones)
            if block_index % LOG_EVERY_N_BLOCKS == 0:
                hb.log('stormwater zonal: row %d of %d' % (row_0, src.height))

    df = countries_df.copy()
    df['retention_m3'] = sums[df[ZONE_ID_FIELD].to_numpy(dtype=int)]
    return df


def main():
    """Rasterize the country ids onto the retention grid, then write the per-country table."""
    p = hb.ProjectFlow(project_name=PROJECT_NAME, run_mode='check')
    retention_path = os.path.join(p.intermediate_dir, INVEST_WORKSPACE_DIR_NAME,
                                  RETENTION_VOLUME_FILE_NAME)
    countries_path = p.get_path(COUNTRIES_REF_PATH, possible_dirs=[p.base_data_dir])
    countries_on_grid_path = p.get_path(COUNTRIES_ON_GRID_REF_PATH,
                                        possible_dirs=[p.base_data_dir], raise_error_if_fail=False)
    zone_ids_path = p.get_path(ZONE_IDS_REF_PATH, possible_dirs=[p.base_data_dir],
                               raise_error_if_fail=False)
    output_path = p.get_path(RETENTION_BY_COUNTRY_REF_PATH, possible_dirs=[p.base_data_dir],
                             raise_error_if_fail=False)

    if not hb.path_exists(countries_on_grid_path):
        write_countries_on_retention_grid(countries_path, countries_on_grid_path)
    if not hb.path_exists(zone_ids_path):
        write_zone_id_raster(countries_on_grid_path, retention_path, zone_ids_path)

    countries_df = gpd.read_file(countries_path, ignore_geometry=True)[
        [ZONE_ID_FIELD, ZONE_LABEL_FIELD]]
    df = retention_m3_by_country(retention_path, zone_ids_path, countries_df)
    df.to_csv(output_path, index=False)
    hb.log('stormwater retention: %.4g m3/yr over %d countries with positive retention; '
           'wrote %s' % (df['retention_m3'].sum(), (df['retention_m3'] > 0).sum(), output_path))


if __name__ == '__main__':
    main()

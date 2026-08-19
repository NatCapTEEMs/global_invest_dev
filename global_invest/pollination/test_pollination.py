"""Unit tests for the pollination GEP valuation (synthetic, self-contained).

End-to-end through the real chain: a tiny USD value raster -> utilities.summarize_raster_by_region
(real zonal statistics) -> gep_calculation. Pins the r250 contract: a split country (two r264
sub-regions, one iso3) is summed once, and the map gpkg carries per-sub-region rows that are never
summed. The ES-shock tasks have their own coverage via the shared resolver/invariant tests.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from global_invest.pollination import pollination_tasks as pt

ATTRS = {'iso3_r250_label': {156: 'CHN', 528: 'NLD'},
         'iso3_r250_name': {156: 'China', 528: 'Netherlands'},
         'continent': {156: 'Asia', 528: 'Europe'},
         'region_un': {156: 'Asia', 528: 'Europe'},
         'region_wb': {156: 'EAP', 528: 'ECA'},
         'income_grp': {156: 'UMIC', 528: 'HIC'},
         'subregion': {156: 'Eastern Asia', 528: 'Western Europe'}}


def _write_tif(path, array):
    from osgeo import gdal, osr
    array = np.asarray(array, dtype='float32')
    h, w = array.shape
    ds = gdal.GetDriverByName('GTiff').Create(str(path), w, h, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((-180.0, 360.0 / w, 0.0, 90.0, 0.0, -180.0 / h))
    srs = osr.SpatialReference(); srs.ImportFromEPSG(4326); ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1); band.SetNoDataValue(float('nan'))
    band.WriteArray(array); band.FlushCache(); ds = None


def test_pollination_gep_sums_split_country_once(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box

    # 4x4 global raster, columns 90 deg wide. Region 1 = col 0 (10 USD/cell), region 2 = col 1
    # (5 USD/cell), region 3 = col 2 (2 USD/cell); col 3 belongs to no region. Regions 1+2 are the
    # split country's sub-regions: CHN = 40 + 20 = 60, NLD = 8, total 68.
    vals = np.zeros((4, 4), dtype='float32')
    vals[:, 0] = 10.0; vals[:, 1] = 5.0; vals[:, 2] = 2.0
    _write_tif(tmp_path / 'poll_value.tif', vals)

    ids = [1, 2, 3]
    iso = [156, 156, 528]
    gdf = gpd.GeoDataFrame(
        {'ee_r264_id': ids, 'iso3_r250_id': iso,
         **{col: [m[i] for i in iso] for col, m in ATTRS.items()},
         'geometry': [box(-175, -85, -95, 85), box(-85, -85, -5, 85), box(5, -85, 85, 85)]},
        crs='EPSG:4326')
    gdf.to_file(str(tmp_path / 'regions.gpkg'), driver='GPKG')

    p = SimpleNamespace(run_this=True, cur_dir=str(tmp_path), results={}, base_year=2023,
                        gep_quantity_input_path=str(tmp_path / 'poll_value.tif'),
                        gdf_countries_vector_path=str(tmp_path / 'regions.gpkg'),
                        gdf_countries_simplified=str(tmp_path / 'regions.gpkg'))

    pt.task_summarize_pollination_value_by_region(p)
    total = pt.gep_calculation(p)

    assert total == 68.0                              # split country counted once, not 128
    out = pd.read_csv(tmp_path / 'gep_by_country_base_year.csv').set_index('iso3_r250_label')
    assert len(out) == 2                              # one row per iso3 country
    assert out.loc['CHN', 'pollination_gep'] == 60.0
    assert out.loc['NLD', 'pollination_gep'] == 8.0

    # Map contract: r264-expanded rows each carry the COUNTRY value, never summed.
    m = gpd.read_file(tmp_path / 'gep_by_country_base_year.gpkg')
    chn_rows = m[m['ee_r264_id'].isin([1, 2])]
    assert (chn_rows['pollination_gep'] == 60.0).all()

"""Unit tests for the terrestrial_carbon module.

Fast, self-contained: the scenario-map resolver is pure; the raster functions run on tiny synthetic
GeoTIFFs written to tmp_path, so no base data or full-globe rasters are needed.
"""
import numpy as np
import pandas as pd

from global_invest import utilities
from global_invest.terrestrial_carbon import terrestrial_carbon_functions as tcf


# --- utilities.resolve_raw_scenario (identity default + loud skip) --------------------------------

def test_resolve_scenario_identity_default():
    labels = ['below_2c', 'current_policies', 'net_zero_2050']
    # no map entry, but the table already uses our name -> identity resolves it
    assert utilities.resolve_raw_scenario(labels, {}, 'below_2c', 'svc') == 'below_2c'


def test_resolve_scenario_explicit_map_first_present_wins():
    labels = ['net_zero_2050', 'current_policies']
    m = {'net_zero': ['net_zero', 'net_zero_2050'], 'stress_test': ['current_policies']}
    # 'net_zero' is absent, 'net_zero_2050' present -> the second candidate wins
    assert utilities.resolve_raw_scenario(labels, m, 'net_zero', 'svc') == 'net_zero_2050'
    assert utilities.resolve_raw_scenario(labels, m, 'stress_test', 'svc') == 'current_policies'


def test_resolve_scenario_absent_warns_loudly_and_returns_none():
    msgs = []
    got = utilities.resolve_raw_scenario(['below_2c'], {}, 'net_zero', 'terrestrial_carbon', log=msgs.append)
    assert got is None                       # never a silent match
    assert len(msgs) == 1                     # and it warned
    assert 'net_zero' in msgs[0] and 'terrestrial_carbon' in msgs[0] and 'below_2c' in msgs[0]


# --- tiny-raster helpers --------------------------------------------------------------------------

def _write_tif(path, array, dtype, nodata=None):
    from osgeo import gdal, osr
    array = np.asarray(array)
    h, w = array.shape
    ds = gdal.GetDriverByName('GTiff').Create(str(path), w, h, 1, dtype)
    ds.SetGeoTransform((-180.0, 360.0 / w, 0.0, 90.0, 0.0, -180.0 / h))
    srs = osr.SpatialReference(); srs.ImportFromEPSG(4326); ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(float(nodata))
    band.WriteArray(array); band.FlushCache(); ds = None


# --- generate_carbon_density_raster (lookup mapping; missing pair -> NaN) --------------------------

def test_generate_density_maps_pairs_and_leaves_missing_as_nan(tmp_path):
    from osgeo import gdal
    lulc, cz, out = tmp_path / 'lulc.tif', tmp_path / 'cz.tif', tmp_path / 'dens.tif'
    _write_tif(lulc, [[10, 10], [20, 99]], gdal.GDT_Byte)          # (99, 101) has no lookup row
    _write_tif(cz, [[101, 101], [101, 101]], gdal.GDT_UInt32)
    pd.DataFrame({'lulc_id': [10, 20], 'carbon_zone_id': [101, 101],
                  'carbon_density_mean': [5.0, 7.0]}).to_csv(tmp_path / 'lut.csv', index=False)

    tcf.generate_carbon_density_raster(str(lulc), str(cz), str(tmp_path / 'lut.csv'), str(out))

    a = gdal.Open(str(out)).ReadAsArray()
    assert a[0, 0] == 5.0 and a[0, 1] == 5.0 and a[1, 0] == 7.0
    assert np.isnan(a[1, 1])                  # (lulc 99, zone 101) absent -> NaN, not a silent 0


# --- stack_layers_to_csv (group a value raster by two category rasters) ---------------------------

def test_stack_groups_by_two_rasters_with_mean_and_count(tmp_path):
    from osgeo import gdal
    lulc, cz, val, out = tmp_path / 'lulc.tif', tmp_path / 'cz.tif', tmp_path / 'val.tif', tmp_path / 'lut.csv'
    _write_tif(lulc, [[10, 10], [20, 20]], gdal.GDT_Byte)
    _write_tif(cz, [[101, 101], [101, 101]], gdal.GDT_UInt32)
    _write_tif(val, [[4.0, 6.0], [8.0, 10.0]], gdal.GDT_Float32, nodata=np.nan)

    tcf.stack_layers_to_csv(str(lulc), str(cz), str(val), str(out),
                            group1_name='lulc_id', group2_name='carbon_zone_id', value_name='carbon_density')

    df = pd.read_csv(out).set_index(['lulc_id', 'carbon_zone_id'])
    assert df.loc[(10, 101), 'carbon_density_mean'] == 5.0     # mean of [4, 6]
    assert df.loc[(10, 101), 'carbon_density_count'] == 2
    assert df.loc[(20, 101), 'carbon_density_mean'] == 9.0     # mean of [8, 10]

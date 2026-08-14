"""Unit tests specific to the terrestrial_carbon module.

Fast and self-contained: the raster functions run on tiny synthetic GeoTIFFs written to tmp_path, and the
static shock runs on a small synthetic dependency table -- no base data or full-globe rasters. The shared
scenario resolver is tested once in global_invest_tests/test_es_utilities.py, not here.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from global_invest.terrestrial_carbon import terrestrial_carbon_functions as tcf
from global_invest.terrestrial_carbon import terrestrial_carbon_tasks as tct


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


# --- task_compute_terrestrial_carbon_shock_static (linear ramp, differencing, loud skip) ----------

def test_static_shock_ramps_differences_and_skips_absent(tmp_path):
    dep = tmp_path / 'carbon_storage_dependency.csv'
    # end-year (2050) rows only; base = baseline_ignore_dependencies. percentage_change is scaled x100.
    pd.DataFrame({
        'scenario': ['baseline_ignore_dependencies', 'baseline_ignore_dependencies', 'below_2c', 'below_2c'],
        'year': [2050, 2050, 2050, 2050],
        'ENDW': ['AEZ1', 'AEZ2', 'AEZ1', 'AEZ2'],
        'REG': ['usa', 'usa', 'usa', 'usa'],
        'percentage_change': [0.10, 0.20, 0.15, 0.20],   # below_2c - base: AEZ1 +0.05 -> +5 pts, AEZ2 0
    }).to_csv(dep, index=False)
    out = tmp_path / 'terrestrial_carbon_interpolated.csv'
    p = SimpleNamespace(run_this=True, es_shock_base_year=2020, es_shock_end_year=2050,
                        es_shock_scenarios=['below_2c', 'net_zero'],   # net_zero absent from table
                        terrestrial_carbon_dependency_path=str(dep),
                        terrestrial_carbon_shock_output_path=str(out))

    tct.task_compute_terrestrial_carbon_shock_static(p)

    df = pd.read_csv(out)
    assert set(df['scenario'].unique()) == {'below_2c'}       # net_zero absent -> skipped, not zeroed
    aez1 = df[df['ENDW'] == 'AEZ1'].set_index('year')['shock_pct']
    assert abs(aez1.loc[2050] - 5.0) < 1e-9                    # full shock at end year
    assert abs(aez1.loc[2020] - 0.0) < 1e-9                    # 0 at base year
    assert abs(aez1.loc[2035] - 2.5) < 1e-9                    # linear ramp at the midpoint
    assert abs(df[df['ENDW'] == 'AEZ2'].set_index('year')['shock_pct'].loc[2050]) < 1e-9  # AEZ2 shock 0

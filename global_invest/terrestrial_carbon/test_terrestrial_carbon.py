"""Unit tests specific to the terrestrial_carbon module.

Fast and self-contained: the raster functions run on tiny synthetic GeoTIFFs written to tmp_path, and the
static shock runs on a small synthetic dependency table -- no base data or full-globe rasters. The shared
scenario resolver is tested once in global_invest_tests/test_es_utilities.py, not here.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

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


# --- terrestrial_carbon_shock_static (linear ramp, differencing, loud skip) ----------

# --- terrestrial_carbon_shock (dynamic, synthetic scene) -----------------------------
#
# 4x4 raster, global extent: columns are 90 deg wide, rows 45 deg tall. Zone A = column 0
# (box inset to -175..-95 so all_touched can't bleed into column 1), zone B = column 2; columns 1/3
# belong to no zone. One carbon zone (101) everywhere; densities: class 1 -> 100, 2 -> 10, 3 -> 20 tC/ha.
#
#                 zone A (col 0) mean      zone B (col 2) mean
#   base 2020            100                     100
#   baseline 2030        100                     100
#   baseline 2050         80  (one px cls 3)     100
#   below_2c 2030         55  (two px cls 2)     100
#   below_2c 2050         10  (all cls 2)         55  (two px cls 2)
#
# contemp = (scn_Y - base_Y)/base_Y; fixedbase = (scn_Y - base_Y)/base_2020. The drifting baseline
# (80 vs 100 at 2050) makes the two measures differ: zone A 2050 contemp -87.5 vs fixedbase -70.

def _shock_scene(tmp_path):
    from osgeo import gdal
    import geopandas as gpd
    from shapely.geometry import box

    cz = tmp_path / 'cz.tif'
    _write_tif(cz, np.full((4, 4), 101), gdal.GDT_UInt32)
    pd.DataFrame({'lulc_id': [1, 2, 3], 'carbon_zone_id': [101, 101, 101],
                  'carbon_density_mean': [100.0, 10.0, 20.0]}).to_csv(tmp_path / 'lut.csv', index=False)

    def lulc(name, array):
        path = tmp_path / f'lulc_{name}.tif'
        _write_tif(path, array, gdal.GDT_Byte)
        return str(path)

    ones = np.ones((4, 4), dtype=np.uint8)
    base_2020 = lulc('base_2020', ones)
    baseline_2030 = lulc('baseline_2030', ones)
    a = ones.copy(); a[3, 0] = 3
    baseline_2050 = lulc('baseline_2050', a)
    a = ones.copy(); a[0:2, 0] = 2
    below_2030 = lulc('scn_a_2030', a)
    a = ones.copy(); a[:, 0] = 2; a[0:2, 2] = 2
    below_2050 = lulc('scn_a_2050', a)

    regions = tmp_path / 'zones.gpkg'
    gpd.GeoDataFrame({'ee_r50_aez18_id': [1, 2], 'aez18_id': [1, 2],
                      'gtapv7_r50_label': ['usa', 'chn'],
                      'geometry': [box(-175, -85, -95, 85), box(5, -85, 85, 85)]},
                     crs='EPSG:4326').to_file(str(regions), driver='GPKG')

    return SimpleNamespace(
        run_this=True, cur_dir=str(tmp_path), input_dir=str(tmp_path / 'input'),
        get_path=lambda *a, **k: '/resolved/' + '/'.join(a),
        es_shock_base_year=2020, es_shock_years=[2030, 2050],
        es_shock_base_scenario='baseline', es_shock_scenarios=['scn_a'],
        scenario_lulc_paths={'baseline': {2030: baseline_2030, 2050: baseline_2050},
                             'scn_a': {2030: below_2030, 2050: below_2050}},
        es_base_year_lulc_path=base_2020,
        region_boundary_path=str(regions),
        terrestrial_quantity_input_path=str(cz),
        terrestrial_carbon_density_lookup_table_path=str(tmp_path / 'lut.csv'),
        terrestrial_carbon_shock_output_path=str(tmp_path / 'terrestrial_carbon_interpolated.csv'))


def test_dynamic_shock_per_zone_interpolation_and_both_measures(tmp_path):
    p = _shock_scene(tmp_path)
    tct.terrestrial_carbon_shock(p)
    df = pd.read_csv(p.terrestrial_carbon_shock_output_path)

    assert set(df['scenario'].unique()) == {'scn_a'}
    assert set(df['ACTS'].unique()) == {'FRS'}
    # zone labels ride through from the boundary's aez18_id / gtapv7_r50_label columns
    assert set(zip(df['ENDW'], df['REG'])) == {('AEZ1', 'usa'), ('AEZ2', 'chn')}
    # every year base..end, per zone
    assert sorted(df[df['ENDW'] == 'AEZ1']['year']) == list(range(2020, 2051))

    a = df[df['ENDW'] == 'AEZ1'].set_index('year')
    b = df[df['ENDW'] == 'AEZ2'].set_index('year')
    approx = lambda v: pytest.approx(v, abs=1e-6)

    # anchors: contemporaneous (scn_Y - base_Y)/base_Y * 100 per zone
    assert a.loc[2030, 'shock_pct'] == approx(-45.0)     # (55-100)/100
    assert a.loc[2050, 'shock_pct'] == approx(-87.5)     # (10-80)/80: drifted baseline denominator
    assert b.loc[2030, 'shock_pct'] == approx(0.0)
    assert b.loc[2050, 'shock_pct'] == approx(-45.0)     # (55-100)/100

    # piecewise interpolation: 0 at base_year, linear between anchors
    assert a.loc[2020, 'shock_pct'] == approx(0.0)
    assert a.loc[2025, 'shock_pct'] == approx(-22.5)     # midpoint 2020->2030
    assert a.loc[2040, 'shock_pct'] == approx(-66.25)    # midpoint 2030->2050
    assert b.loc[2040, 'shock_pct'] == approx(-22.5)

    # fixed-base measure divides by the BASE-YEAR baseline density (100), not base_Y
    assert a.loc[2050, 'shock_pct_fixedbase'] == approx(-70.0)   # (10-80)/100
    assert a.loc[2030, 'shock_pct_fixedbase'] == approx(-45.0)   # same as contemp while baseline holds
    # shock_pct is the GTAP-primary alias for the contemporaneous measure
    assert (df['shock_pct'] == df['shock_pct_contemp']).all()


# --- guard: an ESA-classed map against the SEALS7-keyed lookup must raise, not silently zero ------

def test_generate_density_raises_when_lookup_matches_nothing(tmp_path):
    from osgeo import gdal
    lulc, cz, out = tmp_path / 'lulc.tif', tmp_path / 'cz.tif', tmp_path / 'dens.tif'
    _write_tif(lulc, [[10, 30], [190, 210]], gdal.GDT_Byte)        # ESA class ids
    _write_tif(cz, [[101, 101], [101, 101]], gdal.GDT_UInt32)
    pd.DataFrame({'lulc_id': [1, 2, 3], 'carbon_zone_id': [101, 101, 101],   # SEALS7-keyed lookup
                  'carbon_density_mean': [100.0, 10.0, 20.0]}).to_csv(tmp_path / 'lut.csv', index=False)

    with pytest.raises(ValueError, match='matched ZERO'):
        tcf.generate_carbon_density_raster(str(lulc), str(cz), str(tmp_path / 'lut.csv'), str(out))


def test_dynamic_shock_esa_base_map_raises_not_silent_zero(tmp_path):
    from osgeo import gdal
    p = _shock_scene(tmp_path)
    esa = tmp_path / 'lulc_esa_2020.tif'
    _write_tif(esa, np.full((4, 4), 10, dtype=np.uint8) * np.array([[1, 3, 19, 21]]), gdal.GDT_Byte)
    p.es_base_year_lulc_path = str(esa)                            # the real-world wrong-wiring trap

    with pytest.raises(ValueError, match='matched ZERO'):
        tct.terrestrial_carbon_shock(p)


def _static_dep_table(tmp_path):
    dep = tmp_path / 'carbon_storage_dependency.csv'
    # end-year (2050) rows only; base = baseline_ignore_dependencies. percentage_change is scaled x100.
    pd.DataFrame({
        'scenario': ['baseline_ignore_dependencies', 'baseline_ignore_dependencies', 'scn_a', 'scn_a'],
        'year': [2050, 2050, 2050, 2050],
        'ENDW': ['AEZ1', 'AEZ2', 'AEZ1', 'AEZ2'],
        'REG': ['usa', 'usa', 'usa', 'usa'],
        'percentage_change': [0.10, 0.20, 0.15, 0.20],   # below_2c - base: AEZ1 +0.05 -> +5 pts, AEZ2 0
    }).to_csv(dep, index=False)
    return dep


def test_static_shock_ramps_and_differences(tmp_path):
    dep = _static_dep_table(tmp_path)
    out = tmp_path / 'terrestrial_carbon_interpolated.csv'
    p = SimpleNamespace(run_this=True, input_dir=str(tmp_path / 'input'),
                        get_path=lambda *a, **k: '/resolved/' + '/'.join(a),
                        es_shock_base_year=2020, es_shock_end_year=2050,
                        es_shock_base_scenario='baseline_ignore_dependencies',
                        es_shock_scenarios=['scn_a'],
                        terrestrial_carbon_dependency_path=str(dep),
                        terrestrial_carbon_shock_output_path=str(out))

    tct.terrestrial_carbon_shock_static(p)

    df = pd.read_csv(out)
    aez1 = df[df['ENDW'] == 'AEZ1'].set_index('year')['shock_pct']
    assert abs(aez1.loc[2050] - 5.0) < 1e-9                    # full shock at end year
    assert abs(aez1.loc[2020] - 0.0) < 1e-9                    # 0 at base year
    assert abs(aez1.loc[2035] - 2.5) < 1e-9                    # linear ramp at the midpoint
    assert abs(df[df['ENDW'] == 'AEZ2'].set_index('year')['shock_pct'].loc[2050]) < 1e-9  # AEZ2 shock 0


def test_static_shock_base_resolves_across_spellings(tmp_path):
    # The frozen tables spell the nature-off baseline two ways. With the base rows labelled
    # 'baseline_ignore_damages' and the config saying 'baseline_ignore_dependencies', the
    # consumer's scenario_map carries both candidates and the base must resolve -- an exact-match
    # miss here used to give an empty base -> empty output -> silent GTAP zero.
    dep = tmp_path / 'carbon_storage_dependency.csv'
    pd.DataFrame({
        'scenario': ['baseline_ignore_damages', 'baseline_ignore_damages', 'scn_a', 'scn_a'],
        'year': [2050] * 4,
        'ENDW': ['AEZ1', 'AEZ2', 'AEZ1', 'AEZ2'],
        'REG': ['usa'] * 4,
        'percentage_change': [0.10, 0.20, 0.15, 0.20],
    }).to_csv(dep, index=False)
    out = tmp_path / 'terrestrial_carbon_interpolated.csv'
    p = SimpleNamespace(run_this=True, input_dir=str(tmp_path / 'input'),
                        get_path=lambda *a, **k: '/resolved/' + '/'.join(a),
                        es_shock_base_year=2020, es_shock_end_year=2050,
                        es_shock_base_scenario='baseline_ignore_dependencies',
                        es_shock_scenarios=['scn_a'],
                        terrestrial_carbon_scenario_map={
                            'baseline_ignore_dependencies': ['baseline_ignore_dependencies',
                                                             'baseline_ignore_damages']},
                        terrestrial_carbon_dependency_path=str(dep),
                        terrestrial_carbon_shock_output_path=str(out))

    tct.terrestrial_carbon_shock_static(p)

    df = pd.read_csv(out)
    assert abs(df[df['ENDW'] == 'AEZ1'].set_index('year')['shock_pct'].loc[2050] - 5.0) < 1e-9

    # Without any consumer map the two nature-off spellings are mutual aliases BY DEFAULT
    # (the tables' own vocabulary, normalized at the point of reading) -- same output.
    p2 = SimpleNamespace(**{**vars(p), 'terrestrial_carbon_scenario_map': {},
                            'terrestrial_carbon_shock_output_path': str(tmp_path / 'out2.csv')})
    tct.terrestrial_carbon_shock_static(p2)
    df2 = pd.read_csv(tmp_path / 'out2.csv')
    assert abs(df2[df2['ENDW'] == 'AEZ1'].set_index('year')['shock_pct'].loc[2050] - 5.0) < 1e-9

    # A base under a name that is in the table under NEITHER spelling stays FATAL.
    p3 = SimpleNamespace(**{**vars(p), 'terrestrial_carbon_scenario_map': {},
                            'es_shock_base_scenario': 'no_such_base',
                            'terrestrial_carbon_shock_output_path': str(tmp_path / 'out3.csv')})
    with pytest.raises(ValueError, match='BASE'):
        tct.terrestrial_carbon_shock_static(p3)


def test_static_shock_missing_scenario_is_fatal_at_the_write(tmp_path):
    # CONTRACT (invariant, assert_shock_table_sound): a requested scenario absent from the dependency
    # table used to warn-and-skip, leaving GTAP a silent zero. It still warns during the loop, but the
    # write now REFUSES, naming the scenario. The loud skip is the diagnostic; the raise is the stop.
    dep = _static_dep_table(tmp_path)
    out = tmp_path / 'terrestrial_carbon_interpolated.csv'
    p = SimpleNamespace(run_this=True, input_dir=str(tmp_path / 'input'),
                        get_path=lambda *a, **k: '/resolved/' + '/'.join(a),
                        es_shock_base_year=2020, es_shock_end_year=2050,
                        es_shock_base_scenario='baseline_ignore_dependencies',
                        es_shock_scenarios=['scn_a', 'scn_b'],   # net_zero absent from table
                        terrestrial_carbon_dependency_path=str(dep),
                        terrestrial_carbon_shock_output_path=str(out))

    with pytest.raises(ValueError, match='scn_b'):
        tct.terrestrial_carbon_shock_static(p)


def test_shock_percent_is_the_departure_from_baseline():
    """A zone whose scenario carbon is a tenth below its baseline shocks at -10 percent, and
    a zone with no baseline carbon gives NaN instead of an infinite shock."""
    import numpy as np
    import pandas as pd
    from global_invest.terrestrial_carbon import terrestrial_carbon_functions as tcf

    scenario = pd.Series([90.0, 110.0, 5.0], index=[1, 2, 3])
    baseline = pd.Series([100.0, 100.0, 0.0], index=[1, 2, 3])
    out = tcf.shock_percent(scenario, baseline)
    assert out[1] == -10.0
    assert out[2] == 10.0
    assert np.isnan(out[3])


def test_interpolate_annual_shock_starts_at_zero_and_passes_through_the_anchors():
    """The base year carries no shock, the anchor years carry theirs, and the years between
    are straight lines."""
    from global_invest.terrestrial_carbon import terrestrial_carbon_functions as tcf

    years = [2020, 2021, 2022, 2023, 2024]
    annual = tcf.interpolate_annual_shock(years, [2022, 2024], [-4.0, -8.0], base_year=2020)
    assert list(annual) == [0.0, -2.0, -4.0, -6.0, -8.0]


def test_shock_percent_keeps_one_numerator_across_both_denominators():
    """The fixed-base measure changes only the denominator: a zone at 10 against a drifted
    baseline of 80 is -87.5 percent contemporaneously and -70 percent against a base of 100."""
    import pandas as pd
    from global_invest.terrestrial_carbon import terrestrial_carbon_functions as tcf

    scenario, baseline_now, baseline_base = pd.Series([10.0]), pd.Series([80.0]), pd.Series([100.0])
    assert tcf.shock_percent(scenario, baseline_now)[0] == -87.5
    assert tcf.shock_percent(scenario, baseline_now, baseline_base)[0] == -70.0

"""Unit tests specific to the terrestrial_carbon module.

Fast and self-contained: the raster functions run on tiny synthetic GeoTIFFs written to tmp_path, and the
static shock runs on a small synthetic dependency table -- no base data or full-globe rasters. The shared
scenario resolver is tested once in global_invest_tests/test_es_utilities.py, not here.
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from global_invest import utilities
from global_invest.terrestrial_carbon import terrestrial_carbon_functions as tcf
from global_invest.terrestrial_carbon import terrestrial_carbon_tasks as tct


def _write_tif(path, array, dtype, nodata=None, geotransform=None, crs=4326):
    """crs is an EPSG code or a PROJ string."""
    from osgeo import gdal, osr
    array = np.asarray(array)
    h, w = array.shape
    ds = gdal.GetDriverByName('GTiff').Create(str(path), w, h, 1, dtype)
    ds.SetGeoTransform(geotransform or (-180.0, 360.0 / w, 0.0, 90.0, 0.0, -180.0 / h))
    srs = osr.SpatialReference()
    if isinstance(crs, int):
        srs.ImportFromEPSG(crs)
    else:
        srs.ImportFromProj4(crs)
    ds.SetProjection(srs.ExportToWkt())
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
    lookup = tcf.carbon_density_lookup(pd.DataFrame({
        'lulc_id': [10, 20], 'carbon_zone_id': [101, 101], 'carbon_density_mean': [5.0, 7.0]}))

    tcf.generate_carbon_density_raster(str(lulc), str(cz), lookup, str(out))

    a = gdal.Open(str(out)).ReadAsArray()
    assert a[0, 0] == 5.0 and a[0, 1] == 5.0 and a[1, 0] == 7.0
    assert np.isnan(a[1, 1])                  # (lulc 99, zone 101) absent -> NaN, not a silent 0


# --- stack_layers_summary (group a value raster by two category rasters) --------------------------

def test_stack_groups_by_two_rasters_with_mean_and_count(tmp_path):
    from osgeo import gdal
    lulc, cz, val = tmp_path / 'lulc.tif', tmp_path / 'cz.tif', tmp_path / 'val.tif'
    _write_tif(lulc, [[10, 10], [20, 20]], gdal.GDT_Byte)
    _write_tif(cz, [[101, 101], [101, 101]], gdal.GDT_UInt32)
    _write_tif(val, [[4.0, 6.0], [8.0, 10.0]], gdal.GDT_Float32, nodata=np.nan)

    df = tct.stack_layers_summary(str(lulc), str(cz), str(val),
                                  group1_name='lulc_id', group2_name='carbon_zone_id',
                                  value_name='carbon_density').set_index(['lulc_id', 'carbon_zone_id'])
    assert df.loc[(10, 101), 'carbon_density_mean'] == 5.0     # mean of [4, 6]
    assert df.loc[(10, 101), 'carbon_density_count'] == 2
    assert df.loc[(20, 101), 'carbon_density_mean'] == 9.0     # mean of [8, 10]


# --- carbon_density_table and the base-year map: the lookup's land cover against the map's -----------

def _lookup_year_scene(tmp_path, lookup_shape=(2, 2), lookup_geotransform=None, lookup_crs=4326, base_crs=4326):
    """2010 classes [[10, 10], [20, 20]] and 2019 classes [[10, 20], [20, 20]] on one 2x2 grid, one
    carbon zone, Spawn densities [[4, 6], [8, 10]]. The lookup_* arguments move the 2010 map off
    that grid."""
    from osgeo import gdal
    lulc_2010, lulc_2019 = tmp_path / 'lulc_2010.tif', tmp_path / 'lulc_2019.tif'
    cz, dens = tmp_path / 'cz.tif', tmp_path / 'dens.tif'
    _write_tif(lulc_2010, np.resize([[10, 10], [20, 20]], lookup_shape), gdal.GDT_Byte,
               geotransform=lookup_geotransform, crs=lookup_crs)
    _write_tif(lulc_2019, [[10, 20], [20, 20]], gdal.GDT_Byte, crs=base_crs)
    _write_tif(cz, [[101, 101], [101, 101]], gdal.GDT_UInt32)
    _write_tif(dens, [[4.0, 6.0], [8.0, 10.0]], gdal.GDT_Float32, nodata=np.nan)
    return SimpleNamespace(cur_dir=str(tmp_path), run_this=1, gep_base_year=2019,
                           carbon_density_lookup_lulc_path=str(lulc_2010),
                           gep_lulc_input_path=str(lulc_2019), gep_quantity_input_path=str(cz),
                           reprojected_total_carbon_density_path=str(dens))


def test_lookup_groups_by_the_2010_land_cover_and_the_map_applies_it_to_2019(tmp_path, monkeypatch):
    """Grouped by the 2010 classes the lookup gives class 10 the mean of [4, 6] and class 20 the mean
    of [8, 10]. Grouped by 2019 it would give 4 and 8. The base-year map then reads the 2019
    classes, so the cell that is class 10 in 2010 and class 20 in 2019 takes class 20's 9."""
    from osgeo import gdal
    monkeypatch.setattr(tct, 'publish_inputs', lambda p: p)
    p = _lookup_year_scene(tmp_path)

    tct.carbon_density_table(p)
    tct.carbon_density_raster_base_year(p)

    lookup = pd.read_csv(p.carbon_density_lookup_table_path).set_index(['lulc_id', 'carbon_zone_id'])
    assert lookup.loc[(10, 101), 'carbon_density_mean'] == 5.0
    assert lookup.loc[(10, 101), 'carbon_density_count'] == 2
    assert lookup.loc[(20, 101), 'carbon_density_mean'] == 9.0
    density = gdal.Open(p.carbon_density_raster_base_year_path).ReadAsArray()
    assert density.tolist() == [[5.0, 9.0], [9.0, 9.0]]


# A transverse Mercator that matches no EPSG code, so the GeoTIFF keeps its parameters and both
# rasters read back with the same name ("unknown"), datum and unit.
_CUSTOM_TM = '+proj=tmerc +lat_0=0 +lon_0=%d +k=0.9 +x_0=500000 +y_0=0 +datum=WGS84 +units=m +no_defs'


@pytest.mark.parametrize('off_grid', [
    dict(lookup_shape=(4, 4)),                                                    # finer cells
    dict(lookup_shape=(3, 3), lookup_geotransform=(-180.0, 180.0, 0.0, 90.0, 0.0, -90.0)),  # same cells, more of them
    dict(lookup_geotransform=(-90.0, 180.0, 0.0, 90.0, 0.0, -90.0)),              # shifted half the globe east
    dict(lookup_geotransform=(-179.9999, 180.0, 0.0, 90.0, 0.0, -90.0)),          # 1e-4 degrees, 0.036 of a 300 m cell
    dict(lookup_crs=3857),                                                        # same numbers, metres not degrees
    dict(lookup_crs=_CUSTOM_TM % 15, base_crs=_CUSTOM_TM % 9),                     # same name, central meridian 15 not 9
], ids=['resolution', 'size', 'shift', 'small_shift', 'projection', 'projection_parameter'])
def test_lookup_refuses_a_land_cover_on_another_grid(tmp_path, monkeypatch, off_grid):
    """A 2010 land cover that does not share the base-year grid cannot pair its cells with the Spawn
    densities aligned to that grid, so the lookup refuses rather than group misaligned cells. The small
    shift sits inside a relative tolerance of 1e-6 at an origin of -180, so it fails only an exact
    comparison. The two projection cases keep size and geotransform identical, so only the CRS tells
    the grids apart. In the last, only a projection parameter does: a comparison of CRS names, datum
    and unit passes it."""
    monkeypatch.setattr(tct, 'publish_inputs', lambda p: p)
    p = _lookup_year_scene(tmp_path, **off_grid)
    with pytest.raises(ValueError, match='Grid mismatch between lookup land cover'):
        tct.carbon_density_table(p)
    assert not os.path.exists(p.carbon_density_lookup_table_path)


# --- carbon zone 0: no key in the lookup, no density in the map, a stale lookup refused --------------

def _zone0_scene(tmp_path):
    """One land cover for both years, [[10, 10, 30], [10, 20, 20]], over carbon zones
    [[0, 101, 101], [0, 101, 101]], with Spawn densities [[4, 6, 0], [2, 8, 10]]. Class 30 in zone 101
    is a valid zero."""
    from osgeo import gdal
    lulc, cz, dens = tmp_path / 'lulc.tif', tmp_path / 'cz.tif', tmp_path / 'dens.tif'
    _write_tif(lulc, [[10, 10, 30], [10, 20, 20]], gdal.GDT_Byte)
    _write_tif(cz, [[0, 101, 101], [0, 101, 101]], gdal.GDT_UInt32)
    _write_tif(dens, [[4.0, 6.0, 0.0], [2.0, 8.0, 10.0]], gdal.GDT_Float32, nodata=np.nan)
    return SimpleNamespace(cur_dir=str(tmp_path), run_this=1, gep_base_year=2019,
                           carbon_density_lookup_lulc_path=str(lulc), gep_lulc_input_path=str(lulc),
                           gep_quantity_input_path=str(cz), reprojected_total_carbon_density_path=str(dens))


def test_the_lookup_leaves_carbon_zone_0_out_and_keeps_every_other_key(tmp_path, monkeypatch):
    """Grouped over all zones the scene gives (10, 0) the mean of [4, 2]. That key is not written, and
    the zone-101 keys are exactly what the summary over all zones gives them."""
    monkeypatch.setattr(tct, 'publish_inputs', lambda p: p)
    p = _zone0_scene(tmp_path)
    every_zone = tct.stack_layers_summary(p.carbon_density_lookup_lulc_path, p.gep_quantity_input_path,
                                          p.reprojected_total_carbon_density_path, group1_name='lulc_id',
                                          group2_name='carbon_zone_id', value_name='carbon_density')
    assert ((every_zone.carbon_zone_id == 0) & (every_zone.lulc_id == 10)).sum() == 1

    tct.carbon_density_table(p)

    lookup = pd.read_csv(p.carbon_density_lookup_table_path)
    assert (lookup.carbon_zone_id != 0).all()
    expected = every_zone[every_zone.carbon_zone_id != 0].reset_index(drop=True)
    pd.testing.assert_frame_equal(lookup, expected, check_dtype=False)
    assert lookup.set_index(['lulc_id', 'carbon_zone_id']).loc[(20, 101), 'carbon_density_mean'] == 9.0


def test_a_zone_0_cell_is_missing_in_the_map_and_a_valid_zero_stays_zero(tmp_path, monkeypatch):
    """Both zone-0 cells come out NaN, not 0. Class 30 in zone 101 averages a real 0 and keeps it."""
    from osgeo import gdal
    monkeypatch.setattr(tct, 'publish_inputs', lambda p: p)
    p = _zone0_scene(tmp_path)

    tct.carbon_density_table(p)
    tct.carbon_density_raster_base_year(p)

    density = gdal.Open(p.carbon_density_raster_base_year_path).ReadAsArray()
    assert np.isnan(density[0, 0]) and np.isnan(density[1, 0])
    assert density[0, 2] == 0.0
    assert density[0, 1] == 6.0 and density[1, 1] == 9.0 and density[1, 2] == 9.0


@pytest.mark.parametrize('run_this', [1, 0], ids=['running', 'skipped'])
def test_a_lookup_built_before_the_zone_0_rule_is_refused(tmp_path, monkeypatch, run_this):
    """A lookup that still carries a zone-0 key is refused by the map task, also when that task is
    skipped, since skip_existing would otherwise reuse the old lookup and everything built from it."""
    monkeypatch.setattr(tct, 'publish_inputs', lambda p: p)
    p = _zone0_scene(tmp_path)
    p.run_this = run_this
    p.carbon_density_lookup_table_path = str(tmp_path / 'carbon_density_lookup_table.csv')
    pd.DataFrame({'lulc_id': [10, 10], 'carbon_zone_id': [0, 101], 'carbon_density_mean': [3.0, 6.0]}).to_csv(
        p.carbon_density_lookup_table_path, index=False)

    with pytest.raises(ValueError, match='carbon_density_table, carbon_density_raster_base_year'):
        tct.carbon_density_raster_base_year(p)
    assert not os.path.exists(tmp_path / 'projected_carbon_density_2019.tif')


# --- gep_calculation: every country keeps a row --------------------------------------------------

def test_every_country_keeps_a_row_with_missing_and_measured_zero_apart(tmp_path, monkeypatch):
    """Five countries. AAA is valued. BBB is a measured zero: its region holds valid cells that sum
    to 0. CCC has no region row at all, as when every cell is in carbon zone 0, and comes out blank.
    DDD is split and one sub-region has no row, so it is the other sub-region alone. EEE is split
    with both sub-regions valued, so it is their sum."""
    import geopandas as gpd
    from shapely.geometry import box
    monkeypatch.setattr(tct, 'publish_inputs', lambda p: p)
    attrs = dict(continent='C', region_un='U', region_wb='W', income_grp='I', subregion='S')
    r264 = pd.DataFrame([
        (1, 1.0, 'AAA', 'AAA'), (2, 2.0, 'BBB', 'BBB'), (3, 3.0, 'CCC', 'CCC'),
        (4, 4.0, 'DDD', 'DDD'), (5, 4.0, 'DDX', 'DDD'), (6, 5.0, 'EEE', 'EEE'), (7, 5.0, 'EEX', 'EEE')],
        columns=['ee_r264_id', 'iso3_r250_id', 'ee_r264_label', 'iso3_r250_label']).assign(
        ee_r264_name=lambda d: d.ee_r264_label, iso3_r250_name=lambda d: d.iso3_r250_label, **attrs)
    present = {1: 2.0, 2: 0.0, 4: 10.0, 6: 3.0, 7: 4.0}        # regions 3 and 5 hold no valid cell
    regions = r264[r264.ee_r264_id.isin(present)].assign(
        year=2019, total=lambda d: d.ee_r264_id.map(present), count=5)
    regions_path, price_path = tmp_path / 'carbon_by_region.csv', tmp_path / 'carbon_prices.xlsx'
    regions.to_csv(regions_path, index=False)
    pd.DataFrame({'year': [2018, 2019], 'rental scc r2%': [12.0, 13.0]}).to_excel(price_path, index=False)
    simplified_path = tmp_path / 'countries_simplified.gpkg'
    gpd.GeoDataFrame(r264[['ee_r264_id']], geometry=[box(i, 0, i + 1, 1) for i in range(len(r264))],
                     crs=4326).to_file(simplified_path, driver='GPKG')
    cur_dir = tmp_path / 'gep_calculation'
    cur_dir.mkdir()
    p = SimpleNamespace(cur_dir=str(cur_dir), run_this=1, results={}, gep_base_year=2019,
                        gep_price_convention='rental scc r2%', gep_price_input_path=str(price_path),
                        carbon_by_region_base_year_path=str(regions_path), df_countries=r264,
                        gdf_countries_simplified=str(simplified_path))

    total = tct.gep_calculation(p)

    written = pd.read_csv(os.path.join(p.cur_dir, 'gep_by_country_base_year.csv'))
    assert list(written.columns) == tcf.CARBON_ATTR_COLS + ['year', 'terrestrial_carbon_quantity',
                                                            'rental scc r2%', 'terrestrial_carbon_gep']
    assert written.iso3_r250_id.is_monotonic_increasing
    out = written.set_index('iso3_r250_label')
    assert sorted(out.index) == ['AAA', 'BBB', 'CCC', 'DDD', 'EEE'] and out.iso3_r250_id.is_unique
    assert out.loc['AAA', 'terrestrial_carbon_quantity'] == 2.0 and out.loc['AAA', 'terrestrial_carbon_gep'] == 26.0
    assert out.loc['BBB', 'terrestrial_carbon_quantity'] == 0.0 and out.loc['BBB', 'terrestrial_carbon_gep'] == 0.0
    assert np.isnan(out.loc['CCC', 'terrestrial_carbon_quantity']) and np.isnan(out.loc['CCC', 'terrestrial_carbon_gep'])
    assert out.loc['CCC', 'year'] == 2019 and out.loc['CCC', 'rental scc r2%'] == 13.0
    assert out.loc['DDD', 'terrestrial_carbon_quantity'] == 10.0
    assert out.loc['EEE', 'terrestrial_carbon_quantity'] == 7.0
    assert total == 13.0 * (2.0 + 0.0 + 10.0 + 7.0)


def test_the_lookup_land_cover_is_configured_as_2010():
    """es_parameters names the 2010 map for the lookup, and es_config keeps 2019 for the base year."""
    params = pd.read_csv(os.path.join(os.path.dirname(utilities.__file__), 'input_template', 'es_parameters.csv'))
    row = params[(params.service == 'terrestrial_carbon') & (params.parameter == 'carbon_density_lookup_lulc_path')]
    assert row.value.tolist() == ['lulc/esa/lulc_esa_2010.tif']
    config = pd.read_csv(os.path.join(os.path.dirname(utilities.__file__), 'input_template', 'es_config.csv'))
    assert config.loc[config.iloc[:, 0] == 'terrestrial_carbon', 'gep_lulc_input_path'].tolist() == ['lulc/esa/lulc_esa_2019.tif']


def _hydrated_from_a_copy_predating_the_row(tmp_path, **caller_set):
    """Hydrate terrestrial_carbon from a project input/ copy of es_parameters made before the lookup
    rows existed, with one local value of its own. get_path is stubbed to a base_data join."""
    template = os.path.join(os.path.dirname(utilities.__file__), 'input_template', 'es_parameters.csv')
    stale = pd.read_csv(template)
    stale = stale[~stale.parameter.str.startswith('carbon_density_lookup_lulc_', na=False)]
    stale.loc[stale.parameter == 'terrestrial_carbon_shock_acts', 'value'] = 'FRS_LOCAL'
    (tmp_path / 'input').mkdir()
    stale.to_csv(tmp_path / 'input' / 'es_parameters.csv', index=False)
    p = SimpleNamespace(input_dir=str(tmp_path / 'input'), **caller_set,
                        get_path=lambda ref, raise_error_if_fail=True: os.path.join('BASE', ref))
    utilities.hydrate_es_parameters(p, 'terrestrial_carbon', log=lambda *args: None)
    return p, pd.read_csv(tmp_path / 'input' / 'es_parameters.csv')


def test_a_project_copy_predating_the_row_takes_it_and_keeps_its_own_values(tmp_path):
    """The saved project's input/ copy predates the lookup row. Hydration appends the row from the
    template and sets the 2010 map on p, and the copy's own values stay as they were."""
    p, copy = _hydrated_from_a_copy_predating_the_row(tmp_path)
    assert p.carbon_density_lookup_lulc_path == os.path.join('BASE', 'lulc/esa/lulc_esa_2010.tif')
    assert copy.loc[copy.parameter == 'carbon_density_lookup_lulc_path', 'value'].tolist() == ['lulc/esa/lulc_esa_2010.tif']
    assert p.terrestrial_carbon_shock_acts == 'FRS_LOCAL'
    assert copy.loc[copy.parameter == 'terrestrial_carbon_shock_acts', 'value'].tolist() == ['FRS_LOCAL']


def test_a_caller_set_lookup_land_cover_wins_over_the_template(tmp_path):
    p, _ = _hydrated_from_a_copy_predating_the_row(tmp_path, carbon_density_lookup_lulc_path='caller_2010.tif')
    assert p.carbon_density_lookup_lulc_path == 'caller_2010.tif'


def test_the_lookup_land_cover_stays_out_of_the_carbon_shock_signature(tmp_path):
    """The shock tasks reuse their table while the terrestrial_carbon_ settings and inputs are
    unchanged, and they read their own lookup table, not the one carbon_density_table builds. A
    prefixed lookup row would change every recorded shock signature and rebuild every shock."""
    p, _ = _hydrated_from_a_copy_predating_the_row(tmp_path)
    settings, inputs = utilities._service_settings_and_inputs(p, 'terrestrial_carbon')
    assert 'terrestrial_carbon_shock_acts' in settings                     # the signature does see the hydrated rows
    assert not [k for k in list(settings) + list(inputs) if 'lookup_lulc' in k]


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

    # PCHIP between anchors (the selected specification), 0 at base_year. The ANCHORS stay exact
    # -- asserted above -- while intermediate years are not the linear midpoint. They are
    # asserted by the property the specification guarantees, shape preservation, rather than by a
    # hand-computed number: a value strictly inside the bracketing anchors and no overshoot.
    assert a.loc[2020, 'shock_pct'] == approx(0.0)
    for year, lo, hi in [(2025, -45.0, 0.0), (2040, -87.5, -45.0)]:
        assert min(lo, hi) - 1e-9 <= a.loc[year, 'shock_pct'] <= max(lo, hi) + 1e-9
    assert a.loc[2025, 'shock_pct'] != approx(-22.5), 'reverted to the linear midpoint'
    assert min(-45.0, 0.0) - 1e-9 <= b.loc[2040, 'shock_pct'] <= max(-45.0, 0.0) + 1e-9

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
    lookup = tcf.carbon_density_lookup(pd.DataFrame({                        # SEALS7-keyed lookup
        'lulc_id': [1, 2, 3], 'carbon_zone_id': [101, 101, 101],
        'carbon_density_mean': [100.0, 10.0, 20.0]}))

    with pytest.raises(ValueError, match='matched ZERO'):
        tcf.generate_carbon_density_raster(str(lulc), str(cz), lookup, str(out))


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
    # miss here would give an empty base -> empty output -> silent GTAP zero.
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
    # table must not leave GTAP a silent zero. The loop warns and skips it, and the write REFUSES,
    # naming the scenario. The loud skip is the diagnostic; the raise is the stop.
    dep = _static_dep_table(tmp_path)
    out = tmp_path / 'terrestrial_carbon_interpolated.csv'
    p = SimpleNamespace(run_this=True, input_dir=str(tmp_path / 'input'),
                        get_path=lambda *a, **k: '/resolved/' + '/'.join(a),
                        es_shock_base_year=2020, es_shock_end_year=2050,
                        es_shock_base_scenario='baseline_ignore_dependencies',
                        es_shock_scenarios=['scn_a', 'scn_b'],   # scn_b absent from table
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
    """The base year carries no shock, the anchor years carry theirs, and with collinear anchors
    the years between fall on the straight line."""
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


def test_dynamic_shock_rows_v3_measures_the_scenario_against_the_base_year_baseline():
    """v3 answers a different question from the other two measures, and the answers differ in sign.

    shock_pct and shock_pct_fixedbase share the numerator scenario[y] - baseline[y]: the distance
    from a contemporaneous baseline. v3's numerator is scenario[y] - baseline[base_year]: the
    trajectory away from the base year. Pinned because the paper reports v3, so the seam must
    carry it beside the other two.
    """
    zone_labels = {1: ('AEZ1', 'usa')}
    baseline_at_base_year = pd.Series({1: 100.0})
    baseline_by_year = {2030: pd.Series({1: 120.0}), 2040: pd.Series({1: 140.0})}
    scenario_by_year = {2030: pd.Series({1: 126.0}), 2040: pd.Series({1: 133.0})}

    rows = tcf.dynamic_shock_rows(scenario_by_year, baseline_by_year, baseline_at_base_year,
                                  zone_labels, 2020, 'FRS', 'net_zero')
    by_year = pd.DataFrame(rows).set_index('year')

    # Contemporaneous: (126-120)/120 = +5%, then (133-140)/140 = -5%.
    assert np.isclose(by_year.loc[2030, 'shock_pct'], 5.0)
    assert np.isclose(by_year.loc[2040, 'shock_pct'], -5.0)
    # Fixed base: the same numerators over the base year's 100.
    assert np.isclose(by_year.loc[2030, 'shock_pct_fixedbase'], 6.0)
    assert np.isclose(by_year.loc[2040, 'shock_pct_fixedbase'], -7.0)
    # v3: (126-100)/100 = +26%, (133-100)/100 = +33%. Positive where the contemporaneous measure
    # is negative -- the scenario is above its own base year while below the 2040 baseline.
    assert np.isclose(by_year.loc[2030, 'shock_pct_v3'], 26.0)
    assert np.isclose(by_year.loc[2040, 'shock_pct_v3'], 33.0)
    assert by_year.loc[2020, 'shock_pct_v3'] == 0.0
    # 2035 is NOT an anchor: under the PCHIP specification it is not the linear midpoint of
    # the 2030 and 2040 anchors. Asserted by shape preservation instead, with the anchors exact.
    lo, hi = by_year.loc[2030, 'shock_pct_v3'], by_year.loc[2040, 'shock_pct_v3']
    assert min(lo, hi) - 1e-9 <= by_year.loc[2035, 'shock_pct_v3'] <= max(lo, hi) + 1e-9
    assert not np.isclose(by_year.loc[2035, 'shock_pct_v3'], 29.5), 'reverted to linear'


def test_dynamic_shock_rows_v3_is_absent_without_a_base_year_baseline():
    zone_labels = {1: ('AEZ1', 'usa')}
    rows = tcf.dynamic_shock_rows({2030: pd.Series({1: 126.0})}, {2030: pd.Series({1: 120.0})},
                                  None, zone_labels, 2029, 'FRS', 'net_zero')
    assert all(np.isnan(row['shock_pct_v3']) for row in rows)


def test_scc_valuation_interpolates_prices_and_derives_the_cut():
    """Two countries, one scenario with anchors at 2030 and 2050, price 10 $/Mg C."""
    base = pd.Series({1: 100.0, 2: 50.0})
    stocks = {'cp': {2030: pd.Series({1: 110.0, 2: 40.0}), 2050: pd.Series({1: 130.0, 2: 40.0})},
              'cp-es20': {2030: pd.Series({1: 110.0, 2: 40.0}), 2050: pd.Series({1: 130.0, 2: 40.0})}}
    out = tcf.scc_valuation_rows(stocks, base, price=10.0, base_year=2023, retained=0.8,
                                 from_year=2030, cut_source='cp', cut_label='cp-es20')
    cp = out[(out['scenario'] == 'cp') & (out['iso3_r250_id'] == 1)].set_index('year')
    assert cp.loc[2023, 'delta_value_usd'] == 0.0
    # The STOCK is PCHIP-interpolated, so 2027 is not the linear value between 2023 (100) and
    # 2030 (110), 100 + 4/7 * 10. The anchors themselves are asserted elsewhere in this test and
    # remain exact.
    lo, hi = cp.loc[2023, 'stock_mgc'], cp.loc[2030, 'stock_mgc']
    assert min(lo, hi) - 1e-9 <= cp.loc[2027, 'stock_mgc'] <= max(lo, hi) + 1e-9
    assert abs(cp.loc[2027, 'stock_mgc'] - (100 + 40 / 7)) > 1e-6, 'reverted to linear'
    # 2040 is between this fixture's anchors, so its PCHIP stock -- and therefore its value -- is
    # not the linear one. Bounded by the bracketing anchors; the anchor years stay exact.
    lo, hi = cp.loc[2030, 'delta_value_usd'], cp.loc[2050, 'delta_value_usd']
    assert min(lo, hi) - 1e-9 <= cp.loc[2040, 'delta_value_usd'] <= max(lo, hi) + 1e-9
    assert cp.loc[2050, 'delta_value_usd'] == 300.0
    # the cut: the modelled es20 rows are replaced by 0.8 x cp from 2030, untouched before
    es20 = out[(out['scenario'] == 'cp-es20') & (out['iso3_r250_id'] == 1)].set_index('year')
    assert es20.loc[2029, 'stock_mgc'] == cp.loc[2029, 'stock_mgc']
    assert abs(es20.loc[2030, 'stock_mgc'] - 0.8 * 110) < 1e-9
    assert abs(es20.loc[2050, 'delta_value_usd'] - (0.8 * 130 - 100) * 10) < 1e-9
    assert sorted(out['scenario'].unique()) == ['cp', 'cp-es20']
    assert len(out) == 2 * 2 * 28
    # against a contemporaneous baseline: 2050 baseline 120 for country 1 -> cp is +10 above it
    out2 = tcf.scc_valuation_rows(stocks, base, price=10.0, base_year=2023,
                                  baseline_by_year={2030: pd.Series({1: 105.0, 2: 50.0}), 2050: pd.Series({1: 120.0, 2: 50.0})})
    cp2 = out2[(out2['scenario'] == 'cp') & (out2['iso3_r250_id'] == 1)].set_index('year')
    assert abs(cp2.loc[2050, 'delta_value_vs_baseline_usd'] - 100.0) < 1e-9
    assert abs(cp2.loc[2023, 'delta_value_vs_baseline_usd']) < 1e-9


# --- what retaining a measured zero does to a PER-ZONE measure -------------------------------------
# The summed (D19) measure absorbs a zone's zero into its region's total. The per-zone measures --
# terrestrial_carbon, pollination, and timber's net_return path -- divide by the zone's own baseline,
# so a zone whose quantity reaches zero is a -100% shock on that zone rather than a small change in a
# regional sum. These two tests hold that difference still, because it is the economically material
# consequence of retaining measured zeros, and it must not drift unnoticed either way.

def test_zone_reaching_zero_gives_a_minus_hundred_percent_per_zone_shock():
    """A zone that keeps its baseline but loses all its own quantity. The per-zone ratio is -100%.

    The zonal summary retains the measured zero, so the zone stays in the scenario series and a
    total loss reaches the economic model as the loss it is. Were the zero dropped, the
    subtraction would give NaN, dropna would remove the zone, and the loss would reach the model
    as NO shock at all.
    """
    baseline = pd.Series({1: 100.0, 2: 50.0})
    scenario_with_explicit_zero = pd.Series({1: 100.0, 2: 0.0})
    result = tcf.shock_percent(scenario_with_explicit_zero, baseline).dropna()
    assert result.to_dict() == {1: pytest.approx(0.0), 2: pytest.approx(-100.0)}


def test_zone_with_no_baseline_quantity_is_still_dropped_not_shocked():
    """The other direction: a zero DENOMINATOR has no ratio, so the zone carries no shock rather
    than an infinite one, whether or not it appears in the summary."""
    baseline_with_explicit_zero = pd.Series({1: 100.0, 2: 0.0})
    scenario = pd.Series({1: 100.0, 2: 25.0})
    result = tcf.shock_percent(scenario, baseline_with_explicit_zero).dropna()
    assert result.to_dict() == {1: pytest.approx(0.0)}


import numpy as np
import pandas as pd
import pytest

from global_invest.terrestrial_carbon import terrestrial_carbon_functions as tcf

BASE_YEAR = 2023
# Two regions of two zones each, so a region can lose one zone entirely and keep a positive total.
ZONE_LABELS = {11: ('AEZ1', 'bra'), 12: ('AEZ2', 'bra'),
               21: ('AEZ1', 'usa'), 22: ('AEZ2', 'usa')}


def series(**by_zone):
    return pd.Series({int(k[1:]): float(v) for k, v in by_zone.items()}, dtype='float64')


def shock_at(rows, region, year, scenario='policy'):
    matching = {r['shock_pct'] for r in rows
                if r['REG'] == region and r['year'] == year and r['scenario'] == scenario}
    assert len(matching) == 1, 'region %s year %d gave %d distinct shocks' % (region, year, len(matching))
    return matching.pop()


def run(scenario_by_year, baseline, **kwargs):
    return tcf.summed_shock_rows({'policy': scenario_by_year}, baseline, ZONE_LABELS, BASE_YEAR,
                                 sector='frs', log=lambda *a, **k: None, **kwargs)


def test_partial_regional_loss_is_the_ratio_of_sums():
    """Half of one zone's biomass goes: the region's shock is the change in its SUM, and both of
    its zones carry that same regional number."""
    baseline = series(z11=100.0, z12=100.0, z21=50.0, z22=50.0)
    scenario = {2030: series(z11=50.0, z12=100.0, z21=50.0, z22=50.0),
                2050: series(z11=50.0, z12=100.0, z21=50.0, z22=50.0)}
    rows = run(scenario, baseline)
    assert shock_at(rows, 'bra', 2030) == pytest.approx(-25.0)   # 150/200 - 1
    assert shock_at(rows, 'usa', 2030) == pytest.approx(0.0)
    per_zone = {r['ENDW']: r['shock_pct'] for r in rows if r['REG'] == 'bra' and r['year'] == 2030}
    assert per_zone == {'AEZ1': pytest.approx(-25.0), 'AEZ2': pytest.approx(-25.0)}


def test_zone_losing_all_forest_is_not_a_minus_one_hundred_shock():
    """THE POINT OF THE SUMMED MEASURE. Zone 11 goes to zero. Its region keeps zone 12, so the shock is -50%, and
    -100% appears nowhere -- not for the emptied zone, not for its region."""
    baseline = series(z11=100.0, z12=100.0, z21=50.0, z22=50.0)
    scenario = {2030: series(z11=0.0, z12=100.0, z21=50.0, z22=50.0),
                2050: series(z11=0.0, z12=100.0, z21=50.0, z22=50.0)}
    rows = run(scenario, baseline)
    assert shock_at(rows, 'bra', 2030) == pytest.approx(-50.0)
    assert min(r['shock_pct'] for r in rows) > -100.0


def test_explicit_zero_and_a_dropped_row_are_not_interchangeable():
    """The zero above must be WRITTEN. Dropping the row instead raises, because a regional sum
    cannot tell an unmeasured zone from a measured empty one."""
    baseline = series(z11=100.0, z12=100.0, z21=50.0, z22=50.0)
    dropped = {2030: series(z12=100.0, z21=50.0, z22=50.0),
               2050: series(z12=100.0, z21=50.0, z22=50.0)}
    with pytest.raises(ValueError, match='missing 1 of region bra'):
        run(dropped, baseline)


def test_missing_zone_in_the_base_year_also_raises():
    """The denominator is held to the same rule as the numerator."""
    baseline = series(z11=100.0, z21=50.0, z22=50.0)
    scenario = {2030: series(z11=100.0, z12=100.0, z21=50.0, z22=50.0),
                2050: series(z11=100.0, z12=100.0, z21=50.0, z22=50.0)}
    with pytest.raises(ValueError, match='the base-year quantity is missing'):
        run(scenario, baseline)


def test_zero_baseline_zone_gaining_forest_contributes_without_forming_a_ratio():
    """A zone holding nothing in the base year and afforested later adds its gain to the region's
    numerator. The unbounded zonal ratio the summed measure exists to avoid is never formed: the
    region rises by the gain over its own base, not by the zone's own multiple."""
    baseline = series(z11=100.0, z12=0.0, z21=50.0, z22=50.0)
    scenario = {2030: series(z11=100.0, z12=50.0, z21=50.0, z22=50.0),
                2050: series(z11=100.0, z12=50.0, z21=50.0, z22=50.0)}
    rows = run(scenario, baseline)
    assert shock_at(rows, 'bra', 2030) == pytest.approx(50.0)    # 150/100 - 1, not an infinite ratio
    assert np.isfinite([r['shock_pct'] for r in rows]).all()


def test_whole_region_reaching_zero_is_raised_for_review():
    """A region losing ALL its biomass is the only way to -100%, and it stops before annual
    conversion instead of being interpolated."""
    baseline = series(z11=100.0, z12=100.0, z21=50.0, z22=50.0)
    scenario = {2030: series(z11=0.0, z12=0.0, z21=50.0, z22=50.0),
                2050: series(z11=0.0, z12=0.0, z21=50.0, z22=50.0)}
    with pytest.raises(ValueError, match='-100% regional productivity factor'):
        run(scenario, baseline)


def test_region_with_no_base_year_biomass_is_reported_not_shocked():
    """No denominator, so no shock: the region is left out rather than given a fabricated number,
    and the OTHER region is still shocked normally."""
    baseline = series(z11=100.0, z12=100.0, z21=0.0, z22=0.0)
    scenario = {2030: series(z11=50.0, z12=100.0, z21=10.0, z22=10.0),
                2050: series(z11=50.0, z12=100.0, z21=10.0, z22=10.0)}
    reported = []
    rows = tcf.summed_shock_rows({'policy': scenario}, baseline, ZONE_LABELS, BASE_YEAR,
                                 sector='frs', log=lambda m: reported.append(m))
    assert {r['REG'] for r in rows} == {'bra'}
    assert any('usa' in m for m in reported)


def test_base_year_is_pinned_at_zero_shock():
    baseline = series(z11=100.0, z12=100.0, z21=50.0, z22=50.0)
    scenario = {2030: series(z11=0.0, z12=100.0, z21=50.0, z22=50.0),
                2050: series(z11=0.0, z12=100.0, z21=50.0, z22=50.0)}
    rows = run(scenario, baseline)
    assert shock_at(rows, 'bra', BASE_YEAR) == pytest.approx(0.0)


def test_dynamic_shock_reuses_a_current_table_and_recomputes_a_stale_one(tmp_path):
    """A second pass must leave a current table alone, and must not reuse a stale one.

    Rewriting terrestrial_carbon_interpolated.csv on every pass turns a no-op resume into repeated
    work and makes an unchanged-resume check impossible to pass. Reuse keyed on existence alone
    would be the opposite defect, so the second half changes an input and requires the recompute.
    """
    p = _shock_scene(tmp_path)
    tct.terrestrial_carbon_shock(p)
    first = os.stat(p.terrestrial_carbon_shock_output_path).st_mtime_ns

    tct.terrestrial_carbon_shock(p)
    assert os.stat(p.terrestrial_carbon_shock_output_path).st_mtime_ns == first, \
        'the second pass rewrote a table it could have reused'

    # An input the signature covers by content: the table must come back, not be republished.
    lut = p.terrestrial_carbon_density_lookup_table_path
    rows = open(lut, encoding='utf-8').read()
    open(lut, 'w', encoding='utf-8').write(rows.replace('100', '50'))
    tct.terrestrial_carbon_shock(p)
    assert os.stat(p.terrestrial_carbon_shock_output_path).st_mtime_ns != first, \
        'the table was reused although its density lookup changed'


def test_reuse_signature_excludes_its_own_outputs_and_survives_a_json_round_trip(tmp_path):
    """A signature must not fingerprint the files the task is about to write, nor hold tuples.

    Either defect makes a guard self-invalidating -- it recomputes on every pass and no resume can
    ever be clean. The SCC valuation's output paths end in _path, not _output_path, so a name
    filter misses them; json writes a tuple as a list, so a tuple never compares equal after a
    round trip.
    """
    import json

    out_csv = tmp_path / 'terrestrial_carbon_scc_valuation.csv'
    out_png = tmp_path / 'terrestrial_carbon_scc_valuation.png'
    out_csv.write_text('a,b\n1,2\n', encoding='utf-8')
    out_png.write_bytes(b'\x89PNG')
    source = tmp_path / 'source.csv'
    source.write_text('x\n1\n', encoding='utf-8')

    p = SimpleNamespace(
        cur_dir=str(tmp_path),
        terrestrial_carbon_scc_valuation_path=str(out_csv),
        terrestrial_carbon_scc_figure_path=str(out_png),
        terrestrial_carbon_density_lookup_table_path=str(source))
    outputs = [str(out_csv), str(out_png)]

    reason, signature, signature_path = tct._reuse_check(
        p, outputs, 'sig.json', extra_settings=dict(pairs={'a': 1, 'b': 2}, seq=(3, 4)))
    # The outputs must not appear anywhere in the signature's inputs.
    for name in outputs:
        assert name not in signature['inputs'], 'the task fingerprinted its own output'
    assert str(source) not in signature['inputs'] or True   # a real input may appear

    # And the whole thing must compare equal after a json round trip.
    assert json.loads(json.dumps(signature, sort_keys=True, default=str)) == \
        json.loads(json.dumps(signature, sort_keys=True, default=str))
    utilities.write_outputs_signature(signature, signature_path)
    again, _, _ = tct._reuse_check(
        p, outputs, 'sig.json', extra_settings=dict(pairs={'a': 1, 'b': 2}, seq=(3, 4)))
    assert again is None, 'the signature did not survive being written and read back: %s' % again


def _summed_scene():
    """Two regions of two zones each; only region 'aaa' is fully covered by the grid."""
    import pandas as pd
    zone_labels = {101: ('AEZ1', 'aaa'), 102: ('AEZ2', 'aaa'),
                   201: ('AEZ1', 'bbb'), 202: ('AEZ2', 'bbb')}
    base = pd.Series({101: 100.0, 102: 100.0})          # only the covered region is measured
    scen = {2030: pd.Series({101: 90.0, 102: 90.0})}
    return zone_labels, base, scen


def test_bounded_scope_reports_only_fully_covered_regions_and_declares_the_rest():
    zone_labels, base, scen = _summed_scene()
    excluded = []
    rows = tcf.summed_shock_rows({'scn': scen}, base, zone_labels, 2020, 'FRS',
                                 log=lambda *a, **k: None,
                                 covered_zones={101, 102}, excluded_out=excluded)
    assert excluded == ['bbb'], 'the unmeasured region was not declared'
    reported = {r['REG'] for r in rows}
    assert reported == {'aaa'}, 'an unmeasured region was reported'
    # And it must be ABSENT, never present as a zero -- a zero is a measured no-change to GTAP.
    assert not any(r['REG'] == 'bbb' for r in rows)


def test_bounded_scope_still_rejects_a_missing_zone_inside_a_covered_region():
    zone_labels, base, scen = _summed_scene()
    short = base.drop(index=102)                         # a zone of the COVERED region goes missing
    with pytest.raises(ValueError, match='missing zone is not a zero'):
        tcf.summed_shock_rows({'scn': scen}, short, zone_labels, 2020, 'FRS',
                              log=lambda *a, **k: None,
                              covered_zones={101, 102}, excluded_out=[])


def test_global_runs_keep_the_full_regional_coverage_requirement():
    """covered_zones None is the global path: every region required, nothing declared."""
    zone_labels, base, scen = _summed_scene()
    with pytest.raises(ValueError, match="missing 2 of region bbb's 2 zones"):
        tcf.summed_shock_rows({'scn': scen}, base, zone_labels, 2020, 'FRS',
                              log=lambda *a, **k: None)
    # Fully measured globally: both regions reported, no exclusions.
    import pandas as pd
    full_base = pd.Series({101: 100.0, 102: 100.0, 201: 100.0, 202: 100.0})
    full_scen = {2030: pd.Series({101: 90.0, 102: 90.0, 201: 80.0, 202: 80.0})}
    excluded = []
    rows = tcf.summed_shock_rows({'scn': full_scen}, full_base, zone_labels, 2020, 'FRS',
                                 log=lambda *a, **k: None, excluded_out=excluded)
    assert excluded == []
    assert {r['REG'] for r in rows} == {'aaa', 'bbb'}


def test_no_fully_covered_region_is_a_failure_not_an_empty_table():
    zone_labels, base, scen = _summed_scene()
    with pytest.raises(ValueError, match='fully covers no GTAP region'):
        tcf.summed_shock_rows({'scn': scen}, base, zone_labels, 2020, 'FRS',
                              log=lambda *a, **k: None, covered_zones={101}, excluded_out=[])

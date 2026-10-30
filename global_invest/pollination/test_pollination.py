"""Unit tests for pollination.

Two layers. The shock and valuation arithmetic is pinned as pure functions over hand-built series
whose expected values are written out in the assertions. Below that, one end-to-end test runs the
real calculation (a tiny USD value raster through utilities.summarize_raster_by_region into
gep_calculation) to pin the r250 contract: a split country is summed once, and the map gpkg
carries per-sub-region rows that are never summed.
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import hazelbean as hb
from global_invest import utilities

from global_invest.pollination import pollination_functions as pf
from global_invest.pollination import pollination_tasks as pt


def _base_data_project():
    """A bare ProjectFlow, only for its base_data_dir.

    The anchors are inputs, so a test finds them the way a run does rather than by walking
    directories of its own.
    """
    import tempfile
    import hazelbean as hb
    return hb.ProjectFlow(project_dir=os.path.join(tempfile.mkdtemp(), 'anchors'))


ATTRS = {'iso3_r250_label': {156: 'CHN', 528: 'NLD'},
         'iso3_r250_name': {156: 'China', 528: 'Netherlands'},
         'continent': {156: 'Asia', 528: 'Europe'},
         'region_un': {156: 'Asia', 528: 'Europe'},
         'region_wb': {156: 'EAP', 528: 'ECA'},
         'income_grp': {156: 'UMIC', 528: 'HIC'},
         'subregion': {156: 'Eastern Asia', 528: 'Western Europe'}}

ZONE_A = ('AEZ1', 'usa')
ZONE_B = ('AEZ2', 'bra')


def _zone_frame(columns_by_year):
    """An anchor table shaped the way anchor_shock_tables builds it: rows keyed on (ENDW, REG)."""
    return pd.DataFrame({year: pd.Series(values) for year, values in columns_by_year.items()})


# ---------------------------------------------------------------------------
# Shock arithmetic.
# ---------------------------------------------------------------------------

def test_anchor_shock_tables_subtract_the_baseline_and_rebase_onto_it():
    # Both inputs are percent changes against the same fixed base-year value. Zone A: scenario is
    # 4 points below its baseline against the fixed base, and the baseline itself has grown 25%,
    # so contemporaneously it is -4 / 1.25 = -3.2 percent.
    scenario = {2030: pd.Series({ZONE_A: 21.0, ZONE_B: -5.0})}
    baseline = {2030: pd.Series({ZONE_A: 25.0, ZONE_B: -50.0})}

    fixedbase, contemporaneous = pf.anchor_shock_tables(scenario, baseline)
    assert fixedbase.loc[ZONE_A, 2030] == -4.0
    assert np.isclose(contemporaneous.loc[ZONE_A, 2030], -3.2)
    # Zone B: 45 points above its baseline against the fixed base, baseline halved, so 45 / 0.5.
    assert fixedbase.loc[ZONE_B, 2030] == 45.0
    assert np.isclose(contemporaneous.loc[ZONE_B, 2030], 90.0)


def test_anchor_shock_tables_give_nan_where_the_baseline_has_gone_to_zero():
    """A baseline that has lost all its value would divide the contemporaneous shock by zero."""
    scenario = {2030: pd.Series({ZONE_A: -100.0})}
    baseline = {2030: pd.Series({ZONE_A: -100.0})}       # growth factor exactly 0
    fixedbase, contemporaneous = pf.anchor_shock_tables(scenario, baseline)
    assert fixedbase.loc[ZONE_A, 2030] == 0.0
    assert np.isnan(contemporaneous.loc[ZONE_A, 2030])


def test_anchor_shock_tables_drop_a_zone_missing_from_any_anchor_year():
    """The two tables must carry identical rows, so a zone the later year does not cover is
    dropped from both rather than being carried with a hole in it."""
    scenario = {2030: pd.Series({ZONE_A: 10.0, ZONE_B: 10.0}), 2040: pd.Series({ZONE_A: 20.0})}
    baseline = {2030: pd.Series({ZONE_A: 0.0, ZONE_B: 0.0}), 2040: pd.Series({ZONE_A: 0.0})}
    fixedbase, contemporaneous = pf.anchor_shock_tables(scenario, baseline)
    assert fixedbase.index.tolist() == [ZONE_A]
    assert contemporaneous.index.tolist() == [ZONE_A]


def test_dynamic_shock_rows_ramp_from_zero_at_the_base_year_through_each_anchor():
    fixedbase = _zone_frame({2025: {ZONE_A: -4.0}, 2030: {ZONE_A: -8.0}})
    contemporaneous = _zone_frame({2025: {ZONE_A: -3.2}, 2030: {ZONE_A: -6.4}})
    level_usd = pd.Series({ZONE_A: 1000.0})

    rows = pf.dynamic_shock_rows(fixedbase, contemporaneous, level_usd, 'net_zero',
                                 ('V_F', 'OSD'), base_year=2020)
    out = pd.DataFrame(rows)

    # 11 years (2020..2030) x 2 sectors.
    assert len(out) == 22
    by_year = out[out['ACTS'] == 'V_F'].set_index('year')
    assert by_year.loc[2020, 'shock_pct_fixedbase'] == 0.0
    assert np.isclose(by_year.loc[2022, 'shock_pct_fixedbase'], -1.6)   # 2/5 of the way to -4
    assert by_year.loc[2025, 'shock_pct_fixedbase'] == -4.0
    assert np.isclose(by_year.loc[2027, 'shock_pct_fixedbase'], -5.6)   # 2/5 of -4 to -8
    assert by_year.loc[2030, 'shock_pct_fixedbase'] == -8.0
    # shock_pct is the contemporaneous measure, matching carbon.
    assert (out['shock_pct'] == out['shock_pct_contemp']).all()
    assert by_year.loc[2030, 'shock_pct'] == -6.4
    assert (out['value_usd_base'] == 1000.0).all()
    assert out['REG'].unique().tolist() == ['usa']
    assert out['ENDW'].unique().tolist() == ['AEZ1']


def test_dynamic_shock_rows_leave_the_level_missing_for_a_zone_the_denominator_lacks():
    fixedbase = _zone_frame({2030: {ZONE_A: -8.0}})
    rows = pf.dynamic_shock_rows(fixedbase, fixedbase, pd.Series({ZONE_B: 5.0}), 'net_zero',
                                 ('V_F',), base_year=2029)
    assert all(np.isnan(row['value_usd_base']) for row in rows)


def test_static_shock_rows_ramp_the_difference_linearly_to_the_end_year():
    baseline = pd.Series({ZONE_A: 10.0, ZONE_B: 4.0})
    scenario = pd.Series({ZONE_A: 12.0, ZONE_B: 4.0})

    rows = pf.static_shock_rows(baseline, scenario, 'net_zero', ('V_F',), 2020, 2024)
    out = pd.DataFrame(rows).set_index(['REG', 'year'])['shock_pct']

    assert out.loc[('usa', 2020)] == 0.0
    assert out.loc[('usa', 2021)] == 0.5       # a quarter of the way to +2
    assert out.loc[('usa', 2024)] == 2.0
    assert out.loc[('bra', 2024)] == 0.0       # no difference to ramp
    assert len(rows) == 2 * 5                  # two zones x five years, one sector


def test_static_shock_rows_skip_a_zone_the_scenario_does_not_cover():
    """Shocking a zone the scenario table has no row for would put an invented number into the
    economic model, so only the zones both series carry are ramped."""
    baseline = pd.Series({ZONE_A: 10.0, ZONE_B: 4.0})
    scenario = pd.Series({ZONE_A: 12.0})
    rows = pf.static_shock_rows(baseline, scenario, 'net_zero', ('V_F',), 2020, 2024)
    assert {row['REG'] for row in rows} == {'usa'}


# ---------------------------------------------------------------------------
# Zonal arithmetic.
# ---------------------------------------------------------------------------

def test_zonal_pct_change_weights_by_area_and_drops_a_zone_without_baseline_value():
    # Zone 1 spans two pixels of unequal area; zone 2's only pixel has a NaN baseline, so its
    # denominator is zero and the zone appears in neither series.
    zones = np.array([[1, 1, 2]])
    baseline = np.array([[10.0, 20.0, np.nan]])
    diff = np.array([[1.0, -2.0, 5.0]])
    area = np.array([[2.0, 1.0, 1.0]])
    labels = {1: ZONE_A, 2: ZONE_B}

    pct, level = pf.zonal_pct_change(diff, baseline, area, zones, labels)

    assert list(pct.index) == [ZONE_A] and list(level.index) == [ZONE_A]
    assert level.loc[ZONE_A] == 40.0                    # 10x2 + 20x1
    assert pct.loc[ZONE_A] == 0.0                       # (1x2 - 2x1) / 40 x 100


def test_zonal_pct_change_treats_nan_diff_pixels_as_zero_change():
    zones = np.array([[1, 1]])
    baseline = np.array([[10.0, 10.0]])
    diff = np.array([[3.0, np.nan]])                    # the NaN pixel contributes no numerator
    area = np.array([[1.0, 1.0]])

    pct, level = pf.zonal_pct_change(diff, baseline, area, zones, {1: ZONE_A})

    assert level.loc[ZONE_A] == 20.0
    assert pct.loc[ZONE_A] == 15.0                      # 3 / 20 x 100


def test_zone_labels_from_boundary_keeps_one_row_per_zone():
    boundary = pd.DataFrame({'ee_r50_aez18_id': [101, 101, 102], 'aez18_id': [1, 1, 2],
                             'gtapv7_r50_label': ['usa', 'usa', 'chn']})
    labels = pf.zone_labels_from_boundary(
        boundary, 'ee_r50_aez18_id', 'aez18_id', 'gtapv7_r50_label', 'AEZ%d')
    assert labels == {101: ('AEZ1', 'usa'), 102: ('AEZ2', 'chn')}


# ---------------------------------------------------------------------------
# GEP valuation arithmetic.
# ---------------------------------------------------------------------------

def _region_frame():
    """Three r264 regions, two of them sub-regions of the same country."""
    rows = []
    for region_id, iso, total in [(1, 156, 40.0), (2, 156, 20.0), (3, 528, 8.0)]:
        row = {'ee_r264_id': region_id, 'iso3_r250_id': iso, 'year': 2023, 'total': total}
        row.update({col: mapping[iso] for col, mapping in ATTRS.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def test_collapse_regions_to_countries_sums_a_split_country_once():
    out = pf.collapse_regions_to_countries(_region_frame())
    assert list(out.columns) == utilities.GEP_COUNTRY_ATTR_COLS + ['year', 'pollination_gep']
    by_country = out.set_index('iso3_r250_label')
    assert len(by_country) == 2
    assert by_country.loc['CHN', 'pollination_gep'] == 60.0     # 40 + 20, not 40 and 20
    assert by_country.loc['NLD', 'pollination_gep'] == 8.0
    assert by_country.loc['CHN', 'region_wb'] == 'EAP'


def test_expand_country_values_to_regions_repeats_the_national_value_per_sub_region():
    df_regions = _region_frame()
    out = utilities.expand_country_values_to_regions(
        df_regions, pf.collapse_regions_to_countries(df_regions), 'pollination_gep')
    assert len(out) == 3
    assert out.set_index('ee_r264_id')['pollination_gep'].loc[[1, 2]].tolist() == [60.0, 60.0]
    # The per-region raster totals are kept beside the national value, not overwritten by it.
    assert out.set_index('ee_r264_id')['total'].loc[1] == 40.0


# ---------------------------------------------------------------------------
# End to end through the real zonal chain.
# ---------------------------------------------------------------------------

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

    p = SimpleNamespace(run_this=True, cur_dir=str(tmp_path), results={}, gep_base_year=2023,
                        input_dir=str(tmp_path / 'input'),
                        get_path=lambda *a, **k: '/resolved/' + '/'.join(a),
                        # A real country frame, because the published table is put on the
                        # account's r250 list: every country appears, and one the service has no
                        # value for is NA rather than absent. A placeholder cannot express that.
                        # Shaped like the real correspondence: a split country carries its
                        # sub-region rows plus the canonical one whose r264 label equals its r250
                        # label, and that canonical row is the one a country join keeps.
                        df_countries=pd.DataFrame({
                            'ee_r264_id': [1, 2, 156, 528],
                            'ee_r264_label': ['chn_a', 'chn_b', 'CHN', 'NLD'],
                            'ee_r264_name': ['China A', 'China B', 'China', 'Netherlands'],
                            'iso3_r250_id': [156, 156, 156, 528],
                            'iso3_r250_label': ['CHN', 'CHN', 'CHN', 'NLD'],
                            'iso3_r250_name': ['China', 'China', 'China', 'Netherlands'],
                            **{c: [m[i] for i in [156, 156, 156, 528]] for c, m in ATTRS.items()
                               if c not in ('iso3_r250_label', 'iso3_r250_name')}}),
                        # our own raster, in USD per cell rather than a density
                        pollination_value_raster_path=str(tmp_path / 'poll_value.tif'),
                        pollination_value_raster_rebuilt_path=str(tmp_path / 'poll_value.tif'),
                        pollination_reported_variant='unweighted',
                        gep_quantity_input_path=str(tmp_path / 'poll_value.tif'),
                        gep_regions_input_path=str(tmp_path / 'regions.gpkg'),
                        gep_regions_id_col='ee_r264_id',
                        gdf_countries_simplified=str(tmp_path / 'regions.gpkg'))

    pt.pollination_value_by_region(p)
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


def test_the_module_imports_without_the_crop_benefits_package():
    # The sufficiency and value science lives in this repo, so no pollination module may import
    # crop_benefits, a package declared in no pyproject and configured through a gitignored
    # local.yaml: on a machine without it, the service must still import.
    import importlib
    import sys

    class Blocker:
        def find_module(self, name, path=None):
            return self if name == 'crop_benefits' or name.startswith('crop_benefits.') else None

        def load_module(self, name):
            raise ImportError(f"No module named '{name}'")

    blocker = Blocker()
    sys.meta_path.insert(0, blocker)
    cached = {name: sys.modules.pop(name) for name in list(sys.modules)
              if name.startswith('crop_benefits')}
    try:
        for name in ('pollination_functions', 'pollination_tasks', 'pollination_initialize'):
            module = importlib.import_module(f'global_invest.pollination.{name}')
            assert module is not None
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(cached)


def test_the_settings_object_replaces_the_config_that_was_loaded_from_a_gitignored_file():
    # Seven fields are everything the raster steps read off the crop_benefits Config. They are
    # plain arguments, so a missing config file cannot silently produce a run against the wrong
    # paths.

    from global_invest.pollination import pollination_functions as pf

    settings = pf.SufficiencySettings(
        output_dir='/tmp/out', value_raster_dir='/tmp/base',
        country_raster_path='/tmp/base/poll_value_global_2023usd.tif',
        tile_size=2048, n_workers=4, lulc_classes_path='/tmp/base/classes.csv')
    assert settings.tile_size == 2048 and settings.n_workers == 4
    assert settings.pa_raster_300m_path is None
    # The compression profiles are named constants rather than Config fields.
    assert set(pf.COMPRESSION_PROFILES) == {'continuous', 'categorical', 'defaults'}


# ---------------------------------------------------------------------------
# The 5 km grid: CropGrids' float32 transform and the exact grid it stands for.
# ---------------------------------------------------------------------------

def _write_raster(path, array, transform, nodata=np.nan, crs='EPSG:4326'):
    import rasterio
    with rasterio.open(path, 'w', driver='GTiff', height=array.shape[0], width=array.shape[1],
                       count=1, dtype='float32', crs=crs, transform=transform,
                       nodata=nodata) as dst:
        dst.write(array.astype('float32'), 1)


def test_a_float32_offset_template_does_not_leak_sufficiency_into_an_empty_cell(tmp_path):
    # One farmland pixel (0.8) at row 8, column 17: the last column of the first 5 km cell. On the
    # exact 0.05-degree grid it belongs to that cell alone. CropGrids' float32 coordinates put the
    # template's x origin 6e-6 degrees west, so the pixel overlaps the next cell by a sliver, and
    # because the average ignores missing pixels, that sliver alone would give the empty cell the
    # full 0.8.
    import rasterio
    from affine import Affine
    fine = np.full((36, 36), np.nan)
    fine[8, 17] = 0.8
    _write_raster(tmp_path / 'pollination_sufficiency_t_300m.tif', fine,
                  Affine(0.05 / 18, 0, 0.0, 0, -0.05 / 18, 0.1))
    stored = Affine(0.05, 0, -0.000006, 0, -0.05, 0.1)
    _write_raster(tmp_path / 'template.tif', np.zeros((2, 2)), stored, nodata=-9999.0)
    settings = pf.SufficiencySettings(
        output_dir=str(tmp_path), value_raster_dir=str(tmp_path),
        country_raster_path=str(tmp_path / 'template.tif'), tile_size=8192, n_workers=1,
        lulc_classes_path='')
    with rasterio.open(pt.run_pollination_sufficiency_5km(settings, scenario='t')) as result:
        sufficiency = result.read(1)
        assert result.transform == stored        # still on the grid of the raster it weights
    assert sufficiency[0, 0] == pytest.approx(0.8)
    assert np.isnan(sufficiency[0, 1]), sufficiency
    assert np.isnan(sufficiency[1]).all()


def test_the_nominal_grid_is_the_exact_grid_a_float32_transform_stands_for():
    from affine import Affine
    fitted = Affine(0.0500000017, 0, -180.0000061044, 0, -0.0499999992, 89.9999984737)
    assert pf.nominal_grid_transform(fitted) == Affine(0.05, 0, -180.0, 0, -0.05, 90.0)
    # A transform further from an exact grid than float32 rounding explains is another grid.
    with pytest.raises(ValueError):
        pf.nominal_grid_transform(Affine(0.05, 0, -180.01, 0, -0.05, 90.0))


def test_a_rounded_global_template_lands_on_hazelbeans_global_grid():
    from affine import Affine
    fitted = Affine(0.0500000017, 0, -180.0000061044, 0, -0.0499999992, 89.9999984737)
    assert pf.nominal_grid_transform(fitted) == Affine.from_gdal(
        *hb.get_global_geotransform_from_resolution(0.05))


def test_a_cropped_template_keeps_its_own_extent_on_the_global_lattice():
    from affine import Affine
    # Offsets of float32 size from whole cells snap to those cells, and the grid is not moved
    # to the global origin.
    assert pf.nominal_grid_transform(Affine(0.05, 0, -0.000006, 0, -0.05, 0.1)) == Affine(
        0.05, 0, 0.0, 0, -0.05, 0.1)
    assert pf.nominal_grid_transform(Affine(0.0500000017, 0, 10.0000003, 0, -0.05, 45.0000002)) == \
        Affine(0.05, 0, 10.0, 0, -0.05, 45.0)
    # Already exact, already cropped: unchanged.
    exact = Affine(0.25, 0, 7.25, 0, -0.25, -12.5)
    assert pf.nominal_grid_transform(exact) == exact


@pytest.mark.parametrize('transform', [
    (0.05, 0.001, -180.0, 0, -0.05, 90.0),     # rotated
    (0.05, 0, -180.0, 0, 0.05, -90.0),         # south-up
    (0.0503, 0, -180.0, 0, -0.05, 90.0),       # a cell size that is not 1/n degree
    (0.05, 0, -180.0, 0, -0.05, 89.99),        # an origin between grid lines
    (0.05, 0, 10.02, 0, -0.05, 45.0),          # a cropped origin between grid lines
])
def test_a_grid_further_off_than_float32_rounding_is_refused(transform):
    from affine import Affine
    with pytest.raises(ValueError):
        pf.nominal_grid_transform(Affine(*transform))


def test_the_weighting_refuses_a_second_grid_rather_than_resampling_onto_it(tmp_path):
    import rasterio
    from affine import Affine
    stored = Affine(0.05, 0, -0.000006, 0, -0.05, 0.1)
    _write_raster(tmp_path / 'value.tif', np.ones((2, 2)), stored, nodata=-9999.0)
    _write_raster(tmp_path / 'same.tif', np.ones((2, 2)), stored)
    _write_raster(tmp_path / 'snapped.tif', np.ones((2, 2)), Affine(0.05, 0, 0.0, 0, -0.05, 0.1))
    _write_raster(tmp_path / 'mercator.tif', np.ones((2, 2)), stored, crs='EPSG:3857')
    with rasterio.open(tmp_path / 'value.tif') as value:
        with rasterio.open(tmp_path / 'same.tif') as same:
            pt._require_same_grid(value, same)
        for other in ('snapped.tif', 'mercator.tif'):
            with rasterio.open(tmp_path / other) as different, pytest.raises(ValueError):
                pt._require_same_grid(value, different)


# ---------------------------------------------------------------------------
# The pollination value raster: production x price x dependence, and its units.
# ---------------------------------------------------------------------------

def test_the_value_is_a_density_and_only_becomes_money_after_cell_area():
    # The production rasters hold tonnes per square kilometre, so the value they generate is USD
    # per square kilometre, and a raster like that summed straight over a country adds densities:
    # a plausible number about 26 times too small. A cell's own area is what makes it money.
    production_density = np.array([[10.0, 20.0]])           # tonnes/km2
    poll, crop = pf.crop_pollination_value_density(production_density, 100.0, 0.5)
    assert crop.tolist() == [[1000.0, 2000.0]]              # USD/km2
    assert poll.tolist() == [[500.0, 1000.0]]

    # Two cells of the same density but different size hold different amounts of money.
    area_km2 = np.array([[1.0, 4.0]])
    assert pf.value_density_to_per_cell(poll, area_km2).tolist() == [[500.0, 4000.0]]


def test_a_negative_production_is_missing_rather_than_a_negative_value():
    # The source rasters carry sentinels below zero for pixels outside a crop's extent. Left as
    # numbers they would subtract value from the country total.
    poll, _ = pf.crop_pollination_value_density(np.array([[-9999.0, 5.0]]), 10.0, 1.0)
    assert np.isnan(poll[0, 0]) and poll[0, 1] == 50.0


def test_a_cell_with_no_crop_value_has_no_pollination_share_rather_than_zero():
    # Zero would say pollinators contribute nothing to this farmland; missing says there is no
    # farmland. Only the second is true of open ocean, and a mean must not average the first in.
    share = pf.local_pollination_share(np.array([[5.0, 0.0]]), np.array([[20.0, 0.0]]))
    assert share[0, 0] == pytest.approx(0.25)
    assert np.isnan(share[0, 1])


def test_coffee_dependence_follows_what_each_country_actually_grows():
    # FAO files arabica and robusta under one item code with different dependence ratios, 0.25
    # and 0.65. Collapsing on the item code keeps whichever row comes first, which values every
    # coffee country as pure arabica. Colombia really is pure arabica; Vietnam is nearly all
    # robusta, so one global ratio is wrong in opposite directions for two big producers.
    # The real file gives each country five seasons and a multi-year summary. Brazil's seasons
    # bracket its summary, so a reader that took a row by position rather than by name would
    # still return something plausible -- which is why the season rows are here.
    splits = pd.DataFrame({
        'area_code_m49': [170, 704, 76, 76, 76],
        'year': [pf.COFFEE_SPLIT_SUMMARY_YEAR, pf.COFFEE_SPLIT_SUMMARY_YEAR,
                 '2021_22', pf.COFFEE_SPLIT_SUMMARY_YEAR, 'Dec_2025_26'],
        'prop_arabica': [1.0, 0.034832, 0.626506, 0.644762, 0.603175],
        'prop_robusta': [0.0, 0.965168, 0.373494, 0.355238, 0.396825]})
    by_country = pf.coffee_dependence_by_country(splits)
    assert by_country[170] == pytest.approx(0.25)                      # Colombia, all arabica
    assert by_country[704] == pytest.approx(0.6361, abs=1e-3)          # Vietnam, mostly robusta
    # The summary row, not the last row: Brazil's final season would give 0.4087.
    assert by_country[76] == pytest.approx(0.3921, abs=1e-3)

    # And the ratio reaches the pixels through the country raster, with a country we have no
    # split for falling back rather than dropping out.
    country_ids = np.array([[170, 704], [76, 999]])
    ratios = pf.dependence_raster_from_country_lookup(country_ids, by_country, default=0.25)
    assert ratios[0, 0] == pytest.approx(0.25)
    assert ratios[0, 1] == pytest.approx(0.6361, abs=1e-3)
    assert ratios[1, 1] == pytest.approx(0.25)                          # unknown -> the default


def test_the_world_average_row_is_the_fallback_and_not_a_country():
    """The file carries a world-average row under area_code_m49 9999 for the 58 of 78 coffee
    countries it does not name. It is what those countries are valued at, and it is not itself a
    country, so it must not appear in the per-country lookup."""
    splits = pd.DataFrame({
        'area_code_m49': [170, pf.COFFEE_SPLIT_FALLBACK_CODE],
        'year': [pf.COFFEE_SPLIT_SUMMARY_YEAR, pf.COFFEE_SPLIT_SUMMARY_YEAR],
        'prop_arabica': [1.0, 0.603917],
        'prop_robusta': [0.0, 0.396083]})

    by_country = pf.coffee_dependence_by_country(splits)
    assert set(by_country) == {170}                       # 9999 is not a country
    # 0.603917 x 0.25 + 0.396083 x 0.65, the source pipeline's own fallback rather than arabica.
    assert pf.coffee_dependence_fallback(splits) == pytest.approx(0.4084, abs=1e-4)
    assert pf.coffee_dependence_fallback(splits) != pytest.approx(0.25, abs=1e-3)


def test_the_coffee_blend_refuses_a_file_with_no_multi_year_summary():
    # Selecting by position returns the summary only because the exporter writes it last. If a
    # re-export drops it, averaging the seasons here would report a different window than the
    # author does, and the two pipelines would part by a fraction of a percent with nothing said.
    seasons = pd.DataFrame({'area_code_m49': [76, 76],
                            'year': ['2021_22', 'Dec_2025_26'],
                            'prop_arabica': [0.626506, 0.603175],
                            'prop_robusta': [0.373494, 0.396825]})
    with pytest.raises(NameError, match='coffee split file'):
        pf.coffee_dependence_by_country(seasons)



def test_the_cpi_table_matches_the_series_the_deflator_used_to_hold():
    """The staged CPI table carries the series every deflator here reads, value for value."""
    import csv
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(utilities.service_data_dir(_base_data_project(), 'pollination'), 'us_cpi_u_annual.csv')
    with open(path, encoding='utf-8-sig') as f:
        cpi = {int(r['year']): float(r['cpi_u']) for r in csv.DictReader(f)}
    assert cpi[2019] == 255.7 and cpi[2020] == 258.8 and cpi[1993] == 144.5
    assert sorted(cpi) == list(range(1993, 2026))


def test_the_upstream_package_is_never_imported():
    """The science lives in this repo, so a run must not depend on an outside package being there.

    An import that works on one machine and not another is the failure this prevents: the tests
    pass wherever the package happens to be installed and the run dies wherever it is not.
    """
    import os
    import re
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ('pollination_tasks.py', 'pollination_functions.py'):
        text = open(os.path.join(here, name), encoding='utf-8').read()
        for line in text.split('\n'):
            assert not re.match(r'\s*(import crop_benefits|from crop_benefits)', line), \
                '%s imports crop_benefits: %s' % (name, line.strip())

def test_source_provenance_records_the_file_it_actually_read(tmp_path):
    """A stale staged raster is silent: same name, same shape, a slightly different number.

    Recording the hash is what makes "which copy produced this total?" answerable after the fact,
    rather than a question nobody can settle once the file has been overwritten.
    """
    raster = tmp_path / 'poll_value_global_2019usd.tif'
    raster.write_bytes(b'not really a raster, but bytes are bytes')
    out = pt.write_source_provenance(str(raster), str(tmp_path / 'prov' / 'provenance.csv'))
    row = pd.read_csv(out, encoding='utf-8-sig').iloc[0]
    assert row['source_raster'] == 'poll_value_global_2019usd.tif'
    assert row['bytes'] == raster.stat().st_size
    assert len(row['sha256']) == 64

    raster.write_bytes(b'not really a raster, but bytes are different now')
    changed = pd.read_csv(pt.write_source_provenance(str(raster), str(tmp_path / 'prov' / 'p2.csv')),
                          encoding='utf-8-sig').iloc[0]
    assert changed['sha256'] != row['sha256'], 'a changed file must change the recorded hash'


def test_the_latest_vintage_wins_when_the_base_year_is_unpublished(tmp_path):
    """His year files are separate model vintages, not one raster restated in other dollars.

    His 2024 file deflates to $386.76bn at 2019 prices and his 2023 file to $398.74bn, three
    percent apart, which no price index produces. Choosing by proximity would take
    the older model whenever the base year sits below the newest release.
    """
    class FakeProject:
        pollination_author_value_dir = str(tmp_path)
    for year in (2023, 2024):
        (tmp_path / ('poll_value_global_%dusd.tif' % year)).write_bytes(b'x')
    path, year = pf.find_source_value_raster(FakeProject(), 2019)
    assert year == 2024, 'nearest-year selection would have taken 2023'

    (tmp_path / 'poll_value_global_2019usd.tif').write_bytes(b'x')
    path, year = pf.find_source_value_raster(FakeProject(), 2019)
    assert year == 2019, 'the exact base year must win when it exists'


# =================================================================================================
# Condition 12: the two reproduction claims, checked rather than asserted.
#
# These pin what the pipeline can actually re-derive (the dependence panel against the author's
# table, and the rebuilt raster against his), and they run against every pollination project on
# the machine so a stale directory cannot stand in for a fresh one.
# =================================================================================================

def _pollination_runs():
    """Every pollination project directory on this machine, cold starts included."""
    import glob
    pattern = os.path.join(os.path.expanduser('~'), 'Files', 'global_invest', 'projects',
                           'gep_pollination*', 'intermediate')
    return sorted(p for p in glob.glob(pattern) if os.path.isdir(p))


def test_our_dependence_panel_reproduces_the_authors_published_table():
    """Our `pollination_1993_2024.parquet` against the author's csv, column by column.

    Reproducing it proves the port carries his FAO-item crosswalk and his blend weights
    faithfully, and proves nothing about whether those are right. Pinning it is what stops the
    agreement quietly lapsing the next time either side regenerates.
    """
    base = os.path.join(os.path.expanduser('~'), 'Files', 'base_data', 'global_invest', 'pollination')
    ours_path = os.path.join(base, 'fao', 'pollination', 'pollination_1993_2024.parquet')
    his_path = os.path.join(base, 'pollination_1993_2024.csv')
    if not (os.path.exists(ours_path) and os.path.exists(his_path)):
        pytest.skip('the dependence panel or the author table is not on this machine')

    ours = pd.read_parquet(ours_path)
    his = pd.read_csv(his_path)
    assert len(ours) == len(his) == 287285

    for frame in (ours, his):
        frame['area_code_m49'] = frame['area_code_m49'].astype('int64')
    shared = [c for c in his.columns if c in ours.columns]
    assert len(shared) == len(his.columns), 'the author table has a column ours does not carry'

    # The key repeats on 2,477 rows in both files, so ordering on the key alone is not stable
    # enough to compare row against row. Sorting on every shared column is.
    o = ours[shared].sort_values(shared).reset_index(drop=True)
    h = his[shared].sort_values(shared).reset_index(drop=True)

    for column in shared:
        if pd.api.types.is_numeric_dtype(h[column]):
            a, b = o[column].astype(float), h[column].astype(float)
            both = a.notna() & b.notna()
            relative = ((a[both] - b[both]).abs() / b[both].abs().replace(0, np.nan)).max()
            assert pd.isna(relative) or relative < 1e-12, column
            assert (a.isna() == b.isna()).all(), column
        else:
            # A missing label reads as None out of parquet and as nan out of csv, which is the file
            # format and not a difference; every other text value must match exactly.
            a = o[column].where(o[column].notna(), None).astype(str).replace('None', '')
            b = h[column].where(h[column].notna(), None).astype(str).replace('None', '')
            assert (a == b).all(), column


def test_our_rebuilt_raster_stays_close_to_the_authors_staged_one():
    """The independence check, pinned. Ours is built from FAO, CropGrids and Monfreda rather than
    read from his file, so this is two pipelines meeting rather than one reproducing the other, and
    the tolerance is set where a real divergence would show without the ordinary difference between
    two builds tripping it.

    The staged author raster is regenerated from his own code (his repo run here with the
    corrected crosswalk, the staged FAO bulks and the staged IMF FX), so the comparison also
    measures a DESIGNED divergence: the account prices on the 2019-only window with its
    geographic fallback, while his raster keeps the five-year window. A correlation collapse
    below the pinned floor is a loud signal, not noise."""
    checked = 0
    for run in _pollination_runs():
        path = os.path.join(run, 'pollination_value_independence_check',
                            'value_raster_independence.csv')
        if not os.path.exists(path):
            continue
        row = pd.read_csv(path).iloc[0]
        assert abs(row['pct_difference']) < 10.0, (run, row['pct_difference'])
        assert row['correlation_where_both'] > 0.97, (run, row['correlation_where_both'])
        assert row['cells_in_both'] > 2_000_000, (run, row['cells_in_both'])
        # Ours under the account's price rule; his under the geographic hierarchy.
        assert 3.3e11 < row['ours_independent_usd'] < 4.2e11, run
        assert 3.5e11 < row['author_raster_usd'] < 4.2e11, run
        checked += 1
    if not checked:
        pytest.skip('no pollination run with an independence check on this machine')


def test_dynamic_shock_rows_v3_is_the_scenario_against_its_own_paired_base():
    """v3 is a trajectory, not a distance from a contemporaneous baseline.

    Pinned because the two are easy to conflate and they disagree in magnitude and in sign. The
    seam emits the v3 the paper reports, so it needs no separate script rebuilding it from
    rasters.
    """
    fixedbase = _zone_frame({2030: {ZONE_A: -8.0}, 2040: {ZONE_A: -12.0}})
    paired_base = {2030: pd.Series({ZONE_A: 200.0}), 2040: pd.Series({ZONE_A: 400.0})}
    paired_scen = {2030: pd.Series({ZONE_A: 180.0}), 2040: pd.Series({ZONE_A: 300.0})}

    rows = pf.dynamic_shock_rows(fixedbase, fixedbase, None, 'net_zero', ('V_F',), base_year=2020,
                                 paired_base_by_year=paired_base, paired_scen_by_year=paired_scen)
    by_year = pd.DataFrame(rows).set_index('year')

    # 180/200 - 1 = -10%;  300/400 - 1 = -25%.
    assert np.isclose(by_year.loc[2030, 'shock_pct_v3'], -10.0)
    assert np.isclose(by_year.loc[2040, 'shock_pct_v3'], -25.0)
    # Pinned at zero at the base year, like every other measure here.
    assert by_year.loc[2020, 'shock_pct_v3'] == 0.0
    # 2025 is not an anchor; PCHIP does not put it on the straight line.
    lo, hi = by_year.loc[2023, 'shock_pct_v3'], by_year.loc[2030, 'shock_pct_v3']
    assert min(lo, hi) - 1e-9 <= by_year.loc[2025, 'shock_pct_v3'] <= max(lo, hi) + 1e-9
    assert not np.isclose(by_year.loc[2025, 'shock_pct_v3'], -5.0), 'reverted to linear'
    lo, hi = by_year.loc[2030, 'shock_pct_v3'], by_year.loc[2040, 'shock_pct_v3']
    assert min(lo, hi) - 1e-9 <= by_year.loc[2035, 'shock_pct_v3'] <= max(lo, hi) + 1e-9
    # The ratio is taken anchor by anchor. Interpolating the two LEVELS to 2035 and dividing there
    # would give 240/300 - 1 = -20%, which is not the same number.
    assert not np.isclose(by_year.loc[2035, 'shock_pct_v3'], -20.0)


def test_dynamic_shock_rows_v3_is_missing_rather_than_infinite_on_a_zero_paired_base():
    fixedbase = _zone_frame({2030: {ZONE_A: -8.0}})
    rows = pf.dynamic_shock_rows(fixedbase, fixedbase, None, 'net_zero', ('V_F',), base_year=2029,
                                 paired_base_by_year={2030: pd.Series({ZONE_A: 0.0})},
                                 paired_scen_by_year={2030: pd.Series({ZONE_A: 5.0})})
    by_year = pd.DataFrame(rows).set_index('year')
    # The anchor is unmeasurable, so it is missing rather than an infinite shock. The base year
    # stays zero: it is zero by construction, not a measurement the zero denominator could spoil.
    assert np.isnan(by_year.loc[2030, 'shock_pct_v3'])
    assert by_year.loc[2029, 'shock_pct_v3'] == 0.0


def test_dynamic_shock_rows_v3_is_absent_without_the_paired_halves():
    fixedbase = _zone_frame({2030: {ZONE_A: -8.0}})
    rows = pf.dynamic_shock_rows(fixedbase, fixedbase, None, 'net_zero', ('V_F',), base_year=2029)
    assert all(np.isnan(row['shock_pct_v3']) for row in rows)


# ---------------------------------------------------------------------------------------------
# Crop-to-GTAP-sector split
#
# Without the split, the per-crop loop sums every crop into one raster, which destroys the sector
# identity: everything downstream gives each sector the whole zone's value rather than its share --
# invisible while only shock_pct is read, and wrong as soon as a value becomes a denominator (a
# sector's pollination value can then exceed its output).
# ---------------------------------------------------------------------------------------------

def _correspondence_and_crosswalk():
    """The shipped crop->sector correspondence and the crop list the pollination task iterates."""
    import glob
    import os

    import pandas as pd

    corr = glob.glob(os.path.expanduser(
        '~/Files/gtap_invest/projects/elasticities_assessment/elasticities_assessment/'
        'input_template/cropgrids_c153_gtapv7_s8_correspondence.csv'))
    summary = glob.glob(os.path.expanduser(
        '~/Files/gep_repos/crop_benefits_outputs/rasters_*/pollination/value_*/'
        'poll_value_summary_*usd.csv'))
    if not corr or not summary:
        import pytest
        pytest.skip('correspondence or value summary not staged on this machine')
    return (pd.read_csv(corr[0], encoding='utf-8-sig'),
            pd.read_csv(sorted(summary)[-1], encoding='utf-8-sig'))


def test_every_valued_crop_maps_to_a_gtap_sector():
    """A crop with no sector keeps its value in the total and loses it from every sector raster.

    That is a silent leak: the totals still reconcile, so nothing looks wrong, while the per-sector
    shares are all quietly too small.
    """
    corr, summary = _correspondence_and_crosswalk()
    mapped = set(corr['cropgrids_label'].astype(str))
    valued = set(summary['cropgrids_crop'].astype(str))
    missing = sorted(valued - mapped)
    assert not missing, (
        '%d valued crops have no GTAP sector, so their value would vanish from the per-sector '
        'rasters while remaining in the total: %s' % (len(missing), missing[:10]))


def test_sector_split_conserves_total_value():
    """Summing the sectors must reproduce the ungrouped total, to the cent.

    This is the invariant the split has to preserve: it REDISTRIBUTES value between sectors, it
    never creates or destroys any.
    """
    corr, summary = _correspondence_and_crosswalk()
    m = summary.merge(corr[['cropgrids_label', 'gtapv7_label']],
                      left_on='cropgrids_crop', right_on='cropgrids_label', how='left')
    poll_col = [c for c in summary.columns if c.startswith('total_poll_value')][0]
    ungrouped = float(summary[poll_col].sum())
    by_sector = float(m.groupby('gtapv7_label')[poll_col].sum().sum())
    assert abs(by_sector - ungrouped) < 1.0, (
        'sector split does not conserve value: %.2f grouped vs %.2f ungrouped'
        % (by_sector, ungrouped))


def test_v_f_and_osd_do_not_own_everything():
    """The two sectors NGFS writes must not be assumed to hold all pollination value.

    They hold ~92%; the rest sits in OCR, PFB and GRO and currently receives no shock at all. If
    this ever reads 100%, the correspondence has collapsed and the split is doing nothing.
    """
    corr, summary = _correspondence_and_crosswalk()
    m = summary.merge(corr[['cropgrids_label', 'gtapv7_label']],
                      left_on='cropgrids_crop', right_on='cropgrids_label', how='left')
    poll_col = [c for c in summary.columns if c.startswith('total_poll_value')][0]
    by_sector = m.groupby('gtapv7_label')[poll_col].sum()
    share_two = by_sector.reindex(['V_F', 'OSD']).fillna(0.0).sum() / by_sector.sum()
    assert 0.80 < share_two < 0.99, (
        'V_F+OSD hold %.1f%% of pollination value; outside the expected band, so either the '
        'correspondence changed or the split collapsed' % (100 * share_two))


def test_dynamic_shock_rows_accepts_per_sector_crop_levels_without_changing_shock_pct():
    """The output-denominated shock's crop levels are delivered, and nothing else moves.

    The point of the test is the second half. Handing a new denominator to the row writer must not
    change shock_pct, which is what the solver reads; if it did, the published afeall results would
    shift the moment the crop rasters exist. So: same shock_pct with and without the crop levels,
    and value_usd_base takes the SECTOR's pollination level rather than the zone total.
    """
    import pandas as pd

    from global_invest.pollination import pollination_functions as pf

    zone = ('AEZ3', 'CHL')
    fixed = pd.DataFrame({2030: [10.0], 2050: [20.0]}, index=pd.MultiIndex.from_tuples([zone]))
    contemp = fixed.copy()
    zone_total = pd.Series({zone: 2.0e9})                 # the whole zone, all crops
    poll_by_sector = {'V_F': pd.Series({zone: 1.7e9}), 'OSD': pd.Series({zone: 0.3e9})}
    crop_by_sector = {'V_F': pd.Series({zone: 9.0e9}), 'OSD': pd.Series({zone: 0.21e9})}

    without = pf.dynamic_shock_rows(fixed, contemp, zone_total, 'net_zero', ['V_F', 'OSD'], 2023)
    with_levels = pf.dynamic_shock_rows(fixed, contemp, zone_total, 'net_zero', ['V_F', 'OSD'], 2023,
                                        level_usd_by_sector=poll_by_sector,
                                        crop_usd_by_sector=crop_by_sector)

    a = pd.DataFrame(without).sort_values(['ACTS', 'year']).reset_index(drop=True)
    b = pd.DataFrame(with_levels).sort_values(['ACTS', 'year']).reset_index(drop=True)
    assert len(a) == len(b) and len(a) > 0
    # What the solver reads is untouched.
    assert (a['shock_pct'].values == b['shock_pct'].values).all()
    # The value column carries the sector's own share, not the zone total copied twice.
    vf = b[b['ACTS'] == 'V_F']['value_usd_base'].iloc[0]
    osd = b[b['ACTS'] == 'OSD']['value_usd_base'].iloc[0]
    assert vf == 1.7e9 and osd == 0.3e9, (vf, osd)
    assert abs((vf + osd) - 2.0e9) < 1.0, 'sector shares must reproduce the zone total, not multiply it'
    # And without the levels: the zone total on every sector row.
    assert (a['value_usd_base'] == 2.0e9).all()


def test_shock_pct_output_is_the_v3_dollars_over_crop_value():
    """The output-denominated column is the PAPER'S numerator over the crop-value denominator.

    The export selects shock_pct_v3 for pollination: the scenario's own paired trajectory from the
    base year, 100 (S_t - S_2023) / S_2023. shock_pct_output must be that same dollar change,
    d_usd = v3/100 x the sector's base-year pollination value, over the sector's crop value --
    NOT the contemporaneous measure relabelled. Expected values are hand-computed from the levels
    below, independently of the function. NaN, never 0, where the crop level or the v3 series is
    absent.
    """
    import math

    import pandas as pd

    from global_invest.pollination import pollination_functions as pf

    zone = ('AEZ3', 'CHL')
    idx = pd.MultiIndex.from_tuples([zone])
    # contemporaneous measure deliberately DIFFERENT from v3, so a relabelling would be caught
    fixed = pd.DataFrame({2030: [10.0], 2050: [20.0]}, index=idx)
    contemp = pd.DataFrame({2030: [3.0], 2050: [7.0]}, index=idx)
    # paired levels: base 2.0e9 (S_2023), scenario 2.2e9 at 2030 and 2.5e9 at 2050
    paired_base = {2030: pd.Series({zone: 2.0e9}), 2050: pd.Series({zone: 2.0e9})}
    paired_scen = {2030: pd.Series({zone: 2.2e9}), 2050: pd.Series({zone: 2.5e9})}
    poll = {'V_F': pd.Series({zone: 1.7e9}), 'OSD': pd.Series({zone: 0.3e9})}
    crop = {'V_F': pd.Series({zone: 9.0e9})}          # OSD deliberately has no crop level

    rows = pd.DataFrame(pf.dynamic_shock_rows(
        fixed, contemp, pd.Series({zone: 2.0e9}), 'net_zero', ['V_F', 'OSD'], 2023,
        paired_base_by_year=paired_base, paired_scen_by_year=paired_scen,
        level_usd_by_sector=poll, crop_usd_by_sector=crop))
    vf = rows[rows['ACTS'] == 'V_F'].set_index('year')

    # v3 at 2050: 100 (2.5/2.0 - 1) = 25 %; the V_F sector holds 1.7e9 of the zone's 2.0e9, so its
    # dollar change is 0.25 x 1.7e9 = 0.425e9; over 9.0e9 crop value = 4.7222 %
    assert abs(vf.loc[2050, 'shock_pct_v3'] - 25.0) < 1e-9
    assert abs(vf.loc[2050, 'shock_pct_output'] - 100.0 * (0.25 * 1.7e9) / 9.0e9) < 1e-9
    # at 2030: v3 = 10 %, output share = 0.10 x 1.7 / 9.0 = 1.8889 %
    assert abs(vf.loc[2030, 'shock_pct_output'] - 100.0 * (0.10 * 1.7e9) / 9.0e9) < 1e-9
    # between anchors the v3 series is interpolated and the output share follows it.
    # 2040 lies between this fixture's anchors, so the PCHIP value is not the linear one. The
    # quantity under test -- v3 dollars over crop value -- is unchanged; only the path between
    # anchors is.
    lo, hi = vf.loc[2030, 'shock_pct_output'], vf.loc[2050, 'shock_pct_output']
    assert min(lo, hi) - 1e-9 <= vf.loc[2040, 'shock_pct_output'] <= max(lo, hi) + 1e-9
    # the ratio output/v3 IS the pollination share of crop value, and it is NOT the contemp ratio
    assert abs(vf.loc[2050, 'shock_pct_output'] / vf.loc[2050, 'shock_pct_v3'] - 1.7e9 / 9.0e9) < 1e-12
    assert abs(vf.loc[2050, 'shock_pct_output'] - vf.loc[2050, 'shock_pct'] * 1.7e9 / 9.0e9) > 1e-3
    # base year: no shock
    assert vf.loc[2023, 'shock_pct_output'] == 0.0

    # OSD: no crop level -> NaN
    osd = rows[(rows['ACTS'] == 'OSD') & (rows['year'] == 2050)].iloc[0]
    assert math.isnan(osd['shock_pct_output'])

    # No paired levels -> no v3 -> output share NaN throughout, never zero
    plain = pd.DataFrame(pf.dynamic_shock_rows(
        fixed, contemp, pd.Series({zone: 2.0e9}), 'net_zero', ['V_F'], 2023,
        level_usd_by_sector=poll, crop_usd_by_sector=crop))
    assert plain['shock_pct_output'].isna().all()


import ast
import math
from pathlib import Path
from typing import Tuple
import unittest
import numpy as np
from scipy.ndimage import convolve

SOURCE = Path(__file__).with_name('pollination_tasks.py')
TREE = ast.parse(SOURCE.read_text())
NS = dict(np=np, math=math, Tuple=Tuple, convolve=convolve)
HELPERS = {'_compute_radii_pixels', '_exact_radii_pixels', '_make_elliptical_kernel'}
nodes = [n for n in TREE.body if
         isinstance(n, ast.FunctionDef) and n.name in HELPERS or
         isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
         t.id in {'_COS_LAT_FLOOR', '_RADIUS_METERS', '_WGS84_A_KM', '_WGS84_F',
                  '_MAX_RY', '_MAX_RX'} for t in n.targets)]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), NS)
worker = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == '_process_tile')
count = next(n for n in ast.walk(worker) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'counts' for t in n.targets))
EXPR = compile(ast.Expression(count.value), str(SOURCE), 'eval')

def production(mask, kernel):
    # The worker counts per band, slicing nat_mask by s0:s1; the whole mask is one band here.
    return eval(EXPR, dict(NS, nat_mask=mask, kernel=kernel, s0=0, s1=None))

class HabitatCounts(unittest.TestCase):
    def test_center_matches_exact_neighbor_sum(self):
        for latitude in (0, 30, 50, 55, 60, 70):
            with self.subTest(latitude=latitude):
                ry, rx = NS['_compute_radii_pixels'](latitude, 1/360, 1/360)
                exact = NS['_exact_radii_pixels'](latitude, 1/360, 1/360)
                kernel = NS['_make_elliptical_kernel'](ry, rx, exact=exact)
                mask = kernel.copy()
                mask[ry, rx] = 0  # focal pixel is cropland
                self.assertEqual(int(production(mask, kernel)[ry, rx]), int(mask.sum()))

    def test_exact_radius_trims_the_whole_pixel_ellipse(self):
        # The whole-pixel radii are a ceiling, so the exact 2 km ellipse can only be a subset
        # of the whole-pixel one, and at the equator it is strictly smaller (6.51 px vs 7).
        ry, rx = NS['_compute_radii_pixels'](0.0, 1/360, 1/360)
        exact = NS['_exact_radii_pixels'](0.0, 1/360, 1/360)
        trimmed = NS['_make_elliptical_kernel'](ry, rx, exact=exact)
        whole = NS['_make_elliptical_kernel'](ry, rx)
        self.assertTrue(np.all(whole >= trimmed))
        self.assertLess(int(trimmed.sum()), int(whole.sum()))
        self.assertLess(exact[0], ry)

    def test_habitat_removal_cannot_increase_sufficiency(self):
        kernel = NS['_make_elliptical_kernel'](7, 12)
        mask = kernel.copy()
        mask[7, 12] = 0
        previous = 1.0
        for row, col in np.argwhere(mask):
            mask[row, col] = 0
            count = int(production(mask, kernel)[7, 12])
            sufficiency = min(count / int(kernel.sum()) / .3, 1.)
            self.assertLessEqual(sufficiency, previous)
            previous = sufficiency
        self.assertEqual(previous, 0.)

if __name__ == '__main__':
    unittest.main()


import numpy as np
import pytest

from global_invest.pollination import pollination_functions as pd_mod
from global_invest.pollination.pollination_functions import retained_value_change


def test_hectare_components_partition_baseline_cropland():
    base = np.array([[2, 2, 3], [2, 3, 3]])
    future = np.array([[2, 3, 2], [2, 3, 3]])
    ha = np.full((2, 3), 10.0)
    got = pd_mod.component_hectares(base, future, ha)
    assert got['retained_ha'] == 20.0            # (0,0) and (1,0)
    assert got['lost_ha'] == 10.0                # (0,1) crop -> non-crop
    assert got['new_ha'] == 10.0                 # (0,2) non-crop -> crop
    assert got['retained_ha'] + got['lost_ha'] == got['base_cropland_ha']
    assert got['retained_ha'] + got['new_ha'] == got['future_cropland_ha']


def test_retained_term_matches_the_fed_measure():
    """The decomposition must not quietly redefine the estimand it is diagnosing."""
    value, base_ha, retained_ha = 1000.0, 100.0, 60.0
    suff_base, suff_future = 0.4, 0.7
    got = pd_mod.decompose(value, retained_ha, 40.0, 0.0, base_ha, suff_base, suff_future)
    fed = retained_value_change(value, retained_ha / base_ha, suff_base, suff_future)
    assert np.isclose(got['retained_change'], fed)


def test_identity_reconciles():
    got = pd_mod.decompose(500.0, 50.0, 30.0, 20.0, 80.0, 0.5, 0.6)
    summary = pd_mod.reconcile(got)
    assert np.isclose(summary['total_change'],
                      summary['retained_change'] + summary['new_provision'] - summary['lost_provision'])


def test_new_cropland_without_a_base_rate_is_reported_not_zeroed():
    """A cell with no baseline cropland cannot yield a value per hectare; the area must survive."""
    got = pd_mod.decompose(0.0, 0.0, 0.0, 25.0, 0.0, 0.3, 0.9)
    assert got['new_provision'] == 0.0
    assert got['unvalued_new_ha'] == 25.0
    summary = pd_mod.reconcile(got)
    assert summary['unvalued_new_ha'] == 25.0
    assert 'no defensible crop value' in summary['caveat']


def test_unvalued_assumption_declines_to_value_any_new_land():
    got = pd_mod.decompose(1000.0, 50.0, 0.0, 50.0, 50.0, 0.5, 0.5,
                           new_cropland_valuation='unvalued')
    assert got['new_provision'] == 0.0
    assert got['unvalued_new_ha'] == 50.0


def test_one_sided_sufficiency_coverage_contributes_nothing():
    """A component computed where only one year has sufficiency would difference against nothing."""
    got = pd_mod.decompose(1000.0, 50.0, 10.0, 10.0, 60.0, np.nan, 0.8)
    assert got['retained_change'] == 0.0
    assert got['lost_provision'] == 0.0
    assert got['new_provision'] == 0.0


def test_components_that_overrun_baseline_cropland_are_rejected():
    with pytest.raises(ValueError, match='do not partition'):
        pd_mod.decompose(100.0, 60.0, 60.0, 0.0, 100.0, 0.5, 0.5)


def test_unknown_valuation_assumption_is_rejected():
    with pytest.raises(ValueError, match='unknown new-cropland valuation'):
        pd_mod.decompose(100.0, 10.0, 0.0, 0.0, 10.0, 0.5, 0.5, new_cropland_valuation='zero')


def test_label_says_valued_support_when_some_new_cropland_is_unvalued():
    got = pd_mod.decompose(0.0, 0.0, 0.0, 25.0, 0.0, 0.3, 0.9)
    summary = pd_mod.reconcile(got)
    assert summary['quantity'] == 'provision change on valued support'


def test_label_says_total_only_when_everything_is_valued_and_covered():
    got = pd_mod.decompose(1000.0, 60.0, 40.0, 0.0, 100.0, 0.4, 0.7)
    summary = pd_mod.reconcile(got)
    assert summary['quantity'] == 'total provision change'
    assert summary['unvalued_new_ha'] == 0.0
    assert summary['excluded_for_missing_sufficiency_ha'] == 0.0


def test_area_excluded_for_missing_sufficiency_is_reported_not_dropped():
    got = pd_mod.decompose(1000.0, 50.0, 10.0, 10.0, 60.0, np.nan, 0.8)
    summary = pd_mod.reconcile(got)
    assert summary['excluded_for_missing_sufficiency_ha'] == 70.0
    assert summary['quantity'] == 'provision change on valued support'


from global_invest.pollination.pollination_functions import (_overlap, output_share_pct,
                                                             annual_retained_rows)


def test_coincident_edges_do_not_create_neighbour_slivers():
    weights = _overlap(2, 0, 1000. + 1e-12, 1., 0., 1., 1003).toarray()
    assert np.count_nonzero(weights) == 2
    np.testing.assert_array_equal(weights[1000:1002], np.eye(2))


def test_real_small_overlap_is_preserved():
    weights = _overlap(1, 0, 1000. + 1e-5, 1., 0., 1., 1002).toarray()
    assert np.count_nonzero(weights) == 2
    assert weights[1001, 0] == pytest.approx(1e-5, abs=1e-10)
    assert weights.sum() == pytest.approx(1.)


def test_half_retained_does_not_receive_whole_cell_value():
    # $100 crop value; $20 dependent; half retained; sufficiency .4 -> .6.
    delta = retained_value_change(20, .5, .4, .6)
    assert float(delta) == pytest.approx(2.)
    assert output_share_pct(delta, 100) == pytest.approx(2.)


def test_turnover_alone_is_not_habitat_productivity_change():
    np.testing.assert_allclose(retained_value_change([20, 20], [0, .5], [.4, .4], [.4, .4]), 0)


def test_habitat_loss_never_produces_gain():
    delta = retained_value_change([20, 10], [.5, 1], [.8, .5], [.2, .4])
    assert np.all(delta < 0)


def test_sector_denominator_includes_all_baseline_crop_value():
    delta = retained_value_change([20, 10], [.5, 0], [.4, np.nan], [.6, np.nan])
    assert output_share_pct(delta, [100, 100]) == pytest.approx(1.)


def test_inconsistent_coverage_and_missing_aggregation_refused():
    with pytest.raises(ValueError):
        retained_value_change(20, .5, .4, np.nan)
    with pytest.raises(ValueError):
        output_share_pct([1, np.nan], [100, 100])


def test_common_value_unit_conversion_cancels():
    assert output_share_pct([2, -1], [100, 50]) == pytest.approx(
        output_share_pct([2000, -1000], [100000, 50000]))


def test_annual_dollars_allow_zero_baseline_and_preserve_anchor_change():
    import pandas as pd
    index = pd.MultiIndex.from_tuples([('AEZ1', 'USA')])
    zero = pd.Series([0.], index=index)
    future = pd.Series([2.], index=index)
    rows = annual_retained_rows({2030: zero}, {2030: future}, zero, 'current_policies', 2023, ['V_F'])
    assert rows.iloc[0].delta_pollination_usd == 0.
    assert rows.iloc[-1].delta_pollination_usd == 2.
    np.testing.assert_allclose(rows.delta_pollination_usd, np.linspace(0, 2, 8))


def test_retained_fraction_raster_runs_the_folded_path_and_caches(tmp_path):
    """End-to-end through pollination_tasks.retained_fraction_raster on a 4x4 world.

    Quadrant design makes the expected fractions exact under any latitude weighting:
    full retention (1.0), full loss (0.0), no cropland (0.0), and column-wise half
    retention (0.5, the area weights cancel). The second call must hit the signature
    sidecar instead of recomputing. It runs the tasks module's own code path, so the test
    fails on any unbound name there.
    """
    import rasterio
    base = np.ones((4, 4));  base[:2, :] = 2.;  base[2:, 2:] = 2.
    future = np.ones((4, 4)); future[:2, :2] = 2.; future[2:, 2] = 2.
    _write_tif(tmp_path / 'base.tif', base)
    _write_tif(tmp_path / 'future.tif', future)
    _write_tif(tmp_path / 'template.tif', np.zeros((2, 2)))
    out = tmp_path / 'fraction.tif'
    got = pt.retained_fraction_raster(str(tmp_path / 'base.tif'), str(tmp_path / 'future.tif'),
                                      str(tmp_path / 'template.tif'), str(out))
    with rasterio.open(got) as ds:
        np.testing.assert_allclose(ds.read(1), [[1., 0.], [0., .5]], atol=1e-6)
    assert os.path.exists(str(out) + '.json')
    again = pt.retained_fraction_raster(str(tmp_path / 'base.tif'), str(tmp_path / 'future.tif'),
                                        str(tmp_path / 'template.tif'), str(out))
    assert again == str(out)


def test_totals_pair_matches_the_account_anchor():
    """The anchor pair: $392.5915bn unweighted and $164.3089bn weighted — the account under the
    adopted price rule (2019-only windows; the country's price, then the same crop's subregion,
    region and world median; no cross-year step), with the 5 km sufficiency averaged on the
    nominal 0.05-degree grid. The unweighted total reproduces the author's independently reported
    $392.59bn to his rounding. His weighted $164.96bn is the offset-grid pair below: his code
    averages onto the float32-offset grid, which adds 0.39%.
    Runs built under earlier rules carry earlier pairs and are recognised rather than failed:
    $392.5915/$164.9579 (that offset grid), $347.3652/$146.1518 (the superseded own-country rule),
    $379.5011/$160.2358 (the author's revision with five-year windows), and the `october` and
    `august` pairs. At least one run must carry the CURRENT pair or the pin would pass on history
    alone.
    Rebuilds must force the import (PYTHONPATH to the intended tree) and assert
    global_invest.__file__: the environment's editable install points at the MAIN checkout, so a
    run launched as a script (rather than inline from a worktree cwd) silently imports whatever
    branch that checkout is on and produces another code tree's numbers."""
    import glob as _glob
    checked = 0
    current_runs = 0
    for run in _pollination_runs():
        path = os.path.join(run, 'pollination_sufficiency_weighted',
                            'pollination_totals_with_and_without_sufficiency.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path).set_index('sufficiency_applied')['total_usd_2019_bn']
        unweighted, weighted = float(df[False]), float(df[True])
        current = abs(unweighted - 392.5915) < 0.05 and abs(weighted - 164.3089) < 0.05
        offset_grid = abs(unweighted - 392.5915) < 0.05 and abs(weighted - 164.9579) < 0.05
        own_country = abs(unweighted - 347.3652) < 0.05 and abs(weighted - 146.1518) < 0.05
        author_rev = abs(unweighted - 379.5011) < 0.05 and abs(weighted - 160.2358) < 0.05
        october = abs(unweighted - 379.5011) < 0.05 and abs(weighted - 157.4983) < 0.05
        august = abs(unweighted - 388.1426) < 0.05 and abs(weighted - 163.4501) < 0.2
        assert current or offset_grid or own_country or author_rev or october or august, (run, unweighted, weighted)
        checked += 1
        current_runs += int(current)
    assert checked >= 1, 'no run on this machine carries the totals pair'
    assert current_runs >= 1, 'no run carries the current anchor (392.5915 / 164.3089)'


def test_yield_change_keeps_late_only_country_yields_for_the_target_method():
    rows = []
    for code, year, value in [('001', 2000, 10.0), ('001', 2019, 20.0),
                              ('002', 2019, 30.0), ('003', 2000, 5.0)]:
        rows.append({'element': 'Yield', 'area_code_m49': code, 'item_code_fao': 1,
                     'year': year, 'value': value, 'item_fao': 'Crop'})
    crosswalk = pd.DataFrame({'area_code_m49': ['001', '002', '003'],
                              'iso3': ['AAA', 'BBB', 'CCC'],
                              'region_fao': ['R'] * 3, 'subregion_fao': ['S'] * 3})
    ratios = pf.yield_change_ratios(pd.DataFrame(rows), crosswalk, [2000], [2019])
    country = ratios[ratios['agg_level'] == 'country'].set_index('area_code_m49')
    # a complete pair keeps its ratio and its late yield
    assert country.loc['001', 'yield_ratio'] == pytest.approx(2.0)
    assert country.loc['001', 'late_yield'] == pytest.approx(20.0)
    # a late-only pair keeps its own 2019 yield, with no invented ratio
    assert country.loc['002', 'late_yield'] == pytest.approx(30.0)
    assert np.isnan(country.loc['002', 'yield_ratio'])
    # an early-only pair still has no late yield
    assert np.isnan(country.loc['003', 'late_yield'])
    late_lookup, provenance = pf.build_hierarchical_lookup(ratios, crosswalk, 'late_yield')
    assert late_lookup[2] == pytest.approx(30.0) and provenance[2] == pf.PROVENANCE_COUNTRY
    # the early-only country falls back to the subregion median of (20, 30)
    assert late_lookup[3] == pytest.approx(25.0) and provenance[3] == pf.PROVENANCE_SUBREGION
    # the ratio method: the NaN-ratio country falls back exactly as a country with no row does
    ratio_lookup, ratio_provenance = pf.build_hierarchical_lookup(ratios, crosswalk, 'yield_ratio')
    assert ratio_lookup[2] == pytest.approx(2.0) and ratio_provenance[2] == pf.PROVENANCE_SUBREGION


def test_annual_values_fall_back_country_subregion_region_world_in_that_order():
    """The adopted rule (the author's crop_benefits hierarchy): the country's own price for the
    year, else the same crop's subregion median, else region, else world — and never another
    YEAR's price. Value is production tonnes times USD per tonne."""
    prod = pd.DataFrame({
        'element': ['Production'] * 4,
        'year': [2019] * 4,
        'area_code_m49': ['004', '008', '012', '016'],
        'subregion_fao': ['S1', 'S1', 'S2', 'S3'],
        'region_fao': ['R1', 'R1', 'R1', 'R2'],
        'item_code_fao': [515] * 4,
        'value': [100.0, 50.0, 70.0, 10.0]})
    price_country = pd.DataFrame({
        'year': [2019, 2017], 'area_code_m49': ['004', '008'], 'item_code_fao': [515, 515],
        'price_usd_tonne': [10.0, 999.0]})
    price_subregion = pd.DataFrame({'year': [2019], 'subregion_fao': ['S1'], 'item_code_fao': [515],
                                    'subreg_price_usd_tonne': [20.0]})
    price_region = pd.DataFrame({'year': [2019], 'region_fao': ['R1'], 'item_code_fao': [515],
                                 'reg_price_usd_tonne': [30.0]})
    price_world = pd.DataFrame({'year': [2019], 'item_code_fao': [515],
                                'world_price_usd_tonne': [40.0]})
    merged = pt._merge_production_value(prod, price_country, price_subregion, price_region, price_world)
    keyed = merged.set_index('area_code_m49')
    assert keyed.at['004', 'price_source_agg_level'] == 'country'
    assert keyed.at['004', 'value_usd'] == pytest.approx(100.0 * 10.0)
    # 008 has only a 2017 own price: that year is NOT used; its subregion median prices it
    assert keyed.at['008', 'price_source_agg_level'] == 'subregion'
    assert keyed.at['008', 'price_usd_tonne'] == pytest.approx(20.0)
    assert keyed.at['012', 'price_source_agg_level'] == 'region'
    assert keyed.at['016', 'price_source_agg_level'] == 'world'
    assert keyed.at['016', 'value_usd'] == pytest.approx(10.0 * 40.0)
    assert 'price_donor_year' not in merged.columns


def test_price_raster_falls_back_to_the_world_median_for_an_unpriced_country():
    rows = pd.DataFrame({
        'agg_level': ['country', 'country', 'world'],
        'area_code_m49': ['002', '003', '999'],
        'region_fao': ['R', 'R', None], 'subregion_fao': ['S', 'S', None],
        'price_median_usd_tonne': [40.0, 60.0, 50.0]})
    crosswalk = pd.DataFrame({'area_code_m49': ['002', '003', '004'],
                              'region_fao': ['R'] * 3, 'subregion_fao': ['S'] * 3})
    ids = np.array([[2, 3], [4, 0]])
    price, source = pf.price_raster_by_country(rows, ids, crosswalk, 1.0)
    assert price[0, 0] == pytest.approx(40.0) and source[0, 0] == pf.PROVENANCE_COUNTRY
    assert price[0, 1] == pytest.approx(60.0)
    assert np.isfinite(price).all(), 'every pixel is priced under the hierarchy'
    assert price[1, 1] == pytest.approx(50.0) and source[1, 1] == pf.PROVENANCE_WORLD


def test_price_tables_follow_the_adopted_hierarchy_with_no_temporal_step():
    """The adopted rule, checked on the artifacts themselves: the median table carries all four
    levels, and every priced row in the values table names one of them. A temporal-donor row
    or a donor-year column is a regression to the superseded own-country rule."""
    median_path = os.path.join(_base_data_project().base_data_dir,
                               'global_invest', 'pollination', 'fao', 'median_prices',
                               'price_median_usd_tonne_2019_2019.parquet')
    if not os.path.exists(median_path):
        pytest.skip('no 2019-window median table on this machine')
    medians = pd.read_parquet(median_path)
    assert set(medians['agg_level']) == {'country', 'subregion', 'region', 'world'}, \
        sorted(set(medians['agg_level']))
    values_path = os.path.join(os.path.dirname(median_path), '..', 'fao_values_1993_2024.csv')
    values = pd.read_csv(values_path, low_memory=False)
    assert set(values['price_source_agg_level']) <= {'country', 'subregion', 'region', 'world',
                                                     'missing'}, sorted(set(values['price_source_agg_level']))
    assert 'price_donor_year' not in values.columns


def test_weighted_variant_is_bounded_by_the_unweighted_and_reconciles_to_the_country_table():
    """Habitat sufficiency is a 0..1 weight, so the weighted total can never exceed the
    unweighted one. The published country table must be the weighted raster's own zonal sum:
    conservation is pinned at 99.5 percent (measured 99.86; the remainder is boundary pixels
    outside every region polygon), so a wrong-raster export or a zonal regression trips it."""
    checked = 0
    for run in _pollination_runs():
        totals_path = os.path.join(run, 'pollination_sufficiency_weighted',
                                   'pollination_totals_with_and_without_sufficiency.csv')
        country_path = os.path.join(run, 'gep_calculation', 'gep_by_country_base_year.csv')
        if not (os.path.exists(totals_path) and os.path.exists(country_path)):
            continue
        totals = pd.read_csv(totals_path).set_index('sufficiency_applied')['total_usd']
        assert totals[True] <= totals[False], run
        country_sum = pd.read_csv(country_path)['pollination_gep'].sum()
        weighted_ratio = country_sum / totals[True]
        unweighted_ratio = country_sum / totals[False]
        # A run reconciles with exactly one of its own rasters; historical runs carried the
        # unweighted variant and are recognised, not failed.
        is_weighted = 0.995 <= weighted_ratio <= 1.001
        is_historical_unweighted = 0.995 <= unweighted_ratio <= 1.001
        assert is_weighted or is_historical_unweighted, (run, weighted_ratio, unweighted_ratio)
        checked += int(is_weighted)
    assert checked >= 1, 'no run publishes the weighted variant within conservation tolerance'


def test_weighting_preserves_missing_values_and_genuine_zeros():
    """Pins the missing-data rule: nodata and NaN crop values must stay missing through the
    weighting (converted to zero, they publish unmeasured countries as zeros in the country
    export), a valued cell with missing sufficiency stays missing (unknown habitat is not
    confirmed absence), and a genuine zero stays zero."""
    value = np.array([-9999.0, np.nan, 100.0, 0.0])
    sufficiency = np.array([0.5, 0.5, 0.5, 0.5])
    weighted, unweighted_total, weighted_total = pf.value_weighted_by_sufficiency(
        value, sufficiency, value_nodata=-9999.0)
    assert np.isnan(weighted[0]), 'nodata value must stay missing'
    assert np.isnan(weighted[1]), 'NaN value must stay missing'
    assert weighted[2] == pytest.approx(50.0)
    assert weighted[3] == 0.0, 'a genuine zero stays zero'
    assert unweighted_total == pytest.approx(100.0)
    assert weighted_total == pytest.approx(50.0)
    # missing sufficiency on a VALUED cell: missing, not zero
    weighted2, _, total2 = pf.value_weighted_by_sufficiency(
        np.array([100.0, 100.0]), np.array([np.nan, 0.25]))
    assert np.isnan(weighted2[0]) and weighted2[1] == pytest.approx(25.0)
    assert total2 == pytest.approx(25.0)


def test_read_band_missing_as_nan_normalizes_finite_sentinels(tmp_path):
    """The sufficiency read path: a FINITE declared nodata (say -9999) must come back as NaN, or
    the clip to [0, 1] turns it into zero sufficiency -- confirmed absence invented from a
    sentinel. A NaN-declared raster and an undeclared one must pass through unchanged."""
    import rasterio
    data = np.array([[0.5, -9999.0], [1.0, 0.25]], dtype='float32')

    def write(path, nodata):
        profile = {'driver': 'GTiff', 'width': 2, 'height': 2, 'count': 1, 'dtype': 'float32',
                   'crs': 'EPSG:4326', 'transform': rasterio.transform.from_bounds(0, 0, 2, 2, 2, 2),
                   'nodata': nodata}
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(data, 1)

    finite_path = str(tmp_path / 'finite_sentinel.tif')
    write(finite_path, -9999.0)
    with rasterio.open(finite_path) as src:
        out = pt.read_band_missing_as_nan(src)
    assert np.isnan(out[0, 1]), 'the finite sentinel must become NaN'
    assert out[0, 0] == pytest.approx(0.5) and out[1, 0] == pytest.approx(1.0)

    nan_path = str(tmp_path / 'nan_nodata.tif')
    write(nan_path, np.nan)
    with rasterio.open(nan_path) as src:
        out = pt.read_band_missing_as_nan(src)
    # NaN is already the missing encoding: the read is an identity, -9999.0 stays a (nonsense)
    # finite number because nothing declared it missing.
    assert out[0, 1] == pytest.approx(-9999.0)

    none_path = str(tmp_path / 'no_nodata.tif')
    write(none_path, None)
    with rasterio.open(none_path) as src:
        out = pt.read_band_missing_as_nan(src)
    assert out[0, 1] == pytest.approx(-9999.0)


def test_a_country_factor_within_tolerance_reads_as_near_equal_not_as_a_factor():
    """The near-equal test must be parenthesised as the author's is: `&` binds before `<=`, so an
    unparenthesised mask compares the whole chain with the tolerance. A country factor within 1%
    would then read as a full factor (and fall to the later currency check), and a row with no
    factor at all would be labelled near-equal."""
    nan = np.nan
    pp_wide = pd.DataFrame({
        'country': ['A', 'A', 'B', 'B', 'C', 'D'],
        'Item': ['x'] * 6,
        'year': [2000, 2001, 2000, 2001, 2001, 2001],
        'slc_tonne': [100.0, 200.0, 100.0, 200.0, 50.0, nan],
        'lcu_tonne': [100.5, nan, 150.0, nan, nan, nan],
        'price_index': [nan] * 6,
    })
    out = pf._reconstruct_slc_lcu(pp_wide, SimpleNamespace(tol_near_equal_lcu_slc=0.01))
    row = out.set_index(['country', 'year'])
    assert row.loc[('A', 2001), 'lcu_source'] == 'slc_filled_country_factor_near_equal'
    assert row.loc[('A', 2001), 'lcu_filled'] == pytest.approx(200.0)
    assert row.loc[('B', 2001), 'lcu_source'] == 'slc_filled_country_factor'
    assert row.loc[('B', 2001), 'lcu_filled'] == pytest.approx(300.0)
    assert row.loc[('C', 2001), 'lcu_source'] == 'slc_filled_assume_equal'
    assert row.loc[('D', 2001), 'lcu_source'] == 'missing'

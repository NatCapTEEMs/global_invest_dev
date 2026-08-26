"""Unit tests for pollination.

Two layers. The shock and valuation arithmetic is pinned as pure functions over hand-built series
whose expected values are written out in the assertions. Below that, one end-to-end test runs the
real calculation (a tiny USD value raster through utilities.summarize_raster_by_region into
gep_calculation) to pin the r250 contract: a split country is summed once, and the map gpkg
carries per-sub-region rows that are never summed.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from global_invest.pollination import pollination_functions as pf
from global_invest.pollination import pollination_tasks as pt

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
    labels = pf.zone_labels_from_boundary(boundary)
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
    assert list(out.columns) == pf.POLLINATION_ATTR_COLS + ['year', 'pollination_gep']
    by_country = out.set_index('iso3_r250_label')
    assert len(by_country) == 2
    assert by_country.loc['CHN', 'pollination_gep'] == 60.0     # 40 + 20, not 40 and 20
    assert by_country.loc['NLD', 'pollination_gep'] == 8.0
    assert by_country.loc['CHN', 'region_wb'] == 'EAP'


def test_expand_country_values_to_regions_repeats_the_national_value_per_sub_region():
    df_regions = _region_frame()
    out = pf.expand_country_values_to_regions(df_regions, pf.collapse_regions_to_countries(df_regions))
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
                        df_countries=pd.DataFrame({'placeholder': [1]}),   # trips the caller-wins guard
                        # our own raster now, in USD per cell rather than a density from elsewhere
                        pollination_value_raster_path=str(tmp_path / 'poll_value.tif'),
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
    # The sufficiency and value science used to come from an installed package, crop_benefits,
    # that was declared in no pyproject and configured through a gitignored local.yaml. On a
    # machine without it the whole module failed at import, which is how it reached a collaborator.
    # It is vendored now, so nothing here may reach for that package again.
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
    # Seven fields are everything the vendored steps ever read off the crop_benefits Config, and
    # three of them our own driver already overrode. They are plain arguments now, so a missing
    # config file cannot silently produce a run against the wrong paths.

    from global_invest.pollination import pollination_functions as pf

    settings = pf.SufficiencySettings(
        output_dir='/tmp/out', value_raster_dir='/tmp/base',
        country_raster_path='/tmp/base/poll_value_global_2023usd.tif')
    assert settings.tile_size == 2048 and settings.n_workers == 4
    assert settings.lulc_path is None and settings.pa_raster_300m_path is None
    # The compression profiles came off that Config too, and are now named constants.
    assert set(pf.COMPRESSION_PROFILES) == {'continuous', 'categorical', 'defaults'}


# ---------------------------------------------------------------------------
# The pollination value raster: production x price x dependence, and its units.
# ---------------------------------------------------------------------------

def test_the_value_is_a_density_and_only_becomes_money_after_cell_area():
    # This is the arithmetic behind the defect the account carried: the production rasters hold
    # tonnes per square kilometre, so the value they generate is USD per square kilometre, and a
    # raster like that summed straight over a country adds densities. It gives a plausible number
    # 26 times too small, which is why nothing caught it. A cell's own area is what makes it money.
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
    # and 0.65. Collapsing on the item code keeps whichever row came first, which valued every
    # coffee country as pure arabica. Colombia really is pure arabica; Vietnam is nearly all
    # robusta, so one global ratio is wrong in opposite directions for two big producers.
    splits = pd.DataFrame({'area_code_m49': [170, 704, 76],
                           'prop_arabica': [1.0, 0.034832, 0.626506],
                           'prop_robusta': [0.0, 0.965168, 0.373494]})
    by_country = pf.coffee_dependence_by_country(splits)
    assert by_country[170] == pytest.approx(0.25)                      # Colombia, all arabica
    assert by_country[704] == pytest.approx(0.6361, abs=1e-3)          # Vietnam, mostly robusta
    assert by_country[76] == pytest.approx(0.3994, abs=1e-3)           # Brazil, a real mix

    # And the ratio reaches the pixels through the country raster, with a country we have no
    # split for falling back rather than dropping out.
    country_ids = np.array([[170, 704], [76, 999]])
    ratios = pf.dependence_raster_from_country_lookup(country_ids, by_country, default=0.25)
    assert ratios[0, 0] == pytest.approx(0.25)
    assert ratios[0, 1] == pytest.approx(0.6361, abs=1e-3)
    assert ratios[1, 1] == pytest.approx(0.25)                          # unknown -> the default


def test_the_deflator_refuses_a_year_it_has_no_index_for():
    # Returning 1.0 for an unknown year would leave the value undeflated and looking correct.
    assert pf.usd_deflator(2020, 2020) == 1.0
    assert pf.usd_deflator(2020, 2019) == pytest.approx(0.98802, abs=1e-5)
    with pytest.raises(KeyError):
        pf.usd_deflator(2020, 1850)

"""Unit tests for NTFP: the valuation, the FAO area, the alternatives, the configuration rows and
the accessibility stage."""
import inspect
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from global_invest.ntfp import ntfp_functions
from global_invest.ntfp import ntfp_functions as nf


def test_the_published_rate_is_applied_to_the_named_area_in_the_base_year():
    # $1/ha over 200 hectares of FAO forest is $200. Only the base-year rate is used: AAA's 2020
    # row would have given a different value.
    area = pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB'], 'fao_forest_ha': [200.0, 50.0]})
    values = pd.DataFrame({'iso3_r250_label': ['AAA', 'AAA', 'BBB'],
                           'year': [2019, 2020, 2019],
                           'nwfp_value_per_ha': [1.0, 4.5, 2.0]})
    out = ntfp_functions.ntfp_gep_by_country(area, values, 2019, 'fao_forest_ha').set_index('iso3_r250_label')
    assert out['ntfp_gep'].to_dict() == {'AAA': 200.0, 'BBB': 100.0}
    assert out.loc['AAA', 'nwfp_value_per_ha'] == 1.0


def test_a_missing_area_or_rate_stays_missing_and_a_zero_stays_zero():
    area = pd.DataFrame({'iso3_r250_label': ['NOAREA', 'NORATE', 'ZEROAREA', 'ZERORATE'],
                         'fao_forest_ha': [np.nan, 10.0, 0.0, 10.0]})
    values = pd.DataFrame({'iso3_r250_label': ['NOAREA', 'ZEROAREA', 'ZERORATE'], 'year': [2019] * 3,
                           'nwfp_value_per_ha': [3.0, 3.0, 0.0]})
    out = nf.ntfp_gep_by_country(area, values, 2019, 'fao_forest_ha').set_index('iso3_r250_label')['ntfp_gep']
    assert np.isnan(out['NOAREA']) and np.isnan(out['NORATE'])
    assert out['ZEROAREA'] == 0.0 and out['ZERORATE'] == 0.0


def test_a_repeated_country_in_either_table_is_refused():
    area = pd.DataFrame({'iso3_r250_label': ['AAA', 'AAA'], 'fao_forest_ha': [1.0, 2.0]})
    values = pd.DataFrame({'iso3_r250_label': ['AAA'], 'year': [2019], 'nwfp_value_per_ha': [1.0]})
    with pytest.raises(ValueError):
        nf.ntfp_gep_by_country(area, values, 2019, 'fao_forest_ha')
    with pytest.raises(ValueError):
        nf.ntfp_gep_by_country(area.iloc[:1], pd.concat([values, values]), 2019, 'fao_forest_ha')


def test_source_script_constants_are_pinned():
    """Every value the source module chose, pinned so a claim of following it cannot go stale.

    The docs say this service reproduces the source module's method step for step. That claim is
    only worth as much as the values behind it, so each one is asserted here against the number
    in the source scripts rather than left to be re-read by hand.
    """
    from global_invest.ntfp import ntfp_tasks

    # run_ntfp.py: the buffer distance, and the ESA class range create_forest_mask keeps.
    assert ntfp_functions.NTFP_ACCESS_BUFFER_M == 10_000
    assert (ntfp_functions.ESA_FOREST_CLASS_MIN, ntfp_functions.ESA_FOREST_CLASS_MAX) == (50, 90)

    # run_ntfp.py: the NDVI screen, on the MOD13Q1 convention of NDVI times 10,000.
    assert ntfp_functions.NDVI_MIN_THRESHOLD == 0.20
    assert ntfp_functions.NDVI_SCALE_FACTOR == 0.0001
    assert ntfp_functions.NDVI_NODATA == -9999

    # ntfp_tasks.py: the analysis grid is the account's own, so there is no second projection and
    # no grid constant to pin. What is pinned is that the Mollweide machinery stays out -- by name,
    # not by the word, which the comments use to say why the account's grid is used instead.
    for gone in ('MOLLWEIDE_WKT', 'MOLLWEIDE_BBOX', 'HECTARES_PER_CELL',
                 'ACCESSIBILITY_CELL_SIZE_M', 'reproject_vector', 'buffer_and_union_access'):
        assert not hasattr(ntfp_tasks, gone), '%s came back' % gone
    assert 'ha_per_cell' in inspect.getsource(ntfp_tasks)
    assert ntfp_tasks.REACHABLE_MAX_LATITUDE == 84.0


def test_es_config_row_hydrates_ntfp(tmp_path):
    from global_invest import utilities
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'ntfp', log=lambda *a: None)
    assert p.gep_base_year == 2019


def test_the_template_names_the_complete_roads_layer():
    """The staged globalroads.shp holds 21,438,033 of GRIP4's 25,758,453 lines and no road of types 1 to 3. The
    template names the GeoPackage merged from the seven regional geodatabases, which holds all of them."""
    import csv
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, 'input_template', 'es_parameters.csv')
    rows = {row[1]: row[2] for row in csv.reader(open(path, encoding='utf-8-sig'))
            if len(row) > 2 and row[0] == 'ntfp'}
    assert rows['ntfp_roads_vector_path'] == 'global_invest/ntfp/input/global_roads/grip4_global_roads.gpkg'
    assert rows['ntfp_roads_source'].startswith('GRIP4 ')


# ---------------------------------------------------------------------------
# Accessibility: forest within 10 km of a road or a river.
# ---------------------------------------------------------------------------

def test_forest_mask_covers_the_esa_forest_classes_only():
    block = np.array([10, 49, 50, 70, 90, 91, 200])
    assert nf.forest_mask(block).tolist() == [False, False, True, True, True, False, False]


def test_forest_mask_excludes_nodata_even_inside_the_class_range():
    block = np.array([70, 255])
    assert nf.forest_mask(block, ndv=255).tolist() == [True, False]


def test_access_sources_are_roads_or_rivers():
    roads = np.array([0.0, 120.0, 0.0, 0.0])
    rivers = np.array([0.0, 0.0, 1.0, 0.0])
    assert nf.access_source_mask(roads, rivers).tolist() == [False, True, True, False]


def test_a_cell_exactly_on_the_buffer_edge_counts_as_accessible():
    distance = np.array([0.0, 9.999, 10.0, 10.001])
    assert nf.accessible_mask(distance, buffer_km=10.0).tolist() == [True, True, True, False]


def test_accessible_forest_hectares_needs_both_forest_and_access():
    forest = np.array([True, True, False, False])
    access = np.array([True, False, True, False])
    ha = np.array([9.0, 9.0, 9.0, 9.0])
    assert nf.accessible_forest_hectares(forest, access, ha).tolist() == [9.0, 0.0, 0.0, 0.0]


def test_hectares_by_zone_sums_into_the_zone_id_and_keeps_outside_at_index_zero():
    ha = np.array([[5.0, 5.0], [2.0, 0.0]])
    zones = np.array([[1, 2], [1, 0]])
    out = nf.hectares_by_zone(ha, zones, n_zones=3)
    assert out[1] == 7.0        # two cells in zone 1
    assert out[2] == 5.0
    assert out[3] == 0.0        # a zone with no accessible forest is zero, not missing
    assert out[0] == 0.0        # outside every zone


def test_the_account_values_all_forest_and_the_earlier_valuation_only_the_reachable_part():
    # The rate is an average over all of a country's forest, so the account multiplies it by all
    # of it: 1,000 and 500 FAO hectares at $12 and $10 are CWoN's own $12,000 and $5,000. The
    # earlier valuation, kept as its own table, applies the same rate to the reachable hectares.
    fao = pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB'], 'fao_forest_ha': [1000.0, 500.0]})
    accessible = pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB'], 'accessible_forest_ha': [600.0, 250.0]})
    rates = pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB'], 'year': [2019] * 2,
                          'nwfp_value_per_ha': [12.0, 10.0]})

    main = nf.ntfp_gep_by_country(fao, rates, 2019, 'fao_forest_ha').set_index('iso3_r250_label')
    earlier = nf.ntfp_gep_by_country(accessible, rates, 2019, 'accessible_forest_ha',
                                     'ntfp_gep_accessible_forest').set_index('iso3_r250_label')

    assert main['ntfp_gep'].to_dict() == {'AAA': 12000.0, 'BBB': 5000.0}
    assert earlier['ntfp_gep_accessible_forest'].to_dict() == {'AAA': 7200.0, 'BBB': 2500.0}


def test_ndvi_floor_drops_bare_and_unseen_forest():
    # Four forest cells and one non-forest. Raw NDVI is the value times 10,000, so 2500 is 0.25
    # and passes, 1500 is 0.15 and fails, and -9999 is no data and fails with it. The non-forest
    # cell stays out however green it is, because the floor narrows the mask and cannot widen it.
    forest = np.array([True, True, True, True, False])
    ndvi = np.array([2500, 1500, -9999, 2000, 9000], dtype=np.int16)

    kept = nf.vegetated_forest_mask(forest, ndvi)

    assert list(kept) == [True, False, False, True, False]


def test_ndvi_floor_is_inclusive_at_the_threshold():
    # 0.20 exactly is kept: the threshold is the floor a cell must reach, not clear.
    forest = np.array([True, True])
    ndvi = np.array([2000, 1999], dtype=np.int16)
    assert list(nf.vegetated_forest_mask(forest, ndvi)) == [True, False]


def test_burning_the_country_id_matches_zonal_statistics_over_the_same_polygons(tmp_path):
    """The country stage agrees with the source module's own aggregation, cell for cell.

    The source module takes zonal statistics over the reprojected boundary polygons. The library
    burns the country id onto the analysis grid instead and sums by id in blocks, because reading
    a per-country window out of a seven-billion-cell raster is what the block pass exists to
    avoid. Both assign a cell by which polygon covers its centre, so the totals must agree
    exactly rather than approximately.
    """
    gpd = pytest.importorskip('geopandas')
    zonal_stats = pytest.importorskip('rasterstats').zonal_stats
    from osgeo import gdal
    from shapely.geometry import box

    from global_invest.ntfp import ntfp_tasks

    # Three countries side by side on a 30-cell strip, with deliberately ragged edges so a
    # boundary that falls inside a cell is exercised rather than landing on a cell edge.
    polygons = gpd.GeoDataFrame(
        {'iso3_r250_id': [4, 7, 11]},
        geometry=[box(0.0, 0.0, 9.4, 10.0), box(9.4, 0.0, 20.6, 10.0), box(20.6, 0.0, 30.0, 10.0)],
        crs='EPSG:3857')
    vector_path = str(tmp_path / 'countries.gpkg')
    polygons.to_file(vector_path, driver='GPKG')

    # A one-cell grid over the strip carrying the per-cell hectares the country stage sums.
    values = np.arange(1, 31, dtype='float32').reshape(3, 10) * 1.5
    reference_path = str(tmp_path / 'values.tif')
    raster = gdal.GetDriverByName('GTiff').Create(reference_path, 10, 3, 1, gdal.GDT_Float32)
    raster.SetGeoTransform((0.0, 3.0, 0.0, 10.0, 0.0, -10.0 / 3.0))
    raster.SetProjection(polygons.crs.to_wkt())
    raster.GetRasterBand(1).WriteArray(values)
    raster = None

    burned_path = str(tmp_path / 'country_ids.tif')
    ntfp_tasks.rasterize_polygon_to_grid(vector_path, reference_path, burned_path,
                                         attribute='iso3_r250_id', output_type=gdal.GDT_Int32)
    # The dataset is held while the band is read: chaining the two lets it be collected first.
    burned = gdal.Open(burned_path)
    ids = burned.GetRasterBand(1).ReadAsArray()
    ours = nf.hectares_by_zone(values, ids, 11)

    theirs = zonal_stats(polygons, reference_path, stats=['sum'], nodata=-9999)
    for polygon_id, statistics in zip(polygons['iso3_r250_id'], theirs):
        assert ours[polygon_id] == pytest.approx(statistics['sum'], rel=1e-12)

    # And nothing is lost or double counted between them.
    assert ours.sum() == pytest.approx(values.sum(), rel=1e-12)


def test_the_reach_widens_with_latitude_because_a_cell_narrows(tmp_path):
    """A 100 km reach must cover twice as many columns at 60 N as at the equator.

    Growing the reach in metres is what makes an equal-area grid unnecessary. A cell on
    the account's grid is the same height everywhere and 309*cos(latitude) metres wide, so a
    fixed reach in metres has to cross more of them the further north it runs. Buffering in
    degrees would give the same column count at both latitudes, which is a 10 km buffer at the
    equator and a 20 km one at 60 N.
    """
    gpd = pytest.importorskip('geopandas')
    from osgeo import gdal, osr
    from shapely.geometry import Point

    from global_invest.ntfp import ntfp_tasks

    width, height, cell = 1440, 720, 0.25
    template = str(tmp_path / 'template.tif')
    dataset = gdal.GetDriverByName('GTiff').Create(template, width, height, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform((-180.0, cell, 0, 90.0, 0, -cell))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(np.ones((height, width), dtype='float32'))
    dataset = None

    lines = str(tmp_path / 'lines.gpkg')
    gpd.GeoDataFrame(geometry=[Point(0.125, 0.125), Point(0.125, 60.125)],
                     crs='EPSG:4326').to_file(lines, driver='GPKG')

    out = str(tmp_path / 'reach.tif')
    ntfp_tasks.reachable_mask_on_pyramid([lines], template, out, 100_000.0, log=lambda *a: None)
    reach = gdal.Open(out).ReadAsArray()

    at_equator = int((reach[int((90 - 0.125) // cell)] > 0).sum())
    at_sixty = int((reach[int((90 - 60.125) // cell)] > 0).sum())
    # 3.6 cells each way at the equator and 7.2 at 60 N, rounded to whole cells.
    assert at_equator == 7
    assert at_sixty == 15


def test_burning_in_stripes_gives_the_same_raster_as_one_call(tmp_path):
    """Striping is an implementation detail and must not change a single cell.

    A one-call burn segfaults on the global grid: 21 million road features into an
    8.4-billion-cell target, held at once. Stripes bound what GDAL holds, and each stripe's
    spatial filter means it only sees the features inside it -- so the burn has to come out
    identical either way, or striping has quietly changed the answer.
    """
    gpd = pytest.importorskip('geopandas')
    from osgeo import gdal, osr
    from shapely.geometry import LineString

    from global_invest.ntfp import ntfp_tasks

    width, height, cell = 720, 360, 0.5
    template = str(tmp_path / 'template.tif')
    dataset = gdal.GetDriverByName('GTiff').Create(template, width, height, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform((-180.0, cell, 0, 90.0, 0, -cell))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(np.ones((height, width), dtype='float32'))
    dataset = None

    lines = str(tmp_path / 'lines.gpkg')
    gpd.GeoDataFrame(geometry=[LineString([(-170, -80), (170, 80)]),
                               LineString([(-20, 10), (60, 10)])],
                     crs='EPSG:4326').to_file(lines, driver='GPKG')

    original = ntfp_tasks.BURN_STRIPE_ROWS
    try:
        burned = {}
        for stripe in (4096, 37):        # one stripe covering everything, then many
            ntfp_tasks.BURN_STRIPE_ROWS = stripe
            out = str(tmp_path / ('burn_%d.tif' % stripe))
            ntfp_tasks.burn_lines_in_stripes([lines], template, out, log=lambda *a: None)
            burned[stripe] = gdal.Open(out).ReadAsArray()
    finally:
        ntfp_tasks.BURN_STRIPE_ROWS = original

    assert (burned[4096] > 0).sum() > 0
    assert np.array_equal(burned[4096], burned[37])


def test_a_geopackage_and_a_shapefile_of_the_same_lines_burn_the_same_cells(tmp_path):
    """The roads moved from a shapefile to a GeoPackage, which has no 4 GiB limit. The format must not change a cell.

    The shapefile has no spatial index beside it, the GeoPackage carries its own, and each stripe filters the
    layer to its own extent, so narrow stripes test the filter on both.
    """
    gpd = pytest.importorskip('geopandas')
    from osgeo import gdal, osr
    from shapely.geometry import LineString

    from global_invest.ntfp import ntfp_tasks

    width, height, cell = 720, 360, 0.5
    template = str(tmp_path / 'template.tif')
    dataset = gdal.GetDriverByName('GTiff').Create(template, width, height, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform((-180.0, cell, 0, 90.0, 0, -cell))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(np.ones((height, width), dtype='float32'))
    dataset = None

    lines = gpd.GeoDataFrame(geometry=[LineString([(-170, -80), (170, 80)]),
                                       LineString([(-20, 10), (60, 10)]),
                                       LineString([(100, -40), (100.2, -39.9)])],
                             crs='EPSG:4326')
    original = ntfp_tasks.BURN_STRIPE_ROWS
    try:
        ntfp_tasks.BURN_STRIPE_ROWS = 37
        burned = {}
        for driver, extension in (('GPKG', 'gpkg'), ('ESRI Shapefile', 'shp')):
            path = str(tmp_path / ('roads.%s' % extension))
            lines.to_file(path, driver=driver)
            out = str(tmp_path / ('burn_%s.tif' % extension))
            ntfp_tasks.burn_lines_in_stripes([path], template, out, log=lambda *a: None)
            burned[extension] = gdal.Open(out).ReadAsArray()
    finally:
        ntfp_tasks.BURN_STRIPE_ROWS = original

    assert (burned['gpkg'] > 0).sum() > 0
    assert np.array_equal(burned['gpkg'], burned['shp'])


# ---------------------------------------------------------------------------
# FAO total forest: FRA 2020 interpolated in every country. CWoN's published area apart.
# ---------------------------------------------------------------------------

def cwon_table(rows, series=nf.CWON_AREA_SERIES):
    return pd.DataFrame({'countrycode': [r[0] for r in rows], 'series': series,
                         'YR2019': [r[1] for r in rows]})


def test_cwon_area_reads_hectares_and_keeps_its_missing_mark_missing():
    area = nf.cwon_forest_area(cwon_table([('AAA', 1500), ('BBB', '..'), ('CCC', 0)]), 2019)
    assert area['AAA'] == 1500.0 and area['CCC'] == 0.0
    assert np.isnan(area['BBB'])


def test_cwon_area_refuses_other_text_another_series_or_a_repeated_code():
    with pytest.raises(ValueError):
        nf.cwon_forest_area(cwon_table([('AAA', 'n.a.')]), 2019)
    with pytest.raises(ValueError):
        nf.cwon_forest_area(cwon_table([('AAA', 1)], series='Forest area (1000 ha)'), 2019)
    with pytest.raises(ValueError):
        nf.cwon_forest_area(cwon_table([('AAA', 1), ('AAA', 2)]), 2019)


def test_fra_area_interpolates_between_the_nearest_reported_years_in_thousand_hectares():
    fra = pd.DataFrame({'iso3': ['AAA'] * 3 + ['BBB'] * 2 + ['CCC'] * 2,
                        'year': [2010, 2015, 2020, 2010, 2015, 2010, 2020],
                        'forest': [90.0, 100.0, 150.0, 5.0, 6.0, 10.0, 20.0]})
    area = nf.fra_forest_area(fra, 2019)
    # 100 + (150 - 100) * 4 / 5 thousand hectares.
    assert area['AAA'] == pytest.approx(140_000.0, rel=1e-15)
    # No year after 2019 reported: missing, not extrapolated.
    assert np.isnan(area['BBB'])
    # 2015 missing: the nearest years on each side are 2010 and 2020.
    assert area['CCC'] == pytest.approx(19_000.0, rel=1e-15)
    assert nf.fra_forest_area(fra, 2015)['AAA'] == 100_000.0
    with pytest.raises(ValueError):
        nf.fra_forest_area(pd.concat([fra, fra.iloc[:1]]), 2019)


def test_fao_area_is_fra_in_every_country_and_fills_nothing_else():
    countries = pd.DataFrame({'iso3_r250_id': [1, 2, 3, 4], 'iso3_r250_label': ['CWN', 'FRA', 'NON', 'ZER']})
    fra = pd.Series({'CWN': 999.0, 'FRA': 250.0, 'ZER': 0.0})
    out = nf.fao_forest_ha_by_country(countries, fra).set_index('iso3_r250_label')
    assert out.loc['CWN', 'fao_forest_ha'] == 999.0 and out.loc['CWN', 'fao_forest_source'] == nf.FAO_SOURCE_FRA
    assert out.loc['FRA', 'fao_forest_ha'] == 250.0 and out.loc['FRA', 'fao_forest_source'] == nf.FAO_SOURCE_FRA
    assert np.isnan(out.loc['NON', 'fao_forest_ha']) and out.loc['NON', 'fao_forest_source'] is None
    assert out.loc['ZER', 'fao_forest_ha'] == 0.0
    assert list(out['iso3_r250_id']) == [1, 2, 3, 4]
    with pytest.raises(ValueError):
        nf.fao_forest_ha_by_country(pd.concat([countries, countries.iloc[:1]]), fra)


def test_cwon_area_table_carries_the_published_area_and_fills_nothing():
    countries = pd.DataFrame({'iso3_r250_id': [1, 2, 3], 'iso3_r250_label': ['CWN', 'FRA', 'ZER']})
    out = nf.cwon_forest_ha_by_country(countries, pd.Series({'CWN': 500.0, 'FRA': np.nan, 'ZER': 0.0}))
    out = out.set_index('iso3_r250_label')
    assert list(out.columns) == ['iso3_r250_id', 'cwon_forest_ha']
    assert out.loc['CWN', 'cwon_forest_ha'] == 500.0 and out.loc['ZER', 'cwon_forest_ha'] == 0.0
    assert np.isnan(out.loc['FRA', 'cwon_forest_ha'])


def test_the_account_area_never_reads_cwon_and_the_comparison_never_reads_fra():
    from global_invest.ntfp import ntfp_tasks
    for task in (ntfp_tasks.fao_forest, ntfp_tasks.gep_calculation, ntfp_tasks.divide_by_five_sensitivity):
        source = inspect.getsource(task)
        assert 'ntfp_cwon_' not in source and 'cwon_forest' not in source
    for task in (ntfp_tasks.cwon_forest, ntfp_tasks.cwon_forest_comparison):
        source = inspect.getsource(task)
        assert 'ntfp_fra_' not in source and 'fao_forest' not in source


def _staged_fao_sources():
    import tempfile
    import hazelbean as hb
    base = hb.ProjectFlow(project_dir=os.path.join(tempfile.mkdtemp(), 'fao')).base_data_dir
    folder = os.path.join(base, 'global_invest', 'ntfp', 'cwon_2024_forest_area')
    paths = (os.path.join(folder, 'forest_nontimber_total_area.xlsx'),
             os.path.join(folder, 'for_fao_fra2020_bulk_download.csv'),
             os.path.join(base, 'global_invest', 'ntfp', 'nontimber_price_iucn_edited.csv'))
    if not all(os.path.exists(path) for path in paths):
        pytest.skip('the CWoN 2024 forest area or the rate table is not staged in base data on this machine')
    return paths


def test_the_staged_sources_are_the_package_files_and_fra_reproduces_cwon_where_it_should():
    """The staged files are the package's and the rate table by SHA-256, FRA 2020 interpolated to
    2019 gives CWoN's own area in every country CWoN publishes except the six it extrapolates from
    FRA 2010, and FRA 2020 has an area wherever CWoN does."""
    import hashlib
    cwon_path, fra_path, rate_path = _staged_fao_sources()
    assert hashlib.sha256(open(cwon_path, 'rb').read()).hexdigest().startswith('62e64298c1aae48e')
    assert hashlib.sha256(open(fra_path, 'rb').read()).hexdigest().startswith('ca906a4595fef3e1')
    assert hashlib.sha256(open(rate_path, 'rb').read()).hexdigest().startswith('0094846d3180e343')

    cwon = nf.cwon_forest_area(pd.read_excel(cwon_path), 2019).dropna()
    fra = nf.fra_forest_area(pd.read_csv(fra_path, low_memory=False), 2019)
    assert len(cwon) == 151
    assert fra.reindex(cwon.index).notna().all()
    gap = (fra[cwon.index] - cwon).abs()
    apart = sorted(gap[gap > 1e-6 * cwon.clip(lower=1.0)].index)
    assert apart == sorted(['BOL', 'CIV', 'CZE', 'GBR', 'MKD', 'SWZ'])


def test_the_account_and_the_cwon_comparison_totals_on_the_staged_inputs():
    """The account's value and the comparison, rebuilt from the staged inputs with the library's
    own functions. Pinned to the cent: the CSV parser moves an area by at most a float64 step or
    two between machines, which is millionths of a dollar on these totals."""
    import math
    cwon_path, fra_path, rate_path = _staged_fao_sources()
    rate = nf.nwfp_rate_long(pd.read_csv(rate_path, encoding='utf-8-sig'))
    rate = rate[rate['year'] == 2019].set_index('iso3_r250_label')['nwfp_value_per_ha']
    fra = nf.fra_forest_area(pd.read_csv(fra_path, low_memory=False), 2019)
    cwon = nf.cwon_forest_area(pd.read_excel(cwon_path), 2019)
    account = (rate * fra.reindex(rate.index)).dropna()
    comparison = (rate * cwon.reindex(rate.index)).dropna()
    assert len(account) == 223 and len(comparison) == 149
    assert math.fsum(account) == pytest.approx(31_737_899_452.62, abs=0.01)
    assert math.fsum(comparison) == pytest.approx(30_281_055_011.16, abs=0.01)
    assert math.fsum(account[comparison.index]) == pytest.approx(30_209_130_853.06, abs=0.01)
    six = ['BOL', 'CIV', 'CZE', 'GBR', 'MKD', 'SWZ']
    assert math.fsum(account[six] - comparison[six]) == pytest.approx(-71_924_158.11, abs=0.01)


# ---------------------------------------------------------------------------
# The alternatives: each its own task and table, none of them the account's value.
# ---------------------------------------------------------------------------

def test_divided_rates_divides_only_the_named_countries_and_refuses_an_absent_one():
    rates = pd.DataFrame({'iso3_r250_label': ['COD', 'MOZ', 'BRA'], 'year': [2019] * 3,
                          'nwfp_value_per_ha': [10.0, 5.0, 2.0]})
    out = nf.divided_rates(rates, ['COD', 'MOZ'], 5).set_index('iso3_r250_label')['nwfp_value_per_ha']
    assert out.to_dict() == {'COD': 2.0, 'MOZ': 1.0, 'BRA': 2.0}
    assert rates['nwfp_value_per_ha'].tolist() == [10.0, 5.0, 2.0]
    with pytest.raises(ValueError):
        nf.divided_rates(rates, ['CAF'], 5)


def test_the_account_value_reads_the_published_rate_and_the_divide_by_five_only_its_own_table():
    from global_invest.ntfp import ntfp_tasks
    assert 'divided_rates' not in inspect.getsource(ntfp_tasks.gep_calculation)
    assert 'fao_forest_ha' in inspect.getsource(ntfp_tasks.gep_calculation)
    assert 'divided_rates' in inspect.getsource(ntfp_tasks.divide_by_five_sensitivity)


def test_the_template_registers_the_fao_area_and_the_divide_by_five():
    import csv
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, 'input_template', 'es_parameters.csv')
    rows = {row[1]: row[2] for row in csv.reader(open(path, encoding='utf-8-sig'))
            if len(row) > 2 and row[0] == 'ntfp'}
    assert rows['ntfp_cwon_forest_area_input_path'].endswith('cwon_2024_forest_area/forest_nontimber_total_area.xlsx')
    assert rows['ntfp_fra_forest_input_path'].endswith('cwon_2024_forest_area/for_fao_fra2020_bulk_download.csv')
    assert json.loads(rows['ntfp_divide_by_five_labels']) == ['COD', 'MOZ', 'CAF']
    assert json.loads(rows['ntfp_divide_by_five_divisor']) == 5
    # Both FAO inputs can be fetched by utilities.download_missing_inputs: the package ZIP and the
    # member that is the staged file.
    package = 'FR_WLD_2024_195/Reproducibility package/'
    for name, member in (('ntfp_cwon_forest_area_input', 'Output/Latest/forest_nontimber_total_area.xlsx'),
                         ('ntfp_fra_forest_input', 'Input/User_update/for_fao_fra2020_bulk_download.csv')):
        assert rows[name + '_source_url'] == 'https://reproducibility.worldbank.org/catalog/194/download/549'
        assert rows[name + '_source_archive_member'] == package + member
    # The rate table is not in that package, so it carries a note naming its source, not a URL
    # that would save the package ZIP under the rate table's name.
    assert 'ntfp_value_per_ha_source_url' not in rows
    assert 'geo_ntfp' in rows['ntfp_value_per_ha_source_note']
    # The rivers are a shapefile in a ZIP. One fetched file cannot carry its sidecars, so a URL row
    # would save the ZIP under the .shp name. The note keeps the URL for a person to unzip.
    assert 'ntfp_rivers_source_url' not in rows
    assert 'ne_10m_rivers_lake_centerlines.zip' in rows['ntfp_rivers_source_note']


def test_the_full_tree_runs_every_alternative_as_its_own_task_with_no_switch():
    from global_invest.ntfp import ntfp_initialize

    class Tree:
        def __init__(self):
            self.names = []

        def add_task(self, function, **kwargs):
            self.names.append(function.__name__)
            return function.__name__

    calculation = ntfp_initialize.build_gep_service_calculation_task_tree(Tree()).names
    assert calculation == ['fao_forest', 'gep_calculation']
    full = ntfp_initialize.build_gep_service_task_tree(Tree()).names
    assert full == ['fao_forest', 'gep_calculation', 'divide_by_five_sensitivity', 'cwon_forest',
                    'cwon_forest_comparison', 'accessible_forest', 'esa_forest', 'esa_forest_sensitivity',
                    'accessible_forest_valuation', 'gep_result']


def _grid(tmp_path, name, array, gdal_type):
    from osgeo import gdal, osr
    path = str(tmp_path / name)
    ds = gdal.GetDriverByName('GTiff').Create(path, array.shape[1], array.shape[0], 1, gdal_type)
    ds.SetGeoTransform((0.0, 1.0, 0.0, 3.0, 0.0, -1.0))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(array)
    ds = None
    return path


def test_esa_forest_counts_every_screened_forest_cell_whatever_the_reach(tmp_path):
    """The land cover passed as the reach covers every forest cell, so the accessible-forest call
    gives all forest: ESA classes 50 to 90 that pass the NDVI floor, summed by country."""
    from osgeo import gdal
    from global_invest.ntfp import ntfp_tasks

    lulc = np.array([[50, 70, 10], [90, 100, 60], [55, 0, 80]], dtype=np.int16)
    ids = np.array([[1, 1, 2], [2, 2, 1], [1, 0, 2]], dtype=np.int32)
    ha = np.arange(1, 10, dtype=np.float64).reshape(3, 3)
    ndvi = np.array([[3000, 1500, 9000], [2000, 9000, -9999], [2500, 2500, 4000]], dtype=np.int16)
    reach = np.zeros((3, 3), dtype=np.uint8)
    paths = {k: _grid(tmp_path, k + '.tif', a, t) for k, a, t in (
        ('lulc', lulc, gdal.GDT_Int16), ('ids', ids, gdal.GDT_Int32), ('ha', ha, gdal.GDT_Float64),
        ('ndvi', ndvi, gdal.GDT_Int16), ('reach', reach, gdal.GDT_Byte))}

    total = ntfp_tasks.accessible_forest_hectares_by_country(
        paths['lulc'], paths['lulc'], paths['ids'], paths['ha'], 2, ndvi_path=paths['ndvi'])
    forest = (lulc >= 50) & (lulc <= 90) & (ndvi != -9999) & (ndvi >= 2000)
    expected = np.bincount(ids.ravel(), weights=np.where(forest, ha, 0.0).ravel(), minlength=3)
    assert total.tolist() == expected.tolist()
    # Country 1: cells of 1 ha (class 50, NDVI 0.3) and 7 ha (class 55); the 2 ha cell fails the
    # floor and the 6 ha cell has no NDVI. Country 2: 4 ha (class 90 at the floor) and 9 ha.
    assert total[1] == 8.0 and total[2] == 13.0

    reached = ntfp_tasks.accessible_forest_hectares_by_country(
        paths['lulc'], paths['reach'], paths['ids'], paths['ha'], 2, ndvi_path=paths['ndvi'])
    assert reached.sum() == 0.0

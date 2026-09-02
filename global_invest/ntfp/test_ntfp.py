"""Unit tests for the NTFP skeleton (valuation function + configuration rows)."""
import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from global_invest.ntfp import ntfp_functions
from global_invest.ntfp import ntfp_functions as nf


def test_the_published_rate_is_applied_to_the_reachable_area():
    # The rate is CWoN's own, per hectare, and it is applied to the hectares people can reach.
    # AAA: $1/ha over 100 reachable hectares is $100, not the $200 CWoN spread over all 200.
    area = pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB'], 'accessible_forest_ha': [100.0, 50.0]})
    values = pd.DataFrame({'iso3_r250_label': ['AAA', 'AAA', 'BBB'],
                           'year': [2019, 2020, 2019],
                           'nwfp_value_per_ha': [1.0, 4.5, 2.0]})
    out = ntfp_functions.ntfp_gep_by_country(area, values, 2019).set_index('iso3_r250_label')
    assert out['ntfp_gep'].to_dict() == {'AAA': 100.0, 'BBB': 100.0}
    # Only the requested year is used: AAA's 2020 row would have given a different rate.
    assert out.loc['AAA', 'nwfp_value_per_ha'] == 1.0


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
    # no grid constant to pin. What is pinned is that none came back -- the machinery, not the
    # word, because the comments still say why Mollweide was dropped and that history is the
    # point of them.
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


def test_the_country_total_falls_below_cwons_because_unreachable_forest_is_dropped():
    # CWoN's rate covers all forest. Applying it to only the reachable part drops the value
    # notionally earned where nobody can go, which is the source module's choice and ours.
    accessible = pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB', 'NOACCESS'],
                               'accessible_forest_ha': [600.0, 250.0, 0.0]})
    rates = pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB', 'NOACCESS'], 'year': [2019] * 3,
                          'nwfp_value_per_ha': [12.0, 10.0, 2.0]})

    out = nf.ntfp_gep_by_country(accessible, rates, 2019).set_index('iso3_r250_label')

    assert out.loc['AAA', 'ntfp_gep'] == pytest.approx(7200.0)   # 600 ha at $12
    assert out.loc['BBB', 'ntfp_gep'] == pytest.approx(2500.0)   # 250 ha at $10
    # CWoN priced 1,000 and 500 hectares, so its own totals would be $12,000 and $5,000.
    assert out['ntfp_gep'].sum() == pytest.approx(9700.0)
    # A country with no reachable forest earns nothing rather than an undefined amount.
    assert out.loc['NOACCESS', 'ntfp_gep'] == pytest.approx(0.0)


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

    This is the whole reason the old Mollweide grid existed and the reason it is gone. A cell on
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

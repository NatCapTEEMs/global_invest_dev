"""Unit tests for coastal_carbon: the science pinned on hand-built inputs.

Synthetic data only (the whole calculation needs the coastal source data). Every pure function in
coastal_carbon_functions is checked here on arrays and frames small enough that the expected
value is written out in the comment beside it.

Two contracts get their own tests because they are where the reported number comes from:
the marine surface is valued on `_EEZ` rows only (the service's open question -- see
`test_eez_only_is_a_strict_subset_of_all_rows`), and the r264 correspondence is collapsed to
its canonical row per country before the join, so a split country is counted once (see
global_invest/utilities.py).
"""
import os
from types import SimpleNamespace

import geopandas as gpd
import hazelbean as hb
import numpy as np
import pandas as pd
import pytest
import rasterio
import rasterio.transform
from pyproj import Geod
from shapely.geometry import box

from global_invest.coastal_carbon import coastal_carbon_functions as ccf
from global_invest.coastal_carbon import coastal_carbon_tasks as cct

ID = 'eemarine_r566_id'
# The configured soil depth over the depth the soil products report: 30 of 100 cm.
SOIL_DEPTH_SHARE = 30 / 100


# ---------------------------------------------------------------------------
# Per-pixel density: mangrove
# ---------------------------------------------------------------------------

def test_mangrove_agb_follows_the_hamilton_friess_latitude_regression():
    lat = np.array([0.0, 40.0, 50.0])
    d = ccf.calculate_mangrove_density_array(lat, SOIL_DEPTH_SHARE)
    # AGB (t/ha) = -6.4305 x |lat| + 271.747, then x 0.48 to carbon.
    assert d['agb_c_mg_per_ha'][0] == pytest.approx(271.747 * 0.48)          # 130.43856
    assert d['agb_c_mg_per_ha'][1] == pytest.approx(14.527 * 0.48)           # 6.97296
    # The regression goes negative past ~42.3 deg; the floor keeps biomass at zero, not below.
    assert d['agb_c_mg_per_ha'][2] == 0.0
    assert d['bgb_c_mg_per_ha'][2] == 0.0


def test_mangrove_bgb_ratio_uses_the_ipcc_zone_the_precipitation_puts_the_pixel_in():
    lat = np.array([10.0, 10.0, 30.0])
    precip = np.array([3000.0, 1000.0, 3000.0])
    agb_t = -6.4305 * 10.0 + 271.747                                          # 207.442
    d = ccf.calculate_mangrove_density_array(lat, SOIL_DEPTH_SHARE, precipitation_arr=precip)
    assert d['bgb_c_mg_per_ha'][0] == pytest.approx(agb_t * 0.49 * 0.39)      # tropical wet
    assert d['bgb_c_mg_per_ha'][1] == pytest.approx(agb_t * 0.29 * 0.39)      # tropical dry
    # Past 23.5 deg the pixel is subtropical whatever the rain does.
    agb_t_30 = -6.4305 * 30.0 + 271.747
    assert d['bgb_c_mg_per_ha'][2] == pytest.approx(agb_t_30 * 0.96 * 0.39)

    # With no precipitation raster the whole tropics take the wet ratio.
    dry_free = ccf.calculate_mangrove_density_array(lat, SOIL_DEPTH_SHARE)
    assert dry_free['bgb_c_mg_per_ha'][1] == pytest.approx(agb_t * 0.49 * 0.39)


def test_mangrove_soc_raster_is_scaled_from_one_metre_to_the_configured_depth():
    lat = np.array([5.0, 5.0])
    d = ccf.calculate_mangrove_density_array(lat, SOIL_DEPTH_SHARE, soc_arr=np.array([500.0, 0.0]))
    assert d['soil_c_mg_per_ha'][0] == pytest.approx(150.0)              # 500 x 30/100
    assert d['soil_c_mg_per_ha'][1] == 0.0
    assert d['total_c_mg_per_ha'][0] == pytest.approx(
        d['agb_c_mg_per_ha'][0] + d['bgb_c_mg_per_ha'][0] + 150.0)


def test_mangrove_soc_fallback_steps_down_away_from_the_equator():
    lat = np.array([5.0, 12.0, 17.0, 22.0, 40.0])
    soc = ccf.calculate_mangrove_density_array(lat, SOIL_DEPTH_SHARE)['soil_c_mg_per_ha']
    # The 1 m bands 400, 375, 350, 300 and 250 Mg C/ha, at 30 of 100 cm.
    assert soc == pytest.approx([120.0, 112.5, 105.0, 90.0, 75.0])


def test_the_soil_share_scales_soil_alone():
    lat = np.array([5.0, 30.0])
    at_30 = ccf.calculate_mangrove_density_array(lat, 0.3, soc_arr=np.array([400.0, 200.0]))
    at_20 = ccf.calculate_mangrove_density_array(lat, 0.2, soc_arr=np.array([400.0, 200.0]))
    # Moving from 20 to 30 cm multiplies soil by 1.5 and leaves both biomass pools as they were.
    assert at_30['soil_c_mg_per_ha'] == pytest.approx(at_20['soil_c_mg_per_ha'] * 1.5)
    assert (at_30['agb_c_mg_per_ha'] == at_20['agb_c_mg_per_ha']).all()
    assert (at_30['bgb_c_mg_per_ha'] == at_20['bgb_c_mg_per_ha']).all()
    marsh_30 = ccf.calculate_salt_marsh_density_array(lat, 0.3)
    marsh_20 = ccf.calculate_salt_marsh_density_array(lat, 0.2)
    assert marsh_30['soil_c_mg_per_ha'] == pytest.approx(marsh_20['soil_c_mg_per_ha'] * 1.5)
    assert (marsh_30['agb_c_mg_per_ha'] == marsh_20['agb_c_mg_per_ha']).all()


# ---------------------------------------------------------------------------
# Per-pixel density: salt marsh
# ---------------------------------------------------------------------------

def test_salt_marsh_biomass_takes_the_tropical_boost_and_the_root_ratio():
    d = ccf.calculate_salt_marsh_density_array(np.array([10.0, 30.0]), SOIL_DEPTH_SHARE)
    # Tropical: 5 t/ha x 1.2 = 6 -> AGB C 2.7, BGB 6 x 2.5 = 15 t/ha -> C 6.15.
    assert d['agb_c_mg_per_ha'][0] == pytest.approx(6.0 * 0.45)
    assert d['bgb_c_mg_per_ha'][0] == pytest.approx(6.0 * 2.5 * 0.41)
    # Beyond 25 deg the boost is gone: 5 t/ha -> AGB C 2.25, BGB C 5.125.
    assert d['agb_c_mg_per_ha'][1] == pytest.approx(5.0 * 0.45)
    assert d['bgb_c_mg_per_ha'][1] == pytest.approx(5.0 * 2.5 * 0.41)


def test_salt_marsh_soc_raster_and_the_fallback_that_rises_with_latitude():
    lat = np.array([10.0, 30.0, 40.0, 50.0])
    # The 1 m bands 180, 220, 250 and 350 Mg C/ha, at 30 of 100 cm.
    assert ccf.calculate_salt_marsh_density_array(lat, SOIL_DEPTH_SHARE)['soil_c_mg_per_ha'] \
        == pytest.approx([54.0, 66.0, 75.0, 105.0])
    d = ccf.calculate_salt_marsh_density_array(np.array([10.0]), SOIL_DEPTH_SHARE,
                                               soc_arr=np.array([300.0]))
    assert d['soil_c_mg_per_ha'][0] == pytest.approx(90.0)
    assert d['total_c_mg_per_ha'][0] == pytest.approx(2.7 + 6.15 + 90.0)


# ---------------------------------------------------------------------------
# One density for every mapped pixel: seagrass
# ---------------------------------------------------------------------------

def test_seagrass_densities_split_the_biomass_and_add_the_soil():
    d = ccf.seagrass_pool_densities(1.55, 0.3, 24.2)
    assert d['agb_c_mg_per_ha'] == pytest.approx(0.465)              # 1.55 x 0.3
    assert d['bgb_c_mg_per_ha'] == pytest.approx(1.085)              # 1.55 x 0.7
    assert d['soil_c_mg_per_ha'] == 24.2                             # top 30 cm, not scaled
    assert d['total_c_mg_per_ha'] == pytest.approx(25.75)


# ---------------------------------------------------------------------------
# Per-pixel stock accumulated per region
# ---------------------------------------------------------------------------

def _constant_density(latitude_arr, **_):
    """1 / 2 / 3 Mg C/ha per pool, so the block arithmetic is readable by hand."""
    ones = np.ones_like(np.asarray(latitude_arr, dtype=np.float64))
    return {'agb_c_mg_per_ha': ones * 1.0, 'bgb_c_mg_per_ha': ones * 2.0,
            'soil_c_mg_per_ha': ones * 3.0, 'total_c_mg_per_ha': ones * 99.0}


def test_pixel_stock_sums_counts_only_masked_pixels_inside_a_region_with_area():
    mask = np.array([[1, 1, 0], [1, 0, 1]])
    region = np.array([[1, 2, 1], [0, 1, 2]])
    ha = np.array([[10.0, 20.0, 30.0], [40.0, 0.0, 50.0]])
    sums = ccf.pixel_stock_sums(mask, region, ha, np.zeros((2, 3)), _constant_density,
                                n_region_ids=3)
    # Counted: (0,0) 10 ha in region 1; (0,1) 20 ha and (1,2) 50 ha in region 2.
    # Dropped: (0,2) unmasked, (1,0) region 0 = outside, (1,1) unmasked and zero-area.
    assert sums['agb'].tolist() == [0.0, 10.0, 70.0]
    assert sums['bgb'].tolist() == [0.0, 20.0, 140.0]
    assert sums['soil'].tolist() == [0.0, 30.0, 210.0]
    # The total is the three pools added, never the density function's own total column, so
    # it cannot drift away from the pools it is reported beside.
    assert sums['total'].tolist() == [0.0, 60.0, 420.0]


def test_pixel_stock_sums_treats_nodata_soil_as_zero_rather_than_poisoning_the_region():
    def nan_soil_density(latitude_arr, **_):
        d = _constant_density(latitude_arr)
        d['soil_c_mg_per_ha'] = np.array([[np.nan, 3.0]])
        return d

    sums = ccf.pixel_stock_sums(np.array([[1, 1]]), np.array([[1, 1]]),
                                np.array([[10.0, 10.0]]), np.zeros((1, 2)),
                                nan_soil_density, n_region_ids=2)
    assert sums['soil'][1] == 30.0                     # only the non-nodata pixel
    assert sums['agb'][1] == 20.0                      # biomass at the nodata pixel still counts
    assert sums['total'][1] == 20.0 + 40.0 + 30.0


def test_stock_by_region_frame_keeps_the_given_region_order_and_names_the_ecosystem():
    sums = {'agb': np.array([0.0, 1.0, 2.0, 3.0]), 'bgb': np.array([0.0, 10.0, 20.0, 30.0]),
            'soil': np.array([0.0, 100.0, 200.0, 300.0]),
            'total': np.array([0.0, 111.0, 222.0, 333.0])}
    df = ccf.stock_by_region_frame([3, 1], sums, 'mangrove', ID)
    assert df[ID].tolist() == [3, 1]
    assert list(df.columns) == [ID] + ccf.stock_columns('mangrove')
    assert df['mangrove_total_c_stock_mg'].tolist() == [333.0, 111.0]


def test_stock_and_value_column_names_are_one_convention():
    assert ccf.stock_columns('seagrass') == [
        'seagrass_agb_c_total_mg', 'seagrass_bgb_c_total_mg',
        'seagrass_soil_c_total_mg', 'seagrass_total_c_stock_mg']
    mapping = ccf.stock_to_value_columns('seagrass')
    assert list(mapping.values()) == [
        'seagrass_agb_storage_value', 'seagrass_bgb_storage_value',
        'seagrass_soil_storage_value', 'seagrass_storage_value']
    # The ecosystem total drops the pool word: that is the column gep_calculation sums.
    assert mapping['seagrass_total_c_stock_mg'] == 'seagrass_storage_value'


# ---------------------------------------------------------------------------
# Habitat extent on the marine surface
# ---------------------------------------------------------------------------

def _regions(ids=(1, 2, 3)):
    return gpd.GeoDataFrame(
        {ID: list(ids), 'eemarine_r566_label': [f'R{i}_EEZ' for i in ids],
         'geometry': [box(i - 1, 0, i, 1) for i in ids]}, crs='EPSG:4326')


def test_intersect_features_with_regions_clips_each_piece_to_its_region():
    regions = _regions()
    features = gpd.GeoDataFrame(
        {'GENUS': ['Zostera', 'Posidonia'],
         'geometry': [box(0.5, 0, 1.5, 1), box(50, 50, 51, 51)]}, crs='EPSG:4326')

    pieces = ccf.intersect_features_with_regions(features, regions, desc='test')
    # The straddling polygon is split into a piece per region; region 3 is empty and the far
    # polygon reaches nothing.
    assert sorted(pieces[ID].tolist()) == [1, 2]
    assert set(pieces['GENUS']) == {'Zostera'}
    # Each piece is clipped to its own region, so neither carries the other region's half.
    assert sorted(g.bounds for g in pieces.geometry) == [(0.5, 0.0, 1.0, 1.0),
                                                        (1.0, 0.0, 1.5, 1.0)]


def test_intersect_features_with_regions_raises_when_nothing_intersects():
    far_away = gpd.GeoDataFrame({'geometry': [box(50, 50, 51, 51)]}, crs='EPSG:4326')
    with pytest.raises(ValueError, match='No intersections found'):
        ccf.intersect_features_with_regions(far_away, _regions(), desc='nothing')


def test_add_equal_area_ha_measures_in_the_equal_area_projection():
    gdf = gpd.GeoDataFrame({'id': [1]}, geometry=[box(0, 0, 1, 1)], crs='EPSG:4326')
    out = ccf.add_equal_area_ha(gdf)
    assert out.crs.to_epsg() == 6933
    # A 1x1 degree cell on the equator is ~12,308 km2 = ~1.2308 million ha.
    assert out['area_ha'].iloc[0] == pytest.approx(1_230_846, rel=1e-4)
    assert out['area_ha'].iloc[0] == out['area_m2'].iloc[0] / 10000.0


def test_area_by_region_right_keeps_empty_regions_at_zero_and_left_drops_them():
    regions = _regions()
    pieces = gpd.GeoDataFrame(
        {ID: [1, 1, 2], 'area_ha': [3.0, 4.0, 10.0],
         'geometry': [box(0, 0, 1, 1)] * 3}, crs='EPSG:4326')

    keep_all = ccf.area_by_region(pieces, regions, ID, how='right')
    assert keep_all[ID].tolist() == [1, 2, 3]
    assert keep_all['area_ha'].tolist() == [7.0, 10.0, 0.0]     # region 3 has no habitat
    assert keep_all.crs == regions.crs
    # The geometry came in on the region-id merge, so it belongs to the region on its row.
    assert keep_all.loc[keep_all[ID] == 3, 'geometry'].iloc[0].equals(box(2, 0, 3, 1))

    only_reached = ccf.area_by_region(pieces, regions, ID, how='left')
    assert only_reached[ID].tolist() == [1, 2]


def test_seagrass_stock_by_region_is_the_area_times_each_density():
    area = pd.DataFrame({ID: [1, 2, 3], 'area_ha': [2.0, 10.0, 0.0]})
    df = ccf.seagrass_stock_by_region(area, ID, ccf.seagrass_pool_densities(1.55, 0.3, 24.2))
    row = df.set_index(ID)
    assert row.loc[1, 'seagrass_agb_c_total_mg'] == pytest.approx(0.93)          # 2 x 0.465
    assert row.loc[2, 'seagrass_soil_c_total_mg'] == pytest.approx(242.0)        # 10 x 24.2
    assert row.loc[2, 'seagrass_total_c_stock_mg'] == pytest.approx(257.5)       # 10 x 25.75
    assert row.loc[3, 'seagrass_total_c_stock_mg'] == 0.0      # a region with no seagrass stays
    assert list(df.columns) == [ID] + ccf.stock_columns('seagrass')


def test_presence_area_by_zone_counts_a_shared_pixel_in_each_zone_and_once_as_overlap():
    presence = np.array([[1, 1, 0], [1, 0, 1]], dtype=bool)
    masks = {7: np.array([[1, 1, 0], [0, 0, 0]], dtype=bool),
             9: np.array([[0, 1, 1], [0, 0, 1]], dtype=bool),
             4: np.array([[0, 0, 1], [0, 1, 0]], dtype=bool)}       # holds only absent pixels
    row_area = np.array([[10.0], [20.0]])
    out = ccf.presence_area_by_zone(presence, masks, row_area)
    # Zone 7 holds (0,0) and (0,1); zone 9 holds (0,1) and (1,2). (0,1) is in both.
    assert out['area_ha'] == {7: 20.0, 9: 30.0}
    assert out['shared_ha'] == {7: 10.0, 9: 10.0}
    assert out['presence_ha'] == 60.0                # (1,0), 20 ha, is present and in no zone
    assert out['unique_ha'] == 40.0
    assert out['overlap_counted_ha'] == 10.0         # the zone totals, 50, less unique_ha
    assert sum(out['area_ha'].values()) == out['unique_ha'] + out['overlap_counted_ha']


def _row_ha(top, cell, n_rows):
    """Hectares per cell of n_rows rows of square cells below latitude top, as the seagrass pass computes them."""
    return hb.get_area_of_pixel_column_from_center_lats(cell, top - cell * (np.arange(n_rows) + 0.5)) / 10_000.0


def test_the_row_area_is_the_ellipsoid_area_pyproj_measures():
    geod = Geod(ellps='WGS84')
    cell = 0.000089831528412                         # the GlobalSeagrass pixel, about 10 m
    for top in (72.338455221235890, 30.0, 1e-4, -45.123):
        rows = _row_ha(top, cell, 3)
        for k, hectares in enumerate(rows):
            upper, lower = top - k * cell, top - (k + 1) * cell
            m2, _ = geod.polygon_area_perimeter([0, cell, cell, 0], [upper, upper, lower, lower])
            # Measured worst case over these rows 1.4e-9, rounding in a difference of two large
            # cumulative areas; the sign is random, so it averages out over a tile.
            assert hectares == pytest.approx(abs(m2) / 10_000.0, rel=5e-9)


def _seagrass_fixture(tmp_path):
    """An 8 x 8 presence tile of 0.001 degree cells over (0, 0)-(0.008, 0.008), and four regions.

    A covers columns 0-3 and B columns 3.0-8 in x, so column 3 lies in both. C covers part of
    pixel (7, 7) but not its centre. D lies off the tile.
    """
    tiles = tmp_path / 'tiles'
    tiles.mkdir()
    presence = np.zeros((8, 8), dtype='uint8')
    for row, col in ((0, 3), (1, 4), (7, 7), (5, 0)):
        presence[row, col] = 1
    with rasterio.open(tiles / 'tile_a.tif', 'w', driver='GTiff', width=8, height=8, count=1,
                       dtype='uint8', nodata=0, crs='EPSG:4326',
                       transform=rasterio.transform.from_origin(0.0, 0.008, 0.001, 0.001)) as dst:
        dst.write(presence, 1)
    regions = gpd.GeoDataFrame(
        {ID: [1.0, 2.0, 3.0, 4.0], 'eemarine_r566_label': ['A_EEZ', 'B_EEZ', 'C', 'D_EEZ'],
         'geometry': [box(0.0, 0.0, 0.004, 0.008), box(0.003, 0.0, 0.008, 0.008),
                      box(0.0076, 0.0, 0.008, 0.001), box(5, 5, 6, 6)]}, crs='EPSG:4326')
    regions.to_file(tmp_path / 'regions.gpkg', driver='GPKG')
    return str(tiles), regions


@pytest.mark.parametrize('window', [8, 4])
def test_seagrass_tile_pass_assigns_by_centre_and_reports_the_overlap(tmp_path, monkeypatch, window):
    # At 4 the tile is four windows, A and B each cover two of them whole, and only the windows a
    # boundary crosses are burned. At 8 every region is burned. The answer must not move.
    monkeypatch.setattr(cct, 'SEAGRASS_WINDOW_PIXELS', window)
    tiles, regions = _seagrass_fixture(tmp_path)
    area, shared, diagnostic = cct._seagrass_tile_area(os.path.join(tiles, 'tile_a.tif'), regions, ID)

    row_ha = _row_ha(0.008, 0.001, 8)
    assert area[1] == pytest.approx(row_ha[0] + row_ha[5])                # A: (0,3) and (5,0)
    assert area[2] == pytest.approx(row_ha[0] + row_ha[1] + row_ha[7])    # B: (0,3), (1,4), (7,7)
    assert 3 not in area                 # C reaches into (7,7) but does not hold its centre
    assert 4 not in area
    assert shared[1] == pytest.approx(row_ha[0]) and shared[2] == pytest.approx(row_ha[0])
    assert diagnostic['positive_pixels'] == 4
    assert diagnostic['presence_ha'] == pytest.approx(row_ha[[0, 1, 5, 7]].sum())
    assert diagnostic['unique_ha'] == pytest.approx(diagnostic['presence_ha'])
    assert diagnostic['overlap_counted_ha'] == pytest.approx(row_ha[0])


def test_seagrass_area_task_writes_every_region_and_leaves_no_partial_file(tmp_path):
    tiles, _ = _seagrass_fixture(tmp_path)
    p = SimpleNamespace(run_this=True, cur_dir=str(tmp_path), input_dir=str(tmp_path / 'input'),
                        get_path=lambda *a, **k: '/resolved/' + '/'.join(a),
                        df_countries=pd.DataFrame(), seagrass_extent_dir_path=tiles,
                        gep_regions_input_path=str(tmp_path / 'regions.gpkg'), gep_regions_id_col=ID)
    cct.seagrass_area_within_countries(p)

    area = pd.read_csv(tmp_path / 'seagrass_area_by_countries2019.csv').set_index(ID)
    row_ha = _row_ha(0.008, 0.001, 8)
    assert sorted(area.index) == [1, 2, 3, 4]                  # every region, zero included
    assert area.loc[2, 'area_ha'] == pytest.approx(row_ha[0] + row_ha[1] + row_ha[7])
    assert area.loc[2, 'overlap_ha'] == pytest.approx(row_ha[0])
    assert area.loc[3, 'area_ha'] == 0.0 and area.loc[4, 'area_ha'] == 0.0
    tiles_table = pd.read_csv(tmp_path / 'seagrass_tiles2019.csv')
    assert tiles_table['tile'].tolist() == ['tile_a.tif']
    assert not [name for name in os.listdir(tmp_path) if '_partial' in name]


# ---------------------------------------------------------------------------
# Physical stock -> GEP
# ---------------------------------------------------------------------------

def test_rental_price_for_year_reads_the_base_year_row_not_the_first_row():
    prices = pd.DataFrame({'year': [2018, 2019, 2020], 'rental scc r2%': [12.0, 13.19, 14.0]})
    assert ccf.rental_price_for_year(prices, 2019, 'rental scc r2%') == 13.19
    with pytest.raises(IndexError):
        ccf.rental_price_for_year(prices, 1999, 'rental scc r2%')


def test_apply_rental_price_prices_every_pool_and_records_what_it_used():
    stock = pd.DataFrame({ID: [1, 2],
                          'mangrove_agb_c_total_mg': [10.0, 0.0],
                          'mangrove_bgb_c_total_mg': [20.0, 0.0],
                          'mangrove_soil_c_total_mg': [30.0, 0.0],
                          'mangrove_total_c_stock_mg': [60.0, 0.0]})
    out = ccf.apply_rental_price(stock, ccf.stock_to_value_columns('mangrove'), 13.19, 2019,
                                 'rental scc r2%')
    assert out['mangrove_agb_storage_value'].tolist() == [10.0 * 13.19, 0.0]
    assert out['mangrove_storage_value'].tolist() == [60.0 * 13.19, 0.0]
    # The value equals the pools priced separately, so the total cannot drift from its parts.
    assert out['mangrove_storage_value'][0] == pytest.approx(
        out['mangrove_agb_storage_value'][0] + out['mangrove_bgb_storage_value'][0]
        + out['mangrove_soil_storage_value'][0])
    assert out['year'].tolist() == [2019, 2019]
    assert out['rental scc r2%'].tolist() == [13.19, 13.19]
    assert 'mangrove_agb_storage_value' not in stock.columns      # the input frame is untouched


def _area_frame(ecosystem, ids, areas):
    return pd.DataFrame({ID: ids, 'area_ha': areas})


def _stock_frame(ecosystem, ids, totals):
    agb, bgb, soil, total = ccf.stock_columns(ecosystem)
    return pd.DataFrame({ID: ids, agb: totals, bgb: totals, soil: totals,
                         total: [t * 3 for t in totals]})


def test_combine_ecosystem_areas_and_stocks_adds_the_two_totals_across_habitats():
    regions = pd.DataFrame({ID: [1, 2], 'eemarine_r566_label': ['A_EEZ', 'B_EEZ']})
    df = ccf.combine_ecosystem_areas_and_stocks(
        area_frames={'mangrove': _area_frame('mangrove', [1], [10.0]),
                     'salt_marsh': _area_frame('salt_marsh', [1, 2], [4.0, 6.0]),
                     'seagrass': _area_frame('seagrass', [1, 2], [1.0, 0.0])},
        stock_frames={'mangrove': _stock_frame('mangrove', [1], [100.0]),
                      'salt_marsh': _stock_frame('salt_marsh', [1, 2], [10.0, 20.0]),
                      'seagrass': _stock_frame('seagrass', [1, 2], [1.0, 2.0])},
        df_regions=regions, id_col=ID)

    row1 = df.set_index(ID).loc[1]
    row2 = df.set_index(ID).loc[2]
    assert row1['total_coastal_carbon_area_ha'] == 15.0            # 10 + 4 + 1
    assert row1['total_carbon_stock_mg'] == 333.0                  # (100 + 10 + 1) x 3
    # Region 2 is absent from the mangrove tables: zero mangrove, not a dropped region.
    assert row2['mangrove_area_ha'] == 0.0
    assert row2['mangrove_total_c_stock_mg'] == 0.0
    assert row2['total_coastal_carbon_area_ha'] == 6.0
    assert row2['eemarine_r566_label'] == 'B_EEZ'                  # region attributes joined


def test_combine_ecosystem_areas_and_stocks_without_seagrass_reports_zero_not_missing():
    regions = pd.DataFrame({ID: [1]})
    df = ccf.combine_ecosystem_areas_and_stocks(
        area_frames={'mangrove': _area_frame('mangrove', [1], [10.0]),
                     'salt_marsh': _area_frame('salt_marsh', [1], [4.0])},
        stock_frames={'mangrove': _stock_frame('mangrove', [1], [100.0]),
                      'salt_marsh': _stock_frame('salt_marsh', [1], [10.0])},
        df_regions=regions, id_col=ID)

    # A tree built with include_seagrass=False still writes every seagrass column, at zero, so
    # the combined table has one shape whatever the run included.
    assert df['seagrass_area_ha'].tolist() == [0]
    for col in ccf.stock_columns('seagrass'):
        assert df[col].tolist() == [0]
    assert df['total_coastal_carbon_area_ha'].tolist() == [14.0]
    assert df['total_carbon_stock_mg'].tolist() == [330.0]


def test_storage_value_frame_prices_the_base_year_and_drops_regions_with_no_habitat():
    areas = pd.DataFrame({ID: [1, 2, 3],
                          'total_coastal_carbon_area_ha': [10.0, 5.0, 0.0]})
    prices = pd.DataFrame({'year': [2018, 2019], 'rental scc r2%': [12.0, 13.19]})
    value_frames = {
        'mangrove': pd.DataFrame({ID: [1, 2, 3], 'mangrove_storage_value': [100.0, 50.0, 7.0]}),
        'salt_marsh': pd.DataFrame({ID: [1], 'salt_marsh_storage_value': [30.0]}),
        'seagrass': pd.DataFrame({ID: [1, 2], 'seagrass_storage_value': [20.0, 5.0]}),
    }
    df = ccf.coastal_carbon_storage_value_frame(areas, prices, value_frames, ID, 2019)

    assert df[ID].tolist() == [1, 2]                 # region 3 has no coastal habitat at all
    assert df['rental scc r2%'].tolist() == [13.19, 13.19]
    assert df['coastal_carbon_storage_value'].tolist() == [150.0, 55.0]
    # Region 2 is missing from the salt marsh table: no value there, not NaN swallowing the sum.
    assert df.set_index(ID).loc[2, 'salt_marsh_storage_value'] == 0.0


# ---------------------------------------------------------------------------
# Marine surface -> one row per country
# ---------------------------------------------------------------------------

def _r566_frame():
    """A marine surface holding EEZ rows and the land/territorial rows for the same coasts."""
    return pd.DataFrame({
        'eemarine_r566_label': ['CHN_EEZ', 'NLD_EEZ', 'CHN_LAND', 'NLD_TERRITORIAL'],
        'iso3_r250_label': ['CHN', 'NLD', 'CHN', 'NLD'],
        'coastal_carbon_storage_value': [200.0, 20.0, 150.0, 5.0],
    })


def test_eez_only_is_a_strict_subset_of_all_rows():
    """The service's open question, pinned as arithmetic rather than prose.

    The r566 surface carries a country's coast more than once: an `_EEZ` row plus land and
    territorial rows over the same habitat. Valuing every row therefore returns a strictly
    larger total than valuing the EEZ rows, and the gap is exactly the non-EEZ rows: the
    EEZ-only figure the module writes and the all-rows figure the r566 table sums to are two
    different numbers, not a number and a rounding error.
    """
    df_r566 = _r566_frame()
    eez = ccf.eez_storage_value_by_iso3(df_r566)

    assert eez['iso3_r250_label'].tolist() == ['CHN', 'NLD']
    assert eez['value'].sum() == 220.0                         # 200 + 20
    all_rows_total = df_r566['coastal_carbon_storage_value'].sum()
    assert all_rows_total == 375.0
    assert all_rows_total - eez['value'].sum() == 155.0        # exactly the non-EEZ rows
    assert 'coastal_carbon_storage_value' not in eez.columns   # renamed to the 'value' contract


def test_collapse_to_iso3_r250_counts_a_split_country_once_and_zeroes_the_absent():
    # CHN is split across two r264 sub-regions; only the canonical row (ee_r264_label ==
    # iso3_r250_label) may carry the country's value, or the sum doubles it.
    df_r264 = pd.DataFrame({
        'ee_r264_label': ['CHN', 'CHN_x', 'NLD', 'BOL'],
        'iso3_r250_label': ['CHN', 'CHN', 'NLD', 'BOL'],
        'iso3_r250_name': ['China', 'China', 'Netherlands', 'Bolivia'],
    })
    out = ccf.collapse_to_iso3_r250(ccf.eez_storage_value_by_iso3(_r566_frame()), df_r264)

    assert out['iso3_r250_label'].tolist() == ['BOL', 'CHN', 'NLD']
    assert out.set_index('iso3_r250_label')['value'].to_dict() == {
        'BOL': 0.0, 'CHN': 200.0, 'NLD': 20.0}       # landlocked BOL is zero, not dropped
    assert out.set_index('iso3_r250_label').loc['CHN', 'iso3_r250_name'] == 'China'


# ---------------------------------------------------------------------------
# End to end: the gep_calculation task on synthetic inputs
# ---------------------------------------------------------------------------

def test_gep_calculation_eez_only_and_one_row_per_country(tmp_path):
    # r566 combined-area frame: CHN and NLD EEZ rows, plus a CHN land row that must be EXCLUDED
    # (only *_EEZ rows are valued) and would double CHN if it leaked through.
    pd.DataFrame({
        ID: [1, 2, 3],
        'eemarine_r566_label': ['CHN_EEZ', 'NLD_EEZ', 'CHN_LAND'],
        'iso3_r250_label': ['CHN', 'NLD', 'CHN'],
        'mangrove_total_c_stock_mg': [50.0, 5.0, 50.0],
        'salt_marsh_total_c_stock_mg': [30.0, 3.0, 30.0],
        'seagrass_total_c_stock_mg': [20.0, 2.0, 20.0],
        'total_coastal_carbon_area_ha': [10.0, 1.0, 10.0],
    }).to_csv(tmp_path / 'combined_areas.csv', index=False)

    pd.DataFrame({'year': [2019], 'p': [2.0]}).to_excel(tmp_path / 'carbon_prices.xlsx', index=False)

    # The storage_value tasks own stock x price and gep_calculation merges their totals.
    # Values here = the fixture stocks x price 2.0.
    for eco, stocks in (('mangrove', [50.0, 5.0, 50.0]), ('salt_marsh', [30.0, 3.0, 30.0]),
                        ('seagrass', [20.0, 2.0, 20.0])):
        pd.DataFrame({ID: [1, 2, 3],
                      f'{eco}_storage_value': [v * 2.0 for v in stocks]
                      }).to_csv(tmp_path / f'{eco}_storage_value.csv', index=False)

    # r264 correspondence: CHN split into two r264 rows (canonical row = ee_r264_label == 'CHN');
    # without the canonical filter the join would duplicate CHN and the sum would double it.
    pd.DataFrame({
        'ee_r264_label': ['CHN', 'CHN_x', 'NLD'],
        'iso3_r250_label': ['CHN', 'CHN', 'NLD'],
        'iso3_r250_name': ['China', 'China', 'Netherlands'],
        'continent': ['Asia', 'Asia', 'Europe'],
    }).to_csv(tmp_path / 'ee_r264_correspondence.csv', index=False)

    gpd.GeoDataFrame({ID: [1, 2, 3],
                      'geometry': [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)]},
                     crs='EPSG:4326').to_file(str(tmp_path / 'marine.gpkg'), driver='GPKG')

    p = SimpleNamespace(run_this=True, cur_dir=str(tmp_path), results={},
                        input_dir=str(tmp_path / 'input'),
                        get_path=lambda *a, **k: '/resolved/' + '/'.join(a),
                        df_countries=pd.read_csv(tmp_path / 'ee_r264_correspondence.csv'),
                        mangrove_storage_value_path=str(tmp_path / 'mangrove_storage_value.csv'),
                        salt_marsh_storage_value_path=str(tmp_path / 'salt_marsh_storage_value.csv'),
                        seagrass_storage_value_path=str(tmp_path / 'seagrass_storage_value.csv'),
                        combined_area_path=str(tmp_path / 'combined_areas.csv'),
                        gep_price_input_path=str(tmp_path / 'carbon_prices.xlsx'), gep_price_convention='p',
                        gep_regions_input_path=str(tmp_path / 'marine.gpkg'),
                        df_countries_csv_path=str(tmp_path / 'ee_r264_correspondence.csv'))

    total = cct.gep_calculation(p)

    # CHN EEZ = (50+30+20)*2 = 200, NLD EEZ = (5+3+2)*2 = 20. Land row excluded, split country
    # counted once -> 220 (not 400 via the duplicate r264 join, not 420 via the land leak).
    assert total == 220.0
    # The written table carries the service-named column every other service writes, so the
    # account reads coastal carbon the same way it reads the rest.
    out = pd.read_csv(tmp_path / 'gep_by_country_base_year.csv').set_index('iso3_r250_label')
    assert out.loc['CHN', 'coastal_carbon_gep'] == 200.0
    assert out.loc['NLD', 'coastal_carbon_gep'] == 20.0
    assert (out['coastal_carbon_gep'].index.value_counts() == 1).all()   # one row per iso3 country
    assert list(out.reset_index().columns[-2:]) == ['year', 'coastal_carbon_gep']

    # Stage-1 r566 detail written for csv + map gpkg (per-region values, never summed downstream).
    r566 = pd.read_csv(tmp_path / 'gep_by_country_base_year_r566.csv')
    assert r566.loc[r566['eemarine_r566_label'] == 'CHN_EEZ', 'coastal_carbon_storage_value'].item() == 200.0
    # The same frame valued on every row would be 420: the EEZ filter is what makes it 220.
    assert r566['coastal_carbon_storage_value'].sum() == 420.0


def test_es_config_and_parameters_rows_hydrate_coastal_carbon(tmp_path):
    from global_invest import utilities
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'coastal_carbon', log=lambda *a: None)
    assert p.gep_base_year == 2019
    # The marine surface, not the land one: the aggregation key the whole module runs on.
    assert p.gep_regions_id_col == 'eemarine_r566_id'
    assert p.gep_regions_input_path.endswith('eemarine_r566_correspondence.gpkg')
    utilities.hydrate_es_parameters(p, 'coastal_carbon', log=lambda *a: None)
    assert p.mangrove_vector_path.endswith('gmw_v3_2019_vec.shp')
    # Soil to 30 cm of the 1 m the soil products report, and the GlobalSeagrass extent with its
    # uniform densities.
    assert (p.coastal_carbon_soil_depth_cm, p.coastal_carbon_soil_source_depth_cm) == (30, 100)
    assert p.seagrass_extent_dir_path.endswith('global_invest/coastal_carbon/GlobalSeagrass2019_2020')
    assert p.seagrass_extent_md5 == '2046bf6c62d85f4d57e678bf6ef1cd07'
    assert p.seagrass_extent_source_url == 'https://doi.org/10.5281/zenodo.18612240'
    assert (p.seagrass_biomass_c_mg_per_ha, p.seagrass_agb_fraction, p.seagrass_soil_c_mg_per_ha) \
        == (1.55, 0.3, 24.2)
    assert not hasattr(p, 'seagrass_vector_path')

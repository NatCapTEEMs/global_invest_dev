"""Unit tests for the flood_control port (committed-output anchored).

reference/ ships the extracted per-country avoided-damage table AND the pipeline's own global
summary, so every run verifies the anchor against the pipeline's own bookkeeping: the total,
the country counts, and the zero/NaN distinction (assessed-zero vs never-assessed).

The chain section pins the ported per-step science (flood_control_chain) on hand-built
inputs: 3x3 arrays and 5-row tables whose expected values are computed in the comments.
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from global_invest.flood_control import flood_control_chain as fchain
from global_invest.flood_control import flood_control_functions as fc

REFERENCE_DIR = os.path.join(os.path.dirname(fc.__file__), 'reference')


def test_anchor_reproduces_the_pipelines_own_global_summary():
    avoided = pd.read_csv(os.path.join(REFERENCE_DIR, 'country_avoided_damage_usd2019.csv'))
    summary = pd.read_csv(os.path.join(REFERENCE_DIR, 'summary_global.csv')).iloc[0]
    assert np.isclose(avoided['avoided_damage_usd2019'].sum() / 1e9,
                      summary['global_total_expected_avoided_damage_usd2019_bil'], rtol=1e-9)
    assert int((avoided['avoided_damage_usd2019'] > 0).sum()) == int(summary['n_nonzero'])
    assert int(avoided['avoided_damage_usd2019'].notna().sum()) == int(summary['n_countries'])
    assert int((avoided['avoided_damage_usd2019'] == 0).sum()) == int(summary['n_zero'])


def test_join_keeps_all_countries_and_distinguishes_zero_from_unassessed():
    avoided = pd.read_csv(os.path.join(REFERENCE_DIR, 'country_avoided_damage_usd2019.csv'))
    countries = avoided[['iso3_r250_id', 'iso3_r250_label']]
    out = fc.flood_control_gep_by_country(avoided, countries)
    assert len(out) == 250
    assert (out['flood_control_gep'] == 0).sum() > 0        # assessed, zero avoided damage
    # ESH (Western Sahara) was never assessed: NaN, not zero.
    assert np.isnan(out.set_index('iso3_r250_label').loc['ESH', 'flood_control_gep'])


def test_es_config_and_parameters_rows_hydrate_flood_control(tmp_path):
    from global_invest import utilities
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'flood_control', log=lambda *a: None)
    assert p.gep_base_year == 2019
    utilities.hydrate_es_parameters(p, 'flood_control', log=lambda *a: None)
    assert p.flood_control_avoided_damage_path.endswith('country_avoided_damage_usd2019.csv')


# ---------------------------------------------------------------------------
# Chain section: the ported FINAL_GEP_FLOOD per-step science.
# ---------------------------------------------------------------------------

def test_currency_constant_matches_the_pipelines_committed_multiplier():
    # FRED DEXUSEU 2010 average x GDPDEF 2019/2010 ratio; the currency-audit notebook
    # commits combined = 1.538344.
    assert abs(fchain.EUR2010_TO_USD2019 - 1.538344) < 5e-6


def test_landtype_ladder_normalizes_and_maps_to_sda():
    assert fchain.normalize_landtype_label('Residential') == 'Residential buildings'
    assert fchain.normalize_landtype_label('roads') == 'Infrastructure - roads'
    assert fchain.normalize_landtype_label('Agriculture') == 'Agriculture'
    assert fchain.LANDTYPE_TO_SDA['Agriculture'] == 'crop'
    assert fchain.LANDTYPE_TO_SDA['Infrastructure - roads'] == 'artif'
    assert 'Grassland' not in fchain.LANDTYPE_TO_SDA           # unmapped stays sector-only
    assert fchain.fmt_depth_col(1.0) == '1m'
    assert fchain.fmt_depth_col(0.5) == '0.5m'


def test_build_canonical_damage_table_joins_and_drops():
    fractional = pd.DataFrame({
        'JRC_Region': ['R1', 'R1', 'R1', 'R1', 'R2', 'R2'],
        'LandType': ['Agriculture', 'Agriculture', 'Residential buildings',
                     'Residential buildings', 'Agriculture', 'Agriculture'],
        'depth_m': [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        'fraction': [0.0, 0.5, 0.0, 0.4, 0.0, 0.8],
    })
    # 'Residential' (short label) must normalize to the fraction-curve label to join.
    max_damage = pd.DataFrame({
        'iso3': ['AAA', 'AAA', 'BBB', 'CCC'],
        'LandType': ['Agriculture', 'Residential', 'Agriculture', 'Agriculture'],
        'max_damage_eur_m2': [100.0, 200.0, 50.0, 10.0],
    })
    iso3_region = pd.DataFrame({'iso3': ['AAA', 'BBB'], 'JRC_Region': ['R1', 'R2']})

    out = fchain.build_canonical_damage_table(fractional, max_damage, iso3_region)
    idx = out.set_index(['iso3', 'landtype', 'depth_m'])['damage_per_m2']
    assert idx.loc[('AAA', 'Agriculture', 1.0)] == 50.0            # 100 x 0.5
    assert idx.loc[('AAA', 'Residential buildings', 1.0)] == 80.0  # 200 x 0.4
    assert idx.loc[('BBB', 'Agriculture', 1.0)] == 40.0            # 50 x 0.8
    assert 'CCC' not in out['iso3'].values                          # no JRC region -> dropped


def test_usd_conversion_and_sda_aggregation():
    eur_long = pd.DataFrame({
        'iso3': ['AAA'] * 3,
        'landtype': ['Agriculture', 'Residential buildings', 'Commercial buildings'],
        'depth_m': [1.0, 1.0, 1.0],
        'damage_per_m2': [50.0, 80.0, 120.0],
    })
    usd = fchain.convert_damage_table_to_usd2019(eur_long, combined_multiplier=2.0)
    assert usd['damage_per_m2'].tolist() == [100.0, 160.0, 240.0]
    assert (usd['currency'] == 'USD2019').all()

    sda = fchain.aggregate_damage_table_to_sda(usd, set_pasture_equal_crop=True)
    assert set(sda['sda_type']) == {'crop', 'artif', 'pasture'}
    # pasture rows are copies of the crop rows.
    assert (sda[sda['sda_type'] == 'pasture']['damage_per_m2'].values
            == sda[sda['sda_type'] == 'crop']['damage_per_m2'].values).all()

    wide = fchain.pivot_damage_table_wide(sda, group_cols=['iso3', 'sda_type'])
    w = wide.set_index('sda_type')['1m']
    assert w.loc['artif'] == 200.0    # the artif curve is the MEAN of the built-asset curves
    assert w.loc['crop'] == 100.0


def test_build_sda_class_priorities_floodplain_and_roads():
    lulc = np.array([[190, 10, 130],
                     [50, 20, 255],
                     [130, 190, 30]])
    floodplain = np.array([[True, True, True],
                           [True, False, True],
                           [True, False, True]])
    sda = fchain.build_sda_class(lulc, floodplain=floodplain, nodata=255)
    expected = np.array([[1, 2, 3],
                         [0, 0, 0],    # forest; outside floodplain; nodata
                         [3, 0, 2]])
    assert (sda == expected).all()

    no_pasture = fchain.build_sda_class(lulc, floodplain=floodplain, nodata=255,
                                        include_pasture=False)
    assert no_pasture[0, 2] == 0 and no_pasture[2, 0] == 0

    roads = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]])
    with_roads = fchain.build_sda_class(lulc, floodplain=floodplain, nodata=255, roads=roads)
    assert with_roads[0, 1] == 4 and with_roads[2, 0] == 4     # roads override in floodplain

    global_sda = fchain.build_sda_class(lulc, nodata=255)      # no floodplain: global raster
    assert global_sda[1, 0] == 0 and global_sda[1, 1] == 2 and global_sda[2, 1] == 1


def test_build_depthbin_index_clamps_and_flags_nodata():
    depth = np.array([[-1.0, 0.0, 0.3],
                      [0.5, 5.9, 99.0]])
    nodata_mask = np.array([[False, False, False],
                            [False, False, True]])
    idx = fchain.build_depthbin_index(depth, nodata_mask, max_depth=6.0)
    expected = np.array([[0, 0, 1],
                         [1, 12, -1]], dtype=np.int16)   # 99 clamps to 6m (bin 12) but is nodata
    assert (idx == expected).all()
    assert idx.dtype == np.int16


def test_service_flow_from_spa_clips_and_zeroes_outside_sda():
    spa = np.array([[0.5, 1.5, 255.0],
                    [-0.2, 0.7, 0.3]], dtype='float32')
    sda_mask = np.array([[True, True, True],
                         [True, False, True]])
    flow = fchain.service_flow_from_spa(spa, sda_mask, nodata=255.0)
    assert flow[0, 0] == np.float32(0.5)
    assert flow[0, 1] == 1.0            # clipped
    assert np.isnan(flow[0, 2])          # nodata inside SDA stays NaN
    assert flow[1, 0] == 0.0            # negative clipped
    assert flow[1, 1] == 0.0            # outside SDA -> exactly zero
    assert flow[1, 2] == np.float32(0.3)


def test_damage_curves_and_interpolation():
    row = {'iso3': 'AAA', 'sda_type': 'artif'}
    row.update({fchain.fmt_depth_col(d): 10.0 * d for d in fchain.DEPTH_BINS_M})
    curves = fchain.damage_curves_from_wide(pd.DataFrame([row]))
    xs, ys = curves[('AAA', 'artif')]
    assert xs.tolist() == [float(d) for d in fchain.DEPTH_BINS_M]

    d = fchain.interp_damage_per_m2(np.array([0.25, 2.5, 7.0, -1.0]), xs, ys)
    assert np.allclose(d, [2.5, 25.0, 60.0, 0.0])    # linear; clamped to [0, 6]


def test_pixel_damage_totals_hand_computed_3x3():
    depth = np.array([[0.25, 1.0, 7.0],
                      [-5.0, 3.0, 0.0],
                      [2.0, 0.5, np.nan]], dtype='float32')
    sda = np.array([[1, 1, 1],
                    [2, 2, 2],
                    [3, 0, 1]])
    curves = {}
    for sda_type, slope in (('artif', 10.0), ('crop', 2.0)):    # no pasture curve
        xs = np.array(fchain.DEPTH_BINS_M, dtype='float32')
        curves[('AAA', sda_type)] = (xs, slope * xs)

    raster, totals, total = fchain.pixel_damage_totals(depth, sda, curves, 'AAA',
                                                       pixel_area_m2=2.0)
    # artif: depths 0.25, 1, 7->6 -> per m2 2.5+10+60 -> x2 = 145. crop: only depth 3
    # (negative and zero depths are not flooded) -> 6 x 2 = 12. pasture: no curve -> 0.
    assert totals['artif'] == 145.0
    assert totals['crop'] == 12.0
    assert totals['pasture'] == 0.0
    assert total == 157.0
    assert raster[0, 0] == 5.0 and raster[1, 1] == 12.0
    assert raster[1, 0] == 0.0 and raster[2, 2] == 0.0


def test_ead_from_rp_damages_hand_computed():
    # Points in p-space with anchor + flat tail: (0, 200) (0.01, 200) (0.1, 100) (1, 0).
    # Trapezoids: 0.01x200 + 0.09x150 + 0.9x50 = 2 + 13.5 + 45 = 60.5.
    ead, pts = fchain.ead_from_rp_damages([10, 100], [100.0, 200.0])
    assert np.isclose(ead, 60.5)
    assert len(pts) == 4

    # Zero tail replaces the first trapezoid: 0.01x100 = 1 -> 59.5.
    ead_zero, _ = fchain.ead_from_rp_damages([10, 100], [100.0, 200.0], tail_mode='zero')
    assert np.isclose(ead_zero, 59.5)

    # No anchor: integration stops at p = 0.1 -> 2 + 13.5 = 15.5.
    ead_no_anchor, _ = fchain.ead_from_rp_damages([10, 100], [100.0, 200.0],
                                                  add_p1_zero=False)
    assert np.isclose(ead_no_anchor, 15.5)

    # Duplicate RPs average before integrating.
    ead_dup, _ = fchain.ead_from_rp_damages([10, 10, 100], [100.0, 200.0, 200.0])
    assert np.isclose(ead_dup, fchain.ead_from_rp_damages([10, 100], [150.0, 200.0])[0])

    # enforce_monotone lifts the decreasing point: damages [200, 100] -> [200, 200]
    # -> 0.01x200 + 0.09x200 + 0.9x100 = 110.
    ead_mono, _ = fchain.ead_from_rp_damages([10, 100], [200.0, 100.0],
                                             enforce_monotone=True)
    assert np.isclose(ead_mono, 110.0)

    assert fchain.ead_from_rp_damages([], [])[0] == 0.0


def test_ead_from_rp_stack_matches_the_point_version_per_pixel():
    stack = np.array([[[100.0, 40.0], [0.0, 10.0]],
                      [[200.0, 80.0], [0.0, 30.0]]])
    ead = fchain.ead_from_rp_stack(stack, [10, 100])
    for i in range(2):
        for j in range(2):
            expected, _ = fchain.ead_from_rp_damages([10, 100], stack[:, i, j])
            assert np.isclose(ead[i, j], expected, rtol=1e-6)

    # RP order must not matter.
    ead_rev = fchain.ead_from_rp_stack(stack[::-1], [100, 10])
    assert np.allclose(ead, ead_rev)


def test_country_ead_table_fills_missing_with_zero_and_status():
    admin0 = pd.DataFrame({'iso3': ['AAA', 'BBB', 'CCC'],
                           'region_wb': ['R1', 'R1', 'R2']})
    country = pd.DataFrame({'iso3': ['AAA', 'BBB'], 'ead_usd2019': [5.0, 0.0]})

    out = fchain.country_ead_table(country, admin0).set_index('iso3')
    assert out.loc['AAA', 'status'] == 'ok'
    assert out.loc['BBB', 'status'] == 'ok'                    # assessed zero stays 'ok'
    assert out.loc['CCC', 'status'] == 'missing_step4c'
    assert out.loc['CCC', 'ead_usd2019'] == 0.0
    assert out.loc['AAA', 'region_wb'] == 'R1'

    strict = fchain.country_ead_table(country, admin0, fill_missing_zero=False)
    assert strict['iso3'].tolist() == ['AAA', 'BBB']

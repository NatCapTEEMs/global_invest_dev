"""Unit tests for moisture_recycling's re-attribution.

Every function is pinned on a hand-built three-country matrix, so the orientation convention
(columns are the destination whose precipitation is being sourced) and the conservation the
construction promises are both stated as executable facts.
"""
import numpy as np
import pandas as pd

from global_invest.moisture_recycling import moisture_recycling_functions as mr


def a_matrix_frame():
    """Three countries as shipped: m_-prefixed labels, columns are destinations.

    AAA sends 0.2 of BBB's rain and keeps 0.1 of its own; BBB keeps 0.3 of its own;
    CCC receives only ocean rain and sends nothing.
    """
    return pd.DataFrame({
        'Unnamed: 0': ['m_AAA', 'm_BBB', 'm_CCC'],
        'm_AAA': [0.1, 0.0, 0.0],
        'm_BBB': [0.2, 0.3, 0.0],
        'm_CCC': [0.0, 0.0, 0.0],
    })


def test_the_prefix_is_stripped_and_the_frame_is_square():
    m = mr.moisture_matrix(a_matrix_frame())
    assert list(m.index) == ['AAA', 'BBB', 'CCC']
    assert list(m.columns) == ['AAA', 'BBB', 'CCC']


def test_a_volume_table_is_refused():
    volumes = a_matrix_frame()
    volumes['m_AAA'] = [33.4, 0.1, 0.0]
    try:
        mr.moisture_matrix(volumes)
    except NameError:
        return
    raise AssertionError('a cell above 1 must be refused: it means mm, not fractions')


def test_the_terrestrial_share_is_the_column_sum():
    shares = mr.terrestrial_precipitation_share(mr.moisture_matrix(a_matrix_frame()))
    assert np.isclose(shares['BBB'], 0.5)          # 0.2 from AAA + 0.3 self-recycled
    assert np.isclose(shares['CCC'], 0.0)          # ocean rain only


def test_value_moves_upstream_and_nothing_is_lost():
    m = mr.moisture_matrix(a_matrix_frame())
    water = pd.Series({'AAA': 100.0, 'BBB': 1000.0, 'CCC': 500.0})
    out = mr.reattributed_water_value(m, water, ecosystem_share=1.0).set_index('iso3_r250_label')
    # AAA is credited its own recycled share plus what it exports to BBB.
    assert np.isclose(out.loc['AAA', 'moisture_recycling_gep'], 0.1 * 100 + 0.2 * 1000)
    assert np.isclose(out.loc['AAA', 'moisture_recycling_gep_own_part'], 0.1 * 100)
    assert np.isclose(out.loc['AAA', 'moisture_recycling_gep_export_part'], 0.2 * 1000)
    # CCC's water is ocean-fed and its land sends nothing, so it holds no value on either side.
    assert out.loc['CCC', 'moisture_recycling_gep'] == 0.0
    assert out.loc['CCC', 'moisture_recycling_sink_side_value'] == 0.0
    # Conservation: the source-side and sink-side views are the same money.
    assert np.isclose(out['moisture_recycling_gep'].sum(),
                      out['moisture_recycling_sink_side_value'].sum())


def test_the_ecosystem_share_scales_everything_linearly():
    m = mr.moisture_matrix(a_matrix_frame())
    water = pd.Series({'AAA': 100.0, 'BBB': 1000.0, 'CCC': 500.0})
    full = mr.reattributed_water_value(m, water, ecosystem_share=1.0)
    half = mr.reattributed_water_value(m, water, ecosystem_share=0.5)
    assert np.isclose(half['moisture_recycling_gep'].sum(),
                      0.5 * full['moisture_recycling_gep'].sum())


def test_an_untracked_destination_is_named_and_receives_no_attribution():
    """The shipped matrix carries all-NaN columns for islands below the tracking grid. Their
    water value must go unattributed rather than poisoning every dot product with NaN."""
    frame = a_matrix_frame()
    frame['m_CCC'] = [np.nan, np.nan, np.nan]
    m = mr.moisture_matrix(frame)
    assert mr.untracked_destinations(m) == ['CCC']
    water = pd.Series({'AAA': 100.0, 'BBB': 1000.0, 'CCC': 500.0})
    out = mr.reattributed_water_value(m, water, ecosystem_share=1.0).set_index('iso3_r250_label')
    assert np.isfinite(out['moisture_recycling_gep']).all()
    assert np.isclose(out['moisture_recycling_gep'].sum(), 0.1 * 100 + 0.2 * 1000 + 0.3 * 1000)
    assert out.loc['CCC', 'moisture_recycling_sink_side_value'] == 0.0


def test_a_destination_absent_from_the_matrix_is_dropped_not_invented():
    m = mr.moisture_matrix(a_matrix_frame())
    water = pd.Series({'AAA': 100.0, 'BBB': 1000.0, 'CCC': 500.0, 'DDD': 9999.0})
    out = mr.reattributed_water_value(m, water, ecosystem_share=1.0)
    assert 'DDD' not in set(out['iso3_r250_label'])
    assert np.isclose(out['moisture_recycling_gep'].sum(), 0.1 * 100 + 0.2 * 1000 + 0.3 * 1000)

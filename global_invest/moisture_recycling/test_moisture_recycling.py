"""Unit tests for moisture_recycling's source-side diagnostic.

Every function is pinned on a hand-built three-country matrix, so the orientation convention
(columns are the destination whose precipitation is being sourced) and the agreement of the
diagnostic's source-side and sink-side totals are both stated as executable facts. That
agreement is one double sum computed two ways, not a conservation of the headline.
"""
import os

import numpy as np
import pandas as pd
import pytest

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
    # The diagnostic's source-side and sink-side totals agree: one double sum computed two ways.
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


def test_own_part_is_the_labelled_diagonal_under_any_column_order():
    """The own/export split must not depend on the matrix's column order: the own part is read
    from the labelled diagonal, so a column-reordered matrix gives the same frame."""
    matrix = pd.DataFrame([[0.5, 0.2, 0.0], [0.1, 0.6, 0.1], [0.0, 0.1, 0.7]],
                          index=['AAA', 'BBB', 'CCC'], columns=['AAA', 'BBB', 'CCC'])
    value = pd.Series({'AAA': 100.0, 'BBB': 200.0, 'CCC': 300.0})
    ordered = mr.reattributed_water_value(matrix, value, 1.0).set_index('iso3_r250_label')
    shuffled = mr.reattributed_water_value(matrix[['BBB', 'CCC', 'AAA']], value, 1.0
                                            ).set_index('iso3_r250_label')
    assert ordered['moisture_recycling_gep_own_part'].tolist() == [50.0, 120.0, 210.0]
    pd.testing.assert_frame_equal(ordered, shuffled)


def test_duplicated_matrix_labels_are_refused():
    matrix = pd.DataFrame([[0.5, 0.2], [0.1, 0.6]], index=['AAA', 'AAA'], columns=['AAA', 'BBB'])
    value = pd.Series({'AAA': 100.0, 'BBB': 200.0})
    with pytest.raises(ValueError, match='duplicated country labels'):
        mr.reattributed_water_value(matrix, value, 1.0)


def test_the_water_input_is_resolved_and_pinned(tmp_path):
    """The resolved path lands on p as moisture_recycling_water_gep_path, which is what puts the
    table's content fingerprint into the run signature; an explicit setting wins unchanged."""
    import os
    from types import SimpleNamespace
    from global_invest.moisture_recycling import moisture_recycling_tasks as mt

    p = SimpleNamespace(project_dir=str(tmp_path),
                        moisture_recycling_water_gep_relative_file='intermediate/x/gep.csv')
    path = mt.water_input_path(p)
    assert path == os.path.normpath(os.path.join(str(tmp_path), 'intermediate/x/gep.csv'))
    assert p.moisture_recycling_water_gep_path == path
    water = os.path.join(str(tmp_path), 'water.csv')
    explicit = SimpleNamespace(moisture_recycling_water_gep_path=water)
    assert mt.water_input_path(explicit) == water


def test_rainfall_reuse_decision_tracks_the_live_water_table(tmp_path):
    """The four cases of the caching contract, on temporary tables through the task's own
    resolution and the shared signature machinery: an output with no prior signature cannot be
    reused; unchanged inputs reuse; a content change at the same path recomputes; a path change
    with identical bytes recomputes (the fingerprint carries the path)."""
    from types import SimpleNamespace
    from global_invest import utilities
    from global_invest.moisture_recycling import moisture_recycling_tasks as mt
    from global_invest.moisture_recycling import moisture_recycling_functions as mr_f

    out_dir = os.path.join(str(tmp_path), 'gep_calculation')
    os.makedirs(out_dir)
    output = os.path.join(out_dir, 'gep_by_country_base_year.csv')
    pd.DataFrame({'iso3_r250_label': ['nor'], 'moisture_recycling_gep': [1.0]}).to_csv(
        output, index=False)
    water = os.path.join(str(tmp_path), 'water_gep_by_country.csv')
    with open(water, 'w') as f:
        f.write('iso3_r250_label,water_use_rainfed_vmr_gep\nnor,2.0\n')
    p = SimpleNamespace(project_dir=str(tmp_path),
                        moisture_recycling_water_gep_path=water,
                        moisture_recycling_ecosystem_share=0.5)
    mt.water_input_path(p)
    outputs = [output]
    dependencies = [mt.__file__, mr_f.__file__]

    # absent prior signature: what produced the output is unknown, so it cannot be reused
    reason = utilities.reuse_reason(p, 'moisture_recycling', outputs, light_inputs=dependencies)
    assert reason is not None and 'signature' in reason

    utilities.write_reuse_signature(p, 'moisture_recycling', outputs, light_inputs=dependencies)
    # unchanged inputs: reuse
    assert utilities.reuse_reason(p, 'moisture_recycling', outputs,
                                  light_inputs=dependencies) is None

    # content change at the SAME path: recompute, and the reason names the water table
    with open(water, 'w') as f:
        f.write('iso3_r250_label,water_use_rainfed_vmr_gep\nnor,3.0\n')
    reason = utilities.reuse_reason(p, 'moisture_recycling', outputs, light_inputs=dependencies)
    assert reason is not None and 'moisture_recycling_water_gep_path' in reason

    # path change with IDENTICAL bytes: still recompute -- which table was read is part of
    # what produced the output
    utilities.write_reuse_signature(p, 'moisture_recycling', outputs, light_inputs=dependencies)
    moved = os.path.join(str(tmp_path), 'water_gep_by_country_moved.csv')
    with open(water) as src, open(moved, 'w') as dst:
        dst.write(src.read())
    p.moisture_recycling_water_gep_path = moved
    reason = utilities.reuse_reason(p, 'moisture_recycling', outputs, light_inputs=dependencies)
    assert reason is not None and 'moisture_recycling_water_gep_path' in reason


def test_the_task_resolves_and_checks_reuse_before_the_early_return_and_records_after_writing():
    """Source order, the shock-table house pattern: the water input is resolved and the signature
    consulted BEFORE any output-exists return, and the signature is recorded after the outputs
    are written. A bare `if already_done: return` is exactly the stale-rainfall defect."""
    import inspect
    from global_invest.moisture_recycling import moisture_recycling_tasks as mt

    src = inspect.getsource(mt.gep_calculation)
    assert 'if already_done:' not in src, 'the output-exists return must consult the signature'
    resolve = src.index('water_input_path(')
    check = src.index('already_done and utilities.reuse_reason(')
    heavy = src.index('moisture_matrix(')
    record = src.index('utilities.write_reuse_signature(')
    write = src.index('write_gep_by_country(')
    assert resolve < check < heavy, 'resolve the live input, decide, only then read anything heavy'
    assert write < record, 'the signature must describe outputs that exist'

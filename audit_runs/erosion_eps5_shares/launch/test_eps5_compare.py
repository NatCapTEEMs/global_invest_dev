"""Tests for the EPS-5 comparison and gates, on small synthetic files.

Run: python -I -B -m pytest launch/test_eps5_compare.py -q -p no:cacheprovider
"""
import importlib.util
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ec, eg = load('eps5_compare'), load('eps5_gates')
HEADER = 'component,ISO3,crop,elasticity_used,protected_production_tons,total_production_tons,share_protected_production\n'
ROWS = ['onfarm,CUB,bana,0.08,65.82103368127747,260.69230263087445,0.2524855280229594\n',
        'onfarm,NAM,whea,0.3,0.0,10.0,0.0\n',
        'upstream,CUB,bana,0.08,1.5,3.0,0.5\n',
        'combined,ATA,rice,0.2,0.0,0.0,\n']
SETTINGS = [{'setting': 'threshold_high', 'value': '5.0', 'task_attribute': 'erosion_threshold_high_t_ha_yr', 'author_manifest_key': 'threshold_high_t_ha_yr'},
            {'setting': 'apply_severe_filter', 'value': 'true', 'task_attribute': 'erosion_apply_severe_filter', 'author_manifest_key': 'apply_severe_filter'},
            {'setting': 'rasterize_all_touched', 'value': 'false', 'task_attribute': 'erosion_rasterize_all_touched', 'author_manifest_key': ''}]


def table(tmp_path, name, rows=ROWS, header=HEADER):
    path = tmp_path / name
    path.write_text(header + ''.join(rows))
    return str(path)


def test_the_same_bytes_are_byte_identical(tmp_path):
    result = ec.compare_long(table(tmp_path, 'a.csv'), table(tmp_path, 'b.csv'))
    assert result['verdict'] == ec.BYTE_IDENTICAL and result['summary']['keys_shared'] == 4 and result['rows_that_differ'].empty


def test_the_same_values_in_another_row_order_are_values_identical(tmp_path):
    result = ec.compare_long(table(tmp_path, 'a.csv'), table(tmp_path, 'b.csv', rows=ROWS[::-1]))
    assert result['verdict'] == ec.VALUES_IDENTICAL and result['summary']['row_order_identical'] is False


def test_a_last_digit_difference_counts_with_no_tolerance(tmp_path):
    changed = [ROWS[0].replace('65.82103368127747', '65.82103368127748')] + ROWS[1:]
    result = ec.compare_long(table(tmp_path, 'a.csv'), table(tmp_path, 'b.csv', rows=changed))
    assert result['verdict'] == ec.DIFFERENT
    differing = result['rows_that_differ']
    assert len(differing) == 1 and differing.loc[0, 'column'] == 'protected_production_tons' and differing.loc[0, 'ISO3'] == 'CUB'
    assert differing.loc[0, 'left_minus_right'] != 0


def test_a_share_that_is_missing_on_one_side_is_a_mask_difference(tmp_path):
    changed = ROWS[:3] + ['combined,ATA,rice,0.2,0.0,0.0,0.0\n']
    columns = ec.compare_long(table(tmp_path, 'a.csv'), table(tmp_path, 'b.csv', rows=changed))['columns']
    row = columns[(columns['group'] == 'all') & (columns['column'] == 'share_protected_production')].iloc[0]
    assert row['missing_mismatch'] == 1 and row['missing_left'] == 1 and row['missing_right'] == 0


def test_a_value_that_is_zero_on_one_side_is_a_zero_mask_difference(tmp_path):
    changed = [ROWS[0], ROWS[1].replace(',0.0,10.0,', ',0.5,10.0,')] + ROWS[2:]
    columns = ec.compare_long(table(tmp_path, 'a.csv'), table(tmp_path, 'b.csv', rows=changed))['columns']
    row = columns[(columns['group'] == 'onfarm') & (columns['column'] == 'protected_production_tons')].iloc[0]
    assert row['zero_mismatch'] == 1 and row['max_abs_difference'] == 0.5


def test_a_label_that_reads_as_missing_stays_a_key(tmp_path):
    rows = ['onfarm,NA,whea,0.3,1.0,2.0,0.5\n', 'onfarm,NAM,whea,0.3,1.0,2.0,0.5\n']
    result = ec.compare_long(table(tmp_path, 'a.csv', rows=rows), table(tmp_path, 'b.csv', rows=rows[::-1]))
    assert result['summary']['keys_shared'] == 2 and result['verdict'] == ec.VALUES_IDENTICAL


def test_a_key_on_one_side_is_reported_by_side(tmp_path):
    result = ec.compare_long(table(tmp_path, 'a.csv'), table(tmp_path, 'b.csv', rows=ROWS[:3]))
    assert result['verdict'] == ec.DIFFERENT and result['summary']['keys_left_only'] == 1 and result['summary']['keys_right_only'] == 0
    assert result['keys_one_side'].loc[0, 'ISO3'] == 'ATA' and result['keys_one_side'].loc[0, 'side'] == 'left only'


def test_a_repeated_key_stops_the_value_comparison(tmp_path):
    result = ec.compare_long(table(tmp_path, 'a.csv', rows=ROWS + ROWS[:1]), table(tmp_path, 'b.csv'))
    assert result['verdict'] == ec.DIFFERENT and result['summary']['repeated_keys_left'] == 1 and result['columns'].empty


def test_a_different_column_set_is_different(tmp_path):
    other = HEADER.replace('share_protected_production', 'share')
    result = ec.compare_long(table(tmp_path, 'a.csv'), table(tmp_path, 'b.csv', header=other))
    assert result['verdict'] == ec.DIFFERENT and result['summary']['columns_identical'] is False


def test_a_text_column_difference_is_listed(tmp_path):
    a = table(tmp_path, 'a.csv', rows=['XKX,Serbia,2.0\n'], header='ISO3,country_name,threshold_t_ha_yr\n')
    b = table(tmp_path, 'b.csv', rows=['XKX,Kosovo,2.0\n'], header='ISO3,country_name,threshold_t_ha_yr\n')
    result = ec.compare_tables(a, b, ['ISO3'])
    assert result['verdict'] == ec.DIFFERENT and result['text_differences'].loc[0, 'left'] == 'Serbia' and result['rows_that_differ'].empty


def raster(tmp_path, name, array, nodata=-9999.0, compress=None, transform=None):
    import rasterio
    from rasterio.transform import from_origin
    path = str(tmp_path / name)
    with rasterio.open(path, 'w', driver='GTiff', width=array.shape[1], height=array.shape[0], count=1, dtype='float32', nodata=nodata,
                       crs='EPSG:8857', transform=transform or from_origin(0, 0, 1000, 1000), **({'compress': compress} if compress else {})) as d:
        d.write(array.astype('float32'), 1)
    return path


GRID = np.array([[0.0, 0.5, -9999.0], [0.25, 0.0, 1.0]])


def test_rasters_with_other_compression_and_nodata_are_values_identical(tmp_path):
    other = np.where(GRID == -9999.0, 3e38, GRID)
    result = ec.compare_rasters(raster(tmp_path, 'a.tif', GRID), raster(tmp_path, 'b.tif', other, nodata=3e38, compress='deflate'), block_rows=1)
    assert result['verdict'] == ec.VALUES_IDENTICAL and result['valid_left'] == 5 and result['zero_left'] == 2 and result['cells'] == 6


def test_a_raster_cell_difference_and_a_mask_difference_are_counted_apart(tmp_path):
    other = GRID.copy()
    other[0, 1] = 0.75
    other[1, 2] = -9999.0
    result = ec.compare_rasters(raster(tmp_path, 'a.tif', GRID), raster(tmp_path, 'b.tif', other))
    assert result['verdict'] == ec.DIFFERENT and result['values_differ'] == 1 and result['valid_mismatch'] == 1 and result['max_abs_difference'] == 0.25


def test_a_zero_on_one_side_of_a_raster_counts_in_the_zero_mask(tmp_path):
    other = GRID.copy()
    other[0, 0] = 0.5
    result = ec.compare_rasters(raster(tmp_path, 'a.tif', GRID), raster(tmp_path, 'b.tif', other))
    assert result['zero_mismatch'] == 1 and result['verdict'] == ec.DIFFERENT


def test_another_grid_stops_the_raster_comparison(tmp_path):
    from rasterio.transform import from_origin
    result = ec.compare_rasters(raster(tmp_path, 'a.tif', GRID), raster(tmp_path, 'b.tif', GRID, transform=from_origin(1000, 0, 1000, 1000)))
    assert result['verdict'] == 'GRID_DIFFERS' and 'values_differ' not in result


def test_the_policy_moves_only_default_high_rows():
    text = b'ISO3,country_name,threshold_t_ha_yr,reason\nITA,Italy,11.0,default-high\nALB,Albania,2.0,small-area\nX11,"A, 11.0,b",2.0,low-elevation\n'
    moved, count = eg.policy_at_threshold(text)
    assert count == 1 and b'ITA,Italy,5.0,default-high' in moved and b'X11,"A, 11.0,b",2.0,low-elevation' in moved and len(moved) == len(text) - 1


def test_the_author_settings_gate_stops_on_another_threshold():
    eg.check_author_settings({'threshold_high_t_ha_yr': 5.0, 'apply_severe_filter': True}, SETTINGS)
    with pytest.raises(eg.Mismatch, match='threshold_high_t_ha_yr'):
        eg.check_author_settings({'threshold_high_t_ha_yr': 11.0, 'apply_severe_filter': True}, SETTINGS)


def test_the_task_settings_gate_reads_what_hydration_left():
    ok = SimpleNamespace(erosion_threshold_high_t_ha_yr=5.0, erosion_apply_severe_filter=True, erosion_rasterize_all_touched=False)
    eg.check_task_settings(ok, SETTINGS)
    with pytest.raises(eg.Mismatch, match='erosion_threshold_high_t_ha_yr'):
        eg.check_task_settings(SimpleNamespace(erosion_threshold_high_t_ha_yr=11.0, erosion_apply_severe_filter=True, erosion_rasterize_all_touched=False), SETTINGS)


def test_the_author_manifests_may_differ_only_by_threshold_outputs_and_time():
    eps5 = {'timestamp_utc': 'a', 'threshold_high_t_ha_yr': 5.0, 'inputs': {'usle_path': 'x'}, 'outputs': {'o': '1'}, 'elapsed_minutes': 1}
    ses11 = {'timestamp_utc': 'b', 'threshold_high_t_ha_yr': 11.0, 'inputs': {'usle_path': 'x'}, 'outputs': {'o': '2'}, 'elapsed_minutes': 2}
    eg.check_manifests_differ_only_by_threshold(eps5, ses11)
    with pytest.raises(eg.Mismatch, match='inputs.usle_path'):
        eg.check_manifests_differ_only_by_threshold(eps5, dict(ses11, inputs={'usle_path': 'y'}))


def test_a_path_into_the_repair_is_refused():
    eg.refuse_repaired(['/scratch/base_data/global_invest/erosion/spam2020_yield_stack_TA.tif'])
    for path in ['/x/spam_stack_repair_20261103/output/a.tif', '/x/spam2020_yield_stack_TA_repaired_20261103.tif']:
        with pytest.raises(eg.Mismatch, match='repaired'):
            eg.refuse_repaired([path])


def test_the_stacks_gate_needs_the_author_bytes():
    rows = [{'parameter': 'erosion_area_stack_path', 'path': '/b/area.tif', 'sha256': 'a'}, {'parameter': 'erosion_yield_stack_path', 'path': '/b/yield.tif', 'sha256': 'y'},
            {'parameter': 'erosion_bandmap_csv_path', 'path': '/b/map.csv', 'sha256': 'm'}]
    pins = [{'role': 'area_stack', 'sha256': 'a'}, {'role': 'yield_stack', 'sha256': 'y'}, {'role': 'bandmap', 'sha256': 'm'}]
    eg.check_original_stacks(rows, pins)
    with pytest.raises(eg.Mismatch, match='erosion_yield_stack_path'):
        eg.check_original_stacks(rows, [pins[0], {'role': 'yield_stack', 'sha256': 'other'}, pins[2]])


def test_the_run_manifest_gate_names_settings_and_inputs(tmp_path):
    path = tmp_path / 'manifest.json'
    good = {'threshold_high_t_ha_yr': 5.0, 'apply_severe_filter': True, 'inputs': {'area_stack': '/b/area.tif', 'usle_path': '/b/usle.tif'}}
    path.write_text(json.dumps(good))
    eg.check_run_manifest(str(path), SETTINGS, {'area_stack': '/b/area.tif'})
    with pytest.raises(eg.Mismatch, match='area_stack'):
        eg.check_run_manifest(str(path), SETTINGS, {'area_stack': '/other.tif'})
    path.write_text(json.dumps(dict(good, inputs={'area_stack': '/x/spam2020_harvested_area_stack_TA_repaired_20261103.tif'})))
    with pytest.raises(eg.Mismatch, match='repaired'):
        eg.check_run_manifest(str(path), SETTINGS, {})


def share_raster(path, values):
    import rasterio
    from rasterio.transform import from_origin
    array = np.asarray(values, dtype='float32')
    with rasterio.open(path, 'w', driver='GTiff', height=array.shape[0], width=array.shape[1], count=1, dtype='float32', crs='EPSG:8857', transform=from_origin(0, 0, 1, 1)) as d:
        d.write(array, 1)
    return str(path)


def prediction_cells(row=1, col=1, library=0.5, author=0.25):
    columns = {'row': [0, row], 'col': [0, col]}
    for part in ('onfarm', 'upstream', 'combined'):
        columns['eps5_%s_predicted_library' % part] = [0.0, library if part == 'onfarm' else 0.0]
        columns['eps5_%s_stored_author' % part] = [0.0, author if part == 'onfarm' else 0.0]
    import pandas as pd
    return pd.DataFrame(columns)


def test_a_raster_differing_only_at_the_predicted_cell_is_exact(tmp_path):
    ep = load('eps5_prediction')
    left, right = share_raster(tmp_path / 'l.tif', [[0, 0], [0, 0.5]]), share_raster(tmp_path / 'r.tif', [[0, 0], [0, 0.25]])
    result = ep.check_raster('onfarm', left, right, prediction_cells())
    assert result['as_predicted'] and result['magnitude'] == ep.EXACT


def test_a_raster_differing_at_another_cell_falsifies_the_prediction(tmp_path):
    ep = load('eps5_prediction')
    left, right = share_raster(tmp_path / 'l.tif', [[0, 0.75], [0, 0.5]]), share_raster(tmp_path / 'r.tif', [[0, 0], [0, 0.25]])
    result = ep.check_raster('onfarm', left, right, prediction_cells())
    assert not result['as_predicted'] and not result['location_as_predicted'] and 'outside the prediction 1' in result['detail']
    assert result['magnitude'] == ep.NOT_EVALUATED


def test_a_predicted_cell_that_does_not_differ_falsifies_the_prediction(tmp_path):
    ep = load('eps5_prediction')
    left, right = share_raster(tmp_path / 'l.tif', [[0, 0], [0, 0.25]]), share_raster(tmp_path / 'r.tif', [[0, 0], [0, 0.25]])
    assert not ep.check_raster('onfarm', left, right, prediction_cells())['as_predicted']


def test_a_predicted_cell_off_by_its_last_bit_falsifies_the_prediction(tmp_path):
    ep = load('eps5_prediction')
    off = float(np.nextafter(np.float32(0.5), np.float32(1.0)))
    left, right = share_raster(tmp_path / 'l.tif', [[0, 0], [0, off]]), share_raster(tmp_path / 'r.tif', [[0, 0], [0, 0.25]])
    result = ep.check_raster('onfarm', left, right, prediction_cells())
    assert not result['as_predicted'] and result['location_as_predicted'] and result['magnitude'] == ep.OFF_PREDICTION


def test_an_author_cell_off_its_stored_value_falsifies_the_prediction(tmp_path):
    ep = load('eps5_prediction')
    left, right = share_raster(tmp_path / 'l.tif', [[0, 0], [0, 0.5]]), share_raster(tmp_path / 'r.tif', [[0, 0], [0, 0.125]])
    assert not ep.check_raster('onfarm', left, right, prediction_cells())['as_predicted']


def test_rasters_on_another_grid_are_not_read(tmp_path):
    ep = load('eps5_prediction')
    left, right = share_raster(tmp_path / 'l.tif', [[0, 0], [0, 0.5]]), share_raster(tmp_path / 'r.tif', [[0, 0, 0], [0, 0.25, 0]])
    result = ep.check_raster('onfarm', left, right, prediction_cells())
    assert not result['as_predicted'] and result['magnitude'] == ep.NOT_EVALUATED


LONG_HEADER = 'component,ISO3,crop,elasticity_used,protected_production_tons,total_production_tons,share_protected_production\n'
LONG_ROWS = ['onfarm,SRB,whea,0.3,10.0,100.0,0.1\n', 'onfarm,FRA,whea,0.3,20.0,100.0,0.2\n', 'combined,SRB,whea,0.3,30.0,100.0,0.3\n']
BOUNDS = {('onfarm', 'SRB', 'whea', 'protected_production_tons'): 1e-8, ('onfarm', 'SRB', 'whea', 'share_protected_production'): 1e-10}


def long_comparison(tmp_path, rows):
    a = table(tmp_path, 'a.csv', rows=rows, header=LONG_HEADER)
    b = table(tmp_path, 'b.csv', rows=LONG_ROWS, header=LONG_HEADER)
    return ec.compare_long(a, b)


def test_identical_tables_pass_with_no_difference(tmp_path):
    ep = load('eps5_prediction')
    result = ep.check_protected_table('t', long_comparison(tmp_path, LONG_ROWS), BOUNDS)
    assert result['as_predicted'] and result['magnitude'] == ep.NO_DIFFERENCE


def test_a_value_within_its_bound_passes_and_says_it_is_a_bound(tmp_path):
    ep = load('eps5_prediction')
    rows = ['onfarm,SRB,whea,0.3,10.000000001,100.0,0.1\n'] + LONG_ROWS[1:]
    result = ep.check_protected_table('t', long_comparison(tmp_path, rows), BOUNDS)
    assert result['as_predicted'] and result['location_as_predicted'] and result['magnitude'] == ep.WITHIN_BOUND


def test_a_huge_error_on_a_permitted_row_falsifies_the_prediction(tmp_path):
    ep = load('eps5_prediction')
    rows = ['onfarm,SRB,whea,0.3,1000000000000.0,100.0,0.1\n'] + LONG_ROWS[1:]
    result = ep.check_protected_table('t', long_comparison(tmp_path, rows), BOUNDS)
    assert not result['as_predicted'] and result['location_as_predicted'] and result['magnitude'] == ep.OVER_BOUND


def test_an_error_just_over_the_bound_falsifies_the_prediction(tmp_path):
    ep = load('eps5_prediction')
    rows = ['onfarm,SRB,whea,0.3,10.00000002,100.0,0.1\n'] + LONG_ROWS[1:]
    assert not ep.check_protected_table('t', long_comparison(tmp_path, rows), BOUNDS)['as_predicted']


def test_a_share_error_on_a_permitted_row_falsifies_the_prediction(tmp_path):
    ep = load('eps5_prediction')
    rows = ['onfarm,SRB,whea,0.3,10.0,100.0,0.11\n'] + LONG_ROWS[1:]
    assert not ep.check_protected_table('t', long_comparison(tmp_path, rows), BOUNDS)['as_predicted']


@pytest.mark.parametrize('rows', [['onfarm,FRA,whea,0.3,20.000000001,100.0,0.2\n'.replace('FRA', 'SRB')] + LONG_ROWS[1:] + ['onfarm,SRB,whea,0.3,10.0,100.0,0.1\n'],
                                  LONG_ROWS + [LONG_ROWS[0]]])
def test_repeated_keys_fail_closed_though_the_frames_are_empty(tmp_path, rows):
    ep = load('eps5_prediction')
    comparison = long_comparison(tmp_path, rows)
    assert comparison['rows_that_differ'].empty and comparison['keys_one_side'].empty
    result = ep.check_protected_table('t', comparison, BOUNDS)
    assert not result['as_predicted'] and result['magnitude'] == ep.NOT_EVALUATED and 'repeated keys' in result['structure_problems']


def test_another_schema_fails_closed(tmp_path):
    ep = load('eps5_prediction')
    a = table(tmp_path, 'a.csv', rows=['onfarm,SRB,whea,10.0\n'], header='component,ISO3,crop,protected_production_tons\n')
    b = table(tmp_path, 'b.csv', rows=LONG_ROWS, header=LONG_HEADER)
    result = ep.check_protected_table('t', ec.compare_long(a, b), BOUNDS)
    assert not result['as_predicted'] and 'the columns differ' in result['structure_problems']


def test_a_zero_mask_change_on_a_permitted_row_fails_closed(tmp_path):
    ep = load('eps5_prediction')
    a = table(tmp_path, 'a.csv', rows=['onfarm,SRB,whea,0.3,0.0,100.0,0.1\n'] + LONG_ROWS[1:], header=LONG_HEADER)
    b = table(tmp_path, 'b.csv', rows=LONG_ROWS, header=LONG_HEADER)
    result = ep.check_protected_table('t', ec.compare_long(a, b), {k: 1e12 for k in BOUNDS})
    assert not result['as_predicted'] and 'zero-mask' in result['structure_problems']


def test_a_missing_mask_change_on_a_permitted_row_fails_closed(tmp_path):
    ep = load('eps5_prediction')
    rows = ['onfarm,SRB,whea,0.3,,100.0,0.1\n'] + LONG_ROWS[1:]
    result = ep.check_protected_table('t', long_comparison(tmp_path, rows), {k: 1e12 for k in BOUNDS})
    assert not result['as_predicted'] and 'missing-mask' in result['structure_problems']


def test_a_key_on_one_side_or_another_row_order_fails_closed(tmp_path):
    ep = load('eps5_prediction')
    assert 'keys on one side' in ep.check_protected_table('t', long_comparison(tmp_path, LONG_ROWS[:2]), BOUNDS)['structure_problems']
    assert 'row order' in ep.check_protected_table('t', long_comparison(tmp_path, list(reversed(LONG_ROWS))), BOUNDS)['structure_problems']


def test_a_difference_outside_the_predicted_rows_or_columns_fails(tmp_path):
    ep = load('eps5_prediction')
    for rows in (['onfarm,SRB,whea,0.3,10.0,100.0,0.1\n', 'onfarm,FRA,whea,0.3,20.000000001,100.0,0.2\n', LONG_ROWS[2]],
                 ['onfarm,SRB,whea,0.3,10.0,100.000000001,0.1\n'] + LONG_ROWS[1:],
                 ['onfarm,SRB,whea,0.31,10.0,100.0,0.1\n'] + LONG_ROWS[1:]):
        result = ep.check_protected_table('t', long_comparison(tmp_path, rows), BOUNDS)
        assert not result['as_predicted'] and not result['location_as_predicted'] and result['magnitude'] == ep.NOT_EVALUATED


NAMED_HEADER = 'ISO3,country_name,mean_ps_onfarm_cropland_severe\n'
NAMED_ROWS = ['XKX,Serbia,0.5\n', 'SRB,Serbia,0.25\n']


def named_comparison(tmp_path, rows):
    a = table(tmp_path, 'a.csv', rows=rows, header=NAMED_HEADER)
    b = table(tmp_path, 'b.csv', rows=NAMED_ROWS, header=NAMED_HEADER)
    baseline = tmp_path / 'baseline.csv'
    baseline.write_text('ISO3,column,left,right\nXKX,country_name,Kosovo,Serbia\n')
    return ec.compare_tables(a, b, ['ISO3']), baseline


def test_named_text_must_equal_the_baseline_and_values_stay_within_the_bound_on_file(tmp_path):
    ep = load('eps5_prediction')
    bounds = {('', 'SRB', '', ep.DIAGNOSTIC_VALUE_COLUMN): 1e-8}
    good, baseline = named_comparison(tmp_path, ['XKX,Kosovo,0.5\n', 'SRB,Serbia,0.250000001\n'])
    result = ep.check_named_text('t', good, baseline, bounds)
    assert result['as_predicted'] and result['magnitude'] == ep.WITHIN_BOUND
    assert not ep.check_named_text('t', good, baseline)['as_predicted'], 'no bound on file'
    huge, baseline = named_comparison(tmp_path, ['XKX,Kosovo,0.5\n', 'SRB,Serbia,0.9\n'])
    assert ep.check_named_text('t', huge, baseline, bounds)['magnitude'] == ep.OVER_BOUND
    other, baseline = named_comparison(tmp_path, ['XKX,Kosovo,0.500000001\n', 'SRB,Serbia,0.25\n'])
    assert not ep.check_named_text('t', other, baseline, bounds)['location_as_predicted']
    renamed, baseline = named_comparison(tmp_path, ['XKX,Albania,0.5\n', 'SRB,Serbia,0.25\n'])
    assert not ep.check_named_text('t', renamed, baseline, bounds)['as_predicted']


def test_named_text_with_repeated_keys_or_another_schema_fails_closed(tmp_path):
    ep = load('eps5_prediction')
    repeated, baseline = named_comparison(tmp_path, ['XKX,Kosovo,0.5\n', 'XKX,Kosovo,0.5\n', 'SRB,Serbia,0.25\n'])
    assert 'repeated keys' in ep.check_named_text('t', repeated, baseline)['structure_problems']
    a = table(tmp_path, 'c.csv', rows=['XKX,Kosovo\n', 'SRB,Serbia\n'], header='ISO3,country_name\n')
    b = table(tmp_path, 'd.csv', rows=NAMED_ROWS, header=NAMED_HEADER)
    assert 'the columns differ' in ep.check_named_text('t', ec.compare_tables(a, b, ['ISO3']), baseline)['structure_problems']


AUDIT_HEADER = 'band,crop,total_area_ha_src\n'


def audit_comparison(tmp_path, rows):
    a = table(tmp_path, 'a.csv', rows=rows, header=AUDIT_HEADER)
    b = table(tmp_path, 'b.csv', rows=['1,bana,5216161\n'], header=AUDIT_HEADER)
    baseline = tmp_path / 'b_rows.csv'
    baseline.write_text('band,crop,column,left,right,left_minus_right\n1,bana,total_area_ha_src,5216159.5,5216161,-1.5\n')
    return ec.compare_tables(a, b, ['band', 'crop']), baseline


def test_baseline_values_must_match_key_column_and_both_sides_and_repeats_fail_closed(tmp_path):
    ep = load('eps5_prediction')
    good, baseline = audit_comparison(tmp_path, ['1,bana,5216159.5\n'])
    assert ep.check_baseline_values('t', good, baseline)['as_predicted']
    off, baseline = audit_comparison(tmp_path, ['1,bana,5216159.0\n'])
    assert not ep.check_baseline_values('t', off, baseline)['as_predicted']
    none, baseline = audit_comparison(tmp_path, ['1,bana,5216161\n'])
    assert not ep.check_baseline_values('t', none, baseline)['as_predicted']
    repeated, baseline = audit_comparison(tmp_path, ['1,bana,5216159.5\n', '1,bana,5216159.5\n'])
    result = ep.check_baseline_values('t', repeated, baseline)
    assert not result['as_predicted'] and 'repeated keys' in result['structure_problems']


def test_a_clean_table_with_a_difference_or_repeated_keys_fails(tmp_path):
    ep = load('eps5_prediction')
    clean = (ec.BYTE_IDENTICAL, ec.VALUES_IDENTICAL)
    assert ep.check_clean('t', long_comparison(tmp_path, LONG_ROWS), clean)['as_predicted']
    assert not ep.check_clean('t', long_comparison(tmp_path, ['onfarm,SRB,whea,0.3,10.5,100.0,0.1\n'] + LONG_ROWS[1:]), clean)['as_predicted']
    assert not ep.check_clean('t', long_comparison(tmp_path, LONG_ROWS + [LONG_ROWS[0]]), clean)['as_predicted']


def test_the_derived_bound_holds_on_the_tasks_own_arithmetic():
    """Two runs that differ by one float32 step of share on four cells, through float32 products and a float64 sum."""
    ep = load('eps5_prediction')
    random = np.random.default_rng(4242)
    used = 0.0
    for _ in range(200):
        cells = 20000
        share = random.uniform(0.0, 1.0, cells).astype('float32')
        yields, area = random.uniform(1e-4, 0.05, cells).astype('float32'), random.uniform(0.0, 3000.0, cells).astype('float32')
        moved = random.choice(cells, 4, replace=False)
        share[moved] = random.uniform(0.5, 0.99, 4).astype('float32')
        other = share.copy()
        other[moved] = np.nextafter(share[moved], np.float32(1.0))
        step = float(other[moved[0]]) - float(share[moved[0]])
        def protected(values):
            return float(np.bincount(np.ones(cells, dtype='int32'), weights=(values * yields * area).astype('float64'), minlength=2)[1])
        a, b = protected(share), protected(other)
        delta = step * float((yields[moved].astype('float64') * area[moved].astype('float64')).sum())
        bound = ep.protected_bound(delta, step, a, cells)
        assert abs(a - b) <= bound
        used = max(used, abs(a - b) / bound)
    assert 0.0 < used <= 1.0, 'the bound must be neither crossed nor vacuous at the scale of the real rows'
    assert ep.protected_bound(1e-8, 2.0 ** -24, 100.0, 21000) < 1e-6, 'far under any real error'


def test_the_share_and_mean_bounds_are_the_stated_formulas():
    ep = load('eps5_prediction')
    assert ep.share_bound(1e-8, 100.0, 0.5) == 1e-10 + 2 * ep.U64 * 0.5
    assert ep.mean_bound(4 * 2.0 ** -24, 1000, 21000, 0.5) == 4 * 2.0 ** -24 / 1000 + 2 * 21000 * ep.U64 * 0.5

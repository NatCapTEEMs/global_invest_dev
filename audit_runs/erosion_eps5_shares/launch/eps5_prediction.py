"""Whether a finished EPS-5 comparison stays inside the prediction on file.

The staged rasters fail the identity gate against the author's, and prediction/ says what that must do to the
shares. Each check reports two things apart. Location: the outputs have the expected structure and masks, and only
predicted cells, keys and columns differ. Magnitude: raster cells must equal their predicted value to the last bit,
and table values must lie within the bound in magnitude_bounds.csv, which run_eps5_prediction_bounds.py derives from
the predicted share change and the rounding of the task's own float32 and float64 arithmetic. That bound is no
tolerance on the science, but it is not a last-bit prediction either, so the result says which of the two it
verified. Any comparison whose structure cannot be read fails. Nothing here writes a file.
"""
import os

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

CELLS = 'sdr_reach_cells.csv'
EXPECTED = 'sdr_reach_expected_differences.csv'
BOUNDS = 'magnitude_bounds.csv'
DIAGNOSTIC_VALUE_COLUMN = 'mean_ps_onfarm_cropland_severe'
PARTS = {'onfarm': 'ps_onfarm_cropland_severe.tif', 'upstream': 'ps_upstream_cropland_severe.tif', 'combined': 'ps_combined_union_cropland_severe.tif'}
MAX_LISTED = 100000
U32, U64 = 2.0 ** -24, 2.0 ** -53
EXACT, NO_DIFFERENCE, WITHIN_BOUND = 'exact to the last bit', 'no difference', 'within the derived float bound'
OVER_BOUND, OFF_PREDICTION, NOT_EVALUATED = 'over the derived float bound', 'off the predicted value', 'not evaluated'
HELD = (EXACT, NO_DIFFERENCE, WITHIN_BOUND)


def read_prediction(directory):
    """The predicted cells, and the bound on each table value that may move at EPS-5, by key and column."""
    cells = pd.read_csv(os.path.join(directory, CELLS), encoding='utf-8-sig', float_precision='round_trip')
    bounds = pd.read_csv(os.path.join(directory, BOUNDS), dtype={'component': str, 'crop': str}, keep_default_na=False, encoding='utf-8-sig', float_precision='round_trip')
    return cells, {(r.component, r.ISO3, r.crop, r.column): float(r.bound) for r in bounds.itertuples()}


def predicted_raster(cells, part):
    """Every cell the prediction covers for one share, with its predicted value, and the ones that differ from the author's."""
    frame = pd.DataFrame({'row': cells['row'], 'col': cells['col'], 'library': cells['eps5_%s_predicted_library' % part], 'author': cells['eps5_%s_stored_author' % part]})
    return frame, frame[frame['library'] != frame['author']]


def raster_differences(left_path, right_path, block_rows=512):
    """Each cell where two share rasters on one grid differ, read in strips, listing at most MAX_LISTED."""
    rows, count = [], 0
    with rasterio.open(left_path) as a, rasterio.open(right_path) as b:
        for start in range(0, a.height, block_rows):
            window = Window(0, start, a.width, min(block_rows, a.height - start))
            x, y = a.read(1, window=window), b.read(1, window=window)
            hit = ~((x == y) | (np.isnan(x) & np.isnan(y)))
            r, c = np.nonzero(hit)
            count += r.size
            if len(rows) < MAX_LISTED:
                rows += [(int(i) + start, int(j), float(x[i, j]), float(y[i, j])) for i, j in zip(r, c)][:MAX_LISTED - len(rows)]
    return pd.DataFrame(rows, columns=['row', 'col', 'library', 'author']), count


def values_at(path, rows, cols):
    with rasterio.open(path) as d:
        band = d.read(1)
    return band[np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)].astype('float64')


def raster_layout(dataset):
    nodata = None if dataset.nodata is None or np.isnan(dataset.nodata) else dataset.nodata
    return dataset.crs, dataset.transform, dataset.shape, dataset.count, dataset.dtypes, nodata


def same_raster_layout(left_path, right_path):
    with rasterio.open(left_path) as a, rasterio.open(right_path) as b:
        return raster_layout(a) == raster_layout(b)


def protected_bound(delta, share_step, protected, country_cells):
    """The largest difference a predicted protected-production value can show, from the task's own arithmetic.

    The task computes share x yield x area in float32 and sums cells per country in float64. So two runs that differ
    only by the predicted share step on some cells differ by that step times their production, plus the rounding of two
    float32 products on each side of each cell, plus the float64 rounding of the two country sums.

    Args:
        delta (float): the predicted difference, the share step times the production on the moved cells.
        share_step (float): the share difference on those cells, equal across them.
        protected (float): the country-crop value, which sizes the float64 sum.
        country_cells (int): how many cells the sum runs over.

    Returns:
        float: the bound, in the units of the table.
    """
    production = abs(delta) / share_step
    return abs(delta) + (4 * U32 + 2 * U32 ** 2) * production + 2 * (country_cells - 1) * U64 * abs(protected)


def share_bound(protected_difference_bound, total, share):
    """The share is protected over an unchanged total, in float64."""
    return protected_difference_bound / total + 2 * U64 * abs(share)


def mean_bound(sum_of_share_steps, valued_cells_at_least, country_cells, mean):
    """A country's mean share: the moved share over the cells averaged, plus float64 rounding of the sum and the division."""
    return sum_of_share_steps / valued_cells_at_least + 2 * country_cells * U64 * abs(mean)


def result_row(item, structure, outside, magnitude, detail):
    location = not structure and not outside
    return {'item': item, 'as_predicted': location and magnitude in HELD, 'location_as_predicted': location, 'magnitude': magnitude, 'structure_problems': ', '.join(structure), 'detail': detail}


def check_raster(part, library_path, author_path, cells):
    """The differing cells are the predicted ones, and each covered cell holds its predicted value on both sides, to the last bit."""
    if not same_raster_layout(library_path, author_path):
        return result_row(PARTS[part], ['the two rasters differ in grid, type or nodata'], False, NOT_EVALUATED, 'layout differs')
    covered, moved = predicted_raster(cells, part)
    found, count = raster_differences(library_path, author_path)
    predicted_set, found_set = set(zip(moved['row'], moved['col'])), set(zip(found['row'], found['col']))
    wrong_library = int((values_at(library_path, covered['row'], covered['col']) != covered['library'].to_numpy()).sum())
    wrong_author = int((values_at(author_path, covered['row'], covered['col']) != covered['author'].to_numpy()).sum())
    outside = count - len(found_set & predicted_set)
    absent = len(predicted_set - found_set)
    magnitude = OFF_PREDICTION if wrong_library or wrong_author else NOT_EVALUATED if outside or absent else EXACT if count else NO_DIFFERENCE
    return result_row(PARTS[part], [], bool(outside or absent), magnitude, 'cells that differ %d, predicted %d, outside the prediction %d, predicted and absent %d, covered cells off the predicted library value %d, off the stored author value %d'
                      % (count, len(predicted_set), outside, absent, wrong_library, wrong_author))


def structure_problems(result):
    """What stops a table comparison being read against the prediction.

    compare_tables returns empty frames when it stops early on repeated keys or other columns, so an empty frame proves
    nothing until this list is empty.
    """
    summary, problems = result['summary'], []
    if summary.get('repeated_keys_left') or summary.get('repeated_keys_right'):
        problems.append('repeated keys %s library, %s author' % (summary.get('repeated_keys_left'), summary.get('repeated_keys_right')))
    if not summary.get('columns_identical'):
        problems.append('the columns differ')
    if 'keys_shared' not in summary:
        problems.append('the comparison stopped before reading its keys')
    if len(result['keys_one_side']):
        problems.append('%d keys on one side' % len(result['keys_one_side']))
    if 'keys_shared' in summary and not summary.get('row_order_identical'):
        problems.append('the row order differs')
    columns = result['columns']
    if len(columns):
        everything = columns[columns['group'] == 'all']
        for column, label in (('missing_mismatch', 'missing'), ('zero_mismatch', 'zero')):
            if everything[column].sum():
                problems.append('%d %s-mask changes' % (everything[column].sum(), label))
    return problems


def bound_check(rows, key_of, bounds):
    """How many differing values sit where no bound exists, how many exceed theirs, and the largest used share of a bound."""
    outside, over, largest = 0, 0, 0.0
    for r in rows.to_dict('records'):
        bound = bounds.get(key_of(r))
        gap = abs(r['left'] - r['right'])
        if bound is None:
            outside += 1
        elif not gap <= bound:
            over += 1
        elif bound > 0:
            largest = max(largest, gap / bound)
    return outside, over, largest


def magnitude_of(structure, rows, over, outside):
    """A value with no bound on file has no predicted magnitude, so the row is not reported as within one."""
    if over:
        return OVER_BOUND
    if structure or outside:
        return NOT_EVALUATED
    if not len(rows):
        return NO_DIFFERENCE
    return WITHIN_BOUND


def check_protected_table(name, result, bounds):
    """Readable structure, unchanged masks, differences only on predicted rows and columns, each within its derived bound."""
    structure, rows = structure_problems(result), result['rows_that_differ']
    outside, over, largest = bound_check(rows, lambda r: (r['component'], r['ISO3'], r['crop'], r['column']), bounds) if len(rows) and not structure else (0, 0, 0.0)
    return result_row(name, structure, bool(outside or len(result['text_differences'])), magnitude_of(structure, rows, over, outside),
                      'text cells %d, values that differ %d, outside the predicted rows %d, over their bound %d, largest fraction of a bound used %.3g' % (len(result['text_differences']), len(rows), outside, over, largest))


def text_rows(frame):
    return sorted(tuple(str(v) for v in row) for row in frame.itertuples(index=False))


def check_named_text(name, result, baseline_path, bounds=None):
    """Text differences equal the baseline, and values differ only where a bound is on file, within it."""
    baseline = pd.read_csv(baseline_path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    structure, rows = structure_problems(result), result['rows_that_differ']
    same_text = text_rows(result['text_differences']) == text_rows(baseline)
    outside, over, largest = bound_check(rows, lambda r: ('', r['ISO3'], '', r['column']), bounds or {}) if len(rows) and not structure else (0, 0, 0.0)
    return result_row(name, structure, bool(outside) or not same_text, magnitude_of(structure, rows, over, outside),
                      'text cells %d, equal to the baseline %s, values that differ %d, outside the prediction %d, over their bound %d, largest fraction of a bound used %.3g' % (len(result['text_differences']), same_text, len(rows), outside, over, largest))


def check_baseline_values(name, result, baseline_path):
    """Every differing value equals the baseline's, key, column and both sides, to the last bit."""
    baseline = pd.read_csv(baseline_path, encoding='utf-8-sig', float_precision='round_trip')
    structure, found = structure_problems(result), result['rows_that_differ']
    columns = ['band', 'crop', 'column', 'left', 'right']

    def keyed(frame):
        return sorted((str(a), str(b), str(c), float(d), float(e)) for a, b, c, d, e in frame[columns].itertuples(index=False)) if len(frame) else []
    same = keyed(found) == keyed(baseline)
    magnitude = NOT_EVALUATED if structure else EXACT if same else OFF_PREDICTION
    return result_row(name, structure, bool(len(result['text_differences'])) or not same, magnitude, 'values that differ %d, baseline %d, equal %s' % (len(found), len(baseline), same))


def check_clean(name, result, clean):
    """No value may differ, with a readable structure."""
    structure = structure_problems(result)
    ok = result['verdict'] in clean
    return result_row(name, structure, not ok, NOT_EVALUATED if structure else NO_DIFFERENCE if ok else OFF_PREDICTION, 'verdict %s' % result['verdict'])

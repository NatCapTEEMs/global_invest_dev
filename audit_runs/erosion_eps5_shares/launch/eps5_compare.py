"""Exact comparison of two prevention-share outputs, with no tolerance.

Every number is read as Python reads a decimal string, so equal text gives equal values and a
difference in the last bit is reported as a difference. Masks are compared before values: a row or
cell missing on one side, or zero on one side, is counted on its own and not folded into a
magnitude. Nothing here writes a file or decides what a difference means.
"""
import hashlib

import numpy as np
import pandas as pd

LONG_KEYS = ['component', 'ISO3', 'crop']
BYTE_IDENTICAL, VALUES_IDENTICAL, DIFFERENT = 'BYTE_IDENTICAL', 'VALUES_IDENTICAL', 'DIFFERENT'
MISSING_TEXT = {'', 'nan', 'NaN', 'NA', 'None'}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_text_table(path):
    """Every cell as its text, so a label such as NA stays a label and a number keeps its digits."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')


def numeric_values(cells):
    """A float array with NaN for missing cells, or None if any other cell is not a number."""
    out = np.full(len(cells), np.nan)
    for i, cell in enumerate(cells):
        if cell in MISSING_TEXT:
            continue
        try:
            out[i] = float(cell)
        except ValueError:
            return None
    return out


def compare_column(left, right):
    """Counts and magnitudes for one numeric column on rows already aligned."""
    both = ~np.isnan(left) & ~np.isnan(right)
    gap = np.abs(left[both] - right[both])
    with np.errstate(divide='ignore', invalid='ignore'):
        relative = np.where(right[both] != 0, gap / np.abs(right[both]), np.where(gap == 0, 0.0, np.inf))
    return {'n_rows': len(left), 'n_exact_equal': int((left[both] == right[both]).sum() + (np.isnan(left) & np.isnan(right)).sum()),
            'missing_left': int(np.isnan(left).sum()), 'missing_right': int(np.isnan(right).sum()),
            'missing_mismatch': int((np.isnan(left) != np.isnan(right)).sum()),
            'zero_left': int((left == 0).sum()), 'zero_right': int((right == 0).sum()),
            'zero_mismatch': int((((left == 0) != (right == 0)) & both).sum()),
            'above_zero_left': int((left > 0).sum()), 'above_zero_right': int((right > 0).sum()),
            'sum_left': float(np.nansum(left)), 'sum_right': float(np.nansum(right)),
            'max_abs_difference': float(gap.max()) if gap.size else 0.0,
            'max_relative_difference': float(relative.max()) if relative.size else 0.0}


def compare_tables(left_path, right_path, keys, group_by=None):
    """Compare two tables on their keys, every other column, and masks.

    Args:
        left_path (str): one table.
        right_path (str): the other.
        keys (list): the columns that name a row. A repeated key stops the value comparison and is reported.
        group_by (str): a key column to break the numeric summary down by, as well as over all rows.

    Returns:
        dict: `verdict`, `summary` (facts about the two files), `columns` (pd.DataFrame, numeric
        columns), `text_differences`, `keys_one_side` and `rows_that_differ` (each a pd.DataFrame).
    """
    left, right = read_text_table(left_path), read_text_table(right_path)
    summary = {'sha256_left': sha256_file(left_path), 'sha256_right': sha256_file(right_path), 'rows_left': len(left), 'rows_right': len(right),
               'columns_identical': list(left.columns) == list(right.columns)}
    summary['bytes_identical'] = summary['sha256_left'] == summary['sha256_right']
    empty = pd.DataFrame()
    repeats = int(left.duplicated(keys).sum()), int(right.duplicated(keys).sum())
    summary['repeated_keys_left'], summary['repeated_keys_right'] = repeats
    if any(repeats) or not summary['columns_identical']:
        return {'verdict': DIFFERENT, 'summary': summary, 'columns': empty, 'text_differences': empty, 'keys_one_side': empty, 'rows_that_differ': empty}
    merged = left.merge(right, on=keys, how='outer', suffixes=('__left', '__right'), indicator=True)
    one_side = merged.loc[merged['_merge'] != 'both', keys + ['_merge']].rename(columns={'_merge': 'side'})
    one_side['side'] = one_side['side'].map({'left_only': 'left only', 'right_only': 'right only'}).astype(str)
    shared = merged[merged['_merge'] == 'both'].reset_index(drop=True)
    summary['keys_left_only'] = int((one_side['side'] == 'left only').sum())
    summary['keys_right_only'] = int((one_side['side'] == 'right only').sum())
    summary['keys_shared'] = len(shared)
    summary['row_order_identical'] = left[keys].equals(right[keys])
    groups = [('all', np.ones(len(shared), dtype=bool))]
    if group_by:
        groups += [(g, (shared[group_by] == g).to_numpy()) for g in sorted(shared[group_by].unique())]
    summary_rows, text_rows, value_rows = [], [], []
    for column in [c for c in left.columns if c not in keys]:
        a_cells, b_cells = shared[column + '__left'].to_numpy(), shared[column + '__right'].to_numpy()
        a, b = numeric_values(a_cells), numeric_values(b_cells)
        if a is None or b is None:
            for i in np.flatnonzero(a_cells != b_cells):
                text_rows.append(dict({k: shared.loc[i, k] for k in keys}, column=column, left=a_cells[i], right=b_cells[i]))
            continue
        for name, mask in groups:
            summary_rows.append(dict({'group': name, 'column': column}, **compare_column(a[mask], b[mask])))
        differs = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        for i in np.flatnonzero(differs):
            value_rows.append(dict({k: shared.loc[i, k] for k in keys}, column=column, left=a[i], right=b[i], left_minus_right=a[i] - b[i]))
    columns = pd.DataFrame(summary_rows)
    text_differences, rows_that_differ = pd.DataFrame(text_rows), pd.DataFrame(value_rows)
    clean = not (len(one_side) or len(text_differences) or len(rows_that_differ))
    verdict = BYTE_IDENTICAL if summary['bytes_identical'] else VALUES_IDENTICAL if clean else DIFFERENT
    return {'verdict': verdict, 'summary': summary, 'columns': columns, 'text_differences': text_differences,
            'keys_one_side': one_side.reset_index(drop=True), 'rows_that_differ': rows_that_differ}


def compare_long(left_path, right_path):
    """The country, crop and component comparison, broken down by component."""
    return compare_tables(left_path, right_path, LONG_KEYS, group_by='component')


def raster_grid(dataset):
    return {'crs': dataset.crs.to_wkt() if dataset.crs else None, 'width': dataset.width, 'height': dataset.height,
            'bands': dataset.count, 'transform': tuple(dataset.transform)[:6]}


def valid_cells(values, nodata):
    valid = np.isfinite(values)
    if nodata is not None and not np.isnan(nodata):
        valid &= values != nodata
    return valid


def compare_rasters(left_path, right_path, block_rows=512):
    """Cell by cell on one band, read in strips so the memory stays one strip of each raster.

    Stored numbers are compared as float64, which holds float32 exactly. A different nodata value
    alone is not a difference in what the raster says, so it is reported and the valid masks decide.
    A different grid stops the comparison and is the verdict.
    """
    import rasterio
    from rasterio.windows import Window
    out = {'sha256_left': sha256_file(left_path), 'sha256_right': sha256_file(right_path)}
    out['bytes_identical'] = out['sha256_left'] == out['sha256_right']
    with rasterio.open(left_path) as a, rasterio.open(right_path) as b:
        out.update(width=a.width, height=a.height, data_type_left=a.dtypes[0], data_type_right=b.dtypes[0], nodata_left=a.nodata, nodata_right=b.nodata,
                   compression_left=a.compression.value if a.compression else None, compression_right=b.compression.value if b.compression else None)
        if raster_grid(a) != raster_grid(b):
            out.update(verdict='GRID_DIFFERS', grid_left=str(raster_grid(a)), grid_right=str(raster_grid(b)))
            return out
        counts = dict.fromkeys(['cells', 'valid_left', 'valid_right', 'valid_mismatch', 'zero_left', 'zero_right', 'zero_mismatch', 'values_differ'], 0)
        largest, sum_left, sum_right = 0.0, 0.0, 0.0
        for row in range(0, a.height, block_rows):
            window = Window(0, row, a.width, min(block_rows, a.height - row))
            x, y = a.read(1, window=window).astype('float64'), b.read(1, window=window).astype('float64')
            vx, vy = valid_cells(x, a.nodata), valid_cells(y, b.nodata)
            both = vx & vy
            counts['cells'] += x.size
            counts['valid_left'] += int(vx.sum())
            counts['valid_right'] += int(vy.sum())
            counts['valid_mismatch'] += int((vx != vy).sum())
            counts['zero_left'] += int((vx & (x == 0)).sum())
            counts['zero_right'] += int((vy & (y == 0)).sum())
            counts['zero_mismatch'] += int((both & ((x == 0) != (y == 0))).sum())
            counts['values_differ'] += int((both & (x != y)).sum())
            if both.any():
                largest = max(largest, float(np.abs(x[both] - y[both]).max()))
            sum_left += float(x[vx].sum())
            sum_right += float(y[vy].sum())
    clean = not (counts['valid_mismatch'] or counts['values_differ'])
    out.update(counts, max_abs_difference=largest, sum_left=sum_left, sum_right=sum_right,
               verdict=BYTE_IDENTICAL if out['bytes_identical'] else VALUES_IDENTICAL if clean else DIFFERENT)
    return out

"""Rebuild the SPAM 2020 harvested-area and yield stacks from the raw V2r0 GeoTIFFs, without resampling.

The supplied stacks (spam2020_*_stack_TA.tif) were built by average-resampling each raw file onto the
grid of the first harvested-area file, which put values into NoData cells for the 20 crops whose
raw grids differ from that grid by a sliver, and divided a yield already in t/ha by 1000. These
functions copy each raw array onto the reference grid as it is, so every valid cell keeps its raw
float32 value and every NoData cell stays NoData (-9999), and keep yield in t/ha.

The functions take explicit paths and open nothing on import. The run files beside the code folder
(run_spam_stack_preflight.py, run_spam_stack_manifest.py, run_spam_stack_repair.py) call them with
the paths in spam_stack_repair_parameters.csv, each with no arguments.

Every input the build and the check read is read once into memory, hashed, compared with the
pinned manifest and decoded from those same bytes, so a file that changes on disk cannot be read
unverified. Every mismatch stops the run: a different size, CRS, rotation or data type, a grid
displaced by more than MAX_DISPLACEMENT_M anywhere, a negative value, a valid value equal to the
stack NoData, a missing or unlisted file, or an input whose SHA-256 differs from the manifest.
"""
import contextlib
import csv
import datetime
import hashlib
import io
import math
import os
import platform
import shutil

import numpy as np
import rasterio
from rasterio.io import MemoryFile

STACK_NODATA = np.float32(-9999.0)
# Largest allowed distance between a raw cell centre and the same cell's centre on the reference
# grid, anywhere on the grid. A SPAM 2020 cell is about 9.3 km at the equator, so 1 m is 1e-4 of a
# cell. The largest displacement among the raw files is 0.033 m.
MAX_DISPLACEMENT_M = 1.0
# Metres per degree on the WGS 84 equator, an upper bound for longitude and within 0.4% for latitude.
M_PER_DEG = math.pi / 180.0 * 6378137.0
YIELD_DIVISOR = np.float32(1000.0)
# The two units the raw yield could be in, as production (t) per harvested area (ha) times yield.
UNIT_SCALES = {'t_per_ha': 1.0, 'kg_per_ha': 1e-3}
# Each raw file stores values to a fixed number of decimals (one, in every file read so far). The
# check measures it per file, up to this many, and a file with no fixed precision is unclassifiable.
MAX_DECIMALS = 6
RAW_DIRS = {
    'H': os.path.join('spam2020V2r0_global_harvested_area.geotiff', 'spam2020V2r0_global_harvested_area'),
    'Y': os.path.join('spam2020V2r0_global_yield.geotiff', 'spam2020V2r0_global_yield'),
    'P': os.path.join('spam2020V2r0_global_production.geotiff', 'spam2020V2r0_global_production'),
}
OUTPUT_FILES = {'H': 'spam2020_harvested_area_stack_TA_repaired_20261103.tif',
                'Y': 'spam2020_yield_stack_TA_repaired_20261103.tif'}
OUTPUT_HASHES = 'outputs_sha256.txt'
UNITS = {'H': 'ha', 'Y': 't/ha'}
SOURCE = 'IFPRI SPAM 2020 V2r0 global GeoTIFFs, all technologies (_A)'
METHOD = 'raw arrays copied onto the reference grid without resampling, NoData to -9999, values unchanged'


class Mismatch(Exception):
    """An input that does not match what the stacks require. The run stops on the first one."""


def read_parameters(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return {r['parameter']: r['value'] for r in csv.DictReader(f)}


def raw_path(raw_root, var, crop):
    return os.path.join(raw_root, RAW_DIRS[var], 'spam2020_V2r0_global_%s_%s_A.tif' % (var, crop.upper()))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_manifest(path):
    with open(path, newline='', encoding='utf-8') as f:
        return {os.path.abspath(r['path']): r['sha256'] for r in csv.DictReader(f)}


def read_verified(manifest, path):
    """The file's bytes, after their SHA-256 matches the manifest. Stops otherwise."""
    key = os.path.abspath(path)
    if key not in manifest:
        raise Mismatch('%s is not in the pinned manifest' % path)
    if not os.path.exists(path):
        raise Mismatch('%s is missing' % path)
    with open(path, 'rb') as f:
        data = f.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != manifest[key]:
        raise Mismatch('%s has SHA-256 %s, the manifest says %s' % (path, digest, manifest[key]))
    return data, digest


@contextlib.contextmanager
def open_verified(manifest, path):
    """A raster decoded from the same bytes that were hashed, and their SHA-256."""
    data, digest = read_verified(manifest, path)
    with MemoryFile(data) as mem, mem.open() as src:
        yield src, digest


def read_bandmap(data, label):
    """Crops in band order, from the band map's bytes. Bands must run 1..n with no gap and no crop twice."""
    rows = list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))
    bands = [int(r['band']) for r in rows]
    crops = [r['crop'].strip().lower() for r in rows]
    if bands != list(range(1, len(rows) + 1)):
        raise Mismatch('%s: bands are %s, not 1..%d in order' % (label, bands, len(rows)))
    if len(set(crops)) != len(crops):
        raise Mismatch('%s: a crop appears twice' % label)
    return crops


def grid_displacement_m(transform, reference, width, height):
    """Largest distance in metres between a cell centre on `transform` and on `reference`.

    The difference of two affine maps is affine, so its largest value over the grid is at one of
    the four corner cell centres. It is formed from the coefficient differences, not by subtracting
    two coordinates near 180 degrees, which would lose about 1e-7 of a centimetre-sized result.
    """
    t, r = transform, reference
    worst = 0.0
    for col in (0.5, width - 0.5):
        for row in (0.5, height - 0.5):
            dx = (t.a - r.a) * col + (t.b - r.b) * row + (t.c - r.c)
            dy = (t.d - r.d) * col + (t.e - r.e) * row + (t.f - r.f)
            worst = max(worst, math.hypot(dx * M_PER_DEG, dy * M_PER_DEG))
    return worst


def check_grid(src, reference, label):
    """Stop unless `src` is a one-band float32 raster on the reference grid within the bound."""
    t, r = src.transform, reference['transform']
    problems = []
    if (src.width, src.height) != (reference['width'], reference['height']):
        problems.append('size %dx%d, reference %dx%d' % (src.width, src.height, reference['width'], reference['height']))
    if src.crs is None or src.crs.to_epsg() != reference['crs'].to_epsg():
        problems.append('CRS %s, reference EPSG:%s' % (src.crs, reference['crs'].to_epsg()))
    if t.b != 0.0 or t.d != 0.0:
        problems.append('rotated transform %r' % (t,))
    if src.count != 1:
        problems.append('%d bands' % src.count)
    if src.dtypes[0] != 'float32':
        problems.append('data type %s' % src.dtypes[0])
    if problems:
        raise Mismatch('%s: %s' % (label, '; '.join(problems)))
    displacement = grid_displacement_m(t, r, src.width, src.height)
    if not displacement <= MAX_DISPLACEMENT_M:
        raise Mismatch('%s: cell centres move up to %.6g m from the reference grid, more than %g m'
                       % (label, displacement, MAX_DISPLACEMENT_M))
    return {'cell_size_x_deg': t.a, 'cell_size_y_deg': t.e, 'origin_x_deg': t.c, 'origin_y_deg': t.f,
            'cell_size_x_minus_reference_deg': t.a - r.a, 'cell_size_y_minus_reference_deg': t.e - r.e,
            'max_displacement_m': displacement, 'nodata': src.nodata}


def valid_mask(array, nodata):
    valid = np.isfinite(array)
    if nodata is not None and np.isfinite(nodata):
        valid &= array != nodata
    return valid


def raw_array(src, reference, label):
    """The raw band, its valid mask and grid record, after the grid and value checks."""
    grid = check_grid(src, reference, label)
    array = src.read(1)
    valid = valid_mask(array, src.nodata)
    if (array[valid] < 0).any():
        raise Mismatch('%s: %d negative values' % (label, int((array[valid] < 0).sum())))
    if (array[valid] == STACK_NODATA).any():
        raise Mismatch('%s: a valid value equals the stack NoData %g' % (label, STACK_NODATA))
    return array, valid, grid


def reference_profile(src, label):
    profile = src.profile.copy()
    if profile['dtype'] != 'float32' or profile['nodata'] != STACK_NODATA:
        raise Mismatch('%s: reference stack is %s with NoData %s, not float32 with %g'
                       % (label, profile['dtype'], profile['nodata'], STACK_NODATA))
    return profile


def decimal_places(values):
    """The fewest decimals k <= MAX_DECIMALS such that every value is the float32 of a multiple of 10^-k."""
    v64 = values.astype('float64')
    for k in range(MAX_DECIMALS + 1):
        if np.array_equal((np.round(v64 * 10.0 ** k) / 10.0 ** k).astype(np.float32), values):
            return k
    return None


def unit_evidence(h, hv, y, yv, p, pv):
    """Which unit the raw yield is in, from production = area x yield cell by cell.

    Each stored value is the float32 of a number rounded to the file's measured decimals, so the
    true value lies within half a step plus half a float32 spacing of it. A cell is consistent with
    a unit when the production interval meets the interval of area times yield times the unit's
    scale. The crop is in a unit when every cell is consistent with it and at least one cell is not
    consistent with the other unit. Anything else, including a crop whose production is 1000 times
    area times yield, a file with no fixed precision or no cell to test, is 'unknown'. No share of
    cells is allowed to disagree.
    """
    both = hv & yv & pv
    row = {'unit_cells': int(both.sum())}
    arrays = {'H': h[both], 'Y': y[both], 'P': p[both]}
    decimals = {var: decimal_places(a) for var, a in arrays.items()}
    row.update({'decimals_%s' % var: ('' if k is None else k) for var, k in decimals.items()})
    hy = arrays['H'].astype('float64') * arrays['Y'].astype('float64')
    ratio = arrays['P'][hy > 0].astype('float64') / hy[hy > 0]
    row['p_over_hy_median'] = float(np.median(ratio)) if ratio.size else float('nan')
    row['p_total_over_hy_total'] = float(arrays['P'].astype('float64').sum() / hy.sum()) if hy.sum() > 0 else float('nan')
    if None in decimals.values() or not both.any():
        row.update({'cells_consistent_%s' % u: '' for u in UNIT_SCALES})
        row['yield_unit'] = 'unknown'
        return row
    bounds = {}
    for var, a in arrays.items():
        a64 = a.astype('float64')
        half = 0.5 * 10.0 ** -decimals[var] + np.spacing(np.abs(a)).astype('float64') / 2.0
        bounds[var] = (np.maximum(a64 - half, 0.0), a64 + half)
    (h_lo, h_hi), (y_lo, y_hi), (p_lo, p_hi) = bounds['H'], bounds['Y'], bounds['P']
    consistent = {u: (p_lo <= s * h_hi * y_hi) & (p_hi >= s * h_lo * y_lo) for u, s in UNIT_SCALES.items()}
    n = row['unit_cells']
    row.update({'cells_consistent_%s' % u: int(c.sum()) for u, c in consistent.items()})
    unit = 'unknown'
    for u, other in (('t_per_ha', 'kg_per_ha'), ('kg_per_ha', 't_per_ha')):
        if row['cells_consistent_%s' % u] == n and row['cells_consistent_%s' % other] < n:
            unit = u
    row['yield_unit'] = unit
    return row


def write_csv(path, rows, columns=None):
    columns = columns or list(rows[0])
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def input_list(raw_root, bandmap_path, area_stack_path, yield_stack_path, crops):
    paths = [('band_map', '', '', bandmap_path), ('reference_stack', '', '', area_stack_path),
             ('old_yield_stack', '', '', yield_stack_path)]
    for crop in crops:
        for var in ('H', 'Y', 'P'):
            paths.append(('raw', crop, var, raw_path(raw_root, var, crop)))
    return paths


def make_manifest(raw_root, bandmap_path, area_stack_path, yield_stack_path, out_path):
    """SHA-256, size and mtime of every input. Refuses to replace an existing manifest."""
    if os.path.exists(out_path):
        raise Mismatch('%s already exists. A pinned manifest is never replaced: move it aside first.' % out_path)
    with open(bandmap_path, 'rb') as f:
        crops = read_bandmap(f.read(), bandmap_path)
    rows = []
    for role, crop, var, path in input_list(raw_root, bandmap_path, area_stack_path, yield_stack_path, crops):
        st = os.stat(path)
        rows.append({'role': role, 'crop': crop, 'var': var, 'path': os.path.abspath(path), 'bytes': st.st_size,
                     'mtime_utc': datetime.datetime.fromtimestamp(st.st_mtime, datetime.timezone.utc).isoformat(),
                     'sha256': sha256_file(path)})
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    write_csv(out_path, rows)
    print('manifest: %d inputs to %s' % (len(rows), out_path))
    return 0


def preflight(raw_root, bandmap_path, area_stack_path, yield_stack_path, out_dir, report_path, grid_table, expect_crops):
    """Headers only, for a login node. Writes a PASS/FAIL line per check and the raw grid table."""
    if os.path.exists(report_path):
        raise Mismatch('%s already exists: move it aside to rerun the preflight' % report_path)
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    lines, failures = [], []

    def report(ok, text):
        lines.append('%s  %s' % ('PASS' if ok else 'FAIL', text))
        if not ok:
            failures.append(text)

    with open(bandmap_path, 'rb') as f:
        crops = read_bandmap(f.read(), bandmap_path)
    report(len(crops) == expect_crops, 'band map has %d crops, %d expected' % (len(crops), expect_crops))
    with rasterio.open(area_stack_path) as ref:
        reference = reference_profile(ref, area_stack_path)
    report(reference['count'] == len(crops), 'reference stack has %d bands' % reference['count'])
    with rasterio.open(yield_stack_path) as old:
        same = (old.transform == reference['transform'] and old.crs == reference['crs']
                and (old.width, old.height, old.count) == (reference['width'], reference['height'], reference['count']))
    report(same, 'supplied yield stack on the reference grid with %d bands' % reference['count'])
    rows = []
    for role, crop, var, path in input_list(raw_root, bandmap_path, area_stack_path, yield_stack_path, crops):
        if role != 'raw':
            continue
        if not os.path.exists(path):
            report(False, 'missing %s' % path)
            continue
        try:
            with rasterio.open(path) as src:
                grid = check_grid(src, reference, os.path.basename(path))
            rows.append({'crop': crop, 'var': var, 'file': os.path.basename(path), 'status': 'ok', **grid})
        except Mismatch as e:
            report(False, str(e))
            rows.append({'crop': crop, 'var': var, 'file': os.path.basename(path), 'status': str(e)})
    ok = [r for r in rows if r['status'] == 'ok']
    report(len(ok) == 3 * len(crops), '%d of %d raw files on the reference grid within %g m'
           % (len(ok), 3 * len(crops), MAX_DISPLACEMENT_M))
    if ok:
        worst = max(ok, key=lambda r: r['max_displacement_m'])
        lines.append('      largest displacement %.6g m (%s)' % (worst['max_displacement_m'], worst['file']))
        columns = ['crop', 'var', 'file', 'status'] + [k for k in ok[0] if k not in ('crop', 'var', 'file', 'status')]
        write_csv(grid_table, rows, columns)
    for path in (out_dir, out_dir.rstrip(os.sep) + '.partial'):
        report(not os.path.exists(path), 'output path absent: %s' % path)
    parent = os.path.dirname(os.path.abspath(out_dir))
    free = shutil.disk_usage(parent).free
    report(free > 2e9, '%.1f GB free under %s' % (free / 1e9, parent))
    lines.append('      python %s, numpy %s, rasterio %s, GDAL %s' % (platform.python_version(), np.__version__,
                                                                    rasterio.__version__, rasterio.__gdal_version__))
    lines.append('preflight: %s' % ('FAIL, %d checks' % len(failures) if failures else 'all checks pass'))
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 1 if failures else 0


def build(raw_root, bandmap_path, area_stack_path, manifest_path, out_dir):
    """Write the two stacks into out_dir, through out_dir.partial, from manifest-verified bytes.

    A first pass validates every raw file (hash, grid, values) and writes nothing. Only then is
    out_dir.partial created and filled, and it is renamed to out_dir when both stacks, the receipt
    and the output hashes are written.
    """
    out_dir = os.path.normpath(out_dir)
    partial = out_dir + '.partial'
    for path in (out_dir, partial):
        if os.path.exists(path):
            raise Mismatch('%s already exists. The build writes only into an empty, new directory.' % path)
    manifest = read_manifest(manifest_path)
    crops = read_bandmap(read_verified(manifest, bandmap_path)[0], bandmap_path)
    with open_verified(manifest, area_stack_path) as (ref, _):
        profile = reference_profile(ref, area_stack_path)
    if profile['count'] != len(crops):
        raise Mismatch('reference stack has %d bands, the band map %d crops' % (profile['count'], len(crops)))
    # Every input is validated in full before anything is written, so a stop leaves no output behind.
    # The write pass then reads each file again from freshly verified bytes.
    for crop in crops:
        for var in OUTPUT_FILES:
            path = raw_path(raw_root, var, crop)
            with open_verified(manifest, path) as (src, _):
                raw_array(src, profile, os.path.basename(path))
    os.makedirs(partial)
    receipt = []
    targets = {var: os.path.join(partial, name) for var, name in OUTPUT_FILES.items()}
    datasets = {var: rasterio.open(path, 'w', **profile) for var, path in targets.items()}
    try:
        for var, dst in datasets.items():
            dst.update_tags(UNITS=UNITS[var], SOURCE=SOURCE, METHOD=METHOD)
        for band, crop in enumerate(crops, start=1):
            for var, dst in datasets.items():
                path = raw_path(raw_root, var, crop)
                with open_verified(manifest, path) as (src, digest):
                    array, valid, grid = raw_array(src, profile, os.path.basename(path))
                out = np.where(valid, array, STACK_NODATA).astype(np.float32, copy=False)
                dst.write(out, band)
                dst.set_band_description(band, crop)
                dst.update_tags(band, source_file=os.path.basename(path), source_sha256=digest, units=UNITS[var])
                receipt.append({'band': band, 'crop': crop, 'var': var, 'source_path': os.path.abspath(path),
                                'source_sha256': digest, 'valid_cells': int(valid.sum()),
                                'zero_cells': int((valid & (array == 0)).sum()),
                                'max_displacement_m': grid['max_displacement_m']})
            print('band %2d %s' % (band, crop), flush=True)
    finally:
        for dst in datasets.values():
            dst.close()
    write_csv(os.path.join(partial, 'build_receipt.csv'), receipt)
    with open(os.path.join(partial, OUTPUT_HASHES), 'w', encoding='utf-8') as f:
        for var, path in targets.items():
            f.write('%s  %s\n' % (sha256_file(path), os.path.basename(path)))
    os.rename(partial, out_dir)
    print('build: %d bands into %s' % (len(crops), out_dir))
    return 0


def bits(array):
    return array.view(np.uint32)


def check(raw_root, bandmap_path, area_stack_path, yield_stack_path, manifest_path, built_dir, out_csv):
    """Every band against its raw files, all read from manifest-verified bytes. 1 on any failure."""
    if os.path.exists(out_csv):
        raise Mismatch('%s already exists: the check writes a new table only' % out_csv)
    manifest = read_manifest(manifest_path)
    crops = read_bandmap(read_verified(manifest, bandmap_path)[0], bandmap_path)
    with open(os.path.join(built_dir, OUTPUT_HASHES), encoding='utf-8') as f:
        built_hashes = {os.path.abspath(os.path.join(built_dir, name)): digest
                        for digest, name in (line.split() for line in f if line.strip())}
    rows, failures = [], []
    with contextlib.ExitStack() as stack:
        ref, _ = stack.enter_context(open_verified(manifest, area_stack_path))
        reference = reference_profile(ref, area_stack_path)
        built = {var: stack.enter_context(open_verified(built_hashes, os.path.join(built_dir, name)))[0]
                 for var, name in OUTPUT_FILES.items()}
        old = {'H': ref, 'Y': stack.enter_context(open_verified(manifest, yield_stack_path))[0]}
        for var, ds in built.items():
            same = (ds.transform == reference['transform'] and ds.crs == reference['crs'] and ds.count == len(crops)
                    and (ds.width, ds.height) == (reference['width'], reference['height'])
                    and ds.dtypes[0] == 'float32' and ds.nodata == STACK_NODATA)
            if not same:
                failures.append('%s: not on the reference grid, profile or band count' % OUTPUT_FILES[var])
            if ds.tags().get('UNITS') != UNITS[var]:
                failures.append('%s: UNITS tag %r' % (OUTPUT_FILES[var], ds.tags().get('UNITS')))
        for band, crop in enumerate(crops, start=1):
            row = {'band': band, 'crop': crop}
            raw = {}
            for var in ('H', 'Y', 'P'):
                path = raw_path(raw_root, var, crop)
                with open_verified(manifest, path) as (src, _):
                    raw[var] = raw_array(src, reference, os.path.basename(path))
                row['max_displacement_m_%s' % var] = raw[var][2]['max_displacement_m']
            for var in ('H', 'Y'):
                array, valid, _ = raw[var]
                new = built[var].read(band)
                row['raw_valid_%s' % var] = int(valid.sum())
                row['mask_exact_%s' % var] = bool(np.array_equal(new != STACK_NODATA, valid))
                row['values_exact_%s' % var] = bool(np.array_equal(bits(new[valid]), bits(array[valid])))
                if built[var].descriptions[band - 1] != crop:
                    failures.append('band %d description %r, crop %s' % (band, built[var].descriptions[band - 1], crop))
            (h, hv, _), (y, yv, _), (p, pv, _) = raw['H'], raw['Y'], raw['P']
            row.update(unit_evidence(h, hv, y, yv, p, pv))
            # The unit factor alone predicts supplied yield = float32(raw) / float32(1000) and supplied
            # area = raw, bit for bit, on every cell valid in both. Anything else is spill or blending.
            for var in ('H', 'Y'):
                array, valid, _ = raw[var]
                o = old[var].read(band)
                ov = o != STACK_NODATA
                common = valid & ov
                expected = array[common] / YIELD_DIVISOR if var == 'Y' else array[common]
                exact = bits(o[common]) == bits(expected.astype(np.float32, copy=False))
                row['old_cells_valid_not_raw_%s' % var] = int((ov & ~valid).sum())
                row['raw_cells_valid_not_old_%s' % var] = int((valid & ~ov).sum())
                row['old_cells_off_unit_factor_%s' % var] = int((~exact).sum())
                row['old_total_%s' % var] = float(o[ov].astype('float64').sum())
                row['raw_total_%s' % var] = float(array[valid].astype('float64').sum())
            unit_only = all(row['old_cells_valid_not_raw_%s' % v] == 0 and row['raw_cells_valid_not_old_%s' % v] == 0
                            and row['old_cells_off_unit_factor_%s' % v] == 0 for v in ('H', 'Y'))
            row['old_stack_differs_by'] = 'unit factor only' if unit_only else 'unit factor, spill and blending'
            passed = all(row[k] for k in ('mask_exact_H', 'values_exact_H', 'mask_exact_Y', 'values_exact_Y')) \
                and row['yield_unit'] == 't_per_ha'
            row['passed'] = passed
            if not passed:
                failures.append('band %d %s failed' % (band, crop))
            rows.append(row)
            print('band %2d %-4s %s' % (band, crop, 'pass' if passed else 'FAIL'), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    write_csv(out_csv, rows)
    offset = [r['crop'] for r in rows if max(r['max_displacement_m_H'], r['max_displacement_m_Y']) > 0]
    spilled = [r['crop'] for r in rows if r['old_stack_differs_by'] != 'unit factor only']
    print('supplied stacks: unit factor only in %d crops, spill or blending in %d (%s)'
          % (len(rows) - len(spilled), len(spilled), ' '.join(spilled)))
    print('raw grids off the reference grid: %d (%s)' % (len(offset), ' '.join(offset)))
    print('check: %s' % ('FAIL: ' + '; '.join(failures) if failures else 'all %d bands pass' % len(rows)))
    return 1 if failures else 0

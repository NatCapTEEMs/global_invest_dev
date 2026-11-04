"""Tests for the SPAM 2020 stack repair on a small synthetic raw tree.

Covers the exact copy, every stop, the manifest-verified reads in the build and the check, the
unit classifier against t/ha, kg/ha, a 1000 scale and an unfixed precision, the separation of the
unit factor from spill, and the three run files running with no arguments.

Run: python -I -m pytest code/test_spam_stack_repair.py -q -p no:cacheprovider
"""
import csv
import importlib.util
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location('spam_stack_repair', os.path.join(HERE, 'spam_stack_repair.py'))
ssr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ssr)

CROPS = ['bana', 'vege', 'yams']
WIDTH, HEIGHT = 12, 6
REFERENCE = Affine(30.0, 0.0, -180.0, 0.0, -30.0, 90.0)
RUN_FILES = ['run_spam_stack_preflight.py', 'run_spam_stack_manifest.py', 'run_spam_stack_repair.py']


def write_raster(path, array, transform=REFERENCE, crs='EPSG:4326', nodata=float('nan')):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bands = array if array.ndim == 3 else array[None]
    profile = dict(driver='GTiff', dtype=str(bands.dtype), width=bands.shape[2], height=bands.shape[1],
                   count=bands.shape[0], crs=crs, transform=transform, nodata=nodata,
                   compress='deflate', tiled=False, blockxsize=bands.shape[2], blockysize=1, interleave='pixel')
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(bands)


def make_tree(root, offset=None, override=None, production='t_per_ha'):
    """Raw H, Y, P per crop at one decimal, a band map and supplied-style stacks. Returns the paths."""
    rng = np.random.default_rng(0)
    raw_root = os.path.join(root, 'raw')
    raw = {}
    for crop in CROPS:
        valid = rng.random((HEIGHT, WIDTH)) > 0.4
        h = np.where(valid, rng.choice([0.0, 1.5, 7.8, 22.0], (HEIGHT, WIDTH)), np.nan)
        y = np.where(valid, np.round(rng.uniform(0.5, 50.0, (HEIGHT, WIDTH)), 1), np.nan)
        if production == 'kg_per_ha':      # the yield file in kg/ha, production in t
            y = np.round(y * 1000.0)
        scale = {'t_per_ha': 1.0, 'kg_per_ha': 1e-3, 'x1000': 1e3}[production]
        p = np.round(h * y * scale, 1)
        raw[crop] = {'H': h.astype(np.float32), 'Y': y.astype(np.float32), 'P': p.astype(np.float32)}
        transform = REFERENCE if crop != 'vege' else Affine(30.0 - 1e-12, 0.0, -180.0, 0.0, -(30.0 - 1e-12), 90.0)
        if offset and crop == offset[0]:
            transform = offset[1]
        for var, array in raw[crop].items():
            kwargs = dict((override or {}).get((crop, var), {}))
            write_raster(ssr.raw_path(raw_root, var, crop), kwargs.pop('array', array), transform=transform, **kwargs)
    base = os.path.join(root, 'base')
    os.makedirs(base)
    bandmap = os.path.join(base, 'spam2020_bandmap.csv')
    with open(bandmap, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['band', 'crop'])
        w.writerows([(i + 1, c) for i, c in enumerate(CROPS)])
    old_h = np.stack([np.where(np.isfinite(raw[c]['H']), raw[c]['H'], ssr.STACK_NODATA) for c in CROPS]).astype(np.float32)
    old_y = np.stack([np.where(np.isfinite(raw[c]['Y']), raw[c]['Y'] / ssr.YIELD_DIVISOR, ssr.STACK_NODATA)
                      for c in CROPS]).astype(np.float32)
    # vege takes a neighbour's value in one NoData cell, as the supplied stacks do
    r, c = np.argwhere(~np.isfinite(raw['vege']['H']))[0]
    old_h[1, r, c], old_y[1, r, c] = 7.8, 0.0518
    area = os.path.join(base, 'spam2020_harvested_area_stack_TA.tif')
    yld = os.path.join(base, 'spam2020_yield_stack_TA.tif')
    write_raster(area, old_h, nodata=-9999.0)
    write_raster(yld, old_y, nodata=-9999.0)
    return {'raw_root': raw_root, 'bandmap': bandmap, 'area': area, 'yield': yld, 'raw': raw, 'root': root,
            'manifest': os.path.join(root, 'manifest', 'input_manifest.csv'), 'out_dir': os.path.join(root, 'output'),
            'check_csv': os.path.join(root, 'check', 'stack_checks_by_band.csv')}


def manifest(t):
    return ssr.make_manifest(t['raw_root'], t['bandmap'], t['area'], t['yield'], t['manifest'])


def build(t):
    return ssr.build(t['raw_root'], t['bandmap'], t['area'], t['manifest'], t['out_dir'])


def check(t):
    if os.path.exists(t['check_csv']):
        os.remove(t['check_csv'])
    return ssr.check(t['raw_root'], t['bandmap'], t['area'], t['yield'], t['manifest'], t['out_dir'], t['check_csv'])


def read_check(t):
    with open(t['check_csv'], newline='') as f:
        return {r['crop']: r for r in csv.DictReader(f)}


def rewrite(path, band, change):
    with rasterio.open(path, 'r+') as ds:
        a = ds.read(band)
        change(a)
        ds.write(a, band)


@pytest.fixture
def tree(tmp_path):
    t = make_tree(str(tmp_path))
    assert manifest(t) == 0
    return t


def test_build_copies_raw_values_and_masks_exactly(tree):
    assert build(tree) == 0
    assert not os.path.exists(tree['out_dir'] + '.partial')
    for var, name in ssr.OUTPUT_FILES.items():
        with rasterio.open(os.path.join(tree['out_dir'], name)) as ds:
            assert ds.transform == REFERENCE and ds.nodata == -9999.0 and ds.count == len(CROPS)
            assert ds.tags()['UNITS'] == ssr.UNITS[var] and list(ds.descriptions) == CROPS
            for band, crop in enumerate(CROPS, start=1):
                raw = tree['raw'][crop][var]
                new = ds.read(band)
                valid = np.isfinite(raw)
                assert np.array_equal(new != -9999.0, valid)
                assert np.array_equal(new[valid].view(np.uint32), raw[valid].view(np.uint32))


def test_check_passes_and_separates_unit_factor_from_spill(tree):
    assert build(tree) == 0 and check(tree) == 0
    rows = read_check(tree)
    assert rows['bana']['old_stack_differs_by'] == 'unit factor only'
    assert rows['yams']['old_stack_differs_by'] == 'unit factor only'
    assert rows['vege']['old_stack_differs_by'] == 'unit factor, spill and blending'
    assert rows['vege']['old_cells_valid_not_raw_H'] == '1'
    for r in rows.values():
        assert r['yield_unit'] == 't_per_ha' and r['passed'] == 'True' and r['decimals_Y'] == '1'
        assert r['cells_consistent_t_per_ha'] == r['unit_cells'] != r['cells_consistent_kg_per_ha']


@pytest.mark.parametrize('production, unit', [('kg_per_ha', 'kg_per_ha'), ('x1000', 'unknown')])
def test_check_fails_unless_every_cell_fits_t_per_ha(tmp_path, production, unit):
    t = make_tree(str(tmp_path), production=production)
    assert manifest(t) == 0 and build(t) == 0
    assert check(t) == 1
    assert {r['yield_unit'] for r in read_check(t).values()} == {unit}


def test_unit_evidence_rejects_one_cell_outside_the_rounding():
    h = np.array([7.8, 22.0, 1.5], np.float32)
    y = np.array([51.8, 3.2, 10.0], np.float32)
    p = np.array([401.9, 70.4, 15.0], np.float32)       # 7.8 x 51.8 = 404.0 rounds within 0.05 x (7.8 + 51.8)
    ok = np.ones(3, bool)
    assert ssr.unit_evidence(h, ok, y, ok, p, ok)['yield_unit'] == 't_per_ha'
    p[2] = 16.0                                           # 1.5 x 10.0 is at most 1.55 x 10.05 = 15.58
    assert ssr.unit_evidence(h, ok, y, ok, p, ok)['yield_unit'] == 'unknown'


def test_unit_evidence_is_unknown_without_a_fixed_precision():
    h = np.array([1.5, 2.0], np.float32)
    y = np.array([0.12345679, 0.2], np.float32)           # more than six decimals
    p = np.array([0.2, 0.4], np.float32)
    ok = np.ones(2, bool)
    row = ssr.unit_evidence(h, ok, y, ok, p, ok)
    assert row['decimals_Y'] == '' and row['yield_unit'] == 'unknown'


def test_check_fails_when_the_build_wrote_a_wrong_value(tree):
    assert build(tree) == 0
    path = os.path.join(tree['out_dir'], ssr.OUTPUT_FILES['Y'])
    rewrite(path, 2, lambda a: a.__setitem__(tuple(np.argwhere(a != -9999.0)[0]),
                                             np.nextafter(a[tuple(np.argwhere(a != -9999.0)[0])], np.float32(np.inf))))
    hashes = os.path.join(tree['out_dir'], ssr.OUTPUT_HASHES)
    with open(hashes, 'w') as f:
        for name in ssr.OUTPUT_FILES.values():
            f.write('%s  %s\n' % (ssr.sha256_file(os.path.join(tree['out_dir'], name)), name))
    assert check(tree) == 1
    assert read_check(tree)['vege']['values_exact_Y'] == 'False'


@pytest.mark.parametrize('target', ['raw_production', 'raw_yield', 'supplied_yield', 'built_output'])
def test_check_stops_on_drift_after_the_manifest(tree, target):
    assert build(tree) == 0
    path = {'raw_production': ssr.raw_path(tree['raw_root'], 'P', 'yams'),
            'raw_yield': ssr.raw_path(tree['raw_root'], 'Y', 'yams'),
            'supplied_yield': tree['yield'],
            'built_output': os.path.join(tree['out_dir'], ssr.OUTPUT_FILES['H'])}[target]
    rewrite(path, 1, lambda a: a.__setitem__((0, 0), 3.0))
    with pytest.raises(ssr.Mismatch, match='SHA-256'):
        check(tree)


@pytest.mark.parametrize('case', ['displaced', 'size', 'crs', 'rotated', 'negative', 'nodata_collision', 'int_type'])
def test_build_stops_on_mismatch(tmp_path, case):
    h = np.full((HEIGHT, WIDTH), 1.0, np.float32)
    offset, override = None, None
    if case == 'displaced':   # 11.5 cells x 1e-6 degrees, about 1.3 m, over the 1 m bound
        offset = ('yams', Affine(30.0 - 1e-6, 0.0, -180.0, 0.0, -30.0, 90.0))
    if case == 'rotated':
        offset = ('yams', Affine(30.0, 1e-9, -180.0, 0.0, -30.0, 90.0))
    if case == 'size':
        override = {('yams', 'H'): {'array': np.ones((HEIGHT, WIDTH + 1), np.float32)}}
    if case == 'crs':
        override = {('yams', 'H'): {'crs': 'EPSG:3857'}}
    if case == 'negative':
        h[0, 0] = -1.0
        override = {('yams', 'H'): {'array': h}}
    if case == 'nodata_collision':
        h[0, 0] = -9999.0
        override = {('yams', 'H'): {'array': h}}
    if case == 'int_type':
        override = {('yams', 'H'): {'array': np.ones((HEIGHT, WIDTH), np.int16), 'nodata': 0}}
    t = make_tree(str(tmp_path), offset=offset, override=override)
    assert manifest(t) == 0
    with pytest.raises(ssr.Mismatch):
        build(t)
    assert not os.path.exists(t['out_dir']) and not os.path.exists(t['out_dir'] + '.partial')


def test_build_stops_on_changed_missing_or_unlisted_input_and_existing_output(tree):
    path = ssr.raw_path(tree['raw_root'], 'Y', 'yams')
    rewrite(path, 1, lambda a: a.__setitem__((0, 0), 3.0))
    with pytest.raises(ssr.Mismatch, match='SHA-256'):
        build(tree)
    assert not os.path.exists(tree['out_dir'] + '.partial')
    os.remove(path)
    with pytest.raises(ssr.Mismatch, match='missing'):
        build(tree)
    with pytest.raises(ssr.Mismatch, match='not in the pinned manifest'):
        ssr.build(tree['raw_root'] + '_other', tree['bandmap'], tree['area'], tree['manifest'], tree['out_dir'])
    os.makedirs(tree['out_dir'])
    with pytest.raises(ssr.Mismatch, match='already exists'):
        build(tree)


def test_manifest_and_preflight_refuse_to_replace(tree):
    with pytest.raises(ssr.Mismatch, match='never replaced'):
        manifest(tree)
    report = os.path.join(tree['root'], 'preflight', 'preflight.txt')
    grid = os.path.join(tree['root'], 'preflight', 'raw_grid_check.csv')
    assert ssr.preflight(tree['raw_root'], tree['bandmap'], tree['area'], tree['yield'], tree['out_dir'], report, grid, 3) == 0
    with pytest.raises(ssr.Mismatch, match='already exists'):
        ssr.preflight(tree['raw_root'], tree['bandmap'], tree['area'], tree['yield'], tree['out_dir'], report, grid, 3)


def test_run_files_take_no_arguments(tmp_path):
    t = make_tree(str(tmp_path / 'data'))
    run_dir = tmp_path / 'run'
    shutil.copytree(HERE, run_dir / 'code', ignore=shutil.ignore_patterns('__pycache__', 'test_*'))
    for name in RUN_FILES:
        shutil.copy(os.path.join(RUN_DIR, name), run_dir / name)
    with open(run_dir / 'spam_stack_repair_parameters.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerows([('parameter', 'value'), ('raw_root', t['raw_root']), ('bandmap_path', t['bandmap']),
                     ('supplied_area_stack_path', t['area']), ('supplied_yield_stack_path', t['yield']),
                     ('expect_crops', len(CROPS))])
    for name in RUN_FILES:
        done = subprocess.run([sys.executable, '-I', '-B', str(run_dir / name)], cwd=str(tmp_path), capture_output=True, text=True)
        assert done.returncode == 0, done.stdout + done.stderr
    assert os.path.exists(run_dir / 'preflight' / 'preflight.txt')
    assert os.path.exists(run_dir / 'manifest' / 'input_manifest.csv')
    assert sorted(os.listdir(run_dir / 'output')) == sorted(list(ssr.OUTPUT_FILES.values()) + ['build_receipt.csv', ssr.OUTPUT_HASHES])
    assert os.path.exists(run_dir / 'check' / 'stack_checks_by_band.csv')


def test_grid_displacement_of_the_raw_files_is_three_centimetres():
    ref = Affine(0.08333333333331436, 0.0, -180.0, 0.0, -0.08333333333331433, 89.99999999999999)
    raw = Affine(0.08333333327272728, 0.0, -180.0, 0.0, -0.08333333327272725, 89.99999999999999)
    expected = math.hypot(4319.5 * (raw.a - ref.a), 2159.5 * (raw.e - ref.e)) * math.pi / 180.0 * 6378137.0
    got = ssr.grid_displacement_m(raw, ref, 4320, 2160)
    assert got == pytest.approx(expected, rel=1e-12)
    assert 0.032 < got < 0.033 < ssr.MAX_DISPLACEMENT_M

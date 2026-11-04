"""Tests for the prevention_shares launch gates, on small synthetic folders.

Run: python -I -B -m pytest launch/test_shares_launch.py -q -p no:cacheprovider
"""
import csv
import hashlib
import importlib.util
import io
import os
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SHARES = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location('shares_launch', os.path.join(HERE, 'shares_launch.py'))
sl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sl)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(data)


CROPS = ['c%02d' % b for b in range(1, 47)]


def bandmap(root, crops=CROPS):
    path = os.path.join(root, 'bandmap.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['band', 'crop'])
        w.writerows([(i + 1, c) for i, c in enumerate(crops)])
    return path


def check_table(rows):
    """The check table as bytes, from (band, crop, passed, unit, exact) rows."""
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=['band', 'crop', 'passed', 'yield_unit'] + list(sl.EXACT_COLUMNS))
    w.writeheader()
    for band, crop, passed, unit, exact in rows:
        w.writerow(dict({'band': band, 'crop': crop, 'passed': passed, 'yield_unit': unit},
                        **{c: exact for c in sl.EXACT_COLUMNS}))
    return out.getvalue().encode()


def good_rows():
    return [(i + 1, c, 'True', 't_per_ha', 'True') for i, c in enumerate(CROPS)]


def make_repair(root, rows=None):
    """A repair folder as the repair job leaves it, its job record agreeing with whatever rows it is given."""
    out, check = os.path.join(root, 'output'), os.path.join(root, 'check')
    built = []
    for name in sl.STACKS.values():
        write(os.path.join(out, name), name.encode())
        built.append('%s  %s\n' % (sha(name.encode()), name))
    write(os.path.join(out, 'outputs_sha256.txt'), ''.join(built).encode())
    write(os.path.join(out, 'build_receipt.csv'), b'band,crop\n')
    write(os.path.join(root, sl.CHECK_CSV), check_table(good_rows() if rows is None else rows))
    record_job(root)
    return bandmap(root)


def record_job(root, names=None):
    """Rewrite check/result_sha256.txt from what is on disk, as the repair job's sha256sum line does."""
    lines = ['%s  %s\n' % (sl.sha256_file(os.path.join(root, n)), n) for n in sorted(names or sl.RESULT_FILES)]
    write(os.path.join(root, 'check', 'result_sha256.txt'), ''.join(lines).encode())


def test_check_repair_returns_both_stacks_after_a_full_pass(tmp_path):
    bm = make_repair(str(tmp_path))
    stacks = sl.check_repair(str(tmp_path), bm)
    assert sorted(stacks) == ['H', 'Y'] and all(os.path.exists(p) for p in stacks.values())


def test_check_repair_refuses_before_the_repair_ran(tmp_path):
    with pytest.raises(sl.Mismatch, match='has not completed'):
        sl.check_repair(str(tmp_path), bandmap(str(tmp_path)))


def test_check_repair_stops_on_a_check_table_edited_after_the_job(tmp_path):
    rows = good_rows()
    rows[6] = (7, CROPS[6], 'False', 't_per_ha', 'False')
    bm = make_repair(str(tmp_path), rows)          # the job recorded a failing band
    write(os.path.join(str(tmp_path), sl.CHECK_CSV), check_table(good_rows()))    # someone turns it to a pass
    with pytest.raises(sl.Mismatch, match='differs from the job record'):
        sl.check_repair(str(tmp_path), bm)


@pytest.mark.parametrize('changed', sorted(sl.RESULT_FILES - {sl.CHECK_CSV}))
def test_check_repair_stops_on_any_recorded_file_changed_after_the_job(tmp_path, changed):
    bm = make_repair(str(tmp_path))
    write(os.path.join(str(tmp_path), changed), b'changed')
    with pytest.raises(sl.Mismatch, match='differs from the job record'):
        sl.check_repair(str(tmp_path), bm)


def test_check_repair_stops_when_the_job_record_lists_other_files(tmp_path):
    bm = make_repair(str(tmp_path))
    record_job(str(tmp_path), sorted(sl.RESULT_FILES - {sl.CHECK_CSV}))
    with pytest.raises(sl.Mismatch, match='not exactly'):
        sl.check_repair(str(tmp_path), bm)


def duplicate_row():
    rows = good_rows()
    rows[5] = rows[4]                              # band 5 twice, band 6 gone, still 46 rows
    return rows


def duplicate_band_other_crop():
    rows = good_rows()
    rows[5] = (5, CROPS[5], 'True', 't_per_ha', 'True')
    return rows


def duplicate_crop_other_band():
    rows = good_rows()
    rows[5] = (6, CROPS[4], 'True', 't_per_ha', 'True')
    return rows


def extra_duplicate_row():
    return good_rows() + [good_rows()[0]]


def swapped_rows():
    rows = good_rows()
    rows[0], rows[1] = rows[1], rows[0]
    return rows


def other_crop():
    rows = good_rows()
    rows[9] = (10, 'zzzz', 'True', 't_per_ha', 'True')
    return rows


@pytest.mark.parametrize('rows', [duplicate_row(), duplicate_band_other_crop(), duplicate_crop_other_band(),
                                  extra_duplicate_row(), good_rows()[:-1], swapped_rows(), other_crop()],
                         ids=['same row twice', 'band twice', 'crop twice', '47 rows', '45 rows', 'order', 'other crop'])
def test_check_repair_requires_the_band_map_row_for_row_even_with_a_consistent_record(tmp_path, rows):
    bm = make_repair(str(tmp_path), rows)
    with pytest.raises(sl.Mismatch, match='not the band map row for row'):
        sl.check_repair(str(tmp_path), bm)


@pytest.mark.parametrize('row', [(7, CROPS[6], 'False', 't_per_ha', 'True'), (7, CROPS[6], 'True', 'unknown', 'True'),
                                 (7, CROPS[6], 'True', 't_per_ha', 'False')], ids=['failed', 'unit', 'inexact'])
def test_check_repair_stops_on_a_band_that_did_not_pass(tmp_path, row):
    rows = good_rows()
    rows[6] = row
    bm = make_repair(str(tmp_path), rows)
    with pytest.raises(sl.Mismatch, match='did not pass'):
        sl.check_repair(str(tmp_path), bm)


def test_check_repair_stops_on_a_band_map_of_the_wrong_size(tmp_path):
    make_repair(str(tmp_path))
    with pytest.raises(sl.Mismatch, match='45 rows, 46 expected'):
        sl.check_repair(str(tmp_path), bandmap(str(tmp_path), CROPS[:-1]))


def test_check_repair_stops_when_build_hashes_disagree_with_the_job_record(tmp_path):
    bm = make_repair(str(tmp_path))
    name = list(sl.STACKS.values())[0]
    write(os.path.join(str(tmp_path), 'output', 'outputs_sha256.txt'), ('%s  %s\n' % ('0' * 64, name)).encode())
    record_job(str(tmp_path))
    with pytest.raises(sl.Mismatch, match='the build says'):
        sl.check_repair(str(tmp_path), bm)


class Tree:
    """A built task tree with the given p.<name>_task attributes."""
    def __init__(self, names):
        for name in names:
            setattr(self, name + '_task', object())


def test_task_lists_skip_everything_but_prevention_shares_with_and_without_the_optional_task():
    every, skipped = sl.task_lists(Tree(sl.REQUIRED_TASKS))
    assert every == sl.REQUIRED_TASKS and set(every) - set(skipped) == {'prevention_shares'}
    every, skipped = sl.task_lists(Tree(sl.REQUIRED_TASKS + sl.OPTIONAL_TASKS))
    assert 'per_crop_valuation' in every and 'per_crop_valuation' in skipped and 'prevention_shares' not in skipped
    with pytest.raises(sl.Mismatch, match='gep_calculation'):
        sl.task_lists(Tree([t for t in sl.REQUIRED_TASKS if t != 'gep_calculation']))


def test_snapshot_commit_gate(tmp_path):
    info = tmp_path / 'snapshot_info.txt'
    info.write_text('source commit: abc\non remote branches: origin/x\n')
    assert sl.check_snapshot_commit(str(info), 'abc')['on remote branches'] == 'origin/x'
    with pytest.raises(sl.Mismatch, match='not def'):
        sl.check_snapshot_commit(str(info), 'def')
    info.write_text('source commit: abc\n')
    with pytest.raises(sl.Mismatch, match='no remote branch'):
        sl.check_snapshot_commit(str(info), 'abc')


def git(repo, *args):
    env = dict(os.environ, GIT_AUTHOR_NAME='t', GIT_AUTHOR_EMAIL='t@t', GIT_COMMITTER_NAME='t', GIT_COMMITTER_EMAIL='t@t')
    return subprocess.run(['git', '-C', str(repo)] + list(args), capture_output=True, text=True, check=True, env=env).stdout.strip()


def test_library_gate_refuses_uncommitted_and_unpushed_code(tmp_path):
    bare, repo = tmp_path / 'origin.git', tmp_path / 'repo'
    subprocess.run(['git', 'init', '-q', '--bare', str(bare)], check=True)
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    git(repo, 'remote', 'add', 'origin', str(bare))
    (repo / 'a.py').write_text('x = 1\n')
    git(repo, 'add', 'a.py')
    git(repo, 'commit', '-q', '-m', 'one')
    with pytest.raises(sl.Mismatch, match='no remote branch'):
        sl.check_library_committed(str(repo))
    git(repo, 'push', '-q', 'origin', 'HEAD:refs/heads/main')
    git(repo, 'fetch', '-q', 'origin')
    assert sl.check_library_committed(str(repo)) == git(repo, 'rev-parse', 'HEAD')
    (repo / 'a.py').write_text('x = 2\n')
    with pytest.raises(sl.Mismatch, match='uncommitted'):
        sl.check_library_committed(str(repo))


def test_verify_tree_catches_a_changed_or_missing_file(tmp_path):
    root = tmp_path / 'tree'
    write(str(root / 'a' / 'b.py'), b'x = 1\n')
    record = tmp_path / 'snapshot.csv'
    with open(record, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['path', 'sha256'])
        w.writeheader()
        w.writerow({'path': 'a/b.py', 'sha256': sha(b'x = 1\n')})
    assert sl.verify_tree(str(record), str(root)) == 1
    write(str(root / 'a' / 'b.py'), b'x = 2\n')
    with pytest.raises(sl.Mismatch, match='snapshot says'):
        sl.verify_tree(str(record), str(root))
    os.remove(root / 'a' / 'b.py')
    with pytest.raises(sl.Mismatch, match='missing'):
        sl.verify_tree(str(record), str(root))


def test_input_manifest_pins_present_and_absent_rows_and_checks_the_receipt(tmp_path):
    base = tmp_path / 'base'
    es = tmp_path / 'es_parameters.csv'
    sdr = base / sl.SDR_REF_DIR
    for attribute, name in sl.SDR.items():
        write(str(sdr / name), name.encode())
    write(str(sdr / 'SHA256SUMS.source'), ''.join('%s  x/%s\n' % (sha(n.encode()), n) for n in sl.SDR.values()).encode())
    for ref in sl.COUNTRY_REFS:
        write(str(base / ref), b'country')
    write(str(base / 'global_invest' / 'erosion' / 'dem.tif'), b'dem')
    with open(es, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['service', 'parameter', 'value'])
        w.writerow(['erosion', 'erosion_dem_path', 'global_invest/erosion/dem.tif'])
        w.writerow(['erosion', 'erosion_reference_path', 'global_invest/erosion/missing.csv'])
        w.writerow(['erosion', 'erosion_drainage_path', ''])
        w.writerow(['pollination', 'other_path', 'x.tif'])
        for attribute in sl.SDR:
            w.writerow(['erosion', attribute, os.path.join(sl.SDR_REF_DIR, sl.SDR[attribute])])
    out = tmp_path / 'm' / 'manifest.csv'
    rows = sl.make_input_manifest(str(es), str(base), str(out))
    status = {r['parameter']: r['status'] for r in rows}
    assert status['erosion_dem_path'] == 'present' and status['erosion_reference_path'] == 'absent'
    assert 'erosion_drainage_path' not in status and 'other_path' not in status
    assert len(sl.verify_inputs(str(out))) == len(rows)
    write(str(base / 'global_invest' / 'erosion' / 'missing.csv'), b'now here')
    with pytest.raises(sl.Mismatch, match='was absent'):
        sl.verify_inputs(str(out))
    os.remove(base / 'global_invest' / 'erosion' / 'missing.csv')
    write(str(base / 'global_invest' / 'erosion' / 'dem.tif'), b'changed')
    with pytest.raises(sl.Mismatch, match='manifest says'):
        sl.verify_inputs(str(out))
    with pytest.raises(sl.Mismatch, match='never replaced'):
        sl.make_input_manifest(str(es), str(base), str(out))
    write(str(sdr / sl.SDR['erosion_usle_path']), b'other')
    with pytest.raises(sl.Mismatch, match='staging receipt'):
        sl.make_input_manifest(str(es), str(base), str(tmp_path / 'm2.csv'))


def test_tree_hashes_round_trip_and_catch_an_extra_file(tmp_path):
    root = tmp_path / 'shares'
    write(str(root / 'a.csv'), b'a')
    write(str(root / 'sub' / 'b.tif'), b'b')
    hashes = str(tmp_path / 'h.txt')
    assert sl.write_tree_hashes(str(root), hashes) == 2
    assert sl.verify_tree_hashes(str(root), hashes) == 2
    write(str(root / 'c.txt'), b'c')
    with pytest.raises(sl.Mismatch, match='holds 3 files'):
        sl.verify_tree_hashes(str(root), hashes)
    with pytest.raises(sl.Mismatch, match='already exists'):
        sl.write_tree_hashes(str(root), hashes)


def test_valuation_skips_leave_exactly_the_two_valuation_tasks():
    every = sl.REQUIRED_TASKS + sl.OPTIONAL_TASKS
    assert set(every) - set(sl.VALUATION_ONLY) == {'per_crop_valuation', 'gep_calculation'}


def test_run_files_compile():
    for name in os.listdir(SHARES):
        if name.startswith('run_') and name.endswith('.py'):
            with open(os.path.join(SHARES, name)) as f:
                compile(f.read(), name, 'exec')


def test_heartbeat_cpu_parser():
    spec2 = importlib.util.spec_from_file_location('hb_check', os.path.join(SHARES, 'run_shares_heartbeat_check.py'))
    m = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(m)
    assert m.cpu_seconds('3600') == 3600 and m.cpu_seconds('01:00:00') == 3600 and m.cpu_seconds('1-00:00:01') == 86401


def test_the_real_repair_output_passes_the_gate_and_its_stack_names_agree(tmp_path, monkeypatch):
    """Builder, check and the sbatch's sha256sum line, end to end on a synthetic tree, then the gate."""
    code = os.path.join(os.path.dirname(SHARES), 'code')
    spec2 = importlib.util.spec_from_file_location('test_spam_stack_repair', os.path.join(code, 'test_spam_stack_repair.py'))
    repair_tests = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(repair_tests)
    ssr = repair_tests.ssr
    assert ssr.OUTPUT_FILES == sl.STACKS
    t = repair_tests.make_tree(str(tmp_path))
    assert repair_tests.manifest(t) == 0 and repair_tests.build(t) == 0 and repair_tests.check(t) == 0
    root = str(tmp_path)
    names = sorted(os.path.relpath(os.path.join(d, n), root) for sub in ('output', 'check')
                   for d, _, fs in os.walk(os.path.join(root, sub)) for n in fs)
    assert set(names) == sl.RESULT_FILES
    record_job(root, names)
    monkeypatch.setattr(sl, 'EXPECTED_BANDS', len(repair_tests.CROPS))
    stacks = sl.check_repair(root, t['bandmap'])
    assert os.path.basename(stacks['Y']) == sl.STACKS['Y'] and os.path.basename(stacks['H']) == sl.STACKS['H']
    with open(os.path.join(root, sl.CHECK_CSV), 'a') as f:
        f.write('1,bana\n')
    with pytest.raises(sl.Mismatch, match='differs from the job record'):
        sl.check_repair(root, t['bandmap'])


def test_the_rule37_resolution_needs_a_pushed_commit_holding_the_exact_bytes(tmp_path):
    spec3 = importlib.util.spec_from_file_location('run_rule37', os.path.join(SHARES, 'run_rule37_resolution.py'))
    rule37 = importlib.util.module_from_spec(spec3)
    spec3.loader.exec_module(rule37)
    bare, repo, package = tmp_path / 'origin.git', tmp_path / 'repo', tmp_path / 'package'
    subprocess.run(['git', 'init', '-q', '--bare', str(bare)], check=True)
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    git(repo, 'remote', 'add', 'origin', str(bare))
    write(str(package / 'code' / 'a.py'), b'x = 1\n')
    write(str(repo / 'sub' / 'code' / 'a.py'), b'x = 1\n')
    git(repo, 'add', '.')
    git(repo, 'commit', '-q', '-m', 'one')
    head = git(repo, 'rev-parse', 'HEAD')
    with pytest.raises(rule37.sl.Mismatch, match='no remote branch'):
        rule37.commit_holds(str(repo), 'sub', head, str(package), ['code/a.py'])
    git(repo, 'push', '-q', 'origin', 'HEAD:refs/heads/main')
    git(repo, 'fetch', '-q', 'origin')
    assert rule37.commit_holds(str(repo), 'sub', head, str(package), ['code/a.py']) == ['origin/main']
    with pytest.raises(rule37.sl.Mismatch, match='does not hold'):
        rule37.commit_holds(str(repo), 'other', head, str(package), ['code/a.py'])
    write(str(package / 'code' / 'a.py'), b'x = 2\n')
    with pytest.raises(rule37.sl.Mismatch, match='differs'):
        rule37.commit_holds(str(repo), 'sub', head, str(package), ['code/a.py'])


def test_the_rule37_code_list_is_every_code_file_in_the_package():
    spec3 = importlib.util.spec_from_file_location('run_rule37', os.path.join(SHARES, 'run_rule37_resolution.py'))
    rule37 = importlib.util.module_from_spec(spec3)
    spec3.loader.exec_module(rule37)
    package = os.path.dirname(SHARES)
    found = set()
    for folder in (package, os.path.join(package, 'code'), SHARES, HERE):
        for name in os.listdir(folder):
            if name.endswith(('.py', '.sbatch')):
                found.add(os.path.relpath(os.path.join(folder, name), package))
    assert found == set(rule37.CODE_FILES)

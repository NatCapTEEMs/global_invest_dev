"""Gates for the erosion prevention_shares rerun on the repaired SPAM 2020 stacks.

The run files in the folder above call these. Each gate stops with Mismatch on the first thing
that differs from what was pinned: the library snapshot, the repaired stacks and their check, the
erosion inputs on MSI, the staged SDR rasters, the hazelbean checkout, or a project folder that
already exists. Nothing here writes outside the folder it is given.
"""
import csv
import hashlib
import io
import os
import subprocess

# The tasks a snapshot must have. per_crop_valuation exists only in later commits, so it is skipped
# when the snapshot has it and not required.
REQUIRED_TASKS = ['invest_sdr', 'upstream_prevention_share', 'prevention_shares', 'gep_calculation', 'erosion_gep_result']
OPTIONAL_TASKS = ['per_crop_valuation']
VALUATION_ONLY = ['invest_sdr', 'upstream_prevention_share', 'prevention_shares', 'erosion_gep_result']
BANDMAP_REF = os.path.join('global_invest', 'erosion', 'spam2020_bandmap.csv')
CHECK_CSV = os.path.join('check', 'stack_checks_by_band.csv')
# Exactly what the repair job's sha256sum line records once its check has passed.
RESULT_FILES = {os.path.join('output', name) for name in
                ('spam2020_harvested_area_stack_TA_repaired_20261103.tif',
                 'spam2020_yield_stack_TA_repaired_20261103.tif', 'build_receipt.csv', 'outputs_sha256.txt')} | {CHECK_CSV}
EXACT_COLUMNS = ('mask_exact_H', 'values_exact_H', 'mask_exact_Y', 'values_exact_Y')
STACKS = {'H': 'spam2020_harvested_area_stack_TA_repaired_20261103.tif',
          'Y': 'spam2020_yield_stack_TA_repaired_20261103.tif'}
SDR = {'erosion_usle_path': 'usle_fresh.tif', 'erosion_avoided_erosion_path': 'avoided_erosion_fresh.tif',
       'erosion_upstream_prevention_share_path': 'upstream_prevention_share_fresh.tif'}
SDR_MANIFEST_KEYS = {'erosion_usle_path': 'usle_path', 'erosion_avoided_erosion_path': 'avoided_path',
                     'erosion_upstream_prevention_share_path': 'ups_path'}
SDR_REF_DIR = os.path.join('global_invest', 'erosion', 'author_repo_2026_10', 'full_chain_618f67e')
# Read by initialize_country_paths in every task, beside the erosion es_parameters rows.
# What per_crop_valuation and gep_calculation read locally besides the returned prevention shares.
VALUATION_REFS = [('crop_values', os.path.join('..', 'gep_crop_provision', 'intermediate', 'gep_calculation',
                                               'gep_by_country_year_crop.csv')),
                  ('correspondence', os.path.join('global_invest', 'erosion', 'fao_item_spam2020_crop_correspondence.csv')),
                  ('elasticity', os.path.join('global_invest', 'erosion', 'elasticity_crops_fao_revised.csv')),
                  ('bandmap', BANDMAP_REF)]
COUNTRY_REFS = [os.path.join('cartographic', 'ee', 'ee_r264_correspondence.gpkg'),
                os.path.join('cartographic', 'ee', 'ee_r264_correspondence.csv')]
EXPECTED_BANDS = 46


class Mismatch(Exception):
    """Something differs from what was pinned. The run stops on the first one."""


def read_parameters(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return {r['parameter']: r['value'] for r in csv.DictReader(f)}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_hash_list(path):
    """`sha256  relative/path` lines, as sha256sum writes them, keyed by path."""
    with open(path, encoding='utf-8') as f:
        return {name: digest for digest, name in (line.rstrip('\n').split('  ', 1) for line in f if line.strip())}


def verify_tree(snapshot_csv, root):
    """Stop unless every file the snapshot lists is under root with the same SHA-256 and no tracked file is missing."""
    with open(snapshot_csv, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        path = os.path.join(root, r['path'])
        if not os.path.exists(path):
            raise Mismatch('snapshot file missing: %s' % path)
        digest = sha256_file(path)
        if digest != r['sha256']:
            raise Mismatch('%s has SHA-256 %s, the snapshot says %s' % (path, digest, r['sha256']))
    return len(rows)


def task_lists(tree):
    """(every task of this tree, every task but prevention_shares), from a built task tree.

    Stops unless each required task resolves as p.<name>_task, the only form every hazelbean's
    skip_tasks resolves, because a name that resolves nowhere would leave its task enabled.
    """
    missing = [t for t in REQUIRED_TASKS if getattr(tree, t + '_task', None) is None]
    if missing:
        raise Mismatch('no p.<name>_task for %s, so skip_tasks would leave them enabled' % missing)
    every = REQUIRED_TASKS + [t for t in OPTIONAL_TASKS if getattr(tree, t + '_task', None) is not None]
    return every, [t for t in every if t != 'prevention_shares']


def read_bandmap_rows(path):
    """(band, crop) pairs in file order."""
    with open(path, newline='', encoding='utf-8-sig') as f:
        return [(int(r['band']), r['crop'].strip().lower()) for r in csv.DictReader(f)]


def check_repair(repair_dir, bandmap_path):
    """The two repaired stacks, once the repair job's record and check prove them.

    check/result_sha256.txt, which the repair job writes only after its check exits 0, must list
    exactly the two stacks, the receipt, the output hashes and the check table, each with the
    SHA-256 on disk. The check table, read from the bytes just hashed, must then be the band map row
    for row: 46 bands, each band and crop once, in order, every one passing exactly and in t/ha. So
    an edited table, a duplicated or dropped row, or rows for other crops all stop the run.
    """
    result = os.path.join(repair_dir, 'check', 'result_sha256.txt')
    if not os.path.exists(result):
        raise Mismatch('the stack repair has not completed: %s is missing' % result)
    recorded = read_hash_list(result)
    if set(recorded) != RESULT_FILES:
        raise Mismatch('the job record lists %s, not exactly %s' % (sorted(recorded), sorted(RESULT_FILES)))
    for name, digest in recorded.items():
        path = os.path.join(repair_dir, name)
        if not os.path.exists(path):
            raise Mismatch('%s is missing, the job record lists it' % path)
        if name != CHECK_CSV and sha256_file(path) != digest:
            raise Mismatch('%s differs from the job record' % path)
    with open(os.path.join(repair_dir, CHECK_CSV), 'rb') as f:
        data = f.read()
    if hashlib.sha256(data).hexdigest() != recorded[CHECK_CSV]:
        raise Mismatch('%s differs from the job record' % CHECK_CSV)
    rows = list(csv.DictReader(io.StringIO(data.decode('utf-8'))))
    expected = read_bandmap_rows(bandmap_path)
    if len(expected) != EXPECTED_BANDS:
        raise Mismatch('the band map has %d rows, %d expected' % (len(expected), EXPECTED_BANDS))
    try:
        found = [(int(r['band']), r['crop']) for r in rows]
    except (KeyError, TypeError, ValueError) as e:
        raise Mismatch('%s has no usable band and crop columns: %s' % (CHECK_CSV, e))
    if found != expected:
        wrong = [pair for pair in found if found.count(pair) > 1 or pair not in expected]
        raise Mismatch('%s has %d rows and is not the band map row for row (first differing or repeated: %s)'
                       % (CHECK_CSV, len(found), wrong[:3] or [p for p in expected if p not in found][:3] or 'order'))
    failed = [r['crop'] for r in rows if r.get('passed') != 'True' or r.get('yield_unit') != 't_per_ha'
              or any(r.get(c) != 'True' for c in EXACT_COLUMNS)]
    if failed:
        raise Mismatch('%s: these bands did not pass exactly in t/ha: %s' % (CHECK_CSV, failed))
    built = read_hash_list(os.path.join(repair_dir, 'output', 'outputs_sha256.txt'))
    stacks = {}
    for var, name in STACKS.items():
        if built.get(name) != recorded[os.path.join('output', name)]:
            raise Mismatch('%s: the build says %s, the job record %s' % (name, built.get(name), recorded[os.path.join('output', name)]))
        stacks[var] = os.path.abspath(os.path.join(repair_dir, 'output', name))
    return stacks


def check_snapshot_commit(info_path, commit):
    """Stop unless the library snapshot is the export of this commit, which must be on a remote."""
    with open(info_path, encoding='utf-8') as f:
        info = dict(line.rstrip('\n').split(': ', 1) for line in f if ': ' in line)
    if info.get('source commit') != commit:
        raise Mismatch('the snapshot is from commit %s, not %s' % (info.get('source commit'), commit))
    if not info.get('on remote branches'):
        raise Mismatch('the snapshot commit %s is on no remote branch' % commit)
    return info


def check_library_committed(library_dir):
    """HEAD of library_dir, once every tracked file is committed and HEAD is on a remote (rule 37)."""
    def git(*args):
        return subprocess.run(['git', '-C', library_dir] + list(args), capture_output=True, text=True, check=True).stdout.strip()
    dirty = git('status', '--porcelain', '--untracked-files=no').split('\n')
    dirty = [line for line in dirty if line]
    if dirty:
        raise Mismatch('rule 37: %s has %d files with uncommitted changes, so this run could not be reproduced from a clone'
                       % (library_dir, len(dirty)))
    head = git('rev-parse', 'HEAD')
    if not git('branch', '-r', '--contains', head):
        raise Mismatch('rule 37: %s is on no remote branch' % head)
    return head


def erosion_input_refs(es_parameters_csv):
    """(parameter, reference path) for every erosion *_path row with a value, plus the country table."""
    with open(es_parameters_csv, newline='', encoding='utf-8-sig') as f:
        rows = [(r['parameter'], r['value']) for r in csv.DictReader(f)
                if r['service'] == 'erosion' and r['parameter'].endswith('_path') and r['value']]
    return rows + [('country_table', ref) for ref in COUNTRY_REFS]


def make_input_manifest(es_parameters_csv, base_data_dir, out_path):
    """SHA-256 of every erosion input present under base_data_dir, and 'absent' for the rest.

    The three staged SDR rasters are also checked against the staging receipt beside them.
    Refuses to replace an existing manifest.
    """
    if os.path.exists(out_path):
        raise Mismatch('%s already exists. A pinned manifest is never replaced: move it aside first.' % out_path)
    receipt = read_receipt(base_data_dir)
    rows = []
    for parameter, ref in erosion_input_refs(es_parameters_csv):
        path = os.path.join(base_data_dir, ref)
        if os.path.exists(path):
            digest = sha256_file(path)
            if parameter in SDR and receipt[SDR[parameter]] != digest:
                raise Mismatch('%s has SHA-256 %s, the staging receipt says %s' % (path, digest, receipt[SDR[parameter]]))
            rows.append({'parameter': parameter, 'path': os.path.abspath(path), 'status': 'present',
                         'bytes': os.path.getsize(path), 'sha256': digest})
        else:
            rows.append({'parameter': parameter, 'path': os.path.abspath(path), 'status': 'absent', 'bytes': '', 'sha256': ''})
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def make_file_manifest(named_paths, out_path):
    """SHA-256 and size of each (name, path), every one required present. Refuses to replace."""
    if os.path.exists(out_path):
        raise Mismatch('%s already exists. A pinned manifest is never replaced: move it aside first.' % out_path)
    rows = []
    for name, path in named_paths:
        if not os.path.exists(path):
            raise Mismatch('%s (%s) is missing' % (path, name))
        rows.append({'parameter': name, 'path': os.path.abspath(path), 'status': 'present',
                     'bytes': os.path.getsize(path), 'sha256': sha256_file(path)})
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def read_receipt(base_data_dir):
    """The staging receipt of the three SDR rasters, by file name."""
    path = os.path.join(base_data_dir, SDR_REF_DIR, 'SHA256SUMS.source')
    with open(path, encoding='utf-8') as f:
        return {os.path.basename(name): digest for digest, name in (line.split() for line in f if line.strip())}


def verify_inputs(manifest_csv):
    """Stop unless every present input still has its pinned SHA-256 and every absent one is still absent."""
    with open(manifest_csv, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r['status'] == 'absent':
            if os.path.exists(r['path']):
                raise Mismatch('%s was absent when pinned and now exists' % r['path'])
            continue
        if not os.path.exists(r['path']):
            raise Mismatch('pinned input missing: %s' % r['path'])
        digest = sha256_file(r['path'])
        if digest != r['sha256']:
            raise Mismatch('%s has SHA-256 %s, the manifest says %s' % (r['path'], digest, r['sha256']))
    return rows


def check_hazelbean(module, repo_dir, commit):
    """Stop unless hazelbean is imported from repo_dir, at commit, with a clean working tree."""
    location = os.path.realpath(os.path.dirname(module.__file__))
    if not location.startswith(os.path.realpath(repo_dir) + os.sep):
        raise Mismatch('hazelbean is imported from %s, not %s' % (location, repo_dir))
    head = subprocess.run(['git', '-C', repo_dir, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', repo_dir, 'status', '--porcelain', '--untracked-files=no'],
                           capture_output=True, text=True, check=True).stdout.strip()
    if head != commit or dirty:
        raise Mismatch('hazelbean at %s is %s%s, %s expected' % (repo_dir, head, ' with local changes' if dirty else '', commit))
    return head


def check_imported_from(module, root):
    location = os.path.realpath(module.__file__)
    if not location.startswith(os.path.realpath(root) + os.sep):
        raise Mismatch('%s is imported from %s, not from %s' % (module.__name__, location, root))


def write_tree_hashes(root, out_path):
    """sha256sum-style lines for every file under root, paths relative to root. Refuses to replace."""
    if os.path.exists(out_path):
        raise Mismatch('%s already exists' % out_path)
    lines = []
    for dirpath, _, files in sorted(os.walk(root)):
        for name in sorted(files):
            path = os.path.join(dirpath, name)
            lines.append('%s  %s\n' % (sha256_file(path), os.path.relpath(path, root)))
    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    return len(lines)


def verify_tree_hashes(root, hash_list):
    """Stop unless root holds exactly the files hash_list names, each with its SHA-256."""
    expected = read_hash_list(hash_list)
    found = sorted(os.path.relpath(os.path.join(d, n), root) for d, _, fs in os.walk(root) for n in fs)
    if sorted(expected) != found:
        raise Mismatch('%s holds %d files, the list %d: %s' % (root, len(found), len(expected),
                                                               sorted(set(found) ^ set(expected))[:5]))
    for name, digest in expected.items():
        if sha256_file(os.path.join(root, name)) != digest:
            raise Mismatch('%s differs from %s' % (os.path.join(root, name), hash_list))
    return len(expected)

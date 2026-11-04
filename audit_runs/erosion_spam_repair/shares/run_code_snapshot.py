"""Export the library at a pushed commit into code/global_invest_dev. Takes no arguments.

Local, before anything goes to MSI. Reads library_dir and source_commit from local_parameters.csv
and stops unless the commit is on a remote branch, so the job runs code anyone can clone. Exports
with git archive, never from the working tree, so no uncommitted edit can enter, and proves each
exported file equal to its git blob. Writes code_snapshot/snapshot_files.csv (path, SHA-256) and
code_snapshot/snapshot_info.txt. Also records what the working tree holds beyond that commit, for
the valuation that needs it: code_snapshot/working_tree_beyond_commit.diff and
code_snapshot/functions_changed_since_commit.csv, each top-level function, class, import and
assignment whose code differs, docstrings ignored. Refuses if the snapshot exists.

Run: python -I -B run_code_snapshot.py
"""
import ast
import csv
import hashlib
import io
import os
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT = os.path.join(HERE, 'code', 'global_invest_dev')
RECORD = os.path.join(HERE, 'code_snapshot')


def git(library, *args):
    return subprocess.run(['git', '-C', library] + list(args), capture_output=True, check=True).stdout


def without_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body \
                and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], 'value', None), ast.Constant) \
                and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:] or [ast.Pass()]
    return tree


def top_level(source):
    table = {}
    for node in without_docstrings(ast.parse(source)).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            table['def ' + node.name] = ast.dump(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            table['import ' + ast.unparse(node)] = ''
        elif isinstance(node, ast.Assign):
            table['assign ' + ', '.join(ast.unparse(t) for t in node.targets)] = ast.dump(node.value)
    return table


def functions_changed(library, commit):
    rows = []
    for name in git(library, 'diff', '--name-only', commit).decode().split():
        if not name.endswith('.py'):
            rows.append({'file': name, 'item': '(not python)', 'state': 'changed'})
            continue
        old = top_level(git(library, 'show', '%s:%s' % (commit, name)).decode())
        with open(os.path.join(library, name), encoding='utf-8') as f:
            new = top_level(f.read())
        for item in sorted(set(old) | set(new)):
            if old.get(item) != new.get(item):
                rows.append({'file': name, 'item': item,
                             'state': 'added' if item not in old else 'removed' if item not in new else 'changed'})
    return rows


def main():
    with open(os.path.join(HERE, 'local_parameters.csv'), newline='', encoding='utf-8-sig') as f:
        q = {r['parameter']: r['value'] for r in csv.DictReader(f)}
    library, commit = q['library_dir'], q['source_commit']
    for path in (SNAPSHOT, RECORD):
        if os.path.exists(path):
            print('STOP: %s already exists' % path, file=sys.stderr)
            return 2
    remotes = [r.strip() for r in git(library, 'branch', '-r', '--contains', commit).decode().splitlines() if r.strip()]
    if not remotes:
        print('STOP: %s is on no remote branch, so a clone could not reach it' % commit, file=sys.stderr)
        return 2
    blobs = {}
    for line in git(library, 'ls-tree', '-r', commit).decode().splitlines():
        meta, name = line.split('\t', 1)
        blobs[name] = meta.split()[2]
    os.makedirs(SNAPSHOT)
    with tarfile.open(fileobj=io.BytesIO(git(library, 'archive', '--format=tar', commit))) as tar:
        tar.extractall(SNAPSHOT, filter='data')
    rows = []
    for name in sorted(blobs):
        path = os.path.join(SNAPSHOT, name)
        if not os.path.exists(path):
            print('STOP: the export has no %s' % name, file=sys.stderr)
            return 2
        with open(path, 'rb') as f:
            data = f.read()
        if hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest() != blobs[name]:
            print('STOP: the export of %s differs from its git blob' % name, file=sys.stderr)
            return 2
        rows.append({'path': name, 'sha256': hashlib.sha256(data).hexdigest()})
    exported = sum(len(files) for _, _, files in os.walk(SNAPSHOT))
    if exported != len(rows):
        print('STOP: the export has %d files, the commit %d' % (exported, len(rows)), file=sys.stderr)
        return 2
    os.makedirs(RECORD)
    with open(os.path.join(RECORD, 'snapshot_files.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['path', 'sha256'])
        w.writeheader()
        w.writerows(rows)
    diff = git(library, 'diff', '--binary', commit)
    with open(os.path.join(RECORD, 'working_tree_beyond_commit.diff'), 'wb') as f:
        f.write(diff)
    changed = functions_changed(library, commit)
    with open(os.path.join(RECORD, 'functions_changed_since_commit.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['file', 'item', 'state'])
        w.writeheader()
        w.writerows(changed)
    info = ['source commit: %s' % commit,
            'on remote branches: %s' % ' '.join(remotes),
            'files: %d, each equal to its git blob' % len(rows),
            'working tree beyond the commit: %d files, diff SHA-256 %s'
            % (len(git(library, 'diff', '--name-only', commit).decode().split()), hashlib.sha256(diff).hexdigest())]
    with open(os.path.join(RECORD, 'snapshot_info.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(info) + '\n')
    print('\n'.join(info))
    return 0


if __name__ == '__main__':
    sys.exit(main())

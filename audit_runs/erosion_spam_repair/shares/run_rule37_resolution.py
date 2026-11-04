"""Record the pushed commit that holds the stack builder and this launch code. Takes no arguments.

Local, once that commit exists. Reads code_repo_dir, code_repo_subdir and code_commit from
local_parameters.csv, blank until the decision is made. Stops unless the commit is on a remote branch
and holds every code file of this package byte for byte under code_repo_subdir, then writes
../manifest/rule37_resolution.txt, which both sbatch files require and which is copied to MSI with
the rest. So the gate can only be opened by a commit that really holds what runs. Refuses if the
file exists.

Run: python -I -B run_rule37_resolution.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE = os.path.dirname(HERE)
RESOLUTION = os.path.join(PACKAGE, 'manifest', 'rule37_resolution.txt')
sys.path.insert(0, os.path.join(HERE, 'launch'))
import shares_launch as sl

CODE_FILES = ['code/spam_stack_repair.py', 'code/test_spam_stack_repair.py', 'run_spam_stack_preflight.py',
              'run_spam_stack_manifest.py', 'run_spam_stack_repair.py', 'build_spam_stacks.sbatch',
              'shares/launch/shares_launch.py', 'shares/launch/test_shares_launch.py', 'shares/run_code_snapshot.py',
              'shares/run_shares_manifest.py', 'shares/run_shares_preflight.py', 'shares/run_erosion_spam_repair_shares.py',
              'shares/run_erosion_spam_repair_shares.sbatch', 'shares/run_shares_heartbeat_check.py',
              'shares/run_valuation_manifest.py', 'shares/run_erosion_spam_repair_valuation.py',
              'shares/run_rule37_resolution.py']


def git(repo, *args):
    return subprocess.run(['git', '-C', repo] + list(args), capture_output=True, check=True).stdout


def commit_holds(repo, subdir, commit, root, files):
    """The commit's remote branches, once it is pushed and holds each file of root byte for byte under subdir."""
    remotes = git(repo, 'branch', '-r', '--contains', commit).decode().split()
    if not remotes:
        raise sl.Mismatch('rule 37: %s is on no remote branch' % commit)
    for name in files:
        target = '/'.join(p for p in (subdir, name) if p)
        try:
            held = git(repo, 'show', '%s:%s' % (commit, target))
        except subprocess.CalledProcessError:
            raise sl.Mismatch('rule 37: %s does not hold %s' % (commit, target))
        with open(os.path.join(root, name), 'rb') as f:
            if f.read() != held:
                raise sl.Mismatch('rule 37: %s in %s differs from the file in this package' % (target, commit))
    return remotes


def main():
    q = sl.read_parameters(os.path.join(HERE, 'local_parameters.csv'))
    repo, subdir, commit = q.get('code_repo_dir'), q.get('code_repo_subdir', ''), q.get('code_commit')
    if not repo or not commit:
        raise sl.Mismatch('rule 37: code_repo_dir and code_commit are blank in local_parameters.csv. '
                          'The builder and launch code are in no repository yet.')
    if os.path.exists(RESOLUTION):
        raise sl.Mismatch('%s already exists' % RESOLUTION)
    full = git(repo, 'rev-parse', commit).decode().strip()
    remotes = commit_holds(repo, subdir, full, PACKAGE, CODE_FILES)
    url = git(repo, 'remote', 'get-url', 'origin').decode().strip()
    with open(RESOLUTION, 'w', encoding='utf-8') as f:
        f.write('%s at %s on %s holds %d code files byte for byte under %s\n'
                % (url, full, ' '.join(remotes), len(CODE_FILES), subdir or 'the repository root'))
    print('written: %s' % RESOLUTION)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except sl.Mismatch as e:
        print('STOP: %s' % e, file=sys.stderr)
        sys.exit(2)

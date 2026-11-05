"""Record the pushed commit that holds this launch code. Takes no arguments.

Local, once that commit exists. Reads code_repo_dir, code_repo_subdir and code_commit from
local_parameters.csv, with code_commit blank until the decision is made. Stops unless the commit is on a remote branch and holds
every code file of this package byte for byte, then writes manifest/rule37_resolution.txt, which the
sbatch requires. Refuses if the file exists.

Run: python -I -B run_eps5_rule37_resolution.py
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESOLUTION = os.path.join(HERE, 'manifest', 'rule37_resolution.txt')
sys.path.insert(0, os.path.join(HERE, 'launch'))
import eps5_gates as eg

COMMITTED = os.path.join(HERE, 'code', 'global_invest_dev', 'audit_runs', 'erosion_spam_repair', 'shares', 'run_rule37_resolution.py')
CODE_FILES = ['README.md', 'parameters_template.csv', 'run_eps5_prediction_bounds.py', 'prediction/magnitude_bounds.csv', 'prediction/boundary_reach.csv', 'prediction/eps5_preflight_first_run.txt', 'prediction/prediction_sha256.txt', 'prediction/sdr_rasters_library_vs_author.csv',
              'prediction/sdr_reach_cells.csv', 'prediction/sdr_reach_expected_differences.csv', 'prediction/sdr_reach_summary.txt', 'prediction/ses11_area_conservation_audit_rows_that_differ.csv',
              'prediction/ses11_country_ps_diagnostics_text_differences.csv', 'prediction/ses11_threshold_policy_text_differences.csv', 'launch/eps5_prediction.py',
              'launch/eps5_compare.py', 'launch/eps5_gates.py', 'launch/eps5_probe.py', 'launch/test_eps5_compare.py', 'run_eps5_code_snapshot.py',
              'run_eps5_local_checks.py', 'run_eps5_manifest.py', 'run_eps5_preflight.py', 'run_eps5_shares.py', 'run_eps5_shares.sbatch',
              'run_eps5_heartbeat_check.py', 'run_eps5_compare.py', 'run_eps5_rule37_resolution.py', 'run_eps5_raster_cells.py', 'run_eps5_raster_reach.py', 'eps5_settings.csv', 'author_pins.csv']


def main():
    q = eg.read_parameters(os.path.join(HERE, 'local_parameters.csv'))
    repo, subdir, commit = q.get('code_repo_dir'), q.get('code_repo_subdir', ''), q.get('code_commit')
    if not repo or not commit:
        raise eg.Mismatch('rule 37: code_commit is blank in local_parameters.csv. This launch code is in no repository yet.')
    if os.path.exists(RESOLUTION):
        raise eg.Mismatch('%s already exists' % RESOLUTION)
    os.makedirs(os.path.dirname(RESOLUTION), exist_ok=True)
    spec = importlib.util.spec_from_file_location('run_rule37_resolution', COMMITTED)
    committed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(committed)
    full = committed.git(repo, 'rev-parse', commit).decode().strip()
    try:
        remotes = committed.commit_holds(repo, subdir, full, HERE, CODE_FILES)
    except committed.sl.Mismatch as e:
        raise eg.Mismatch(str(e))
    url = committed.git(repo, 'remote', 'get-url', 'origin').decode().strip()
    with open(RESOLUTION, 'w', encoding='utf-8') as f:
        f.write('%s at %s on %s holds %d code files byte for byte under %s\n' % (url, full, ' '.join(remotes), len(CODE_FILES), subdir or 'the repository root'))
    print('written: %s' % RESOLUTION)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except eg.Mismatch as e:
        print('STOP: %s' % e, file=sys.stderr)
        sys.exit(2)

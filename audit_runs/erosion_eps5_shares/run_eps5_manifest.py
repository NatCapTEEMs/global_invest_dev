"""Pin everything the EPS-5 run and comparison read on MSI, and this package. Takes no arguments.

For a login node. Writes manifest/eps5_input_manifest.csv (every erosion input the library can read),
manifest/eps5_author_manifest.csv (the author's EPS-5 and SES-11 files, each role in author_pins.csv
required at its pinned SHA-256) and manifest/pinned_sha256.txt, which the job checks first. Refuses to
replace any of them.

Run: python -I -B run_eps5_manifest.py
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT = os.path.join(HERE, 'code', 'global_invest_dev')
sys.path.insert(0, os.path.join(HERE, 'launch'))
import eps5_gates as eg
import eps5_probe as ep

ES_PARAMETERS = os.path.join(SNAPSHOT, 'global_invest', 'input_template', 'es_parameters.csv')
AUTHOR_EXTRA = ['results/eps5_sensitivity/manifest.json', 'results/eps5_sensitivity/run_metadata.txt', 'results/eps5_sensitivity/country_ps_diagnostics.csv',
                'results/eps5_sensitivity/country_crop_protected_production_onfarm.csv', 'results/eps5_sensitivity/country_crop_protected_production_upstream.csv',
                'results/eps5_sensitivity/country_crop_protected_production_combined.csv', 'results/ses11_headline/manifest.json',
                'results/ses11_headline/ps_onfarm_cropland_severe.tif', 'results/ses11_headline/ps_upstream_cropland_severe.tif',
                'results/ses11_headline/ps_combined_union_cropland_severe.tif', 'repro_test_valuation.sh', 'MANIFEST.md']
PREDICTION_FILES = ['prediction/boundary_reach.csv', 'prediction/eps5_preflight_first_run.txt', 'prediction/magnitude_bounds.csv', 'prediction/prediction_sha256.txt', 'prediction/sdr_rasters_library_vs_author.csv', 'prediction/sdr_reach_cells.csv',
                    'prediction/sdr_reach_expected_differences.csv', 'prediction/sdr_reach_summary.txt', 'prediction/ses11_area_conservation_audit_rows_that_differ.csv',
                    'prediction/ses11_country_ps_diagnostics_text_differences.csv', 'prediction/ses11_threshold_policy_text_differences.csv']
PACKAGE_FILES = PREDICTION_FILES + ['launch/eps5_compare.py', 'launch/eps5_gates.py', 'launch/eps5_prediction.py', 'launch/eps5_probe.py', 'launch/test_eps5_compare.py', 'msi_parameters.csv', 'eps5_settings.csv',
                 'author_pins.csv', 'run_eps5_manifest.py', 'run_eps5_preflight.py', 'run_eps5_shares.py', 'run_eps5_shares.sbatch', 'run_eps5_heartbeat_check.py',
                 'run_eps5_compare.py', 'code_snapshot/snapshot_files.csv', 'code_snapshot/snapshot_info.txt', 'manifest/eps5_input_manifest.csv', 'manifest/eps5_author_manifest.csv']


def main():
    q = eg.read_parameters(os.path.join(HERE, 'msi_parameters.csv'))
    sl = ep.load_shared_launch(SNAPSHOT)
    pinned = os.path.join(HERE, 'manifest', 'pinned_sha256.txt')
    if os.path.exists(pinned):
        raise eg.Mismatch('%s already exists: move it aside first' % pinned)
    os.makedirs(os.path.dirname(pinned), exist_ok=True)
    rows = sl.make_input_manifest(ES_PARAMETERS, q['base_data_dir'], os.path.join(HERE, 'manifest', 'eps5_input_manifest.csv'))
    pins = eg.read_rows(os.path.join(HERE, 'author_pins.csv'))
    eg.check_original_stacks(rows, pins)
    archive = q['author_archive_dir']
    named = [(r['role'], os.path.join(archive, r['relative_path'])) for r in pins] + [(name, os.path.join(archive, name)) for name in AUTHOR_EXTRA]
    author = sl.make_file_manifest(named, os.path.join(HERE, 'manifest', 'eps5_author_manifest.csv'))
    digest = {r['parameter']: r['sha256'] for r in author}
    for r in pins:
        if digest[r['role']] != r['sha256']:
            raise eg.Mismatch('%s has SHA-256 %s, the pin %s' % (r['relative_path'], digest[r['role']], r['sha256']))
    lines = []
    for name in PACKAGE_FILES:
        with open(os.path.join(HERE, name), 'rb') as f:
            lines.append('%s  %s\n' % (hashlib.sha256(f.read()).hexdigest(), name))
    with open(pinned, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    present = [r for r in rows if r['status'] == 'present']
    print('eps5 manifest: %d library inputs pinned, %d absent (%s), %d author files, %d package files'
          % (len(present), len(rows) - len(present), ' '.join(r['parameter'] for r in rows if r['status'] == 'absent'), len(author), len(lines)))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (eg.Mismatch,) as e:
        print('STOP: %s' % e, file=sys.stderr)
        sys.exit(2)

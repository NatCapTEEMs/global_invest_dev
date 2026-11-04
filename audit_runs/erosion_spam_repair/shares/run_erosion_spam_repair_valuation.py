"""Value erosion per crop on the returned prevention shares, locally. Takes no arguments.

After the MSI job, and after its outputs are copied back as the README says. Runs per_crop_valuation
and gep_calculation (the per-crop table and the account row, and the published 30 arc-second share
map) in a new project, project_dir in local_parameters.csv, whose intermediate/prevention_shares is
the copied-back MSI output. Skips invest_sdr, upstream_prevention_share, prevention_shares and the
report. Nothing in gep_erosion or base_data is written.

Stops unless: per_crop_valuation, gep_calculation and result_valuation/ are absent; the copied-back
prevention shares match result/prevention_shares_sha256.txt file for file; the valuation inputs
match manifest/valuation_input_manifest.csv; the copied-back stacks pass the same repair gate as on
MSI; hazelbean is the pinned clean checkout; and the library has no uncommitted change and its HEAD
is on a remote (rule 37). The per-crop code is not in the commit the MSI job ran, so it must be
committed first: this run refuses until it is. Writes the library commit and the SHA-256 of both
task folders into result_valuation/.

Run: python -I -B run_erosion_spam_repair_valuation.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT = os.path.join(HERE, 'result_valuation')
sys.path.insert(0, os.path.join(HERE, 'launch'))
import shares_launch as sl


def main():
    q = sl.read_parameters(os.path.join(HERE, 'local_parameters.csv'))
    project = q['project_dir']
    intermediate = os.path.join(project, 'intermediate')
    for path in (os.path.join(intermediate, 'per_crop_valuation'), os.path.join(intermediate, 'gep_calculation'), RESULT):
        if os.path.exists(path):
            raise sl.Mismatch('%s already exists: this run writes new outputs only' % path)
    n_shares = sl.verify_tree_hashes(os.path.join(intermediate, 'prevention_shares'),
                                     os.path.join(HERE, 'result', 'prevention_shares_sha256.txt'))
    sl.verify_inputs(os.path.join(HERE, 'manifest', 'valuation_input_manifest.csv'))
    stacks = sl.check_repair(os.path.dirname(HERE), os.path.join(q['base_data_dir'], sl.BANDMAP_REF))
    library_head = sl.check_library_committed(q['library_dir'])
    sys.path.insert(0, q['library_dir'])
    import hazelbean as hb
    import global_invest
    from global_invest.erosion import run_erosion
    sl.check_hazelbean(hb, q['hazelbean_dir'], q['hazelbean_commit'])
    sl.check_imported_from(global_invest, q['library_dir'])

    p = hb.ProjectFlow(project_dir=project, run_mode='check')
    p.base_data_dir = q['base_data_dir']
    p.erosion_area_stack_path = stacks['H']
    p.erosion_yield_stack_path = stacks['Y']
    p.tasks_to_skip = sl.VALUATION_ONLY
    run_erosion.run_project(p)

    os.makedirs(RESULT)
    counts = {name: sl.write_tree_hashes(os.path.join(intermediate, name), os.path.join(RESULT, '%s_sha256.txt' % name))
              for name in ('per_crop_valuation', 'gep_calculation')}
    country = hb.df_read(os.path.join(intermediate, 'per_crop_valuation', 'country_gep_per_crop.csv'))
    account = hb.df_read(os.path.join(intermediate, 'gep_calculation', 'gep_by_country_base_year.csv'))
    with open(os.path.join(RESULT, 'library_commit.txt'), 'w', encoding='utf-8') as f:
        f.write(library_head + '\n')
    print('prevention shares: %d files verified; library at %s' % (n_shares, library_head))
    print('per crop, combined: %s USD over %d countries; account row: %s USD'
          % (format(country['gep_const2019_usd_combined'].sum(), ',.2f'),
             int(country['gep_const2019_usd_combined'].notna().sum()), format(account['erosion_gep'].sum(), ',.2f')))
    print('hashed: %s' % counts)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except sl.Mismatch as e:
        print('STOP: %s' % e, file=sys.stderr)
        sys.exit(2)

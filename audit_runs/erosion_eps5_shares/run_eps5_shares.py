"""The library's prevention_shares at the EPS-5 threshold on the original stacks, on MSI. Takes no arguments.

The whole task runs from nothing in a project of its own: the cropland mask, the threshold policy,
the three prevention-share rasters, the country diagnostics and the country-crop protected
production. It reads the staged SDR rasters and the original stacks, and only the high threshold
differs from the library's SES-11 run. The author's EPS-5 outputs are not read, so what comes out can
be compared with them. Before anything runs it stops unless project/ and result/ are absent, the
snapshot matches its record, every pinned input matches, and hazelbean is the pinned clean checkout.
Two passes: the first only publishes paths and checks them, the second runs the task. Afterwards the
task's manifest must hold the pinned settings and name the same inputs, and result/ records the
SHA-256 of every file the task wrote. Writes nothing outside this folder.

Run: python -I -B run_eps5_shares.py (from run_eps5_shares.sbatch)
"""
import datetime
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT = os.path.join(HERE, 'code', 'global_invest_dev')
PROJECT_DIR = os.path.join(HERE, 'project', 'gep_erosion_eps5_library')
RESULT = os.path.join(HERE, 'result')
sys.path.insert(0, os.path.join(HERE, 'launch'))
import eps5_gates as eg
import eps5_probe as ep


def main():
    started = datetime.datetime.now(datetime.timezone.utc)
    q = eg.read_parameters(os.path.join(HERE, 'msi_parameters.csv'))
    settings = eg.read_rows(os.path.join(HERE, 'eps5_settings.csv'))
    base = q['base_data_dir']
    sl = ep.load_shared_launch(SNAPSHOT)
    for path in (PROJECT_DIR, RESULT):
        if os.path.exists(path):
            raise eg.Mismatch('%s already exists. This run starts from nothing: move it aside first.' % path)
    sl.check_snapshot_commit(os.path.join(HERE, 'code_snapshot', 'snapshot_info.txt'), q['library_commit'])
    n_files = sl.verify_tree(os.path.join(HERE, 'code_snapshot', 'snapshot_files.csv'), SNAPSHOT)
    rows = sl.verify_inputs(os.path.join(HERE, 'manifest', 'eps5_input_manifest.csv'))
    eg.check_original_stacks(rows, eg.read_rows(os.path.join(HERE, 'author_pins.csv')))
    sys.path.insert(0, SNAPSHOT)
    import hazelbean as hb
    import global_invest
    from global_invest.erosion import run_erosion
    hazelbean_head = sl.check_hazelbean(hb, q['hazelbean_dir'], q['hazelbean_commit'])
    sl.check_imported_from(global_invest, SNAPSHOT)
    sl.check_imported_from(run_erosion, SNAPSHOT)

    every_task, all_but_shares = ep.publish_only_pass(hb, run_erosion, sl, eg, PROJECT_DIR, base, settings)
    hb.log('pass 1: threshold 5.0, the staged SDR rasters, the original stacks and the band map are what the task reads')

    p = hb.ProjectFlow(project_dir=PROJECT_DIR, run_mode='check')
    ep.configure(p, base, settings)
    p.tasks_to_skip = all_but_shares
    p = run_erosion.run_project(p)

    expected = ep.expected_paths(sl, base)
    named = dict({key: os.path.normpath(expected[attribute]) for attribute, key in sl.SDR_MANIFEST_KEYS.items()},
                 area_stack=os.path.normpath(expected['erosion_area_stack_path']), yield_stack=os.path.normpath(expected['erosion_yield_stack_path']),
                 bandmap_csv=os.path.normpath(expected['erosion_bandmap_csv_path']))
    shares_dir = os.path.join(PROJECT_DIR, 'intermediate', 'prevention_shares')
    recorded = eg.check_run_manifest(os.path.join(p.erosion_gep_output_dir, 'manifest.json'), settings, named)
    os.makedirs(RESULT)
    n_outputs = sl.write_tree_hashes(shares_dir, os.path.join(RESULT, 'prevention_shares_sha256.txt'))
    import numpy
    import rasterio
    record = ['started_utc: %s' % started.isoformat(),
              'finished_utc: %s' % datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'task elapsed minutes: %s' % recorded.get('elapsed_minutes'),
              'threshold high and low: %s %s' % (recorded['threshold_high_t_ha_yr'], recorded['threshold_low_t_ha_yr']),
              'project: %s' % PROJECT_DIR,
              'library: %d files exported from commit %s' % (n_files, q['library_commit']),
              'hazelbean: %s at %s' % (q['hazelbean_dir'], hazelbean_head),
              'python %s, numpy %s, rasterio %s, GDAL %s' % (platform.python_version(), numpy.__version__, rasterio.__version__, rasterio.__gdal_version__),
              'prevention_shares files hashed: %d' % n_outputs]
    with open(os.path.join(RESULT, 'run_record.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(record) + '\n')
    hb.log('\n'.join(record))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except eg.Mismatch as e:
        print('STOP: %s' % e, file=sys.stderr)
        sys.exit(2)

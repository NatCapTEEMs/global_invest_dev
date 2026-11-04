"""Rerun erosion's prevention_shares on the repaired SPAM 2020 stacks, on MSI. Takes no arguments.

The whole task reruns: the cropland mask (the union of all 46 harvested-area bands), the three
prevention-share rasters, the country diagnostics, soil retained and the country-crop protected
production. The SDR rasters are the pinned staged ones, never rerun. Nothing else runs: invest_sdr,
upstream_prevention_share, per_crop_valuation, gep_calculation and the report are skipped, and the
valuation follows locally (run_erosion_spam_repair_valuation.py).

The library is the export of a pushed commit (code_snapshot/snapshot_info.txt), which this file
checks against library_commit in msi_parameters.csv. Before anything runs it stops unless: project/
is absent, the snapshot matches code_snapshot/snapshot_files.csv, every erosion input matches
manifest/shares_input_manifest.csv, the stack repair beside this folder passed its check, proved
by its job record and row for row against the band map, and hazelbean is the pinned clean checkout. Two passes: the first skips every task, so
each only publishes its paths, and checks them; the second runs prevention_shares. Afterwards the
task's manifest must name the staged SDR rasters and the repaired stacks, and result/ records the
SHA-256 of every file the task wrote. Writes nothing outside this folder.

Run: python -I -B run_erosion_spam_repair_shares.py (from run_erosion_spam_repair_shares.sbatch)
"""
import datetime
import json
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT = os.path.join(HERE, 'code', 'global_invest_dev')
PROJECT_DIR = os.path.join(HERE, 'project', 'gep_erosion_spam_repair')
RESULT = os.path.join(HERE, 'result')
sys.path.insert(0, os.path.join(HERE, 'launch'))
import shares_launch as sl


def configure(p, base_data_dir, stacks):
    """What this run sets before hydration, so it wins over es_parameters: base data and the two stacks."""
    p.base_data_dir = base_data_dir
    p.erosion_area_stack_path = stacks['H']
    p.erosion_yield_stack_path = stacks['Y']


def main():
    started = datetime.datetime.now(datetime.timezone.utc)
    q = sl.read_parameters(os.path.join(HERE, 'msi_parameters.csv'))
    base = q['base_data_dir']
    for path in (PROJECT_DIR, RESULT):
        if os.path.exists(path):
            raise sl.Mismatch('%s already exists. This run starts from nothing: move it aside first.' % path)
    sl.check_snapshot_commit(os.path.join(HERE, 'code_snapshot', 'snapshot_info.txt'), q['library_commit'])
    n_files = sl.verify_tree(os.path.join(HERE, 'code_snapshot', 'snapshot_files.csv'), SNAPSHOT)
    sl.verify_inputs(os.path.join(HERE, 'manifest', 'shares_input_manifest.csv'))
    stacks = sl.check_repair(os.path.dirname(HERE), os.path.join(base, sl.BANDMAP_REF))
    sys.path.insert(0, SNAPSHOT)
    import hazelbean as hb
    import global_invest
    from global_invest.erosion import run_erosion
    hazelbean_head = sl.check_hazelbean(hb, q['hazelbean_dir'], q['hazelbean_commit'])
    sl.check_imported_from(global_invest, SNAPSHOT)
    sl.check_imported_from(run_erosion, SNAPSHOT)

    staged = {attribute: os.path.join(base, sl.SDR_REF_DIR, name) for attribute, name in sl.SDR.items()}
    expected = dict(staged, erosion_area_stack_path=stacks['H'], erosion_yield_stack_path=stacks['Y'],
                    erosion_bandmap_csv_path=os.path.join(base, sl.BANDMAP_REF))

    def run(tasks_to_skip):
        p = hb.ProjectFlow(project_dir=PROJECT_DIR, run_mode='check')
        configure(p, base, stacks)
        p.tasks_to_skip = tasks_to_skip
        return run_erosion.run_project(p)

    probe = hb.ProjectFlow(project_dir=PROJECT_DIR, run_mode='check')
    run_erosion.build_task_tree(probe)
    every_task, all_but_shares = sl.task_lists(probe)

    p = run(every_task)
    for attribute, path in expected.items():
        if os.path.normpath(str(getattr(p, attribute))) != os.path.normpath(path):
            raise sl.Mismatch('pass 1: %s resolves to %s, not %s' % (attribute, getattr(p, attribute), path))
    shares_dir = os.path.join(PROJECT_DIR, 'intermediate', 'prevention_shares')
    if os.path.exists(shares_dir):
        raise sl.Mismatch('pass 1 ran prevention_shares, which it must only publish')
    hb.log('pass 1: the staged SDR rasters, the repaired stacks and the band map are the ones the task reads')

    p = run(all_but_shares)
    with open(os.path.join(p.erosion_gep_output_dir, 'manifest.json')) as f:
        recorded = json.load(f)
    inputs = recorded['inputs']
    named = dict({key: staged[attribute] for attribute, key in sl.SDR_MANIFEST_KEYS.items()},
                 area_stack=stacks['H'], yield_stack=stacks['Y'], bandmap_csv=expected['erosion_bandmap_csv_path'])
    for key, path in named.items():
        if os.path.normpath(inputs[key]) != os.path.normpath(path):
            raise sl.Mismatch('the prevention-share manifest records %s = %s, not %s' % (key, inputs[key], path))
    os.makedirs(RESULT)
    n_outputs = sl.write_tree_hashes(shares_dir, os.path.join(RESULT, 'prevention_shares_sha256.txt'))
    import numpy
    import rasterio
    record = ['started_utc: %s' % started.isoformat(),
              'finished_utc: %s' % datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'task elapsed minutes: %s' % recorded.get('elapsed_minutes'),
              'project: %s' % PROJECT_DIR,
              'library: %d files exported from commit %s' % (n_files, q['library_commit']),
              'hazelbean: %s at %s' % (q['hazelbean_dir'], hazelbean_head),
              'python %s, numpy %s, rasterio %s, GDAL %s' % (platform.python_version(), numpy.__version__,
                                                             rasterio.__version__, rasterio.__gdal_version__),
              'stacks: %s, %s' % (stacks['H'], stacks['Y']),
              'prevention_shares files hashed: %d' % n_outputs]
    with open(os.path.join(RESULT, 'run_record.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(record) + '\n')
    hb.log('\n'.join(record))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except sl.Mismatch as e:
        print('STOP: %s' % e, file=sys.stderr)
        sys.exit(2)

"""Every gate of the prevention_shares rerun that can be checked before the stack repair runs. No arguments.

For a login node. Writes preflight/shares_preflight.txt. Checks that project/, result/ and
heartbeat.tsv are absent, the library snapshot is the export of the pinned pushed commit and matches
its record, the pinned inputs still match, hazelbean is the pinned clean checkout, and the snapshot
imports and its task tree resolves. Reports as BLOCK the rule 37 resolution both jobs require
(../manifest/rule37_resolution.txt, naming the commit that holds the builder and launch code), and as
WAIT whether the stack repair has passed. The tree is built in preflight/probe_project, so project/
stays absent. Exits 1 on a failed gate or a BLOCK. Refuses if the report exists.

Run: python -I -B run_shares_preflight.py
"""
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT = os.path.join(HERE, 'code', 'global_invest_dev')
REPORT = os.path.join(HERE, 'preflight', 'shares_preflight.txt')
sys.path.insert(0, os.path.join(HERE, 'launch'))
import shares_launch as sl


def main():
    if os.path.exists(REPORT):
        raise sl.Mismatch('%s already exists: move it aside to rerun the preflight' % REPORT)
    q = sl.read_parameters(os.path.join(HERE, 'msi_parameters.csv'))
    lines, failures = [], []

    def gate(text, check):
        try:
            detail = check()
            lines.append('PASS  %s%s' % (text, '' if detail is None else ': %s' % detail))
        except (sl.Mismatch, ImportError, OSError) as e:
            lines.append('FAIL  %s: %s' % (text, e))
            failures.append(text)

    def absent(name):
        if os.path.exists(os.path.join(HERE, name)):
            raise sl.Mismatch('exists')

    for name in ('project', 'result', 'heartbeat.tsv'):
        gate('%s absent' % name, lambda name=name: absent(name))
    gate('library snapshot is the export of the pushed commit',
         lambda: 'commit %s on %s' % (q['library_commit'][:7], sl.check_snapshot_commit(
             os.path.join(HERE, 'code_snapshot', 'snapshot_info.txt'), q['library_commit'])['on remote branches']))
    gate('library snapshot matches its record', lambda: '%d files' % sl.verify_tree(os.path.join(HERE, 'code_snapshot', 'snapshot_files.csv'), SNAPSHOT))
    gate('pinned erosion inputs unchanged', lambda: '%d rows' % len(sl.verify_inputs(os.path.join(HERE, 'manifest', 'shares_input_manifest.csv'))))
    sys.path.insert(0, SNAPSHOT)

    def imports():
        import hazelbean as hb
        import global_invest
        from global_invest.erosion import run_erosion
        head = sl.check_hazelbean(hb, q['hazelbean_dir'], q['hazelbean_commit'])
        sl.check_imported_from(global_invest, SNAPSHOT)
        probe = hb.ProjectFlow(project_dir=os.path.join(HERE, 'preflight', 'probe_project'), run_mode='check')
        run_erosion.build_task_tree(probe)
        every, skipped = sl.task_lists(probe)
        return 'hazelbean %s, global_invest from the snapshot, tasks %s, all but prevention_shares skipped' % (head[:7], ' '.join(every))
    gate('imports and task tree', imports)
    resolution = os.path.join(os.path.dirname(HERE), 'manifest', 'rule37_resolution.txt')
    if os.path.exists(resolution) and os.path.getsize(resolution):
        with open(resolution, encoding='utf-8') as f:
            lines.append('PASS  rule 37 resolution: %s' % f.read().strip())
    else:
        lines.append('BLOCK rule 37: the stack builder and this launch code are in no repository, and both '
                     'sbatch files refuse until manifest/rule37_resolution.txt names the commit that holds them')
        failures.append('rule 37')
    try:
        stacks = sl.check_repair(os.path.dirname(HERE), os.path.join(q['base_data_dir'], sl.BANDMAP_REF))
        lines.append('PASS  stack repair passed: %s' % ', '.join(os.path.basename(s) for s in stacks.values()))
    except sl.Mismatch as e:
        lines.append('WAIT  stack repair: %s (the job refuses until it passes)' % e)
    import numpy
    import rasterio
    lines.append('      python %s, numpy %s, rasterio %s, GDAL %s' % (platform.python_version(), numpy.__version__,
                                                                    rasterio.__version__, rasterio.__gdal_version__))
    lines.append('shares preflight: %s' % ('NOT CLEARED: %s' % ', '.join(failures) if failures else 'all gates pass'))
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 1 if failures else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except sl.Mismatch as e:
        print('STOP: %s' % e, file=sys.stderr)
        sys.exit(2)

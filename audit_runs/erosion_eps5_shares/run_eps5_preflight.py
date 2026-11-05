"""Every gate of the EPS-5 run that can be checked before it starts, on MSI. Takes no arguments.

For a login node, one strip of each raster in memory at a time. Writes preflight/eps5_preflight.txt
and tables beside it. Checks that nothing from an earlier attempt exists, the snapshot is the export
of the pushed commit, the library inputs and author files match their manifests, the author's EPS-5
settings hold and differ from SES-11 only by the threshold, the stacks are the author's bytes and
never the repaired ones, the staged SDR rasters decode to the author's rasters on the same grid, the
library's SES-11 rasters and tables equal the author's, the policy file follows from SES-11, and the
publish-only pass resolves threshold 5 and these inputs. The identity gate has no tolerance and fails on
the staged rasters, and it stays failed on the record: the next gate checks that its counts equal the ones
on file in prediction/ and that the prediction's falsifiers held. Reports the rule 37 resolution and the
raster gate decision as BLOCK until each exists. Exits 1 on a failed gate or a BLOCK. Refuses if the report
exists. Runs no prevention_shares and submits nothing.

Run: python -I -B run_eps5_preflight.py
"""
import json
import os
import platform
import shutil
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT = os.path.join(HERE, 'code', 'global_invest_dev')
OUT = os.path.join(HERE, 'preflight')
PREDICTION = os.path.join(HERE, 'prediction')
REPORT = os.path.join(OUT, 'eps5_preflight.txt')
sys.path.insert(0, os.path.join(HERE, 'launch'))
import eps5_compare as ec
import eps5_gates as eg
import eps5_prediction as epr
import eps5_probe as ep

CLEAN = (ec.BYTE_IDENTICAL, ec.VALUES_IDENTICAL)
RASTERS = ['ps_onfarm_cropland_severe.tif', 'ps_upstream_cropland_severe.tif', 'ps_combined_union_cropland_severe.tif']
LONG = 'country_crop_protected_production_long.csv'
OTHER_TABLES = [('threshold_policy.csv', ['ISO3']), ('country_ps_diagnostics.csv', ['ISO3']), ('area_conservation_audit.csv', ['band', 'crop']), ('elasticity_audit.csv', ['crop_key'])]


def main():
    if os.path.exists(REPORT):
        raise eg.Mismatch('%s already exists: move it aside to rerun the preflight' % REPORT)
    os.makedirs(OUT, exist_ok=True)
    q = eg.read_parameters(os.path.join(HERE, 'msi_parameters.csv'))
    settings = eg.read_rows(os.path.join(HERE, 'eps5_settings.csv'))
    pins = eg.read_rows(os.path.join(HERE, 'author_pins.csv'))
    role = {r['role']: os.path.join(q['author_archive_dir'], r['relative_path']) for r in pins}
    base, archive, library_ses11 = q['base_data_dir'], q['author_archive_dir'], q['library_ses11_dir']
    sl = ep.load_shared_launch(SNAPSHOT)
    lines, failures = [], []

    def gate(text, check):
        try:
            detail = check()
            lines.append('PASS  %s%s' % (text, '' if detail is None else ': %s' % detail))
        except (eg.Mismatch, sl.Mismatch, ImportError, OSError, KeyError, ValueError) as e:
            lines.append('FAIL  %s: %s' % (text, e))
            failures.append(text)

    def absent(name):
        if os.path.exists(os.path.join(HERE, name)):
            raise eg.Mismatch('exists')

    for name in ('project', 'result', 'heartbeat.tsv', 'comparison'):
        gate('%s absent' % name, lambda name=name: absent(name))

    def separate():
        eg.refuse_repaired([HERE] + [r['path'] for r in eg.read_rows(os.path.join(HERE, 'manifest', 'eps5_input_manifest.csv'))])
        return 'this folder and every pinned input are outside the repair folder'
    gate('repaired stacks kept out', separate)
    gate('library snapshot is the export of the pushed commit',
         lambda: 'commit %s on %s' % (q['library_commit'][:7], sl.check_snapshot_commit(os.path.join(HERE, 'code_snapshot', 'snapshot_info.txt'), q['library_commit'])['on remote branches']))
    gate('library snapshot matches its record', lambda: '%d files' % sl.verify_tree(os.path.join(HERE, 'code_snapshot', 'snapshot_files.csv'), SNAPSHOT))
    gate('pinned library inputs unchanged', lambda: '%d rows' % len(sl.verify_inputs(os.path.join(HERE, 'manifest', 'eps5_input_manifest.csv'))))
    gate('pinned author files unchanged', lambda: '%d rows' % len(sl.verify_inputs(os.path.join(HERE, 'manifest', 'eps5_author_manifest.csv'))))

    def author_settings():
        with open(os.path.join(archive, 'results', 'eps5_sensitivity', 'manifest.json'), encoding='utf-8') as f:
            eps5 = json.load(f)
        with open(os.path.join(archive, 'results', 'ses11_headline', 'manifest.json'), encoding='utf-8') as f:
            ses11 = json.load(f)
        eg.check_author_settings(eps5, settings)
        eg.check_manifests_differ_only_by_threshold(eps5, ses11)
        with open(os.path.join(archive, 'results', 'eps5_sensitivity', 'run_metadata.txt'), encoding='utf-8') as f:
            text = f.read()
        for line in ('Default threshold (high): 5 t/ha/yr', 'Low threshold (special cases): 2 t/ha/yr'):
            if line not in text:
                raise eg.Mismatch('run_metadata.txt has no line %r' % line)
        with open(os.path.join(archive, 'repro_test_valuation.sh'), encoding='utf-8') as f:
            script = f.read()
        for line in ('eps5)  THRESH=5;', 'GEP_EROSION_THRESH_LOW=2', 'GEP_EROSION_ALL_TOUCHED=false'):
            if line not in script:
                raise eg.Mismatch('repro_test_valuation.sh has no %r' % line)
        return 'threshold 5 and 2, area 50000, elevation 250, EPSG 8857, severe filter on, all_touched off. The run tag in the author EPS-5 manifest still reads %s' % eps5['run_tag']
    gate('author EPS-5 settings', author_settings)

    def stacks():
        eg.check_original_stacks(eg.read_rows(os.path.join(HERE, 'manifest', 'eps5_input_manifest.csv')), pins)
        return 'area stack, yield stack and band map are the bytes in the author archive'
    gate('original stacks', stacks)

    def author_read_paths():
        with open(os.path.join(archive, 'results', 'eps5_sensitivity', 'manifest.json'), encoding='utf-8') as f:
            inputs = json.load(f)['inputs']
        rows = []
        for key, name in (('usle_path', 'usle'), ('avoided_path', 'avoided'), ('ups_path', 'upstream'), ('boundary_gpkg', 'boundary'), ('dem_path_for_thresholds', 'dem'),
                          ('yield_stack', 'yield_stack'), ('area_stack', 'area_stack'), ('bandmap_csv', 'bandmap'), ('elasticity_csv', 'elasticity')):
            path, present = inputs[key], os.path.exists(inputs[key])
            rows.append({'manifest_key': key, 'path_the_author_run_read': path, 'present': present, 'sha256': ec.sha256_file(path) if present else '',
                         'modified': pd.Timestamp(os.path.getmtime(path), unit='s').isoformat() if present else '', 'archive_sha256': ec.sha256_file(role[name])})
        frame = pd.DataFrame(rows).assign(equals_archive=lambda f: f['sha256'] == f['archive_sha256'])
        frame.to_csv(os.path.join(OUT, 'author_read_paths.csv'), index=False)
        different = frame.loc[frame['present'] & ~frame['equals_archive'], 'manifest_key'].tolist()
        if different:
            raise eg.Mismatch('the file the author run read now differs from the archive for %s, see author_read_paths.csv' % different)
        return '%d of %d still present, each equal to the archive' % (frame['present'].sum(), len(frame))
    gate('the paths the author EPS-5 run read', author_read_paths)

    def sdr_rasters():
        rows = []
        for name, file_name in (('usle', 'usle_fresh.tif'), ('avoided', 'avoided_erosion_fresh.tif'), ('upstream', 'upstream_prevention_share_fresh.tif')):
            rows.append(dict(ec.compare_rasters(os.path.join(base, sl.SDR_REF_DIR, file_name), role[name]), raster=name))
        frame = pd.DataFrame(rows)
        frame.to_csv(os.path.join(OUT, 'sdr_rasters_library_vs_author.csv'), index=False)
        bad = frame.loc[~frame['verdict'].isin(CLEAN), ['raster', 'verdict']].values.tolist()
        if bad:
            raise eg.Mismatch('%s, see sdr_rasters_library_vs_author.csv' % bad)
        return '; '.join('%s %s, %dx%d, %d valid cells' % (r['raster'], r['verdict'], r['width'], r['height'], r['valid_left']) for _, r in frame.iterrows())
    gate('staged SDR rasters against the author rasters', sdr_rasters)

    def prediction_on_file():
        with open(os.path.join(PREDICTION, 'prediction_sha256.txt'), encoding='utf-8') as f:
            rows = [line.split() for line in f if line.strip()]
        for digest, name in rows:
            if ec.sha256_file(os.path.join(PREDICTION, name)) != digest:
                raise eg.Mismatch('prediction/%s differs from prediction_sha256.txt' % name)
        with open(os.path.join(PREDICTION, 'sdr_reach_summary.txt'), encoding='utf-8') as f:
            if not f.readline().startswith('falsifiers: HOLD'):
                raise eg.Mismatch('the prediction on file does not record that its falsifiers held')
        with open(os.path.join(OUT, 'sdr_rasters_library_vs_author.csv'), 'rb') as f, open(os.path.join(PREDICTION, 'sdr_rasters_library_vs_author.csv'), 'rb') as g:
            if f.read() != g.read():
                raise eg.Mismatch('the raster identity counts differ from the ones the prediction rests on')
        cells, bounds = epr.read_prediction(PREDICTION)
        expected = pd.read_csv(os.path.join(PREDICTION, epr.EXPECTED), encoding='utf-8-sig')
        expected = expected[expected['run'] == 'eps5']
        for r in expected.itertuples():
            for column in ('protected_production_tons', 'share_protected_production'):
                if (r.component, r.ISO3, r.crop, column) not in bounds:
                    raise eg.Mismatch('no magnitude bound for %s %s %s %s' % (r.component, r.ISO3, r.crop, column))
        moved = {part: len(epr.predicted_raster(cells, part)[1]) for part in epr.PARTS}
        return '%d files, identity counts unchanged, cells that must differ at EPS-5: %s, table rows that may move: %d, countries %s, magnitude bounds: %d' % (len(rows), moved, len(expected), ' '.join(sorted(expected['ISO3'].unique())), len(bounds))
    gate('quantified prediction on file for the failed identity gate', prediction_on_file)

    def ses11_control():
        result = ec.compare_long(os.path.join(library_ses11, LONG), role['ses11_long'])
        if result['verdict'] != ec.BYTE_IDENTICAL:
            raise eg.Mismatch('long table verdict %s' % result['verdict'])
        rows = [dict(ec.compare_rasters(os.path.join(library_ses11, name), os.path.join(archive, 'results', 'ses11_headline', name)), raster=name) for name in RASTERS]
        frame = pd.DataFrame(rows)
        frame.to_csv(os.path.join(OUT, 'ses11_rasters_library_vs_author.csv'), index=False)
        bad = frame.loc[~frame['verdict'].isin(CLEAN), ['raster', 'verdict']].values.tolist()
        if bad:
            raise eg.Mismatch('%s, see ses11_rasters_library_vs_author.csv' % bad)
        return 'long table BYTE_IDENTICAL on %d rows, rasters %s' % (result['summary']['keys_shared'], ' '.join(frame['verdict']))
    gate('SES-11 control on MSI files', ses11_control)

    def ses11_other_tables():
        rows = []
        for name, keys in OTHER_TABLES:
            result = ec.compare_tables(os.path.join(library_ses11, name), os.path.join(archive, 'results', 'ses11_headline', name), keys)
            rows.append({'table': name, 'verdict': result['verdict'], 'keys_shared': result['summary'].get('keys_shared'), 'values_that_differ': len(result['rows_that_differ']),
                         'text_cells_that_differ': len(result['text_differences'])})
            for part in ('text_differences', 'rows_that_differ'):
                if not result[part].empty:
                    result[part].to_csv(os.path.join(OUT, 'ses11_%s_%s.csv' % (name[:-4], part)), index=False, encoding='utf-8-sig', float_format='%.17g')
        frame = pd.DataFrame(rows)
        frame.to_csv(os.path.join(OUT, 'ses11_other_tables_library_vs_author.csv'), index=False)
        return 'baseline to read the EPS-5 comparison against: ' + '; '.join('%s %s, %d values, %d text cells' % (r['table'], r['verdict'], r['values_that_differ'], r['text_cells_that_differ']) for _, r in frame.iterrows())
    gate('SES-11 other tables', ses11_other_tables)

    def policy():
        with open(role['ses11_threshold_policy'], 'rb') as f:
            moved, count = eg.policy_at_threshold(f.read())
        with open(role['eps5_threshold_policy'], 'rb') as f:
            if f.read() != moved:
                raise eg.Mismatch('the author EPS-5 policy is not the SES-11 policy with default-high rows at 5.0')
        return '%d default-high rows at 5.0, every other byte as in SES-11' % count
    gate('author threshold policy follows from SES-11', policy)

    sys.path.insert(0, SNAPSHOT)

    def publish_only():
        import hazelbean as hb
        import global_invest
        from global_invest.erosion import run_erosion
        head = sl.check_hazelbean(hb, q['hazelbean_dir'], q['hazelbean_commit'])
        sl.check_imported_from(global_invest, SNAPSHOT)
        sl.check_imported_from(run_erosion, SNAPSHOT)
        project = os.path.join(OUT, 'probe_project')
        if os.path.exists(project):
            shutil.rmtree(project)
        every, _ = ep.publish_only_pass(hb, run_erosion, sl, eg, project, base, settings)
        return 'hazelbean %s, snapshot imported, tasks %s, threshold 5.0 and the staged rasters and original stacks resolve' % (head[:7], ' '.join(every))
    gate('publish-only pass', publish_only)

    free = shutil.disk_usage(HERE).free / 1e12
    lines.append('      scratch free %.0f TB. Expected outputs under 0.2 GB. Expected run 10 h, 26 GB, see README' % free)
    resolution = os.path.join(HERE, 'manifest', 'rule37_resolution.txt')
    if os.path.exists(resolution) and os.path.getsize(resolution):
        with open(resolution, encoding='utf-8') as f:
            lines.append('PASS  rule 37 resolution: %s' % f.read().strip())
    else:
        lines.append('BLOCK rule 37: this launch code is in no repository, and the sbatch refuses until manifest/rule37_resolution.txt names the pushed commit that holds it')
        failures.append('rule 37')
    decision = os.path.join(HERE, 'manifest', 'raster_gate_decision.txt')
    digest = ec.sha256_file(os.path.join(PREDICTION, 'prediction_sha256.txt'))
    if os.path.exists(decision) and digest in open(decision, encoding='utf-8').read():
        with open(decision, encoding='utf-8') as f:
            lines.append('PASS  raster gate decision: %s' % f.read().strip())
    else:
        lines.append('BLOCK raster gate decision: the sbatch refuses until manifest/raster_gate_decision.txt records the decision to run against the prediction %s' % digest[:16])
        failures.append('raster gate decision')
    import numpy
    import rasterio
    lines.append('      python %s, numpy %s, rasterio %s, GDAL %s' % (platform.python_version(), numpy.__version__, rasterio.__version__, rasterio.__gdal_version__))
    lines.append('eps5 preflight: %s' % ('NOT CLEARED: %s' % ', '.join(failures) if failures else 'all gates pass'))
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 1 if failures else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except eg.Mismatch as e:
        print('STOP: %s' % e, file=sys.stderr)
        sys.exit(2)

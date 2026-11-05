"""Every EPS-5 check that the laptop's own files can settle before the MSI preflight. Takes no arguments.

Writes preflight/local_checks.txt and its tables. Controls on the saved shares: the library's SES-11
table against the author's must be byte identical, and the author's SES-11 against EPS-5 must show
the comparison finds differences. Also the threshold policy predicted from the SES-11 file against
the author's EPS-5 hash, the library inputs against the author pins, the two country boundary
files, the library snapshot against the frozen code of the SES-11 run, and the publish-only pass of
the run on local base data. Nothing runs prevention_shares. Refuses if the report exists.

Run: python -I -B run_eps5_local_checks.py
"""
import hashlib
import os
import shutil
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT = os.path.join(HERE, 'code', 'global_invest_dev')
OUT = os.path.join(HERE, 'preflight')
REPORT = os.path.join(OUT, 'local_checks.txt')
sys.path.insert(0, os.path.join(HERE, 'launch'))
import eps5_compare as ec
import eps5_gates as eg
import eps5_probe as ep

# The frozen copies of the code that produced the SES-11 match, against the snapshot this run uses.
FROZEN_FILES = ['__init__.py', 'utilities.py', 'erosion/erosion_tasks.py', 'erosion/erosion_functions.py', 'erosion/erosion_initialize.py', 'erosion/run_erosion.py']


def main():
    if os.path.exists(REPORT):
        raise eg.Mismatch('%s already exists: move it aside to rerun' % REPORT)
    os.makedirs(OUT, exist_ok=True)
    q = eg.read_parameters(os.path.join(HERE, 'local_parameters.csv'))
    settings = eg.read_rows(os.path.join(HERE, 'eps5_settings.csv'))
    pins = eg.read_rows(os.path.join(HERE, 'author_pins.csv'))
    pin = {r['role']: r['sha256'] for r in pins}
    lines, failures = [], []

    def gate(text, check):
        try:
            detail = check()
            lines.append('PASS  %s%s' % (text, '' if detail is None else ': %s' % detail))
        except (eg.Mismatch, ImportError, OSError, KeyError, ValueError) as e:
            lines.append('FAIL  %s: %s' % (text, e))
            failures.append(text)

    saved, library_ses11 = q['saved_shares_dir'], q['library_ses11_dir']
    long_name = 'country_crop_protected_production_long.csv'

    def saved_shares_are_the_pinned_files():
        for role, folder in (('eps5_long', 'eps5_sensitivity'), ('ses11_long', 'ses11_headline')):
            if ec.sha256_file(os.path.join(saved, folder, long_name)) != pin[role]:
                raise eg.Mismatch('the saved %s is not the pinned author file' % folder)
        return 'EPS-5 %s, SES-11 %s' % (pin['eps5_long'][:8], pin['ses11_long'][:8])
    gate('saved author shares are the pinned files', saved_shares_are_the_pinned_files)

    def positive_control():
        result = ec.compare_long(os.path.join(library_ses11, long_name), os.path.join(saved, 'ses11_headline', long_name))
        if result['verdict'] != ec.BYTE_IDENTICAL:
            raise eg.Mismatch('verdict %s' % result['verdict'])
        return '%d rows, verdict %s' % (result['summary']['keys_shared'], result['verdict'])
    gate('positive control, library SES-11 against author SES-11', positive_control)

    def negative_control():
        result = ec.compare_long(os.path.join(saved, 'ses11_headline', long_name), os.path.join(saved, 'eps5_sensitivity', long_name))
        if result['verdict'] != ec.DIFFERENT or result['rows_that_differ'].empty:
            raise eg.Mismatch('the comparison found no difference between two thresholds')
        result['columns'].to_csv(os.path.join(OUT, 'negative_control_author_ses11_vs_eps5_columns.csv'), index=False, float_format='%.17g')
        s = result['summary']
        by = result['rows_that_differ'].groupby('column').size().to_dict()
        if set(by) != {'protected_production_tons', 'share_protected_production'} or s['keys_left_only'] or s['keys_right_only']:
            raise eg.Mismatch('the threshold moved something it cannot reach: %s, keys one side %d' % (by, s['keys_left_only'] + s['keys_right_only']))
        return 'keys identical at %d rows, so keys, elasticity_used and total_production_tons at EPS-5 already equal the library SES-11 table. Differing values %s' % (s['keys_shared'], by)
    gate('negative control, author SES-11 against author EPS-5', negative_control)

    def predicted_policy():
        with open(q['author_ses11_policy_path'], 'rb') as f:
            data = f.read()
        if hashlib.sha256(data).hexdigest() != pin['ses11_threshold_policy']:
            raise eg.Mismatch('the held SES-11 policy is not the pinned author file')
        moved, count = eg.policy_at_threshold(data)
        digest = hashlib.sha256(moved).hexdigest()
        if digest != pin['eps5_threshold_policy']:
            raise eg.Mismatch('SES-11 moved to 5.0 on %d rows has SHA-256 %s, the author EPS-5 policy %s' % (count, digest, pin['eps5_threshold_policy']))
        with open(os.path.join(library_ses11, 'threshold_policy.csv'), 'rb') as f:
            library_moved, library_count = eg.policy_at_threshold(f.read())
        predicted = os.path.join(OUT, 'library_threshold_policy_predicted_at_eps5.csv')
        with open(predicted, 'wb') as f:
            f.write(library_moved)
        author = os.path.join(OUT, 'author_threshold_policy_at_eps5_from_ses11.csv')
        with open(author, 'wb') as f:
            f.write(moved)
        result = ec.compare_tables(predicted, author, ['ISO3'])
        return ('author SES-11 with %d default-high rows at 5.0 hashes to the author EPS-5 policy %s. Library prediction %d rows moved, numeric differences %d, text differences %s'
                % (count, digest[:8], library_count, len(result['rows_that_differ']), result['text_differences'][['ISO3', 'left', 'right']].values.tolist()))
    gate('threshold policy at EPS-5', predicted_policy)

    def input_identity():
        erosion = os.path.join(q['base_data_dir'], 'global_invest', 'erosion')
        table = [('dem', os.path.join(erosion, 'global_dem_reproj.tif')), ('elasticity', os.path.join(erosion, 'elasticity_crops_fao_revised.csv')),
                 ('bandmap', os.path.join(erosion, 'spam2020_bandmap.csv')), ('area_stack', os.path.join(erosion, 'spam2020_harvested_area_stack_TA.tif')),
                 ('yield_stack', os.path.join(erosion, 'spam2020_yield_stack_TA.tif')),
                 ('usle', os.path.join(erosion, 'author_repo_2026_10', 'full_chain_618f67e', 'usle_fresh.tif')),
                 ('avoided', os.path.join(erosion, 'author_repo_2026_10', 'full_chain_618f67e', 'avoided_erosion_fresh.tif')),
                 ('upstream', os.path.join(erosion, 'author_repo_2026_10', 'full_chain_618f67e', 'upstream_prevention_share_fresh.tif')),
                 ('boundary', os.path.join(q['base_data_dir'], 'cartographic', 'ee', 'ee_r250.gpkg'))]
        rows = [{'role': role, 'library_path': path, 'library_sha256': ec.sha256_file(path), 'author_sha256': pin[role]} for role, path in table]
        frame = pd.DataFrame(rows).assign(bytes_identical=lambda f: f['library_sha256'] == f['author_sha256'])
        frame.to_csv(os.path.join(OUT, 'library_inputs_vs_author_pins.csv'), index=False)
        must = frame[frame['role'].isin(['dem', 'elasticity', 'bandmap', 'area_stack', 'yield_stack'])]
        if not must['bytes_identical'].all():
            raise eg.Mismatch('%s differ from the author bytes' % must.loc[~must['bytes_identical'], 'role'].tolist())
        eg.refuse_repaired(frame['library_path'])
        return 'identical: %s. Different bytes, decoded by the MSI preflight or below: %s' % (
            ' '.join(frame.loc[frame['bytes_identical'], 'role']), ' '.join(frame.loc[~frame['bytes_identical'], 'role']))
    gate('library inputs against author pins', input_identity)

    def boundary():
        import geopandas as gpd
        if ec.sha256_file(q['author_boundary_path']) != pin['boundary']:
            raise eg.Mismatch('the held boundary file is not the pinned author file')
        library = gpd.read_file(os.path.join(q['base_data_dir'], 'cartographic', 'ee', 'ee_r250.gpkg'))
        author = gpd.read_file(q['author_boundary_path'])
        def iso(frame):
            column = next(c for c in frame.columns if c.lower() in ('iso3', 'iso3_r250_label'))
            return frame.assign(_iso=frame[column].astype(str).str.upper()).set_index('_iso')
        a, b = iso(library), iso(author)
        if a.index.duplicated().any() or b.index.duplicated().any():
            raise eg.Mismatch('a boundary file repeats a country')
        shared = a.index.intersection(b.index)
        if len(shared) != len(a) or len(shared) != len(b):
            raise eg.Mismatch('the boundary files hold different countries: %s' % sorted(set(a.index) ^ set(b.index)))
        area_a, area_b = a.loc[shared].to_crs(8857).area, b.loc[shared].to_crs(8857).area
        ratio = (area_a / area_b).rename('library_over_author')
        pd.DataFrame({'library_km2': area_a / 1e6, 'author_km2': area_b / 1e6, 'library_over_author': ratio}).to_csv(os.path.join(OUT, 'boundary_areas_in_analysis_crs.csv'), index_label='iso3')
        return ('both %d countries. Stored in %s and %s, so raw coordinates differ on every row. Area in the analysis CRS, library over author, from %.6f (%s) to %.6f (%s). '
                'SES-11 total production is byte identical by country and crop, which a country raster differing on production cells would disturb'
                % (len(shared), a.crs.to_string(), b.crs.to_string(), ratio.min(), ratio.idxmin(), ratio.max(), ratio.idxmax()))
    gate('country boundary files', boundary)

    def code_path():
        different = [name for name in FROZEN_FILES
                     if ec.sha256_file(os.path.join(q['frozen_ses11_code_dir'], name)) != ec.sha256_file(os.path.join(SNAPSHOT, 'global_invest', name))]
        if different:
            raise eg.Mismatch('these differ from the code of the SES-11 run: %s' % different)
        return '%d files byte identical: %s' % (len(FROZEN_FILES), ' '.join(FROZEN_FILES))
    gate('library snapshot against the frozen SES-11 code', code_path)

    def publish_only():
        sys.path.insert(0, SNAPSHOT)
        import hazelbean as hb
        import global_invest
        from global_invest.erosion import run_erosion
        sl = ep.load_shared_launch(SNAPSHOT)
        sl.check_imported_from(global_invest, SNAPSHOT)
        project = os.path.join(OUT, 'local_probe_project')
        if os.path.exists(project):
            shutil.rmtree(project)
        every, skipped = ep.publish_only_pass(hb, run_erosion, sl, eg, project, q['base_data_dir'], settings)
        return 'tasks %s, all but prevention_shares skipped in the run. Threshold 5.0 and the other pins reach the task, the staged rasters and original stacks resolve' % ' '.join(every)
    gate('publish-only pass on local base data', publish_only)

    lines.append('local checks: %s' % ('NOT CLEARED: %s' % ', '.join(failures) if failures else 'all pass'))
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

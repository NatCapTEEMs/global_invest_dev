"""Compare the library's EPS-5 prevention shares with the author's, on MSI after the job. Takes no arguments.

For a login node, one strip of each raster in memory at a time. Stops unless the job's outputs match
result/prevention_shares_sha256.txt and the author files match their manifest. Compares every
country, crop and component key, the missing and zero masks, tonnes and shares, then the component
files, the threshold policy, the country diagnostics, the area and elasticity audits and the three
prevention-share rasters. There is no tolerance: a table is byte identical, values identical or
different, and every difference is written out. Then checks each output against prediction/, which says
what the failed raster identity gate must do to the shares. comparison/prediction_check.csv reports location and
magnitude apart for each output: raster cells to the last bit, table values within a bound derived from float
rounding. A difference outside the predicted cells, keys and columns, a broken structure or mask, or a value over
its bound falsifies the prediction. Writes comparison/, refuses if it exists, and
reads nothing from the repaired stacks. The exit code says the comparison ran, not what it found.

Run: python -I -B run_eps5_compare.py
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT = os.path.join(HERE, 'code', 'global_invest_dev')
SHARES = os.path.join(HERE, 'project', 'gep_erosion_eps5_library', 'intermediate', 'prevention_shares')
OUT = os.path.join(HERE, 'comparison')
PREDICTION = os.path.join(HERE, 'prediction')
sys.path.insert(0, os.path.join(HERE, 'launch'))
import eps5_compare as ec
import eps5_gates as eg
import eps5_prediction as epr
import eps5_probe as ep

TABLES = [('long', 'country_crop_protected_production_long.csv', ec.LONG_KEYS, 'component'),
          ('onfarm', 'country_crop_protected_production_onfarm.csv', ec.LONG_KEYS, None),
          ('upstream', 'country_crop_protected_production_upstream.csv', ec.LONG_KEYS, None),
          ('combined', 'country_crop_protected_production_combined.csv', ec.LONG_KEYS, None),
          ('threshold_policy', 'threshold_policy.csv', ['ISO3'], None),
          ('country_ps_diagnostics', 'country_ps_diagnostics.csv', ['ISO3'], None),
          ('area_conservation_audit', 'area_conservation_audit.csv', ['band', 'crop'], None),
          ('elasticity_audit', 'elasticity_audit.csv', ['crop_key'], None)]
RASTERS = ['ps_onfarm_cropland_severe.tif', 'ps_upstream_cropland_severe.tif', 'ps_combined_union_cropland_severe.tif']


def main():
    q = eg.read_parameters(os.path.join(HERE, 'msi_parameters.csv'))
    sl = ep.load_shared_launch(SNAPSHOT)
    if os.path.exists(OUT):
        raise eg.Mismatch('%s already exists: move it aside to rerun' % OUT)
    sl.verify_tree_hashes(SHARES, os.path.join(HERE, 'result', 'prevention_shares_sha256.txt'))
    sl.verify_inputs(os.path.join(HERE, 'manifest', 'eps5_author_manifest.csv'))
    author = os.path.join(q['author_archive_dir'], 'results', 'eps5_sensitivity')
    eg.refuse_repaired([SHARES, author])
    os.makedirs(OUT)
    verdicts = []
    results = {}
    for label, name, keys, group_by in TABLES:
        result = ec.compare_tables(os.path.join(SHARES, name), os.path.join(author, name), keys, group_by=group_by)
        results[label] = result
        with open(os.path.join(OUT, label + '_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(dict(result['summary'], verdict=result['verdict']), f, indent=1)
        for part in ('columns', 'text_differences', 'keys_one_side', 'rows_that_differ'):
            if not result[part].empty:
                result[part].to_csv(os.path.join(OUT, '%s_%s.csv' % (label, part)), index=False, encoding='utf-8-sig', float_format='%.17g')
        verdicts.append({'item': name, 'verdict': result['verdict'], 'keys_shared': result['summary'].get('keys_shared'), 'keys_library_only': result['summary'].get('keys_left_only'),
                         'keys_author_only': result['summary'].get('keys_right_only'), 'values_that_differ': len(result['rows_that_differ']), 'text_cells_that_differ': len(result['text_differences'])})
    rasters = pd.DataFrame([dict(ec.compare_rasters(os.path.join(SHARES, name), os.path.join(author, name)), raster=name) for name in RASTERS])
    rasters.to_csv(os.path.join(OUT, 'rasters.csv'), index=False, encoding='utf-8-sig')
    for _, r in rasters.iterrows():
        verdicts.append({'item': r['raster'], 'verdict': r['verdict'], 'values_that_differ': r.get('values_differ'), 'valid_mask_mismatch': r.get('valid_mismatch'), 'zero_mask_mismatch': r.get('zero_mismatch')})
    table = pd.DataFrame(verdicts)
    table.to_csv(os.path.join(OUT, 'verdicts.csv'), index=False, encoding='utf-8-sig')
    print(table.to_string(index=False))
    cells, bounds = epr.read_prediction(PREDICTION)
    clean = (ec.BYTE_IDENTICAL, ec.VALUES_IDENTICAL)
    checks = [epr.check_protected_table(TABLES[0][1], results['long'], bounds), epr.check_protected_table(TABLES[1][1], results['onfarm'], bounds),
              epr.check_clean(TABLES[2][1], results['upstream'], clean), epr.check_clean(TABLES[3][1], results['combined'], clean),
              epr.check_named_text(TABLES[4][1], results['threshold_policy'], os.path.join(PREDICTION, 'ses11_threshold_policy_text_differences.csv')),
              epr.check_named_text(TABLES[5][1], results['country_ps_diagnostics'], os.path.join(PREDICTION, 'ses11_country_ps_diagnostics_text_differences.csv'), bounds),
              epr.check_baseline_values(TABLES[6][1], results['area_conservation_audit'], os.path.join(PREDICTION, 'ses11_area_conservation_audit_rows_that_differ.csv')),
              epr.check_clean(TABLES[7][1], results['elasticity_audit'], clean)]
    checks += [epr.check_raster(part, os.path.join(SHARES, name), os.path.join(author, name), cells) for part, name in epr.PARTS.items()]
    frame = pd.DataFrame(checks)
    frame.to_csv(os.path.join(OUT, 'prediction_check.csv'), index=False, encoding='utf-8-sig')
    failed = frame.loc[~frame['as_predicted'], 'item'].tolist()
    within = int((frame['magnitude'] == epr.WITHIN_BOUND).sum())
    print('prediction: %s' % ('HOLDS on all %d outputs, %d of them only to the derived float bound and not to the last bit' % (len(frame), within) if not failed else 'FALSIFIED by %s' % ', '.join(failed)))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except eg.Mismatch as e:
        print('STOP: %s' % e, file=sys.stderr)
        sys.exit(2)

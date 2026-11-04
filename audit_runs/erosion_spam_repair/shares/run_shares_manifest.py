"""Pin every erosion input the prevention_shares rerun can read on MSI. Takes no arguments.

For a login node. Reads msi_parameters.csv and the snapshot's es_parameters.csv, and writes
manifest/shares_input_manifest.csv: SHA-256 and size of every erosion *_path row and the country
table present under base_data_dir, and 'absent' for rows that resolve nowhere. The three staged
SDR rasters must also match their staging receipt. Refuses to replace an existing manifest.

Run: python -I -B run_shares_manifest.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'launch'))
import shares_launch as sl

ES_PARAMETERS = os.path.join(HERE, 'code', 'global_invest_dev', 'global_invest', 'input_template', 'es_parameters.csv')


def main():
    q = sl.read_parameters(os.path.join(HERE, 'msi_parameters.csv'))
    rows = sl.make_input_manifest(ES_PARAMETERS, q['base_data_dir'], os.path.join(HERE, 'manifest', 'shares_input_manifest.csv'))
    present = [r for r in rows if r['status'] == 'present']
    print('shares manifest: %d inputs pinned (%.2f GB), %d absent: %s'
          % (len(present), sum(int(r['bytes']) for r in present) / 1e9, len(rows) - len(present),
             ' '.join(r['parameter'] for r in rows if r['status'] == 'absent')))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except sl.Mismatch as e:
        print('STOP: %s' % e, file=sys.stderr)
        sys.exit(2)

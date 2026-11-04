"""Pin what the local per-crop valuation reads besides the returned shares. Takes no arguments.

Local. Reads local_parameters.csv and writes manifest/valuation_input_manifest.csv: crop
provision's per-crop table (found from the project the way erosion_crop_values_relative_file
finds it), the FAO item to SPAM 2020 correspondence, the elasticity table, the band map and the
country table.
Refuses to replace an existing manifest.

Run: python -I -B run_valuation_manifest.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'launch'))
import shares_launch as sl


def main():
    q = sl.read_parameters(os.path.join(HERE, 'local_parameters.csv'))
    named = [('crop_values', os.path.normpath(os.path.join(q['project_dir'], sl.VALUATION_REFS[0][1])))]
    named += [(name, os.path.join(q['base_data_dir'], ref)) for name, ref in sl.VALUATION_REFS[1:]]
    named += [('country_table', os.path.join(q['base_data_dir'], ref)) for ref in sl.COUNTRY_REFS]
    rows = sl.make_file_manifest(named, os.path.join(HERE, 'manifest', 'valuation_input_manifest.csv'))
    print('valuation manifest: %d inputs pinned' % len(rows))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except sl.Mismatch as e:
        print('STOP: %s' % e, file=sys.stderr)
        sys.exit(2)

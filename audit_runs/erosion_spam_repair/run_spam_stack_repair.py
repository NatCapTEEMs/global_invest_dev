"""Build the repaired SPAM 2020 stacks and check all bands against the raw files. Takes no arguments.

Reads spam_stack_repair_parameters.csv and manifest/input_manifest.csv beside this file. Writes the
two stacks, build_receipt.csv and outputs_sha256.txt into output/ (through output.partial/), then
check/stack_checks_by_band.csv. Every input is read from bytes verified against the manifest.
Exits 2 on a mismatch, 1 if any band fails the check, 0 when all pass. Writes nothing outside
this folder.

Run: python -I -B run_spam_stack_repair.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'code'))
import spam_stack_repair as ssr


def main():
    q = ssr.read_parameters(os.path.join(HERE, 'spam_stack_repair_parameters.csv'))
    manifest = os.path.join(HERE, 'manifest', 'input_manifest.csv')
    output = os.path.join(HERE, 'output')
    ssr.build(q['raw_root'], q['bandmap_path'], q['supplied_area_stack_path'], manifest, output)
    return ssr.check(q['raw_root'], q['bandmap_path'], q['supplied_area_stack_path'], q['supplied_yield_stack_path'],
                     manifest, output, os.path.join(HERE, 'check', 'stack_checks_by_band.csv'))


if __name__ == '__main__':
    try:
        sys.exit(main())
    except ssr.Mismatch as e:
        print('STOP: %s' % e, file=sys.stderr)
        sys.exit(2)

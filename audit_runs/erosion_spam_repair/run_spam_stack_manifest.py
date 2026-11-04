"""Pin every input of the SPAM 2020 stack repair by SHA-256. Takes no arguments.

Reads spam_stack_repair_parameters.csv beside this file and writes manifest/input_manifest.csv:
the band map, the two supplied stacks and the 138 raw files, with path, size, mtime and SHA-256.
Refuses to replace an existing manifest.

Run: python -I -B run_spam_stack_manifest.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'code'))
import spam_stack_repair as ssr


def main():
    q = ssr.read_parameters(os.path.join(HERE, 'spam_stack_repair_parameters.csv'))
    return ssr.make_manifest(q['raw_root'], q['bandmap_path'], q['supplied_area_stack_path'], q['supplied_yield_stack_path'],
                             os.path.join(HERE, 'manifest', 'input_manifest.csv'))


if __name__ == '__main__':
    try:
        sys.exit(main())
    except ssr.Mismatch as e:
        print('STOP: %s' % e, file=sys.stderr)
        sys.exit(2)

"""Preflight for the SPAM 2020 stack repair: headers only, safe on a login node. Takes no arguments.

Reads spam_stack_repair_parameters.csv beside this file, writes preflight/preflight.txt and
preflight/raw_grid_check.csv, and exits 1 if any check fails. Refuses if the report exists.

Run: python -I -B run_spam_stack_preflight.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'code'))
import spam_stack_repair as ssr


def main():
    q = ssr.read_parameters(os.path.join(HERE, 'spam_stack_repair_parameters.csv'))
    return ssr.preflight(q['raw_root'], q['bandmap_path'], q['supplied_area_stack_path'], q['supplied_yield_stack_path'],
                         os.path.join(HERE, 'output'), os.path.join(HERE, 'preflight', 'preflight.txt'),
                         os.path.join(HERE, 'preflight', 'raw_grid_check.csv'), int(q['expect_crops']))


if __name__ == '__main__':
    try:
        sys.exit(main())
    except ssr.Mismatch as e:
        print('STOP: %s' % e, file=sys.stderr)
        sys.exit(2)

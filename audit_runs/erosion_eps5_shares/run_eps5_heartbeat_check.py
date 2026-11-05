"""Read heartbeat.tsv of the running EPS-5 job and say whether it is progressing. Takes no arguments.

Runs the committed check of the repaired-stack package against this folder, so there is one check.

Run: python -I -B run_eps5_heartbeat_check.py
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CHECK = os.path.join(HERE, 'code', 'global_invest_dev', 'audit_runs', 'erosion_spam_repair', 'shares', 'run_shares_heartbeat_check.py')


def main():
    spec = importlib.util.spec_from_file_location('run_shares_heartbeat_check', CHECK)
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    check.HERE = HERE
    return check.main()


if __name__ == '__main__':
    sys.exit(main())

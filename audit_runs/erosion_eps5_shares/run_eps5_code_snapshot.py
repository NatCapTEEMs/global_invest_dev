"""Export the library at the pushed commit into code/global_invest_dev. Takes no arguments.

Local, before anything goes to MSI. Runs the committed exporter of the repaired-stack package against
this folder, so there is one exporter: it stops unless source_commit is on a remote branch, uses git
archive, proves each file equal to its git blob and refuses if the snapshot exists.

Run: python -I -B run_eps5_code_snapshot.py
"""
import csv
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    with open(os.path.join(HERE, 'local_parameters.csv'), newline='', encoding='utf-8-sig') as f:
        library = {r['parameter']: r['value'] for r in csv.DictReader(f)}['library_dir']
    path = os.path.join(library, 'audit_runs', 'erosion_spam_repair', 'shares', 'run_code_snapshot.py')
    spec = importlib.util.spec_from_file_location('run_code_snapshot', path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    exporter.HERE = HERE
    exporter.SNAPSHOT = os.path.join(HERE, 'code', 'global_invest_dev')
    exporter.RECORD = os.path.join(HERE, 'code_snapshot')
    return exporter.main()


if __name__ == '__main__':
    sys.exit(main())

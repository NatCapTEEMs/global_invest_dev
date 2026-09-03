"""Resume an erosion cold start whose expensive stages completed.

A cold start that fails AFTER `prevention_shares` -- nine and a half hours of work already on
disk -- must not be restarted with `run_mode='full'`, which would mint a fresh directory and
redo all of it. This re-enters the SAME project directory: tasks whose outputs exist skip, and
only the aggregation reruns.

    EROSION_COLDSTART_DIR=/path/to/gep_erosion_coldstart_<stamp> python run_erosion_coldstart_resume.py
"""
import os

import hazelbean as hb

from global_invest.erosion.run_erosion import run_project


if __name__ == '__main__':

    project_dir = os.environ.get('EROSION_COLDSTART_DIR')
    if not project_dir or not os.path.isdir(project_dir):
        raise SystemExit('set EROSION_COLDSTART_DIR to the existing cold-start project directory')

    p = hb.ProjectFlow(project_dir=project_dir, run_mode='check')
    p.tasks_to_skip = ['invest_sdr']
    run_project(p)

    print('RESUMED PROJECT DIR: ' + p.project_dir)
    print('Compare the total against the published $18,240,930,373 over 250 countries.')

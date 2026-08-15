"""Full erosion GEP run: InVEST SDR -> prevention-share valuation -> maps/figures.

Thin runner: one tree, executed. The chain is cluster-scale (global InVEST SDR); the SDR input/
output locations and table paths are set on p here or by the submitting script -- the configure_*
functions in erosion_functions.py read them off p at run time (their defaults point at the source
repo's cluster layout). The ES-shock runner is run_erosion_shock.py -- a separate tree, a separate
thin runner, no mode switch.
"""
import os
import hazelbean as hb

from global_invest.erosion import erosion_initialize

if __name__ == '__main__':

    p = hb.ProjectFlow()
    p.project_name = 'gep_erosion'
    p.project_dir = os.path.join(os.path.expanduser('~'), 'Files', 'global_invest', 'projects', p.project_name)
    p.set_project_dir(p.project_dir)

    erosion_initialize.build_gep_service_task_tree(p)

    p.results = {}

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script
           + '\n    with base_data set at ' + p.base_data_dir)
    p.execute()

"""Full landslide-mitigation GEP run. Thin runner: one tree, executed; inputs resolve via
initialize_paths reference paths. Folded from m-braaksma/landslide_mitigation v0.2.0."""
import os
import hazelbean as hb

from global_invest.landslide_mitigation import landslide_mitigation_initialize

if __name__ == '__main__':

    p = hb.ProjectFlow()
    p.project_name = 'gep_landslide_mitigation'
    p.project_dir = os.path.join(os.path.expanduser('~'), 'Files', 'global_invest', 'projects', p.project_name)
    p.set_project_dir(p.project_dir)

    # The tree builder's iterators read run configuration, so initialize before building the tree.
    p.results = {}
    landslide_mitigation_initialize.initialize_paths(p)
    landslide_mitigation_initialize.build_gep_service_task_tree(p)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script
           + '\n    with base_data set at ' + p.base_data_dir)
    p.execute()

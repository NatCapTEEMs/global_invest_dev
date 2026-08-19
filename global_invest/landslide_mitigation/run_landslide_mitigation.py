"""Full landslide-mitigation GEP run. Thin runner: one tree, executed; inputs resolve via
initialize_paths reference paths. Folded from m-braaksma/landslide_mitigation v0.2.0."""
import hazelbean as hb

from global_invest.landslide_mitigation import landslide_mitigation_initialize

def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    landslide_mitigation_initialize.build_gep_service_task_tree(p)


if __name__ == '__main__':

    p = hb.ProjectFlow(project_name='gep_landslide_mitigation', run_mode='check')

    # The tree builder's iterators read run configuration, so initialize before building the tree.
    p.results = {}
    landslide_mitigation_initialize.initialize_paths(p)
    build_task_tree(p)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

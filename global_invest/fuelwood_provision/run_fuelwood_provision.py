import hazelbean as hb

from global_invest.fuelwood_provision import fuelwood_provision_initialize


def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    fuelwood_provision_initialize.build_gep_service_task_tree(p)


def run_project(p):
    build_task_tree(p)
    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()
    return p


if __name__ == '__main__':
    p = hb.ProjectFlow(project_name='gep_fuelwood_provision', run_mode='check')
    run_project(p)

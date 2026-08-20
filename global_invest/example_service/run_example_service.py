"""Standalone runner for the example service -- the template in miniature, and the living demo
of the configuration approach: a bare ProjectFlow construction, configuration hydrated from
es_config as a defaults layer, and a get_path reference that resolves to a fixture the library
SHIPS (the standard-seals test map seeds into the project's input/), so this runs on any
machine with zero staging. The tasks themselves are deliberately trivial.
"""
import hazelbean as hb

from global_invest import utilities
from global_invest.example_service import example_service_tasks


def build_task_tree(p):
    p.example_parent_task = p.add_task(example_service_tasks.example_parent_task)
    p.example_service_task = p.add_task(example_service_tasks.example_task, parent=p.example_parent_task)


def run_project(p):
    # The service's configuration row: its one cell, gep_lulc_input_path, resolves to the
    # shipped fixture -- demonstrating the whole chain (template -> input/ seeding -> get_path).
    utilities.hydrate_es_config(p, 'example_service')

    build_task_tree(p)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

    return p


if __name__ == '__main__':

    # Create ProjectFlow object
    p = hb.ProjectFlow(project_name='gep_example_service', run_mode='check')

    # Run the project
    run_project(p)

    result = 'Done!'

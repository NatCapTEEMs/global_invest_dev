"""Standalone runner for the terrestrial-carbon GEP valuation -- values the carbon stock per country.

Consumers (ngfs_pnas, nff_global) do NOT use this script -- they graft add_terrestrial_carbon_tasks(p)
into their own task tree. This is for standalone GEP runs and smoke tests. For the standalone GTAP
productivity shock, see run_terrestrial_carbon_shock.py (one run file per purpose, no MODE switch).
"""
import hazelbean as hb

from global_invest.terrestrial_carbon import terrestrial_carbon_initialize


def build_task_tree(p):
    # Add the tasks to the ProjectFlow object p. This is the main logic of the model.
    terrestrial_carbon_initialize.build_gep_service_task_tree(p)
    
def run_project(p):
    # Task tree. Every task publishes its own inputs (publish_inputs in the tasks module), so
    # there is no setup call here: open the workspace, build the tree, go.
    build_task_tree(p)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)

    p.execute()

    return p    

if __name__ == '__main__':
    
    # Create ProjectFlow object
    p = hb.ProjectFlow(project_name='gep_terrestrial_carbon', run_mode='check')

    # Run the project
    run_project(p)

    result = 'Done!'

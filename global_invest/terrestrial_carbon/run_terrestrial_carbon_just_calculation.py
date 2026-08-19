"""Standalone runner for the terrestrial-carbon GEP CALCULATION only (no results report).

Runs the compute chain and writes the per-country GEP, without rendering the report. For the full GEP
run (calculation + report) use run_terrestrial_carbon.py; for the shock, run_terrestrial_carbon_shock.py.
base_data_dir is resolved by ProjectFlow (default / machine.env); inputs are resolved in initialize_paths.
"""
import pandas as pd
import hazelbean as hb

from global_invest.terrestrial_carbon import terrestrial_carbon_initialize


def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    # Compose additional library subtrees or project-specific tasks here if the
    # project's pipeline ever diverges; only tree construction belongs in this function.
    terrestrial_carbon_initialize.build_gep_service_calculation_task_tree(p)


if __name__ == '__main__':

    # One call does the whole directory setup: it validates run_mode, infers the
    # project dir git-aware from this repo (~/Files/global_invest/projects/
    # gep_terrestrial_carbon -- the same path the hand-built one produced), creates
    # it, and seeds input/ from input_template/. It also sets p.base_data_dir
    # (default / machine.env), p.input_dir, p.intermediate_dir, p.output_dir.
    # run_mode: 'check' resumes in place | 'fresh_intermediate' rebuilds all
    # computation but keeps input/ (test projects only) | 'full' timestamps a new dir.
    p = hb.ProjectFlow(project_name='gep_terrestrial_carbon', run_mode='check')

    # Task tree
    build_task_tree(p) # Defines the actual logic of the model. Navigate into here to see what the model does.

    # Project-level attributes resolved in initialize_paths (one source of truth, all via get_path).
    p.results = {}  # All results will be stored here by each child task.
    terrestrial_carbon_initialize.initialize_paths(p)

    # Run the model
    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script + '\n    with base_data set at ' + p.base_data_dir)
    p.execute()

    result = 'Done!'








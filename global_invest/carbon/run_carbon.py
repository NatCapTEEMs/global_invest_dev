# Copyright (c) 2025, Yanxu Long
# This file is part of the Global GEP project: carbon storage and sequestration
#
# Stand-alone runner for the carbon model. Builds the SEALS7 carbon-density lookup
# (Stage 1, via carbon_initialize.add_carbon_tasks) then applies it to the base-year map
# and summarizes by region (Stage 2, standalone only). Consumers (nff/ngfs) do NOT run this
# file: they read the lookup and call carbon_functions.generate_carbon_density_raster.

import os
import hazelbean as hb

from global_invest.carbon import carbon_tasks, carbon_initialize


def build_task_tree(p):
    # Stage 1: build the SEALS7 carbon-density lookup (the reusable seam).
    carbon_initialize.add_carbon_tasks(p)
    # Stage 2 (standalone only): apply the lookup to the base-year map, then summarize by region.
    p.task_generate_carbon_density_raster_base_year = p.add_task(carbon_tasks.task_generate_carbon_density_raster_base_year)
    p.task_summarize_carbon_density_by_region = p.add_task(carbon_tasks.task_summarize_carbon_density_by_region)


if __name__ == '__main__':
    p = hb.ProjectFlow()

    # Project directories.
    p.user_dir = os.path.expanduser('~')
    p.extra_dirs = ['Files', 'global_invest', 'projects']
    p.project_name = p.project_name + '_' + hb.pretty_time()  # comment out to reuse an existing project
    p.project_dir = os.path.join(p.user_dir, os.sep.join(p.extra_dirs), p.project_name)
    p.set_project_dir(p.project_dir)

    # Base data. Portable default; carbon inputs live under base_data/carbon_storage.
    p.base_data_dir = os.path.join(p.user_dir, 'Files', 'base_data')
    p.aoi = 'global'

    # Resolve carbon input paths (SEALS7 base map, carbon zones, lookup, regions).
    carbon_initialize.initialize_paths(p)

    build_task_tree(p)
    p.fail_fast = True
    p.verbosity = 2
    p.debug = True
    p.execute()

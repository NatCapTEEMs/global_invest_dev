"""Full recreation GEP run: site quality -> gravity-model visits -> travel-cost value -> report.

Thin runner: builds ONE tree and executes it. Inputs are published by each task itself
(publish_inputs in the tasks module); base_data_dir is resolved by ProjectFlow (default /
machine.env), never hardcoded here.

Requires (es_parameters rows, staged under base_data/global_invest/recreation/ from the GEP
consortium drive's Recreation/data/ tree -- 0_inputs/* to the module root,
0_processed_raster_inputs/<sub>/* to <sub>/*):
  - lulc/: the six SEALS7 2010 1 km class-share rasters
  - pa/: WDPA accessible-area share       - distance_to_roads/: distance-to-road raster
  - grip4_road_length/: the 1 km reference grid   - worldpop/: 2010 population, aligned
  - module root: fuel-cost CSV, UNWTO all-data workbook (xlsx -> needs openpyxl), hotels gpkg

Verification anchor: the reference results_by_country.csv from the source pipeline's project
dir (not committed to its repo) -- compare per-country daily/tourist visits and values once the
data is staged. See the flagged travel-cost units note in recreation_functions.
"""
import hazelbean as hb

from global_invest.recreation import recreation_initialize


def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    recreation_initialize.build_gep_service_task_tree(p)


def run_project(p):

    # Every task publishes its own inputs (publish_inputs in the tasks module): no setup call.
    build_task_tree(p)

    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script)
    p.execute()

    return p


if __name__ == '__main__':

    # Create ProjectFlow object
    p = hb.ProjectFlow(project_name='gep_recreation', run_mode='check')

    # Run the project
    run_project(p)

    result = 'Done!'

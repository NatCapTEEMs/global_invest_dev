import os

import pandas as pd
import hazelbean as hb
from global_invest import utilities

from global_invest.renewable_energy_provision import renewable_energy_provision_initialize
from global_invest.renewable_energy_provision import renewable_energy_provision_functions as rf


def publish_inputs(p):
    """Every task's first line: the renewable-energy valuation's es_config row (defaults layer -- a caller-set value wins)
    plus the shared country references and the results registry."""
    utilities.hydrate_es_config(p, 'renewable_energy_provision', log=hb.log)
    utilities.hydrate_es_parameters(p, 'renewable_energy_provision', log=hb.log)
    utilities.initialize_country_paths(p, simplified='30sec')
    if not hasattr(p, 'results'):
        p.results = {}
    return p

def renewable_energy_provision(p):
    """
    Parent task for renewable energy provision. Inputs resolve in publish_inputs.
    """
    publish_inputs(p)

def gep_preprocess(p):
    """
    Preprocessing tasks are assumed NOT to be run by the user. Instead, it is assumed that the output of a preprocess
    task is an input to the actual model, saved at the canonical project attribute p.renewable_energy_provision_input_path.
    These are preprocessing tasks are still provided for reference, but are not intended to be run directly by the user.
    We will "promote" the data outputed by a preprocess task to the base_data_dir provided to users.
    """
    publish_inputs(p)
    pass # NYI

def gep_calculation(p):
    """GEP valuation for renewable energy: each resource's generation priced at the country-year
    electricity price and scaled by nature's share of the resource rent, one CSV per resource."""
    publish_inputs(p)
    service_results = {'gep_by_country_base_year': os.path.join(
        p.cur_dir, "renewable_energy_provision_gep_by_country_base_year.csv")}
    p.results['renewable_energy_provision'] = service_results
    subservices = {subservice: {'gep_by_country_base_year': os.path.join(
        p.cur_dir, f"{subservice}_gep_by_country_base_year.csv")}
        for subservice in rf.SUBSERVICE_TECHNOLOGIES}
    service_results['subservices'] = subservices

    output_paths = ([service_results['gep_by_country_base_year']]
                    + [s['gep_by_country_base_year'] for s in subservices.values()])
    if hb.path_all_exist(output_paths):
        hb.log("All results already exist. Skipping GEP calculation for renewable energy provision.")
        return
    hb.log("Starting GEP calculation for renewable energy provision.")

    generation_frames = rf.generation_by_technology(hb.df_read(p.gep_quantity_input_path))
    df_price = rf.price_in_usd_per_gwh(hb.df_read(p.gep_price_input_path))
    priced_frames = rf.merge_price_onto_generation(df_price, generation_frames)
    df_valued = rf.valued_generation(priced_frames, hb.df_read(p.gep_attribution_input_path))
    df_gep = rf.base_year_valued_rows(df_valued, int(p.gep_base_year))

    # The source is keyed by ISO3 strings, so the join matches on the r250 label. One row per
    # country: r264 splits large countries, so the correspondence is collapsed before the join.
    ee_r264_to_250 = utilities.collapse_countries_to_r250(p.df_countries)
    valued_rows_before_join = int(df_gep['renewable_energy_provision_gep'].notna().sum())
    df_gep = hb.df_merge(ee_r264_to_250, df_gep,
                         left_on='iso3_r250_label', right_on='iso3_r250_label', how='left')
    utilities.assert_join_coverage(df_gep, 'renewable_energy_provision_gep',
                                   valued_rows_before_join, 'renewable_energy_provision', log=hb.log)

    hb.df_write(df_gep, service_results['gep_by_country_base_year'], index=False)
    by_resource = rf.split_by_resource(df_gep)
    for subservice, technology in rf.SUBSERVICE_TECHNOLOGIES.items():
        hb.df_write(by_resource[technology], subservices[subservice]['gep_by_country_base_year'],
                    index=False)

    # Map only: the r264-expanded boundaries, each sub-region carrying its country's value.
    gdf = hb.df_merge(p.gdf_countries_simplified, df_gep, how='outer',
                      left_on='ee_r264_id', right_on='ee_r264_id')
    gdf.to_file(service_results['gep_by_country_base_year'].replace('.csv', '.gpkg'), driver='GPKG')

    value_gep_base_year = df_gep['renewable_energy_provision_gep'].sum()
    hb.log(f"Total GEP value for base year {int(p.gep_base_year)}: {value_gep_base_year}")
    return value_gep_base_year

def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)


def gep_load_results(p):
    publish_inputs(p)
    
    # Learn the paths by creating a temp task treep
    p_temp = hb.ProjectFlow()
    renewable_energy_provision_initialize.build_gep_service_calculation_task_tree(p_temp)
    p_temp.set_all_tasks_to_skip_if_dir_exists()
    p_temp.execute()
    
    print(p_temp.results)
    pass
        
def gep_results_distribution(p):
    """Distribute the results of the GEP calculation."""
    publish_inputs(p)
    # This task is intended to copy the results to the output directory.
    hb.log("Distributing GEP results...")
    
    for key, value in p.results['renewable_energy_provision'].items():
        output_path = os.path.join(p.output_dir, key)
        hb.path_copy(value, output_path)
        hb.log(f"Distributed {key} to {output_path}")
    
    hb.log("GEP results distribution complete.")
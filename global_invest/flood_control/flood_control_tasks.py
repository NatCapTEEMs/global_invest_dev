"""Flood-control GEP tasks: the committed avoided-damage valuation on the r250 rows."""
import os

import pandas as pd
import hazelbean as hb
from global_invest import utilities
from global_invest.flood_control import flood_control_functions as fc


def publish_inputs(p):
    """Every GEP task's first line: the flood_control es_config row and the data reference
    from es_parameters (defaults layer -- a caller-set value prevails), the shared country
    references and the results registry."""
    utilities.hydrate_es_config(p, 'flood_control', log=hb.log)
    utilities.hydrate_es_parameters(p, 'flood_control', log=hb.log)
    utilities.initialize_country_paths(p)
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def gep_calculation(p):
    """GEP valuation for flood control: the committed per-country expected avoided damage."""
    publish_inputs(p)
    service_results = {}
    p.results['flood_control'] = service_results
    service_results['gep_by_country_base_year'] = os.path.join(p.cur_dir, 'gep_by_country_base_year.csv')

    if hb.path_all_exist(list(service_results.values())):
        hb.log('All results already exist. Skipping GEP calculation for flood_control.')
        return
    hb.log('Starting GEP calculation for flood_control.')

    avoided = pd.read_csv(p.flood_control_avoided_damage_path)
    attr_cols = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
                 'continent', 'region_un', 'region_wb', 'income_grp', 'subregion']
    countries = p.df_countries[attr_cols].drop_duplicates('iso3_r250_id')
    df_gep = fc.flood_control_gep_by_country(avoided, countries)
    df_gep['year'] = int(p.gep_base_year)
    hb.df_write(df_gep[attr_cols + ['year', 'flood_control_gep']],
                service_results['gep_by_country_base_year'])

    hb.log(f'Total flood_control GEP for base year {p.gep_base_year}: '
           f'{df_gep["flood_control_gep"].sum():,.2f}')
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)

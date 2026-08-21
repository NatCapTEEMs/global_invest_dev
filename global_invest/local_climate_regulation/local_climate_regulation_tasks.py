"""Local-climate-regulation GEP tasks: the committed urban-cooling valuation on r250 rows."""
import os

import pandas as pd
import hazelbean as hb
from global_invest import utilities
from global_invest.local_climate_regulation import local_climate_regulation_functions as lc


def publish_inputs(p):
    """Every GEP task's first line: the local_climate_regulation es_config row and the data
    reference from es_parameters (defaults layer -- a caller-set value prevails), the shared
    country references and the results registry."""
    utilities.hydrate_es_config(p, 'local_climate_regulation', log=hb.log)
    utilities.hydrate_es_parameters(p, 'local_climate_regulation', log=hb.log)
    utilities.initialize_country_paths(p)
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def gep_calculation(p):
    """GEP valuation for local climate regulation: the committed final table, one row per
    country (the v04 correction's own inputs are the open asks; see the functions module)."""
    publish_inputs(p)
    service_results = {}
    p.results['local_climate_regulation'] = service_results
    service_results['gep_by_country_base_year'] = os.path.join(p.cur_dir, 'gep_by_country_base_year.csv')

    if hb.path_all_exist(list(service_results.values())):
        hb.log('All results already exist. Skipping GEP calculation for local_climate_regulation.')
        return
    hb.log('Starting GEP calculation for local_climate_regulation.')

    final = pd.read_csv(p.local_climate_regulation_final_path)
    attr_cols = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
                 'continent', 'region_un', 'region_wb', 'income_grp', 'subregion']
    countries = p.df_countries[attr_cols].drop_duplicates('iso3_r250_id')
    df_gep = lc.local_climate_gep_by_country(final, countries)
    df_gep['year'] = int(p.gep_base_year)
    hb.df_write(df_gep[attr_cols + ['year', 'local_climate_regulation_gep']],
                service_results['gep_by_country_base_year'])

    hb.log(f'Total local_climate_regulation GEP for base year {p.gep_base_year}: '
           f'{df_gep["local_climate_regulation_gep"].sum():,.2f}')
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)

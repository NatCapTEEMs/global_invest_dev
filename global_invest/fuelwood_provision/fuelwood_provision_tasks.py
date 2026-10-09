"""Fuelwood-provision GEP tasks: the CAS per-country results carried onto the account list.

This layer owns every file read and write. The science it calls lives in
fuelwood_provision_functions, which never opens a file.
"""
import os

import hazelbean as hb
import pandas as pd

from global_invest import utilities
from global_invest.fuelwood_provision import fuelwood_provision_functions as fw


def publish_inputs(p):
    """Every GEP task's first line: the fuelwood_provision es_config row and the parameter
    rows from es_parameters (defaults layer -- a caller-set value prevails), the shared
    country references and the results registry."""
    utilities.hydrate_es_config(p, 'fuelwood_provision', log=hb.log)
    utilities.hydrate_es_parameters(p, 'fuelwood_provision', log=hb.log)
    utilities.initialize_country_paths(p)
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def gep_calculation(p):
    """GEP valuation for fuelwood provision: the CAS results sheet onto one row per country.

    The sheet's values are thousands of constant-2019 USD by its own title; a code the
    account list cannot carry (the dissolved Netherlands Antilles) is written beside the
    table rather than dropped silently, and any new unmatched code fails the run.
    """
    publish_inputs(p)
    service_results, already_done = utilities.begin_gep_calculation(p, 'fuelwood_provision')
    if already_done:
        return

    results = pd.read_excel(str(p.get_path(p.fuelwood_provision_results_path)))
    gep, unallocated = fw.fuelwood_gep_by_country(results, p.df_countries)

    attr_cols = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
                 'continent', 'region_un', 'region_wb', 'income_grp', 'subregion']
    attrs = utilities.collapse_countries_to_r250(p.df_countries)[attr_cols]
    df_gep = attrs.merge(gep, how='left', on='iso3_r250_id')
    df_gep['year'] = int(p.gep_base_year)
    hb.df_write(df_gep[attr_cols + ['year', 'fuelwood_provision_gep']],
                service_results['gep_by_country_base_year'])

    map_df = (p.df_countries[['ee_r264_id', 'iso3_r250_id']]
              .merge(df_gep[['iso3_r250_id', 'iso3_r250_name', 'fuelwood_provision_gep']],
                     how='left', on='iso3_r250_id'))
    gdf = hb.df_merge(p.gdf_countries_simplified, map_df,
                      how='outer', left_on='ee_r264_id', right_on='ee_r264_id')
    gdf.to_file(service_results['gep_by_country_base_year'].replace('.csv', '.gpkg'),
                driver='GPKG')

    unallocated_path = os.path.join(p.cur_dir, 'unallocated_rows.csv')
    unallocated.to_csv(unallocated_path, index=False)

    total = df_gep['fuelwood_provision_gep'].sum()
    hb.log('Total fuelwood provision GEP for base year %s: %s over %d countries '
           '(the CAS results at their stated thousands of 2019 USD); %d row(s) the account '
           'list cannot carry, %s, written to %s'
           % (p.gep_base_year, f'{total:,.2f}',
              int(df_gep['fuelwood_provision_gep'].notna().sum()), len(unallocated),
              f"{unallocated['value_usd'].sum():,.0f}" if len(unallocated) else '$0',
              unallocated_path))
    return total


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)

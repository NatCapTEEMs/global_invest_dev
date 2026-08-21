"""Extractive-energy GEP tasks: CWoN 2024 fuel rents computed onto the r250 rows."""
import os

import pandas as pd
import hazelbean as hb
from global_invest import utilities
from global_invest.extractive_energy import extractive_energy_functions as xe


def publish_inputs(p):
    """Every GEP task's first line: the extractive_energy es_config row and the data references
    from es_parameters (defaults layer -- a caller-set value prevails), the shared country
    references and the results registry."""
    utilities.hydrate_es_config(p, 'extractive_energy', log=hb.log)
    utilities.hydrate_es_parameters(p, 'extractive_energy', log=hb.log)
    utilities.initialize_country_paths(p)
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def gep_calculation(p):
    """GEP valuation for extractive energy: gas + coal + petroleum, one row per country."""
    publish_inputs(p)
    service_results = {}
    p.results['extractive_energy'] = service_results
    service_results['gep_by_country_base_year'] = os.path.join(p.cur_dir, 'gep_by_country_base_year.csv')

    if hb.path_all_exist(list(service_results.values())):
        hb.log('All results already exist. Skipping GEP calculation for extractive_energy.')
        return
    hb.log('Starting GEP calculation for extractive_energy.')

    gas_cwon = pd.read_stata(p.extractive_energy_cwon_gas_path)
    coal_cwon = pd.read_stata(p.extractive_energy_cwon_coal_path)
    oil_cwon = pd.read_stata(p.extractive_energy_cwon_oil_path)
    attr_cols = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
                 'continent', 'region_un', 'region_wb', 'income_grp', 'subregion']
    countries = p.df_countries[attr_cols].drop_duplicates('iso3_r250_id')
    df_gep = xe.extractive_energy_gep_from_cwon(gas_cwon, coal_cwon, oil_cwon, countries,
                                                p.gep_base_year)
    df_gep['year'] = int(p.gep_base_year)
    hb.df_write(df_gep[attr_cols + ['year', 'extractive_energy_gas_gep',
                                    'extractive_energy_coal_gep', 'extractive_energy_oil_gep',
                                    'extractive_energy_gep']],
                service_results['gep_by_country_base_year'])

    hb.log(f'Total extractive_energy GEP for base year {p.gep_base_year}: '
           f'{df_gep["extractive_energy_gep"].sum():,.2f}')
    committed_total = sum(pd.read_csv(path).iloc[:, -1].sum() for path in (
        p.extractive_energy_gas_path, p.extractive_energy_coal_path, p.extractive_energy_oil_path))
    hb.log(f'Committed drive tables sum to {committed_total:,.2f}; the difference is the '
           f'territory join artifact plus the Kosovo coal row (see the functions docstring).')
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)

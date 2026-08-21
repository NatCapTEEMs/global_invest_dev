"""Water-supply GEP tasks. First component: hydropower (CWoN resource-rent method).

The hydropower rent derives from CWoN 2024's capitalized wealth (see the functions module for
the identified method and its anchor); the agriculture and household components join here when
their science surfaces.
"""
import os

import pandas as pd
import hazelbean as hb
from global_invest import utilities
from global_invest.water_supply import water_supply_functions as wf


def publish_inputs(p):
    """Every GEP task's first line: the water_supply es_config row and the CWoN data reference
    from es_parameters (defaults layer -- a caller-set value prevails), the shared country
    references and the results registry."""
    utilities.hydrate_es_config(p, 'water_supply', log=hb.log)
    utilities.hydrate_es_parameters(p, 'water_supply', log=hb.log)
    utilities.initialize_country_paths(p)
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def hydropower_rent(p):
    """CWoN capitalized hydropower wealth -> the implied constant annual rent per country."""
    publish_inputs(p)
    p.hydropower_rent_path = os.path.join(p.cur_dir, 'hydropower_rent.csv')
    if not p.run_this:
        return
    if not hb.path_exists(p.hydropower_rent_path):
        wealth = pd.read_stata(p.water_supply_cwon_hydro_wealth_path)
        wf.hydropower_rent_from_wealth(wealth).to_csv(p.hydropower_rent_path, index=False)
    return True


def gep_calculation(p):
    """GEP valuation for water_supply: the hydropower component on the r250 country list,
    one row per country. water_supply_gep currently equals the hydropower component; the
    agriculture and household components add columns here when they arrive."""
    publish_inputs(p)
    service_results = {}
    p.results['water_supply'] = service_results
    service_results['gep_by_country_base_year'] = os.path.join(p.cur_dir, 'gep_by_country_base_year.csv')

    if hb.path_all_exist(list(service_results.values())):
        hb.log('All results already exist. Skipping GEP calculation for water_supply.')
        return
    hb.log('Starting GEP calculation for water_supply.')

    hydropower = pd.read_csv(p.hydropower_rent_path)
    attr_cols = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
                 'continent', 'region_un', 'region_wb', 'income_grp', 'subregion']
    countries = p.df_countries[attr_cols].drop_duplicates('iso3_r250_id')
    df_gep = wf.water_supply_gep_by_country(hydropower, countries)
    df_gep['year'] = int(p.gep_base_year)
    df_gep['water_supply_gep'] = df_gep['hydropower_gep']
    hb.df_write(df_gep[attr_cols + ['year', 'hydropower_gep', 'water_supply_gep']],
                service_results['gep_by_country_base_year'])

    hb.log(f'Total water_supply GEP (hydropower component) for base year {p.gep_base_year}: '
           f'{df_gep["hydropower_gep"].sum():,.2f}')
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)

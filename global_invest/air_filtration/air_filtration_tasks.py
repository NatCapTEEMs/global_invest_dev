"""Air-filtration GEP tasks: the drive workbook's avoided-mortality valuation, two channels.

air_filtration_gep is the deposition channel (the sheet's air_filtration service);
sandstorm_prevention_gep is the dust channel (the sheet's sandstorm prevention service) --
one workbook, one module, two sheet rows. See the functions module for the identified rules
and the VSL vintage gap.
"""
import os

import pandas as pd
import hazelbean as hb
from global_invest import utilities
from global_invest.air_filtration import air_filtration_functions as af

MODULE_REFERENCE_DIR = os.path.join(os.path.dirname(__file__), 'reference')


def publish_inputs(p):
    """Every GEP task's first line: the air_filtration es_config row and the workbook reference
    from es_parameters (defaults layer -- a caller-set value prevails), the shared country
    references and the results registry."""
    utilities.hydrate_es_config(p, 'air_filtration', log=hb.log)
    utilities.hydrate_es_parameters(p, 'air_filtration', log=hb.log)
    utilities.initialize_country_paths(p)
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def gep_calculation(p):
    """GEP valuation for air quality: the workbook's deaths x VSL per channel, one row per
    country, on the r250 ids (positional join, name-floor guarded)."""
    publish_inputs(p)
    service_results = {}
    p.results['air_filtration'] = service_results
    service_results['gep_by_country_base_year'] = os.path.join(p.cur_dir, 'gep_by_country_base_year.csv')

    if hb.path_all_exist(list(service_results.values())):
        hb.log('All results already exist. Skipping GEP calculation for air_filtration.')
        return
    hb.log('Starting GEP calculation for air_filtration.')

    workbook = pd.read_excel(p.air_filtration_workbook_path)
    af.verify_global_average_fill(workbook)
    r250_order = hb.df_read(os.path.join(MODULE_REFERENCE_DIR, 'r250_gpkg_order.csv'))
    df = af.air_quality_gep_by_country(workbook, r250_order)

    attr_cols = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
                 'continent', 'region_un', 'region_wb', 'income_grp', 'subregion']
    attrs = p.df_countries[attr_cols].drop_duplicates('iso3_r250_id')
    df_gep = df.drop(columns=['iso3_r250_label']).merge(attrs, how='left', on='iso3_r250_id')
    df_gep['year'] = int(p.gep_base_year)
    hb.df_write(df_gep[attr_cols + ['year', 'air_filtration_gep', 'sandstorm_prevention_gep']],
                service_results['gep_by_country_base_year'])

    hb.log(f'Total air_filtration GEP (deposition channel) for base year {p.gep_base_year}: '
           f'{df_gep["air_filtration_gep"].sum():,.2f}')
    hb.log(f'Total sandstorm_prevention GEP (dust channel): '
           f'{df_gep["sandstorm_prevention_gep"].sum():,.2f}')
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)

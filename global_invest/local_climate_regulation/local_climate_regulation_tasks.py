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


def read_city_savings(merged_dir):
    """The per-country city-month valuation files, concatenated.

    One file per country, each a row per city and month. Reading them here rather than in the
    science module keeps the summing testable on a small frame.
    """
    import glob
    paths = sorted(glob.glob(os.path.join(merged_dir, '*_all_urban_valuations.csv')))
    if not paths:
        raise FileNotFoundError(f'no city valuation files under {merged_dir}')
    return pd.concat([hb.df_read(path) for path in paths], ignore_index=True)


def gep_calculation(p):
    """GEP valuation for local climate regulation, computed from the calculation's own city-month
    valuations: avoided cooling energy priced at the national electricity price, summed over
    every city in a country.

    The committed accounting table is carried beside it as the comparison anchor rather than
    reported. The two disagree by a wide and country-varying margin, which is the open ask:
    something sits between these city values and that table which is not in anything we hold.
    """
    publish_inputs(p)
    service_results, already_done = utilities.begin_gep_calculation(p, 'local_climate_regulation')
    if already_done:
        return

    city = read_city_savings(p.local_climate_regulation_city_savings_path)
    ours = lc.city_savings_by_country(city)

    attr_cols = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
                 'continent', 'region_un', 'region_wb', 'income_grp', 'subregion']
    countries = utilities.collapse_countries_to_r250(p.df_countries)[attr_cols]
    df_gep = countries.merge(ours, on='iso3_r250_id', how='left')

    anchor = lc.local_climate_gep_by_country(
        hb.df_read(p.local_climate_regulation_final_path), countries[['iso3_r250_label']])
    df_gep = df_gep.merge(
        anchor.rename(columns={'local_climate_regulation_gep':
                               'local_climate_regulation_gep_committed'}),
        on='iso3_r250_label', how='left')
    df_gep['year'] = int(p.gep_base_year)
    hb.df_write(df_gep[attr_cols + ['year', 'local_climate_regulation_gep',
                                    'local_climate_regulation_gep_committed']],
                service_results['gep_by_country_base_year'])

    ours_total = df_gep['local_climate_regulation_gep'].sum()
    committed_total = df_gep['local_climate_regulation_gep_committed'].sum()
    hb.log(f'Total local_climate_regulation GEP for base year {p.gep_base_year}: '
           f'{ours_total:,.2f} over '
           f'{int(df_gep["local_climate_regulation_gep"].notna().sum())} countries')
    hb.log(f'  the committed accounting table totals {committed_total:,.2f}; the gap is not a '
           f'constant factor and is the open ask on this service')
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)

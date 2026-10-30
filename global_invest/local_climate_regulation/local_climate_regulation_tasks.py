"""Local-climate-regulation GEP tasks: the author's delivered urban-cooling valuation on r250 rows."""
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
    """GEP valuation for local climate regulation: the author's delivered final, the code register's
    results file of 13 July 2026, joined onto the r250 country list.

    That final is the December 2025 city sums with the old global marginal-consumption slope
    divided out and each country's own cooling consumption per degree-day applied. The step that
    made it is not in the registered code, so the final is imported, not recomputed. The
    December 2025 city-sum lineage and the CAS 2019 results are carried beside it for comparison.
    """
    publish_inputs(p)
    service_results, already_done = utilities.begin_gep_calculation(p, 'local_climate_regulation')
    if already_done:
        return

    if not p.run_this:
        return p
    attr_cols = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
                 'continent', 'region_un', 'region_wb', 'income_grp', 'subregion']
    countries = utilities.collapse_countries_to_r250(p.df_countries)[attr_cols]

    # The account's cooling row is the author's delivered final.
    adopted = lc.local_climate_gep_by_country(
        hb.df_read(p.local_climate_regulation_final_path), countries[['iso3_r250_label']])
    df_gep = countries.merge(adopted, on='iso3_r250_label', how='left')

    # Beside it: the December 2025 city sums the final was built from, and the CAS 2019
    # results the code register names as the cross-check.
    city = read_city_savings(p.local_climate_regulation_city_savings_path)
    lineage = lc.city_savings_by_country(city).rename(
        columns={'local_climate_regulation_gep': 'local_climate_regulation_gep_city_sum_lineage'})
    cas = pd.read_excel(str(p.get_path(p.local_climate_regulation_cas_results_path)),
                        sheet_name='2019 Results')
    cas = lc.cas_cooling_by_country(cas, countries).rename(
        columns={'local_climate_regulation_gep': 'local_climate_regulation_gep_cas',
                 'cooling_kwh': 'cooling_kwh_cas'})
    df_gep = df_gep.merge(lineage, on='iso3_r250_id', how='left')
    df_gep = df_gep.merge(cas, on='iso3_r250_id', how='left')
    df_gep['year'] = int(p.gep_base_year)
    hb.df_write(df_gep[attr_cols + ['year', 'local_climate_regulation_gep',
                                    'local_climate_regulation_gep_city_sum_lineage',
                                    'local_climate_regulation_gep_cas', 'cooling_kwh_cas']],
                service_results['gep_by_country_base_year'])

    hb.log('Total local climate regulation GEP (the author\'s delivered final) for base year %s: '
           '%s over %d countries. Beside it: city-sum lineage %s, CAS %s'
           % (p.gep_base_year, f"{df_gep['local_climate_regulation_gep'].sum():,.2f}",
              int(df_gep['local_climate_regulation_gep'].notna().sum()),
              f"{df_gep['local_climate_regulation_gep_city_sum_lineage'].sum():,.0f}",
              f"{df_gep['local_climate_regulation_gep_cas'].sum():,.0f}"))
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    if not p.run_this:
        return p
    utilities.render_service_results(p)

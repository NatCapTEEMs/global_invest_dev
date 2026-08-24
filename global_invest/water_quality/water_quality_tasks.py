"""Water-quality GEP tasks: the verified USD calculation plus the committed international-dollar
accounting value (conversion stage unidentified -- see the functions module)."""

import hazelbean as hb
from global_invest import utilities
from global_invest.water_quality import water_quality_functions as wq


def publish_inputs(p):
    """Every GEP task's first line: the water_quality es_config row and the data references
    from es_parameters (defaults layer -- a caller-set value prevails), the shared country
    references and the results registry."""
    utilities.hydrate_es_config(p, 'water_quality', log=hb.log)
    utilities.hydrate_es_parameters(p, 'water_quality', log=hb.log)
    utilities.initialize_country_paths(p)
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def gep_calculation(p):
    """GEP valuation for water quality: recompute the USD calculation from the retention estimates
    and attach the committed international-dollar value, one row per country."""
    publish_inputs(p)
    service_results, already_done = utilities.begin_gep_calculation(p, 'water_quality')
    if already_done:
        return

    retention = hb.df_read(p.water_quality_retention_path)
    international = hb.df_read(p.water_quality_international_path)
    attr_cols = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
                 'continent', 'region_un', 'region_wb', 'income_grp', 'subregion']
    countries = utilities.collapse_countries_to_r250(p.df_countries)[attr_cols]
    df_gep = wq.water_quality_gep_by_country(retention, international, countries)
    df_gep['year'] = int(p.gep_base_year)
    hb.df_write(df_gep[attr_cols + ['year', 'nitrogen_gep_usd', 'phosphorus_gep_usd',
                                    'water_quality_gep_usd', 'water_quality_gep']],
                service_results['gep_by_country_base_year'])

    hb.log(f'Total water_quality GEP for base year {p.gep_base_year}: '
           f'{df_gep["water_quality_gep_usd"].sum():,.2f} USD; '
           f'{df_gep["water_quality_gep"].sum():,.2f} international dollars (the committed '
           f'accounting value; conversion stage unidentified).')
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)

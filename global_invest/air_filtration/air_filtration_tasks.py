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
    # The air quality group's country-level value-of-life table, which is the source behind the
    # workbook's VSL column and the one the valuation reads.
    if not hasattr(p, 'air_filtration_vsl_path'):
        p.air_filtration_vsl_path = p.get_path(
            os.path.join('global_invest', 'air_filtration', 'data', 'clean', 'vsl.csv'))
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def gep_calculation(p):
    """GEP valuation for air quality: the workbook's deaths x VSL per channel, one row per
    country, on the r250 ids (positional join, name-floor guarded)."""
    publish_inputs(p)
    service_results, already_done = utilities.begin_gep_calculation(p, 'air_filtration')
    if already_done:
        return

    workbook = pd.read_excel(p.air_filtration_workbook_path)
    af.verify_global_average_fill(workbook)

    r250_order = hb.df_read(os.path.join(MODULE_REFERENCE_DIR, 'r250_gpkg_order.csv'))

    # The valuation reads the group's country table rather than the workbook's VSL column, and
    # says where the two disagree. They are the same build, so a disagreement is a question for
    # the group, not a reason to stop: the run reports it and carries on with the table.
    vsl_table = hb.df_read(p.air_filtration_vsl_path)
    vsl, matched, disagreeing, unsourced = af.vsl_from_country_table(
        workbook, r250_order, vsl_table)
    hb.log(f'VSL sourced from the country table for {matched} of {len(r250_order)} countries.')
    for row in unsourced.itertuples():
        hb.log(f'  {row.country} ({row.iso3}) is priced per country in the workbook at '
               f'{row.workbook_vsl:,.0f} but is absent from the table, so the workbook figure '
               f'stands and this value is not one we can source.')
    for row in disagreeing.itertuples():
        hb.log(f'  VSL differs for {row.country} ({row.iso3}): workbook {row.workbook_vsl:,.0f}, '
               f'table {row.table_vsl:,.0f}, {row.relative_difference:.1%} apart.')

    df = af.air_quality_gep_by_country(workbook, r250_order, vsl=vsl)

    attr_cols = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
                 'continent', 'region_un', 'region_wb', 'income_grp', 'subregion']
    attrs = utilities.collapse_countries_to_r250(p.df_countries)[attr_cols]
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

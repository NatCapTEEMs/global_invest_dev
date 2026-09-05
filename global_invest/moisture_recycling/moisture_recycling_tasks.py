"""Moisture-recycling GEP tasks: the water-supply value re-attributed to the rain's source countries.

This layer owns every file read and write. The science it calls lives in
moisture_recycling_functions, which never opens a file.

⚠ The published total is a re-attribution of a SUBSET of the water-supply row (its irrigation and
domestic components, scaled by each destination's terrestrial-origin precipitation share). It is
never added to water supply; the two rows partition the same money.
"""
import json
import os

import hazelbean as hb

from global_invest import utilities
from global_invest.moisture_recycling import moisture_recycling_functions as mr


def publish_inputs(p):
    """Every GEP task's first line: the moisture_recycling es_config row and the parameter rows
    from es_parameters (defaults layer -- a caller-set value prevails), the shared country
    references and the results registry."""
    utilities.hydrate_es_config(p, 'moisture_recycling', log=hb.log)
    utilities.hydrate_es_parameters(p, 'moisture_recycling', log=hb.log)
    utilities.initialize_country_paths(p, simplified='30sec')
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def gep_calculation(p):
    """GEP valuation for moisture recycling: water value attributed to the rain's sources.

    Each destination's irrigation and domestic water GEP is scaled by the terrestrial-origin
    share of its precipitation and distributed over source countries with the WAM2layers
    sink-fraction matrix. Source-side and sink-side totals agree by construction and the task
    asserts that they do.
    """
    publish_inputs(p)
    service_results, already_done = utilities.begin_gep_calculation(p, 'moisture_recycling')
    if already_done:
        return

    matrix = mr.moisture_matrix(hb.df_read(str(p.get_path(p.moisture_recycling_sink_fraction_input_path))))
    water = hb.df_read(str(p.get_path(p.moisture_recycling_water_gep_input_path)))
    component_cols = json.loads(p.moisture_recycling_water_gep_component_cols) \
        if isinstance(p.moisture_recycling_water_gep_component_cols, str) \
        else list(p.moisture_recycling_water_gep_component_cols)
    water_value = water.set_index('iso3_r250_label')[component_cols].sum(axis=1)

    dropped = sorted(set(water_value.index) - set(matrix.columns))
    if dropped:
        hb.log('  ⚠ %d countries carry water value but are absent from the moisture matrix and '
               'receive no attribution: %s (their value: %s)'
               % (len(dropped), ', '.join(dropped[:8]) + ('...' if len(dropped) > 8 else ''),
                  f'{water_value.reindex(dropped).sum():,.0f}'))
    untracked = mr.untracked_destinations(matrix)
    if untracked:
        hb.log('  ⚠ %d destinations are below the tracking grid (all-NaN columns) and their '
               'water value goes unattributed: %s (their value: %s)'
               % (len(untracked), ', '.join(untracked),
                  f'{water_value.reindex(untracked).fillna(0.0).sum():,.0f}'))

    df_gep = mr.reattributed_water_value(matrix, water_value, float(p.moisture_recycling_ecosystem_share))
    source_total = df_gep['moisture_recycling_gep'].sum()
    sink_total = df_gep['moisture_recycling_sink_side_value'].sum()
    if abs(source_total - sink_total) > 1.0:
        raise NameError('moisture recycling lost money in the re-attribution: source side %r '
                        'against sink side %r. The matrix and the value series disagree on the '
                        'country set.' % (source_total, sink_total))

    # The full collapse frame keeps ee_r264_id: the published table drops it through
    # published_country_columns, and the map merge is what needs it.
    countries = utilities.collapse_countries_to_r250(p.df_countries)
    aggregate_sources = sorted(set(df_gep['iso3_r250_label']) - set(countries['iso3_r250_label']))
    if aggregate_sources:
        lost = df_gep.set_index('iso3_r250_label').loc[aggregate_sources, 'moisture_recycling_gep'].sum()
        hb.log('  ⚠ %d matrix sources are aggregates with no account country and leave the '
               'published table: %s (their value: %s)'
               % (len(aggregate_sources), ', '.join(aggregate_sources), f'{lost:,.0f}'))
    df_gep['year'] = int(p.gep_base_year)
    utilities.write_gep_by_country(
        p, df_gep[utilities.published_country_columns(df_gep, 'moisture_recycling')],
        service_results['gep_by_country_base_year'])

    gdf = hb.df_merge(p.gdf_countries_simplified, df_gep, how='outer',
                      left_on='ee_r264_id', right_on='ee_r264_id')
    gdf.to_file(service_results['gep_by_country_base_year'].replace('.csv', '.gpkg'), driver='GPKG')

    hb.log(f'Total moisture_recycling GEP for base year {p.gep_base_year}: {source_total:,.2f}')
    hb.log('  ⚠ a re-attribution of the water-supply row, never an addition to it: the same '
           'total read on the sink side is %s. Of the source-side value, %s serves the source '
           'country itself and %s is exported as rain.'
           % (f'{sink_total:,.2f}', f'{df_gep["moisture_recycling_gep_own_part"].sum():,.2f}',
              f'{df_gep["moisture_recycling_gep_export_part"].sum():,.2f}'))
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)

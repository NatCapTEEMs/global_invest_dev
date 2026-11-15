"""Moisture-recycling GEP tasks: the VMR crop value at the receiving country, with a source-side diagnostic.

This layer owns every file read and write. The science it calls lives in
moisture_recycling_functions, which never opens a file.

The published total is the VMR crop value at the receiving country, read live from the
water_supply run, and it does not use the WAM2layers matrix. The source-side diagnostic
beside it credits the land-sourced part of that value to the countries whose evaporation
supplied the rain. Both are intermediate contributions carved out of crop-provision value:
they overlap crop provision, agricultural water and each other, so none of them may be added
to crop provision or to one another without explicit overlap treatment.
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


def water_input_path(p):
    """Resolve and PIN the water_supply country table the calculation reads.

    An explicit `moisture_recycling_water_gep_path` wins; otherwise the relative file against the
    project dir. Pinning the resolved path on p is what puts the table's CONTENT fingerprint into
    the run signature (the signature fingerprints every service `_path` attribute), so a fresh
    water_supply run at the same path invalidates the saved rainfall table -- the
    agriculture_water convention. Must run BEFORE any reuse decision.
    """
    path = getattr(p, 'moisture_recycling_water_gep_path', None) or None
    if path is None:
        path = os.path.normpath(os.path.join(
            p.project_dir, p.moisture_recycling_water_gep_relative_file))
    p.moisture_recycling_water_gep_path = str(path)
    return str(path)


def gep_calculation(p):
    """GEP valuation for rainfall pattern regulation: the VMR crop value at the receiving country.

    The headline is the two VMR components the water_supply run computes (the VMR
    coefficients times the rainfed and irrigated 2019 crop-provision values), summed per
    country and REPORTED AT THE RECEIVING COUNTRY, the paper's approach. It is an
    intermediate contribution already inside crop provision, like agricultural water, never an
    addition.

    The WAM2layers source-side diagnostic is computed BESIDE the headline as
    `moisture_recycling_source_attributed`. The matrix columns sum below one, so it credits only
    the land-sourced part of the value, and its source-side and sink-side totals agree with each
    other, not with the headline. The matrix's missing rows, its untracked destinations and the
    ecosystem share touch the diagnostic only.
    """
    publish_inputs(p)
    service_results, already_done = utilities.begin_gep_calculation(p, 'moisture_recycling')
    # The live water input is resolved BEFORE the reuse decision. An output-exists return that
    # never looks at the table it was computed from republishes a stale rainfall figure after
    # every fresh water_supply run -- the exact failure the run signature exists for.
    water_path = water_input_path(p)
    outputs = [service_results['gep_by_country_base_year']]
    dependencies = [__file__, mr.__file__]
    if already_done and utilities.reuse_reason(
            p, 'moisture_recycling', outputs, light_inputs=dependencies) is None:
        return

    if not p.run_this:
        return p
    matrix = mr.moisture_matrix(hb.df_read(str(p.get_path(p.moisture_recycling_sink_fraction_input_path))))
    if not os.path.isfile(str(water_path)):
        raise FileNotFoundError('Rainfall regulation requires the water_supply country table with '
                                f'the VMR components: {water_path}. Run water_supply first or set '
                                'moisture_recycling_water_gep_path.')
    water = hb.df_read(str(water_path))
    component_cols = json.loads(p.moisture_recycling_water_gep_component_cols) \
        if isinstance(p.moisture_recycling_water_gep_component_cols, str) \
        else list(p.moisture_recycling_water_gep_component_cols)
    water_value = water.set_index('iso3_r250_label')[component_cols].sum(axis=1)

    # The headline at the receiving country: a country with both VMR components missing stays
    # missing, one present component counts.
    received = water.set_index('iso3_r250_label')[component_cols].sum(axis=1, min_count=1)
    received.name = 'moisture_recycling_gep'

    dropped = sorted(set(water_value.index) - set(matrix.columns))
    if dropped:
        hb.log('  %d countries carry VMR value but are absent from the moisture matrix, so the '
               'source-side DIAGNOSTIC cannot place them: %s (their value: %s)'
               % (len(dropped), ', '.join(dropped[:8]) + ('...' if len(dropped) > 8 else ''),
                  f'{water_value.reindex(dropped).sum():,.0f}'))
    untracked = mr.untracked_destinations(matrix)
    if untracked:
        hb.log('  %d destinations are below the tracking grid (all-NaN columns) and stay out of '
               'the source-side diagnostic: %s (their value: %s)'
               % (len(untracked), ', '.join(untracked),
                  f'{water_value.reindex(untracked).fillna(0.0).sum():,.0f}'))

    # The source-side DIAGNOSTIC: a WAM-weighted source-side diagnostic, not a redistribution
    # of the full headline. Published beside the headline, never as it.
    df_source = mr.reattributed_water_value(matrix, water_value.fillna(0.0),
                                            float(p.moisture_recycling_ecosystem_share))
    df_source = df_source.rename(columns={
        'moisture_recycling_gep': 'moisture_recycling_source_attributed',
        'moisture_recycling_gep_own_part': 'moisture_recycling_source_attributed_own_part',
        'moisture_recycling_gep_export_part': 'moisture_recycling_source_attributed_export_part'})
    source_total = df_source['moisture_recycling_source_attributed'].sum()
    sink_total = df_source['moisture_recycling_sink_side_value'].sum()
    if abs(source_total - sink_total) > 1.0:
        raise NameError('the moisture-recycling source-side diagnostic does not balance: source '
                        'side %r against sink side %r. The matrix and the value series disagree on '
                        'the country set.' % (source_total, sink_total))

    # The full collapse frame keeps ee_r264_id: the published table drops it through
    # published_country_columns, and the map merge is what needs it.
    countries = utilities.collapse_countries_to_r250(p.df_countries)
    aggregate_sources = sorted(set(df_source['iso3_r250_label']) - set(countries['iso3_r250_label']))
    if aggregate_sources:
        lost = df_source.set_index('iso3_r250_label').loc[
            aggregate_sources, 'moisture_recycling_source_attributed'].sum()
        hb.log('  %d matrix sources are aggregates with no account country and leave the '
               'published diagnostic: %s (their value: %s)'
               % (len(aggregate_sources), ', '.join(aggregate_sources), f'{lost:,.0f}'))
    df_gep = countries.merge(received.reset_index(), on='iso3_r250_label', how='left')
    df_gep = df_gep.merge(df_source, on='iso3_r250_label', how='left')
    df_gep['year'] = int(p.gep_base_year)
    utilities.write_gep_by_country(
        p, df_gep[utilities.published_country_columns(df_gep, 'moisture_recycling')],
        service_results['gep_by_country_base_year'])

    gdf = hb.df_merge(p.gdf_countries_simplified, df_gep, how='outer',
                      left_on='ee_r264_id', right_on='ee_r264_id')
    gdf.to_file(service_results['gep_by_country_base_year'].replace('.csv', '.gpkg'), driver='GPKG')
    utilities.write_reuse_signature(p, 'moisture_recycling', outputs, light_inputs=dependencies)

    headline_total = df_gep['moisture_recycling_gep'].sum()
    hb.log(f'Total moisture_recycling (rainfall regulation) GEP for base year {p.gep_base_year}: '
           f'{headline_total:,.2f} at the receiving country — an intermediate contribution '
           f'already inside crop provision, never an addition.')
    hb.log('  WAM-weighted source-side diagnostic, not a redistribution of the full headline: '
           'sums to %s (sink side %s); %s serves the source country itself and %s is exported '
           'as rain.'
           % (f'{source_total:,.2f}', f'{sink_total:,.2f}',
              f'{df_gep["moisture_recycling_source_attributed_own_part"].sum():,.2f}',
              f'{df_gep["moisture_recycling_source_attributed_export_part"].sum():,.2f}'))
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    if not p.run_this:
        return p
    utilities.render_service_results(p)

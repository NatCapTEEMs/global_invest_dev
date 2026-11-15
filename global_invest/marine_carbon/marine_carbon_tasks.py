"""Marine carbon GEP tasks: ocean areas from the marine correspondence layer, then the valuation.

`ocean_areas` measures each country's EEZ and the total ocean in an equal-area CRS (about 80
seconds on the full layer), `gep_calculation` shares the four pools out by those areas and writes
the country table plus the full allocation with its high-seas row.
"""
import os

import geopandas as gpd
import hazelbean as hb
import pandas as pd
from global_invest import utilities
from global_invest.marine_carbon import marine_carbon_functions as mcf


def publish_inputs(p):
    """Every GEP task's first line: the marine_carbon es_config row (the ocean layer as the
    regions surface), its es_parameters rows (pools, name patterns, area CRS), the shared country
    references and the results registry."""
    utilities.hydrate_es_config(p, 'marine_carbon', log=hb.log)
    utilities.hydrate_es_parameters(p, 'marine_carbon', log=hb.log)
    utilities.initialize_country_paths(p)
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def ocean_areas(p):
    """EEZ area per country and the total ocean area, from the marine correspondence layer."""
    publish_inputs(p)
    p.marine_carbon_eez_areas_path = os.path.join(p.cur_dir, 'eez_areas_by_country.csv')
    p.marine_carbon_total_ocean_area_path = os.path.join(p.cur_dir, 'total_ocean_area.csv')
    if not p.run_this:
        return p
    outputs = [p.marine_carbon_eez_areas_path, p.marine_carbon_total_ocean_area_path]
    light_inputs = [p.gep_regions_input_path, __file__, mcf.__file__]
    reason = utilities.reuse_reason(p, 'marine_carbon', outputs, light_inputs=light_inputs)
    if reason is None:
        hb.log('Ocean areas already computed from the current layer and settings; reused.')
        return p
    hb.log('Measuring ocean areas because %s.' % reason)

    gdf = gpd.read_file(p.gep_regions_input_path)
    eez_areas = mcf.eez_areas_by_country(gdf, p.marine_carbon_eez_name_pattern,
                                         p.marine_carbon_area_crs)
    total_km2 = mcf.total_ocean_area_km2(gdf, p.marine_carbon_ocean_name_pattern,
                                         p.marine_carbon_area_crs)
    utilities.write_csv(eez_areas, p.marine_carbon_eez_areas_path)
    utilities.write_csv(pd.DataFrame([{'total_ocean_area_km2': total_km2}]),
                        p.marine_carbon_total_ocean_area_path)
    utilities.write_reuse_signature(p, 'marine_carbon', outputs, light_inputs=light_inputs)
    hb.log('EEZs of %d countries cover %.0f km2 of %.0f km2 of ocean.'
           % (len(eez_areas), eez_areas['eez_area_km2'].sum(), total_km2))
    return p


def gep_calculation(p):
    """GEP valuation for marine carbon: the pools shared out by ocean area, one row per country,
    with the high-seas row kept in the full allocation table."""
    publish_inputs(p)
    allocation_path = os.path.join(p.cur_dir, 'allocation_with_high_seas_base_year.csv')
    service_results, already_done = utilities.begin_gep_calculation(p, 'marine_carbon', extra_results={
        'allocation_with_high_seas': allocation_path})
    outputs = list(service_results.values())
    light_inputs = [__file__, mcf.__file__]
    if already_done and utilities.reuse_reason(p, 'marine_carbon', outputs,
                                               light_inputs=light_inputs) is None:
        return
    if not p.run_this:
        return p

    # round_trip: pandas' default float parser can move a value by one unit in the last place, and
    # the valuation reproduces the source's areas to the last digit.
    eez_areas = pd.read_csv(p.marine_carbon_eez_areas_path, float_precision='round_trip')
    total_km2 = float(pd.read_csv(p.marine_carbon_total_ocean_area_path, float_precision='round_trip')
                      ['total_ocean_area_km2'].iloc[0])
    allocation = mcf.allocate_marine_carbon_value(eez_areas, total_km2, p.marine_carbon_pools)
    national, high_seas = mcf.national_and_high_seas(allocation)
    national['year'] = int(p.gep_base_year)

    # The country table goes onto the account's country list by label, so an EEZ row whose label
    # the list does not carry (or a row with no label) would vanish from it with its value.
    listed = set(utilities.collapse_countries_to_r250(p.df_countries)['iso3_r250_label'])
    unlisted = sorted(map(str, set(national['iso3_r250_label']) - listed))
    if unlisted:
        raise ValueError('EEZ rows whose labels the account country list does not carry would be '
                         'dropped from the country table: %s' % unlisted)

    utilities.write_csv(allocation, service_results['allocation_with_high_seas'])
    utilities.write_gep_by_country(
        p, national[utilities.published_country_columns(national, 'marine_carbon')],
        service_results['gep_by_country_base_year'])
    utilities.write_reuse_signature(p, 'marine_carbon', outputs, light_inputs=light_inputs)

    hb.log('Marine carbon value of the four pools, base year %d: %.2f, of which %d countries %.2f '
           'and the high seas %.2f.'
           % (p.gep_base_year, allocation['allocated_value_usd'].sum(), len(national),
              national['marine_carbon_gep'].sum(), high_seas['allocated_value_usd'].iloc[0]))
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    if not p.run_this:
        return p
    utilities.render_service_results(p)

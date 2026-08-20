"""Recreation/tourism GEP tasks: site quality -> visits -> travel-cost value -> country GEP.

Ported from the GEP recreation repo into the house shape: tasks publish their own inputs
(publish_inputs), read data references from es_parameters.csv, and aggregate DIRECTLY on r250
(the source pipeline's own choice -- its country-id raster is iso3_r250_id -- so the aggregation
surface and the one-row-per-country collapse coincide; no r264 sum can double-count here).

Data staging map (drive Recreation/data/ tree -> base_data/global_invest/recreation/):
0_inputs/* -> the module root; 0_processed_raster_inputs/<sub>/* -> <sub>/*. File names are
kept EXACTLY as shipped (including the fuel-cost CSV's spelling) so staging is a copy, not a
rename layer.
"""
import os

import pandas as pd
import hazelbean as hb
from global_invest import utilities
from global_invest.recreation import recreation_functions as rf


def publish_inputs(p):
    """Every GEP task's first line: the recreation es_config row (defaults layer -- a caller-set
    value wins), the recreation data references from es_parameters (the staged drive data), the
    shared country references and the results registry. The aggregation surface
    (gep_regions_input_path) is the r250 country vector itself; gep_base_year (2019) is also the
    UNWTO overnight target year and the fuel-cost price year."""
    utilities.hydrate_es_config(p, 'recreation', log=hb.log)
    utilities.hydrate_es_parameters(p, 'recreation', log=hb.log)
    utilities.initialize_country_paths(p)
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def environment_index(p):
    """Environment class raster (0-3) from the six SEALS7 class-share rasters + PA share."""
    publish_inputs(p)
    p.environment_index_path = os.path.join(p.cur_dir, 'env_index_1km.tif')
    if not p.run_this:
        return
    if not hb.path_exists(p.environment_index_path):
        lulc_share_paths = {
            lulc_class: p.get_path(p.recreation_lulc_share_path_template.format(lulc_class=lulc_class))
            for lulc_class in rf.RECREATION_LULC_CLASSES}
        rf.calculate_environment_index(lulc_share_paths, p.recreation_pa_share_path,
                                       p.environment_index_path)
    return True


def accessibility_index(p):
    """Accessibility class raster (1-5) from urban share + distance-to-road."""
    publish_inputs(p)
    p.accessibility_index_path = os.path.join(p.cur_dir, 'accessibility_index_1km.tif')
    if not p.run_this:
        return
    if not hb.path_exists(p.accessibility_index_path):
        urban_share_path = p.get_path(
            p.recreation_lulc_share_path_template.format(lulc_class='urban'))
        rf.calculate_accessibility_index(urban_share_path, p.recreation_road_distance_path,
                                         p.accessibility_index_path)
    return True


def recreation_sites(p):
    """Site rank raster (1-9) from the two indices, and the high-quality-site (rank 9) mask."""
    publish_inputs(p)
    p.recreation_sites_ranked_path = os.path.join(p.cur_dir, 'rec_sites_ranked_1km.tif')
    p.recreation_hq_sites_path = os.path.join(p.cur_dir, 'high_quality_sites_1km.tif')
    if not p.run_this:
        return
    if not hb.path_exists(p.recreation_sites_ranked_path):
        rf.rank_recreation_sites(p.accessibility_index_path, p.environment_index_path,
                                 p.recreation_sites_ranked_path)
    if not hb.path_exists(p.recreation_hq_sites_path):
        rf.extract_high_quality_sites(p.recreation_sites_ranked_path, p.recreation_hq_sites_path)
    return True


def overnight_allocation(p):
    """UNWTO national overnights allocated to hotel pixels; also rasterizes the r250 country ids
    (the id raster every downstream task keys on). Overnight target year = gep_base_year."""
    publish_inputs(p)
    p.recreation_country_id_path = os.path.join(p.cur_dir, 'iso3_r250_ids_1km.tif')
    p.recreation_hotels_raster_path = os.path.join(p.cur_dir, 'hotels_1km.tif')
    p.recreation_unwto_panel_path = os.path.join(p.cur_dir, 'unwto_panel.csv')
    p.recreation_overnights_path = os.path.join(p.cur_dir, 'overnights_1km.tif')
    if not p.run_this:
        return
    # The road-length raster is used ONLY as the 1 km reference grid for rasterization.
    if not hb.path_exists(p.recreation_hotels_raster_path):
        rf.rasterize_presence(p.recreation_hotels_path, p.recreation_road_length_path,
                              p.recreation_hotels_raster_path)
    if not hb.path_exists(p.recreation_country_id_path):
        rf.rasterize_id_column(p.gep_regions_input_path, p.recreation_road_length_path,
                               p.gep_regions_id_col, p.recreation_country_id_path)
    if not hb.path_exists(p.recreation_overnights_path):
        if hb.path_exists(p.recreation_unwto_panel_path):
            overnight_df = pd.read_csv(p.recreation_unwto_panel_path)
        else:
            overnight_df = rf.clean_unwto_data(p.recreation_unwto_path,
                                               p.recreation_unwto_panel_path)
        country_overnights_map = rf.build_country_overnights_map(overnight_df, int(p.gep_base_year))
        hb.log('recreation: overnight totals mapped for %d countries (%.4g total overnights @%d)'
               % (len(country_overnights_map), sum(country_overnights_map.values()),
                  int(p.gep_base_year)))
        rf.allocate_overnights_raster(country_overnights_map, p.recreation_hotels_raster_path,
                                      p.recreation_country_id_path, p.recreation_overnights_path)
    return True


def daily_recreation(p):
    """Resident visits and travel-cost value into high-quality sites (gravity kernels)."""
    publish_inputs(p)
    p.daily_recreation_visits_path = os.path.join(p.cur_dir, 'site_visits_1km.tif')
    p.daily_recreation_value_path = os.path.join(p.cur_dir, 'site_value_1km.tif')
    if not p.run_this:
        return
    if not hb.path_all_exist([p.daily_recreation_visits_path, p.daily_recreation_value_path]):
        rf.calculate_kernel_recreation(
            p.recreation_population_path, p.recreation_country_id_path,
            p.recreation_fuel_cost_path, p.recreation_hq_sites_path,
            p.daily_recreation_visits_path, p.daily_recreation_value_path,
            working_dir=os.path.join(p.cur_dir, 'kernel'))
    return True


def tourist_recreation(p):
    """Tourist visits and travel-cost value into high-quality sites: the same kernel engine,
    with the allocated-overnights raster in the population role."""
    publish_inputs(p)
    p.tourist_recreation_visits_path = os.path.join(p.cur_dir, 'site_visits_1km.tif')
    p.tourist_recreation_value_path = os.path.join(p.cur_dir, 'site_value_1km.tif')
    if not p.run_this:
        return
    if not hb.path_all_exist([p.tourist_recreation_visits_path, p.tourist_recreation_value_path]):
        rf.calculate_kernel_recreation(
            p.recreation_overnights_path, p.recreation_country_id_path,
            p.recreation_fuel_cost_path, p.recreation_hq_sites_path,
            p.tourist_recreation_visits_path, p.tourist_recreation_value_path,
            working_dir=os.path.join(p.cur_dir, 'kernel'))
    return True


def gep_calculation(p):
    """GEP valuation for recreation: sum the four result rasters per country (already r250 --
    the id raster is iso3_r250_id, so no crosswalk collapse is needed) and write ONE row per
    country. recreation_gep = daily value + tourist value; visit columns are the quantities."""
    publish_inputs(p)
    service_results = {}
    p.results['recreation'] = service_results
    service_results['gep_by_country_base_year'] = os.path.join(p.cur_dir, 'gep_by_country_base_year.csv')

    if hb.path_all_exist(list(service_results.values())):
        hb.log('All results already exist. Skipping GEP calculation for recreation.')
        return
    hb.log('Starting GEP calculation for recreation.')

    # 1. Per-country sums of the four value/visit rasters, keyed on the r250 id raster.
    df = rf.sum_rasters_by_country_id(
        {'daily_visits': p.daily_recreation_visits_path,
         'daily_value': p.daily_recreation_value_path,
         'tourist_visits': p.tourist_recreation_visits_path,
         'tourist_value': p.tourist_recreation_value_path},
        p.recreation_country_id_path)
    df['recreation_gep'] = df['daily_value'] + df['tourist_value']
    df['year'] = int(p.gep_base_year)

    # 2. Attach per-country attributes (from the r264 correspondence, one row per iso3) and
    #    write the per-country CSV (source of truth for every sum).
    attr_cols = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
                 'continent', 'region_un', 'region_wb', 'income_grp', 'subregion']
    attrs = p.df_countries[attr_cols].drop_duplicates('iso3_r250_id')
    keep_cols = attr_cols + ['year', 'daily_visits', 'daily_value',
                             'tourist_visits', 'tourist_value', 'recreation_gep']
    df_gep = df.merge(attrs, how='left', on='iso3_r250_id')[keep_cols]
    hb.df_write(df_gep, service_results['gep_by_country_base_year'])

    # 3. Map only: r264-expanded, each sub-region carries its country's value, never summed.
    gdf = hb.df_merge(p.gdf_countries_simplified, df[['iso3_r250_id', 'recreation_gep']],
                      how='outer', left_on='iso3_r250_id', right_on='iso3_r250_id')
    gdf.to_file(service_results['gep_by_country_base_year'].replace('.csv', '.gpkg'),
                driver='GPKG')

    # 4. National total = sum over the one-row-per-country table.
    value_gep_base_year = df_gep['recreation_gep'].sum()
    hb.log(f'Total recreation GEP for base year {p.gep_base_year}: {value_gep_base_year}')
    return value_gep_base_year


def gep_load_results(p):
    """Load GEP results from a PRIOR calculation run so the report renders without recomputing.
    Fails loudly if absent (run run_recreation.py first). The results-only entry point."""
    publish_inputs(p)
    result_path = os.path.join(p.intermediate_dir, 'gep_calculation', 'gep_by_country_base_year.csv')
    if not hb.path_exists(result_path):
        raise FileNotFoundError(
            f'recreation GEP results not found at {result_path}. '
            f'Run the calculation first (run_recreation.py), then re-run results.')
    p.results.setdefault('recreation', {})
    p.results['recreation']['gep_by_country_base_year'] = result_path


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)

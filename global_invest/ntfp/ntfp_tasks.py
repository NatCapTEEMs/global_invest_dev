"""NTFP GEP tasks: accessible forest hectares per country, then the CWoN NWFP value per hectare.

The accessibility stage runs on a public roads dataset (GRIP4) and Natural Earth river
centrelines, because the source script's own roads file was never committed. That substitution
is the open replication ask, not a modelling choice.
"""
import os

import pandas as pd
import hazelbean as hb
from global_invest import utilities

from global_invest.ntfp import ntfp_accessibility as na
from global_invest.ntfp import ntfp_functions as nf

# The country id raster the zonal totals are keyed on, and the highest id it carries. The r250
# ids are ISO numeric codes, so they run past 250; bincount needs the ceiling, not the count.
COUNTRY_ID_MAX = 900


def publish_inputs(p):
    """Every GEP task's first line: the ntfp es_config row and the data references from
    es_parameters (defaults layer -- a caller-set value prevails), the shared country
    references and the results registry."""
    utilities.hydrate_es_config(p, 'ntfp', log=hb.log)
    utilities.hydrate_es_parameters(p, 'ntfp', log=hb.log)
    utilities.initialize_country_paths(p)
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def accessible_forest(p):
    """Forest hectares within the accessibility buffer of a road or river, per country.

    Everything happens on an equal-area 1 km grid so the buffer is a constant ten kilometres
    and a cell is exactly a hundred hectares. The intermediate rasters are cached in this
    task's directory, so a re-run does the warps once.
    """
    publish_inputs(p)
    p.ntfp_accessible_forest_path = os.path.join(p.cur_dir, 'accessible_forest_ha_by_country.csv')
    if not p.run_this:
        return
    if hb.path_exists(p.ntfp_accessible_forest_path):
        hb.log('Accessible forest already computed. Skipping.')
        return

    import numpy as np
    from osgeo import gdal

    river_mask_path = os.path.join(p.cur_dir, 'rivers_mask.tif')
    if not hb.path_exists(river_mask_path):
        na.rasterize_rivers(p.ntfp_rivers_path, p.ntfp_roads_length_path, river_mask_path)

    access_source_path = os.path.join(p.cur_dir, 'access_source_mask.tif')
    if not hb.path_exists(access_source_path):
        na.build_access_mask(p.ntfp_roads_length_path, river_mask_path, access_source_path)

    access_equal_area_path = os.path.join(p.cur_dir, 'access_source_equal_area_1km.tif')
    if not hb.path_exists(access_equal_area_path):
        na.warp_to_equal_area(access_source_path, access_equal_area_path, 'max',
                              output_type=gdal.GDT_Byte)

    lulc_equal_area_path = os.path.join(p.cur_dir, 'lulc_equal_area_1km.tif')
    if not hb.path_exists(lulc_equal_area_path):
        na.warp_to_equal_area(p.gep_lulc_input_path, lulc_equal_area_path, 'mode',
                              output_type=gdal.GDT_Int16)

    countries_equal_area_path = os.path.join(p.cur_dir, 'countries_equal_area_1km.tif')
    if not hb.path_exists(countries_equal_area_path):
        na.warp_to_equal_area(p.ntfp_country_id_raster_path, countries_equal_area_path,
                              'near', output_type=gdal.GDT_Int32)

    buffer_cells = int(round(nf.NTFP_ACCESS_BUFFER_M / na.ACCESSIBILITY_CELL_SIZE_M))
    source_mask = gdal.Open(access_equal_area_path).ReadAsArray() > 0
    hb.log(f'ntfp: buffering {int(source_mask.sum()):,} road and river cells by '
           f'{buffer_cells} km')
    access_mask = na.buffer_mask_by_cells(source_mask, buffer_cells)

    hectares = na.accessible_forest_hectares_by_country(
        lulc_equal_area_path, access_mask, countries_equal_area_path, COUNTRY_ID_MAX)

    countries = p.df_countries[['iso3_r250_id', 'iso3_r250_label']].drop_duplicates('iso3_r250_id')
    countries = countries[countries['iso3_r250_id'].notna()]
    countries['accessible_forest_ha'] = [
        float(hectares[int(i)]) if int(i) <= COUNTRY_ID_MAX else float('nan')
        for i in countries['iso3_r250_id']]
    hb.df_write(countries, p.ntfp_accessible_forest_path)
    hb.log(f'ntfp: {countries["accessible_forest_ha"].sum():,.0f} accessible forest hectares '
           f'over {int((countries["accessible_forest_ha"] > 0).sum())} countries')
    return True


def gep_calculation(p):
    """GEP valuation for NTFP: accessible forest hectares times the CWoN NWFP value per hectare."""
    publish_inputs(p)
    service_results = {}
    p.results['ntfp'] = service_results
    service_results['gep_by_country_base_year'] = os.path.join(p.cur_dir, 'gep_by_country_base_year.csv')

    if hb.path_all_exist(list(service_results.values())):
        hb.log('All results already exist. Skipping GEP calculation for ntfp.')
        return
    hb.log('Starting GEP calculation for ntfp.')

    accessible = hb.df_read(p.ntfp_accessible_forest_path)
    value_per_ha = hb.df_read(p.ntfp_value_per_ha_path)
    df_gep = nf.ntfp_gep_by_country(accessible, value_per_ha, int(p.gep_base_year))

    attribute_columns = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
                         'continent', 'region_un', 'region_wb', 'income_grp', 'subregion']
    countries = p.df_countries[attribute_columns].drop_duplicates('iso3_r250_id')
    df_gep = countries.merge(df_gep.drop(columns=['iso3_r250_id'], errors='ignore'),
                             on='iso3_r250_label', how='left')
    df_gep['year'] = int(p.gep_base_year)
    hb.df_write(df_gep, service_results['gep_by_country_base_year'])
    hb.log(f'Total ntfp GEP for base year {int(p.gep_base_year)}: '
           f'{df_gep["ntfp_gep"].sum():,.2f}')
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)

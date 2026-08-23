"""Timber-provision GEP tasks: the value raster summed to countries in the library.

The reported number is OUR run: the pipeline's value raster (verified equal to the
net-return layer masked and floored, see timber_provision_functions) summed per country on
the 10-arcsecond country-id raster. The committed Forestry CSV stays as the test anchor
the run is compared against, never as the output."""
import os

import numpy as np
import pandas as pd
import rasterio
import hazelbean as hb
from global_invest import utilities
from global_invest.timber_provision import timber_provision_functions as tp


def publish_inputs(p):
    """Every GEP task's first line: the timber_provision es_config row and the data reference
    from es_parameters (defaults layer -- a caller-set value prevails), the shared country
    references and the results registry."""
    utilities.hydrate_es_config(p, 'timber_provision', log=hb.log)
    utilities.hydrate_es_parameters(p, 'timber_provision', log=hb.log)
    utilities.initialize_country_paths(p)
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def gep_calculation(p):
    """GEP valuation for timber provision: the value raster summed per country."""
    publish_inputs(p)
    service_results = {}
    p.results['timber_provision'] = service_results
    service_results['gep_by_country_base_year'] = os.path.join(p.cur_dir, 'gep_by_country_base_year.csv')

    if hb.path_all_exist(list(service_results.values())):
        hb.log('All results already exist. Skipping GEP calculation for timber_provision.')
        return
    hb.log('Starting GEP calculation for timber_provision.')

    value_src = rasterio.open(p.timber_provision_value_raster_path)
    zone_src = rasterio.open(p.timber_provision_zone_raster_path)
    n_zones = 1000
    zone_sums = np.zeros(n_zones + 1, dtype='float64')
    rows_per_block = 2048
    for row0 in range(0, value_src.shape[0], rows_per_block):
        h = min(rows_per_block, value_src.shape[0] - row0)
        win = rasterio.windows.Window(0, row0, value_src.shape[1], h)
        zone_sums += utilities.sum_by_zone(value_src.read(1, window=win), zone_src.read(1, window=win), n_zones)

    attr_cols = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
                 'continent', 'region_un', 'region_wb', 'income_grp', 'subregion']
    countries = p.df_countries[attr_cols].drop_duplicates('iso3_r250_id')
    df_gep = tp.timber_gep_from_zone_sums(zone_sums, countries)
    df_gep['year'] = int(p.gep_base_year)
    hb.df_write(df_gep[attr_cols + ['year', 'timber_provision_gep']],
                service_results['gep_by_country_base_year'])

    hb.log(f'Total timber_provision GEP for base year {p.gep_base_year}: '
           f'{df_gep["timber_provision_gep"].sum():,.2f}')
    committed = hb.df_read(p.timber_provision_gep_path)
    hb.log(f'Committed Forestry table total: {committed.select_dtypes("number").iloc[:, -1].sum():,.2f} (the test anchor).')
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)

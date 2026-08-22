"""Stormwater GEP tasks: the retained volume from the InVEST run, priced per cubic metre.

The volumes are made outside this task tree, by an InVEST urban stormwater retention run
and the zonal step in stormwater_zonal.py. These tasks read the per-country table those
steps write (stormwater_retention_by_country_path in es_parameters) and price it."""
import os

import hazelbean as hb
from global_invest import utilities


def publish_inputs(p):
    """Every GEP task's first line: the stormwater es_config row and the data references
    from es_parameters (defaults layer -- a caller-set value prevails), the shared country
    references and the results registry."""
    utilities.hydrate_es_config(p, 'stormwater', log=hb.log)
    utilities.hydrate_es_parameters(p, 'stormwater', log=hb.log)
    utilities.initialize_country_paths(p)
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def gep_calculation(p):
    """GEP valuation for stormwater: retained volume times the price per cubic metre.
    The volumes are read from the zonal table stormwater_zonal.py writes, named in
    es_parameters as stormwater_retention_by_country_path."""
    publish_inputs(p)
    service_results = {}
    p.results['stormwater'] = service_results
    service_results['gep_by_country_base_year'] = os.path.join(p.cur_dir, 'gep_by_country_base_year.csv')

    if hb.path_all_exist(list(service_results.values())):
        hb.log('All results already exist. Skipping GEP calculation for stormwater.')
        return
    hb.log('Starting GEP calculation for stormwater.')

    import pandas as pd
    from global_invest.stormwater import stormwater_functions as sf
    retention = pd.read_csv(p.stormwater_retention_by_country_path)
    df_gep = sf.stormwater_gep_by_country(retention, sf.STORMWATER_PRICE_PER_M3_PLACEHOLDER)
    df_gep['year'] = int(p.gep_base_year)
    hb.df_write(df_gep, service_results['gep_by_country_base_year'])
    hb.log('Total stormwater retention: %.4g m3/yr over %d countries; GEP %.4g USD at the '
           'placeholder price of %s per m3 (the open ask).' % (
               df_gep['retention_m3'].sum(), (df_gep['retention_m3'] > 0).sum(),
               df_gep['stormwater_gep'].sum(), sf.STORMWATER_PRICE_PER_M3_PLACEHOLDER))
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)

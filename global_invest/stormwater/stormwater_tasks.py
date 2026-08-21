"""Stormwater GEP tasks. The valuation function is in place and unit-tested; the InVEST
urban stormwater retention run over the staged global inputs is the in-progress part --
it produces the retention volumes the valuation prices."""
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
    The InVEST retention run is in progress; running this task before its output lands
    stops here with the state spelled out."""
    publish_inputs(p)
    service_results = {}
    p.results['stormwater'] = service_results
    service_results['gep_by_country_base_year'] = os.path.join(p.cur_dir, 'gep_by_country_base_year.csv')

    if hb.path_all_exist(list(service_results.values())):
        hb.log('All results already exist. Skipping GEP calculation for stormwater.')
        return
    raise NotImplementedError(
        'The InVEST urban stormwater retention run over the staged global inputs is in '
        'progress -- it produces the per-country retention volumes this task prices. The '
        'valuation function is in place and unit-tested (stormwater_functions), with the '
        'committed price placeholder of 1 held as a named constant until the author names '
        'the intended price.')


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)

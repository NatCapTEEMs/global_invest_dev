"""NTFP GEP tasks. The valuation function is in place and unit-tested; the accessibility
raster stage (forest mask x 10 km road/river buffer, per country) is the in-progress part
of the port -- it needs the public roads dataset staged, then both source-script variants
run through the same task."""
import os

import hazelbean as hb
from global_invest import utilities


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


def gep_calculation(p):
    """GEP valuation for NTFP: accessible forest hectares times the CWoN NWFP value per
    hectare, both script variants. The accessibility stage is being ported; running this
    task before it lands stops here with the state of the port."""
    publish_inputs(p)
    service_results = {}
    p.results['ntfp'] = service_results
    service_results['gep_by_country_base_year'] = os.path.join(p.cur_dir, 'gep_by_country_base_year.csv')

    if hb.path_all_exist(list(service_results.values())):
        hb.log('All results already exist. Skipping GEP calculation for ntfp.')
        return
    raise NotImplementedError(
        'The NTFP accessibility stage is being ported: the forest mask (ESA classes 50-90) '
        'intersected with the 10 km road/river buffer, run for both source-script variants '
        'on a public roads dataset (the committed roads .shp was never committed). The '
        'valuation function it feeds is in place and unit-tested (ntfp_functions).')


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)

"""Unit tests for the extractive_energy port (CWoN rents computed, committed CSVs as anchors)."""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from global_invest import utilities
from global_invest.extractive_energy import extractive_energy_functions as xe


def _base_data_project():
    """A bare ProjectFlow, only for its base_data_dir.

    The anchors are inputs, so a test finds them the way a run does rather than by walking
    directories of its own.
    """
    import tempfile
    import hazelbean as hb
    return hb.ProjectFlow(project_dir=os.path.join(tempfile.mkdtemp(), 'anchors'))


REFERENCE_DIR = utilities.service_data_dir(_base_data_project(), 'extractive_energy')


def test_components_and_sum_reproduce_the_committed_anchors():
    gas = pd.read_csv(os.path.join(REFERENCE_DIR, 'gep-gas.csv'))
    coal = pd.read_csv(os.path.join(REFERENCE_DIR, 'gep-coal.csv'))
    oil = pd.read_csv(os.path.join(REFERENCE_DIR, 'gep-petrolium.csv'))
    countries = gas[['iso3_r250_id', 'iso3_r250_label']]
    out = xe.extractive_energy_gep_by_country(gas, coal, oil, countries)
    assert len(out) == 250
    for src, col, n in ((gas, 'extractive_energy_gas_gep', 111),
                        (coal, 'extractive_energy_coal_gep', 211),
                        (oil, 'extractive_energy_oil_gep', 110)):
        assert out[col].notna().sum() == n
        assert np.isclose(out[col].sum(), src.select_dtypes('number').iloc[:, -1].sum())
    # The service total sums the components that exist; a coal-only country is not NaN.
    coal_only = out[out['extractive_energy_gas_gep'].isna()
                    & out['extractive_energy_coal_gep'].notna()]
    assert len(coal_only) > 0
    assert coal_only['extractive_energy_gep'].notna().all()
    # And a country in no table stays NaN, never zero.
    none = out[[f'extractive_energy_{f}_gep' for f in xe.EXTRACTIVE_ENERGY_FUELS]].isna().all(axis=1)
    assert out.loc[none, 'extractive_energy_gep'].isna().all()


def test_cwon_rents_replicate_the_committed_anchors_except_named_rows():
    """The computed valuation (CWoN <fuel>_rent_cd at YR2019) equals the committed drive
    CSVs country by country. The exceptions are the eight rows named in the functions
    module, which carry another country's full value in the committed CSVs."""
    for dta, csv, vcol in (('gas_rent_cd.dta', 'gep-gas.csv', 'gep_gas'),
                           ('coal_rent_cd.dta', 'gep-coal.csv', 'gep_coal'),
                           ('oil_rent_cd.dta', 'gep-petrolium.csv', 'gep_oil')):
        cwon = pd.read_stata(os.path.join(REFERENCE_DIR, dta))
        committed = pd.read_csv(os.path.join(REFERENCE_DIR, csv))
        computed = xe.fuel_rent_by_country_from_cwon(cwon, 2019)
        j = committed.merge(computed, on='iso3_r250_label', how='left')
        plain = j[~j['iso3_r250_label'].isin(xe.COMMITTED_DUPLICATE_VALUE_ROWS)
                  & j[vcol].notna()]
        assert len(plain) > 60
        matched = plain.dropna(subset=['rent_cd'])
        assert np.allclose(matched[vcol], matched['rent_cd'], rtol=1e-5)
        # A committed value with no CWoN row outside the named duplicates would be a new
        # unexplained source -- there are none.
        assert plain[plain['rent_cd'].isna()][vcol].eq(0).all()


def test_committed_duplicate_rows_equal_their_source_country():
    """Documents the join artifact the port removes: each named row's committed value IS
    another country's value, so the committed totals count those countries twice."""
    for csv, vcol in (('gep-gas.csv', 'gep_gas'), ('gep-coal.csv', 'gep_coal'),
                      ('gep-petrolium.csv', 'gep_oil')):
        committed = pd.read_csv(os.path.join(REFERENCE_DIR, csv)).set_index('iso3_r250_label')[vcol]
        checked = 0
        for row, source in xe.COMMITTED_DUPLICATE_VALUE_ROWS.items():
            if pd.notna(committed.get(row)) and committed.get(row) != 0:
                assert committed[row] == committed[source], (row, source, vcol)
                checked += 1
        assert checked > 0


def test_es_config_and_parameters_rows_hydrate_extractive_energy(tmp_path):
    from global_invest import utilities
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'extractive_energy', log=lambda *a: None)
    assert p.gep_base_year == 2019
    utilities.hydrate_es_parameters(p, 'extractive_energy', log=lambda *a: None)
    assert p.extractive_energy_gas_path.endswith('gep-gas.csv')
    assert p.extractive_energy_coal_path.endswith('gep-coal.csv')
    assert p.extractive_energy_oil_path.endswith('gep-petrolium.csv')
    assert p.extractive_energy_cwon_gas_path.endswith('cwon/gas_rent_cd.dta')
    assert p.extractive_energy_cwon_coal_path.endswith('cwon/coal_rent_cd.dta')
    assert p.extractive_energy_cwon_oil_path.endswith('cwon/oil_rent_cd.dta')

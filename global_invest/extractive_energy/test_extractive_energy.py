"""Unit tests for the extractive_energy port (CWoN rents computed, committed CSVs as anchors)."""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

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
    CSVs country by country, on the rows the committed CSVs value. The exceptions are the eight
    rows named in the functions module, which carry another country's full value in the
    committed CSVs. South Sudan is empty there and so is not compared."""
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


# --- gas and oil rent as production times unit rent -------------------------------------------------

def _cwon_table(series, values):
    """A CWoN-shaped table: countrycode, countryname, series and YR2019, in the order given."""
    return pd.DataFrame({'countrycode': list(values), 'countryname': list(values), 'series': series,
                         'YR2019': [float(v) for v in values.values()]})


def test_rent_is_production_times_unit_rent_matched_by_code():
    """Rows in different orders match by code. A zero on either side is a zero rent, and a missing
    value on either side, or a country in one table only, has no rent."""
    production = _cwon_table(xe.CWON_PRODUCTION_SERIES['oil'],
                             {'EEE': 7, 'AAA': 10, 'DDD': np.nan, 'BBB': 0, 'CCC': 4, 'FFF': 5})
    unit_rent = _cwon_table(xe.CWON_UNIT_RENT_SERIES['oil'],
                            {'CCC': 0, 'BBB': 5, 'AAA': 2, 'GGG': 3, 'DDD': 3, 'EEE': np.nan})
    rent = xe.fuel_rent_from_production_and_unit_rent(
        production, unit_rent, 2019, xe.CWON_PRODUCTION_SERIES['oil'], xe.CWON_UNIT_RENT_SERIES['oil']
    ).set_index('iso3_r250_label')['rent_cd']
    assert rent['AAA'] == 20.0
    assert rent['BBB'] == 0.0 and rent['CCC'] == 0.0
    assert rent[['DDD', 'EEE', 'FFF', 'GGG']].isna().all()
    assert len(rent) == 7


@pytest.mark.parametrize('broken', ['label', 'repeated_code'])
def test_a_wrong_series_label_or_a_repeated_code_is_refused(broken):
    production = _cwon_table(xe.CWON_PRODUCTION_SERIES['gas'], {'AAA': 1, 'BBB': 2})
    unit_rent = _cwon_table(xe.CWON_UNIT_RENT_SERIES['gas'], {'AAA': 3, 'BBB': 4})
    if broken == 'label':
        unit_rent['series'] = xe.CWON_UNIT_RENT_SERIES['oil']
    else:
        unit_rent = pd.concat([unit_rent, unit_rent.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        xe.fuel_rent_from_production_and_unit_rent(
            production, unit_rent, 2019, xe.CWON_PRODUCTION_SERIES['gas'], xe.CWON_UNIT_RENT_SERIES['gas'])


def test_a_valued_code_with_no_r250_row_is_refused():
    countries = pd.DataFrame({'iso3_r250_id': [1], 'iso3_r250_label': ['AAA']})
    tables = {s: _cwon_table(s, {'AAA': 1, 'ZZZ': 2})
              for s in list(xe.CWON_PRODUCTION_SERIES.values()) + list(xe.CWON_UNIT_RENT_SERIES.values())}
    coal = _cwon_table('Coal, rents, total (current US$)', {'AAA': 1, 'ZZZ': np.nan})
    with pytest.raises(ValueError, match='no r250 row'):
        xe.extractive_energy_gep_from_cwon(
            tables[xe.CWON_PRODUCTION_SERIES['gas']], tables[xe.CWON_UNIT_RENT_SERIES['gas']], coal,
            tables[xe.CWON_PRODUCTION_SERIES['oil']], tables[xe.CWON_UNIT_RENT_SERIES['oil']], countries, 2019)


@pytest.mark.parametrize('fuel, countries, published_total', [('oil', 99, 743262045332.98),
                                                              ('gas', 98, 192483771113.38)])
def test_staged_tables_reproduce_the_published_rent(fuel, countries, published_total):
    """Production times unit rent equals CWoN's published 2019 rent country by country, with the same
    missing countries. The bound is two machine epsilon: CWoN's division forming the unit rent and
    the multiplication here, each correctly rounded. The totals agree to the sum of those bounds."""
    cwon_dir = os.path.join(REFERENCE_DIR, 'cwon')
    paths = [os.path.join(cwon_dir, f'{fuel}_{name}.dta') for name in ('production', 'unit_rent_cd', 'rent_cd')]
    if not all(os.path.exists(path) for path in paths):
        pytest.skip('the CWoN production and unit-rent tables are not staged on this machine')
    production, unit_rent, published = (pd.read_stata(path) for path in paths)
    computed = xe.fuel_rent_from_production_and_unit_rent(
        production, unit_rent, 2019, xe.CWON_PRODUCTION_SERIES[fuel], xe.CWON_UNIT_RENT_SERIES[fuel])
    check = xe.compare_with_published_rent(computed, published, 2019)
    eps = np.finfo(float).eps
    assert check['masks_equal']
    assert check['countries_both'] == countries
    assert check['max_relative_difference'] <= 2 * eps
    assert round(check['total_published'], 2) == published_total
    assert abs(check['total_computed'] - check['total_published']) <= 2 * eps * check['total_published']


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
    for fuel in xe.COMPUTED_RENT_FUELS:
        assert getattr(p, f'extractive_energy_cwon_{fuel}_production_path').endswith(f'cwon/{fuel}_production.dta')
        assert getattr(p, f'extractive_energy_cwon_{fuel}_unit_rent_path').endswith(f'cwon/{fuel}_unit_rent_cd.dta')

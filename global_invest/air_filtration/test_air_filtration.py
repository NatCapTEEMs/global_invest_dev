"""Unit tests for the air_filtration port (workbook-anchored, two channels).

The anchor workbook ships in reference/, so every run verifies: the recomputed deaths x VSL
benefits against the workbook's own benefit columns, the identified global-average VSL fill
rule, the positional r250 join with its name floor, and the two channel totals -- deposition
matching the manuscript's air-filtration number. The documented-but-different-vintage VSL
method is pinned on synthetic data.
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from global_invest import utilities
from global_invest.air_filtration import air_filtration_functions as af


def _base_data_project():
    """A bare ProjectFlow, only for its base_data_dir.

    The anchors are inputs, so a test finds them the way a run does rather than by walking
    directories of its own.
    """
    import tempfile
    import hazelbean as hb
    return hb.ProjectFlow(project_dir=os.path.join(tempfile.mkdtemp(), 'anchors'))


REFERENCE_DIR = utilities.service_data_dir(_base_data_project(), 'air_filtration')


def _workbook():
    return pd.read_excel(os.path.join(REFERENCE_DIR, 'air_filtration_gep.xlsx'))


def test_recomputed_benefits_match_the_workbook_exactly():
    w = _workbook()
    out = af.air_quality_benefits(w)
    assert np.allclose(out['air_filtration_gep'], w['Dep_Benefit_USD'], rtol=1e-9)
    assert np.allclose(out['sandstorm_prevention_gep'], w['Dust_Benefit_USD'], rtol=1e-9)
    # The deposition total is the manuscript's air-filtration number.
    assert np.isclose(out['air_filtration_gep'].sum(), 17.81e9, rtol=1e-2)
    assert np.isclose(out['sandstorm_prevention_gep'].sum(), 595.4e9, rtol=1e-2)


def test_global_average_fill_rule_holds_and_returns_the_mean():
    w = _workbook()
    global_average = af.verify_global_average_fill(w)
    country_mean = w.loc[w['VSL_Source'] == 'country', 'VSL'].mean()
    assert np.isclose(global_average, country_mean, rtol=1e-12)
    broken = w.copy()
    broken.loc[broken['VSL_Source'] == 'global_avg', 'VSL'] = 1.0
    with pytest.raises(ValueError, match='fill rule'):
        af.verify_global_average_fill(broken)


def test_positional_r250_join_carries_ids_and_refuses_a_reordered_workbook():
    w = _workbook()
    r250 = pd.read_csv(os.path.join(REFERENCE_DIR, 'r250_gpkg_order.csv'))
    out = af.air_quality_gep_by_country(w, r250)
    assert len(out) == 250
    ita = out[out['iso3_r250_label'] == 'ITA'].iloc[0]
    w_ita = w[w['Country'] == 'Italy'].iloc[0]
    assert np.isclose(ita['air_filtration_gep'], w_ita['Dep_Benefit_USD'], rtol=1e-9)
    with pytest.raises(ValueError, match='positional join refused'):
        af.air_quality_gep_by_country(w.iloc[::-1].reset_index(drop=True), r250)


def test_documented_vsl_method_on_synthetic_data():
    # US: 80 - 40 = 40 life-years, GDP 60k -> VSL per life-year per GDP = 9.9e6 / 40 / 60000.
    # AAA: (75 - 45) = 30 life-years, GDP 20k -> VSL = 20000 * 30 * ratio = 2.475e6 * 30/40 * 20/60... hand: 9.9e6/40/60000 = 4.125; AAA = 20000*30*4.125 = 2.475e6.
    le = pd.DataFrame({'slug': ['united-states', 'aaa'], 'years': [80.0, 75.0]})
    age = pd.DataFrame({'slug': ['united-states', 'aaa'], 'years': [40.0, 45.0]})
    gdp = pd.DataFrame({'slug': ['united-states', 'aaa'], 'gdp_real': [60000.0, 20000.0]})
    vsl = af.gdp_adjusted_vsl(le, age, gdp).set_index('slug')['vsl']
    assert np.isclose(vsl['united-states'], af.AIR_FILTRATION_VSL_USA)
    assert np.isclose(vsl['aaa'], 2_475_000.0)


def test_es_config_and_parameters_rows_hydrate_air_filtration(tmp_path):
    from global_invest import utilities
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'air_filtration', log=lambda *a: None)
    assert p.gep_base_year == 2019
    utilities.hydrate_es_parameters(p, 'air_filtration', log=lambda *a: None)
    assert p.air_filtration_workbook_path.endswith('air_filtration/air_filtration_gep.xlsx')


def test_vsl_comes_from_the_country_table_and_distinguishes_serbia_from_kosovo():
    # The workbook calls both rows Serbia; only the r250 order says the second is Kosovo. A name
    # join hands Serbia's value to both and never resolves Kosovo, which is the failure this
    # guards. XYZ exercises the ISO3 alias, and NOP is priced by the workbook but absent from the
    # table, so it keeps the workbook's figure rather than being averaged away.
    workbook = pd.DataFrame({
        'Country': ['Serbia', 'Serbia', 'Turkey', 'Nowhere', 'Filler'],
        'VSL': [1.0, 2.0, 30.0, 40.0, 99.0],
        'VSL_Source': ['country', 'country', 'country', 'country', 'global_avg'],
    })
    order = pd.DataFrame({
        'ee_r264_description': ['Serbia', 'Kosovo', 'Turkey', 'Nowhere', 'Filler'],
        'iso3_r250_label': ['SRB', 'XKX', 'TUR', 'NOP', 'FIL'],
    })
    table = pd.DataFrame({'country': ['serbia', 'kosovo', 'turkey-turkiye'],
                          'vsl': [10.0, 20.0, 30.0]})

    vsl, matched, disagreeing, unsourced = af.vsl_from_country_table(workbook, order, table)

    assert vsl[0] == 10.0 and vsl[1] == 20.0      # each gets its own country's value
    assert vsl[2] == 30.0                          # TUR resolved through the ISO3 alias
    assert vsl[3] == 40.0                          # absent from the table, workbook figure kept
    assert matched == 3
    assert list(unsourced['iso3']) == ['NOP']
    assert np.isclose(vsl[4], np.mean([10.0, 20.0, 30.0]))   # global_avg row takes the mean
    assert set(disagreeing['iso3']) == {'SRB', 'XKX', 'FIL'}


def test_vsl_table_slug_handles_accents_and_aliases():
    assert af.vsl_table_slug("Côte d'lvoire", 'CIV') == 'cote-divoire'
    assert af.vsl_table_slug('South Korea', 'KOR') == 'korea-south'
    assert af.vsl_table_slug('United States', 'USA') == 'united-states'

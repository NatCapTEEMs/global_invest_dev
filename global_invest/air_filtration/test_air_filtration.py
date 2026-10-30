"""Unit tests for the air_filtration port (workbook-anchored, two channels).

The anchor workbook is read from the service's base-data folder, so every run verifies: the
recomputed deaths x VSL benefits against the workbook's own benefit columns (at the workbook's
own VSL), the workbook's global-average VSL fill rule, the positional r250 join with its name
floor, the two channel totals at the workbook's VSL -- deposition matching the manuscript's
air-filtration number -- and the identified not-estimated list. The documented-but-different-
vintage VSL method is pinned on synthetic data.
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


def _order(labels):
    return pd.DataFrame({'iso3_r250_id': range(1, len(labels) + 1), 'iso3_r250_label': labels})


def test_only_the_listed_countries_publish_as_missing():
    """The identified list, not a zero rule: a listed country publishes missing, an unlisted
    double zero stays a measured zero, and valued rows are untouched."""
    w = pd.DataFrame({'FID': [1, 2, 3], 'Country': ['A', 'B', 'C'],
                      'Dep_Deaths': [0.0, 0.0, 2.0], 'Dust_Deaths': [0.0, 0.0, 3.0],
                      'VSL': [10.0, 10.0, 10.0]})
    mask = af.not_estimated_mask(w, _order(['AAA', 'BBB', 'CCC']), ['AAA'])
    assert mask.tolist() == [True, False, False]


def test_a_listed_country_with_deaths_rejects_the_list():
    w = pd.DataFrame({'Dep_Deaths': [0.0, 4.0], 'Dust_Deaths': [0.0, 1.0]})
    with pytest.raises(ValueError, match='does not match this workbook'):
        af.not_estimated_mask(w, _order(['AAA', 'BBB']), ['AAA', 'BBB'])
    with pytest.raises(ValueError, match='outside the r250 order'):
        af.not_estimated_mask(w, _order(['AAA', 'BBB']), ['ZZZ'])


def test_the_staged_list_is_the_sources_not_estimated_set_for_this_workbook():
    """Row by row against the shipped workbook: the 53 listed countries are exactly its rows
    with zero deaths in both channels (the write-up's 53 not estimated), and the staged
    population evidence puts people in 46 of them."""
    w = _workbook()
    order = pd.read_csv(os.path.join(REFERENCE_DIR, 'r250_gpkg_order.csv'))
    listed = pd.read_csv(os.path.join(REFERENCE_DIR, 'air_filtration_not_estimated.csv'),
                         encoding='utf-8-sig')
    assert len(listed) == 53
    mask = af.not_estimated_mask(w, order, listed['iso3_r250_label'])
    double_zero = ((w['Dep_Deaths'] == 0) & (w['Dust_Deaths'] == 0)).to_numpy()
    assert (mask == double_zero).all()
    assert int(listed['populated_worldpop_2019'].sum()) == 46


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
    # US: 80 - 40 = 40 life-years, GDP 60k -> VSL per life-year per GDP = 9.9e6 / 40 / 60000 = 4.125.
    # AAA: 75 - 45 = 30 life-years, GDP 20k -> VSL = 20000 * 30 * 4.125 = 2.475e6.
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


def test_vsl_comes_from_the_shared_panel_and_a_missing_country_raises():
    # The panel keys on iso3_r250_label, so the workbook's duplicate "Serbia" naming (FID 137
    # and 232) cannot misprice Kosovo: each row takes its own label's value. The workbook
    # column prices nothing, and the repricing report carries the rows that moved.
    workbook = pd.DataFrame({
        'Country': ['Serbia', 'Serbia', 'Turkey'],
        'VSL': [10.0, 20.0, 30.0],
        'VSL_Source': ['country', 'country', 'country'],
    })
    order = pd.DataFrame({
        'ee_r264_description': ['Serbia', 'Kosovo', 'Turkey'],
        'iso3_r250_label': ['SRB', 'XKX', 'TUR'],
    })
    panel = pd.DataFrame({'iso3_r250_label': ['SRB', 'XKX', 'TUR'],
                          'vsl_usd': [10.0, 25.0, 30.0]})

    vsl, repriced = af.vsl_from_shared_panel(workbook, order, panel)

    assert list(vsl) == [10.0, 25.0, 30.0]
    assert list(repriced['iso3']) == ['XKX']
    with pytest.raises(ValueError):
        af.vsl_from_shared_panel(workbook, order, panel[panel['iso3_r250_label'] != 'TUR'])


def _mini_workbook_and_order():
    """Four rows in gpkg order, one duplicated name whose second row is another country's by
    position (the Serbia/Kosovo shape), plus a variant name the r250 list does not carry."""
    workbook = pd.DataFrame({
        'FID': [1, 2, 3, 4],
        'Country': ['Aaa', 'Bbb', 'Mayotte', 'Aaa'],
        'Dep_Deaths': [10.0, 20.0, 30.0, 40.0],
        'Dust_Deaths': [1.0, 2.0, 3.0, 4.0],
        'VSL': [2.0, 2.0, 2.0, 2.0],
    })
    order = pd.DataFrame({
        'brk_name': ['Aaa', 'Bbb', 'France', 'Kkk'],
        'iso3_r250_id': [1, 2, 3, 4],
        'iso3_r250_label': ['AAA', 'BBB', 'MYT', 'KKK'],
    })
    return workbook, order


def test_positional_join_accepts_the_shipped_order_and_the_duplicate_name(monkeypatch):
    monkeypatch.setattr(af, 'AIR_FILTRATION_MIN_NAME_MATCHES', 2)
    workbook, order = _mini_workbook_and_order()
    out = af.air_quality_gep_by_country(workbook, order).set_index('iso3_r250_label')
    assert out.loc['KKK', 'air_filtration_gep'] == 80.0   # the duplicate row lands by position


def test_positional_join_refuses_a_two_row_swap(monkeypatch):
    """Swapping two whole rows carries their FIDs with them, so the FID-order guard refuses it.
    The name floor alone accepts such a swap, and both countries would silently take each
    other's deaths."""
    monkeypatch.setattr(af, 'AIR_FILTRATION_MIN_NAME_MATCHES', 1)
    workbook, order = _mini_workbook_and_order()
    workbook.iloc[[0, 1]] = workbook.iloc[[1, 0]].values
    with pytest.raises(ValueError, match='not in FID order'):
        af.air_quality_gep_by_country(workbook, order)


def test_positional_join_refuses_a_data_column_swap_with_fids_left_in_place(monkeypatch):
    """Reordering only the data columns preserves the FID sequence, so the per-row name guard is
    what refuses it: a known country name at the wrong position. A data-only swap between two
    VARIANT rows (names the r250 list does not carry) leaves no signal at all -- that blind spot
    is recorded here rather than asserted away."""
    monkeypatch.setattr(af, 'AIR_FILTRATION_MIN_NAME_MATCHES', 1)
    workbook, order = _mini_workbook_and_order()
    cols = ['Country', 'Dep_Deaths', 'Dust_Deaths', 'VSL']
    workbook.loc[[0, 1], cols] = workbook.loc[[1, 0], cols].values
    with pytest.raises(ValueError, match='wrong'):
        af.air_quality_gep_by_country(workbook, order)

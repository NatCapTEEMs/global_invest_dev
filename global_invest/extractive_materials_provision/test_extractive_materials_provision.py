"""Unit tests for extractive_materials_provision's valuation.

Every function is pinned on a hand-built frame. The reshape test carries the expression the
module used before the read moved into the task layer, so the two are compared directly rather
than the new one being trusted.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from global_invest.extractive_materials_provision import extractive_materials_provision_functions as em
from global_invest.extractive_materials_provision import extractive_materials_provision_tasks as em_tasks


def a_wide_indicator_frame(first_column='Country Name'):
    """Two countries, three years, in the shape the World Bank publishes."""
    return pd.DataFrame({
        first_column: ['Aruba', 'Angola'],
        'Country Code': ['ABW', 'AGO'],
        'Indicator Name': ['Mineral rents (% of GDP)'] * 2,
        'Indicator Code': ['NY.GDP.MINR.RT.ZS'] * 2,
        '2017': [0.0, 1.5],
        '2018': [0.5, 2.5],
        '2019': [1.0, np.nan],
    })


def test_the_wide_indicator_melts_to_one_row_per_country_and_year():
    out = em.world_bank_wide_to_long(a_wide_indicator_frame(), 'mineral_rent')
    assert list(out.columns) == ['Country Code', 'year', 'mineral_rent']
    assert len(out) == 6                                   # 2 countries x 3 years
    assert list(out['year'].unique()) == ['2017', '2018', '2019']
    ago_2018 = out[(out['Country Code'] == 'AGO') & (out['year'] == '2018')]
    assert ago_2018['mineral_rent'].iloc[0] == 2.5


def test_the_reshape_matches_the_expression_it_replaced():
    """The original dropped three named columns and melted the rest. Selecting the year columns
    instead gives the same frame, and does so whatever the first header is called."""
    wide = a_wide_indicator_frame()
    original = wide.drop(columns=['Country Name', 'Indicator Name', 'Indicator Code']).melt(
        id_vars=['Country Code'], var_name='year', value_name='mineral_rent')
    new = em.world_bank_wide_to_long(wide, 'mineral_rent')
    pd.testing.assert_frame_equal(new, original)


def test_a_byte_order_mark_on_the_first_header_cannot_reach_the_result():
    """Read as ISO-8859-1 the first header arrives as 'i>>?Country Name'. The reshape names the
    year columns rather than the columns to drop, so the mangled header is dropped either way."""
    mangled = em.world_bank_wide_to_long(
        a_wide_indicator_frame(first_column='﻿Country Name'), 'mineral_rent')
    clean = em.world_bank_wide_to_long(a_wide_indicator_frame(), 'mineral_rent')
    pd.testing.assert_frame_equal(mangled, clean)


def test_mineral_rent_gep_is_the_rent_share_of_gdp_times_the_factor():
    """A country with 10 percent mineral rents on 1,000 of GDP holds 100 of rent, of which
    the factor keeps 0.49."""
    assert em.mineral_rent_gep(10.0, 1000.0, 0.49) == 49.0
    assert em.mineral_rent_gep(0.0, 1000.0, 0.49) == 0.0


def test_mineral_rent_gep_runs_over_a_column_the_way_the_task_calls_it():
    df = pd.DataFrame({'mineral_rent': [10.0, 0.0, 2.5], 'GDP_currentUSD': [1000.0, 500.0, 400.0]})
    valued = em.mineral_rent_gep(df['mineral_rent'], df['GDP_currentUSD'], 0.49)
    assert np.allclose(valued.values, [49.0, 0.0, 4.9])


def test_the_factor_the_module_ships_is_the_one_under_question():
    """The 0.49 is the service's open question, so a silent change must fail a test."""
    assert em_tasks.MINERAL_RENT_GEP_FACTOR == 0.49


def test_group_countries_sums_the_value_column_by_year():
    """The country-year rows collapse to one row per year, sorted, whatever order they arrive in."""
    df = pd.DataFrame({'year': [2019, 2018, 2019, 2018],
                       'iso3_r250_label': ['AAA', 'AAA', 'BBB', 'BBB'],
                       'Value': [1.0, 2.0, 10.0, 20.0]})
    by_year = em.group_countries(df).set_index('year')['Value']
    assert list(by_year.index) == [2018, 2019]
    assert by_year.loc[2018] == 22.0
    assert by_year.loc[2019] == 11.0
    assert by_year.sum() == df['Value'].sum()


def test_es_config_row_hydrates_extractive_materials(tmp_path):
    from global_invest import utilities
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'extractive_materials_provision', log=lambda *a: None)
    assert p.gep_base_year == 2019


def test_the_base_year_the_task_filters_on_comes_from_es_config(tmp_path):
    """The row filter reads p.gep_base_year rather than a literal, so the shipped es_config cell
    is what selects the base-year table."""
    from global_invest import utilities
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'extractive_materials_provision', log=lambda *a: None)
    df = pd.DataFrame({'year': [2018, 2019, 2020], 'extractive_materials_provision_gep': [1.0, 2.0, 3.0]})
    base_year_rows = df.loc[df['year'] == int(p.gep_base_year)]
    assert list(base_year_rows['extractive_materials_provision_gep']) == [2.0]

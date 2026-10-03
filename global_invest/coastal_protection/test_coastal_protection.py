"""Unit tests for coastal_protection's valuation steps.

Each function is pinned on a hand-built table small enough that the expected number is written
out in the assertion: the source reader's clean-up, the two component aggregations with their
split-country handling, the combination, and the country join. The task module's workbook
reader is pinned on a workbook written into tmp_path. The coral deflation is a scalar from the
shared utilities.usd_deflation_factor, pinned in global_invest_tests/test_currency_conversion.py.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from global_invest.coastal_protection import coastal_protection_functions as cpf
from global_invest.coastal_protection import coastal_protection_tasks as cpt

# The r264 correspondence, with AAA split into two sub-regions that share the country's NAME.
# The coral table is keyed on name, so both rows match it; the mangrove table is keyed on label,
# so each row matches its own sub-region.
COUNTRIES = pd.DataFrame({
    'ee_r264_id': [1, 2, 3, 4],
    'iso3_r250_id': [10, 20, 30, 10],
    'ee_r264_label': ['AAA', 'BBB', 'CCC', 'Z01'],
    'iso3_r250_label': ['AAA', 'BBB', 'CCC', 'AAA'],
    'ee_r264_name': ['Aaaland', 'Bbbland', 'Cccland', 'Aaaland'],
    'iso3_r250_name': ['Aaaland', 'Bbbland', 'Cccland', 'Aaaland'],
    'continent': ['Africa', 'Asia', 'Europe', 'Africa'],
    'region_un': ['Africa', 'Asia', 'Europe', 'Africa'],
    'region_wb': ['SSA', 'EAP', 'ECA', 'SSA'],
    'income_grp': ['5. Low income', '3. Upper middle income', '1. High income: OECD', '5. Low income'],
    'subregion': ['Western Africa', 'Eastern Asia', 'Western Europe', 'Western Africa'],
})

def test_mangrove_value_is_computed_from_area_and_price_not_read():
    """The workbook publishes a finished value, but it also carries what that value is made of.
    Computing it is what would let a disagreement with the published column surface."""
    raw = pd.DataFrame({
        'countrycode': ['AAA', 'BBB', 'CCC'],
        'countryname': ['Aaaland', 'Bbbland', 'Cccland'],
        'mangrove_ha': [10.0, 20.0, np.nan],
        'value_per_ha_2019': [10.0, 10.0, 10.0],
        'annual_value_2019': [100.0, 199.0, 50.0],
        'year': [2019.0, 2019.0, 2019.0],
    })
    out = cpf.clean_mangrove_values(raw).set_index('ee_r264_label')
    assert out.loc['AAA', 'Value'] == 100.0
    assert out.loc['BBB', 'Value'] == 200.0          # ours, not the published 199
    assert out.loc['BBB', 'Value_published'] == 199.0
    assert pd.isna(out.loc['CCC', 'Value'])          # no area is not no protection
    assert out['year'].dtype.kind == 'i'


def test_mangrove_gep_sums_a_split_country_into_one_row():
    mangrove = pd.DataFrame({
        'ee_r264_label': ['AAA', 'Z01', 'BBB', 'ZZZ'],
        'year': [2019, 2019, 2019, 2019],
        'Value': [100.0, 40.0, 250.0, 999.0],
    })
    out = cpf.mangrove_gep_by_country(COUNTRIES, mangrove).set_index('iso3_r250_label')
    assert out.loc['AAA', 'coastal_protection_gep_mangrove'] == 140.0   # 100 + 40, one row
    assert out.loc['BBB', 'coastal_protection_gep_mangrove'] == 250.0
    assert 'ZZZ' not in out.index          # not in the correspondence, dropped by the inner join
    assert len(out) == 2                   # CCC has no mangrove row at all


def test_coral_reef_gep_deflates_to_the_base_year_and_counts_a_split_country_once():
    coral = pd.DataFrame({
        'ee_r264_name': ['Aaaland', 'Bbbland'],
        'coral_reef_value': [100.0, 50.0],
        'year': [2011, 2011],
    })
    out = cpf.coral_reef_gep_by_country(COUNTRIES, coral, 1.21, base_year=2019)

    assert out['year'].unique().tolist() == [2019]        # the 2011 rows do not survive
    values = out.set_index('iso3_r250_label')['coastal_protection_gep_coral_reef']
    # Aaaland matches both AAA and its sub-region Z01; the de-duplication leaves one row. The
    # factor is a scalar, so every country shares it.
    assert np.isclose(values.loc['AAA'], 121.0)           # 100 x 1.21
    assert np.isclose(values.loc['BBB'], 60.5)            # 50 x 1.21
    assert len(out) == 2


def test_combine_coastal_components_adds_them_and_keeps_a_country_present_in_only_one():
    mangrove = pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB'], 'year': [2019, 2019],
                             'coastal_protection_gep_mangrove': [140.0, 250.0]})
    coral = pd.DataFrame({'iso3_r250_label': ['AAA', 'CCC'], 'year': [2019, 2019],
                          'coastal_protection_gep_coral_reef': [121.0, 7.0]})
    out = cpf.combine_coastal_components(mangrove, coral).set_index('iso3_r250_label')

    assert out.loc['AAA', 'coastal_protection_gep'] == 261.0     # both components
    assert out.loc['BBB', 'coastal_protection_gep'] == 250.0     # mangrove only, coral filled 0
    assert out.loc['CCC', 'coastal_protection_gep'] == 7.0       # coral only, mangrove filled 0
    assert out['Value'].tolist() == out['coastal_protection_gep'].tolist()


def test_combine_coastal_components_keeps_a_missing_coral_value_missing():
    """A country whose coral component arrives missing is unvalued, and reporting its mangrove
    component alone would hide that its coral component is unknown."""
    mangrove = pd.DataFrame({'iso3_r250_label': ['AAA'], 'year': [2019],
                             'coastal_protection_gep_mangrove': [140.0]})
    coral = pd.DataFrame({'iso3_r250_label': ['AAA'], 'year': [2019],
                          'coastal_protection_gep_coral_reef': [np.nan]})
    out = cpf.combine_coastal_components(mangrove, coral).set_index('iso3_r250_label')

    assert np.isnan(out.loc['AAA', 'coastal_protection_gep'])
    assert out.loc['AAA', 'coastal_protection_gep_mangrove'] == 140.0
    # A missing total drops out of the reported sum rather than entering it as a zero.
    assert out['coastal_protection_gep'].sum() == 0.0


def test_attach_country_attributes_gives_one_row_per_country():
    gep = pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB'], 'year': [2019, 2019],
                        'coastal_protection_gep': [261.0, 250.0]})
    out = cpf.attach_country_attributes(gep, COUNTRIES)

    # AAA has two r264 rows; joining against the correspondence uncollapsed would give three rows.
    assert len(out) == 2
    assert out.set_index('iso3_r250_label').loc['AAA', 'ee_r264_id'] == 1
    assert out.set_index('iso3_r250_label').loc['BBB', 'continent'] == 'Asia'
    assert 'geometry' not in out.columns


def test_the_currency_years_that_define_the_deflator_span_are_pinned():
    # The coral table is 2011 USD and the service reports 2019; the shared CPI factor spans them.
    assert cpf.CORAL_REEF_VALUE_YEAR == 2011
    assert cpf.COASTAL_PROTECTION_BASE_YEAR == 2019


def test_task_reader_cleans_the_mangrove_workbook(tmp_path):
    path = str(tmp_path / 'mangroves.xlsx')
    pd.DataFrame({'countrycode': ['AAA'], 'countryname': ['Aaaland'],
                  'mangrove_ha': [10.0], 'value_per_ha_2019': [10.0],
                  'annual_value_2019': [100.0], 'year': [2019.0]}).to_excel(
        path, sheet_name=cpt.SOURCE_SHEET_NAME, index=False)
    out = cpt.read_mangrove_values(path)
    assert out['ee_r264_label'].tolist() == ['AAA']
    assert out['Value'].tolist() == [100.0]
    assert out['year'].dtype.kind == 'i'


def test_es_config_row_hydrates_coastal_protection(tmp_path):
    from global_invest import utilities
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'coastal_protection', log=lambda *a: None)
    assert p.gep_base_year == cpf.COASTAL_PROTECTION_BASE_YEAR
    assert p.gep_quantity_input_path.endswith('data_mangroves_2019.xlsx')

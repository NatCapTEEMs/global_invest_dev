"""Unit tests for coastal_protection's valuation steps.

Each function is pinned on a hand-built table small enough that the expected number is written
out in the assertion: the source reader's clean-up, the two component aggregations with their
split-country handling, the combination, and the country join. The task module's workbook
reader is pinned on a workbook written into tmp_path. The coral deflation is a scalar from the
shared utilities.usd_deflation_factor, pinned in global_invest_tests/test_currency_conversion.py.
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from global_invest.coastal_protection import coastal_protection_functions as cpf
from global_invest.coastal_protection import coastal_protection_tasks as cpt


def _configured_base_data_path(parameter):
    """The base_data file an es_parameters row names, found the way the task finds it."""
    import tempfile
    import hazelbean as hb
    params = pd.read_csv(os.path.join(os.path.dirname(cpf.__file__), '..', 'input_template',
                                      'es_parameters.csv'), encoding='utf-8-sig')
    value = params.loc[params['parameter'] == parameter, 'value'].iloc[0]
    project = hb.ProjectFlow(project_dir=os.path.join(tempfile.mkdtemp(), 'anchors'))
    return os.path.join(project.base_data_dir, value)

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
    assert out.loc['BBB', 'coastal_protection_gep'] == 250.0     # mangrove only, coral missing
    assert np.isnan(out.loc['BBB', 'coastal_protection_gep_coral_reef'])
    assert out.loc['CCC', 'coastal_protection_gep'] == 7.0       # coral only, mangrove missing
    assert np.isnan(out.loc['CCC', 'coastal_protection_gep_mangrove'])
    assert out['Value'].tolist() == out['coastal_protection_gep'].tolist()


def test_combine_coastal_components_reports_a_known_component_beside_a_missing_one():
    """A country whose coral component arrives missing keeps its measured mangrove component as
    the known-component total -- an explicit lower bound -- while the both-components column
    stays missing and the coverage label names the one component the total covers."""
    mangrove = pd.DataFrame({'iso3_r250_label': ['AAA'], 'year': [2019],
                             'coastal_protection_gep_mangrove': [140.0]})
    coral = pd.DataFrame({'iso3_r250_label': ['AAA'], 'year': [2019],
                          'coastal_protection_gep_coral_reef': [np.nan]})
    out = cpf.combine_coastal_components(mangrove, coral).set_index('iso3_r250_label')

    assert out.loc['AAA', 'coastal_protection_gep'] == 140.0
    assert out.loc['AAA', 'coastal_protection_gep_mangrove'] == 140.0
    assert np.isnan(out.loc['AAA', 'coastal_protection_gep_both_components'])
    assert out.loc['AAA', cpf.COVERAGE_COLUMN] == 'mangrove protection only'


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


def test_unrated_positive_area_sums_to_missing_not_zero():
    """Positive mangrove area with no rate is an unknown value, and the per-country sum must say
    so rather than publish 0.0: 27 real countries in the source, with 1,529,289 ha between them,
    are in this shape."""
    df_raw = pd.DataFrame({
        'countrycode': ['AAA', 'AAA'], 'year': [2019, 2019],
        'mangrove_ha': [100.0, 50.0], 'value_per_ha_2019': [np.nan, np.nan],
        'annual_value_2019': [np.nan, np.nan]})
    cleaned = cpf.clean_mangrove_values(df_raw)
    out = cpf.mangrove_gep_by_country(COUNTRIES, cleaned).set_index('iso3_r250_label')

    assert np.isnan(out.loc['AAA', 'coastal_protection_gep_mangrove'])


def test_zero_area_is_a_measured_zero_even_with_no_rate():
    """Zero habitat earns zero whatever a hectare would have been worth, so 0 x missing is a
    measured zero, not an unknown (landlocked rows arrive exactly like this)."""
    df_raw = pd.DataFrame({
        'countrycode': ['AAA'], 'year': [2019],
        'mangrove_ha': [0.0], 'value_per_ha_2019': [np.nan], 'annual_value_2019': [np.nan]})
    cleaned = cpf.clean_mangrove_values(df_raw)
    out = cpf.mangrove_gep_by_country(COUNTRIES, cleaned).set_index('iso3_r250_label')

    assert out.loc['AAA', 'coastal_protection_gep_mangrove'] == 0.0


def test_missing_area_stays_missing():
    """A row with no area at all (the source's Taiwan row) is an unknown, not a zero."""
    df_raw = pd.DataFrame({
        'countrycode': ['AAA'], 'year': [2019],
        'mangrove_ha': [np.nan], 'value_per_ha_2019': [np.nan], 'annual_value_2019': [np.nan]})
    cleaned = cpf.clean_mangrove_values(df_raw)
    out = cpf.mangrove_gep_by_country(COUNTRIES, cleaned).set_index('iso3_r250_label')

    assert np.isnan(out.loc['AAA', 'coastal_protection_gep_mangrove'])


def test_partial_observation_sums_the_measured_rows_alone_and_is_not_complete():
    """A split country with one rated and one unrated sub-region keeps the measured part: the
    sum is a lower bound carried by min_count=1, not a missing value -- and through the
    combine the both-components column stays missing and the label calls the mangrove
    component partial. No real country is in this shape today; the rule is pinned so one can
    be."""
    df_raw = pd.DataFrame({
        'countrycode': ['AAA', 'Z01'], 'year': [2019, 2019],
        'mangrove_ha': [100.0, 50.0], 'value_per_ha_2019': [2.0, np.nan],
        'annual_value_2019': [np.nan, np.nan]})
    cleaned = cpf.clean_mangrove_values(df_raw)
    mang = cpf.mangrove_gep_by_country(COUNTRIES, cleaned)
    assert mang.set_index('iso3_r250_label').loc['AAA', 'coastal_protection_gep_mangrove'] == 200.0

    coral = pd.DataFrame({'iso3_r250_label': ['AAA'], 'year': [2019],
                          'coastal_protection_gep_coral_reef': [7.0]})
    out = cpf.combine_coastal_components(mang, coral).set_index('iso3_r250_label')
    assert out.loc['AAA', 'coastal_protection_gep'] == 207.0
    assert np.isnan(out.loc['AAA', 'coastal_protection_gep_both_components'])
    assert out.loc['AAA', cpf.COVERAGE_COLUMN] == 'partial mangrove and coral protection'


def test_a_missing_area_row_blocks_the_both_components_total():
    """A country with one measured row and one no-area row is partially observed: the known
    components stand, the both-components column does not."""
    df_raw = pd.DataFrame({
        'countrycode': ['AAA', 'Z01'], 'year': [2019, 2019],
        'mangrove_ha': [100.0, np.nan], 'value_per_ha_2019': [2.0, np.nan],
        'annual_value_2019': [np.nan, np.nan]})
    cleaned = cpf.clean_mangrove_values(df_raw)
    mang = cpf.mangrove_gep_by_country(COUNTRIES, cleaned)
    coral = pd.DataFrame({'iso3_r250_label': ['AAA'], 'year': [2019],
                          'coastal_protection_gep_coral_reef': [7.0]})
    out = cpf.combine_coastal_components(mang, coral).set_index('iso3_r250_label')
    assert out.loc['AAA', 'coastal_protection_gep'] == 207.0
    assert np.isnan(out.loc['AAA', 'coastal_protection_gep_both_components'])
    assert out.loc['AAA', cpf.COVERAGE_COLUMN] == 'partial mangrove and coral protection'


def test_combine_distinguishes_every_observation_case():
    """One frame per observation case: both measured, coral absent, mangrove unknown, nothing
    measured, a reported zero mangrove with coral absent, and reported zeros in both -- the two
    totals, the component columns and the coverage label separate them."""
    mangrove = pd.DataFrame({
        'iso3_r250_label': ['A', 'B', 'C', 'D', 'E', 'F'], 'year': [2019] * 6,
        'coastal_protection_gep_mangrove': [10.0, 5.0, np.nan, np.nan, 0.0, 0.0]})
    coral = pd.DataFrame({
        'iso3_r250_label': ['A', 'C', 'F'], 'year': [2019] * 3,
        'coastal_protection_gep_coral_reef': [7.0, 3.0, 0.0]})
    out = cpf.combine_coastal_components(mangrove, coral).set_index('iso3_r250_label')
    both, label = 'coastal_protection_gep_both_components', cpf.COVERAGE_COLUMN

    assert out.loc['A', 'coastal_protection_gep'] == 17.0
    assert out.loc['A', both] == 17.0
    assert out.loc['A', label] == 'mangrove and coral protection'
    assert out.loc['B', 'coastal_protection_gep'] == 5.0          # coral absent beside measured
    assert np.isnan(out.loc['B', 'coastal_protection_gep_coral_reef'])
    assert np.isnan(out.loc['B', both])
    assert out.loc['B', label] == 'mangrove protection only'
    assert out.loc['C', 'coastal_protection_gep'] == 3.0          # known-component lower bound
    assert np.isnan(out.loc['C', 'coastal_protection_gep_mangrove'])
    assert np.isnan(out.loc['C', both])
    assert out.loc['C', label] == 'coral protection only'
    assert np.isnan(out.loc['D', 'coastal_protection_gep'])       # nothing measured
    assert np.isnan(out.loc['D', both])
    assert pd.isna(out.loc['D', label])
    assert out.loc['E', 'coastal_protection_gep'] == 0.0          # reported zero, coral absent
    assert out.loc['E', 'coastal_protection_gep_mangrove'] == 0.0
    assert np.isnan(out.loc['E', both])
    assert out.loc['E', label] == 'mangrove protection only'
    assert out.loc['F', 'coastal_protection_gep'] == 0.0          # reported zeros in both
    assert out.loc['F', both] == 0.0
    assert out.loc['F', label] == 'mangrove and coral protection'


def test_a_carried_missing_coral_row_is_unknown_not_absent():
    """A coral row whose value the table carries as missing is an unknown, not an absence:
    the component stays missing, the total is the measured mangrove alone (a lower bound), the
    both-components column stays missing and the label names the mangrove component alone."""
    coral_raw = pd.DataFrame({'ee_r264_name': ['Aaaland'], 'coral_reef_value': [np.nan],
                              'year': [2010]})
    coral = cpf.coral_reef_gep_by_country(COUNTRIES, coral_raw, 1.1, base_year=2019)
    mangrove = pd.DataFrame({'iso3_r250_label': ['AAA'], 'year': [2019],
                             'coastal_protection_gep_mangrove': [140.0],
                             'mangrove_fully_measured': [True]})
    out = cpf.combine_coastal_components(mangrove, coral).set_index('iso3_r250_label')
    assert np.isnan(out.loc['AAA', 'coastal_protection_gep_coral_reef'])
    assert out.loc['AAA', 'coastal_protection_gep'] == 140.0
    assert np.isnan(out.loc['AAA', 'coastal_protection_gep_both_components'])
    assert out.loc['AAA', cpf.COVERAGE_COLUMN] == 'mangrove protection only'


def test_partial_coral_observation_keeps_the_measured_sum_and_is_labelled_partial():
    """One valued and one missing coral row for the same country: the measured sum stands, the
    both-components column does not, and the label calls the coral component partial (no real
    country is in this shape today; the table is clean)."""
    coral_raw = pd.DataFrame({'ee_r264_name': ['Aaaland', 'Aaaland'],
                              'coral_reef_value': [100.0, np.nan], 'year': [2010, 2010]})
    coral = cpf.coral_reef_gep_by_country(COUNTRIES, coral_raw, 1.0, base_year=2019)
    mangrove = pd.DataFrame({'iso3_r250_label': ['AAA'], 'year': [2019],
                             'coastal_protection_gep_mangrove': [140.0],
                             'mangrove_fully_measured': [True]})
    out = cpf.combine_coastal_components(mangrove, coral).set_index('iso3_r250_label')
    assert out.loc['AAA', 'coastal_protection_gep_coral_reef'] == 100.0
    assert out.loc['AAA', 'coastal_protection_gep'] == 240.0
    assert np.isnan(out.loc['AAA', 'coastal_protection_gep_both_components'])
    assert out.loc['AAA', cpf.COVERAGE_COLUMN] == 'mangrove and partial coral protection'


def test_an_excluded_country_is_missing_with_or_without_coral_and_a_true_zero_stays_zero():
    """The mangrove source removed some countries from its analysis and its processing writes
    them zero area. Listed, their mangrove component is missing in every shape they arrive in:
    a zero row beside coral (EGY, 'coral protection only'), a zero row with no coral (GNB,
    nothing known and no label), and no mangrove row beside coral (COK). An unlisted zero-area
    country stays a measured zero."""
    mangrove = pd.DataFrame({
        'iso3_r250_label': ['EGY', 'GNB', 'ZRO'], 'year': [2019] * 3,
        'coastal_protection_gep_mangrove': [0.0, 0.0, 0.0],
        'mangrove_fully_measured': [True] * 3})
    coral = pd.DataFrame({
        'iso3_r250_label': ['EGY', 'COK'], 'year': [2019, 2019],
        'coastal_protection_gep_coral_reef': [7.0, 3.0],
        'coral_fully_measured': [True, True]})
    out = cpf.combine_coastal_components(
        mangrove, coral, mangrove_excluded_labels=['EGY', 'GNB', 'COK']).set_index('iso3_r250_label')

    assert np.isnan(out.loc['EGY', 'coastal_protection_gep_mangrove'])
    assert out.loc['EGY', 'coastal_protection_gep_coral_reef'] == 7.0
    assert out.loc['EGY', 'coastal_protection_gep'] == 7.0          # known-component lower bound
    assert np.isnan(out.loc['EGY', 'coastal_protection_gep_both_components'])
    assert out.loc['EGY', cpf.COVERAGE_COLUMN] == 'coral protection only'
    for column in ['coastal_protection_gep_mangrove', 'coastal_protection_gep_coral_reef',
                   'coastal_protection_gep', 'coastal_protection_gep_both_components']:
        assert np.isnan(out.loc['GNB', column]), column            # nothing known, not $0
    assert pd.isna(out.loc['GNB', cpf.COVERAGE_COLUMN])
    assert np.isnan(out.loc['COK', 'coastal_protection_gep_mangrove'])
    assert out.loc['COK', 'coastal_protection_gep'] == 3.0
    assert np.isnan(out.loc['COK', 'coastal_protection_gep_both_components'])
    assert out.loc['COK', cpf.COVERAGE_COLUMN] == 'coral protection only'
    assert out.loc['ZRO', 'coastal_protection_gep'] == 0.0           # unlisted: measured zero
    assert out.loc['ZRO', 'coastal_protection_gep_mangrove'] == 0.0
    assert out.loc['ZRO', cpf.COVERAGE_COLUMN] == 'mangrove protection only'


def test_an_exclusion_list_naming_a_valued_country_is_refused():
    """A list written for another table version must fail, not blank a measured value."""
    mangrove = pd.DataFrame({'iso3_r250_label': ['AAA'], 'year': [2019],
                             'coastal_protection_gep_mangrove': [140.0]})
    coral = pd.DataFrame({'iso3_r250_label': ['AAA'], 'year': [2019],
                          'coastal_protection_gep_coral_reef': [1.0]})
    with pytest.raises(ValueError, match='positive value'):
        cpf.combine_coastal_components(mangrove, coral, mangrove_excluded_labels=['AAA'])


def test_the_staged_exclusion_list_is_the_reports_six_geospatial_errors():
    """The six countries Building Coastal Resilience with Mangroves (World Bank 2024, Appendix 1,
    Data Cleaning items 3 and 4) removed for a bug in the geospatial analysis. The 18 removed for
    holding under 100 ha (item 1) are a separate case and are not on this list."""
    path = _configured_base_data_path('coastal_protection_mangrove_excluded_path')
    if not os.path.exists(path):
        pytest.skip('the mangrove exclusion list is not staged on this machine')
    listed = pd.read_csv(path, encoding='utf-8-sig')
    assert sorted(listed['iso3_r250_label']) == ['COD', 'EGY', 'GNB', 'HKG', 'IRN', 'VCT']
    assert sorted(listed['data_cleaning_item'].unique()) == [3, 4]


def test_coverage_label_names_both_one_or_neither_component():
    assert cpf.coverage_label('mangrove', 'coral') == cpf.BOTH_COMPONENTS_LABEL
    assert cpf.coverage_label(None, 'coral') == 'coral protection only'
    assert cpf.coverage_label('mangrove', np.nan) == 'mangrove protection only'
    assert cpf.coverage_label('partial mangrove', 'coral') == 'partial mangrove and coral protection'
    assert cpf.coverage_label(np.nan, None) is None


def test_an_absent_partner_stays_missing_and_known_values_pass_through_unchanged():
    """A country one table leaves out keeps that component missing: the coral table lists 80
    countries and leaves out some that have reefs, so absence is not zero protection. Measured
    values, including reported zeros, reach the output exactly as they arrived."""
    mangrove = pd.DataFrame({'iso3_r250_label': ['BTH', 'MNG', 'ZRO'], 'year': [2019] * 3,
                             'coastal_protection_gep_mangrove': [10.5, 250.25, 0.0]})
    coral = pd.DataFrame({'iso3_r250_label': ['BTH', 'COR'], 'year': [2019] * 2,
                          'coastal_protection_gep_coral_reef': [7.25, 3.5]})
    out = cpf.combine_coastal_components(mangrove, coral).set_index('iso3_r250_label')

    assert out.loc['BTH', 'coastal_protection_gep_mangrove'] == 10.5
    assert out.loc['BTH', 'coastal_protection_gep_coral_reef'] == 7.25
    assert out.loc['MNG', 'coastal_protection_gep_mangrove'] == 250.25
    assert np.isnan(out.loc['MNG', 'coastal_protection_gep_coral_reef'])   # absent, not 0
    assert np.isnan(out.loc['COR', 'coastal_protection_gep_mangrove'])     # absent, not 0
    assert out.loc['COR', 'coastal_protection_gep_coral_reef'] == 3.5
    assert out.loc['ZRO', 'coastal_protection_gep_mangrove'] == 0.0        # reported zero
    assert np.isnan(out.loc['ZRO', 'coastal_protection_gep_coral_reef'])
    assert out[cpf.COVERAGE_COLUMN].to_dict() == {
        'BTH': 'mangrove and coral protection', 'MNG': 'mangrove protection only',
        'COR': 'coral protection only', 'ZRO': 'mangrove protection only'}


def test_the_coverage_label_survives_the_published_table_writer(tmp_path):
    """The label is a supporting column, so the shared column selection keeps it and the
    country-list writer carries it. A country the service never produced (CCC) and one with
    nothing measured get a missing label beside missing values, and reported zeros and missing
    components read back as written."""
    from global_invest import utilities
    mangrove = pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB'], 'year': [2019] * 2,
                             'coastal_protection_gep_mangrove': [0.0, np.nan]})
    coral = pd.DataFrame({'iso3_r250_label': ['AAA'], 'year': [2019],
                          'coastal_protection_gep_coral_reef': [0.0]})
    df = cpf.attach_country_attributes(cpf.combine_coastal_components(mangrove, coral), COUNTRIES)
    df = df.drop(columns=['Value'])
    columns = utilities.published_country_columns(df, 'coastal_protection')
    assert cpf.COVERAGE_COLUMN in columns

    p = SimpleNamespace(df_countries=COUNTRIES)
    path = utilities.write_gep_by_country(p, df[columns], str(tmp_path / 'gep.csv'),
                                          log=lambda *a: None)
    out = pd.read_csv(path).set_index('iso3_r250_label')
    assert out.loc['AAA', cpf.COVERAGE_COLUMN] == 'mangrove and coral protection'
    assert out.loc['AAA', 'coastal_protection_gep_mangrove'] == 0.0
    assert out.loc['AAA', 'coastal_protection_gep_both_components'] == 0.0
    for country in ['BBB', 'CCC']:
        assert pd.isna(out.loc[country, cpf.COVERAGE_COLUMN]), country
        assert np.isnan(out.loc[country, 'coastal_protection_gep']), country

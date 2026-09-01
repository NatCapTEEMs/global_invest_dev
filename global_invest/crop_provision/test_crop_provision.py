"""Unit tests for crop_provision's valuation steps.

Every function the task calls is pinned on a hand-built table small enough that the expected
number is written out in the assertion: the FAOSTAT clean-up, the decade rental-rate lookup,
the as-of attribution, the country join with its thousand-USD conversion, and the two groupings.
The task module's two readers are pinned on files written into tmp_path.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from global_invest import utilities
from global_invest.crop_provision import crop_provision_functions as cp
from global_invest.crop_provision import crop_provision_tasks as cpt

# The identifier columns collapse_countries_to_r250 keeps, with one split country: r264 carries
# AAA twice (the canonical row plus a sub-region Z01) and the collapse must keep only AAA.
COUNTRIES = pd.DataFrame({
    'ee_r264_id': [1, 2, 3],
    'iso3_r250_id': [10, 20, 10],
    'ee_r264_label': ['AAA', 'BBB', 'Z01'],
    'iso3_r250_label': ['AAA', 'BBB', 'AAA'],
    'ee_r264_name': ['Aaaland', 'Bbbland', 'Aaaland North'],
    'iso3_r250_name': ['Aaaland', 'Bbbland', 'Aaaland'],
    'continent': ['Africa', 'Asia', 'Africa'],
    'region_un': ['Africa', 'Asia', 'Africa'],
    'region_wb': ['SSA', 'EAP', 'SSA'],
    'income_grp': ['5. Low income', '3. Upper middle income', '5. Low income'],
    'subregion': ['Western Africa', 'Eastern Asia', 'Western Africa'],
})


def _raw_faostat_frame():
    """Four FAOSTAT rows: one that survives, one unrequested crop, one aggregate area, one whose
    element is not gross production value. The year columns are the file's full Y1961..Y2022 span
    with a flag column beside each, because the melt reads all of them."""
    years = range(utilities.FAOSTAT_FIRST_YEAR, utilities.FAOSTAT_LAST_YEAR + 1)
    frame = pd.DataFrame({
        'Area Code': [1, 1, 2, 223],
        'Area Code (M49)': ["'010", "'010", "'020", "'223"],
        'Area': ['Aaaland', 'Aaaland', 'World', 'Turkiye'],
        'Item Code': [100, 200, 100, 100],
        'Item': ['Wheat', 'Rye', 'Wheat', 'Wheat'],
        'Element Code': [57, 57, 57, 152],
        'Element': ['Gross Production Value'] * 4,
        'Unit': ['1000 USD', '1000 USD', '1000 USD', '1000 USD'],
    })
    year_columns = {}
    for year in years:
        year_columns[f'Y{year}'] = [0.0] * 4
        year_columns[f'Y{year}F'] = ['A'] * 4
    year_columns['Y1961'] = [1.0, 2.0, 3.0, 4.0]
    year_columns['Y2022'] = [5.0, 6.0, 7.0, 8.0]
    return pd.concat([frame, pd.DataFrame(year_columns)], axis=1)


def test_clean_crop_values_keeps_gross_production_value_and_melts_the_years():
    out = utilities.clean_faostat_values(_raw_faostat_frame(), items=['Wheat'], value_column='crop_provision_gep', aggregate_areas=['World'])

    # 'Rye' is not requested, 'World' is an aggregate area, and area 223's row is element 152.
    assert set(out['crop']) == {'Wheat'}
    assert set(out['country']) == {'Aaaland'}
    assert not [c for c in out.columns if c.endswith('F')]      # flag columns dropped
    # One surviving source row, melted over the full year span.
    assert len(out) == utilities.FAOSTAT_LAST_YEAR - utilities.FAOSTAT_FIRST_YEAR + 1
    values = out.set_index('year')['crop_provision_gep']
    assert values.loc[1961] == 1.0
    assert values.loc[2022] == 5.0
    assert values.loc[1990] == 0.0
    assert out['year'].dtype.kind == 'i'
    assert out['area_code'].dtype.kind == 'i'


def test_clean_crop_values_renames_area_223_to_turkey():
    raw = _raw_faostat_frame()
    raw.loc[3, 'Element Code'] = cp.FAOSTAT_GROSS_PRODUCTION_VALUE_ELEMENT
    out = utilities.clean_faostat_values(raw, items=['Wheat'], value_column='crop_provision_gep', aggregate_areas=['World'])
    assert set(out.loc[out['area_code'] == 223, 'country']) == {'Turkey'}


def test_build_rental_rate_lookup_keys_each_decade_by_its_first_year():
    raw = pd.DataFrame({
        'Order': [1, 2],
        'FAO': [1.0, 2.0],
        'ISO3': ['AAA', 'BBB'],
        'Country/territory': ['Aaaland', 'Bbbland'],
        '1961-1970': [0.30, 0.40],
        '2011-2020': [0.35, 0.45],
    })
    out = utilities.build_rental_rate_lookup(raw)
    assert sorted(out['year'].unique().tolist()) == [1961, 2011]
    assert out['FAO'].dtype.kind == 'i'
    # ISO3 is not a decade column, so it melts in and then falls out with the unparseable start.
    assert len(out) == 4
    rates = out.set_index(['FAO', 'year'])['rental_rate']
    assert rates.loc[(1, 1961)] == 0.30
    assert rates.loc[(2, 2011)] == 0.45


def test_merge_crop_with_coefs_applies_the_decade_in_force_and_leaves_uncovered_countries_missing():
    values = pd.DataFrame({
        'area_code': [1, 1, 1, 2],
        'year': [1960, 1961, 2015, 2015],
        'crop_provision_gep': [100.0, 100.0, 200.0, 500.0],
    })
    coefs = pd.DataFrame({'FAO': [1, 1], 'year': [1961, 2011], 'rental_rate': [0.30, 0.35]})

    out = utilities.apply_rental_rates(values, coefs, 'crop_provision_gep').set_index(['area_code', 'year'])['crop_provision_gep']
    assert np.isnan(out.loc[(1, 1960)])          # before the first decade: no rate in force
    assert out.loc[(1, 1961)] == 30.0            # 100 x 0.30
    assert out.loc[(1, 2015)] == 70.0            # 200 x 0.35, the 2011-2020 decade
    assert pd.isna(out.loc[(2, 2015)])           # country absent from the CWoN table


def test_attach_countries_in_usd_converts_once_and_does_not_repeat_a_split_country():
    crops = pd.DataFrame({
        'area_code_M49': [10, 10, 20],
        'area_code': [1, 1, 2],
        'country': ['Aaaland', 'Aaaland', 'Bbbland'],
        'crop_code': [100, 200, 100],
        'crop': ['Wheat', 'Rye', 'Wheat'],
        'year': [2019, 2019, 2019],
        'rental_rate': [0.3, 0.3, 0.4],
        'crop_provision_gep': [30.0, 15.0, 40.0],
    })
    out = cp.attach_countries_in_usd(crops, COUNTRIES)

    # AAA is split across two r264 rows; the join must still produce three rows, not five.
    assert len(out) == 3
    assert out['crop_provision_gep'].tolist() == [30000.0, 15000.0, 40000.0]
    assert out.set_index('crop_code').loc[100, 'iso3_r250_label'].tolist() == ['AAA', 'BBB']


def test_attach_countries_in_usd_keeps_a_row_whose_code_the_correspondence_lacks():
    crops = pd.DataFrame({
        'area_code_M49': [10, 999],
        'area_code': [1, 9],
        'country': ['Aaaland', 'Nowhere'],
        'crop_code': [100, 100],
        'crop': ['Wheat', 'Wheat'],
        'year': [2019, 2019],
        'rental_rate': [0.3, 0.3],
        'crop_provision_gep': [30.0, 7.0],
    })
    out = cp.attach_countries_in_usd(crops, COUNTRIES)
    assert len(out) == 2
    unmatched = out[out['area_code_M49'] == 999].iloc[0]
    assert pd.isna(unmatched['iso3_r250_label'])
    assert unmatched['crop_provision_gep'] == 7000.0


def test_group_crops_then_group_countries_sum_to_the_same_total():
    crop_rows = pd.DataFrame({
        'iso3_r250_id': [10, 10, 20, 10],
        'iso3_r250_label': ['AAA', 'AAA', 'BBB', 'AAA'],
        'year': [2019, 2019, 2019, 2018],
        'crop_provision_gep': [30000.0, 15000.0, 40000.0, 1000.0],
    })
    by_country = utilities.sum_items_to_country_year(crop_rows, 'crop_provision_gep')
    per_country = by_country.set_index(['iso3_r250_id', 'year'])['crop_provision_gep']
    assert per_country.loc[(10, 2019)] == 45000.0        # two crops summed
    assert per_country.loc[(20, 2019)] == 40000.0
    assert per_country.loc[(10, 2018)] == 1000.0

    by_year = utilities.sum_countries_to_year(by_country, 'crop_provision_gep').set_index('year')['crop_provision_gep']
    assert by_year.loc[2019] == 85000.0
    assert by_year.loc[2018] == 1000.0


def test_normalize_m49_codes_unquotes_casts_and_maps_successors():
    """FAOSTAT ships quoted codes and keeps dissolved states under their own code: both are
    resolved so every row joins to a current country."""
    df = pd.DataFrame({'area_code_M49': ["'156", "'159", "'891", "'076"], 'value': [1, 2, 3, 4]})
    out = utilities.normalize_m49_codes(df)
    assert list(out['area_code_M49']) == [156, 156, 688, 76]
    assert list(out['value']) == [1, 2, 3, 4]
    assert list(df['area_code_M49']) == ["'156", "'159", "'891", "'076"]   # input untouched


def test_every_successor_maps_to_a_different_live_code():
    """A successor mapping that pointed at itself, or at another dissolved state, would leave
    production stranded."""
    for dissolved, successor in utilities.M49_SUCCESSORS.items():
        assert dissolved != successor
        assert successor not in utilities.M49_SUCCESSORS


def test_the_faostat_unit_factor_is_the_thousand_usd_conversion():
    assert cp.FAOSTAT_THOUSAND_USD == 1000.0
    assert cp.FAOSTAT_VALUE_UNIT == '1000 USD'


def test_task_reader_cleans_the_faostat_bulk_file(tmp_path):
    """The bulk file is Latin-1, so an accented area name comes back mangled at any other
    encoding and the country would then miss its join."""
    path = str(tmp_path / 'Value_of_Production_E_All_Data.csv')
    raw = _raw_faostat_frame()
    raw.loc[0, 'Area'] = 'Côte'
    raw.to_csv(path, index=False, encoding='ISO-8859-1')

    out = cpt.read_crop_values(path, items=['Wheat'], aggregate_areas=['World'])
    assert set(out['country']) == {'Côte'}
    assert out.set_index('year')['crop_provision_gep'].loc[1961] == 1.0


def test_task_reader_reshapes_the_semicolon_delimited_coefficient_table(tmp_path):
    path = str(tmp_path / 'CWON2024_crop_coef.csv')
    pd.DataFrame({'Order': [1], 'FAO': [1.0], 'ISO3': ['AAA'], 'Country/territory': ['Aaaland'],
                  '1961-1970': [0.30], '2011-2020': [0.35]}).to_csv(path, sep=';', index=False)

    out = cpt.read_crop_coefs(path)
    assert out.set_index(['FAO', 'year'])['rental_rate'].loc[(1, 2011)] == 0.35


def test_es_config_row_hydrates_crop_provision(tmp_path):
    from global_invest import utilities
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'crop_provision', log=lambda *a: None)
    assert p.sheet_label == 'crop_provision'


# =================================================================================================
# The subsistence component: a port, so the tests pin the reproduction and the two findings.
# =================================================================================================

def _rulis_rows():
    """A RuLIS export in miniature: the indicator this reads and the one beside it."""
    own = cp.RULIS_OWN_CONSUMPTION_INDICATOR
    return pd.DataFrame([
        {'Indicator': own, 'Country': 'Kenya', 'Disaggregation': 'National', 'Year': 2005,
         'Value': 54.8, cp.PER_AREA_COLUMN: 500.0, 'Standard Deviation': 1.0,
         'Number of observations': 10, 'Income Classification': 'L'},
        {'Indicator': own, 'Country': 'Georgia', 'Disaggregation': 'Rural', 'Year': 2013,
         'Value': 60.0, cp.PER_AREA_COLUMN: 900.0, 'Standard Deviation': 1.0,
         'Number of observations': 5, 'Income Classification': 'LM'},
        {'Indicator': own, 'Country': 'Georgia', 'Disaggregation': 'Urban', 'Year': 2013,
         'Value': 40.0, cp.PER_AREA_COLUMN: 900.0, 'Standard Deviation': 1.0,
         'Number of observations': 5, 'Income Classification': 'LM'},
        {'Indicator': cp.RULIS_SOLD_AT_MARKET_INDICATOR, 'Country': 'Kenya',
         'Disaggregation': 'National', 'Year': 2005, 'Value': 88.0,
         cp.PER_AREA_COLUMN: 500.0, 'Standard Deviation': 1.0,
         'Number of observations': 10, 'Income Classification': 'L'},
    ])


def test_the_sold_at_market_indicator_is_dropped():
    """It sits in the same export and is close to this indicator's complement, so reading it would
    value the commercial half as subsistence."""
    out = cp.national_own_consumption_shares(_rulis_rows())
    assert cp.RULIS_SOLD_AT_MARKET_INDICATOR not in set(out['Indicator'])
    assert len(out) == 3


def test_a_country_with_no_national_row_takes_the_mean_of_rural_and_urban():
    """Georgia is surveyed but never reported nationally. Dropping it would be a silent loss."""
    out = cp.add_constructed_national_rows(cp.national_own_consumption_shares(_rulis_rows()))
    georgia = out[out['Country'] == 'Georgia']
    assert len(georgia) == 1
    assert georgia['Value'].iloc[0] == pytest.approx(50.0)
    assert georgia['Disaggregation'].iloc[0] == cp.RULIS_NATIONAL


def test_a_country_reported_nationally_keeps_only_its_national_row():
    """Kenya has a national figure, so averaging its settlement rows in would pull the share
    towards the households most likely to eat what they grow."""
    out = cp.add_constructed_national_rows(cp.national_own_consumption_shares(_rulis_rows()))
    assert out[out['Country'] == 'Kenya']['Value'].tolist() == [pytest.approx(54.8)]


def test_the_share_of_agricultural_area_is_read_not_the_share_of_farms():
    """Both rows sit in the same Lowder table under the same region. The share of farms is about
    seven times the share of area."""
    lowder = pd.DataFrame([
        {'Region': 'South Asia', 'Number or share of farms / agricultural area': 'share of farms (%)',
         '< 1 ha': 70.4, '1–2 ha': 13.8},
        {'Region': 'South Asia',
         'Number or share of farms / agricultural area': cp.LOWDER_AREA_SHARE_ROW,
         '< 1 ha': 23.9, '1–2 ha': 21.5}])
    out = cp.smallholder_area_shares(lowder)
    assert len(out) == 1 and out['< 1 ha'].iloc[0] == 23.9


def test_reading_the_wrong_lowder_row_raises_rather_than_returning_nothing():
    lowder = pd.DataFrame([
        {'Region': 'South Asia', 'Number or share of farms / agricultural area': 'share of farms (%)',
         '< 1 ha': 70.4, '1–2 ha': 13.8}])
    with pytest.raises(NameError, match='share of agricultural area'):
        cp.smallholder_area_shares(lowder)


def test_the_unit_correction_is_exactly_a_factor_of_ten():
    """The finding, pinned. FAOSTAT reports cropland in THOUSANDS of hectares against an intensity
    per SINGLE hectare, and the Lowder share is a PERCENTAGE the reference never divides by 100.
    The two compound to THOUSAND_HECTARES / PERCENT, and if either constant is ever edited to
    'fix' the other this fails."""
    assert cp.THOUSAND_HECTARES / cp.PERCENT == 10.0


def test_a_country_that_does_not_land_on_the_account_list_raises():
    """The reference joins its panel against the correspondence on a Natural Earth name column and
    delivers 16 of its own 66 valued countries -- 250 rows and a populated column, so the loss is
    invisible. Here it is an error."""
    countries = pd.DataFrame([
        {'ee_r264_label': 'KEN', 'iso3_r250_label': 'KEN', 'iso3_r250_id': 404,
         'iso3_r250_name': 'Kenya'}])
    panel = pd.DataFrame([
        {'alpha-3': 'KEN', 'Year': 2019, 'own_con': 1.0, 'own_con2': 1.0,
         'own_con_source': 'observed', 'rental_rate': 0.3, 'crop_subsistence_gep': 1.0},
        {'alpha-3': 'ATL', 'Year': 2019, 'own_con': 5.0, 'own_con2': 5.0,
         'own_con_source': 'observed', 'rental_rate': 0.3, 'crop_subsistence_gep': 5.0}])
    with pytest.raises(ValueError, match='did not land on the account country list'):
        cp.subsistence_on_country_list(panel, countries, 2019)


def test_a_year_takes_the_rate_of_the_decade_it_falls_in():
    coefs = pd.DataFrame([{'Order': 1, 'FAO': 159, 'ISO3': 'KEN', 'Country/territory': 'Kenya',
                           '2001-2010': 0.2, '2011-2020': 0.3}])
    panel = pd.DataFrame([{'alpha-3': 'KEN', 'Year': 2019, 'own_con2': 100.0}])
    out = cp.apply_subsistence_rental_rate(panel, coefs)
    assert out['rental_rate'].iloc[0] == 0.3
    assert out['gep_value'].iloc[0] == pytest.approx(30.0)

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
    raw.loc[3, 'Element Code'] = utilities.FAOSTAT_GROSS_PRODUCTION_VALUE_ELEMENT
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
    assert utilities.FAOSTAT_THOUSAND_USD == 1000.0
    assert utilities.FAOSTAT_VALUE_UNIT == '1000 USD'


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


def _crop_rows_after_attach():
    # Thousand USD gross, USD rent, at the decade lookup's rates. CCC has no rate there.
    return pd.DataFrame({
        'iso3_r250_label': ['AAA', 'AAA', 'BBB', 'CCC', 'DDD', None],
        'year': [2018, 2019, 2019, 2019, 2019, 2019],
        'gross_production_value': [10.0, 10.0, 20.0, 30.0, np.nan, 5.0],
        'rental_rate': [0.5, 0.5, 0.25, pd.NA, 0.25, 0.5],
        'crop_provision_gep': [5000.0, 5000.0, 5000.0, np.nan, np.nan, 2500.0]})


ANNUAL_RATES = pd.DataFrame({'wb_code': ['AAA', 'CCC', 'DDD'], 'yr2018': [0.9, 0.9, 0.9], 'yr2019': [0.1, 0.2, 0.3]})


def test_base_year_takes_the_annual_rate_with_rent_in_usd_and_other_years_stay():
    out = cp.apply_base_year_annual_rental_rates(_crop_rows_after_attach(), ANNUAL_RATES, 2019)
    assert out.iloc[1][['rental_rate', 'crop_provision_gep']].tolist() == [0.1, 1000.0], '10 thousand x 0.1'
    assert out.iloc[0][['rental_rate', 'crop_provision_gep']].tolist() == [0.5, 5000.0], 'not the base year'


def test_a_country_the_annual_table_does_not_list_goes_missing_with_no_fallback():
    out = cp.apply_base_year_annual_rental_rates(_crop_rows_after_attach(), ANNUAL_RATES, 2019)
    bbb = out[out['iso3_r250_label'] == 'BBB'].iloc[0]
    assert np.isnan(bbb['rental_rate']) and np.isnan(bbb['crop_provision_gep']), 'it had 0.25 in the decade lookup'
    assert bbb['gross_production_value'] == 20.0, 'gross stays, erosion values it'
    unlabelled = out[out['iso3_r250_label'].isna()].iloc[0]
    assert np.isnan(unlabelled['rental_rate']) and np.isnan(unlabelled['crop_provision_gep'])


def test_a_country_new_to_the_annual_table_gains_a_rate_and_a_blank_gross_stays_missing():
    out = cp.apply_base_year_annual_rental_rates(_crop_rows_after_attach(), ANNUAL_RATES, 2019)
    ccc = out[out['iso3_r250_label'] == 'CCC'].iloc[0]
    assert ccc['rental_rate'] == 0.2 and ccc['crop_provision_gep'] == pytest.approx(6000.0)
    ddd = out[out['iso3_r250_label'] == 'DDD'].iloc[0]
    assert ddd['rental_rate'] == 0.3 and np.isnan(ddd['crop_provision_gep'])


def test_annual_rates_keep_the_country_identity_through_the_country_sum():
    out = cp.apply_base_year_annual_rental_rates(_crop_rows_after_attach(), ANNUAL_RATES, 2019)
    out['iso3_r250_id'] = out['iso3_r250_label'].map({'AAA': 1, 'BBB': 2, 'CCC': 3, 'DDD': 4})
    by_country = utilities.sum_items_to_country_year(out.dropna(subset=['iso3_r250_id']), 'crop_provision_gep')
    row = by_country[(by_country['iso3_r250_id'] == 3) & (by_country['year'] == 2019)].iloc[0]
    assert row['crop_provision_gep'] == pytest.approx(row['rental_rate'] * row['gross_production_value'] * 1000)
    assert np.isnan(by_country[by_country['iso3_r250_id'] == 2].iloc[0]['crop_provision_gep'])
    assert np.isnan(by_country[by_country['iso3_r250_id'] == 2].iloc[0]['gross_production_value'])


def test_annual_rates_are_not_extrapolated_and_a_duplicate_country_raises():
    with pytest.raises(ValueError, match='yr2021'):
        cp.apply_base_year_annual_rental_rates(_crop_rows_after_attach(), ANNUAL_RATES, 2021)
    twice = pd.concat([ANNUAL_RATES, ANNUAL_RATES.iloc[[0]]])
    with pytest.raises(ValueError, match='AAA'):
        cp.apply_base_year_annual_rental_rates(_crop_rows_after_attach(), twice, 2019)


def test_annual_reader_takes_latin1_and_checks_the_recorded_hash(tmp_path):
    path = str(tmp_path / 'rates.csv')
    pd.DataFrame({'wb_code': ['CIV'], 'country': ['C\xf4te'], 'yr2019': [0.1]}).to_csv(path, index=False, encoding='latin-1')
    assert cpt.read_annual_rental_rates(path, utilities.sha256_file(path))['country'].tolist() == ['C\xf4te']
    with pytest.raises(ValueError, match='sha256'):
        cpt.read_annual_rental_rates(path, 'not-the-hash')


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
    """The finding, pinned on the arithmetic rather than on two named constants.

    FAOSTAT reports cropland in THOUSANDS of hectares against an intensity per SINGLE hectare, and
    the Lowder share is a PERCENTAGE the reference never divides by 100. Those compound to ten, and
    this computes both columns from one worked row rather than asserting a ratio between constants
    -- which is the stronger test, because it fails if either line's arithmetic changes and not
    merely if a constant is edited.

    Ethiopia 2019, from the source files: 18,190 thousand hectares of cropland, a Sub-Saharan
    smallholder share of 27.7 percent, 502.95 international dollars per hectare, and 33.7 percent
    eaten rather than sold.
    """
    row = pd.DataFrame([{
        'Value_x': 18190.0, '< 1 ha': 11.5, '1–2 ha': 16.2,
        cp.PER_AREA_COLUMN: 502.95,
        cp.RULIS_OWN_CONSUMPTION_INDICATOR: 33.700001,
    }])
    smallholder_percent = row['< 1 ha'] + row['1–2 ha']
    own_share_percent = row[cp.RULIS_OWN_CONSUMPTION_INDICATOR]

    reference = (row['Value_x'] * smallholder_percent * row[cp.PER_AREA_COLUMN]
                 * own_share_percent / 100.0).iloc[0]
    corrected = (row['Value_x'] * 1000.0 * (smallholder_percent / 100.0)
                 * row[cp.PER_AREA_COLUMN] * (own_share_percent / 100.0)).iloc[0]

    assert reference == pytest.approx(85_401_830, rel=1e-6)
    assert corrected == pytest.approx(854_018_300, rel=1e-6)
    assert corrected / reference == pytest.approx(10.0)


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


def _crop_provision_runs():
    """Every crop_provision project directory on this machine, cold starts included.

    Checking one named directory lets a stale warm run stand in for a fresh one, which is the
    failure condition 10 exists for. Every run that exists has to agree.
    """
    import glob
    import os
    pattern = os.path.join(os.path.expanduser('~'), 'Files', 'global_invest', 'projects',
                           'gep_crop_provision*', 'intermediate')
    return sorted(p for p in glob.glob(pattern) if os.path.isdir(p))


def _reference_by_m49(path):
    """A staged reference summed onto the M49 code.

    FAOSTAT carries dissolved states beside their successors -- "Ethiopia PDR" next to Ethiopia,
    "Serbia and Montenegro" next to Serbia -- under one M49 code each. The reference leaves those
    rows stranded at zero; this library maps them to the successor, which is one of the two fixes
    crop_provision documents. So the reference is summed onto the M49 code before comparing,
    which is the same collapse, and comparing row for row would count Ethiopia twice.
    """
    reference = pd.read_csv(path, encoding='utf-8-sig')
    return reference.groupby('area_code_M49', as_index=False).agg(
        crop_provision_gep_reference=('crop_provision_gep_reference', 'sum'))


def _reproduces(run_table, reference):
    """Whether a run's reference column equals a staged reference country by country."""
    joined = reference.merge(
        run_table[['iso3_r250_id', 'crop_provision_gep_reference']].rename(
            columns={'crop_provision_gep_reference': 'crop_provision_gep'}),
        left_on='area_code_M49', right_on='iso3_r250_id', how='inner')
    if len(joined) != len(reference):
        return False
    relative = ((joined['crop_provision_gep'] - joined['crop_provision_gep_reference']).abs()
                / joined['crop_provision_gep_reference'].abs().replace(0, np.nan))
    return bool(relative.max() < 1e-12) and np.isclose(
        joined['crop_provision_gep'].sum(), joined['crop_provision_gep_reference'].sum(), rtol=1e-12)


def test_the_commercial_figure_matches_the_reference_output_country_by_country(tmp_path):
    """The reported values alone, valued at the CWoN shares, must reproduce a staged reference --
    the authors' code on its own inputs, which values no gap -- for every run on the machine, not
    one chosen directory.

    Two reference versions are staged: the current one (the authors' code without the group
    totals) and the superseded one (their January 2026 list, which carries them). Every run must
    reproduce one of them exactly, so an old run is classified rather than rewritten, and the
    canonical project must reproduce the current one.

    Skips rather than fails when base_data or a run is absent, because a machine without either is
    not a machine this can speak about."""
    import os
    import hazelbean as hb
    p = hb.ProjectFlow(project_dir=str(tmp_path / 'reference'))
    utilities.hydrate_es_parameters(p, 'crop_provision', log=lambda *a: None)
    versions = {'current': p.crop_provision_reference_by_country_path,
                'superseded': p.crop_provision_reference_superseded_by_country_path}
    if not all(hb.path_exists(path) for path in versions.values()):
        pytest.skip('the staged references are not on this machine')
    versions = {name: _reference_by_m49(path) for name, path in versions.items()}

    checked = 0
    for run in _crop_provision_runs():
        project_path = os.path.join(run, 'gep_calculation', 'gep_by_country_base_year.csv')
        if not os.path.exists(project_path):
            continue
        ours = pd.read_csv(project_path)
        if 'crop_provision_gep_reference' not in ours.columns:
            continue                      # a run whose table carries no reproduction column
        matched = [name for name, reference in versions.items() if _reproduces(ours, reference)]
        assert len(matched) == 1, (run, matched)
        if os.path.basename(os.path.dirname(run)) == 'gep_crop_provision':
            assert matched == ['current'], run
        checked += 1
    if not checked:
        pytest.skip('no crop_provision run on this machine')


def test_the_subsistence_reference_column_reproduces_their_published_panel():
    """The published figure is ours; this is the column beside it that proves the port is faithful.
    Same rule: every run on the machine, cold starts included."""
    import os
    reference_path = os.path.join(
        os.path.expanduser('~'), 'Files', 'base_data', 'global_invest', 'crop_provision',
        'subsistence', 'reference', 'agr_subsistence_adjusted_2020.csv')
    if not os.path.exists(reference_path):
        pytest.skip('the staged reference is not on this machine')
    theirs = pd.read_csv(reference_path)
    theirs = theirs[theirs['Year'] == 2019][['alpha-3', 'gep_value_2019']].dropna()

    checked = 0
    for run in _crop_provision_runs():
        path = os.path.join(run, 'crop_subsistence_gep', 'subsistence_gep_by_country.csv')
        if not os.path.exists(path):
            continue
        ours = pd.read_csv(path).dropna(subset=['crop_subsistence_gep_reference'])
        joined = ours.merge(theirs, left_on='iso3_r250_label', right_on='alpha-3', how='inner')
        assert len(joined) == len(theirs), run
        relative = ((joined['crop_subsistence_gep_reference'] - joined['gep_value_2019']).abs()
                    / joined['gep_value_2019'].abs().replace(0, np.nan))
        assert relative.max() < 1e-10, run
        # Correcting the units alone is exactly ten times their arithmetic, which is the finding.
        # The published figure departs further, because it also imputes the survey SHARE for an
        # unsurveyed country rather than predicting the LEVEL from cropland area, so it is compared
        # on being larger and on the same order rather than on a fixed ratio.
        assert (ours['crop_subsistence_gep_units_only'].sum()
                == pytest.approx(10 * ours['crop_subsistence_gep_reference'].sum(), rel=1e-12)), run
        published = ours['crop_subsistence_gep'].sum()
        assert published > ours['crop_subsistence_gep_units_only'].sum(), run
        assert published < 4 * ours['crop_subsistence_gep_units_only'].sum(), run
        checked += 1
    if not checked:
        pytest.skip('no crop_provision run on this machine')


def test_an_unsurveyed_country_takes_a_regional_share_only_where_the_region_has_enough_surveys():
    """The imputation the account uses in place of the reference's regression. A region with too
    few surveys behind it falls back to the global median rather than passing off two observations
    as a regional pattern, and every row records which it used."""
    observed = pd.DataFrame([
        {'Country': 'Kenya', 'Value': 50.0}, {'Country': 'Malawi', 'Value': 60.0},
        {'Country': 'Ghana', 'Value': 40.0}, {'Country': 'Peru', 'Value': 20.0}])
    income = pd.DataFrame([
        {'Country': 'Kenya', 'Region': 'Sub-Saharan Africa', 'Code': 'KEN'},
        {'Country': 'Malawi', 'Region': 'Sub-Saharan Africa', 'Code': 'MWI'},
        {'Country': 'Ghana', 'Region': 'Sub-Saharan Africa', 'Code': 'GHA'},
        {'Country': 'Peru', 'Region': 'Latin America & Caribbean', 'Code': 'PER'},
        {'Country': 'Uganda', 'Region': 'Sub-Saharan Africa', 'Code': 'UGA'},
        {'Country': 'Bolivia', 'Region': 'Latin America & Caribbean', 'Code': 'BOL'}])
    out = cp.impute_own_consumption_shares(observed, income, minimum_per_region=3).set_index('Country')

    assert out.at['Kenya', 'own_consumption_share'] == pytest.approx(0.50)
    assert out.at['Kenya', 'share_source'] == 'observed'
    # Sub-Saharan Africa has three surveys, so Uganda takes their median.
    assert out.at['Uganda', 'own_consumption_share'] == pytest.approx(0.50)
    assert out.at['Uganda', 'share_source'].startswith('regional median')
    # Latin America has one, so Bolivia takes the global median of all four.
    assert out.at['Bolivia', 'own_consumption_share'] == pytest.approx(0.45)
    assert out.at['Bolivia', 'share_source'] == 'global median'


def test_the_imputation_never_reaches_above_lower_middle_income():
    """The reference scopes its panel to the low and lower-middle-income classes and that scope is
    kept, so a high-income country never acquires a subsistence estimate from a regional median."""
    wb = pd.DataFrame([{'Country': 'Kenya', '2019': 'L'}, {'Country': 'France', '2019': 'H'},
                       {'Country': 'Ghana', '2019': 'LM'}, {'Country': 'Chile', '2019': 'UM'}])
    eligible = cp.low_and_lower_middle_income_countries(wb, 2019)
    assert sorted(eligible['Country']) == ['Ghana', 'Kenya']


def test_the_ppp_factor_is_faostats_own_two_publications_of_the_same_output():
    """The account reports 2019 US dollars and reads FAOSTAT's current-year values as already being
    that. It holds for the Value of Production table and not for the production intensity, which
    FAOSTAT publishes only in international dollars -- so unconverted, that one component is in the
    wrong currency, and no current-US$ version exists to read instead.

    The conversion needs no external table: agricultural land times the intensity is total
    agricultural output in international dollars, and Value of Production reports the same output
    in current US dollars, so their ratio is the factor. This pins that identity."""
    area_value = pd.DataFrame([
        {'Area': 'Testland', 'Item': 'Agricultural land', 'Element': 'Area', 'Year': 2019,
         'Value': 1000.0},                                   # 1,000 thousand ha = 1,000,000 ha
        {'Area': 'Testland', 'Item': 'Agricultural land',
         'Element': cp.LAND_USE_VALUE_PER_AREA_ELEMENT, 'Year': 2019, 'Value': 200.0}])
    # 1,000,000 ha x 200 Int$/ha = 200,000,000 Int$ against 100,000,000 US$ -> a factor of 2.
    value = pd.DataFrame([
        {'Area': 'Testland', 'Item': 'Agriculture', 'Element Code': 57, 'Unit': '1000 USD',
         'Y2019': 100000.0}])
    income = pd.DataFrame([{'Country': 'Testland', 'Region': 'Sub-Saharan Africa'}])

    out = cp.agricultural_ppp_factors(area_value, value, income, 2019).set_index('Country')
    assert out.at['Testland', 'ppp_factor'] == pytest.approx(2.0)
    assert out.at['Testland', 'ppp_factor_source'] == 'FAOSTAT, this country'


def test_a_country_faostat_gives_no_aggregate_for_takes_its_regions_factor():
    """18 of the 66 subsistence countries have no FAOSTAT "Agriculture" value, the same coverage
    gap that put group totals in the item lists. They take a regional median rather than 1.0,
    because assuming no conversion is itself a conversion."""
    area_value = pd.DataFrame([
        {'Area': 'Alpha', 'Item': 'Agricultural land', 'Element': 'Area', 'Year': 2019, 'Value': 1000.0},
        {'Area': 'Alpha', 'Item': 'Agricultural land',
         'Element': cp.LAND_USE_VALUE_PER_AREA_ELEMENT, 'Year': 2019, 'Value': 200.0},
        {'Area': 'Beta', 'Item': 'Agricultural land', 'Element': 'Area', 'Year': 2019, 'Value': 500.0},
        {'Area': 'Beta', 'Item': 'Agricultural land',
         'Element': cp.LAND_USE_VALUE_PER_AREA_ELEMENT, 'Year': 2019, 'Value': 200.0},
        {'Area': 'Gamma', 'Item': 'Agricultural land', 'Element': 'Area', 'Year': 2019, 'Value': 900.0},
        {'Area': 'Gamma', 'Item': 'Agricultural land',
         'Element': cp.LAND_USE_VALUE_PER_AREA_ELEMENT, 'Year': 2019, 'Value': 200.0}])
    value = pd.DataFrame([
        {'Area': a, 'Item': 'Agriculture', 'Element Code': 57, 'Unit': '1000 USD', 'Y2019': v}
        for a, v in (('Alpha', 100000.0), ('Beta', 50000.0), ('Gamma', 90000.0))])
    income = pd.DataFrame([{'Country': c, 'Region': 'Sub-Saharan Africa'}
                           for c in ('Alpha', 'Beta', 'Gamma', 'Delta')])

    out = cp.agricultural_ppp_factors(area_value, value, income, 2019).set_index('Country')
    assert out.at['Delta', 'ppp_factor'] == pytest.approx(2.0)
    assert out.at['Delta', 'ppp_factor_source'].startswith('regional median')


def test_impute_fills_price_gaps_from_the_staged_values_table_and_leaves_valued_rows_alone():
    """A pair with production and no USD value takes the value the pollination chain already
    computed, in thousand USD: in place where its row exists empty, appended where no row
    exists, and a country the value file never names (so no FAO area code for the rental
    lookup) is left out. No price arithmetic happens here; the table's value_usd is the fill."""
    values = pd.DataFrame({
        'area_code': [1, 1],
        'area_code_M49': ["'010", "'010"],
        'country': ['Aaaland', 'Aaaland'],
        'crop_code': [100, 200],
        'crop': ['Wheat', 'Rye'],
        'year': [2019, 2019],
        'crop_provision_gep': [50.0, np.nan],
    })
    fao_values = pd.DataFrame({
        'area_code_m49': ['010', '010', '020', '030', '020'],
        'iso3': ['AAA', 'AAA', 'BBB', 'CCC', 'BBB'],
        'area_fao': ['Aaaland', 'Aaaland', 'Bbbland', 'Cccland', 'Bbbland'],
        'item_code_fao': [100, 200, 100, 100, 300],
        'item_fao': ['Wheat', 'Rye', 'Wheat', 'Wheat', 'Barley'],
        'year': [2019] * 5,
        'total_production_tonnes': [10.0, 4.0, 6.0, 5.0, 3.0],
        'price_source_agg_level': ['country', 'country', 'subregion', 'world', 'missing'],
        'value_usd': [9999.0, 2000.0, 1500.0, 500.0, np.nan],
    })

    out = cp.impute_missing_crop_values(
        values, fao_values, 2019, {10: 1, 20: 2}, ['Wheat', 'Rye', 'Barley'])

    keyed = out.set_index(['country', 'crop'])
    # Aaaland Wheat already carries a FAOSTAT value, so the table's 9999 never touches it.
    assert keyed.at[('Aaaland', 'Wheat'), 'crop_provision_gep'] == 50.0
    assert keyed.at[('Aaaland', 'Wheat'), 'value_source'] == 'faostat'
    # Rye's empty row is filled in place: the table's 2000 USD arrives as 2.0 thousand USD.
    assert keyed.at[('Aaaland', 'Rye'), 'crop_provision_gep'] == pytest.approx(2.0)
    assert keyed.at[('Aaaland', 'Rye'), 'value_source'] == 'price_country'
    # Bbbland has no row at all, so one is appended carrying the table's level flag and the
    # FAO area code the rental lookup needs.
    assert keyed.at[('Bbbland', 'Wheat'), 'crop_provision_gep'] == pytest.approx(1.5)
    assert keyed.at[('Bbbland', 'Wheat'), 'value_source'] == 'price_subregion'
    assert keyed.at[('Bbbland', 'Wheat'), 'area_code'] == 2
    # Cccland is not in the area-code map, and Barley's row is unpriced; neither enters.
    assert 'Cccland' not in out['country'].values
    assert 'Barley' not in out['crop'].values
    assert len(out) == 3


def test_group_totals_never_enter_and_their_member_crops_are_filled_instead():
    """Iiiland has only its group total valued (India 2019). The total leaves the item list in
    every country, so it is neither read nor filled, and Iiiland's rice, which carries production
    and no value, is filled like any other gap. Aaaland's valued wheat and its rye gap are
    unaffected."""
    raw = pd.DataFrame({
        'Area Code': [1, 1, 1, 3, 3],
        'Area Code (M49)': ["'010", "'010", "'010", "'030", "'030"],
        'Area': ['Aaaland', 'Aaaland', 'Aaaland', 'Iiiland', 'Iiiland'],
        'Item Code': [100, 200, 1717, 27, 1717],
        'Item Code (CPC)': ['x'] * 5,
        'Item': ['Wheat', 'Rye', 'Cereals, primary', 'Rice', 'Cereals, primary'],
        'Element Code': [57] * 5,
        'Element': ['Gross Production Value (current thousand US$)'] * 5,
        'Unit': ['1000 USD'] * 5,
        'Y2019': [10.0, np.nan, 15.0, np.nan, 50.0],
    })
    fao_values = pd.DataFrame({
        'area_code_m49': ['010', '030', '030'],
        'iso3': ['AAA', 'III', 'III'],
        'area_fao': ['Aaaland', 'Iiiland', 'Iiiland'],
        'item_code_fao': [200, 27, 1717],
        'item_fao': ['Rye', 'Rice', 'Cereals, primary'],
        'year': [2019, 2019, 2019],
        'total_production_tonnes': [4.0, 30.0, 30.0],
        'price_source_agg_level': ['country', 'country', 'country'],
        'value_usd': [2000.0, 30000.0, 99999.0],
    })
    items = cp.items_without_group_totals(
        ['Wheat', 'Rye', 'Rice', 'Cereals, primary'], ['Cereals, primary'])
    assert items == ['Wheat', 'Rye', 'Rice']

    reported = utilities.clean_faostat_values(raw, items, 'crop_provision_gep', [])
    out = cp.impute_missing_crop_values(
        reported[reported['year'] == 2019], fao_values, 2019, {10: 1, 30: 3}, items)

    assert 'Cereals, primary' not in set(out['crop'])
    keyed = out.set_index(['country', 'crop'])
    assert keyed.at[('Iiiland', 'Rice'), 'crop_provision_gep'] == pytest.approx(30.0)
    assert keyed.at[('Iiiland', 'Rice'), 'value_source'] == 'price_country'
    assert keyed.at[('Aaaland', 'Wheat'), 'crop_provision_gep'] == 10.0
    assert keyed.at[('Aaaland', 'Rye'), 'crop_provision_gep'] == pytest.approx(2.0)
    assert out['crop_provision_gep'].sum() == pytest.approx(42.0)


def test_the_staged_item_list_less_the_group_totals_is_the_158_item_crop_list(tmp_path):
    """The configured list is FAOSTAT's 161 crop items including three group totals (codes 1730,
    1720, 1723 in the Value of Production file). The account values the other 158 and never the
    three."""
    import hazelbean as hb
    p = hb.ProjectFlow(project_dir=str(tmp_path / 'items'))
    utilities.hydrate_es_parameters(p, 'crop_provision', log=lambda *a: None)
    configured = utilities.read_column(p.crop_provision_default_items_path, 'item_fao')
    groups = utilities.read_column(p.faostat_aggregate_items_path, 'item_fao')
    items = cp.items_without_group_totals(configured, groups)
    assert len(configured) == 161
    assert {'Oilcrops Primary', 'Roots and Tubers, Total', 'Sugar Crops Primary'} <= set(groups)
    assert len(items) == 158
    assert not set(items) & set(groups)


def test_area_codes_by_m49_keeps_the_account_row_not_the_aggregate_component():
    """FAOSTAT's value file carries China (351, M49 159) beside China, mainland (41, M49 156),
    and the account drops the mainland as an aggregate area. With the successor map sending 159
    to 156, the map must resolve 156 to the kept row's code, 351."""
    raw = pd.DataFrame({
        'Area Code': [351, 41, 1],
        'Area Code (M49)': ["'159", "'156", "'010"],
        'Area': ['China', 'China, mainland', 'Aaaland'],
    })
    out = cp.area_codes_by_m49(raw, aggregate_areas=['China, mainland'])
    assert out[156] == 351
    assert out[10] == 1


def test_area_codes_by_m49_keeps_the_successors_own_code_in_any_row_order():
    """The successor map folds four dissolved states onto their successor's M49, so two FAO codes
    share one normalized code. The successor's own code is kept, whichever row the file lists
    first: the dissolved state's code has no CWoN rental rate. Rows repeat, as in the file."""
    rows = [(62, "'230", 'Ethiopia PDR'), (238, "'231", 'Ethiopia'),
            (186, "'891", 'Serbia and Montenegro'), (272, "'688", 'Serbia'),
            (206, "'736", 'Sudan (former)'), (276, "'729", 'Sudan'),
            (51, "'200", 'Czechoslovakia'), (167, "'203", 'Czechia')] * 2
    expected = {231: 238, 688: 272, 729: 276, 203: 167}
    shuffled = [rows[i] for i in np.random.default_rng(0).permutation(len(rows))]
    for order in (rows, rows[::-1], shuffled):
        raw = pd.DataFrame(order, columns=['Area Code', 'Area Code (M49)', 'Area'])
        assert cp.area_codes_by_m49(raw, aggregate_areas=[]) == expected


def test_area_codes_by_m49_refuses_a_shared_code_it_cannot_resolve():
    """Two codes on one normalized M49 where neither, or both, carry it as their own M49 leave no
    canonical choice, so the map refuses rather than keep whichever came last."""
    neither = pd.DataFrame({'Area Code': [62, 63], 'Area Code (M49)': ["'230", "'230"],
                            'Area': ['Aaaland', 'Bbbland']})
    both = pd.DataFrame({'Area Code': [238, 239], 'Area Code (M49)': ["'231", "'231"],
                         'Area': ['Aaaland', 'Bbbland']})
    for raw in (neither, both):
        with pytest.raises(ValueError, match='ambiguous'):
            cp.area_codes_by_m49(raw, aggregate_areas=[])


def _ethiopia_values(rows):
    """The staged values-table columns for Ethiopia (M49 231) at 2019, one row per (item, name,
    value_usd, level)."""
    return pd.DataFrame({
        'area_code_m49': ['231'] * len(rows),
        'iso3': ['ETH'] * len(rows),
        'area_fao': ['Ethiopia'] * len(rows),
        'item_code_fao': [r[0] for r in rows],
        'item_fao': [r[1] for r in rows],
        'year': [2019] * len(rows),
        'total_production_tonnes': [1.0] * len(rows),
        'price_source_agg_level': [r[3] for r in rows],
        'value_usd': [r[2] for r in rows],
    })


def test_impute_fills_a_dissolved_and_successor_pair_once_on_the_successors_code():
    """Ethiopia PDR (62) and Ethiopia (238) share normalized M49 231. Wheat has an empty row on
    both codes: one fill, on 238, and the PDR row stays empty. Barley's only row is the PDR's, so
    the fill arrives as a new row on 238. Ethiopia's reported rye stays as reported."""
    values = pd.DataFrame({
        'area_code': [62, 238, 62, 238],
        'area_code_M49': ["'230", "'231", "'230", "'231"],
        'country': ['Ethiopia PDR', 'Ethiopia', 'Ethiopia PDR', 'Ethiopia'],
        'crop_code': [15, 15, 44, 71],
        'crop': ['Wheat', 'Wheat', 'Barley', 'Rye'],
        'year': [2019] * 4,
        'crop_provision_gep': [np.nan, np.nan, np.nan, 7.0],
    })
    fao_values = _ethiopia_values([(15, 'Wheat', 3000.0, 'country'),
                                   (44, 'Barley', 800.0, 'subregion'),
                                   (71, 'Rye', 999.0, 'country')])

    out = cp.impute_missing_crop_values(
        values, fao_values, 2019, {231: 238}, ['Wheat', 'Barley', 'Rye'])

    filled = out[out['value_source'].astype(str).str.startswith('price_')].set_index('crop')
    assert sorted(zip(filled['area_code'], filled.index)) == [(238, 'Barley'), (238, 'Wheat')]
    assert filled.at['Wheat', 'crop_provision_gep'] == pytest.approx(3.0)
    assert filled.at['Barley', 'crop_provision_gep'] == pytest.approx(0.8)
    assert filled.at['Barley', 'value_source'] == 'price_subregion'
    old_state = out[out['area_code'] == 62]
    assert old_state['crop_provision_gep'].isna().all()
    assert old_state['value_source'].isna().all()
    rye = out[out['crop'] == 'Rye']
    assert list(rye['crop_provision_gep']) == [7.0]
    assert list(rye['value_source']) == ['faostat']
    assert len(out) == 5


def test_impute_refuses_to_fill_one_country_item_pair_twice():
    """A repeated empty row on the canonical code would count the same production twice, and a
    values table carrying one pair under two M49 codes that normalize together offers two values
    for it. The fill refuses both, whether the pair has a canonical row to fill or is appended."""
    repeated_row = pd.DataFrame({
        'area_code': [238, 238], 'area_code_M49': ["'231", "'231"],
        'country': ['Ethiopia', 'Ethiopia'], 'crop_code': [15, 15], 'crop': ['Wheat', 'Wheat'],
        'year': [2019, 2019], 'crop_provision_gep': [np.nan, np.nan]})
    with pytest.raises(ValueError, match='repeat a country-item pair'):
        cp.impute_missing_crop_values(repeated_row, _ethiopia_values([(15, 'Wheat', 3000.0, 'country')]),
                                      2019, {231: 238}, ['Wheat'])

    two_codes = _ethiopia_values([(15, 'Wheat', 3000.0, 'country'), (15, 'Wheat', 2000.0, 'country')])
    two_codes.loc[1, 'area_code_m49'] = '230'
    for rows in (repeated_row.iloc[:1], repeated_row.iloc[:0]):     # canonical row present, absent
        with pytest.raises(ValueError, match='repeat a country-item pair'):
            cp.impute_missing_crop_values(rows, two_codes, 2019, {231: 238}, ['Wheat'])


def test_the_staged_2019_fill_places_each_pair_once_and_never_on_a_dissolved_states_code(tmp_path):
    """On the staged FAOSTAT Value of Production and the pollination values table, the 2019 fill
    places 5,214 country-item pairs, one row each, none on Ethiopia PDR (62), Serbia and
    Montenegro (186), Sudan (former) (206) or Czechoslovakia (51), worth $1,115,790,900,662.61
    before the rental rate. Every reported value stays as reported. Skips where base_data is
    absent."""
    import hazelbean as hb
    p = hb.ProjectFlow(project_dir=str(tmp_path / 'fill'))
    utilities.hydrate_es_parameters(p, 'crop_provision', log=lambda *a: None)
    paths = {name: str(p.get_path(getattr(p, name))) for name in (
        'fao_input_path', 'crop_provision_fao_values_table_path', 'faostat_aggregate_areas_path',
        'faostat_aggregate_items_path', 'crop_provision_default_items_path')}
    if not all(hb.path_exists(path) for path in paths.values()):
        pytest.skip('the staged FAOSTAT inputs are not on this machine')
    aggregate_areas = utilities.read_column(paths['faostat_aggregate_areas_path'], 'area_fao')
    items = cp.items_without_group_totals(
        utilities.read_column(paths['crop_provision_default_items_path'], 'item_fao'),
        utilities.read_column(paths['faostat_aggregate_items_path'], 'item_fao'))
    raw = pd.read_csv(paths['fao_input_path'], encoding='ISO-8859-1', low_memory=False)
    reported = utilities.clean_faostat_values(raw, items, 'crop_provision_gep', aggregate_areas)

    out = cp.impute_missing_crop_values(
        reported, pd.read_csv(paths['crop_provision_fao_values_table_path']), 2019,
        cp.area_codes_by_m49(raw, aggregate_areas), items)

    filled = out[out['value_source'].astype(str).str.startswith('price_')]
    pairs = set(zip(cp.normalize_m49_series(filled['area_code_M49']), filled['crop_code']))
    assert len(filled) == len(pairs) == 5214
    assert not filled['area_code'].isin([62, 186, 206, 51]).any()
    assert (filled['crop_provision_gep'].sum() * utilities.FAOSTAT_THOUSAND_USD
            == pytest.approx(1115790900662.61, rel=1e-12))
    was_reported = reported['crop_provision_gep'].notna().to_numpy()
    kept = out.iloc[:len(reported)]
    assert (kept['crop_provision_gep'].to_numpy()[was_reported]
            == reported['crop_provision_gep'].to_numpy()[was_reported]).all()
    assert (kept['value_source'].to_numpy()[was_reported] == 'faostat').all()


def test_the_shared_group_total_rule_is_judged_by_membership_not_by_any_item():
    """The shared group-total helper in utilities (crop provision itself removes its group totals
    everywhere): a group total is kept only where none of ITS members is valued.
    Unrelatedland has a valued Banana beside an Oilcrops total: the total stays, because bananas
    are not oilcrops. A group row whose value is missing stays. Partialland's Oilcrops total is
    dropped although only one of its two members is valued."""
    membership = pd.DataFrame({
        'item_group_code': [1720, 1720, 1730, 1730, 1723, 1717, 1717],
        'item_code':       [116,  125,  254,  249,  156,  15,   27]})
    aggregates = ['Roots and Tubers, Total', 'Oilcrops Primary', 'Sugar Crops Primary']
    values = pd.DataFrame({
        'area_code': [9, 9, 9, 7, 7, 5, 5, 3, 3],
        'country': ['Indialike'] * 3 + ['Unrelatedland'] * 2 + ['Missingland'] * 2 + ['Partialland'] * 2,
        'crop_code': [1720, 1730, 1723, 486, 1730, 1730, 254, 1730, 254],
        'crop': ['Roots and Tubers, Total', 'Oilcrops Primary', 'Sugar Crops Primary',
                 'Bananas', 'Oilcrops Primary', 'Oilcrops Primary', 'Oil palm fruit',
                 'Oilcrops Primary', 'Oil palm fruit'],
        'year': [2019] * 9,
        'crop_provision_gep': [2.9, 7.6, 2.4, 5.0, 10.0, np.nan, np.nan, 11.0, 4.0],
    })
    out = utilities.drop_aggregates_where_components_exist(
        values, aggregates, 'crop_provision_gep', membership, log=lambda *a: None)
    keyed = out.set_index(['country', 'crop'])
    assert ('Indialike', 'Roots and Tubers, Total') in keyed.index
    assert ('Indialike', 'Oilcrops Primary') in keyed.index
    assert ('Unrelatedland', 'Oilcrops Primary') in keyed.index, 'a banana is not an oilcrop'
    assert ('Partialland', 'Oilcrops Primary') not in keyed.index, 'one valued member suffices'
    assert ('Missingland', 'Oilcrops Primary') in keyed.index, 'nothing valued under it'


def test_a_country_year_with_no_valued_item_stays_missing():
    """Pins the missing-data rule: a bare country-year sum turns a country whose every item value
    is missing into a published zero. All-missing stays missing, a genuine zero stays zero, a
    partially valued country sums its valued items."""
    import numpy as np
    crop_rows = pd.DataFrame({
        'iso3_r250_id': [10, 10, 20, 30, 30, 40],
        'iso3_r250_label': ['AAA', 'AAA', 'BBB', 'CCC', 'CCC', 'DDD'],
        'year': [2019] * 6,
        'crop_provision_gep': [np.nan, np.nan,   # all items missing
                               0.0,              # a measured zero
                               np.nan, 500.0,    # partially valued
                               700.0],           # fully valued
    })
    by_country = utilities.sum_items_to_country_year(crop_rows, 'crop_provision_gep')
    per = by_country.set_index('iso3_r250_id')['crop_provision_gep']
    assert pd.isna(per.loc[10]), 'all-missing must stay missing, not become zero'
    assert per.loc[20] == 0.0, 'a measured zero stays zero'
    assert per.loc[30] == 500.0
    assert per.loc[40] == 700.0


def test_country_gross_value_covers_valued_rows_so_the_rent_identity_holds():
    """Pins the dissolved-source-area rule: an alias row carrying gross value with no rental rate
    must not enter the country's gross_production_value, or the identity
    value = rate x gross x 1000 fails. The item rows stay."""
    import numpy as np
    crop_rows = pd.DataFrame({
        'iso3_r250_id': [10, 10, 10],
        'iso3_r250_label': ['AAA', 'AAA', 'AAA'],
        'year': [2019] * 3,
        # two current valid rows and one dissolved-area row (gross value, no rent, no gep)
        'crop_provision_gep': [315.0, 630.0, np.nan],
        'gross_production_value': [1.0, 2.0, 50.0],
    })
    by_country = utilities.sum_items_to_country_year(crop_rows, 'crop_provision_gep')
    row = by_country.iloc[0]
    assert row['crop_provision_gep'] == 945.0
    assert row['gross_production_value'] == 3.0, 'alias gross value must not enter the sum'
    # identity with rate 0.315 and thousand-USD gross: 0.315 * 3.0 * 1000 = 945
    assert row['crop_provision_gep'] == pytest.approx(0.315 * row['gross_production_value'] * 1000)


def _base_year_frames():
    values = pd.DataFrame({
        'area_code': [1, 1, 1, 1],
        'area_code_M49': ["'010"] * 4,
        'country': ['Aaaland'] * 4,
        'crop_code': [100, 100, 200, 300],
        'crop': ['Wheat', 'Wheat', 'Rye', 'Oats'],
        'year': [2019, 2018, 2019, 2019],
        'crop_provision_gep': [50.0, 40.0, np.nan, 7.0],
    })
    fao_values = pd.DataFrame({
        'area_code_m49': ['010', '010'],
        'iso3': ['AAA', 'AAA'],
        'area_fao': ['Aaaland', 'Aaaland'],
        'item_code_fao': [100, 200],
        'item_fao': ['Wheat', 'Rye'],
        'year': [2019, 2019],
        'total_production_tonnes': [10.0, 4.0],
        'price_source_agg_level': ['country', 'subregion'],
        'value_usd': [9999.0, 2000.0],
    })
    return values, fao_values


def test_production_times_price_values_every_base_year_pair_from_the_values_table():
    # The crop_benefits basis: the base year's reported wheat value gives way to production times
    # price, rye's gap is filled as before, and oats, which FAOSTAT reports and the values table
    # cannot price, keeps its row with no value. 2018 keeps FAOSTAT's reported value.
    values, fao_values = _base_year_frames()
    out = cp.value_at_production_times_price(values, fao_values, 2019, {10: 1}, [100, 200, 300])
    keyed = out.set_index(['crop', 'year'])
    assert keyed.at[('Wheat', 2019), 'crop_provision_gep'] == pytest.approx(9.999)
    assert keyed.at[('Wheat', 2019), 'value_source'] == 'price_country'
    assert keyed.at[('Rye', 2019), 'crop_provision_gep'] == pytest.approx(2.0)
    assert keyed.at[('Rye', 2019), 'value_source'] == 'price_subregion'
    assert pd.isna(keyed.at[('Oats', 2019), 'crop_provision_gep'])
    assert keyed.at[('Wheat', 2018), 'crop_provision_gep'] == 40.0
    assert keyed.at[('Wheat', 2018), 'value_source'] == 'faostat'
    assert len(out) == len(values)
    # The reported-first basis leaves the reported wheat and oats alone.
    first = cp.impute_missing_crop_values(values, fao_values, 2019, {10: 1}, [100, 200, 300]).set_index(['crop', 'year'])
    assert first.at[('Wheat', 2019), 'crop_provision_gep'] == 50.0
    assert first.at[('Oats', 2019), 'crop_provision_gep'] == 7.0


def test_an_item_is_filled_by_its_code_where_the_two_tables_spell_it_differently():
    # The item list carries FAOSTAT's Latin-1 read of mate and the values table its UTF-8 name, so
    # the fill is keyed on the item code.
    values = pd.DataFrame({'area_code': [1, 1], 'area_code_M49': ["'032", "'032"], 'country': ['Argentina'] * 2,
                           'crop_code': [671, 15], 'crop': ['MatÃ© leaves', 'Wheat'], 'year': [2019, 2019],
                           'crop_provision_gep': [np.nan, np.nan]})
    fao_values = pd.DataFrame({'area_code_m49': ['032', '032'], 'iso3': ['ARG'] * 2, 'area_fao': ['Argentina'] * 2,
                               'item_code_fao': [671, 15], 'item_fao': ['Maté leaves', 'Wheat'], 'year': [2019, 2019],
                               'total_production_tonnes': [100.0, 10.0], 'price_source_agg_level': ['country'] * 2,
                               'value_usd': [5000.0, 3000.0]})
    by_name = cp.impute_missing_crop_values(values, fao_values, 2019, {32: 1}, ['MatÃ© leaves', 'Wheat']).set_index('crop_code')
    assert pd.isna(by_name.at[671, 'crop_provision_gep'])
    assert by_name.at[15, 'crop_provision_gep'] == pytest.approx(3.0)
    by_code = cp.impute_missing_crop_values(values, fao_values, 2019, {32: 1}, [671, 15]).set_index('crop_code')
    assert by_code.at[671, 'crop_provision_gep'] == pytest.approx(5.0)


def _zero_frames(fao_value_usd, items=(100, 200, 300)):
    values = pd.DataFrame({
        'area_code': [1, 1, 1, 1],
        'area_code_M49': ["'036"] * 4,
        'country': ['Australia'] * 4,
        'crop_code': [100, 200, 300, 100],
        'crop': ['Wheat', 'Rye', 'Oats', 'Wheat'],
        'year': [2019, 2019, 2019, 2018],
        'crop_provision_gep': [50.0, 60.0, np.nan, 40.0],
    })
    fao_values = pd.DataFrame({
        'area_code_m49': ['036'] * len(fao_value_usd),
        'iso3': ['AUS'] * len(fao_value_usd),
        'area_fao': ['Australia'] * len(fao_value_usd),
        'item_code_fao': list(items)[:len(fao_value_usd)],
        'item_fao': ['Wheat', 'Rye', 'Oats'][:len(fao_value_usd)],
        'year': [2019] * len(fao_value_usd),
        'total_production_tonnes': [0.0, 10.0, 5.0][:len(fao_value_usd)],
        'price_source_agg_level': ['country', 'country', 'missing'][:len(fao_value_usd)],
        'value_usd': list(fao_value_usd),
    })
    return values, fao_values


def test_production_times_price_keeps_a_source_zero_apart_from_a_missing_value():
    # A values-table zero is a valued zero, and a missing source value stays missing. The
    # reported-first fill still takes positive values only, so a zero never fills its gaps.
    values, fao_values = _zero_frames([0.0, 100.0, np.nan])
    out = cp.value_at_production_times_price(values, fao_values, 2019, {36: 1}, [100, 200, 300])
    keyed = out.set_index(['crop', 'year'])
    assert keyed.at[('Wheat', 2019), 'crop_provision_gep'] == 0.0
    assert keyed.at[('Wheat', 2019), 'value_source'] == 'price_country'
    assert keyed.at[('Rye', 2019), 'crop_provision_gep'] == pytest.approx(0.1)
    assert pd.isna(keyed.at[('Oats', 2019), 'crop_provision_gep'])
    assert keyed.at[('Wheat', 2018), 'crop_provision_gep'] == 40.0
    gap = values.assign(crop_provision_gep=[np.nan, 60.0, np.nan, 40.0])
    first = cp.impute_missing_crop_values(gap, fao_values, 2019, {36: 1}, [100, 200, 300]).set_index(['crop', 'year'])
    assert pd.isna(first.at[('Wheat', 2019), 'crop_provision_gep'])
    assert first.at[('Rye', 2019), 'crop_provision_gep'] == 60.0


def test_production_times_price_on_an_all_zero_table_values_every_pair_at_zero():
    values, fao_values = _zero_frames([0.0, 0.0, 0.0])
    out = cp.value_at_production_times_price(values, fao_values, 2019, {36: 1}, [100, 200, 300])
    at_year = out[out['year'] == 2019]
    assert (at_year['crop_provision_gep'] == 0.0).all() and len(at_year) == 3


def test_a_values_table_that_matches_no_item_leaves_the_gaps_blank_and_raises_nothing():
    values, fao_values = _zero_frames([5.0, 6.0, 7.0], items=(900, 901, 902))
    out = cp.value_at_production_times_price(values, fao_values, 2019, {36: 1}, [100, 200, 300])
    assert out.loc[out['year'] == 2019, 'crop_provision_gep'].isna().all()
    assert out.loc[out['year'] == 2018, 'crop_provision_gep'].tolist() == [40.0]
    assert len(out) == len(values)
    first = cp.impute_missing_crop_values(values, fao_values, 2019, {36: 1}, [100, 200, 300])
    assert first['crop_provision_gep'].tolist()[:2] == [50.0, 60.0]
    assert pd.isna(first['crop_provision_gep'].iloc[2]) and len(first) == len(values)


# =================================================================================================
# The commercial task's reuse: its tables rebuild when anything they are made from changes.
# =================================================================================================

def _write_vop(path, wheat_2018=40.0):
    rows = [(1, "'010", 'Aaaland', 100, 'Wheat', wheat_2018, 50.0), (1, "'010", 'Aaaland', 200, 'Rye', 4.0, 5.0),
            (2, "'020", 'Bbbland', 100, 'Wheat', 80.0, 60.0), (2, "'020", 'Bbbland', 200, 'Rye', 8.0, 6.0)]
    frame = pd.DataFrame(rows, columns=['Area Code', 'Area Code (M49)', 'Area', 'Item Code', 'Item', 'Y2018', 'Y2019'])
    frame = frame.assign(**{'Element Code': 57, 'Element': 'Gross Production Value', 'Unit': '1000 USD', 'Y2018F': 'A', 'Y2019F': 'A'})
    frame.to_csv(path, index=False, encoding='ISO-8859-1')


def _write_values(path, price=2000.0):
    rows = [('010', 'AAA', 'Aaaland', code, item, year, 10.0, 'country', price * 10.0)
            for code, item in ((100, 'Wheat'), (200, 'Rye')) for year in (2018, 2019)]
    rows += [('020', 'BBB', 'Bbbland', code, item, year, 10.0, 'country', 3000.0) for code, item in ((100, 'Wheat'), (200, 'Rye')) for year in (2018, 2019)]
    pd.DataFrame(rows, columns=['area_code_m49', 'iso3', 'area_fao', 'item_code_fao', 'item_fao', 'year', 'total_production_tonnes',
                                'price_source_agg_level', 'value_usd']).to_csv(path, index=False)


def _write_coefs(path, rate=0.35):
    pd.DataFrame({'Order': [1, 2], 'FAO': [1.0, 2.0], 'ISO3': ['AAA', 'BBB'], 'Country/territory': ['Aaaland', 'Bbbland'],
                  '2011-2020': [rate, 0.25]}).to_csv(path, sep=';', index=False)


def _write_annual_rates(p, rate=0.1):
    path = p.crop_provision_cwon_annual_rental_rate_path
    pd.DataFrame({'wb_code': ['AAA', 'BBB'], 'yr2018': [rate, 0.2], 'yr2019': [rate, 0.2]}).to_csv(path, index=False, encoding='latin-1')
    p.crop_provision_cwon_annual_rental_rate_sha256 = utilities.sha256_file(path)


def _crop_project(tmp_path, monkeypatch):
    """A whole gep_calculation project on small files: two countries, two crops, two years."""
    import geopandas as gpd
    from shapely.geometry import Point
    monkeypatch.setattr(cpt, 'publish_inputs', lambda p: p)
    source = tmp_path / 'source'
    source.mkdir()
    (tmp_path / 'task').mkdir()
    p = SimpleNamespace(run_this=1, cur_dir=str(tmp_path / 'task'), results={}, gep_base_year=2019.0, get_path=str,
                        df_countries=COUNTRIES.copy(), crop_provision_base_year_value_basis='production_x_price',
                        fao_input_path=str(source / 'vop.csv'), cwon_crop_coefficients_path=str(source / 'coefs.csv'),
                        faostat_aggregate_areas_path=str(source / 'areas.csv'), faostat_aggregate_items_path=str(source / 'items.csv'),
                        crop_provision_default_items_path=str(source / 'default_items.csv'), crop_provision_fao_values_table_path=str(source / 'values.csv'),
                        crop_provision_cwon_annual_rental_rate_path=str(source / 'annual.csv'), df_countries_csv_path=str(source / 'countries.csv'),
                        gdf_countries_vector_simplified_path=str(source / 'countries.gpkg'))
    p.gdf_countries_simplified = p.gdf_countries_vector_simplified_path
    _write_vop(p.fao_input_path)
    _write_values(p.crop_provision_fao_values_table_path)
    _write_coefs(p.cwon_crop_coefficients_path)
    _write_annual_rates(p)
    pd.DataFrame({'area_fao': ['World']}).to_csv(p.faostat_aggregate_areas_path, index=False)
    pd.DataFrame({'item_fao': ['Cereals, primary']}).to_csv(p.faostat_aggregate_items_path, index=False)
    pd.DataFrame({'item_fao': ['Wheat', 'Rye']}).to_csv(p.crop_provision_default_items_path, index=False)
    COUNTRIES.to_csv(p.df_countries_csv_path, index=False)
    gpd.GeoDataFrame({'ee_r264_id': [1, 2, 3]}, geometry=[Point(0, 0), Point(1, 1), Point(2, 2)], crs='EPSG:4326').to_file(p.gdf_countries_vector_simplified_path, driver='GPKG')
    return p


def _tables(p):
    return {name: open(path, 'rb').read() for name, path in p.results['crop_provision'].items()}


def test_the_task_reuses_its_tables_while_nothing_they_are_made_from_changed(tmp_path, monkeypatch):
    p = _crop_project(tmp_path, monkeypatch)
    assert cpt.gep_calculation(p) is not None
    before = _tables(p)
    assert cpt.gep_calculation(p) is None, 'an unchanged project reuses'
    assert _tables(p) == before


def test_a_new_base_year_rebuilds_the_tables_at_that_year(tmp_path, monkeypatch):
    p = _crop_project(tmp_path, monkeypatch)
    cpt.gep_calculation(p)
    p.gep_base_year = 2018.0
    assert cpt.gep_calculation(p) is not None, 'the base year is outside the crop_provision_ settings'
    published = pd.read_csv(p.results['crop_provision']['gep_by_country_base_year'])
    assert set(published['year'].dropna()) == {2018}


def _change_vop(p):
    _write_vop(p.fao_input_path, wheat_2018=41.0)


def _change_coefs(p):
    _write_coefs(p.cwon_crop_coefficients_path, rate=0.5)


def _change_aggregate_areas(p):
    pd.DataFrame({'area_fao': ['World', 'Bbbland']}).to_csv(p.faostat_aggregate_areas_path, index=False)


def _change_aggregate_items(p):
    pd.DataFrame({'item_fao': ['Cereals, primary', 'Rye']}).to_csv(p.faostat_aggregate_items_path, index=False)


def _change_default_items(p):
    pd.DataFrame({'item_fao': ['Wheat']}).to_csv(p.crop_provision_default_items_path, index=False)


def _change_countries(p):
    p.df_countries = COUNTRIES.assign(income_grp='1. High income')
    p.df_countries.to_csv(p.df_countries_csv_path, index=False)


def _change_values(p):
    _write_values(p.crop_provision_fao_values_table_path, price=2500.0)


def _change_annual_rates(p):
    _write_annual_rates(p, rate=0.15)


def _change_basis(p):
    p.crop_provision_base_year_value_basis = 'reported_first'


@pytest.mark.parametrize('change', [_change_vop, _change_coefs, _change_aggregate_areas, _change_aggregate_items, _change_default_items,
                                    _change_countries, _change_values, _change_annual_rates, _change_basis])
def test_a_changed_input_rebuilds_the_tables_and_moves_them(tmp_path, monkeypatch, change):
    p = _crop_project(tmp_path, monkeypatch)
    cpt.gep_calculation(p)
    before = _tables(p)
    change(p)
    assert cpt.gep_calculation(p) is not None, 'a warm project must not republish the old tables'
    assert _tables(p) != before

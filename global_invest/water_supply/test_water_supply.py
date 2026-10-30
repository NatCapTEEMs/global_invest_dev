"""Unit tests for the water_supply hydropower component (CWoN resource-rent method) and the
water-use calculation.

The hydropower checks pin the annuity identity (the observed constant 0.040808 against
1/annuity(4%, 100y)), the authors' revenue-minus-cost rent with its sign and status, and, where
the private valuation workbook is staged, that rent against the authors' output country by
country. The water-use tests pin the two-script calculation against its two committed
intermediates (cleaning bit-exact, products to the anchor's own CSV rounding) and check the
joins and totals of the committed per-country anchors.
"""
import hashlib
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from global_invest import utilities
from global_invest.water_supply import water_supply_functions as wf


# CWoN's capitalization rate. The run reads it from es_parameters
# (water_supply_hydropower_capitalization_rate), so the tests name it here.
CWON_CAPITALIZATION_RATE = 0.04


def _base_data_project():
    """A bare ProjectFlow, only for its base_data_dir.

    The anchors are inputs, so a test finds them the way a run does rather than by walking
    directories of its own.
    """
    import tempfile
    import hazelbean as hb
    return hb.ProjectFlow(project_dir=os.path.join(tempfile.mkdtemp(), 'anchors'))


REFERENCE_DIR = utilities.service_data_dir(_base_data_project(), 'water_supply')


def test_annuity_identity_matches_the_observed_anchor_ratio():
    factor = wf.annuity_factor(0.04)   # CWoN's capitalization rate, an es_parameters row
    assert np.isclose(factor, 24.504998997, rtol=1e-9)      # sum of 1/1.04^t, t=1..100
    assert np.isclose(1.0 / factor, 0.040808, atol=1e-6)    # the ratio observed in the anchor


def _workbook_path():
    """The private valuation workbook where this machine stages it, or None."""
    params = pd.read_csv(os.path.join(os.path.dirname(wf.__file__), '..', 'input_template',
                                      'es_parameters.csv'), encoding='utf-8-sig')
    value = params.loc[params['parameter'] == 'water_supply_cwon_hydro_workbook_path', 'value'].iloc[0]
    path = os.path.join(_base_data_project().base_data_dir, value)
    return path if os.path.exists(path) else None


def _database(rows):
    """A Database sheet in the workbook's layout: revenues in column W, total costs in Y."""
    columns = ['col%d' % i for i in range(31)]
    columns[0], columns[3], columns[6] = 'Country', 'Year', 'Renewable energy asset'
    columns[22], columns[24] = wf.HYDROPOWER_WORKBOOK_REVENUE, wf.HYDROPOWER_WORKBOOK_COST
    df = pd.DataFrame(np.nan, index=range(len(rows)), columns=columns, dtype=object)
    for i, (country, year, asset, revenue, cost) in enumerate(rows):
        df.loc[i, ['Country', 'Year', 'Renewable energy asset',
                   wf.HYDROPOWER_WORKBOOK_REVENUE, wf.HYDROPOWER_WORKBOOK_COST]] = [
            country, year, asset, revenue, cost]
    return df


def test_the_authors_rent_is_revenue_minus_cost_with_its_sign_and_status():
    """Revenues minus total costs, sign kept. A negative rent stays negative with status
    'negative', is blank in the author-output comparison variant, and leaves the wealth-based
    value the account reports untouched. The workbook's newer country names map through the
    wealth table's older ones."""
    wealth = pd.DataFrame({'countrycode': ['AAA', 'BBB', 'TUR', 'CCC'],
                           'countryname': ['Aland', 'Bland', 'Turkey', 'Cland'],
                           'YR2019': [245.05, 490.10, 100.0, 50.0]})
    database = _database([('Aland', 2019, 'Hydropower', 300.0, 100.0),
                          ('Bland', 2019, 'Hydropower', 100.0, 250.0),
                          ('Turkiye', 2019, 'Hydropower', 'n/a', 10.0),
                          ('Aland', 2018, 'Hydropower', 1.0, 0.0),
                          ('Aland', 2019, 'Solar', 9.0, 0.0)])
    out = wf.hydropower_rent_table(wealth, CWON_CAPITALIZATION_RATE, database).set_index('iso3_r250_label')

    assert out.loc['AAA', 'hydropower_rent_signed_usd'] == 200.0
    assert out.loc['AAA', 'hydropower_rent_status'] == 'positive'
    assert out.loc['AAA', 'hydropower_gep_reference_variant'] == 200.0
    assert out.loc['BBB', 'hydropower_rent_signed_usd'] == -150.0
    assert out.loc['BBB', 'hydropower_rent_status'] == 'negative'
    assert np.isnan(out.loc['BBB', 'hydropower_gep_reference_variant'])
    assert np.isclose(out.loc['BBB', 'hydropower_gep'], 490.10 / wf.annuity_factor(0.04))
    assert out.loc['TUR', 'hydropower_rent_status'] == 'missing'          # revenue is 'n/a'
    assert np.isnan(out.loc['TUR', 'hydropower_rent_signed_usd'])
    assert pd.isna(out.loc['CCC', 'hydropower_rent_status'])               # not in the workbook
    assert np.isclose(out.loc['CCC', 'hydropower_gep'], 50.0 / wf.annuity_factor(0.04))


def test_without_the_workbook_the_rent_columns_are_missing_and_say_why():
    wealth = pd.DataFrame({'countrycode': ['AAA'], 'countryname': ['Aland'], 'YR2019': [245.05]})
    out = wf.hydropower_rent_table(wealth, CWON_CAPITALIZATION_RATE, None)
    assert out['hydropower_rent_status'].tolist() == ['workbook not staged']
    assert out['hydropower_rent_signed_usd'].isna().all()
    assert out['hydropower_gep_reference_variant'].isna().all()
    assert out['hydropower_gep'].notna().all()


def test_a_workbook_out_of_layout_or_with_an_unknown_country_is_refused():
    wealth = pd.DataFrame({'countrycode': ['AAA'], 'countryname': ['Aland'], 'YR2019': [1.0]})
    moved = _database([('Aland', 2019, 'Hydropower', 1.0, 0.0)])
    moved = moved[[c for c in moved.columns if c != 'col1'] + ['col1']]   # shifts W and Y left
    with pytest.raises(ValueError, match='column 23'):
        wf.hydropower_rent_from_revenue_and_cost(moved, wealth)
    unknown = _database([('Nowhere', 2019, 'Hydropower', 1.0, 0.0)])
    with pytest.raises(ValueError, match='Nowhere'):
        wf.hydropower_rent_from_revenue_and_cost(unknown, wealth)


def test_es_parameters_names_the_private_workbook_with_its_hash():
    params = pd.read_csv(os.path.join(os.path.dirname(wf.__file__), '..', 'input_template',
                                      'es_parameters.csv'), encoding='utf-8-sig').set_index('parameter')
    assert params.loc['water_supply_cwon_hydro_workbook_path', 'value'].endswith(
        'Hydropower valuation_CWON 2024_rev4.xlsx')
    assert len(params.loc['water_supply_cwon_hydro_workbook_sha256', 'value']) == 64


def test_the_workbook_rent_reproduces_the_authors_output_country_by_country():
    """Where the private workbook is staged: revenues minus total costs for 2019 give the authors'
    95 positive rents (their file stores whole dollars) and the 17 they leave blank, all of which
    are negative here, with wealth in the CWoN table for every one. 12 of the 17 carry positive
    CWoN wealth, the countries where the wealth-based value and the authors' output differ."""
    path = _workbook_path()
    if path is None:
        pytest.skip('the private hydropower valuation workbook is not staged on this machine')
    wealth = pd.read_stata(os.path.join(REFERENCE_DIR, 'hydro_wealth_cd.dta'))
    anchor = pd.read_csv(os.path.join(REFERENCE_DIR, 'gep_hydro_directuse_CWONresrent_20260720.csv'))
    database = pd.read_excel(path, sheet_name=wf.HYDROPOWER_WORKBOOK_SHEET)
    table = wf.hydropower_rent_table(wealth, CWON_CAPITALIZATION_RATE, database)

    rent = table[table['hydropower_rent_status'].isin(['positive', 'negative', 'zero', 'missing'])]
    assert len(rent) == 112
    assert rent['hydropower_rent_status'].value_counts().to_dict() == {'positive': 95, 'negative': 17}
    assert rent.loc[rent['hydropower_rent_signed_usd'] > 0, 'hydropower_rent_signed_usd'].sum() == \
        pytest.approx(144_680_445_647.89, abs=0.05)

    merged = anchor.merge(table, on='iso3_r250_label', how='left')
    variant, ref = merged['hydropower_gep_reference_variant'], merged['gep_hydro_cwonresrent_2019usd']
    both = variant.notna() & ref.notna()
    assert both.sum() == 95
    assert (variant[both] - ref[both]).abs().max() <= 1.0
    assert not (ref.notna() & variant.isna()).any()
    assert not (ref.isna() & variant.notna()).any()
    negative = table[table['hydropower_rent_status'] == 'negative']
    assert negative['hydropower_wealth_usd'].notna().all()
    assert int((negative['hydropower_wealth_usd'] > 0).sum()) == 12


def test_no_wealth_stays_nan_and_the_join_keeps_all_countries():
    wealth = pd.DataFrame({'countrycode': ['AAA'], 'YR2019': [245.05]})
    hydropower = wf.hydropower_rent_from_wealth(wealth, CWON_CAPITALIZATION_RATE)
    countries = pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB'], 'iso3_r250_id': [1, 2]})
    out = wf.water_supply_gep_by_country(hydropower, countries)
    assert len(out) == 2
    assert np.isclose(out.set_index('iso3_r250_label').loc['AAA', 'hydropower_gep'], 10.0, rtol=1e-3)
    assert np.isnan(out.set_index('iso3_r250_label').loc['BBB', 'hydropower_gep'])


def test_es_config_and_parameters_rows_hydrate_water_supply(tmp_path):
    from global_invest import utilities
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'water_supply', log=lambda *a: None)
    assert p.gep_base_year == 2019
    utilities.hydrate_es_parameters(p, 'water_supply', log=lambda *a: None)
    assert p.water_supply_cwon_hydro_wealth_path.endswith('cwon/hydro_wealth_cd.dta')
    assert p.water_use_efficiency_input_path.endswith('aquastat/aquastat_water_efficiency.csv')
    assert p.water_use_withdrawal_path.endswith('water_use/data/water_withdraw.csv')


# --- Water-use chain (efficiency x withdrawal) ---
def test_water_use_cleaning_replicates_the_drive_table_exactly():
    raw = pd.read_csv(os.path.join(REFERENCE_DIR, 'aquastat_water_efficiency.csv'),
                      encoding='utf-8-sig')
    anchor = pd.read_csv(os.path.join(REFERENCE_DIR, 'aquastats_cleaned.csv'),
                         encoding='utf-8-sig')
    ours = wf.clean_aquastat_water_efficiency(raw)
    a = ours.sort_values(['country', 'year']).reset_index(drop=True)
    b = anchor.sort_values(['country', 'year']).reset_index(drop=True)
    assert a.shape == b.shape == (1816, 6)
    assert list(a.columns) == list(b.columns)
    assert (a[['country', 'year']] == b[['country', 'year']]).all().all()
    for col in a.columns[2:]:
        # Bit-exact: both sides carry the export's values unchanged through the pivot.
        assert ((a[col] == b[col]) | (a[col].isna() & b[col].isna())).all(), col


def test_water_use_gep_replicates_the_drive_intermediate():
    efficiency = pd.read_csv(os.path.join(REFERENCE_DIR, 'aquastats_cleaned.csv'),
                             encoding='utf-8-sig')
    withdrawal = pd.read_csv(os.path.join(REFERENCE_DIR, 'water_withdraw.csv'),
                             encoding='utf-8-sig')
    anchor = pd.read_csv(os.path.join(REFERENCE_DIR, 'gep_wateruse.csv'), encoding='utf-8-sig')
    ours = wf.water_use_gep_by_country_year(efficiency, withdrawal)
    keys = ['iso_code', 'country', 'year']
    a = ours.sort_values(keys, na_position='last').reset_index(drop=True)
    b = anchor.sort_values(keys, na_position='last').reset_index(drop=True)
    assert a.shape == b.shape == (700, 13)
    assert list(a.columns) == list(b.columns)
    assert (a[keys].fillna('') == b[keys].fillna('')).all().all()
    for col in a.columns[3:]:
        x, y, both = a[col], b[col], a[col].notna()
        assert (x.isna() == y.isna()).all(), col
        if col.startswith('gep_'):
            # The anchor CSV stores R fwrite's 15-significant-digit rounding of the identical
            # product, so the recomputed values differ by up to 2.8e-15 relative.
            assert np.allclose(x[both], y[both], rtol=1e-14), col
        else:
            assert (x[both] == y[both]).all(), col   # inputs pass through bit-exact


def test_water_use_sector_arithmetic_on_synthetic_data():
    efficiency = pd.DataFrame({'country': ['Aaa'], 'year': [2015],
                               'wue_irrigation_usdpm3': [0.5], 'wue_industrial_usdpm3': [2.0],
                               'wue_municipal_usdpm3': [3.0], 'wue_general_usdpm3': [1.0]})
    withdrawal = pd.DataFrame({'country': ['Aaa'], 'iso_code': ['AAA'], 'year': [2015],
                               'w_agriculture': [100.0], 'w_industry': [10.0], 'w_munucipal': [5.0]})
    out = wf.water_use_gep_by_country_year(efficiency, withdrawal).iloc[0]
    assert np.isclose(out['gep_water_agricultural'], 50.0)
    assert np.isclose(out['gep_water_industrial'], 20.0)
    assert np.isclose(out['gep_water_municipal'], 15.0)
    # A non-survey year is dropped, as in the source.
    off_year = wf.water_use_gep_by_country_year(efficiency.assign(year=2013),
                                                withdrawal.assign(year=2013))
    assert len(off_year) == 0


def test_water_use_committed_anchors_join_and_total():
    agriculture = pd.read_csv(os.path.join(REFERENCE_DIR, 'wateruse_ag_gep.csv'))
    all_sector = pd.read_csv(os.path.join(REFERENCE_DIR, 'wateruse_gep.csv'))
    countries = agriculture[['iso3_r250_id', 'iso3_r250_label']].merge(
        all_sector[['iso3_r250_id', 'iso3_r250_label']], how='outer')
    out = wf.water_use_components_by_country(agriculture, all_sector, countries)
    assert out['water_use_agriculture_value_added'].notna().sum() == 145
    assert out['water_use_all_sector_value_added'].notna().sum() == 183
    assert np.isclose(out['water_use_agriculture_value_added'].sum(),
                      agriculture['wateruse_ag_gep'].sum())


# ---------------------------------------------------------------------------
# One row per country: the AQUASTAT double-spelling fan-out.
# ---------------------------------------------------------------------------

def _components(rows):
    # irrigation and domestic are the VALUE ADDED columns: SDG 6.4.1 inverts back to value
    # added, so that is what the chain produces, and the account's figure is a share of it.
    return pd.DataFrame(rows, columns=['country', 'iso3_r250_id', 'iso3_r250_label', 'year',
                                       'water_use_agriculture_value_added',
                                       'water_use_irrigation_value_added',
                                       'water_use_domestic_value_added',
                                       'water_use_all_sector_value_added'])


def test_two_spellings_of_one_country_collapse_to_a_single_row():
    """The export names Russia twice. Left-merging both onto the country list would count it
    twice in every total."""
    out = wf.one_row_per_country(_components([
        ['Russian Federation', 643.0, 'RUS', 2015, np.nan, np.nan, np.nan, np.nan],
        ['Russia', 643.0, 'RUS', 2000, 5.0, 5.0, 2.0, 7.0],
    ]))
    assert len(out) == 1
    assert out.iloc[0]['iso3_r250_label'] == 'RUS'
    assert out.iloc[0]['water_use_agriculture_value_added'] == 5.0     # the non-empty spelling wins
    assert out.iloc[0]['water_use_all_sector_value_added'] == 7.0


def test_a_country_the_name_join_could_not_resolve_passes_through():
    """An unresolved name keeps its empty id rather than being dropped, so a name drift in
    the export stays visible instead of silently losing a country."""
    out = wf.one_row_per_country(_components([
        ['Cape Verde', np.nan, np.nan, 2015, 9.0, 9.0, 3.0, 12.0],
        ['Kenya', 404.0, 'KEN', 2015, 1.0, 1.0, 1.0, 2.0],
    ]))
    assert len(out) == 2
    assert set(out['country']) == {'Cape Verde', 'Kenya'}


def test_two_spellings_that_disagree_on_a_value_raise():
    """Combining them would decide, silently, which number the country gets."""
    with pytest.raises(ValueError, match='disagree'):
        wf.one_row_per_country(_components([
            ['Russian Federation', 643.0, 'RUS', 2015, 5.0, 5.0, 2.0, 7.0],
            ['Russia', 643.0, 'RUS', 2000, 6.0, 6.0, 1.0, 7.0],
        ]))


# ---------------------------------------------------------------------------
# The water share: the step that turns a denominator into an account figure.
# ---------------------------------------------------------------------------

def _value_added():
    return pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB'],
                         'water_use_irrigation_value_added': [1000.0, 500.0],
                         'water_use_domestic_value_added': [4000.0, np.nan]})


# ---------------------------------------------------------------------------
# The irrigation premium: what an irrigated hectare earns above the same land rainfed.
# ---------------------------------------------------------------------------

def _aquastat(rows):
    return pd.DataFrame(rows, columns=['m49', 'Year', 'VariableCode', 'Value'])


def test_the_premium_is_the_difference_per_hectare_not_the_whole_value_added():
    """AAA: 100 USD of ag value added, 60% of it from 10 ha irrigated, 90 ha rainfed.

    Irrigated earns 60/10 = 6 per ha, rainfed 40/90 = 0.444, so the premium is 5.556 a hectare
    and 55.56 over the ten hectares. The whole irrigated value added is 60 -- crediting water
    with that is the error the premium exists to fix, because the 0.444 a hectare rainfed land
    earns would be earned with or without irrigation.
    """
    aq = _aquastat([[1, 2019, wf.AQUASTAT_AG_VALUE_ADDED_CODE, 100.0],
                    [1, 2019, wf.AQUASTAT_IRRIGATED_GVA_SHARE_CODE, 60.0],
                    [1, 2019, wf.AQUASTAT_IRRIGATED_AREA_CODE, 0.01]])      # 0.01 x 1000 = 10 ha
    crop = pd.DataFrame({'m49': [1], 'Year': [2019], 'cropland_1000ha': [0.1]})   # 100 ha
    out = wf.irrigation_premium_by_country(aq, crop, 2019).set_index('m49')
    assert out.loc[1, 'irrigated_area_ha'] == pytest.approx(10.0)
    assert out.loc[1, 'premium_usd_per_ha'] == pytest.approx(6.0 - 40.0 / 90.0)
    assert out.loc[1, 'irrigation_premium_usd'] == pytest.approx(55.5556, rel=1e-4)


def test_the_premium_uses_the_account_year_on_both_sources():
    """A country reporting a later year is not used for it: the account's year is the account's.

    Taking each country's latest available year instead gives a median vintage of 2022 and a
    total 26% higher, which is a different year's answer rather than a better one.
    """
    aq = _aquastat([[1, 2019, wf.AQUASTAT_AG_VALUE_ADDED_CODE, 100.0],
                    [1, 2019, wf.AQUASTAT_IRRIGATED_GVA_SHARE_CODE, 60.0],
                    [1, 2019, wf.AQUASTAT_IRRIGATED_AREA_CODE, 0.01],
                    [1, 2022, wf.AQUASTAT_AG_VALUE_ADDED_CODE, 999.0],
                    [1, 2022, wf.AQUASTAT_IRRIGATED_GVA_SHARE_CODE, 60.0],
                    [1, 2022, wf.AQUASTAT_IRRIGATED_AREA_CODE, 0.01]])
    crop = pd.DataFrame({'m49': [1, 1], 'Year': [2019, 2022], 'cropland_1000ha': [0.1, 0.1]})
    out = wf.irrigation_premium_by_country(aq, crop, 2019)
    assert out['irrigation_premium_usd'].sum() == pytest.approx(55.5556, rel=1e-4)


def test_a_country_with_no_rainfed_side_is_dropped_rather_than_dividing_by_zero():
    # Irrigated area equal to cropland leaves no rainfed hectares to compare against.
    aq = _aquastat([[1, 2019, wf.AQUASTAT_AG_VALUE_ADDED_CODE, 100.0],
                    [1, 2019, wf.AQUASTAT_IRRIGATED_GVA_SHARE_CODE, 60.0],
                    [1, 2019, wf.AQUASTAT_IRRIGATED_AREA_CODE, 0.1]])
    crop = pd.DataFrame({'m49': [1], 'Year': [2019], 'cropland_1000ha': [0.1]})
    assert len(wf.irrigation_premium_by_country(aq, crop, 2019)) == 0


# ---------------------------------------------------------------------------
# Domestic: the withdrawn cubic metres are the water; the price turns them into a value.
# ---------------------------------------------------------------------------

def _withdrawal(rows):
    return pd.DataFrame(rows, columns=['m49', 'Year', 'VariableCode', 'Value'])


def test_domestic_volume_is_municipal_plus_industrial_at_the_account_year():
    aq = _withdrawal([[1, 2019, wf.AQUASTAT_MUNICIPAL_WITHDRAWAL_CODE, 2.0],    # 2 km3
                      [1, 2019, wf.AQUASTAT_INDUSTRIAL_WITHDRAWAL_CODE, 3.0],
                      [1, 2022, wf.AQUASTAT_MUNICIPAL_WITHDRAWAL_CODE, 99.0],   # other year ignored
                      [2, 2019, wf.AQUASTAT_MUNICIPAL_WITHDRAWAL_CODE, 1.0]])   # industrial missing
    out = wf.domestic_withdrawal_by_country(aq, 2019).set_index('m49')
    assert out.loc[1, 'domestic_withdrawal_m3'] == pytest.approx(5e9)
    # A country reporting one sector keeps that sector rather than being dropped.
    assert out.loc[2, 'domestic_withdrawal_m3'] == pytest.approx(1e9)


def test_municipal_values_join_by_label_and_refuse_what_cannot_be_placed():
    countries = pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB']})
    out = wf.municipal_gep_by_country(
        pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB'], 'wateruse_municipal_gep': [5.0, np.nan]}),
        countries).set_index('iso3_r250_label')
    assert out.loc['AAA', 'water_use_domestic_gep'] == 5.0
    assert np.isnan(out.loc['BBB', 'water_use_domestic_gep'])       # no value stays missing
    with pytest.raises(ValueError, match='does not carry'):
        wf.municipal_gep_by_country(
            pd.DataFrame({'iso3_r250_label': ['ZZZ'], 'wateruse_municipal_gep': [1.0]}), countries)
    with pytest.raises(ValueError, match='negative'):
        wf.municipal_gep_by_country(
            pd.DataFrame({'iso3_r250_label': ['AAA'], 'wateruse_municipal_gep': [-1.0]}), countries)


def test_the_staged_municipal_table_is_the_authors_final_two_percent_run():
    """The authors' final Results/dr02 table at f94e33a4, matching column N of All Services GEP
    Values 2026_10_01 within its reported precision: 160 valued countries, no zeros."""
    path = os.path.join(REFERENCE_DIR, 'municipal', 'lifeng_f94e33a4_dr02', 'wateruse_municipal_gep.csv')
    if not os.path.exists(path):
        pytest.skip('the authors\' municipal table is not staged on this machine')
    values = pd.read_csv(path)['wateruse_municipal_gep']
    assert values.sum() == pytest.approx(9_231_821_271.88, rel=1e-12)
    assert (values.notna().sum(), (values == 0).sum(), (values < 0).sum()) == (160, 0, 0)


def test_no_price_publishes_no_domestic_gep():
    """A missing price is not a price of zero: the volume is published and no GEP."""
    aq = _withdrawal([[1, 2019, wf.AQUASTAT_MUNICIPAL_WITHDRAWAL_CODE, 2.0]])
    out = wf.apply_raw_water_price(wf.domestic_withdrawal_by_country(aq, 2019), None)
    assert 'water_use_domestic_gep' not in out.columns


def test_a_price_values_the_ecosystem_volume_and_a_negative_price_is_refused():
    aq = _withdrawal([[1, 2019, wf.AQUASTAT_MUNICIPAL_WITHDRAWAL_CODE, 2.0]])
    vol = wf.domestic_withdrawal_by_country(aq, 2019)
    out = wf.apply_raw_water_price(vol, 0.05)
    assert out['water_use_domestic_gep'].iloc[0] == pytest.approx(1e8)
    with pytest.raises(ValueError, match='cannot be negative'):
        wf.apply_raw_water_price(vol, -0.05)


def test_desalinated_water_is_not_ecosystem_water():
    """Kuwait's shape: desalination exceeds domestic withdrawal, so the ecosystem provides zero.

    Desalinated water is manufactured. A country whose desal production covers its whole domestic
    withdrawal gets a floor at zero rather than a negative volume, and a country without desal
    keeps its withdrawal unchanged.
    """
    aq = _withdrawal([[1, 2019, wf.AQUASTAT_MUNICIPAL_WITHDRAWAL_CODE, 1.0],
                      [1, 2019, wf.AQUASTAT_DESALINATED_CODE, 1.5],
                      [2, 2019, wf.AQUASTAT_MUNICIPAL_WITHDRAWAL_CODE, 2.0]])
    out = wf.domestic_withdrawal_by_country(aq, 2019).set_index('m49')
    assert out.loc[1, 'domestic_ecosystem_m3'] == 0.0
    assert out.loc[2, 'domestic_ecosystem_m3'] == pytest.approx(2e9)
    priced = wf.apply_raw_water_price(out.reset_index(), 0.05).set_index('m49')
    assert priced.loc[1, 'water_use_domestic_gep'] == 0.0


def test_irrigation_gep_is_the_premium_times_the_share_with_negatives_clipped():
    premium = pd.DataFrame({'m49': [1, 2], 'irrigation_premium_usd': [1000.0, -400.0]})
    out = wf.irrigation_gep_from_premium(premium, 0.332).set_index('m49')
    assert out.loc[1, 'water_use_irrigation_gep'] == pytest.approx(332.0)
    # An account cannot report negative provisioning; the negative case is a data question.
    assert out.loc[2, 'water_use_irrigation_gep'] == 0.0
    with pytest.raises(ValueError, match='sit in'):
        wf.irrigation_gep_from_premium(premium, 1.4)


def test_the_implied_price_is_what_a_cubic_metre_earns_in_agriculture():
    assert wf.implied_raw_water_price(200.0, 1000.0) == pytest.approx(0.2)
    with pytest.raises(ValueError, match='positive'):
        wf.implied_raw_water_price(200.0, 0.0)


# --- Agriculture (GEP-WaterUse-Agriculture replication): pinned reference, final-stage reproduction ---
# These tests reproduce the author's final valuation and aggregation stage from his own
# processed inputs (theta x GPV 2020). They do not reproduce the upstream lambda, SPAM or
# crop-mapping construction and are not scientific validation.
AGRICULTURE_REFERENCE_DIR = os.path.join(REFERENCE_DIR, 'agriculture', 'shunkei_e229f881')
AGRICULTURE_REFERENCE_SHA256 = {
    'lambda_jr_tbl_CropGBWater.csv': '01a245c7f9d492560d95c736e9274fb846a001a921e08a9fe21c582c4e058f4c',
    'spam_cropgbwater_share.csv': '1d83e570f7dd097d2d616087170ba4925e4e3dbb01d87dbf68364ffee0336df8',
    'GPV_rainfed_irrigated_CropGBWater_2020.csv': 'a7b22fd4d9cbcfcc4c4227d90642362f89b0725361921859179201254d477e46',
    'theta_table_final.csv': 'bd75d855fbf1422dbfeb5a1af59345324f3502f09eee374732012bf4a74d4069',
    'CropGBWater_FAO_Ky_Kc.csv': 'c782e9b21aa09fdb777239729253bb49cb24a8a6d6abddf328efa8c5a4ee3ae0',
    'CropGBWater_FAOSTAT_GPV_crop_correspondence.csv': 'e2803bedd0adcc0bd476c8f947283fa13e325a489be07f5b2b7973d9f3e0e5d4',
    'gep_agwater_country_source.csv': 'c83c721debce654d357d168335dd010d05ee648db38d5fce37bda78c003cd0e5',
    'gep_agwater_crop_country_source.csv': 'a1391945ce768df217fa3ef3959e31478b2ac3afd76a14aaa1ae468af4880fcb',
}
# Comparing float64 arithmetic reordered between R and pandas, in the files' own unit
# (thousand current US$): measured worst deviations are 7e-8 (crop rows) and 6e-7
# (country rows) thousand USD, so atol admits a tenth of a dollar and rtol a billionth
# of the value. A reference cell of exactly zero is compared through atol alone.
AGRICULTURE_RTOL = 1e-9
AGRICULTURE_ATOL_THOUSAND_USD = 1e-4


def _agriculture_reference(file_name):
    path = os.path.join(AGRICULTURE_REFERENCE_DIR, file_name)
    assert os.path.exists(path), (
        f'agriculture reference not staged: {path}. The reproduction cannot be claimed '
        'without it, so this is a failure, not a skip.')
    return pd.read_csv(path)


def test_agriculture_reference_files_match_their_pinned_hashes():
    for file_name, expected in AGRICULTURE_REFERENCE_SHA256.items():
        path = os.path.join(AGRICULTURE_REFERENCE_DIR, file_name)
        assert os.path.exists(path), f'agriculture reference not staged: {path}'
        with open(path, 'rb') as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        assert digest == expected, f'{file_name} differs from the pinned 2026-10-27 retrieval'


def _agriculture_reference_valuation():
    lam = _agriculture_reference('lambda_jr_tbl_CropGBWater.csv').rename(columns={'iso3': 'iso3_r250_label'})
    gpv = _agriculture_reference('GPV_rainfed_irrigated_CropGBWater_2020.csv').rename(columns={'iso3': 'iso3_r250_label'})
    theta = _agriculture_reference('theta_table_final.csv').rename(columns={'iso3': 'iso3_r250_label'})
    assert not theta.duplicated('iso3_r250_label').any(), 'theta must be one row per country'
    keys = ['CropGBWater_crop', 'iso3_r250_label']
    assert not lam.duplicated(keys).any() and not gpv.duplicated(keys).any()
    basis = gpv.merge(theta[['iso3_r250_label', 'theta']], on='iso3_r250_label', how='inner')
    basis['basis_rain'] = basis['theta'] * basis['GPV_rain_jr']
    basis['basis_irr'] = basis['theta'] * basis['GPV_irr_jr']
    lambda_cols = list(wf.AGRICULTURE_LAMBDA_COLUMNS) + list(wf.AGRICULTURE_VMR_LAMBDA_COLUMNS)
    return wf.agriculture_gep_from_lambda(lam[keys + lambda_cols],
                                          basis[keys + ['basis_rain', 'basis_irr']])


def _assert_column_reproduced(merged, ours, reference):
    a, b = merged[ours], merged[reference]
    assert (a.isna() == b.isna()).all(), f'{ours}: missing pattern differs from the reference'
    both = a.notna()
    assert np.isfinite(a[both]).all() and np.isfinite(b[both]).all()
    close = np.isclose(a[both], b[both], rtol=AGRICULTURE_RTOL, atol=AGRICULTURE_ATOL_THOUSAND_USD)
    assert close.all(), f'{ours}: {int((~close).sum())} rows outside tolerance'


def test_agriculture_valuation_reproduces_reference_crop_rows():
    reference = _agriculture_reference('gep_agwater_crop_country_source.csv')
    keys = ['CropGBWater_crop', 'iso3_r250_label']
    assert not reference.duplicated(keys).any()
    actual = _agriculture_reference_valuation()
    assert not actual.duplicated(keys).any()
    assert len(actual) == len(reference) == 1612
    merged = actual.merge(reference, on=keys, how='outer', indicator=True,
                          validate='one_to_one')
    assert (merged['_merge'] == 'both').all(), 'row coverage differs from the reference merge filters'
    for ours, his in [('GEP_gn_rf', 'GEP_gn_rf_jr'), ('GEP_gn_ir', 'GEP_gn_irr_jr'),
                      ('GEP_bl_irr', 'GEP_bl_irr_jr'), ('GEP_total_irr', 'GEP_total_irr_jr'),
                      ('GEP_gn_rf_vmr', 'GEP_gn_rf_vmr_jr'), ('GEP_gn_ir_vmr', 'GEP_gn_irr_vmr_jr'),
                      ('GEP_total', 'GEP_total_jr')]:
        _assert_column_reproduced(merged, ours, his)


def test_agriculture_country_aggregation_reproduces_reference():
    country = wf.aggregate_agriculture_gep_to_country(_agriculture_reference_valuation())
    reference = _agriculture_reference('gep_agwater_country_source.csv')
    assert not reference.duplicated('iso3_r250_label').any()
    assert not country.duplicated('iso3_r250_label').any()
    assert len(country) == len(reference) == 145
    merged = country.merge(reference, on='iso3_r250_label', how='outer',
                           suffixes=('_ours', '_reference'), indicator=True)
    assert (merged['_merge'] == 'both').all(), 'country coverage differs from the reference'
    for column in ['GEP_gn_rf', 'GEP_gn_ir', 'GEP_bl_irr', 'GEP_total_irr',
                   'GEP_gn_rf_vmr', 'GEP_gn_ir_vmr', 'GEP_total']:
        _assert_column_reproduced(merged, column + '_ours', column + '_reference')
    assert (merged['n_crop_ours'] == merged['n_crop_reference']).all()
    # the global check reads the UNROUNDED sum from the staged reference itself
    assert np.isclose(country['GEP_total'].sum(), reference['GEP_total'].sum(),
                      rtol=AGRICULTURE_RTOL, atol=AGRICULTURE_ATOL_THOUSAND_USD)


def test_agriculture_basis_conserves_shared_items_and_preserves_missing():
    values = pd.DataFrame({
        'iso3_r250_label': ['AAA'] * 5, 'year': [2019] * 5,
        'crop': ['Wheat', 'Beans', 'Beans', 'Missing', 'Unmapped'],
        'crop_provision_gep': [100., 40., np.nan, np.nan, 50.]})
    corr = pd.DataFrame({'FAOSTAT_Item': ['Wheat', 'Wheat', 'Beans', 'Missing'],
                         'CropGBWater_crop': ['Spring', 'Winter', 'Beans', 'Missing']})
    spam = pd.DataFrame({'iso3_r250_label': ['AAA'] * 4,
                         'CropGBWater_crop': ['Spring', 'Winter', 'Beans', 'Missing'],
                         'Q_total_for_share': [1., 3., 2., 2.],
                         'share_prod_rain': [.6] * 4, 'share_prod_irr': [.4] * 4})
    basis, diagnostics = wf.agriculture_basis_from_crop_values(
        values, corr, spam, 2019, 'crop_provision_gep')
    indexed = basis.set_index('CropGBWater_crop')
    assert indexed.loc['Spring', 'basis_total'] == 25.
    assert indexed.loc['Winter', 'basis_total'] == 75.
    assert indexed.loc['Beans', 'basis_total'] == 40.
    assert np.isnan(indexed.loc['Missing', 'basis_total'])
    assert diagnostics['basis_value_total'] == pytest.approx(140.)
    assert diagnostics['unmapped_items']['sum'].sum() == 50.
    bad = pd.concat([values, values.iloc[[1]].assign(crop_provision_gep=41.)])
    with pytest.raises(ValueError, match='conflicting duplicate'):
        wf.agriculture_basis_from_crop_values(bad, corr, spam, 2019, 'crop_provision_gep')
    with pytest.raises(ValueError, match='unique crop-country'):
        wf.agriculture_basis_from_crop_values(values, corr, pd.concat([spam, spam.iloc[[0]]]),
                                              2019, 'crop_provision_gep')


def test_agriculture_task_recomputes_changed_inputs_and_registers_paths(tmp_path, monkeypatch):
    from global_invest.water_supply import water_supply_tasks as wt
    monkeypatch.setattr(wt, 'publish_inputs', lambda p: p)
    p = SimpleNamespace(cur_dir=str(tmp_path), project_dir=str(tmp_path), run_this=True,
                        gep_base_year=2019, results={})
    frames = {
        'crop_values': pd.DataFrame({'iso3_r250_label': ['AAA'], 'crop': ['Beans'],
                                     'year': [2019], 'crop_provision_gep': [100.]}),
        'agriculture_correspondence': pd.DataFrame({'CropGBWater_crop': ['Beans'],
                                                    'FAOSTAT_Item': ['Beans']}),
        'agriculture_spam': pd.DataFrame({'iso3': ['AAA'], 'CropGBWater_crop': ['Beans'],
                                         'Q_total_for_share': [10.],
                                         'share_prod_rain': [.6], 'share_prod_irr': [.4]}),
        'agriculture_lambda': pd.DataFrame({'iso3': ['AAA'], 'CropGBWater_crop': ['Beans'],
                                           'lambda_gn_rf': [.5], 'lambda_gn_ir': [.25],
                                           'lambda_bl_ir': [.5], 'lambda_total_ir': [.75]})}
    for name, frame in frames.items():
        path = str(tmp_path / (name + '.csv'))
        frame.to_csv(path, index=False)
        setattr(p, 'water_supply_' + name + '_path', path)
    wt.agriculture_water(p)
    first = pd.read_csv(p.water_supply_agriculture_country_path)
    assert first.loc[0, 'water_use_agriculture_gep'] == pytest.approx(60.)
    assert first.loc[0, 'water_use_irrigation_gep'] == pytest.approx(60.)
    assert first.loc[0, 'water_use_irrigated_total_gep'] == pytest.approx(30.)
    frames['crop_values']['crop_provision_gep'] = 200.
    frames['crop_values'].to_csv(p.water_supply_crop_values_path, index=False)
    wt.agriculture_water(p)
    second = pd.read_csv(p.water_supply_agriculture_country_path)
    assert second.loc[0, 'water_use_agriculture_gep'] == pytest.approx(120.)
    p.run_this = False
    wt.agriculture_water(p)
    assert os.path.isfile(p.water_supply_agriculture_country_path)


def test_water_country_output_replaces_old_agriculture_but_preserves_domestic(tmp_path, monkeypatch):
    from global_invest.water_supply import water_supply_tasks as wt
    monkeypatch.setattr(wt, 'publish_inputs', lambda p: p)
    monkeypatch.setattr(utilities, 'collapse_countries_to_r250', lambda df: df)
    monkeypatch.setattr(utilities, 'write_gep_by_country', lambda p, df, path: df.to_csv(path, index=False))
    countries = pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB'], 'iso3_r250_id': [1, 2],
                              'iso3_r250_name': ['A', 'B'], 'continent': ['X', 'X'],
                              'region_un': ['X', 'X'], 'region_wb': ['X', 'X'],
                              'income_grp': ['X', 'X'], 'subregion': ['X', 'X']})
    p = SimpleNamespace(cur_dir=str(tmp_path), run_this=True, results={}, gep_base_year=2019,
                        df_countries=countries)
    frames = {
        'hydropower_rent': pd.DataFrame({'iso3_r250_label': ['AAA'], 'hydropower_gep': [10.],
                                         'hydropower_gep_reference_variant': [10.]}),
        'water_use_components': pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB'],
                                              'iso3_r250_id': [1, 2],
                                              'water_use_irrigation_gep': [999., 999.],
                                              'water_use_domestic_gep': [7., 8.]}),
        'water_supply_agriculture_country': pd.DataFrame({'iso3_r250_label': ['AAA'],
                                                          'year': [2019],
                                                          'water_use_agriculture_gep': [60.],
                                                          'water_use_irrigation_gep': [60.]}),
        'water_supply_municipal_results': pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB'],
                                                        'wateruse_municipal_gep': [3., np.nan]})}
    for name, df in frames.items():
        path = str(tmp_path / (name + '.csv'))
        df.to_csv(path, index=False)
        setattr(p, name + '_path', path)
    # A stale published output must not bypass the calculation.
    (tmp_path / 'gep_by_country_base_year.csv').write_text('old\n999\n')
    wt.gep_calculation(p)
    result = pd.read_csv(p.results['water_supply']['gep_by_country_base_year'])
    # The domestic value is the authors' municipal table, missing where they have none; the
    # chain's implied-price construction is kept beside it as the variant.
    assert result.loc[0, 'water_use_domestic_gep'] == 3.
    assert np.isnan(result.loc[1, 'water_use_domestic_gep'])
    assert result['water_use_domestic_gep_implied_price_variant'].tolist() == [7., 8.]
    assert result.loc[0, 'water_use_irrigation_gep'] == 60.
    assert np.isnan(result.loc[1, 'water_use_irrigation_gep'])
    assert result.loc[0, 'water_supply_gep'] == 10.


def test_water_task_tree_includes_agriculture_and_accepts_parent():
    from global_invest.water_supply import water_supply_initialize as wi
    registered = []
    p = SimpleNamespace(add_task=lambda fn, **kwargs: registered.append((fn.__name__, kwargs)))
    wi.add_water_supply_tasks(p, parent='parent')
    assert [name for name, _ in registered] == [
        'hydropower_rent', 'water_use_components', 'agriculture_water', 'gep_calculation']
    assert all(kw['parent'] == 'parent' for _, kw in registered)

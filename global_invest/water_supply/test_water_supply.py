"""Unit tests for the water_supply hydropower component (CWoN resource-rent method) and the
water-use calculation.

The hydropower method was identified against the consortium drive's committed output, so the
tests pin BOTH halves of that identification: the annuity identity (the observed constant
0.040808 is exactly 1/annuity(4%, 100y)) and the exact replication of the committed anchor
from the CWoN wealth table (both shipped in reference/). The water-use tests pin the ported
two-script calculation against its two committed intermediates (cleaning bit-exact, products to the
anchor's own CSV rounding) and the adopted per-country outputs' joins and totals.
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from global_invest import utilities
from global_invest.water_supply import water_supply_functions as wf


# CWoN's capitalization rate, which is es_parameters configuration rather than a module
# constant now, so the tests name it here instead of importing one that no longer exists.
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
    factor = wf.annuity_factor(0.04)   # CWoN's capitalization rate, now an es_parameters row
    assert np.isclose(factor, 24.504998997, rtol=1e-9)      # sum of 1/1.04^t, t=1..100
    assert np.isclose(1.0 / factor, 0.040808, atol=1e-6)    # the ratio observed in the anchor


def test_exact_replication_of_the_committed_anchor():
    wealth = pd.read_stata(os.path.join(REFERENCE_DIR, 'hydro_wealth_cd.dta'))
    anchor = pd.read_csv(os.path.join(REFERENCE_DIR, 'gep_hydro_directuse_CWONresrent_20260720.csv'))

    rent = wf.hydropower_rent_from_wealth(wealth, CWON_CAPITALIZATION_RATE)
    merged = anchor.merge(rent, on='iso3_r250_label', how='left')
    variant, ref = merged['hydropower_gep_reference_variant'], merged['gep_hydro_cwonresrent_2019usd']
    both = variant.notna() & ref.notna()
    assert both.sum() == 95                                  # the valued countries, Venezuela included
    # The anchor stores whole dollars, so at 1e7-1e9 magnitudes its rounding is ~1e-7 relative.
    assert np.allclose(variant[both], ref[both], rtol=1e-6)
    # The reference-matching variant must line up with the anchor exactly, country for country.
    assert not (ref.notna() & variant.isna()).any()
    assert not (ref.isna() & variant.notna()).any()
    raw = wealth.set_index('countrycode')['YR2019']
    for label in wf.HYDROPOWER_REFERENCE_EXCLUDED:
        assert pd.notna(raw.get(label)), label      # the exclusion is real: wealth exists


def test_the_reported_value_covers_the_countries_the_reference_drops():
    """The reported column must NOT reproduce the reference's unexplained exclusions. Blanking
    them would fit our number to the anchor, and the discrepancy could never surface."""
    wealth = pd.read_stata(os.path.join(REFERENCE_DIR, 'hydro_wealth_cd.dta'))
    rent = wf.hydropower_rent_from_wealth(wealth, CWON_CAPITALIZATION_RATE).set_index('iso3_r250_label')

    for label in wf.HYDROPOWER_REFERENCE_EXCLUDED:
        assert pd.notna(rent.loc[label, 'hydropower_gep']), label
        assert pd.isna(rent.loc[label, 'hydropower_gep_reference_variant']), label
    extra = int(rent['hydropower_gep'].notna().sum()
                - rent['hydropower_gep_reference_variant'].notna().sum())
    assert extra == len(wf.HYDROPOWER_REFERENCE_EXCLUDED)
    assert rent['hydropower_gep'].sum() > rent['hydropower_gep_reference_variant'].sum()


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
    # ⚠ irrigation and domestic are the VALUE ADDED columns: SDG 6.4.1 inverts back to value
    # added, so that is what the chain produces, and the account's figure is a share of it.
    return pd.DataFrame(rows, columns=['country', 'iso3_r250_id', 'iso3_r250_label', 'year',
                                       'water_use_agriculture_value_added',
                                       'water_use_irrigation_value_added',
                                       'water_use_domestic_value_added',
                                       'water_use_all_sector_value_added'])


def test_two_spellings_of_one_country_collapse_to_a_single_row():
    """The export names Russia twice. Left-merging both onto the country list counted it
    twice in every total, which is what inflated the reported hydropower number."""
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


def test_no_share_publishes_no_gep_at_all():
    """A missing share is not a share of nothing.

    Until 2026-09-02 there was no share step, which is the same as having applied 1.0: the whole
    value added of irrigated agriculture, industry and services went out under a `_gep` name, and
    the totals came out larger than the economies they sit in. With no share set the account now
    publishes the value added and NO gep, so nothing downstream can read one as the other.
    """
    out = wf.apply_water_share_of_value_added(_value_added(), None)
    assert 'water_use_irrigation_gep' not in out.columns
    assert 'water_use_domestic_gep' not in out.columns
    assert out['water_use_irrigation_value_added'].sum() == 1500.0


def test_a_share_is_applied_to_both_components_and_leaves_the_denominator_intact():
    out = wf.apply_water_share_of_value_added(_value_added(), 0.05)
    assert out['water_use_irrigation_gep'].tolist() == [50.0, 25.0]
    assert out['water_use_domestic_gep'].iloc[0] == 200.0
    # A country with no domestic value added gets no domestic gep, not a zero.
    assert pd.isna(out['water_use_domestic_gep'].iloc[1])
    # The denominator survives, so the two can always be compared.
    assert out['water_use_irrigation_value_added'].tolist() == [1000.0, 500.0]


def test_a_share_outside_zero_to_one_is_refused():
    """A share of value added above 1 would say water is worth more than the output it helps
    produce, which is the exact failure the share exists to prevent."""
    with pytest.raises(ValueError, match='share of value added'):
        wf.apply_water_share_of_value_added(_value_added(), 1.4)

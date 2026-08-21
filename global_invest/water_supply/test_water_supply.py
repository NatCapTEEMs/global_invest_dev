"""Unit tests for the water_supply hydropower component (CWoN resource-rent method) and the
water-use chain.

The hydropower method was identified against the consortium drive's committed output, so the
tests pin BOTH halves of that identification: the annuity identity (the observed constant
0.040808 is exactly 1/annuity(4%, 100y)) and the exact replication of the committed anchor
from the CWoN wealth table (both shipped in reference/). The water-use tests pin the ported
two-script chain against its two committed intermediates (cleaning bit-exact, products to the
anchor's own CSV rounding) and the adopted per-country outputs' joins and totals.
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from global_invest.water_supply import water_supply_functions as wf

REFERENCE_DIR = os.path.join(os.path.dirname(wf.__file__), 'reference')


def test_annuity_identity_matches_the_observed_anchor_ratio():
    factor = wf.annuity_factor()
    assert np.isclose(factor, 24.504998997, rtol=1e-9)      # sum of 1/1.04^t, t=1..100
    assert np.isclose(1.0 / factor, 0.040808, atol=1e-6)    # the ratio observed in the anchor


def test_exact_replication_of_the_committed_anchor():
    wealth = pd.read_stata(os.path.join(REFERENCE_DIR, 'hydro_wealth_cd.dta'))
    anchor = pd.read_csv(os.path.join(REFERENCE_DIR, 'gep_hydro_directuse_CWONresrent_20260720.csv'))

    rent = wf.hydropower_rent_from_wealth(wealth)
    merged = anchor.merge(rent, on='iso3_r250_label', how='left')
    ours, ref = merged['hydropower_gep'], merged['gep_hydro_cwonresrent_2019usd']
    both = ours.notna() & ref.notna()
    assert both.sum() == 95                                  # the valued countries, Venezuela included
    # The anchor stores whole dollars, so at 1e7-1e9 magnitudes its rounding is ~1e-7 relative.
    assert np.allclose(ours[both], ref[both], rtol=1e-6)
    # A country the anchor values must never be missing from our side, and our coverage must
    # not exceed the anchor's: the encoded exclusions carry the whole difference.
    assert not (ref.notna() & ours.isna()).any()
    assert not (ref.isna() & ours.notna()).any()
    raw = wealth.set_index('countrycode')['YR2019']
    for label in wf.HYDROPOWER_REFERENCE_EXCLUDED:
        assert pd.notna(raw.get(label)), label      # the exclusion is real: wealth exists


def test_no_wealth_stays_nan_and_the_join_keeps_all_countries():
    wealth = pd.DataFrame({'countrycode': ['AAA'], 'YR2019': [245.05]})
    hydropower = wf.hydropower_rent_from_wealth(wealth)
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
    assert out['water_use_agriculture_gep'].notna().sum() == 145
    assert out['water_use_all_sector_gep'].notna().sum() == 183
    assert np.isclose(out['water_use_agriculture_gep'].sum(),
                      agriculture['wateruse_ag_gep'].sum())

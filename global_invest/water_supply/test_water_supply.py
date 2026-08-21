"""Unit tests for the water_supply hydropower component (CWoN resource-rent method).

The method was identified against the consortium drive's committed output, so the tests pin
BOTH halves of that identification: the annuity identity (the observed constant 0.040808 is
exactly 1/annuity(4%, 100y)) and the exact replication of the committed anchor from the CWoN
wealth table (both shipped in reference/).
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

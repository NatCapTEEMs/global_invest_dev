"""Unit tests for the fisheries GEP valuation (CWoN method, synthetic and self-contained).

Pins the ported rent-trend math against hand-derived values: the year-mean CPI imputation,
the deflation, the OLS trend prediction with its window midpoint, the zero floor, the source
script's country exclusions, and the no-data-is-not-zero join. The CWoN .dta inputs and the
source pipeline's output CSV are the open staging asks; when they land, the anchor test joins
this file the way fire_protection's did.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from global_invest.fisheries import fisheries_functions as ff


def test_cpi_imputation_fills_missing_with_that_years_cross_country_mean():
    cpi = pd.DataFrame({'countrycode': ['AAA', 'BBB', 'CCC', 'AAA'],
                        'year': [2010, 2010, 2010, 2019],
                        'cpi2019': [80.0, np.nan, 120.0, 100.0]})
    out = ff.clean_cwon_cpi(cpi)
    bbb = out[(out['wb_code'] == 'BBB') & (out['year'] == 2010)]['cpi2019'].iloc[0]
    assert np.isclose(bbb, 100.0)                 # mean of 80 and 120, same year only


def test_rent_cleaning_applies_the_source_exclusions():
    rent = pd.DataFrame({'Year': [2010, 2010, 2009, 2011, 2010],
                         'wb_code': ['CYP', 'AAA', 'CUW', 'CUW', 'BBB'],
                         'FAOEconRent': [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = ff.clean_cwon_econ_rent(rent)
    assert 'CYP' not in set(out['wb_code'])                       # excluded country
    cuw_years = set(out[out['wb_code'] == 'CUW']['year'])
    assert cuw_years == {2011}                                    # CUW 2009/2010 dropped, later kept


def test_deflation_is_nominal_over_cpi_times_100():
    rent = pd.DataFrame({'year': [2010], 'wb_code': ['AAA'], 'FAOEconRent': [50.0]})
    cpi = pd.DataFrame({'wb_code': ['AAA'], 'year': [2010], 'cpi2019': [80.0]})
    out = ff.deflate_rent_to_2019usd(rent, cpi)
    assert np.isclose(out['resrent_2019usd'].iloc[0], 62.5)


def test_trend_prediction_matches_hand_derived_ols():
    # AAA: rent = 10 + 2*(t-2009) over 2009-2018 -> mean at midpoint 2013.5 is 19, slope 2,
    # so the 2019 estimate is 19 + 2 * (2019 - 2013.5) = 30.
    years = list(range(2009, 2019))
    deflated = pd.DataFrame({'wb_code': 'AAA', 'year': years,
                             'resrent_2019usd': [10.0 + 2 * (y - 2009) for y in years]})
    trends = ff.fisheries_rent_trends(deflated)
    row = trends.iloc[0]
    assert row['n_years'] == 10
    assert np.isclose(row['beta_hat'], 2.0)
    assert np.isclose(row['resrent_2019_hat'], 30.0)
    assert np.isclose(row['positive_resrent_2019_hat'], 30.0)


def test_negative_prediction_floors_at_zero_and_single_year_gives_nan():
    deflated = pd.DataFrame({
        'wb_code': ['DOWN'] * 10 + ['ONE'],
        'year': list(range(2009, 2019)) + [2015],
        'resrent_2019usd': [100.0 - 20 * (y - 2009) for y in range(2009, 2019)] + [5.0]})
    trends = ff.fisheries_rent_trends(deflated).set_index('wb_code')
    assert trends.loc['DOWN', 'resrent_2019_hat'] < 0
    assert trends.loc['DOWN', 'positive_resrent_2019_hat'] == 0.0    # floored
    assert np.isnan(trends.loc['ONE', 'beta_hat'])                   # <2 years: no slope
    assert np.isnan(trends.loc['ONE', 'positive_resrent_2019_hat'])  # and no estimate


def test_country_join_keeps_all_countries_and_no_data_stays_nan():
    trends = pd.DataFrame({'wb_code': ['AAA'], 'positive_resrent_2019_hat': [7.0],
                           'n_years': [10], 'mean_resrent_2009_2018': [7.0],
                           'beta_hat': [0.0], 'resrent_2019_hat': [7.0]})
    countries = pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB'], 'iso3_r250_id': [1, 2]})
    out = ff.commfish_gep_by_country(trends, countries)
    assert len(out) == 2
    assert np.isclose(out.set_index('iso3_r250_label').loc['AAA', 'commfish_provision'], 7.0)
    assert np.isnan(out.set_index('iso3_r250_label').loc['BBB', 'commfish_provision'])


def test_es_config_and_parameters_rows_hydrate_the_fisheries_gep(tmp_path):
    from global_invest import utilities
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'fisheries', log=lambda *a: None)
    assert p.gep_base_year == 2019
    utilities.hydrate_es_parameters(p, 'fisheries', log=lambda *a: None)
    assert p.fisheries_cwon_cpi_path.endswith('cwon/cpi2019.dta')
    assert p.fisheries_cwon_econ_rent_path.endswith('cwon/EconRent_Analysis_AllYears.dta')

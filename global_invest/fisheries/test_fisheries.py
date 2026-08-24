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
    # The rent now arrives as its parts, so the fixture supplies those rather than a finished
    # column: landed value, the two costs, and subsidy.
    rent = pd.DataFrame({'Year': [2010, 2010, 2009, 2011, 2010],
                         'wb_code': ['CYP', 'AAA', 'CUW', 'CUW', 'BBB'],
                         'FAO_landval2018': [10.0, 20.0, 30.0, 40.0, 50.0],
                         'FAOVarCost': [5.0] * 5, 'FAOFixCost': [2.0] * 5,
                         'SubsidyUSD2018': [1.0] * 5})
    out = ff.clean_cwon_econ_rent(rent)
    assert 'CYP' not in set(out['wb_code'])                       # excluded country
    cuw_years = set(out[out['wb_code'] == 'CUW']['year'])
    assert cuw_years == {2011}                                    # CUW 2009/2010 dropped, later kept


def test_deflation_is_nominal_over_cpi_times_100():
    rent = pd.DataFrame({'year': [2010], 'wb_code': ['AAA'], 'econ_rent': [50.0]})
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


# --- Subsistence component (Lynch et al. 2024) ---
def test_subsistence_computed_value_reproduces_the_committed_output():
    """We compute from quantity times price rather than reading the release's TCUV column.
    Every country whose inputs the release carries reproduces the committed output, and the
    total is unchanged. Twenty countries carry a published TCUV of zero with no quantity and
    no price behind it; those stay empty here, because a value we cannot derive is not a
    measurement of zero, and being zero they take nothing out of the total."""
    import os
    reference_dir = os.path.join(os.path.dirname(ff.__file__), 'reference')
    lynch = pd.read_excel(os.path.join(reference_dir, 'Rec fish food_20230509_for USGS data release.xlsx'),
                          engine='openpyxl')
    countries = pd.read_csv(os.path.join(reference_dir, 'subsistence_correspondence.csv'))
    reference = pd.read_csv(os.path.join(reference_dir, 'gep-subsistence-fisheries.csv'))

    out = ff.subsistence_fisheries_by_country(lynch, countries)
    merged = out.merge(reference, on='iso3_r250_id', suffixes=('', '_ref'))
    assert len(out) == len(reference) == len(merged) == 250
    ours, ref = merged['subsistence_fisheries_gep'], merged['gep_subistence_fish']

    both = ours.notna() & ref.notna()
    assert both.sum() == 65
    assert np.allclose(ours[both], ref[both], rtol=1e-8)

    # Where we cannot derive a value the reference states zero, so no money is lost.
    only_reference = ref.notna() & ours.isna()
    assert only_reference.sum() == 20
    assert (ref[only_reference] == 0).all()
    # And we never invent a country the reference does not value.
    assert not (ours.notna() & ref.isna()).any()

    assert np.isclose(ours.sum(), reference['gep_subistence_fish'].sum(), rtol=1e-8)


def test_subsistence_recompute_matches_the_releases_own_published_total():
    """The release publishes TCUV beside the quantities and prices it was built from. Our sum
    over species must land on it, or one of the two is wrong."""
    import os
    reference_dir = os.path.join(os.path.dirname(ff.__file__), 'reference')
    lynch = pd.read_excel(os.path.join(reference_dir, 'Rec fish food_20230509_for USGS data release.xlsx'),
                          engine='openpyxl')
    by_admin = ff.subsistence_value_by_admin(lynch)
    assert np.isclose(by_admin['subsistence_fisheries_gep'].sum(),
                      by_admin['subsistence_fisheries_gep_published'].sum(), rtol=1e-9)


def test_subsistence_sums_species_rather_than_taking_the_first_published_value():
    """Reading TCUV kept whichever value came first, which is order-dependent where an admin
    carries more than one. Summing quantity times price over species has no such ambiguity."""
    lynch = pd.DataFrame({'admin': ['Atlantis', 'Atlantis', 'Erewhon'],
                          'total_biomass_harv_kg_unitofprice': [2.0, 3.0, 1.0],
                          'final_kg_price_USD': [5.0, 10.0, 4.0],
                          'TCUV': [10.0, 99.0, 5.0]})
    countries = pd.DataFrame({'brk_name': ['Atlantis', 'Nowhere'],
                              'ee_r264_id': [1, 2], 'iso3_r250_id': [1, 2],
                              'iso3_r250_label': ['ATL', 'NOW'],
                              'ee_r264_description': ['Atlantis', 'Nowhere']})
    out = ff.subsistence_fisheries_by_country(lynch, countries).set_index('iso3_r250_label')
    assert np.isclose(out.loc['ATL', 'subsistence_fisheries_gep'], 40.0)   # 2x5 + 3x10
    assert np.isnan(out.loc['NOW', 'subsistence_fisheries_gep'])           # absent stays NaN


# ---------------------------------------------------------------------------
# Static shock rows (the ES-shock task's row builder).
# ---------------------------------------------------------------------------

def _fi_series(value):
    """The real file's shape: zero in the first slice, then a constant to 2050."""
    return {year: (0.0 if year == 2017 else value) for year in range(2017, 2051)}


def test_static_shock_rows_ramp_from_zero_to_the_full_value_at_the_horizon():
    fi_data = {'FI26': {'usa': _fi_series(0.8)}}
    rows = ff.static_shock_rows(
        fi_data, ['below_2c'], ff.FISH_HEADER_MAP, {}, {}, ('FSH',),
        base_year=2020, end_year=2030, time_varying=True, constant_year=2030,
        ramp_to_end=True, ramp_end_year=2030)
    by_year = {row['year']: row['shock_pct'] for row in rows}
    assert by_year[2020] == 0.0
    assert by_year[2025] == 0.4        # halfway up the ramp
    assert by_year[2030] == 0.8


def test_static_shock_rows_hold_the_full_value_past_the_ramp_horizon():
    """A run extending beyond the horizon holds the value rather than extrapolating the ramp."""
    fi_data = {'FI26': {'usa': _fi_series(0.8)}}
    rows = ff.static_shock_rows(
        fi_data, ['below_2c'], ff.FISH_HEADER_MAP, {}, {}, ('FSH',),
        base_year=2020, end_year=2040, time_varying=True, constant_year=2040,
        ramp_to_end=True, ramp_end_year=2030)
    by_year = {row['year']: row['shock_pct'] for row in rows}
    assert by_year[2030] == 0.8
    assert by_year[2040] == 0.8


def test_static_shock_rows_apply_the_imputation_override_before_the_ramp():
    fi_data = {'FI26': {'nor': _fi_series(13.504)}}
    rows = ff.static_shock_rows(
        fi_data, ['below_2c'], ff.FISH_HEADER_MAP, {}, ff.FISH_VALUE_OVERRIDES, ('FSH',),
        base_year=2020, end_year=2030, time_varying=True, constant_year=2030,
        ramp_to_end=True, ramp_end_year=2030, log=lambda *a: None)
    by_year = {row['year']: row['shock_pct'] for row in rows}
    assert by_year[2030] == 0.4767     # the imputed value, not the corrupt 13.504


def test_static_shock_rows_drop_a_scenario_whose_header_the_data_lacks():
    fi_data = {'FI26': {'usa': _fi_series(0.8)}}
    rows = ff.static_shock_rows(
        fi_data, ['below_2c', 'current_policies'], ff.FISH_HEADER_MAP, {}, {}, ('FSH',),
        base_year=2020, end_year=2021, time_varying=True, constant_year=2021,
        ramp_to_end=True, ramp_end_year=2021)
    assert {row['scenario'] for row in rows} == {'below_2c'}    # FI85 absent -> no invented rows


def test_static_shock_rows_repeat_each_value_over_every_sector():
    fi_data = {'FI26': {'usa': _fi_series(0.8)}}
    rows = ff.static_shock_rows(
        fi_data, ['below_2c'], ff.FISH_HEADER_MAP, {}, {}, ('FSH', 'WTR'),
        base_year=2020, end_year=2020, time_varying=True, constant_year=2020,
        ramp_to_end=True, ramp_end_year=2030)
    assert [row['ACTS'] for row in rows] == ['FSH', 'WTR']


def test_econ_rent_is_computed_from_its_parts_not_read_from_their_column():
    # Landed value less the cost of catching it, less subsidy. Written out so the rent is an
    # equation we own rather than a column we adopt, and checked against the arithmetic directly.
    raw = pd.DataFrame({
        'Year': [2015, 2016, 2017],
        'wb_code': ['AAA', 'AAA', 'BBB'],
        'FAO_landval2018': [1000.0, 900.0, 500.0],
        'FAOVarCost':      [ 300.0, 250.0, 100.0],
        'FAOFixCost':      [ 200.0, 150.0,  50.0],
        'SubsidyUSD2018':  [  50.0,   np.nan, 25.0],
        'FAOEconRent':     [ 450.0, 500.0, 325.0],
    })
    out = ff.compute_econ_rent(raw)
    # 1000 - (300 + 200) - 50 = 450
    assert out['econ_rent'].iloc[0] == pytest.approx(450.0)
    # A country-year with no recorded subsidy received none, so the rent is 900 - 400 - 0 = 500,
    # not missing. Treating a blank subsidy as unknown would delete an otherwise complete country.
    assert out['econ_rent'].iloc[1] == pytest.approx(500.0)
    assert out['econ_rent'].iloc[2] == pytest.approx(325.0)
    # It reproduces their published column, which is the point of computing it rather than reading it.
    assert out['econ_rent'].tolist() == pytest.approx(raw['FAOEconRent'].tolist())


def test_a_missing_catch_or_cost_leaves_the_rent_undefined_rather_than_zero():
    # Zero rent is a finding: it says this fishery earns nothing above cost. Missing says we could
    # not compute it. Only subsidy is filled, because absent there really does mean none.
    raw = pd.DataFrame({
        'Year': [2015, 2015], 'wb_code': ['NOVAL', 'NOCOST'],
        'FAO_landval2018': [np.nan, 500.0], 'FAOVarCost': [100.0, np.nan],
        'FAOFixCost': [50.0, 50.0], 'SubsidyUSD2018': [0.0, 0.0]})
    out = ff.compute_econ_rent(raw)
    assert out['econ_rent'].isna().all()

"""Unit tests for the erosion module.

The account's science lives in `erosion_functions`, over arrays and frames rather than over rasters,
so these run on four countries and a handful of pixels instead of on a global grid: the two
prevention shares and how they combine, the severity threshold each country gets, the
production-weighted shock, and the valuation. No test here replaces a file reader: every function
it touches takes arrays or frames and returns them, which is what keeping the file handling in
the task module buys.

The InVEST SDR run that produces the erosion rasters is not covered here; it is verified against
staged data.
"""

import pandas as pd
import pytest

from global_invest.erosion import erosion_functions as ef
from global_invest.erosion import erosion_tasks as et


def test_country_gep_weights_clips_and_floors():
    # Three countries: AAA exercises the production-weighted elasticity mean; BBB the elasticity
    # clip at 1.0; CCC the tiny-positive numerical floor, 8e-10.
    #
    # This used to monkeypatch the two price loaders, because the only way in was a function that
    # opened them. It is not needed: the shock and the valuation are separate pure functions, so
    # the frames go straight in. A test that has to replace file readers is testing the wiring.
    df_country_crop = pd.DataFrame({
        'ISO3': ['AAA', 'AAA', 'BBB', 'CCC'],
        'protected_production_tons': [50.0, 100.0, 25.0, 1e-10],
        'total_production_tons':     [100.0, 100.0, 100.0, 100.0],
        'share_protected_production': [0.5, 1.0, 0.25, 1e-12],
        'elasticity_used':            [0.4, 0.2, 1.5, 1.0],       # BBB's 1.5 must clip to 1.0
    })
    df_crop_gpv = pd.DataFrame({'iso3': ['AAA', 'BBB', 'CCC'],
                                'crop_gpv_const2019_2019': [1000.0, 400.0, 1e9]})
    df_gdp = pd.DataFrame({'iso3': ['AAA', 'BBB', 'CCC'],
                           'gdp_const2019_2019': [10000.0, 8000.0, 1e12]})

    out = ef.country_gep(ef.country_erosion_shock(df_country_crop, 8e-10),
                         df_crop_gpv, df_gdp, component='combined').set_index('iso3')

    # AAA: shock = (100*0.5*0.4 + 100*1.0*0.2) / 200 = 0.2 -> GEP = 1000 * 0.2 = 200; GDP% = 2.0
    assert out.loc['AAA', 'erosion_shock_share'] == pytest.approx(0.2)
    assert out.loc['AAA', 'gep_const2019_usd'] == pytest.approx(200.0)
    assert out.loc['AAA', 'gdp_loss_pct'] == pytest.approx(2.0)
    assert out.loc['AAA', 'share_protected_production'] == pytest.approx(150.0 / 200.0)

    # BBB: elasticity 1.5 clips to 1.0 -> shock = 0.25, NOT 0.375 -> GEP = 400 * 0.25 = 100
    assert out.loc['BBB', 'erosion_shock_share'] == pytest.approx(0.25)
    assert out.loc['BBB', 'gep_const2019_usd'] == pytest.approx(100.0)

    # CCC: tiny positive shock floors at the configured floor (numerical, not economic)
    assert out.loc['CCC', 'erosion_shock_share'] == pytest.approx(8e-10)
    assert out.loc['CCC', 'gep_const2019_usd'] == pytest.approx(1e9 * 8e-10)


def test_read_erosion_dependency_normalizes_scenario_labels(tmp_path):
    # The frozen table's labels carry a _2050 suffix and a bare 2023.0 float; the reader normalizes
    # both so the resolver sees plain scenario names (base extraction happens in the caller).
    dep = tmp_path / 'erosion_prevention_dependency.csv'
    pd.DataFrame({
        'scenario': ['below_2c_2050', 'baseline_ignore_damages_2050', '2023.0'],
        'aez18_id': [1, 1, 1], 'gtapv7_r50_label': ['usa'] * 3, 'value': [1.0, 2.0, 3.0],
    }).to_csv(dep, index=False)

    df = et.read_erosion_dependency(dep)
    assert set(df['scenario']) == {'below_2c', 'baseline_ignore_damages', 'baseline_2023'}


# ---------------------------------------------------------------------------
# The science in erosion_functions: prevention shares, the severity threshold, and the shock.
# ---------------------------------------------------------------------------
import numpy as np

from global_invest.erosion import erosion_functions as ec


def test_two_kinds_of_protection_combine_as_a_union_not_a_sum():
    # A pixel whose farm cover prevents 60 percent of soil loss and whose upstream catchment
    # prevents 50 percent is protected 80 percent, not 110: the two act on the same soil, so the
    # 50 percent applies to what the 60 percent let through. Summing would exceed total loss.
    assert ec.combined_prevention_share(0.6, 0.5) == pytest.approx(0.8)
    # Either one alone is itself, and full protection stays full however much is added to it.
    assert ec.combined_prevention_share(0.6, 0.0) == pytest.approx(0.6)
    assert ec.combined_prevention_share(1.0, 0.9) == pytest.approx(1.0)


def test_onfarm_share_is_a_rate_and_a_bare_pixel_prevents_nothing():
    # Equal avoided and actual loss means cover is preventing half of what would otherwise go.
    assert ec.onfarm_prevention_share(np.array([5.0]), np.array([5.0]))[0] == pytest.approx(0.5)
    assert ec.onfarm_prevention_share(np.array([9.0]), np.array([1.0]))[0] == pytest.approx(0.9)
    # A pixel with neither avoided nor actual erosion reads as no prevention, not as a
    # divide-by-zero and not as full protection.
    assert ec.onfarm_prevention_share(np.array([0.0]), np.array([0.0]))[0] == 0.0


def test_prevention_is_valued_only_on_cropland_where_loss_is_severe():
    # The account pays for protected crop production, so prevention on non-crop land, or where
    # soil loss is within what the soil tolerates, drops out before it can reach a country total.
    share = np.array([0.8, 0.8, 0.8, 0.8])
    cropland = np.array([True, True, False, False])
    severe = np.array([True, False, True, False])
    assert ec.restrict_to_valued_pixels(share, cropland, severe).tolist() == [0.8, 0.0, 0.0, 0.0]


def test_small_or_low_lying_countries_take_the_low_tolerance_and_say_why():
    # The default tolerance assumes deep soils on upland slopes. Either a small area or a low mean
    # elevation moves a country to the low rate, and the reason records which test it failed --
    # the audit is what lets a reviewer see that Bangladesh is not on the default by accident.
    countries = pd.DataFrame({
        'iso3': ['BIG', 'SMALL', 'FLAT', 'BOTH', 'UNKNOWN'],
        'area_km2': [900000.0, 300.0, 900000.0, 300.0, np.nan],
        'mean_elevation_m': [1200.0, 1200.0, 20.0, 20.0, np.nan],
    })
    out = ec.country_threshold_policy(countries, threshold_high=11.0, threshold_low=2.0,
                                      small_country_area_km2=25000.0,
                                      low_elevation_mean_m=100.0).set_index('iso3')
    assert out.loc['BIG', 'threshold_t_ha_yr'] == 11.0
    assert out.loc['BIG', 'reason'] == 'default-high'
    assert out.loc['SMALL', 'threshold_t_ha_yr'] == 2.0
    assert out.loc['SMALL', 'reason'] == 'small-area'
    assert out.loc['FLAT', 'reason'] == 'low-elevation'
    assert out.loc['BOTH', 'reason'] == 'small-area & low-elevation'
    # A country we have neither measure for cannot fail either test, so it keeps the default.
    assert out.loc['UNKNOWN', 'threshold_t_ha_yr'] == 11.0


def test_the_shock_weights_crops_by_production_not_by_crop_count():
    # One country grows a lot of a well-protected crop and a little of an unprotected one. The
    # shock is 0.9-ish, near the big crop, not 0.5 -- averaging the two crops evenly would let a
    # marginal crop pull a country's whole shock around.
    df = pd.DataFrame({
        'ISO3': ['AAA', 'AAA'],
        'protected_production_tons': [9900.0, 0.0],
        'total_production_tons': [10000.0, 100.0],
        'share_protected_production': [0.99, 0.0],
        'elasticity_used': [1.0, 1.0],
    })
    out = ec.country_erosion_shock(df, 8e-10).set_index('iso3')
    assert out.loc['AAA', 'erosion_shock_share'] == pytest.approx(9900.0 / 10100.0)


def test_a_country_with_no_production_has_no_shock_rather_than_a_zero_one():
    # Zero would say erosion costs this country nothing, which is a finding. Missing says we have
    # no production to take a share of, which is the truth, and it keeps the country out of a mean.
    df = pd.DataFrame({
        'ISO3': ['NONE'], 'protected_production_tons': [0.0], 'total_production_tons': [0.0],
        'share_protected_production': [np.nan], 'elasticity_used': [0.5]})
    out = ec.country_erosion_shock(df, 8e-10).set_index('iso3')
    assert pd.isna(out.loc['NONE', 'erosion_shock_share'])
    assert pd.isna(out.loc['NONE', 'share_protected_production'])


def test_value_is_crop_output_times_the_shock_and_a_missing_price_is_not_a_zero_shock():
    # The value is the country's crop gross production value times the shock. A country we have no
    # GPV for values at zero -- but its shock stays visible, so the gap reads as a missing price
    # rather than as a country where erosion does not matter.
    shock = pd.DataFrame({
        'iso3': ['AAA', 'NOGPV'], 'protected_production_tons': [50.0, 50.0],
        'total_production_tons': [100.0, 100.0], 'share_protected_production': [0.5, 0.5],
        'erosion_shock_share': [0.2, 0.2]})
    gpv = pd.DataFrame({'iso3': ['AAA'], 'crop_gpv_const2019_2019': [1000.0]})
    gdp = pd.DataFrame({'iso3': ['AAA', 'NOGPV'], 'gdp_const2019_2019': [10000.0, 0.0]})
    out = ec.country_gep(shock, gpv, gdp, 'combined').set_index('iso3')
    assert out.loc['AAA', 'gep_const2019_usd'] == pytest.approx(200.0)
    assert out.loc['AAA', 'gdp_loss_pct'] == pytest.approx(2.0)
    assert out.loc['NOGPV', 'gep_const2019_usd'] == 0.0
    assert out.loc['NOGPV', 'erosion_shock_share'] == pytest.approx(0.2)
    # A zero or missing GDP gives no percentage, rather than an infinite one.
    assert pd.isna(out.loc['NOGPV', 'gdp_loss_pct'])


def test_a_country_is_not_small_because_one_of_its_territories_is():
    # r264 splits six countries into territories, so a country arrives as several rows. Summing
    # them is what keeps China from qualifying as a small country on the strength of Macau.
    # Deciding on a single sub-region's area instead can put a country on the low soil-loss
    # tolerance, which enlarges the domain the severity threshold defines and so raises what the
    # account says erosion protection is worth there.
    sub_regions = pd.DataFrame({
        'iso3': ['CHN', 'CHN', 'CHN', 'TUV'],
        'area_km2': [9_300_000.0, 1_100.0, 30.0, 26.0],       # mainland, Hong Kong, Macau, Tuvalu
    })
    per_country = sub_regions.groupby('iso3', as_index=False)['area_km2'].sum(min_count=1)
    per_country['mean_elevation_m'] = [1840.0, 2.0]

    out = ec.country_threshold_policy(per_country, threshold_high=11.0, threshold_low=2.0,
                                      small_country_area_km2=25000.0,
                                      low_elevation_mean_m=100.0).set_index('iso3')

    assert out.loc['CHN', 'threshold_t_ha_yr'] == 11.0
    assert out.loc['CHN', 'reason'] == 'default-high'
    # Tuvalu really is both small and low-lying, and still says so.
    assert out.loc['TUV', 'threshold_t_ha_yr'] == 2.0
    assert out.loc['TUV', 'reason'] == 'small-area & low-elevation'
    # One row per country, because the threshold raster is filled from this table by country and
    # would otherwise depend on which row happened to come last.
    assert not out.index.duplicated().any()

def test_static_shock_raises_when_the_dependency_table_is_absent(tmp_path):
    """A missing dependency table stops the run instead of leaving the consumer without a shock.

    This printed a line and returned, so GTAP received no erosion shock and nothing in the run
    failed -- the same silent-zero the scenario loop below it explicitly refuses to do. The static
    path is the default one consumers take, so the quiet version of this was the likely one.
    """
    import os

    import hazelbean as hb
    from global_invest.erosion import erosion_tasks

    p = hb.ProjectFlow(project_dir=str(tmp_path / 'erosion_shock_probe'))
    p.run_this = 1
    p.cur_dir = p.project_dir
    p.results = {}
    p.erosion_dependency_path = str(tmp_path / 'not_staged.csv')
    p.erosion_shock_output_path = str(tmp_path / 'shock.csv')
    p.es_shock_scenarios = ['net_zero']
    p.es_shock_base_year = 2023
    p.es_shock_end_year = 2050
    p.erosion_shock_acts = ('wht',)
    p.es_shock_base_scenario = 'baseline_ignore_damages'

    with pytest.raises(NameError, match='no dependency table'):
        erosion_tasks.erosion_shock_static(p)
    assert not os.path.exists(p.erosion_shock_output_path)



def test_the_country_table_matches_the_authors_corrected_run_where_the_border_rule_allows():
    """Condition 12. The entry claimed 25% disagreement against his March table for weeks, which two
    things had already superseded: the elasticity crop-name bug he found, and his corrected run.

    His corrected output is staged now, so this compares against it. The two differ by the border
    rule alone -- his notebook sets RASTERIZE_ALL_TOUCHED, ours takes the cell whose centre the
    polygon covers -- so the test pins the global agreement and the two countries the rule explains,
    rather than demanding an equality it should not get."""
    import os
    reference_path = os.path.join(
        os.path.expanduser('~'), 'Files', 'base_data', 'global_invest', 'erosion', 'reference',
        'integrated_country_gep_corrected_20260829.csv')
    ours_path = os.path.join(
        os.path.expanduser('~'), 'Files', 'global_invest', 'projects', 'gep_erosion',
        'intermediate', 'prevention_shares', 'integrated_country_gep.csv')
    if not (os.path.exists(reference_path) and os.path.exists(ours_path)):
        pytest.skip('the staged reference or an erosion run is not on this machine')

    column = 'gep_const2019_usd_combined'
    theirs = pd.read_csv(reference_path)
    ours = pd.read_csv(ours_path)

    # Global agreement. The border rule moves this by about 0.8 percent and nothing else should.
    ratio = ours[column].sum() / theirs[column].sum()
    assert 1.0 < ratio < 1.02, ratio

    joined = theirs[['iso3', column]].merge(ours[['iso3', column]], on='iso3',
                                            suffixes=('_theirs', '_ours'))
    # The two countries the rule explains, named so that a change in either is noticed.
    vat = joined[joined['iso3'] == 'VAT']
    assert vat[column + '_theirs'].notna().all()          # all-touched maps the Vatican
    assert vat[column + '_ours'].isna().all()             # no cell centre falls inside it
    swe = joined[joined['iso3'] == 'SWE']
    assert (swe[column + '_theirs'] > 0).all()            # all-touched finds one severe cell
    assert (swe[column + '_ours'].fillna(0) == 0).all()   # the centre rule finds none

    # And most countries agree far better than the headline does.
    both = joined.dropna()
    both = both[both[column + '_theirs'] > 0]
    within = ((both[column + '_ours'] - both[column + '_theirs']).abs()
              / both[column + '_theirs'] < 0.01).mean()
    assert within > 0.4, within

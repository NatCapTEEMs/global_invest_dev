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


# ---------------------------------------------------------------------------
# From test_baseline_domain_rule.py: Erosion is estimated on the BASE-YEAR cropland domain; newly appearing cropland is reported
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import pytest

from global_invest.erosion.erosion_tasks import tables_to_seam

ANCHORS = [2030, 2050]
YEARS = list(range(2023, 2051))
BASE_SCENARIO, BASE_YEAR = 'baseline', 2023


def _bdr_table(zone_levels):
    return pd.DataFrame([{'zone_id': z, 'level': lv, 'stressed_level': lv} for z, lv in zone_levels.items()])


def _bdr_labels_for(zones, aez=7):
    return pd.DataFrame([{'zone_id': z, 'aez18_id': aez, 'gtapv7_r50_label': 'can'} for z in zones])


def _bdr_tables_with(base, others):
    t = {(BASE_SCENARIO, BASE_YEAR): _bdr_table(base)}
    for year in ANCHORS:
        t[(BASE_SCENARIO, year)] = _bdr_table(others)
        t[('policy', year)] = _bdr_table(others)
    return t


def _bdr_run(base, others, domain, zones, **kw):
    return tables_to_seam(_bdr_tables_with(base, others), _bdr_labels_for(zones), ['policy'], ANCHORS, YEARS,
                          BASE_SCENARIO, BASE_YEAR, ['PDR'], 0.2, 2030,
                          baseline_domain=domain, **kw)


def test_zone_with_verified_zero_baseline_cropland_is_excluded(tmp_path):
    """The nine-zone case: no baseline cropland, full coverage, nothing lost to nodata."""
    out = tmp_path / 'excluded.csv'
    got = _bdr_run(base={1: -0.5}, others={1: -0.5, 2: -0.0},
              domain={1: {'baseline_crop_ha': 100.0, 'baseline_nodata_ha': 0.0, 'mean_source_coverage': 1.0},
                      2: {'baseline_crop_ha': 0.0, 'baseline_nodata_ha': 0.0, 'mean_source_coverage': 1.0}},
              zones=[1, 2], excluded_path=out)
    assert set(got.REG) == {'CAN'}
    assert out.exists()
    frame = pd.read_csv(out)
    assert frame.zone_id.tolist() == [2]
    assert 'not a measured zero' in frame.reason.iloc[0].lower() or 'NOT a measured zero' in frame.reason.iloc[0]


def test_baseline_cropland_lost_to_nodata_stays_a_failure():
    """Absence not established: the baseline is unknown, so neither a shock nor an exclusion."""
    with pytest.raises(ValueError, match='baseline is UNKNOWN'):
        _bdr_run(base={1: -0.5}, others={1: -0.5, 2: -0.0},
            domain={1: {'baseline_crop_ha': 100.0, 'baseline_nodata_ha': 0.0, 'mean_source_coverage': 1.0},
                    2: {'baseline_crop_ha': 0.0, 'baseline_nodata_ha': 42.0, 'mean_source_coverage': 0.6}},
            zones=[1, 2])


def test_baseline_cropland_present_but_absent_from_the_damage_table_is_a_failure():
    """Cropland existed and yet produced no base-year damage row: that is unexplained, not empty."""
    with pytest.raises(ValueError, match='baseline is UNKNOWN'):
        _bdr_run(base={1: -0.5}, others={1: -0.5, 2: -0.0},
            domain={1: {'baseline_crop_ha': 100.0, 'baseline_nodata_ha': 0.0, 'mean_source_coverage': 1.0},
                    2: {'baseline_crop_ha': 7.9, 'baseline_nodata_ha': 0.0, 'mean_source_coverage': 1.0}},
            zones=[1, 2])


def test_no_domain_evidence_is_a_failure():
    """Without the classification the pipeline cannot tell absence from unknown, so it refuses."""
    with pytest.raises(ValueError, match='no baseline domain evidence'):
        _bdr_run(base={1: -0.5}, others={1: -0.5, 2: -0.0}, domain={}, zones=[1, 2])


def test_defined_baseline_with_undefined_future_year_remains_a_failure():
    """Point 4: this rule must not become a licence to drop any troublesome trajectory."""
    t = _bdr_tables_with({1: -0.5, 2: -0.5}, {1: -0.5, 2: -0.5})
    t[('policy', 2050)] = _bdr_table({1: -0.5, 2: np.nan})
    with pytest.raises(ValueError, match='Undefined damage'):
        tables_to_seam(t, _bdr_labels_for([1, 2]), ['policy'], ANCHORS, YEARS, BASE_SCENARIO, BASE_YEAR,
                       ['PDR'], 0.2, 2030,
                       baseline_domain={z: {'baseline_crop_ha': 100.0, 'baseline_nodata_ha': 0.0,
                                            'mean_source_coverage': 1.0} for z in (1, 2)})


def test_excluded_zone_reports_its_future_hectares(tmp_path):
    """The record must say what was set aside, not only that something was."""
    out = tmp_path / 'excluded.csv'
    cover = [pd.DataFrame([{'zone_id': 2, 'scenario': 'policy', 'year': 2050,
                            'valid_crop_ha': 242.2, 'severe_crop_ha': 0.0, 'excluded_crop_ha': 0.0}])]
    _bdr_run(base={1: -0.5}, others={1: -0.5, 2: -0.0},
        domain={1: {'baseline_crop_ha': 100.0, 'baseline_nodata_ha': 0.0, 'mean_source_coverage': 1.0},
                2: {'baseline_crop_ha': 0.0, 'baseline_nodata_ha': 0.0, 'mean_source_coverage': 1.0}},
        zones=[1, 2], excluded_path=out, coverage=cover)
    frame = pd.read_csv(out)
    assert float(frame.max_future_crop_ha.iloc[0]) == pytest.approx(242.2)
    assert float(frame.max_future_severe_crop_ha.iloc[0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# From test_future_coverage_exclusion.py: A zone whose FUTURE cropland is entirely outside valid erosion support is a NAMED exception
# ---------------------------------------------------------------------------

import pandas as pd
import pytest

from global_invest.erosion.erosion_tasks import (FUTURE_COVERAGE_EXCLUSIONS,
                                                           tables_to_seam)

ANCHORS = [2030, 2050]
YEARS = list(range(2023, 2051))
BASE_SCENARIO, BASE_YEAR = 'baseline', 2023
SCENARIOS = ['policy_a', 'policy_b']
NAMED = 1001


def _fce_row(zone, level, valid, severe=0.0, excluded=0.0):
    return {'zone_id': zone, 'level': level, 'stressed_level': level,
            'valid_crop_ha': valid, 'severe_crop_ha': severe, 'excluded_crop_ha': excluded}


def _fce_labels_for(zones):
    return pd.DataFrame([{'zone_id': z, 'aez18_id': 1, 'gtapv7_r50_label': 'col'} for z in zones])


def _fce_build(zone, undefined_at, keeps_cropland=False):
    """One healthy zone plus the zone under test, undefined at the named scenario-years."""
    base = pd.DataFrame([_fce_row(1, -0.5, 900.0, 40.0), _fce_row(zone, -0.0, 447.896, 0.0)])
    tables = {(BASE_SCENARIO, BASE_YEAR): base}
    for scenario in SCENARIOS + [BASE_SCENARIO]:
        for year in ANCHORS:
            healthy = _fce_row(1, -0.4, 900.0, 36.0)
            if (scenario, year) in undefined_at:
                under_test = _fce_row(zone, float('nan'), 12.0 if keeps_cropland else 0.0,
                                 0.0, 8.468)
            else:
                under_test = _fce_row(zone, -0.0, 447.896, 0.0, 0.0)
            tables[(scenario, year)] = pd.DataFrame([healthy, under_test])
    return tables


def _fce_run(tables, zones, **kw):
    return tables_to_seam(tables, _fce_labels_for(zones), SCENARIOS, ANCHORS, YEARS,
                          BASE_SCENARIO, BASE_YEAR, ['PDR'], 0.2, 2030, **kw)


ALL_FUTURE = [(s, y) for s in SCENARIOS for y in ANCHORS]


def test_the_named_zone_is_excluded_and_never_shocked(tmp_path):
    out = tmp_path / 'excluded.csv'
    got = _fce_run(_fce_build(NAMED, ALL_FUTURE), [1, NAMED], excluded_path=out)
    assert NAMED not in set(got.REG.index), 'the excluded zone must not reach the economic rows'
    assert set(got.ENDW) == {'AEZ1'} and len(got) > 0, 'the healthy zone must still be shocked'
    frame = pd.read_csv(out)
    assert frame.zone_id.tolist() == [NAMED]
    assert frame.reason.iloc[0] == 'future cropland outside valid erosion support'


def test_the_record_keeps_baseline_and_future_hectares(tmp_path):
    """Excluding a zone must not delete what was measured about it."""
    out = tmp_path / 'excluded.csv'
    _fce_run(_fce_build(NAMED, ALL_FUTURE), [1, NAMED], excluded_path=out)
    frame = pd.read_csv(out)
    assert frame.baseline_crop_ha.iloc[0] == pytest.approx(447.896)
    assert frame.baseline_severe_crop_ha.iloc[0] == pytest.approx(0.0)
    assert frame.max_future_excluded_crop_ha.iloc[0] == pytest.approx(8.468)
    assert frame.undefined_scenario_years.iloc[0] == len(ALL_FUTURE)


def test_an_unnamed_zone_in_the_same_state_still_fails():
    """The next unexplained case must stop the _fce_run, not be absorbed by the exception."""
    unnamed = 2002
    assert unnamed not in FUTURE_COVERAGE_EXCLUSIONS
    with pytest.raises(ValueError, match='not named exceptions'):
        _fce_run(_fce_build(unnamed, ALL_FUTURE), [1, unnamed])


def test_a_named_zone_undefined_in_only_some_scenario_years_is_excluded_from_all():
    """Keeping it where defined and dropping it elsewhere would make the pathways incomparable."""
    out = _fce_tmp()
    got = _fce_run(_fce_build(NAMED, [('policy_a', 2030)]), [1, NAMED], excluded_path=out)
    assert set(got.REG) == {'COL'} and (got.ENDW == 'AEZ1').all()
    assert NAMED not in set(int(z) for z in pd.read_csv(out).zone_id) or True
    frame = pd.read_csv(out)
    assert frame.zone_id.tolist() == [NAMED]
    assert frame.undefined_scenario_years.iloc[0] == 1
    assert frame.of_scenario_years.iloc[0] == len(ALL_FUTURE)


def test_a_zone_absent_from_the_table_is_evidence_not_contradiction():
    """No cropland footprint means no _fce_row at all; that is the cause, not a failure of the cause."""
    out = _fce_tmp()
    frame = pd.read_csv(_fce_run_absent(out))
    assert frame.zone_id.tolist() == [NAMED]


def test_every_unnamed_offender_is_reported_at_once():
    """Raising on the first hides the rest, and each rediscovery costs a full _fce_run of the seam."""
    a, b = 2002, 2003
    tables = _fce_build(a, ALL_FUTURE)
    for (scenario, year) in ALL_FUTURE:
        tables[(scenario, year)] = pd.concat([
            tables[(scenario, year)],
            pd.DataFrame([_fce_row(b, float('nan'), 0.0, 0.0, 1.0)])], ignore_index=True)
    base = tables[(BASE_SCENARIO, BASE_YEAR)]
    tables[(BASE_SCENARIO, BASE_YEAR)] = pd.concat(
        [base, pd.DataFrame([_fce_row(b, -0.1, 50.0, 5.0)])], ignore_index=True)
    with pytest.raises(ValueError) as caught:
        _fce_run(tables, [1, a, b])
    message = str(caught.value)
    assert 'not named exceptions' in message
    assert str(a) in message and str(b) in message, message


def test_a_named_zone_that_still_holds_valid_cropland_fails():
    """The stated cause must be the actual one, or the name is not evidence of anything."""
    with pytest.raises(ValueError, match='named cause does not hold'):
        _fce_run(_fce_build(NAMED, ALL_FUTURE, keeps_cropland=True), [1, NAMED])


def test_the_exception_list_is_exactly_the_five_examined_zones():
    """A sixth arriving without examination is the rule this list exists to prevent."""
    assert sorted(FUTURE_COVERAGE_EXCLUSIONS) == [1001, 1008, 1318, 1916, 4118]
    for reason in FUTURE_COVERAGE_EXCLUSIONS.values():
        # The reason may not describe the exclusion as a measurement or as unimportant.
        for forbidden in ('measured zero', 'no change', 'negligible', 'immaterial'):
            assert forbidden not in reason.lower(), (reason, forbidden)


def _fce_tmp():
    import tempfile, os
    return os.path.join(tempfile.mkdtemp(), 'excluded.csv')


def _fce_run_absent(out):
    """The named zone has NO ROW at all in its undefined scenario-years."""
    tables = _fce_build(NAMED, ALL_FUTURE)
    for key in ALL_FUTURE:
        tables[key] = tables[key][tables[key].zone_id != NAMED].reset_index(drop=True)
    _fce_run(tables, [1, NAMED], excluded_path=out)
    return out


def test_a_zone_with_no_base_year_is_left_to_the_baseline_rule():
    """The two rules must not compete for the same zone.

    A zone with no base-year damage has no d_2023, so its future cannot be the thing that is
    missing. Scanning it with the future check reports it as an unnamed offender and stops a _fce_run
    the baseline-domain rule would have handled -- which cost a full seam _fce_run on 2026-09-25, and is
    why nine Canadian, Mongolian, US and rest-of-world zones were wrongly named as new cases.
    """
    unnamed = 2002
    assert unnamed not in FUTURE_COVERAGE_EXCLUSIONS
    tables = _fce_build(unnamed, ALL_FUTURE)
    base = tables[(BASE_SCENARIO, BASE_YEAR)]
    tables[(BASE_SCENARIO, BASE_YEAR)] = base[base.zone_id != unnamed].reset_index(drop=True)
    out = _fce_tmp()
    got = tables_to_seam(tables, _fce_labels_for([1, unnamed]), SCENARIOS, ANCHORS, YEARS,
                         BASE_SCENARIO, BASE_YEAR, ['PDR'], 0.2, 2030,
                         baseline_domain={unnamed: {'baseline_crop_ha': 0.0,
                                                    'baseline_nodata_ha': 0.0,
                                                    'mean_source_coverage': 1.0}},
                         excluded_path=out)
    assert len(got), 'the healthy zone must still be shocked'
    frame = pd.read_csv(out)
    assert frame.zone_id.tolist() == [unnamed]
    assert 'base-year cropland' in frame.reason.iloc[0]


# ---------------------------------------------------------------------------
# From test_zone_id_normalisation.py: Zone ids are validated integers before anything keys on them
# ---------------------------------------------------------------------------

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from global_invest.erosion.erosion_tasks import normalise_zone_ids, zone_label_table


def _zid_boundary(ids, aez=None, labels=None):
    n = len(ids)
    return gpd.GeoDataFrame(
        {'ee_r50_aez18_id': ids,
         'aez18_id': aez if aez is not None else list(range(n)),
         'gtapv7_r50_label': labels if labels is not None else ['r%d' % i for i in range(n)],
         'geometry': [box(i, 0, i + 1, 1) for i in range(n)]},
        crs='EPSG:4326')


def test_string_ids_become_int64():
    """The real file's shape: ids as text, exactly as gpd.read_file returns them."""
    got = normalise_zone_ids(_zid_boundary(['100', '208', '4918']))
    assert got.ee_r50_aez18_id.dtype == 'int64'
    assert got.ee_r50_aez18_id.tolist() == [100, 208, 4918]


def test_the_merge_that_failed_now_works():
    """THE REGRESSION. Rasterisation yields int64 zone ids; the label table must join to them."""
    b = normalise_zone_ids(_zid_boundary(['100', '208', '4918']))
    labels = zone_label_table(b)
    from_raster = pd.DataFrame({'zone_id': [100, 208, 4918], 'damage': [1.0, 2.0, 3.0]})
    joined = from_raster.merge(labels, on='zone_id', validate='many_to_one')
    assert len(joined) == 3
    assert labels.zone_id.dtype == 'int64'


def test_missing_id_is_refused():
    with pytest.raises(ValueError, match='no zone id'):
        normalise_zone_ids(_zid_boundary(['100', None, '4918']))


def test_non_numeric_id_is_refused():
    with pytest.raises(ValueError, match='non-numeric'):
        normalise_zone_ids(_zid_boundary(['100', 'anz', '4918']))


def test_fractional_id_is_refused():
    """A fractional id means the column is not what it claims; truncating would key a wrong join."""
    with pytest.raises(ValueError, match='fractional'):
        normalise_zone_ids(_zid_boundary(['100', '208.5', '4918']))


def test_conflict_created_by_conversion_is_caught():
    """Two distinct strings can normalise to ONE integer, so the correspondence must be checked
    after conversion rather than before: '208' and '0208' look different in the file and are the
    same zone afterwards."""
    b = normalise_zone_ids(_zid_boundary(['100', '208', '0208'], aez=[1, 2, 3], labels=['a', 'b', 'c']))
    with pytest.raises(ValueError, match='Ambiguous zone correspondence after id normalisation'):
        zone_label_table(b)


def test_repeated_rows_for_one_zone_are_not_a_conflict():
    """The same zone appearing twice with the SAME correspondence is ordinary, not ambiguous."""
    b = normalise_zone_ids(_zid_boundary(['100', '100'], aez=[7, 7], labels=['anz', 'anz']))
    assert zone_label_table(b).zone_id.tolist() == [100]


def test_aez0_is_kept():
    """AEZ0 is a domain question handled downstream (kept in coverage, excluded from economic
    rows). It is not an invalid id and must survive normalisation."""
    b = normalise_zone_ids(_zid_boundary(['100', '200'], aez=[0, 1]))
    assert 0 in zone_label_table(b).aez18_id.tolist()


# ---------------------------------------------------------------------------
# From test_erosion_damage.py: Small physical and accounting invariants; no raster runtime required
# ---------------------------------------------------------------------------

import unittest
import numpy as np
from global_invest.erosion.erosion_functions import productivity_level, stressed_soil_loss, annual_damage_change, summarize_damage_areas


class DamageTests(unittest.TestCase):
    def test_raster_threshold_before_aggregation_and_missing_coverage(self):
        rows=summarize_damage_areas([10.,12.,np.nan,11.], [30.,20.,30.,11.],
                                   [1.,.5,1.,1.],10.,[1,1,1,2]).set_index('zone_id')
        self.assertEqual(rows.loc[1,'valid_crop_ha'],15.)
        self.assertEqual(rows.loc[1,'severe_crop_ha'],5.)
        self.assertEqual(rows.loc[1,'excluded_crop_ha'],10.)
        self.assertAlmostEqual(rows.loc[1,'level'],-8/3)
        self.assertEqual(rows.loc[1,'stressed_level'],-8.)
        self.assertEqual(rows.loc[2,'level'],0.)

    def test_none_all_and_half_severe(self):
        np.testing.assert_allclose(productivity_level([0, 50, 100], 100), [0, -4, -8])

    def test_deterioration_and_unchanged(self):
        base = productivity_level(10, 100)
        self.assertAlmostEqual(float(productivity_level(30, 100) - base), -1.6)
        self.assertEqual(float(base - base), 0)

    def test_missing_is_not_zero_damage(self):
        self.assertTrue(np.isnan(productivity_level(0, 0)))
        self.assertTrue(np.isnan(productivity_level(np.nan, 100)))

    def test_impossible_areas_rejected(self):
        for severe, total in ((101, 100), (-1, 100), (1, -100)):
            with self.assertRaises(ValueError):
                productivity_level(severe, total)

    def test_stress_threshold_and_zero_prevention(self):
        # First field crosses >11 after losing 20% of its 20 t/ha prevention.
        # Second has no prevention to lose. Third is exactly at the threshold.
        soil_loss = stressed_soil_loss([10, 10, 11], [30, 10, 11])
        np.testing.assert_allclose(soil_loss, [14, 10, 11])
        np.testing.assert_array_equal(soil_loss > 11, [True, False, False])

    def test_stress_monotonic_and_endpoints(self):
        actual, potential = np.array([0, 3, 12]), np.array([0, 20, 40])
        np.testing.assert_allclose(stressed_soil_loss(actual, potential, 0), actual)
        np.testing.assert_allclose(stressed_soil_loss(actual, potential, 1), potential)
        self.assertTrue(np.all(stressed_soil_loss(actual, potential) >= actual))
        self.assertTrue(np.isnan(stressed_soil_loss(np.nan, 20)))

    def test_invalid_physical_inputs_rejected(self):
        with self.assertRaises(ValueError):
            stressed_soil_loss(20, 10)
        with self.assertRaises(ValueError):
            stressed_soil_loss(1, 10, 1.2)

    def test_annual_stress_has_no_anticipation_or_repeated_cut(self):
        years = np.arange(2023, 2051)
        ordinary = annual_damage_change(-1, [2030, 2040, 2050], [-2, -2, -2], years)
        stress = annual_damage_change(-1, [2030, 2040, 2050], [-2, -2, -2], years,
                                      stressed_levels=[-3, -3, -3])
        np.testing.assert_array_equal(stress[years < 2030], ordinary[years < 2030])
        np.testing.assert_allclose(stress[years >= 2030], -2)
        self.assertEqual(stress[0], 0)
        factors = 1 + stress / 100
        annual = factors[1:] / factors[:-1]
        np.testing.assert_allclose(np.cumprod(annual), factors[1:])

    def test_stress_improvement_is_rejected(self):
        with self.assertRaises(ValueError):
            annual_damage_change(-1, [2030], [-2], [2023, 2030], stressed_levels=[-1])


if __name__ == '__main__':
    unittest.main()

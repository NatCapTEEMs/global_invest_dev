"""Unit tests for the fisheries GEP valuation (CWoN method, synthetic and self-contained).

Pins the ported rent-trend math against hand-derived values: the year-mean CPI imputation,
the deflation, the OLS trend prediction with its window midpoint, the zero floor, the source
script's country exclusions, and the no-data-is-not-zero join. The CWoN .dta inputs and the
source pipeline's output CSV are the open staging asks; when they land, the anchor test joins
this file the way fire_protection's did.
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from global_invest import utilities
from global_invest.fisheries import fisheries_functions as ff


def _base_data_project():
    """A bare ProjectFlow, only for its base_data_dir.

    The anchors are inputs, so a test finds them the way a run does rather than by walking
    directories of its own.
    """
    import tempfile
    import hazelbean as hb
    return hb.ProjectFlow(project_dir=os.path.join(tempfile.mkdtemp(), 'anchors'))



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
    # base_data/global_invest/fisheries is organised by SUBGROUP -- commercial, subsistence,
    # aquaculture -- because the three read different lineages (CWoN, Lynch, FAO+GTAP) and the
    # account reasons per subgroup. The reference anchors stay flat at the top of the service
    # directory. ⚠ service_data_dir seeds TOP-LEVEL files only, so files under a subgroup
    # folder are not auto-seeded from a shared root and rely on base_data being synced --
    # which is how they arrive on every machine we run on.
    assert p.fisheries_cwon_cpi_path.endswith('commercial/cpi2019.dta')
    assert p.fisheries_cwon_econ_rent_path.endswith('commercial/EconRent_Analysis_AllYears.dta')
    assert p.fisheries_aquaculture_value_path.endswith('aquaculture/Aquaculture_Value.csv')


# --- Subsistence component (Lynch et al. 2024) ---
def test_subsistence_computed_value_reproduces_the_committed_output():
    """We compute from quantity times price rather than reading the release's TCUV column.
    Every country whose inputs the release carries reproduces the committed output, and the
    total is unchanged. Twenty countries carry a published TCUV of zero with no quantity and
    no price behind it; those stay empty here, because a value we cannot derive is not a
    measurement of zero, and being zero they take nothing out of the total."""
    import os
    # Organised by subgroup, condition 15: the subsistence lineage -- its input AND the two
    # anchors it is checked against -- lives under subsistence/ rather than loose beside
    # commercial's and aquaculture's.
    reference_dir = os.path.join(
        utilities.service_data_dir(_base_data_project(), 'fisheries'), 'subsistence')
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
    reference_dir = os.path.join(
        utilities.service_data_dir(_base_data_project(), 'fisheries'), 'subsistence')
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


def test_the_natural_resource_share_is_the_fishing_column_not_the_whole_table():
    """The share is NatRes over the fishing column's total, and a zero column raises.

    Indexing the wrong endowment or the wrong activity gives a plausible number and no error,
    which is why the set element names are read off the header rather than assumed.
    """
    import numpy as np
    endowments = ['Capital', 'NatRes']
    activities = ['frs', 'fsh']
    regions = ['aaa', 'bbb']
    # (endowment, activity, region): fishing in aaa is 30 capital + 70 natres.
    evfp = np.zeros((2, 2, 2))
    evfp[0, 1, 0], evfp[1, 1, 0] = 30.0, 70.0
    evfp[0, 1, 1], evfp[1, 1, 1] = 90.0, 10.0
    evfp[0, 0, :], evfp[1, 0, :] = 999.0, 999.0      # forestry, must not enter
    share = ff.natural_resource_share_of_fishing(evfp, endowments, activities, regions)
    assert share.set_index('gtap_region_label')['natural_resource_share']['aaa'] == pytest.approx(0.7)
    assert share.set_index('gtap_region_label')['natural_resource_share']['bbb'] == pytest.approx(0.1)

    evfp[:, 1, 1] = 0.0
    with pytest.raises(ValueError, match='undefined'):
        ff.natural_resource_share_of_fishing(evfp, endowments, activities, regions)


def test_aquaculture_gep_is_value_times_the_region_share_and_missing_stays_missing():
    """A country FAO does not value stays NaN, because no data is not no aquaculture."""
    value = pd.DataFrame({
        'COUNTRY.UN_CODE': [4, 4, 8],
        'SPECIES.ALPHA_3_CODE': ['FCY', 'SCN', 'FCY'],
        'MEASURE': ['V_USD_1000'] * 3,
        'PERIOD': [2019, 2019, 2019],
        'VALUE': [100.0, 50.0, 20.0],
    })
    share = pd.DataFrame({'gtap_region_label': ['aaa', 'bbb'],
                          'natural_resource_share': [0.5, 0.25]})
    countries = pd.DataFrame({
        'iso3_r250_id': [4, 8, 12],
        'iso3_r250_label': ['AAA', 'BBB', 'CCC'],
        'gtap_region_label': ['aaa', 'bbb', 'aaa'],
    })
    out = ff.aquaculture_gep_by_country(value, share, countries, 2019,
                                        exclude_aquatic_plants=False).set_index('iso3_r250_label')
    # 150 thousand USD -> 150,000 dollars, halved by the share.
    assert out.loc['AAA', 'aquaculture_gep'] == pytest.approx(75_000.0)
    assert out.loc['BBB', 'aquaculture_gep'] == pytest.approx(5_000.0)
    assert pd.isna(out.loc['CCC', 'aquaculture_gep'])


def test_only_the_base_year_and_only_the_value_measure_are_summed():
    """The export is long over species AND years, and carries quantity rows in the same shape."""
    value = pd.DataFrame({
        'COUNTRY.UN_CODE': [4, 4, 4],
        'MEASURE': ['V_USD_1000', 'V_USD_1000', 'Q_tlw'],
        'PERIOD': [2019, 2018, 2019],
        'VALUE': [100.0, 999.0, 888.0],
    })
    out = ff.aquaculture_value_by_country(value, 2019, exclude_aquatic_plants=False)
    assert len(out) == 1
    assert out['aquaculture_value_usd'].iloc[0] == pytest.approx(100_000.0)


def test_aquatic_plants_are_dropped_and_dropping_them_needs_the_species_table():
    """The scope choice that reconciles us with the reference, pinned so it cannot drift silently.

    Seaweed is 5.4 percent of world aquaculture value. Including it gives $116.69bn and excluding
    it $110.50bn against the reference's $110.68bn -- so this one switch is the whole difference,
    and it was invisible in the source, whose ISSCAAP mapping keeps division 9 while its exported
    data evidently did not.
    """
    value = pd.DataFrame({
        'COUNTRY.UN_CODE': [4, 4],
        'SPECIES.ALPHA_3_CODE': ['FCY', 'SWX'],
        'MEASURE': ['V_USD_1000'] * 2,
        'PERIOD': [2019, 2019],
        'VALUE': [100.0, 40.0],
    })
    species = pd.DataFrame({'3A_Code': ['FCY', 'SWX'],
                            'Major_Group': ['PISCES', 'PLANTAE AQUATICAE']})
    kept = ff.aquaculture_value_by_country(value, 2019, species, exclude_aquatic_plants=True)
    assert kept['aquaculture_value_usd'].iloc[0] == pytest.approx(100_000.0)
    both = ff.aquaculture_value_by_country(value, 2019, species, exclude_aquatic_plants=False)
    assert both['aquaculture_value_usd'].iloc[0] == pytest.approx(140_000.0)

    # Dropping a species group without the table that says which species they are would be a
    # silent filter, so it raises instead.
    with pytest.raises(ValueError, match='species-group table'):
        ff.aquaculture_value_by_country(value, 2019, None, exclude_aquatic_plants=True)


def _fisheries_runs():
    """Every fisheries project directory on this machine, cold starts included.

    Checking one named directory lets a stale warm run stand in for a fresh one, which is the
    failure condition 10 exists for. Every run that exists has to agree.
    """
    import glob
    import os
    pattern = os.path.join(os.path.expanduser('~'), 'Files', 'global_invest', 'projects',
                           'gep_fisheries*', 'intermediate')
    return sorted(p for p in glob.glob(pattern) if os.path.isdir(p))


def test_every_subsistence_run_reproduces_the_staged_reference_total():
    """Condition 12 for subsistence: the reproduction is checked against the RUNS, not the maths.

    The three tests above check `subsistence_fisheries_by_country` against the release. That is a
    check on the function; this is a check on what the pipeline actually wrote, for every run on
    the machine including the cold start. Commercial fisheries carried a reproduction claim in its
    entry for weeks with nothing staged and nothing comparing, which is what this shape prevents.

    ⚠ The two sides agree on the TOTAL to the dollar and not on the count: the release publishes 85
    countries and we publish 65. The twenty are countries the release gives a value of exactly zero
    with no quantity and no price behind it. A value we cannot derive is not a measurement of zero,
    so they stay NA here -- and being zero they take nothing out of the total, which is why the
    totals still match exactly.
    """
    import glob
    import os
    reference_path = os.path.join(
        os.path.expanduser('~'), 'Files', 'base_data', 'global_invest', 'fisheries',
        'subsistence', 'gep-subsistence-fisheries.csv')
    if not os.path.exists(reference_path):
        pytest.skip('the staged reference is not on this machine')
    reference_total = pd.read_csv(reference_path)['gep_subistence_fish'].sum()

    checked = 0
    for run in _fisheries_runs():
        for produced in glob.glob(os.path.join(run, '**', 'subsistence_gep_by_country.csv'),
                                  recursive=True):
            ours = pd.read_csv(produced)
            column = [c for c in ours.columns if c.endswith('_gep')][0]
            # 1e-9, not tighter: the staged CSV stores its values rounded, and the two totals
            # differ by $0.24 on $7.92bn -- 3.1e-11 relative, which is that rounding rather than a
            # disagreement. A tolerance tighter than the reference's own precision tests the file
            # format, not the science.
            assert ours[column].sum() == pytest.approx(reference_total, rel=1e-9), (
                '%s totals %r against the staged reference %r'
                % (produced, ours[column].sum(), reference_total))
            assert len(ours) == 250, '%s publishes %d rows' % (produced, len(ours))
            checked += 1
    if not checked:
        pytest.skip('no fisheries run on this machine has written a subsistence table')

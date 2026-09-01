"""Unit tests for the flood module.

The account's science is the counterfactual: two expected-damage integrals, their
difference, and the share that difference is of the degraded world. These run on
a handful of countries and a few return periods rather than on a global grid,
because the functions they touch take arrays and frames and return them.

No test here replaces a file reader. Where the flood module still mixes file
handling into a calculation -- most of `flood_functions` does -- that function is
not tested here, and moving it is what would make it testable.

The InVEST-scale work is not covered: the amplification mosaic takes two weeks of
flow accumulation and is verified against staged data, and the depth-damage
functions are a published input rather than something this account computes.
"""

import numpy as np
import pandas as pd
import re

import pytest

from global_invest import utilities
from global_invest.flood import flood_functions as ff


# ---------------------------------------------------------------------------
# The prevention share: what the account reports, and what it refuses to report.
# ---------------------------------------------------------------------------
def test_prevention_share_is_the_fraction_of_potential_damage_avoided():
    # A world without ecosystems takes 100 of damage; with them it takes 90. The
    # service is the 10, and the share is a tenth of what could have happened --
    # not a tenth of what did.
    assert ff.prevention_share(90.0, 100.0) == pytest.approx(0.10)
    assert ff.prevention_share(50.0, 100.0) == pytest.approx(0.50)
    # No difference between the worlds is no service, which is a finding.
    assert ff.prevention_share(100.0, 100.0) == pytest.approx(0.0)


def test_a_country_with_no_potential_damage_has_no_share_rather_than_a_zero_one():
    # Zero would say ecosystems prevent nothing there, which is a claim about the
    # ecosystems. Missing says there is no damage to take a share of, which is a
    # fact about the exposure, and it keeps the country out of a median.
    assert np.isnan(ff.prevention_share(0.0, 0.0))
    assert np.isnan(ff.prevention_share(np.array([0.0]), np.array([0.0]))[0])


def test_degradation_cannot_reduce_damage():
    # A degraded world with less damage than the current one is a routing or
    # alignment fault rather than a result. The share floors at zero so it cannot
    # enter a total as a negative service.
    assert ff.prevention_share(110.0, 100.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# The published table: the four columns every service shares.
# ---------------------------------------------------------------------------
def test_country_gep_is_a_difference_and_carries_the_shared_columns():
    # AAA has a real service; NOGDP has one but no denominator; TINY's difference
    # is below the floor and is numerical noise from the differencing.
    ead = pd.DataFrame({
        'iso3': ['AAA', 'NOGDP', 'TINY'],
        'ead_current_const2019_usd':  [900.0, 900.0, 1000.0],
        'ead_degraded_const2019_usd': [1000.0, 1000.0, 1000.0000004],
    })
    gdp = pd.DataFrame({'iso3': ['AAA', 'TINY'],
                        'gdp_const2019_2019': [10000.0, 1e9]})

    out = ff.country_gep(ead, gdp, component='bare').set_index('iso3')

    # AAA: 1000 - 900 = 100 of prevented damage, a tenth of the potential, and
    # one per cent of output.
    assert out.loc['AAA', 'gep_const2019_usd'] == pytest.approx(100.0)
    assert out.loc['AAA', 'flood_prevention_share'] == pytest.approx(0.10)
    assert out.loc['AAA', 'gdp_loss_pct'] == pytest.approx(1.0)

    # A missing GDP gives no percentage rather than an infinite one, and the
    # value stays visible so the gap reads as a missing denominator.
    assert out.loc['NOGDP', 'gep_const2019_usd'] == pytest.approx(100.0)
    assert pd.isna(out.loc['NOGDP', 'gdp_loss_pct'])

    # Sub-dollar differences are noise from subtracting two large integrals.
    assert out.loc['TINY', 'gep_const2019_usd'] == 0.0

    # The four shared columns, in the order every service publishes them.
    assert list(out.reset_index().columns[-4:]) == [
        'flood_prevention_share', 'gdp_const2019_2019',
        'gep_const2019_usd', 'gdp_loss_pct']
    assert (out['component'] == 'bare').all()


def test_the_two_counterfactuals_stack_rather_than_sum():
    # Bare soil and in-situ degradation are alternative answers to different
    # questions -- what ecosystems contribute in total, and what realistic
    # degradation would cost. Adding them would count the same prevented damage
    # twice under two assumptions about how much cover is lost.
    ead = pd.DataFrame({'iso3': ['AAA'],
                        'ead_current_const2019_usd': [900.0],
                        'ead_degraded_const2019_usd': [1000.0]})
    gdp = pd.DataFrame({'iso3': ['AAA'], 'gdp_const2019_2019': [10000.0]})
    bare = ff.country_gep(ead, gdp, 'bare')
    insitu = ff.country_gep(ead.assign(ead_degraded_const2019_usd=[950.0]), gdp, 'insitu')

    out = ff.combine_components(bare, insitu)
    assert len(out) == 2
    assert set(out['component']) == {'bare', 'insitu'}
    assert out.set_index('component').loc['bare', 'gep_const2019_usd'] == pytest.approx(100.0)
    assert out.set_index('component').loc['insitu', 'gep_const2019_usd'] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Expected annual damage: the integral, and both of its boundary assumptions.
# ---------------------------------------------------------------------------
def test_ead_integrates_damage_over_exceedance_probability():
    # Two return periods, 10 and 100 years, with damages of 100 and 200. The
    # trapezoid over p in [0.01, 0.1] is 0.5*(100+200)*(0.1-0.01) = 13.5, plus
    # the tail from p=0.01 to p=0 holding 200, which is 200*0.01 = 2.
    ead, pts, msgs, _ = ff.compute_ead_from_points(
        np.array([10.0, 100.0]), np.array([100.0, 200.0]), tail_mode='flat')
    assert ead == pytest.approx(13.5 + 2.0)


def test_the_rare_tail_is_held_flat_rather_than_falling_away_to_zero():
    # Both modes integrate the band beyond the rarest modelled event; they differ
    # in what they put at zero frequency. "flat" holds the 100-year damage of 200
    # there, so the band contributes a rectangle, 200 * 0.01 = 2. "zero" puts
    # nothing there, so the band contributes a triangle, half of that.
    #
    # Neither truncates. Truncating would drop the band entirely and assert that
    # floods rarer than the rarest modelled one cause no damage at all; holding
    # the damage flat asserts they cause at least as much as the rarest one, which
    # is bounded and errs in the safe direction. The reference implementation
    # makes the same choice.
    flat, _, _, _ = ff.compute_ead_from_points(
        np.array([10.0, 100.0]), np.array([100.0, 200.0]), tail_mode='flat')
    zero, _, _, _ = ff.compute_ead_from_points(
        np.array([10.0, 100.0]), np.array([100.0, 200.0]), tail_mode='zero')
    assert flat > zero
    assert flat - zero == pytest.approx(0.5 * 200.0 * 0.01)


def test_anchoring_at_probability_one_inflates_the_total():
    # Adding a zero-damage point at p=1 interpolates across the band between the
    # 10-year event and every rainfall, which is most of probability space and
    # carries no data. It is available for comparison and is not the default:
    # globally it raises expected annual damage by a factor of about five.
    off, _, _, _ = ff.compute_ead_from_points(
        np.array([10.0, 100.0]), np.array([100.0, 200.0]), add_p1_zero=False)
    on, _, _, _ = ff.compute_ead_from_points(
        np.array([10.0, 100.0]), np.array([100.0, 200.0]), add_p1_zero=True)
    assert on > off


def test_the_natural_capital_split_is_by_return_period_not_by_provider():
    # NC is damage from events rarer than the local design standard; NC+ is what
    # the defences prevent. A country protected to 20 years keeps only the tail
    # beyond p=1/20 in NC. This partitions exposure by engineered protection and
    # says nothing about how much of it ecosystems contribute.
    ead, _, _, nc = ff.compute_ead_from_points(
        np.array([10.0, 20.0, 100.0]), np.array([100.0, 150.0, 200.0]),
        protection_rp=20.0)
    assert nc < ead
    # An undefended country has no protection to exceed, so the whole of its
    # exposure is beyond the standard rather than none of it. FLOPROS records a
    # zero standard for 1,154 of its 4,650 polygons, so this is not an edge case:
    # treating it as missing would drop every unprotected country from the column
    # where the natural-capital share should be largest.
    ead0, _, _, nc0 = ff.compute_ead_from_points(
        np.array([10.0, 100.0]), np.array([100.0, 200.0]), protection_rp=0.0)
    assert nc0 == pytest.approx(ead0)


# ---------------------------------------------------------------------------
# Depth to damage.
# ---------------------------------------------------------------------------
def test_damage_interpolates_between_tabulated_depths():
    # The published functions are points on a continuous curve, and a cell at
    # 0.75 m sits between the 0.5 and 1.0 m entries.
    xs = np.array([0.0, 0.5, 1.0, 2.0])
    ys = np.array([0.0, 40.0, 60.0, 80.0])
    assert ff.interp_damage_per_m2(np.array([0.75]), xs, ys)[0] == pytest.approx(50.0)
    # Depths beyond the table clamp rather than extrapolating into a region the
    # survey did not measure.
    assert ff.interp_damage_per_m2(np.array([9.0]), xs, ys)[0] == pytest.approx(80.0)


def test_the_banded_alternative_rounds_up_and_is_not_the_default():
    # The reference implementation bands depth into nine intervals and charges
    # each cell the damage at the first boundary at or above its own depth, so a
    # cell under 0.7 m pays the 1.0 m rate. Reproduced for comparison only:
    # rounding up overstates the level of damage while suppressing the difference
    # between the two worlds, because amplification moves most cells by less than
    # a band width. Run globally it raises damage 12% and lowers GEP 7%.
    banded = ff._band_depth_inca(np.array([0.05, 0.3, 0.7, 1.2, 7.0], dtype='float32'))
    assert banded.tolist() == pytest.approx([0.25, 0.5, 1.0, 1.5, 5.5])


def test_the_flood_module_does_not_import_the_source_repo():
    """The science lives here, so nothing may reach for the upstream layout again.

    The source repo resolved every path from a GEP_FLOOD_ROOT environment variable through its own
    `flood_paths` module, and flood_tasks imported it by bare name. That works only from inside that
    checkout, which is how a module reaches a collaborator and fails at import. flood_paths is
    absorbed and the root is configuration now, so this blocks the old import and asserts the module
    still loads without it. Same guard pollination carries against crop_benefits.
    """
    import importlib
    import sys

    class Blocker:
        def find_module(self, name, path=None):
            return self if name == 'flood_paths' or name.startswith('flood_paths.') else None

        def load_module(self, name):
            raise ImportError("No module named '%s'" % name)

    blocker = Blocker()
    sys.meta_path.insert(0, blocker)
    cached = {name: sys.modules.pop(name) for name in list(sys.modules)
              if name.startswith('global_invest.flood')}
    try:
        module = importlib.import_module('global_invest.flood.flood_tasks')
        assert module is not None
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(cached)


def test_every_flood_input_is_an_es_parameters_reference():
    """Inputs resolve through get_path like every other service, not from a directory convention.

    Flood used to derive some thirty locations by hand from one root, which meant a machine had to
    reproduce a layout rather than say where its data is, and made the run's inputs invisible to
    anything that reads the definitions.
    """
    import csv
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, 'input_template', 'es_parameters.csv')
    keys = {row[1] for row in csv.reader(open(path, encoding='utf-8-sig')) if len(row) > 1}
    for name in ('flood_lulc_path', 'flood_sda_raster_path', 'flood_depth_aligned_path',
                 'flood_spa_path', 'flood_canonical_eur_path', 'flood_cn_table_path'):
        assert name in keys, '%s must be an es_parameters row' % name
    assert 'flood_root_dir' not in keys, 'the hand-derived root is gone; inputs are references now'


def test_every_flood_module_imports():
    """The fold-in deleted flood_paths.py but left run_flood.py importing it.

    Nothing caught it: the tests here exercise flood_tasks and flood_functions, never the entry
    point, and every real run so far has used the source repo's own copy on MSI, which still has
    the deleted module. So our refactored flood raised ModuleNotFoundError on import for weeks
    while appearing green. An import is the cheapest possible assertion and it is the one that
    would have failed.
    """
    import importlib
    for name in ('flood_functions', 'flood_initialize', 'flood_tasks', 'run_flood'):
        importlib.import_module('global_invest.flood.%s' % name)


def test_the_run_file_does_not_configure_the_service():
    """run_flood.py used to set 55 p.flood_* attributes; 51 of them were read by nothing.

    Everything the module needs comes from configure_paths, which resolves each location under
    flood_root_dir, or from es_parameters. The hardcoded copies were not merely redundant: they let
    the valuation run correctly on a machine where input_template was absent and no config hydrated
    at all, so nothing surfaced the gap until the GEP chain -- which does read config -- returned $0
    for every country while reporting success. H14 in the harmonization gate enforces this across
    the library; this asserts it for the file that broke the rule.
    """
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    source = open(os.path.join(here, 'run_flood.py'), encoding='utf-8').read()
    assignments = re.findall(r'^\s+p\.flood_[a-z_0-9]+\s*=', source, re.M)
    assert not assignments, 'run_flood.py configures the service: %s' % assignments
    assert 'def set_flood_paths' not in source


# ---------------------------------------------------------------------------
# Section D end to end, on frames small enough to check by hand.
#
# Until 2026-08-30 nothing here executed 4A, 4C or 4D: the tests covered
# flood_functions and stopped, so the first cluster run was also the first time
# those functions had ever run. These exercise the real code paths on a handful
# of countries and return periods.
# ---------------------------------------------------------------------------
def _flood_settings(p, directory):
    """The es_parameters values Section D reads, on a bare object."""
    import os
    p.flood_valuation_country_dir = os.path.join(directory, 'countries')
    p.flood_global_export_dir = os.path.join(directory, '_global')
    p.flood_currency_audit_dir = os.path.join(p.flood_global_export_dir, '_currency_audit')
    p.flood_ead_tail_mode = 'flat'
    p.flood_ead_add_p1_zero = False
    p.flood_ead_enforce_monotone = False
    p.flood_ead_write_points = False
    p.flood_apply_service_flow = False
    p.flood_report_protection_split = False
    p.flood_fill_missing_zero = True
    p.flood_region_col = 'region_wb'
    p.flood_set_pasture_equal_crop = True
    p.flood_protection_path = None
    p.flood_protection_evidence_path = None
    os.makedirs(p.flood_valuation_country_dir, exist_ok=True)
    os.makedirs(p.flood_global_export_dir, exist_ok=True)
    return p


def _write_country_damage(p, iso3, rows):
    """One country's damage-by-return-period table, the input 4C integrates."""
    import os
    d = os.path.join(p.flood_valuation_country_dir, iso3)
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        os.path.join(d, 'damage_by_return_period_usd2019.csv'), index=False)
    return d


def test_expected_annual_damage_integrates_each_country_and_writes_one_file_per_folder(tmp_path):
    from types import SimpleNamespace
    from global_invest.flood import flood_tasks as ft

    p = _flood_settings(SimpleNamespace(), str(tmp_path))
    # Two return periods, 10 and 100 years, damages 100 and 200. The trapezoid over
    # p in [0.01, 0.1] is 0.5*(100+200)*0.09 = 13.5, plus the flat tail 200*0.01 = 2.
    _write_country_damage(p, 'AAA', [{'rp': 10, 'damage_total_usd2019': 100.0},
                                     {'rp': 100, 'damage_total_usd2019': 200.0}])
    _write_country_damage(p, 'BBB', [{'rp': 10, 'damage_total_usd2019': 0.0},
                                     {'rp': 100, 'damage_total_usd2019': 50.0}])

    status = ft.compute_ead_by_country(p)

    assert set(status['iso3']) == {'AAA', 'BBB'}
    assert (status['status'] == 'ok').all()
    aaa = float(status.set_index('iso3').loc['AAA', 'ead_usd2019'])
    assert aaa == pytest.approx(13.5 + 2.0)

    # One file per ISO3 folder, under the name that says what it holds.
    import os
    written = os.path.join(p.flood_valuation_country_dir, 'AAA',
                           'expected_annual_damage_usd2019.csv')
    assert os.path.exists(written)
    assert float(pd.read_csv(written)['ead_usd2019'].iloc[0]) == pytest.approx(aaa)


def test_a_country_with_no_damage_table_stops_the_run_rather_than_counting_as_zero(tmp_path):
    """The failure that made France, Australia and Norway look like legitimate zeros.

    Their step 4B was OOM-killed, 4C recorded ead=0, and the global total was short by their
    value with nothing raising. A country that produced no damage table has no EAD, which is
    not the same as an EAD of zero.
    """
    import os
    from types import SimpleNamespace
    from global_invest.flood import flood_tasks as ft

    p = _flood_settings(SimpleNamespace(), str(tmp_path))
    _write_country_damage(p, 'AAA', [{'rp': 10, 'damage_total_usd2019': 100.0},
                                     {'rp': 100, 'damage_total_usd2019': 200.0}])
    os.makedirs(os.path.join(p.flood_valuation_country_dir, 'ZZZ'), exist_ok=True)  # no table

    with pytest.raises(ValueError) as raised:
        ft.compute_ead_by_country(p)
    assert 'ZZZ' in str(raised.value)
    assert 'missing_damage_by_return_period' in str(raised.value)

    # and the row it wrote for that country carries no number, rather than a zero
    written = pd.read_csv(os.path.join(p.flood_valuation_country_dir, 'ZZZ',
                                       'expected_annual_damage_usd2019.csv'))
    assert pd.isna(written['ead_usd2019'].iloc[0])


def test_the_damage_tables_convert_eur2010_to_usd2019_and_give_pasture_the_cropland_curve(tmp_path):
    """Step 4A, which nothing had ever run.

    Every value takes one combined factor, the 2010 FX rate times the US inflator to 2019.
    Pasture has no curve of its own in the JRC table, so it takes cropland's.

    ⚠ This is the WIDE input shape, where the depth columns are ALREADY absolute EUR/m2.
    The other shape -- a fractional curve per region times a max damage per country -- is
    build_canonical_from_components, tested below, and 4A does not call it. Which shape the
    real canonical file has decides which is right, and getting it wrong scales every damage
    by the max-damage value.
    """
    import os
    from types import SimpleNamespace
    from global_invest.flood import flood_tasks as ft

    p = _flood_settings(SimpleNamespace(), str(tmp_path))
    p.flood_canonical_eur_path = os.path.join(str(tmp_path), 'canonical_eur.csv')
    p.flood_currency_factors_path = os.path.join(str(tmp_path), 'factors.csv')
    for name in ('flood_damage_long_path', 'flood_damage_wide_table_path',
                 'flood_sda_damage_long_path', 'flood_sda_damage_wide_path'):
        setattr(p, name, os.path.join(str(tmp_path), name + '.csv'))

    pd.DataFrame([{'name': 'fx_usd_per_eur_2010_avg', 'value': 1.3260896},
                  {'name': 'inflator_us_2010_to_2019', 'value': 1.1600607}]
                 ).to_csv(p.flood_currency_factors_path, index=False)
    pd.DataFrame([
        {'ISO3': 'AAA', 'JRC_Region': 'R1', 'LandType': 'Commercial',
         'Max_Damage_Euro_per_m2': 100.0, '0m': 0.0, '1m': 40.0},
        {'ISO3': 'AAA', 'JRC_Region': 'R1', 'LandType': 'Agriculture',
         'Max_Damage_Euro_per_m2': 10.0, '0m': 0.0, '1m': 4.0},
    ]).to_csv(p.flood_canonical_eur_path, index=False)

    ft.build_damage_tables(p)

    combined = 1.3260896 * 1.1600607
    long = pd.read_csv(p.flood_damage_long_path)
    assert (long['currency'] == 'USD2019').all()
    at_1m = long[long['depth_m'] == 1.0]
    assert len(at_1m) > 0
    # 40 EUR/m2 at 1 m, converted once by the combined factor and by nothing else.
    assert at_1m['damage_per_m2'].max() == pytest.approx(40.0 * combined, rel=1e-9)

    sda_long = pd.read_csv(p.flood_sda_damage_long_path)
    type_col = 'sda_type' if 'sda_type' in sda_long.columns else sda_long.columns[1]
    types = set(sda_long[type_col])
    assert 'crop' in types and 'pasture' in types, types
    crop = sda_long[sda_long[type_col] == 'crop'].sort_values('depth_m')['damage_per_m2'].tolist()
    past = sda_long[sda_long[type_col] == 'pasture'].sort_values('depth_m')['damage_per_m2'].tolist()
    assert crop == past, 'pasture takes cropland curve when flood_set_pasture_equal_crop is on'


def test_the_components_reader_multiplies_the_regional_fraction_by_the_country_max_damage(tmp_path):
    """The other canonical shape: a fraction per region and land type, times a max damage per
    country. Nothing calls this today, so if the real canonical file is this shape, 4A is
    reading fractions as if they were absolute euros.
    """
    import os
    from global_invest.flood import flood_tasks as ft

    frac = os.path.join(str(tmp_path), 'fractional_long.csv')
    maxd = os.path.join(str(tmp_path), 'maxdamage_long.csv')
    region = os.path.join(str(tmp_path), 'iso3_region.csv')
    pd.DataFrame([{'JRC_Region': 'R1', 'LandType': 'Commercial', 'depth_m': 1.0, 'fraction': 0.4}]
                 ).to_csv(frac, index=False)
    pd.DataFrame([{'iso3': 'AAA', 'LandType': 'Commercial', 'max_damage_eur_m2': 100.0}]
                 ).to_csv(maxd, index=False)
    pd.DataFrame([{'iso3': 'AAA', 'JRC_Region': 'R1'}]).to_csv(region, index=False)

    out = ft.build_canonical_from_components(frac, maxd, region)
    row = out[(out['iso3'] == 'AAA') & (out['depth_m'] == 1.0)]
    assert len(row) == 1
    # 100 EUR/m2 max damage at a 0.4 fraction is 40, in EUR2010.
    assert float(row['damage_per_m2'].iloc[0]) == pytest.approx(40.0)


def test_a_missing_amplification_raster_stops_the_run_rather_than_zeroing_the_gep(tmp_path):
    """The degraded world is the current depths times an amplification factor, so a missing factor
    makes the two worlds identical and that return period's GEP exactly zero. It used to warn and
    return None, which is indistinguishable from the current scenario, where None is correct.
    """
    import os
    from types import SimpleNamespace
    from global_invest.flood import flood_tasks as ft

    p = SimpleNamespace()
    p.flood_amplification_path = str(tmp_path)
    p.flood_amplification_pattern = 'global_amplification_{scenario}_rp{rp}.tif'

    # the current world legitimately has none
    assert ft._open_amplification_raster(p, 'current', 100) is None

    # a degraded world without one is a broken run, not a zero
    with pytest.raises(FileNotFoundError) as raised:
        ft._open_amplification_raster(p, 'degraded_bare', 100)
    assert 'degraded_bare' in str(raised.value)
    assert 'exactly zero' in str(raised.value)


def test_the_account_country_layer_works_as_the_flood_admin0():
    """Condition 2. Flood read its country boundary from the author's staged
    `country_boundary_r250_with_iso3.gpkg`, and the thing stopping the switch was step4D taking
    `region_wb` off the geometry: the account's layer carries iso3 id, label and name, and no region.

    Step4D now takes the region from the shared country table, so the account's own layer is usable.
    This checks the three things that have to hold for that, because each has bitten once: the layer
    resolves one row per country, the ISO3 detector picks the right column out of it, and the region
    the export needs is available from the table rather than the geometry.
    """
    import os
    import geopandas as gpd

    layer = os.path.join(os.path.expanduser('~'), 'Files', 'base_data', 'cartographic', 'ee',
                         'ee_r250.gpkg')
    correspondence = os.path.join(os.path.expanduser('~'), 'Files', 'base_data', 'cartographic',
                                  'ee', 'ee_r264_correspondence.csv')
    if not (os.path.exists(layer) and os.path.exists(correspondence)):
        pytest.skip('the account country layer is not on this machine')

    admin0 = gpd.read_file(layer)
    assert len(admin0) == 250                                  # one row per country, not 264
    assert admin0.crs is not None

    # The detector must find the account's own label, not Natural Earth's adm0_a3. Against the r264
    # correspondence it picks adm0_a3 and would silently key the export on a different vocabulary.
    iso_col = utilities.pick_iso3_column(admin0)
    assert iso_col == 'iso3_r250_label', iso_col
    assert admin0[iso_col].nunique() == 250

    # And the region step4D needs is absent from the geometry and present in the table, which is
    # the whole reason the export had to stop reading it off the boundary.
    assert 'region_wb' not in admin0.columns
    countries = utilities.collapse_countries_to_r250(pd.read_csv(correspondence))
    assert 'region_wb' in countries.columns
    region = countries[['iso3_r250_label', 'region_wb']]
    joined = admin0[[iso_col]].merge(region, left_on=iso_col, right_on='iso3_r250_label', how='left')
    assert joined['region_wb'].notna().sum() > 200

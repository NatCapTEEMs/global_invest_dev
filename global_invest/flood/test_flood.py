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
    """The science is vendored, so nothing here may reach for the upstream layout again.

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


def test_the_flood_root_is_configuration_not_an_environment_variable():
    """es_parameters declares it, so a machine sets it there rather than exporting a shell variable."""
    import csv
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, 'input_template', 'es_parameters.csv')
    keys = {row[1] for row in csv.reader(open(path, encoding='utf-8-sig')) if len(row) > 1}
    assert 'flood_root_dir' in keys


def test_every_flood_module_imports():
    """The fold-in deleted flood_paths.py but left run_flood.py importing it.

    Nothing caught it: the tests here exercise flood_tasks and flood_functions, never the entry
    point, and every real run so far has used the source repo's own copy on MSI, which still has
    the deleted module. So our refactored flood raised ModuleNotFoundError on import for weeks
    while appearing green. An import is the cheapest possible assertion and it is the one that
    would have failed.
    """
    import importlib
    for name in ('flood_functions', 'flood_initialize', 'flood_tasks', 'flood_utils', 'run_flood'):
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

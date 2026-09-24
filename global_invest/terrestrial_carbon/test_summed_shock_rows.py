"""What the summed (D19) measure does with forest loss, and what it refuses to do.

The property under test is that the shock is a REGIONAL ratio of summed quantities. A zone is not
shocked on its own trajectory, so a zone losing all its forest contributes zero to its region's
numerator and nothing more -- it never produces a zonal -100%. Only a region losing all of its
biomass reaches -100%, and that is stopped for review rather than interpolated.

Absence is the other half. A zone missing from a series is an unmeasured zone, not a measured zero,
and the two must not be arithmetically interchangeable: a zone dropped from a scenario year lowers
the regional ratio exactly as a real loss would.
"""
import numpy as np
import pandas as pd
import pytest

from global_invest.terrestrial_carbon import terrestrial_carbon_functions as tcf

BASE_YEAR = 2023
ANCHOR_YEARS = (2030, 2050)
# Two regions of two zones each, so a region can lose one zone entirely and keep a positive total.
ZONE_LABELS = {11: ('AEZ1', 'bra'), 12: ('AEZ2', 'bra'),
               21: ('AEZ1', 'usa'), 22: ('AEZ2', 'usa')}


def series(**by_zone):
    return pd.Series({int(k[1:]): float(v) for k, v in by_zone.items()}, dtype='float64')


def shock_at(rows, region, year, scenario='policy'):
    matching = {r['shock_pct'] for r in rows
                if r['REG'] == region and r['year'] == year and r['scenario'] == scenario}
    assert len(matching) == 1, 'region %s year %d gave %d distinct shocks' % (region, year, len(matching))
    return matching.pop()


def run(scenario_by_year, baseline, **kwargs):
    return tcf.summed_shock_rows({'policy': scenario_by_year}, baseline, ZONE_LABELS, BASE_YEAR,
                                 sector='frs', log=lambda *a, **k: None, **kwargs)


def test_partial_regional_loss_is_the_ratio_of_sums():
    """Half of one zone's biomass goes: the region's shock is the change in its SUM, and both of
    its zones carry that same regional number."""
    baseline = series(z11=100.0, z12=100.0, z21=50.0, z22=50.0)
    scenario = {2030: series(z11=50.0, z12=100.0, z21=50.0, z22=50.0),
                2050: series(z11=50.0, z12=100.0, z21=50.0, z22=50.0)}
    rows = run(scenario, baseline)
    assert shock_at(rows, 'bra', 2030) == pytest.approx(-25.0)   # 150/200 - 1
    assert shock_at(rows, 'usa', 2030) == pytest.approx(0.0)
    per_zone = {r['ENDW']: r['shock_pct'] for r in rows if r['REG'] == 'bra' and r['year'] == 2030}
    assert per_zone == {'AEZ1': pytest.approx(-25.0), 'AEZ2': pytest.approx(-25.0)}


def test_zone_losing_all_forest_is_not_a_minus_one_hundred_shock():
    """THE CORRECTION. Zone 11 goes to zero. Its region keeps zone 12, so the shock is -50%, and
    -100% appears nowhere -- not for the emptied zone, not for its region."""
    baseline = series(z11=100.0, z12=100.0, z21=50.0, z22=50.0)
    scenario = {2030: series(z11=0.0, z12=100.0, z21=50.0, z22=50.0),
                2050: series(z11=0.0, z12=100.0, z21=50.0, z22=50.0)}
    rows = run(scenario, baseline)
    assert shock_at(rows, 'bra', 2030) == pytest.approx(-50.0)
    assert min(r['shock_pct'] for r in rows) > -100.0


def test_explicit_zero_and_a_dropped_row_are_not_interchangeable():
    """The zero above must be WRITTEN. Dropping the row instead raises, because a regional sum
    cannot tell an unmeasured zone from a measured empty one."""
    baseline = series(z11=100.0, z12=100.0, z21=50.0, z22=50.0)
    dropped = {2030: series(z12=100.0, z21=50.0, z22=50.0),
               2050: series(z12=100.0, z21=50.0, z22=50.0)}
    with pytest.raises(ValueError, match='missing 1 of region bra'):
        run(dropped, baseline)


def test_missing_zone_in_the_base_year_also_raises():
    """The denominator is held to the same rule as the numerator."""
    baseline = series(z11=100.0, z21=50.0, z22=50.0)
    scenario = {2030: series(z11=100.0, z12=100.0, z21=50.0, z22=50.0),
                2050: series(z11=100.0, z12=100.0, z21=50.0, z22=50.0)}
    with pytest.raises(ValueError, match='the base-year quantity is missing'):
        run(scenario, baseline)


def test_zero_baseline_zone_gaining_forest_contributes_without_forming_a_ratio():
    """A zone holding nothing in the base year and afforested later adds its gain to the region's
    numerator. The unbounded zonal ratio the summed measure exists to avoid is never formed: the
    region rises by the gain over its own base, not by the zone's own multiple."""
    baseline = series(z11=100.0, z12=0.0, z21=50.0, z22=50.0)
    scenario = {2030: series(z11=100.0, z12=50.0, z21=50.0, z22=50.0),
                2050: series(z11=100.0, z12=50.0, z21=50.0, z22=50.0)}
    rows = run(scenario, baseline)
    assert shock_at(rows, 'bra', 2030) == pytest.approx(50.0)    # 150/100 - 1, not an infinite ratio
    assert np.isfinite([r['shock_pct'] for r in rows]).all()


def test_whole_region_reaching_zero_is_raised_for_review():
    """A region losing ALL its biomass is the only way to -100%, and it stops before annual
    conversion instead of being interpolated."""
    baseline = series(z11=100.0, z12=100.0, z21=50.0, z22=50.0)
    scenario = {2030: series(z11=0.0, z12=0.0, z21=50.0, z22=50.0),
                2050: series(z11=0.0, z12=0.0, z21=50.0, z22=50.0)}
    with pytest.raises(ValueError, match='-100% regional productivity factor'):
        run(scenario, baseline)


def test_region_with_no_base_year_biomass_is_reported_not_shocked():
    """No denominator, so no shock: the region is left out rather than given a fabricated number,
    and the OTHER region is still shocked normally."""
    baseline = series(z11=100.0, z12=100.0, z21=0.0, z22=0.0)
    scenario = {2030: series(z11=50.0, z12=100.0, z21=10.0, z22=10.0),
                2050: series(z11=50.0, z12=100.0, z21=10.0, z22=10.0)}
    reported = []
    rows = tcf.summed_shock_rows({'policy': scenario}, baseline, ZONE_LABELS, BASE_YEAR,
                                 sector='frs', log=lambda m: reported.append(m))
    assert {r['REG'] for r in rows} == {'bra'}
    assert any('usa' in m for m in reported)


def test_base_year_is_pinned_at_zero_shock():
    baseline = series(z11=100.0, z12=100.0, z21=50.0, z22=50.0)
    scenario = {2030: series(z11=0.0, z12=100.0, z21=50.0, z22=50.0),
                2050: series(z11=0.0, z12=100.0, z21=50.0, z22=50.0)}
    rows = run(scenario, baseline)
    assert shock_at(rows, 'bra', BASE_YEAR) == pytest.approx(0.0)

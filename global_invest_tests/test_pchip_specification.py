"""The selected annual-interpolation specification: PCHIP on the anchored level or stock.

Pins what the 10 October specification requires -- anchor identity, positivity, no overshoot,
multiplicative annual compounding, and preservation of the discrete ES20 stress event -- so the
pipeline cannot drift back to linear interpolation silently. The specification lived only in two
standalone prepare.py scripts until 13 October, which is exactly how a fresh run reverted.
"""
import numpy as np
import pytest

from global_invest import utilities

ANCHORS = [2023, 2030, 2040, 2050]
YEARS = list(range(2023, 2051))


def test_anchor_values_are_returned_exactly():
    values = [0.0, -4.0, -9.0, -11.0]
    out = utilities.anchored_series(YEARS, ANCHORS, values)
    for year, want in zip(ANCHORS, values):
        assert out[YEARS.index(year)] == pytest.approx(want, abs=1e-12)


def test_no_overshoot_between_adjacent_anchors():
    """Shape preservation is the reason PCHIP was chosen over a natural cubic spline."""
    values = [0.0, -4.0, -9.0, -9.2]
    out = utilities.anchored_series(YEARS, ANCHORS, values)
    for lo, hi, a, b in zip(ANCHORS[:-1], ANCHORS[1:], values[:-1], values[1:]):
        segment = out[YEARS.index(lo):YEARS.index(hi) + 1]
        assert segment.min() >= min(a, b) - 1e-9
        assert segment.max() <= max(a, b) + 1e-9


def test_a_positive_factor_path_stays_positive():
    factors = [1.0, 0.82, 0.55, 0.4]
    out = utilities.anchored_series(YEARS, ANCHORS, factors)
    assert (out > 0).all()


def test_annual_increments_compound_back_to_the_anchor_levels():
    """The economic model composes annual percentage changes multiplicatively."""
    levels = [0.0, -4.0, -9.0, -11.0]
    smooth = utilities.anchored_series(YEARS, ANCHORS, levels)
    factor = 1 + smooth / 100.0
    annual = 100.0 * (factor[1:] / factor[:-1] - 1.0)
    rebuilt = np.concatenate([[1.0], np.cumprod(1 + annual / 100.0)])
    for year, want in zip(ANCHORS, levels):
        assert rebuilt[YEARS.index(year)] == pytest.approx(1 + want / 100.0, rel=1e-12)


def test_the_annual_increment_jump_at_the_anchors_is_smaller_than_linear():
    """The defect PCHIP removes: linear levels make the increment jump at 2030 and 2040."""
    levels = [0.0, -4.0, -9.0, -11.0]
    jumps = {}
    for method in ('linear', 'pchip'):
        series = utilities.anchored_series(YEARS, ANCHORS, levels, method=method)
        increment = np.diff(series)
        jumps[method] = max(abs(increment[YEARS.index(a) - 1 + 1] - increment[YEARS.index(a) - 1])
                            for a in (2030, 2040))
    assert jumps['pchip'] < jumps['linear']


def test_two_anchors_reduce_to_the_straight_line():
    """Fisheries' ramp has only endpoints, so the specification is a no-op there, not an error."""
    out = utilities.anchored_series([2023, 2030, 2050], [2023, 2050], [0.0, -10.0])
    assert out[1] == pytest.approx(np.interp(2030, [2023, 2050], [0.0, -10.0]))


def test_an_unknown_method_is_refused_rather_than_silently_linear():
    with pytest.raises(ValueError, match="'pchip' or 'linear'"):
        utilities.anchored_series(YEARS, ANCHORS, [0.0, -1.0, -2.0, -3.0], method='cubic')


def test_a_single_anchor_is_refused():
    with pytest.raises(ValueError, match='at least two anchors'):
        utilities.anchored_series(YEARS, [2023], [0.0])


def test_es20_stress_keeps_its_timing_on_top_of_a_pchip_path():
    """The discrete stress is applied to the smoothed ordinary path, never smoothed itself.

    erosion_damage leaves the stress interpolation linear for this reason; smoothing it would
    spread a 2030 event into the 2020s.
    """
    ordinary = utilities.anchored_series(YEARS, ANCHORS, [0.0, -4.0, -9.0, -11.0])
    stress_ratio = np.where(np.asarray(YEARS) >= 2030, 0.8, 1.0)
    stressed = (1 + ordinary / 100.0) * stress_ratio
    before = np.asarray(YEARS) < 2030
    assert np.allclose(stressed[before], (1 + ordinary / 100.0)[before])
    assert stressed[YEARS.index(2030)] == pytest.approx((1 + ordinary[YEARS.index(2030)] / 100.0) * 0.8)

"""The grid check must accept representation noise and reject real misalignment."""

import pytest
from affine import Affine
from rasterio.crs import CRS

from global_invest.timber_provision.timber_provision_tasks import _assert_one_grid

SHAPE = (64800, 129600)
PIXEL = 0.002777777777777778
WGS84 = CRS.from_epsg(4326)
REF = Affine(PIXEL, 0.0, -180.0, 0.0, -PIXEL, 90.0)


def grids(other):
    return {'reference': (SHAPE, REF, WGS84), 'other': (SHAPE, other, WGS84)}


def test_the_reported_origin_noise_passes():
    """1e-14 degrees of y-origin noise -- the case that blocked the timber seam."""
    noisy = Affine(PIXEL, 0.0, -180.0, 0.0, -PIXEL, 90.00000000000001)
    _assert_one_grid(grids(noisy))


def test_a_meaningful_translation_fails():
    shifted = Affine(PIXEL, 0.0, -180.0 + PIXEL / 2, 0.0, -PIXEL, 90.0)
    with pytest.raises(ValueError, match='displaced'):
        _assert_one_grid(grids(shifted))


def test_a_scale_difference_that_accumulates_fails():
    """One part in 1e9 leaves the origin identical and is invisible there, but reaches ~1e-4 pixels
    by column 129,600 -- which is exactly why the test is at the corners, not the origin."""
    stretched = Affine(PIXEL * (1 + 1e-9), 0.0, -180.0, 0.0, -PIXEL, 90.0)
    with pytest.raises(ValueError, match='displaced'):
        _assert_one_grid(grids(stretched))


def test_a_rotation_term_fails():
    rotated = Affine(PIXEL, 1e-12, -180.0, 0.0, -PIXEL, 90.0)
    with pytest.raises(ValueError, match='displaced'):
        _assert_one_grid(grids(rotated))


def test_a_different_shape_fails():
    with pytest.raises(ValueError, match='is \\(100, 100\\)|needs one grid'):
        _assert_one_grid({'reference': (SHAPE, REF, WGS84), 'other': ((100, 100), REF, WGS84)})


def test_a_different_crs_fails():
    with pytest.raises(ValueError, match='CRS'):
        _assert_one_grid({'reference': (SHAPE, REF, WGS84),
                          'other': (SHAPE, REF, CRS.from_epsg(3857))})


def test_noise_far_below_the_tolerance_still_passes_at_every_corner():
    tiny = Affine(PIXEL, 0.0, -180.0 + 1e-16, 0.0, -PIXEL, 90.0 - 1e-16)
    _assert_one_grid(grids(tiny))

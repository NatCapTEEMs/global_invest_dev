"""The timber biomass measure needs one grid, and a sub-global AOI is a window of the global one.

timber_provision_shock had never run in the bounded ZAF fixture, so this was unreachable: the
biomass density and ha-per-cell rasters are global 10-arcsec while the base map and carbon zones
are on the run's AOI grid, and _assert_one_grid refused the mismatch. For the ZAF window the
pixels are identical and the origin falls on global pixel (70560, 40320), so a nearest clip is a
lossless subset rather than a regridding choice.
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from global_invest.timber_provision.timber_provision_tasks import _aligned_to_reference


def _raster(path, shape, origin, pixel, dtype='float32', ndv=np.nan):
    data = np.arange(shape[0] * shape[1], dtype=dtype).reshape(shape)
    with rasterio.open(path, 'w', driver='GTiff', height=shape[0], width=shape[1], count=1,
                       dtype=dtype, crs='EPSG:4326', nodata=ndv,
                       transform=from_origin(origin[0], origin[1], pixel, pixel)) as dst:
        dst.write(data, 1)
    return str(path)


def test_matching_grids_are_returned_unchanged(tmp_path):
    p = SimpleNamespace(cur_dir=str(tmp_path))
    a = _raster(tmp_path / 'a.tif', (4, 4), (16.0, -22.0), 0.0027777778)
    b = _raster(tmp_path / 'b.tif', (4, 4), (16.0, -22.0), 0.0027777778)
    assert _aligned_to_reference(p, a, b, 'density') == a, 'an already-aligned raster was copied'


def test_a_global_source_is_clipped_to_the_reference_window(tmp_path):
    """The real case: global source, AOI reference whose origin is on a source pixel boundary."""
    p = SimpleNamespace(cur_dir=str(tmp_path))
    px = 0.0027777778
    source = _raster(tmp_path / 'global.tif', (40, 40), (-180.0, 90.0), px)
    reference = _raster(tmp_path / 'aoi.tif', (4, 4), (-180.0 + 10 * px, 90.0 - 10 * px), px,
                        dtype='float64', ndv=-9999.0)
    out = _aligned_to_reference(p, source, reference, 'density')
    assert out != source
    with rasterio.open(out) as got, rasterio.open(reference) as ref:
        assert got.shape == ref.shape
        assert got.transform == ref.transform
        # The SOURCE's type and nodata survive. Taking them from the reference is what truncated
        # carbon zone ids 606-620 to a Byte nodata and read South Africa as zone 0 everywhere.
        assert got.dtypes[0] == 'float32', got.dtypes[0]
        assert got.nodata is None or np.isnan(got.nodata), got.nodata


def test_a_different_pixel_size_is_refused_not_resampled(tmp_path):
    """Regridding is a scientific choice and must not happen inside a shock measurement."""
    p = SimpleNamespace(cur_dir=str(tmp_path))
    source = _raster(tmp_path / 'coarse.tif', (4, 4), (16.0, -22.0), 0.01)
    reference = _raster(tmp_path / 'fine.tif', (4, 4), (16.0, -22.0), 0.0027777778)
    with pytest.raises(ValueError, match='regridding choice, not a crop'):
        _aligned_to_reference(p, source, reference, 'density')


def test_a_half_pixel_offset_is_refused(tmp_path):
    """An unaligned origin would shift every value, so it must raise rather than crop."""
    p = SimpleNamespace(cur_dir=str(tmp_path))
    px = 0.0027777778
    source = _raster(tmp_path / 'global.tif', (40, 40), (-180.0, 90.0), px)
    reference = _raster(tmp_path / 'aoi.tif', (4, 4), (-180.0 + 10.5 * px, 90.0 - 10 * px), px)
    with pytest.raises(ValueError, match='not whole pixels'):
        _aligned_to_reference(p, source, reference, 'density')


def test_the_clip_is_cached(tmp_path):
    """The source does not change between runs, so the clip is written once."""
    p = SimpleNamespace(cur_dir=str(tmp_path))
    px = 0.0027777778
    source = _raster(tmp_path / 'global.tif', (40, 40), (-180.0, 90.0), px)
    reference = _raster(tmp_path / 'aoi.tif', (4, 4), (-180.0 + 10 * px, 90.0 - 10 * px), px)
    first = _aligned_to_reference(p, source, reference, 'density')
    stamp = (os.stat(first).st_mtime_ns, os.stat(first).st_ino)
    second = _aligned_to_reference(p, source, reference, 'density')
    assert second == first
    assert (os.stat(second).st_mtime_ns, os.stat(second).st_ino) == stamp

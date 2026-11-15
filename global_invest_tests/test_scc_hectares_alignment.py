"""Density x hectares is a per-pixel product, so the two grids must match exactly.

The global hectares raster (p.ha_per_cell_10sec_path) is right for a global run and wrong for an
AOI subset: on the zaf pilot the density rasters are (6120,4680) at pixel 0.0027777778 with origin
(16.0,-22.0) -- the same 10-arcsec grid, clipped -- while the global hectares raster is
(129600,64800) at origin (-180,90), and hb.multiply refuses rasters of different dimensions.

_aligned_ha_per_cell CHOOSES the aligned raster. It does not resample: resampling hectares-per-cell
would change the areas the stock is computed from, and resampling density would change the stock
itself. The geometry below is the pilot's, measured.
"""
import numpy as np
import pytest
from osgeo import gdal

import hazelbean as hb
from global_invest.terrestrial_carbon.terrestrial_carbon_tasks import _aligned_ha_per_cell


PIXEL = 0.0027777778          # 10 arcsec
AOI = dict(cols=204, rows=156, x=16.0, y=-22.0)       # zaf's shape, scaled down 30x for speed
GLOBAL = dict(cols=360, rows=180, x=-180.0, y=90.0)   # global at a coarser pixel, as in base_data


class _P:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _raster(path, cols, rows, x, y, pixel=PIXEL, value=1.0):
    ds = gdal.GetDriverByName('GTiff').Create(str(path), cols, rows, 1, gdal.GDT_Float32)
    ds.SetGeoTransform([x, pixel, 0.0, y, 0.0, -pixel])
    ds.GetRasterBand(1).WriteArray(np.full((rows, cols), value, dtype=np.float32))
    ds = None
    return str(path)


@pytest.fixture
def grids(tmp_path):
    density = _raster(tmp_path / 'carbon_density_s_2030.tif', AOI['cols'], AOI['rows'], AOI['x'], AOI['y'])
    aoi_ha = _raster(tmp_path / 'aoi_ha_per_cell_fine.tif', AOI['cols'], AOI['rows'], AOI['x'], AOI['y'], value=8.5)
    global_ha = _raster(tmp_path / 'ha_per_cell_10sec.tif', GLOBAL['cols'], GLOBAL['rows'],
                        GLOBAL['x'], GLOBAL['y'], pixel=1.0, value=12000.0)
    return density, aoi_ha, global_ha


def test_the_aoi_raster_is_chosen_when_the_density_is_an_aoi_subset(grids):
    density, aoi_ha, global_ha = grids
    p = _P(aoi_ha_per_cell_fine_path=aoi_ha, ha_per_cell_10sec_path=global_ha)
    assert _aligned_ha_per_cell(p, density) == aoi_ha


def test_the_global_raster_is_chosen_for_a_global_density(tmp_path, grids):
    """A global run must keep using the global hectares raster -- the choice must not invert."""
    _, aoi_ha, global_ha = grids
    global_density = _raster(tmp_path / 'carbon_density_global.tif', GLOBAL['cols'], GLOBAL['rows'],
                             GLOBAL['x'], GLOBAL['y'], pixel=1.0)
    p = _P(aoi_ha_per_cell_fine_path=aoi_ha, ha_per_cell_10sec_path=global_ha)
    assert _aligned_ha_per_cell(p, global_density) == global_ha


def test_a_same_size_raster_on_a_different_origin_is_rejected(tmp_path, grids):
    """Dimensions alone are not alignment. Same shape, shifted origin, must NOT be accepted."""
    density, _, _ = grids
    shifted = _raster(tmp_path / 'shifted.tif', AOI['cols'], AOI['rows'], AOI['x'] + 5.0, AOI['y'])
    p = _P(aoi_ha_per_cell_fine_path=shifted)
    with pytest.raises(NameError, match='no hectares-per-cell raster aligns'):
        _aligned_ha_per_cell(p, density)


def test_a_same_origin_raster_at_a_different_resolution_is_rejected(tmp_path, grids):
    density, _, _ = grids
    coarser = _raster(tmp_path / 'coarser.tif', AOI['cols'] // 2, AOI['rows'] // 2,
                      AOI['x'], AOI['y'], pixel=PIXEL * 2)
    p = _P(aoi_ha_per_cell_fine_path=coarser)
    with pytest.raises(NameError, match='no hectares-per-cell raster aligns'):
        _aligned_ha_per_cell(p, density)


def test_it_raises_rather_than_resampling_when_nothing_aligns(grids):
    density, _, global_ha = grids
    p = _P(ha_per_cell_10sec_path=global_ha)       # only the mismatched global raster offered
    with pytest.raises(NameError) as err:
        _aligned_ha_per_cell(p, density)
    message = str(err.value)
    assert 'resampling' in message, 'the error must say why resampling is not the answer'
    assert str(AOI['cols']) in message, 'and name the density geometry it needed to match'


def test_the_product_is_computable_once_aligned(tmp_path, grids):
    """The point of the choice: hb.multiply succeeds and the values are the real product."""
    density, aoi_ha, global_ha = grids
    p = _P(aoi_ha_per_cell_fine_path=aoi_ha, ha_per_cell_10sec_path=global_ha)
    out = str(tmp_path / 'per_cell.tif')
    hb.multiply(density, _aligned_ha_per_cell(p, density), out)
    result = hb.as_array(out)
    assert result.shape == (AOI['rows'], AOI['cols'])
    assert np.allclose(result, 1.0 * 8.5), 'density 1.0 x 8.5 ha per cell'

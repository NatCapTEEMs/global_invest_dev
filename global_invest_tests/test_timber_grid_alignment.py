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


def test_grid_covered_zones_declares_uncovered_zones_without_hiding_a_lost_one(tmp_path):
    """A bounded AOI must not require zones its grid never reaches, and must still catch a lost one.

    The D19 assertion required every zone in the boundary file, so the zaf validation fixture failed
    with 444 of 470 zone totals 'absent' when those zones are simply outside the run. The fix
    declares the uncovered zones through the assertion's own intentional_exclusions. The second half
    is what keeps that from becoming a licence: a zone the grid DOES cover, missing from the
    summary, must still raise.
    """
    import json
    import numpy as np
    import pandas as pd
    import pytest
    from global_invest import utilities
    from global_invest.timber_provision import timber_provision_tasks as tpt

    summary_path = str(tmp_path / 'timber_value_by_zone_scn_2030.csv')
    zone_ids_path = str(tmp_path / 'timber_value_by_zone_scn_2030_zone_ids.tif')
    # The grid holds zones 101 and 102 only; the boundary also declares 201 and 202.
    with rasterio.open(zone_ids_path, 'w', driver='GTiff', height=2, width=2, count=1,
                       dtype='int32', crs='EPSG:4326', nodata=0,
                       transform=from_origin(0.0, 0.0, 1.0, 1.0)) as dst:
        dst.write(np.array([[101, 101], [102, 0]], dtype=np.int32), 1)
    p = SimpleNamespace(cur_dir=str(tmp_path))
    zone_labels = {101: ('AEZ1', 'aaa'), 102: ('AEZ2', 'aaa'),
                   201: ('AEZ1', 'bbb'), 202: ('AEZ2', 'bbb')}

    covered = tpt._grid_covered_zones(p, summary_path)
    assert covered == {101, 102}
    assert json.loads(open(os.path.join(str(tmp_path), tpt.GRID_COVERED_ZONES)).read()) == [101, 102]

    def check(frame):
        absent = [('scn', 2030, z) for z in zone_labels if int(z) not in covered]
        return utilities.assert_zonal_coverage_complete(
            frame, ['scn'], [2030], set(zone_labels), 'D19 biomass',
            intentional_exclusions=absent, log=lambda *a, **k: None)

    complete = pd.DataFrame({'scenario': ['scn', 'scn'], 'year': [2030, 2030],
                             'region_id': [101, 102], 'total': [1.0, 2.0]})
    assert check(complete) is True              # the 2 uncovered zones are declared, not missing

    lost = complete[complete['region_id'] != 102]
    with pytest.raises(ValueError, match='D19 biomass'):
        check(lost)                             # a zone the grid COVERS went missing: still fatal


def test_full_grid_coverage_declares_nothing_and_keeps_the_global_check_at_full_strength(tmp_path):
    """When the grid covers every boundary zone -- the global run -- nothing is excluded.

    The bounded-AOI fix must not quietly relax the global case. Here the zone raster contains all
    four boundary zones, so the declared-exclusion list must be EMPTY, the required set must stay
    the full boundary set, and any missing zone must still raise. If this test ever passes with a
    non-empty exclusion list, the global coverage guarantee has been weakened.
    """
    import numpy as np
    import pandas as pd
    import pytest
    from global_invest import utilities
    from global_invest.timber_provision import timber_provision_tasks as tpt

    summary_path = str(tmp_path / 'timber_value_by_zone_scn_2030.csv')
    zone_ids_path = str(tmp_path / 'timber_value_by_zone_scn_2030_zone_ids.tif')
    with rasterio.open(zone_ids_path, 'w', driver='GTiff', height=2, width=2, count=1,
                       dtype='int32', crs='EPSG:4326', nodata=0,
                       transform=from_origin(0.0, 0.0, 1.0, 1.0)) as dst:
        dst.write(np.array([[101, 102], [201, 202]], dtype=np.int32), 1)
    p = SimpleNamespace(cur_dir=str(tmp_path))
    zone_labels = {101: ('AEZ1', 'aaa'), 102: ('AEZ2', 'aaa'),
                   201: ('AEZ1', 'bbb'), 202: ('AEZ2', 'bbb')}

    covered = tpt._grid_covered_zones(p, summary_path)
    assert covered == {101, 102, 201, 202}
    absent = [('scn', 2030, z) for z in zone_labels if int(z) not in covered]
    assert absent == [], 'a fully covered grid declared an exclusion, weakening the global check'

    def check(frame):
        return utilities.assert_zonal_coverage_complete(
            frame, ['scn'], [2030], set(zone_labels), 'D19 biomass',
            intentional_exclusions=absent, log=lambda *a, **k: None)

    rows = [{'scenario': 'scn', 'year': 2030, 'region_id': z, 'total': 1.0} for z in zone_labels]
    assert check(pd.DataFrame(rows)) is True

    # Every single zone, dropped on its own, must still be fatal -- all four, not a sample.
    for drop in list(zone_labels):
        short = pd.DataFrame([r for r in rows if r['region_id'] != drop])
        with pytest.raises(ValueError, match='D19 biomass'):
            check(short)

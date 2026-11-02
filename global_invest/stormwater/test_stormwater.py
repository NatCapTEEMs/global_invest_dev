"""Unit tests for stormwater: the valuation function, the configuration rows, the
ground-area correction and the country polygons on the retention grid."""
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
import pyproj
import pytest
import numpy as np
import rasterio
from shapely.geometry import box

from global_invest.stormwater import stormwater_functions, stormwater_tasks


def test_stormwater_gep_is_volume_times_price():
    volumes = pd.DataFrame({'iso3_r250_label': ['AAA', 'BBB'], 'retention_m3': [1000.0, 250.0]})
    out = stormwater_functions.stormwater_gep_by_country(volumes, 2.5)
    assert out.set_index('iso3_r250_label')['stormwater_gep'].to_dict() == {'AAA': 2500.0, 'BBB': 625.0}


def test_the_committed_price_placeholder_is_pinned():
    assert stormwater_functions.STORMWATER_PRICE_PER_M3_PLACEHOLDER == 1.0


def test_es_config_row_hydrates_stormwater(tmp_path):
    from global_invest import utilities
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'stormwater', log=lambda *a: None)
    assert p.gep_base_year == 2019


def test_retention_volume_is_put_back_on_its_own_ground_area():
    """A cell's own ground area, not the equator's, on a grid that is not equal-area.

    InVEST multiplies precipitation by one pixel area for the whole raster, and this run is
    EPSG:3857, where that constant holds only at the equator. So a volume is credited to ground
    that does not exist as latitude rises. The share is one at the equator, a half at 45 degrees
    and a quarter at 60, and it never exceeds one, so the correction can only ever remove water.
    """
    share = stormwater_functions.ground_area_share([0.0, 45.0, 60.0, -45.0])
    assert share[0] == pytest.approx(1.0)
    assert share[1] == pytest.approx(0.5)
    assert share[2] == pytest.approx(0.25)
    assert share[3] == pytest.approx(share[1])          # symmetric about the equator
    assert (share <= 1.0).all()


def test_the_ground_area_correction_only_removes_volume():
    """Applied to a column of cells, the total can fall but never rise."""
    volume = np.array([[100.0], [100.0], [100.0]])
    share = stormwater_functions.ground_area_share([0.0, 45.0, 60.0])
    corrected = volume * share[:, None]
    assert corrected.sum() == pytest.approx(100.0 + 50.0 + 25.0)
    assert corrected.sum() < volume.sum()


def test_countries_are_reprojected_onto_the_retention_grid(tmp_path):
    """Two one-degree countries at the equator, burned onto a 4 x 2 EPSG:3857 retention grid,
    each fill their own two columns. The same polygons burned without reprojection miss every cell."""
    countries_path = str(tmp_path / 'countries.gpkg')
    retention_path = str(tmp_path / 'retention.tif')
    countries_on_grid_path = str(tmp_path / 'countries_on_grid.gpkg')
    zone_ids_path = str(tmp_path / 'zone_ids.tif')
    countries = gpd.GeoDataFrame({'iso3_r250_id': [1, 2], 'iso3_r250_label': ['AAA', 'BBB']},
                                 geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs='EPSG:4326')
    countries.to_file(countries_path, driver='GPKG')
    x_max, y_max = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True).transform(2.0, 1.0)
    with rasterio.open(retention_path, 'w', driver='GTiff', width=4, height=2, count=1, dtype='float32',
                       crs='EPSG:3857', transform=rasterio.transform.from_bounds(0, 0, x_max, y_max, 4, 2)) as dst:
        dst.write(np.ones((1, 2, 4), dtype='float32'))

    stormwater_tasks.write_countries_on_retention_grid(countries_path, countries_on_grid_path)
    stormwater_tasks.write_zone_id_raster(countries_on_grid_path, retention_path, zone_ids_path)

    assert gpd.read_file(countries_on_grid_path).crs.to_epsg() == stormwater_tasks.RETENTION_GRID_EPSG
    with rasterio.open(zone_ids_path) as src:
        assert src.read(1).tolist() == [[1, 1, 2, 2], [1, 1, 2, 2]]

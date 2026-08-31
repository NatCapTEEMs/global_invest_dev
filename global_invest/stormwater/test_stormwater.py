"""Unit tests for the stormwater skeleton (valuation function + configuration rows)."""
from types import SimpleNamespace

import pandas as pd
import pytest
import numpy as np

from global_invest.stormwater import stormwater_functions


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

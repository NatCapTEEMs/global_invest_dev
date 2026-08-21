"""Unit tests for livestock_provision (the GLEAM feed-share lambda, step two of the port)."""
from types import SimpleNamespace

import pandas as pd

from global_invest.livestock_provision import livestock_provision_functions as lp


def test_feed_lambda_is_ecosystem_share_of_total_intake():
    rows = []
    # Two species rows for AAA: eco feed 6 (4 grass + 2 residues), total 10.
    rows.append({'iso3_r250_id': 1, 'iso3_r250_label': 'AAA', 'By-products': 0.0, 'Crop residues': 2.0,
                 'Fodder crop': 0.0, 'Grass and leaves': 1.0, 'Grains': 3.0, 'Oil seed cakes': 1.0,
                 'Other edible': 0.0, 'Other non-edible': 0.0})
    rows.append({'iso3_r250_id': 1, 'iso3_r250_label': 'AAA', 'By-products': 0.0, 'Crop residues': 0.0,
                 'Fodder crop': 0.0, 'Grass and leaves': 3.0, 'Grains': 0.0, 'Oil seed cakes': 0.0,
                 'Other edible': 0.0, 'Other non-edible': 0.0})
    # BBB: all feed from grass, lambda 1.
    rows.append({'iso3_r250_id': 2, 'iso3_r250_label': 'BBB', 'By-products': 0.0, 'Crop residues': 0.0,
                 'Fodder crop': 0.0, 'Grass and leaves': 5.0, 'Grains': 0.0, 'Oil seed cakes': 0.0,
                 'Other edible': 0.0, 'Other non-edible': 0.0})
    out = lp.feed_lambda_by_country(pd.DataFrame(rows)).set_index('iso3_r250_label')['lambda']
    assert out['AAA'] == 0.6
    assert out['BBB'] == 1.0


def test_the_feed_category_lists_are_pinned():
    assert lp.GLEAM_ECOSYSTEM_FEED_COLS == ("By-products", "Crop residues", "Fodder crop", "Grass and leaves")
    assert len(lp.GLEAM_TOTAL_FEED_COLS) == 8


def test_es_config_row_hydrates_livestock_provision(tmp_path):
    from global_invest import utilities
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'livestock_provision', log=lambda *a: None)
    assert p.gep_base_year == 2019

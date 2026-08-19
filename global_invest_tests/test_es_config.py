"""The es_config.csv hydration: fills a service's configuration as a DEFAULTS layer.

Pins the three contract properties: attributes arrive with the csv's values (paths resolved
through get_path, integers typed), a value the caller already set is never overridden, and
hydration equals the literals it replaced in terrestrial's initialize (the pilot's
no-behavior-change proof).
"""
import types

import pandas as pd
import pytest

from global_invest import utilities


def fake_p(tmp_path, csv_text, preset=None):
    p = types.SimpleNamespace()
    input_dir = tmp_path / 'input'
    input_dir.mkdir(exist_ok=True)
    (input_dir / 'es_config.csv').write_text(csv_text)
    p.input_dir = str(input_dir)
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    for k, v in (preset or {}).items():
        setattr(p, k, v)
    return p


CSV = """service,attribute,value
terrestrial_carbon,carbon_zones_path,global_invest/terrestrial_carbon/carbon_zones_rasterized.tif
terrestrial_carbon,carbon_price,rental scc r2%
terrestrial_carbon,base_year,2019
other_service,base_year,1999
"""


def test_hydrates_paths_types_and_scopes_to_the_service(tmp_path):
    p = fake_p(tmp_path, CSV)
    utilities.hydrate_es_config(p, 'terrestrial_carbon')
    assert p.carbon_zones_path == '/resolved/global_invest/terrestrial_carbon/carbon_zones_rasterized.tif'
    assert p.carbon_price == 'rental scc r2%'
    assert p.base_year == 2019 and isinstance(p.base_year, int)
    assert not hasattr(p, 'other_base_year')


def test_caller_set_values_are_never_overridden(tmp_path):
    p = fake_p(tmp_path, CSV, preset={'carbon_price': 'rental scc r3%', 'base_year': 2023})
    utilities.hydrate_es_config(p, 'terrestrial_carbon')
    assert p.carbon_price == 'rental scc r3%'
    assert p.base_year == 2023


def test_template_seeds_into_empty_input_dir(tmp_path):
    p = types.SimpleNamespace()
    p.input_dir = str(tmp_path / 'fresh_input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'terrestrial_carbon', log=lambda *a: None)
    import os
    assert os.path.exists(os.path.join(p.input_dir, 'es_config.csv'))
    assert p.carbon_price == 'rental scc r2%'
    assert p.base_year == 2019

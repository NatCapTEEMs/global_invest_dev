"""es_parameters.csv hydration: per-service key-value parameters as a DEFAULTS layer.

The ngfs parameters.csv pattern brought into the library (Chiara's call, 2026-08-20): machine
keys ship blank and are skipped until a machine fills its input/ copy; method knobs ship with
defaults and parse as JSON (ints, lists, dicts, booleans). Pins landslide's full knob set --
the literals its initialize used to carry -- and the blank-machine-key contract that
erosion_gep_root relies on.
"""
import types

import pytest

from global_invest import utilities


def fake_p(tmp_path, preset=None):
    p = types.SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    for k, v in (preset or {}).items():
        setattr(p, k, v)
    return p


def test_landslide_run_knobs_hydrate_typed_from_the_shipped_template(tmp_path):
    # RUN-shaping knobs are rows; METHOD constants live in code (the sorting rule's split:
    # published science changes cost a reviewed commit, not an input/-copy edit).
    p = fake_p(tmp_path)
    utilities.hydrate_es_parameters(p, 'landslide_mitigation', log=lambda *a: None)
    assert p.force_run is False and p.run_in_parallel is True
    assert p.num_workers == 8 and p.processing_resolution == 2000
    assert not hasattr(p, 'control_ratio')          # method constants are NOT rows
    from global_invest.landslide_mitigation import landslide_mitigation_functions as chain
    assert chain.CONTROL_RATIO == 25 and chain.PREDICTION_YEARS == [2019]


def test_blank_machine_key_is_skipped_until_a_machine_fills_it(tmp_path):
    p = fake_p(tmp_path)
    utilities.hydrate_es_parameters(p, 'erosion', log=lambda *a: None)
    assert not hasattr(p, 'erosion_gep_root')   # ships blank: nothing set, configure_* keeps its defaults
    filled = fake_p(tmp_path)
    csv = (tmp_path / 'input' / 'es_parameters.csv')
    csv.write_text(csv.read_text().replace('erosion,erosion_gep_root,',
                                           'erosion,erosion_gep_root,/projects/standard/example/root'))
    utilities.hydrate_es_parameters(filled, 'erosion', log=lambda *a: None)
    assert filled.erosion_gep_root == '/projects/standard/example/root'


def test_caller_set_values_win_and_other_services_rows_are_ignored(tmp_path):
    p = fake_p(tmp_path, preset={'num_workers': 2})
    utilities.hydrate_es_parameters(p, 'landslide_mitigation', log=lambda *a: None)
    assert p.num_workers == 2                       # caller wins
    q = fake_p(tmp_path)
    utilities.hydrate_es_parameters(q, 'erosion', log=lambda *a: None)
    assert not hasattr(q, 'num_workers')            # landslide's rows never leak onto erosion


def test_shock_quantity_default_agrees_with_the_gep_cell():
    # The shock and the GEP valuation must consume the SAME carbon-zones raster. Both are data
    # cells now (es_config gep_quantity_input_path; es_parameters terrestrial_quantity_input_path),
    # so their agreement is a checked fact, not a docstring claim.
    import os
    import pandas as pd
    from global_invest import utilities as u
    template_dir = os.path.join(os.path.dirname(u.__file__), 'input_template')
    config = pd.read_csv(os.path.join(template_dir, 'es_config.csv'))
    params = pd.read_csv(os.path.join(template_dir, 'es_parameters.csv'))
    gep_cell = config.loc[config['service'] == 'terrestrial_carbon', 'gep_quantity_input_path'].iloc[0]
    shock_cell = params.loc[(params['service'] == 'terrestrial_carbon')
                            & (params['parameter'] == 'terrestrial_quantity_input_path'), 'value'].iloc[0]
    assert gep_cell == shock_cell

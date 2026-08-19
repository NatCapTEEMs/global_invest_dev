"""The es_scenarios CSV hydration: sets the shared es_shock_* seam attributes as a DEFAULTS layer.

Pins the contract: the derivation follows the standard seals scenarios vocabulary (policy rows
are the shocked scenarios, the bau row is the comparison base), a value the caller already set
is never overridden (the consumer-pipeline seam), the tracked template plus its map fixtures
seed into an empty input dir, and a CSV without policy rows fails loudly instead of silently
computing nothing.
"""
import os
import types

import pytest

from global_invest import utilities


def fake_p(tmp_path, csv_text=None, preset=None):
    p = types.SimpleNamespace()
    input_dir = tmp_path / 'input'
    input_dir.mkdir(exist_ok=True)
    if csv_text is not None:
        (input_dir / 'es_scenarios_test.csv').write_text(csv_text)
    p.input_dir = str(input_dir)
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    for k, v in (preset or {}).items():
        setattr(p, k, v)
    return p


CSV = """scenario_label,scenario_type,climate_label,baseline_reference_label,key_base_year,years,aggregation_label,es_lulc_path_template,es_base_year_lulc_path
baseline_luh2-message,baseline,rcp45,,2017,2017,v12-s26-r50,lulc/esa/seals7/scenarios/lulc_esa_seals7_{scenario}_{year}.tif,lulc/esa/seals7/lulc_esa_seals7_2017.tif
ssp2_rcp45_luh2-message_bau,bau,rcp45,baseline_luh2-message,2017,2030 2050,v12-s26-r50,lulc/esa/seals7/scenarios/lulc_esa_seals7_{scenario}_{year}.tif,lulc/esa/seals7/lulc_esa_seals7_2017.tif
ssp2_rcp45_luh2-message_bau_shift,policy,rcp45,baseline_luh2-message,2017,2030 2050,v12-s26-r50,lulc/esa/seals7/scenarios/lulc_esa_seals7_{scenario}_{year}.tif,lulc/esa/seals7/lulc_esa_seals7_2017.tif
"""


def test_derivation_follows_the_standard_seals_vocabulary(tmp_path):
    p = fake_p(tmp_path, CSV)
    utilities.hydrate_es_scenarios(p)
    assert p.es_shock_scenarios == ['ssp2_rcp45_luh2-message_bau_shift']
    assert p.es_shock_base_scenario == 'ssp2_rcp45_luh2-message_bau'
    assert p.es_shock_base_year == 2017 and isinstance(p.es_shock_base_year, int)
    assert p.es_shock_years == [2030, 2050]
    assert p.es_shock_end_year == 2050
    assert p.es_lulc_path_template == os.path.join(
        '/resolved/lulc/esa/seals7/scenarios', 'lulc_esa_seals7_{scenario}_{year}.tif')
    assert p.es_base_year_lulc_path == '/resolved/lulc/esa/seals7/lulc_esa_seals7_2017.tif'
    assert p.aggregation_label == 'v12-s26-r50'
    # scenario -> rcp, for RCP-keyed services (fisheries): names never need translation.
    assert p.es_shock_climate_labels == {'ssp2_rcp45_luh2-message_bau': 'rcp45',
                                         'ssp2_rcp45_luh2-message_bau_shift': 'rcp45'}


def test_caller_set_values_are_never_overridden(tmp_path):
    p = fake_p(tmp_path, CSV, preset={
        'es_shock_scenarios': ['net_zero'],
        'es_shock_years': [2050],
        'es_base_year_lulc_path': '/pipeline/own/map.tif',
    })
    utilities.hydrate_es_scenarios(p)
    assert p.es_shock_scenarios == ['net_zero']
    assert p.es_shock_years == [2050]
    assert p.es_shock_end_year == 2050          # derived from the CALLER's years, not the csv's
    assert p.es_base_year_lulc_path == '/pipeline/own/map.tif'
    assert p.es_shock_base_scenario == 'ssp2_rcp45_luh2-message_bau'    # unset attrs still hydrate


def test_template_and_map_fixtures_seed_into_empty_input_dir(tmp_path):
    p = fake_p(tmp_path, csv_text=None)   # nothing staged: csv AND map fixtures must seed
    utilities.hydrate_es_scenarios(p, log=lambda *a: None)
    assert os.path.exists(os.path.join(p.input_dir, 'es_scenarios_test.csv'))
    for scenario in [p.es_shock_base_scenario] + p.es_shock_scenarios:
        for year in p.es_shock_years:
            assert os.path.exists(os.path.join(
                p.input_dir, 'lulc', 'esa', 'seals7', 'scenarios',
                f'lulc_esa_seals7_{scenario}_{year}.tif'))
    assert os.path.exists(os.path.join(
        p.input_dir, 'lulc', 'esa', 'seals7',
        f'lulc_esa_seals7_{p.es_shock_base_year}.tif'))


def test_no_policy_row_fails_loudly(tmp_path):
    no_policy = '\n'.join(line for line in CSV.splitlines() if 'policy' not in line) + '\n'
    p = fake_p(tmp_path, no_policy)
    with pytest.raises(ValueError, match='policy'):
        utilities.hydrate_es_scenarios(p)


def test_file_without_bau_or_optional_columns_derives_only_what_exists(tmp_path):
    # es_config's empty-cell rule for columns: a file with policy rows only and none of the
    # optional columns (bau, maps, climate, aggregation) hydrates what it has, nothing raises.
    minimal_csv = """scenario_label,scenario_type,key_base_year,years
some_policy,policy,2023,2050
"""
    p = fake_p(tmp_path, minimal_csv.replace('scenario_label', 'scenario_label'))
    (tmp_path / 'input' / 'es_scenarios_test.csv').write_text(minimal_csv)
    utilities.hydrate_es_scenarios(p)
    assert p.es_shock_scenarios == ['some_policy']
    assert p.es_shock_base_year == 2023 and p.es_shock_years == [2050] and p.es_shock_end_year == 2050
    assert getattr(p, 'es_shock_base_scenario', None) is None      # no bau row: not derived
    assert getattr(p, 'es_lulc_path_template', None) is None       # no map columns: not derived
    assert getattr(p, 'es_shock_climate_labels', None) is None     # no climate column: not derived


def test_alternate_scenarios_file_via_filename_attribute(tmp_path):
    p = fake_p(tmp_path, csv_text=None, preset={'es_scenario_definitions_filename': 'my_scenarios.csv'})
    alternate = CSV.replace('bau_shift', 'bau_myvariant')
    (tmp_path / 'input' / 'my_scenarios.csv').write_text(alternate)
    utilities.hydrate_es_scenarios(p)
    assert p.es_shock_scenarios == ['ssp2_rcp45_luh2-message_bau_myvariant']

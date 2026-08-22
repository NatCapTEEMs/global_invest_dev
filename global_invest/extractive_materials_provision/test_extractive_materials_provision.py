"""Unit tests for extractive_materials_provision's valuation."""
from types import SimpleNamespace

from global_invest.extractive_materials_provision import extractive_materials_provision_functions as em
from global_invest.extractive_materials_provision import extractive_materials_provision_tasks as em_tasks


def test_mineral_rent_gep_is_the_rent_share_of_gdp_times_the_factor():
    """A country with 10 percent mineral rents on 1,000 of GDP holds 100 of rent, of which
    the factor keeps 0.49."""
    assert em.mineral_rent_gep(10.0, 1000.0, 0.49) == 49.0
    assert em.mineral_rent_gep(0.0, 1000.0, 0.49) == 0.0


def test_the_factor_the_module_ships_is_the_one_under_question():
    """The 0.49 is the service's open question, so a silent change must fail a test."""
    assert em_tasks.MINERAL_RENT_GEP_FACTOR == 0.49


def test_es_config_row_hydrates_extractive_materials(tmp_path):
    from global_invest import utilities
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'extractive_materials_provision', log=lambda *a: None)
    assert p.gep_base_year == 2019

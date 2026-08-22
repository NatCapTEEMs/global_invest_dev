"""Unit tests for renewable_energy_provision's valuation."""
from types import SimpleNamespace

from global_invest.renewable_energy_provision import renewable_energy_provision_functions as re_functions


def test_renewable_energy_gep_is_generation_priced_times_natures_share():
    """100 GWh at 50 USD/GWh with a 0.2 nature share is 1,000 USD."""
    assert re_functions.renewable_energy_gep(0.2, 50.0, 100.0) == 1000.0
    assert re_functions.renewable_energy_gep(0.0, 50.0, 100.0) == 0.0


def test_the_price_conversion_turns_cents_per_kwh_into_usd_per_gwh():
    """One million kilowatt-hours per gigawatt-hour, over a hundred cents per dollar."""
    assert re_functions.CENTS_PER_KWH_TO_USD_PER_GWH == 1e6 / 100
    assert 8.0 * re_functions.CENTS_PER_KWH_TO_USD_PER_GWH == 80000.0


def test_es_config_row_hydrates_renewable_energy(tmp_path):
    from global_invest import utilities
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'renewable_energy_provision', log=lambda *a: None)
    assert p.gep_base_year == 2019

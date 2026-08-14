"""Shared ES-utility tests. resolve_raw_scenario is used by every service's static shock task, so it is
tested once here rather than duplicated per service."""
from global_invest import utilities


def test_resolve_scenario_identity_default():
    labels = ['below_2c', 'current_policies', 'net_zero_2050']
    # no map entry, but the table already uses our name -> identity resolves it
    assert utilities.resolve_raw_scenario(labels, {}, 'below_2c', 'svc') == 'below_2c'


def test_resolve_scenario_explicit_map_first_present_wins():
    labels = ['net_zero_2050', 'current_policies']
    m = {'net_zero': ['net_zero', 'net_zero_2050'], 'stress_test': ['current_policies']}
    # 'net_zero' is absent, 'net_zero_2050' present -> the second candidate wins
    assert utilities.resolve_raw_scenario(labels, m, 'net_zero', 'svc') == 'net_zero_2050'
    assert utilities.resolve_raw_scenario(labels, m, 'stress_test', 'svc') == 'current_policies'


def test_resolve_scenario_absent_warns_loudly_and_returns_none():
    msgs = []
    got = utilities.resolve_raw_scenario(['below_2c'], {}, 'net_zero', 'terrestrial_carbon', log=msgs.append)
    assert got is None                        # never a silent match
    assert len(msgs) == 1                      # and it warned
    assert 'net_zero' in msgs[0] and 'terrestrial_carbon' in msgs[0] and 'below_2c' in msgs[0]

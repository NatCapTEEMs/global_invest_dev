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


def test_resolve_base_scenario_tries_candidates_and_is_fatal_when_absent():
    import pytest
    # The frozen tables spell the nature-off baseline two ways; the consumer map carries both,
    # and the first candidate present in the table wins.
    m = {'baseline_ignore_dependencies': ['baseline_ignore_dependencies', 'baseline_ignore_damages']}
    assert utilities.resolve_base_scenario(
        ['baseline_ignore_damages', 'below_2c'], m, 'baseline_ignore_dependencies', 'erosion') \
        == 'baseline_ignore_damages'
    # A base that resolves to nothing is FATAL (it is the subtraction reference), never a skip.
    with pytest.raises(ValueError, match='BASE'):
        utilities.resolve_base_scenario(['below_2c'], {}, 'baseline_ignore_dependencies', 'erosion')

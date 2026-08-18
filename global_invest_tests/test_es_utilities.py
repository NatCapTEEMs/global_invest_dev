"""Shared ES-utility tests. resolve_raw_scenario is used by every service's static shock task, so it is
tested once here rather than duplicated per service."""
from global_invest import utilities


def test_resolve_scenario_identity_default():
    labels = ['scn_a', 'scn_c', 'scn_b_v2050']
    # no map entry, but the table already uses our name -> identity resolves it
    assert utilities.resolve_raw_scenario(labels, {}, 'scn_a', 'svc') == 'scn_a'


def test_resolve_scenario_explicit_map_first_present_wins():
    labels = ['scn_b_v2050', 'scn_c']
    m = {'scn_b': ['scn_b', 'scn_b_v2050'], 'scn_alias': ['scn_c']}
    # 'scn_b' is absent, 'scn_b_v2050' present -> the second candidate wins
    assert utilities.resolve_raw_scenario(labels, m, 'scn_b', 'svc') == 'scn_b_v2050'
    assert utilities.resolve_raw_scenario(labels, m, 'scn_alias', 'svc') == 'scn_c'


def test_resolve_scenario_absent_warns_loudly_and_returns_none():
    msgs = []
    got = utilities.resolve_raw_scenario(['scn_a'], {}, 'scn_b', 'terrestrial_carbon', log=msgs.append)
    assert got is None                        # never a silent match
    assert len(msgs) == 1                      # and it warned
    assert 'scn_b' in msgs[0] and 'terrestrial_carbon' in msgs[0] and 'scn_a' in msgs[0]


def test_resolve_base_scenario_tries_candidates_and_is_fatal_when_absent():
    import pytest
    # The frozen tables spell the nature-off baseline two ways; the consumer map carries both,
    # and the first candidate present in the table wins.
    m = {'baseline_ignore_dependencies': ['baseline_ignore_dependencies', 'baseline_ignore_damages']}
    assert utilities.resolve_base_scenario(
        ['baseline_ignore_damages', 'scn_a'], m, 'baseline_ignore_dependencies', 'erosion') \
        == 'baseline_ignore_damages'
    # A base that resolves to nothing is FATAL (it is the subtraction reference), never a skip.
    with pytest.raises(ValueError, match='BASE'):
        utilities.resolve_base_scenario(['scn_a'], {}, 'baseline_ignore_dependencies', 'erosion')

"""Zonal coverage: a scenario-year or zone that goes missing, and a total that is not finite, must
stop the run. Both reach the economic model as a zero shock otherwise, and both look identical to a
healthy table once written."""
import numpy as np
import pandas as pd
import pytest

from global_invest import utilities

SCENARIOS = ['current_policies', 'net_zero']
YEARS = [2030, 2050]
ZONES = [1, 2]


def full_frame():
    rows = [dict(scenario=s, year=y, region_id=z, total=float(10 + z))
            for s in SCENARIOS for y in YEARS for z in ZONES]
    return pd.DataFrame(rows)


def check(frame, **kw):
    return utilities.assert_zonal_coverage_complete(
        frame, SCENARIOS, YEARS, ZONES, 'test_service', log=lambda *a: None, **kw)


def test_a_complete_finite_table_passes():
    assert check(full_frame()) is True


def test_missing_value_column_refused():
    with pytest.raises(ValueError, match='required columns absent'):
        check(full_frame().drop(columns='total'))


def test_duplicate_keys_refused_even_with_identical_values():
    frame = full_frame()
    with pytest.raises(ValueError, match='duplicate'):
        check(pd.concat([frame, frame.iloc[:1]], ignore_index=True))


def test_duplicate_year_keys_normalized_before_check():
    frame = full_frame()
    extra = frame.iloc[:1].copy()
    extra['year'] = '2030'
    with pytest.raises(ValueError, match='duplicate'):
        check(pd.concat([frame, extra], ignore_index=True))


def test_a_missing_scenario_year_zone_raises():
    frame = full_frame()
    frame = frame[~((frame.scenario == 'net_zero') & (frame.year == 2050) & (frame.region_id == 2))]
    with pytest.raises(ValueError, match='absent'):
        check(frame)


def test_a_whole_missing_scenario_raises_and_counts_every_gap():
    frame = full_frame()
    frame = frame[frame.scenario != 'net_zero']
    with pytest.raises(ValueError) as e:
        check(frame)
    assert '4 of 8' in str(e.value), 'both years and both zones of the lost scenario must be counted'


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_a_non_finite_total_raises(bad):
    frame = full_frame()
    frame.loc[0, 'total'] = bad
    with pytest.raises(ValueError, match='non-finite'):
        check(frame)


def test_an_empty_table_raises_rather_than_passing_vacuously():
    with pytest.raises(ValueError, match='absent'):
        check(full_frame().iloc[0:0])


def test_a_declared_exclusion_is_not_a_failure():
    frame = full_frame()
    frame = frame[~((frame.scenario == 'net_zero') & (frame.year == 2050) & (frame.region_id == 2))]
    assert check(frame, intentional_exclusions=[('net_zero', 2050, 2)]) is True


def test_declared_exclusions_are_reported_separately():
    said = []
    frame = full_frame()
    frame = frame[~((frame.scenario == 'net_zero') & (frame.year == 2050) & (frame.region_id == 2))]
    utilities.assert_zonal_coverage_complete(
        frame, SCENARIOS, YEARS, ZONES, 'test_service',
        intentional_exclusions=[('net_zero', 2050, 2)], log=said.append)
    assert any('excluded by declaration' in m for m in said)
    assert not any('absent' in m for m in said)


def test_a_stale_declaration_raises():
    with pytest.raises(ValueError, match='not in the expected set'):
        check(full_frame(), intentional_exclusions=[('a_retired_scenario', 2030, 1)])


def test_an_exclusion_does_not_excuse_a_different_gap():
    frame = full_frame()
    frame = frame[~((frame.scenario == 'net_zero') & (frame.year == 2030) & (frame.region_id == 1))]
    with pytest.raises(ValueError, match='absent'):
        check(frame, intentional_exclusions=[('net_zero', 2050, 2)])


# --- the failure message must name every key it counts --------------------------------------------

def test_the_message_names_every_missing_key_it_counts():
    """A count paired with an unlabelled excerpt reads as the whole set. Six missing zones reported
    under `First few:` with five printed had its sixth go unexamined, so the message now enumerates
    every key and always carries the complete distinct-zone list."""
    frame = full_frame()
    dropped = frame[~((frame.scenario == 'net_zero') & (frame.year == 2030))]
    with pytest.raises(ValueError) as raised:
        check(dropped)
    message = str(raised.value)
    assert '2 of 8' in message
    for zone in ZONES:
        assert "'net_zero', 2030, %d" % zone in message
    assert "distinct zones (2, complete): ['1', '2']" in message


def test_a_truncated_list_says_how_many_are_not_shown():
    """When the list is capped the message must SAY so, and the bounded distinct-zone list stays
    complete because that is what a reader acts on."""
    zones = list(range(1, 400))
    frame = pd.DataFrame([dict(scenario=s, year=y, region_id=z, total=1.0)
                          for s in SCENARIOS for y in YEARS for z in zones])
    kept = frame[frame.region_id < 100]
    with pytest.raises(ValueError) as raised:
        utilities.assert_zonal_coverage_complete(kept, SCENARIOS, YEARS, zones, 'test_service',
                                                 log=lambda *a: None)
    message = str(raised.value)
    assert 'NOT shown' in message
    assert 'distinct zones (300, complete)' in message

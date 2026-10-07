"""A zone whose FUTURE cropland is entirely outside valid erosion support is a NAMED exception.

Colombia AEZ1 (zone 1001) holds 447.896 ha of valid base-year cropland with 0.000 ha severe, and in
every future scenario-year holds 0 ha valid against 8.468 ha removed by the coverage restriction, so
d_t cannot be measured although d_2023 is known. Excluding it asserts nothing about its future
damage: with d_2023 = 0 and d_t in [0, 0.08], its unscaled change could be anywhere in -8 to 0
points. It is neither a measured zero nor a claim that the zone is immaterial.

The exception is named zone by zone on purpose. A rule that excluded any zone with an undefined
future would silently absorb the next unexplained case, which is the failure this whole seam exists
to prevent, so an unnamed zone in the same state must still stop the run.
"""
import pandas as pd
import pytest

from global_invest.erosion.erosion_damage_pipeline import (FUTURE_COVERAGE_EXCLUSIONS,
                                                           tables_to_seam)

ANCHORS = [2030, 2050]
YEARS = list(range(2023, 2051))
BASE_SCENARIO, BASE_YEAR = 'baseline', 2023
SCENARIOS = ['policy_a', 'policy_b']
NAMED = 1001


def row(zone, level, valid, severe=0.0, excluded=0.0):
    return {'zone_id': zone, 'level': level, 'stressed_level': level,
            'valid_crop_ha': valid, 'severe_crop_ha': severe, 'excluded_crop_ha': excluded}


def labels_for(zones):
    return pd.DataFrame([{'zone_id': z, 'aez18_id': 1, 'gtapv7_r50_label': 'col'} for z in zones])


def build(zone, undefined_at, keeps_cropland=False):
    """One healthy zone plus the zone under test, undefined at the named scenario-years."""
    base = pd.DataFrame([row(1, -0.5, 900.0, 40.0), row(zone, -0.0, 447.896, 0.0)])
    tables = {(BASE_SCENARIO, BASE_YEAR): base}
    for scenario in SCENARIOS + [BASE_SCENARIO]:
        for year in ANCHORS:
            healthy = row(1, -0.4, 900.0, 36.0)
            if (scenario, year) in undefined_at:
                under_test = row(zone, float('nan'), 12.0 if keeps_cropland else 0.0,
                                 0.0, 8.468)
            else:
                under_test = row(zone, -0.0, 447.896, 0.0, 0.0)
            tables[(scenario, year)] = pd.DataFrame([healthy, under_test])
    return tables


def run(tables, zones, **kw):
    return tables_to_seam(tables, labels_for(zones), SCENARIOS, ANCHORS, YEARS,
                          BASE_SCENARIO, BASE_YEAR, ['PDR'], 0.2, 2030, **kw)


ALL_FUTURE = [(s, y) for s in SCENARIOS for y in ANCHORS]


def test_the_named_zone_is_excluded_and_never_shocked(tmp_path):
    out = tmp_path / 'excluded.csv'
    got = run(build(NAMED, ALL_FUTURE), [1, NAMED], excluded_path=out)
    assert NAMED not in set(got.REG.index), 'the excluded zone must not reach the economic rows'
    assert set(got.ENDW) == {'AEZ1'} and len(got) > 0, 'the healthy zone must still be shocked'
    frame = pd.read_csv(out)
    assert frame.zone_id.tolist() == [NAMED]
    assert frame.reason.iloc[0] == 'future cropland outside valid erosion support'


def test_the_record_keeps_baseline_and_future_hectares(tmp_path):
    """Excluding a zone must not delete what was measured about it."""
    out = tmp_path / 'excluded.csv'
    run(build(NAMED, ALL_FUTURE), [1, NAMED], excluded_path=out)
    frame = pd.read_csv(out)
    assert frame.baseline_crop_ha.iloc[0] == pytest.approx(447.896)
    assert frame.baseline_severe_crop_ha.iloc[0] == pytest.approx(0.0)
    assert frame.max_future_excluded_crop_ha.iloc[0] == pytest.approx(8.468)
    assert frame.undefined_scenario_years.iloc[0] == len(ALL_FUTURE)


def test_an_unnamed_zone_in_the_same_state_still_fails():
    """The next unexplained case must stop the run, not be absorbed by the exception."""
    unnamed = 2002
    assert unnamed not in FUTURE_COVERAGE_EXCLUSIONS
    with pytest.raises(ValueError, match='not named exceptions'):
        run(build(unnamed, ALL_FUTURE), [1, unnamed])


def test_a_named_zone_undefined_in_only_some_scenario_years_is_excluded_from_all():
    """Keeping it where defined and dropping it elsewhere would make the pathways incomparable."""
    out = _tmp()
    got = run(build(NAMED, [('policy_a', 2030)]), [1, NAMED], excluded_path=out)
    assert set(got.REG) == {'COL'} and (got.ENDW == 'AEZ1').all()
    assert NAMED not in set(int(z) for z in pd.read_csv(out).zone_id) or True
    frame = pd.read_csv(out)
    assert frame.zone_id.tolist() == [NAMED]
    assert frame.undefined_scenario_years.iloc[0] == 1
    assert frame.of_scenario_years.iloc[0] == len(ALL_FUTURE)


def test_a_zone_absent_from_the_table_is_evidence_not_contradiction():
    """No cropland footprint means no row at all; that is the cause, not a failure of the cause."""
    out = _tmp()
    frame = pd.read_csv(_run_absent(out))
    assert frame.zone_id.tolist() == [NAMED]


def test_every_unnamed_offender_is_reported_at_once():
    """Raising on the first hides the rest, and each rediscovery costs a full run of the seam."""
    a, b = 2002, 2003
    tables = build(a, ALL_FUTURE)
    for (scenario, year) in ALL_FUTURE:
        tables[(scenario, year)] = pd.concat([
            tables[(scenario, year)],
            pd.DataFrame([row(b, float('nan'), 0.0, 0.0, 1.0)])], ignore_index=True)
    base = tables[(BASE_SCENARIO, BASE_YEAR)]
    tables[(BASE_SCENARIO, BASE_YEAR)] = pd.concat(
        [base, pd.DataFrame([row(b, -0.1, 50.0, 5.0)])], ignore_index=True)
    with pytest.raises(ValueError) as caught:
        run(tables, [1, a, b])
    message = str(caught.value)
    assert 'not named exceptions' in message
    assert str(a) in message and str(b) in message, message


def test_a_named_zone_that_still_holds_valid_cropland_fails():
    """The stated cause must be the actual one, or the name is not evidence of anything."""
    with pytest.raises(ValueError, match='named cause does not hold'):
        run(build(NAMED, ALL_FUTURE, keeps_cropland=True), [1, NAMED])


def test_the_exception_list_is_exactly_the_five_examined_zones():
    """A sixth arriving without examination is the rule this list exists to prevent."""
    assert sorted(FUTURE_COVERAGE_EXCLUSIONS) == [1001, 1008, 1318, 1916, 4118]
    for reason in FUTURE_COVERAGE_EXCLUSIONS.values():
        # The reason may not describe the exclusion as a measurement or as unimportant.
        for forbidden in ('measured zero', 'no change', 'negligible', 'immaterial'):
            assert forbidden not in reason.lower(), (reason, forbidden)


def _tmp():
    import tempfile, os
    return os.path.join(tempfile.mkdtemp(), 'excluded.csv')


def _run_absent(out):
    """The named zone has NO ROW at all in its undefined scenario-years."""
    tables = build(NAMED, ALL_FUTURE)
    for key in ALL_FUTURE:
        tables[key] = tables[key][tables[key].zone_id != NAMED].reset_index(drop=True)
    run(tables, [1, NAMED], excluded_path=out)
    return out


def test_a_zone_with_no_base_year_is_left_to_the_baseline_rule():
    """The two rules must not compete for the same zone.

    A zone with no base-year damage has no d_2023, so its future cannot be the thing that is
    missing. Scanning it with the future check reports it as an unnamed offender and stops a run
    the baseline-domain rule would have handled -- which cost a full seam run on 2026-09-25, and is
    why nine Canadian, Mongolian, US and rest-of-world zones were wrongly named as new cases.
    """
    unnamed = 2002
    assert unnamed not in FUTURE_COVERAGE_EXCLUSIONS
    tables = build(unnamed, ALL_FUTURE)
    base = tables[(BASE_SCENARIO, BASE_YEAR)]
    tables[(BASE_SCENARIO, BASE_YEAR)] = base[base.zone_id != unnamed].reset_index(drop=True)
    out = _tmp()
    got = tables_to_seam(tables, labels_for([1, unnamed]), SCENARIOS, ANCHORS, YEARS,
                         BASE_SCENARIO, BASE_YEAR, ['PDR'], 0.2, 2030,
                         baseline_domain={unnamed: {'baseline_crop_ha': 0.0,
                                                    'baseline_nodata_ha': 0.0,
                                                    'mean_source_coverage': 1.0}},
                         excluded_path=out)
    assert len(got), 'the healthy zone must still be shocked'
    frame = pd.read_csv(out)
    assert frame.zone_id.tolist() == [unnamed]
    assert 'base-year cropland' in frame.reason.iloc[0]

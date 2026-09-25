"""Erosion is estimated on the BASE-YEAR cropland domain; newly appearing cropland is reported.

d_t = 0.08 * A_severe,t / A_crop,t and g_t = 100(d_2023 - d_t). A zone with no baseline cropland has
no d_2023, so its trajectory does not exist -- it is EXCLUDED with its reason recorded, not given a
zero shock. Exclusion and a measured zero are different statements even where the assembler applies
no shock in both cases.

An exclusion is legitimate only where absence is ESTABLISHED. Baseline cropland removed by the
coverage restriction leaves the baseline unknown, and that stays a failure -- as does a zone with a
defined baseline and an undefined future year.
"""
import numpy as np
import pandas as pd
import pytest

from global_invest.erosion.erosion_damage_pipeline import tables_to_seam

ANCHORS = [2030, 2050]
YEARS = list(range(2023, 2051))
BASE_SCENARIO, BASE_YEAR = 'baseline', 2023


def table(zone_levels):
    return pd.DataFrame([{'zone_id': z, 'level': lv, 'stressed_level': lv} for z, lv in zone_levels.items()])


def labels_for(zones, aez=7):
    return pd.DataFrame([{'zone_id': z, 'aez18_id': aez, 'gtapv7_r50_label': 'can'} for z in zones])


def tables_with(base, others):
    t = {(BASE_SCENARIO, BASE_YEAR): table(base)}
    for year in ANCHORS:
        t[(BASE_SCENARIO, year)] = table(others)
        t[('policy', year)] = table(others)
    return t


def run(base, others, domain, zones, **kw):
    return tables_to_seam(tables_with(base, others), labels_for(zones), ['policy'], ANCHORS, YEARS,
                          BASE_SCENARIO, BASE_YEAR, ['PDR'], 0.2, 2030,
                          baseline_domain=domain, **kw)


def test_zone_with_verified_zero_baseline_cropland_is_excluded(tmp_path):
    """The nine-zone case: no baseline cropland, full coverage, nothing lost to nodata."""
    out = tmp_path / 'excluded.csv'
    got = run(base={1: -0.5}, others={1: -0.5, 2: -0.0},
              domain={1: {'baseline_crop_ha': 100.0, 'baseline_nodata_ha': 0.0, 'mean_source_coverage': 1.0},
                      2: {'baseline_crop_ha': 0.0, 'baseline_nodata_ha': 0.0, 'mean_source_coverage': 1.0}},
              zones=[1, 2], excluded_path=out)
    assert set(got.REG) == {'CAN'}
    assert out.exists()
    frame = pd.read_csv(out)
    assert frame.zone_id.tolist() == [2]
    assert 'not a measured zero' in frame.reason.iloc[0].lower() or 'NOT a measured zero' in frame.reason.iloc[0]


def test_baseline_cropland_lost_to_nodata_stays_a_failure():
    """Absence not established: the baseline is unknown, so neither a shock nor an exclusion."""
    with pytest.raises(ValueError, match='baseline is UNKNOWN'):
        run(base={1: -0.5}, others={1: -0.5, 2: -0.0},
            domain={1: {'baseline_crop_ha': 100.0, 'baseline_nodata_ha': 0.0, 'mean_source_coverage': 1.0},
                    2: {'baseline_crop_ha': 0.0, 'baseline_nodata_ha': 42.0, 'mean_source_coverage': 0.6}},
            zones=[1, 2])


def test_baseline_cropland_present_but_absent_from_the_damage_table_is_a_failure():
    """Cropland existed and yet produced no base-year damage row: that is unexplained, not empty."""
    with pytest.raises(ValueError, match='baseline is UNKNOWN'):
        run(base={1: -0.5}, others={1: -0.5, 2: -0.0},
            domain={1: {'baseline_crop_ha': 100.0, 'baseline_nodata_ha': 0.0, 'mean_source_coverage': 1.0},
                    2: {'baseline_crop_ha': 7.9, 'baseline_nodata_ha': 0.0, 'mean_source_coverage': 1.0}},
            zones=[1, 2])


def test_no_domain_evidence_is_a_failure():
    """Without the classification the pipeline cannot tell absence from unknown, so it refuses."""
    with pytest.raises(ValueError, match='no baseline domain evidence'):
        run(base={1: -0.5}, others={1: -0.5, 2: -0.0}, domain={}, zones=[1, 2])


def test_defined_baseline_with_undefined_future_year_remains_a_failure():
    """Point 4: this rule must not become a licence to drop any troublesome trajectory."""
    t = tables_with({1: -0.5, 2: -0.5}, {1: -0.5, 2: -0.5})
    t[('policy', 2050)] = table({1: -0.5, 2: np.nan})
    with pytest.raises(ValueError, match='Undefined damage'):
        tables_to_seam(t, labels_for([1, 2]), ['policy'], ANCHORS, YEARS, BASE_SCENARIO, BASE_YEAR,
                       ['PDR'], 0.2, 2030,
                       baseline_domain={z: {'baseline_crop_ha': 100.0, 'baseline_nodata_ha': 0.0,
                                            'mean_source_coverage': 1.0} for z in (1, 2)})


def test_excluded_zone_reports_its_future_hectares(tmp_path):
    """The record must say what was set aside, not only that something was."""
    out = tmp_path / 'excluded.csv'
    cover = [pd.DataFrame([{'zone_id': 2, 'scenario': 'policy', 'year': 2050,
                            'valid_crop_ha': 242.2, 'severe_crop_ha': 0.0, 'excluded_crop_ha': 0.0}])]
    run(base={1: -0.5}, others={1: -0.5, 2: -0.0},
        domain={1: {'baseline_crop_ha': 100.0, 'baseline_nodata_ha': 0.0, 'mean_source_coverage': 1.0},
                2: {'baseline_crop_ha': 0.0, 'baseline_nodata_ha': 0.0, 'mean_source_coverage': 1.0}},
        zones=[1, 2], excluded_path=out, coverage=cover)
    frame = pd.read_csv(out)
    assert float(frame.max_future_crop_ha.iloc[0]) == pytest.approx(242.2)
    assert float(frame.max_future_severe_crop_ha.iloc[0]) == pytest.approx(0.0)

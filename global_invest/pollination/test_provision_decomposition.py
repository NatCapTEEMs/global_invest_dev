"""Tests for the retained/lost/new provision decomposition."""

import numpy as np
import pytest

from global_invest.pollination import provision_decomposition as pd_mod
from global_invest.pollination.retained_value import retained_value_change


def test_hectare_components_partition_baseline_cropland():
    base = np.array([[2, 2, 3], [2, 3, 3]])
    future = np.array([[2, 3, 2], [2, 3, 3]])
    ha = np.full((2, 3), 10.0)
    got = pd_mod.component_hectares(base, future, ha)
    assert got['retained_ha'] == 20.0            # (0,0) and (1,0)
    assert got['lost_ha'] == 10.0                # (0,1) crop -> non-crop
    assert got['new_ha'] == 10.0                 # (0,2) non-crop -> crop
    assert got['retained_ha'] + got['lost_ha'] == got['base_cropland_ha']
    assert got['retained_ha'] + got['new_ha'] == got['future_cropland_ha']


def test_retained_term_matches_the_fed_measure():
    """The decomposition must not quietly redefine the estimand it is diagnosing."""
    value, base_ha, retained_ha = 1000.0, 100.0, 60.0
    suff_base, suff_future = 0.4, 0.7
    got = pd_mod.decompose(value, retained_ha, 40.0, 0.0, base_ha, suff_base, suff_future)
    fed = retained_value_change(value, retained_ha / base_ha, suff_base, suff_future)
    assert np.isclose(got['retained_change'], fed)


def test_identity_reconciles():
    got = pd_mod.decompose(500.0, 50.0, 30.0, 20.0, 80.0, 0.5, 0.6)
    summary = pd_mod.reconcile(got)
    assert np.isclose(summary['total_change'],
                      summary['retained_change'] + summary['new_provision'] - summary['lost_provision'])


def test_new_cropland_without_a_base_rate_is_reported_not_zeroed():
    """A cell with no baseline cropland cannot yield a value per hectare; the area must survive."""
    got = pd_mod.decompose(0.0, 0.0, 0.0, 25.0, 0.0, 0.3, 0.9)
    assert got['new_provision'] == 0.0
    assert got['unvalued_new_ha'] == 25.0
    summary = pd_mod.reconcile(got)
    assert summary['unvalued_new_ha'] == 25.0
    assert 'no defensible crop value' in summary['caveat']


def test_unvalued_assumption_declines_to_value_any_new_land():
    got = pd_mod.decompose(1000.0, 50.0, 0.0, 50.0, 50.0, 0.5, 0.5,
                           new_cropland_valuation='unvalued')
    assert got['new_provision'] == 0.0
    assert got['unvalued_new_ha'] == 50.0


def test_one_sided_sufficiency_coverage_contributes_nothing():
    """A component computed where only one year has sufficiency would difference against nothing."""
    got = pd_mod.decompose(1000.0, 50.0, 10.0, 10.0, 60.0, np.nan, 0.8)
    assert got['retained_change'] == 0.0
    assert got['lost_provision'] == 0.0
    assert got['new_provision'] == 0.0


def test_components_that_overrun_baseline_cropland_are_rejected():
    with pytest.raises(ValueError, match='do not partition'):
        pd_mod.decompose(100.0, 60.0, 60.0, 0.0, 100.0, 0.5, 0.5)


def test_unknown_valuation_assumption_is_rejected():
    with pytest.raises(ValueError, match='unknown new-cropland valuation'):
        pd_mod.decompose(100.0, 10.0, 0.0, 0.0, 10.0, 0.5, 0.5, new_cropland_valuation='zero')


def test_label_says_valued_support_when_some_new_cropland_is_unvalued():
    got = pd_mod.decompose(0.0, 0.0, 0.0, 25.0, 0.0, 0.3, 0.9)
    summary = pd_mod.reconcile(got)
    assert summary['quantity'] == 'provision change on valued support'


def test_label_says_total_only_when_everything_is_valued_and_covered():
    got = pd_mod.decompose(1000.0, 60.0, 40.0, 0.0, 100.0, 0.4, 0.7)
    summary = pd_mod.reconcile(got)
    assert summary['quantity'] == 'total provision change'
    assert summary['unvalued_new_ha'] == 0.0
    assert summary['excluded_for_missing_sufficiency_ha'] == 0.0


def test_area_excluded_for_missing_sufficiency_is_reported_not_dropped():
    got = pd_mod.decompose(1000.0, 50.0, 10.0, 10.0, 60.0, np.nan, 0.8)
    summary = pd_mod.reconcile(got)
    assert summary['excluded_for_missing_sufficiency_ha'] == 70.0
    assert summary['quantity'] == 'provision change on valued support'

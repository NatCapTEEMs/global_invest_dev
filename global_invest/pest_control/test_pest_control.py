"""Unit tests for pest_control's organic cleaning, valuation and Monte Carlo.

Every function is pinned on a hand-built frame, so the benefit convention, the cleaning rules, the
zero and missing rules and the draw structure are stated as executable facts, and the staged
inputs anchor the joins and the published figures.
"""
import csv
import os

import numpy as np
import pandas as pd
import pytest

from global_invest.pest_control import pest_control_functions as pc
from global_invest.pest_control import pest_control_tasks

FRUIT = 'Fruit and nuts'
EFFECT = {'cereals': (0.13, 0.16), 'vegetables': (0.23, 0.10), 'others': (0.28, 0.12)}
YIELD_RATIO = {'cereals': (-0.29, 0.03), 'fruit_nuts': (-0.13, 0.05), 'vegetables': (-0.19, 0.06),
               'pulses': (-0.16, 0.06), 'oilseeds': (-0.19, 0.07), 'stimulants': (-0.16, 0.10)}
ROOTS = 'Edible roots and tubers with high starch or inulin content'
GROUPS = {'Apples': FRUIT, 'Pears': FRUIT, 'Wheat': 'Cereals', 'Potatoes': ROOTS, 'Shea': 'Other oleaginous fruits',
          'Mystery': 'A group in neither table'}
NAMES = {'A': 'AAA', 'B': 'BBB'}
PREMIUMS = pd.DataFrame({'country': ['A', 'A', 'A', 'A', 'A'], 'item': ['Apples', 'Apples', 'Pears', 'Wheat', 'Wheat'],
                         'year': [2019, 2011, 2012, 2012, 2013], 'FAO_group': [FRUIT, FRUIT, FRUIT, 'Cereals', 'Cereals'],
                         'Org_premium': [1.5, 1.0, 1.1, 1.3, 1.1]})


def fibl_rows(*rows):
    return pd.DataFrame(rows, columns=['country', 'year', 'crop', 'organic_ha'])


def harvested_rows(*rows):
    return pd.DataFrame(rows, columns=['area_fao', 'item', 'year', 'harvested_ha'])


# A record of another crop and year, which every cleaning test ignores.
NO_RECENT = fibl_rows(('Norway', 2005, 'Oats', 1.0))


def value_table(organic, fao, include_cereals=False):
    return pc.pest_control_by_crop_country(
        pd.DataFrame(organic, columns=['area_fao', 'item', 'organic_ha']),
        pd.DataFrame(fao, columns=['area_fao', 'item', 'yield_t_ha', 'price_usd_t', 'price_level']),
        PREMIUMS, GROUPS, NAMES, YIELD_RATIO, EFFECT, include_cereals, 2019)


def test_the_benefit_is_the_share_of_observed_production():
    # lnRR 0.2829220 (Boldorini Table S3, other crops): about a quarter of the observed harvest is
    # owed to the predators, smaller than exp(lnRR) - 1.
    assert np.isclose(pc.yield_benefit_from_lnrr(0.2829220), 1.0 - np.exp(-0.2829220))
    assert pc.yield_benefit_from_lnrr(0.0) == 0.0
    assert pc.yield_benefit_from_lnrr(0.2829220) < np.exp(0.2829220) - 1.0


def test_the_value_is_benefit_times_organic_production_times_organic_price():
    d = value_table([('A', 'Apples', 100.0)], [('A', 'Apples', 2.0, 500.0, 'country')])
    expected = (1.0 - np.exp(-0.28)) * 100.0 * 2.0 * np.exp(-0.13) * 500.0 * 1.5
    assert np.isclose(d['pest_control_value'].iloc[0], expected)
    assert d['note'].iloc[0] == '' and d['iso3_r250_label'].iloc[0] == 'AAA'


def test_a_matched_price_ratio_applies_only_to_its_crop_country_and_year():
    d = value_table([('A', 'Apples', 1.0), ('B', 'Apples', 1.0), ('A', 'Pears', 1.0)],
                    [('A', 'Apples', 1.0, 1.0, 'country'), ('B', 'Apples', 1.0, 1.0, 'country'),
                     ('A', 'Pears', 1.0, 1.0, 'country')])
    group_mean = np.mean([1.5, 1.0, 1.1])
    assert d['price_ratio_source'].tolist() == ['observed', 'group_mean', 'group_mean']
    assert np.allclose(d['price_ratio'], [1.5, group_mean, group_mean])


def test_a_crop_group_without_a_benefit_is_a_zero_and_needs_no_ratios():
    d = value_table([('A', 'Potatoes', 50.0)], [('A', 'Potatoes', 20.0, 200.0, 'country')])
    assert d['pest_control_value'].iloc[0] == 0.0 and d['note'].iloc[0] == ''
    assert d['benefit_status'].iloc[0] == 'no_benefit'


def test_an_unclassified_crop_is_missing_never_a_zero():
    d = value_table([('A', 'Mystery', 5.0), ('A', 'Not in the classification', 5.0)],
                    [('A', 'Mystery', 2.0, 9.0, 'country'), ('A', 'Not in the classification', 2.0, 9.0, 'country')])
    assert d['pest_control_value'].isna().all()
    assert d['note'].tolist() == ['no crop group', 'no crop group']
    assert d['benefit_status'].tolist() == ['unclassified', 'unclassified']


def test_cereals_are_a_zero_unless_the_configuration_values_them():
    fao = [('A', 'Wheat', 3.0, 200.0, 'country')]
    assert value_table([('A', 'Wheat', 10.0)], fao)['pest_control_value'].iloc[0] == 0.0
    expected = (1.0 - np.exp(-0.13)) * 10.0 * 3.0 * np.exp(-0.29) * 200.0 * 1.2
    assert np.isclose(value_table([('A', 'Wheat', 10.0)], fao, include_cereals=True)['pest_control_value'].iloc[0], expected)


def test_a_missing_factor_leaves_a_named_na_never_a_zero():
    d = value_table([('A', 'Apples', 1.0), ('B', 'Apples', 1.0), ('A', 'Pears', 1.0), ('A', 'Potatoes', 1.0), ('A', 'Shea', 1.0)],
                    [('A', 'Apples', np.nan, 9.0, 'country'), ('B', 'Apples', 1.0, np.nan, 'missing'),
                     ('A', 'Potatoes', np.nan, 9.0, 'country'), ('A', 'Shea', 1.0, 9.0, 'country')]).set_index(['area_fao', 'item'])
    assert d['pest_control_value'].isna().all()
    assert d.loc[('A', 'Apples'), 'note'] == 'no yield'
    assert d.loc[('B', 'Apples'), 'note'] == 'no price'
    assert d.loc[('A', 'Pears'), 'note'] == 'no yield; no price'
    assert d.loc[('A', 'Potatoes'), 'note'] == 'no yield'
    assert d.loc[('A', 'Shea'), 'note'] == 'no yield ratio; no price ratio'


def test_a_zero_yield_or_price_counts_as_missing():
    d = value_table([('A', 'Apples', 1.0)], [('A', 'Apples', 0.0, 9.0, 'country')])
    assert pd.isna(d['pest_control_value'].iloc[0]) and d['note'].iloc[0] == 'no yield'


def test_a_repeated_faostat_key_raises_instead_of_doubling():
    with pytest.raises(pd.errors.MergeError):
        value_table([('A', 'Apples', 1.0)], [('A', 'Apples', 1.0, 1.0, 'country'), ('A', 'Apples', 2.0, 1.0, 'country')])


def test_a_repeated_organic_key_raises_instead_of_doubling():
    with pytest.raises(pd.errors.MergeError):
        value_table([('A', 'Apples', 1.0), ('A', 'Apples', 2.0)], [('A', 'Apples', 1.0, 1.0, 'country')])


def test_a_repeated_matched_price_ratio_raises_instead_of_doubling():
    repeated = pd.concat([PREMIUMS, PREMIUMS[PREMIUMS['year'] == 2019]], ignore_index=True)
    with pytest.raises(pd.errors.MergeError):
        pc.pest_control_by_crop_country(
            pd.DataFrame([('A', 'Apples', 1.0)], columns=['area_fao', 'item', 'organic_ha']),
            pd.DataFrame([('A', 'Apples', 1.0, 1.0, 'country')], columns=['area_fao', 'item', 'yield_t_ha', 'price_usd_t', 'price_level']),
            repeated, GROUPS, NAMES, YIELD_RATIO, EFFECT, False, 2019)


def test_a_repeated_harvested_area_raises_instead_of_doubling():
    areas = fibl_rows(('Spain', 2019, 'Apples', 1.0))
    harvest = harvested_rows(('Spain', 'Apples', 2019, 5.0), ('Spain', 'Apples', 2019, 6.0))
    with pytest.raises(pd.errors.MergeError):
        pc.clean_organic_areas(areas, NO_RECENT, {'Apples'}, harvest, 2019)


def test_a_fibl_cell_gives_its_number_or_the_first_of_two():
    assert pc.organic_area_from_cell(12.5) == 12.5
    assert pc.organic_area_from_cell('15899.42, 493.11') == 15899.42
    assert pc.organic_area_from_cell("8975.23, 1'342.20") == 8975.23
    assert np.isnan(pc.organic_area_from_cell('-')) and np.isnan(pc.organic_area_from_cell(np.nan))


def test_cleaning_recodes_sums_and_reports_what_leaves_the_area():
    fibl = fibl_rows(('Turkey', 2019, 'Almonds', 5.0), ('Turkey', 2019, 'Peaches', 2.0), ('Turkey', 2019, 'Nectarines', 3.0),
                     ('Kosovo', 2019, 'Almonds', 7.0), ('Turkey', 2019, 'Spelt', 11.0), ('Turkey', 2019, 'Fruit, temperate, other', 13.0),
                     ('Turkey', 2018, 'Almonds', 4.0))
    items = {'Almonds, in shell', 'Peaches and nectarines'}
    out, info = pc.clean_organic_areas(fibl, NO_RECENT, items, harvested_rows(), 2019)
    assert out.set_index('item')['organic_ha'].to_dict() == {'Almonds, in shell': 5.0, 'Peaches and nectarines': 5.0}
    assert set(out['area_fao']) == {'Türkiye'}
    assert info['total_ha'] == 41.0 and info['no_fao_country_ha'] == 7.0
    assert info['no_fao_crop_ha'] == 11.0 and info['aggregate_ha'] == 13.0


def test_cleaning_adds_recent_statistics_and_drops_negative_and_zero_areas():
    fibl = fibl_rows(('Spain', 2019, 'Almonds', -3.0), ('Spain', 2019, 'Apples', 0.0))
    recent = fibl_rows(('Spain', 2019, 'Pears', 4.0))
    out, _ = pc.clean_organic_areas(fibl, recent, {'Almonds, in shell', 'Apples', 'Pears'}, harvested_rows(), 2019)
    assert out[['item', 'organic_ha']].values.tolist() == [['Pears', 4.0]]


def test_an_isolated_record_above_neighbours_and_harvest_takes_their_mean():
    areas = fibl_rows(('Spain', 2018, 'Apples', 2.0), ('Spain', 2019, 'Apples', 100.0), ('Spain', 2020, 'Apples', 4.0))
    harvest = harvested_rows(('Spain', 'Apples', 2019, 50.0))
    out, info = pc.clean_organic_areas(areas, NO_RECENT, {'Apples'}, harvest, 2019)
    assert out['organic_ha'].tolist() == [3.0]
    assert info['spikes']['replacement_ha'].tolist() == [3.0]


def test_a_large_record_within_the_harvested_area_is_a_real_expansion():
    areas = fibl_rows(('Spain', 2018, 'Apples', 2.0), ('Spain', 2019, 'Apples', 100.0), ('Spain', 2020, 'Apples', 4.0))
    out, info = pc.clean_organic_areas(areas, NO_RECENT, {'Apples'}, harvested_rows(('Spain', 'Apples', 2019, 500.0)), 2019)
    assert out['organic_ha'].tolist() == [100.0] and info['spikes'].empty


def test_cleaning_lists_what_it_leaves_out_by_country_and_reason():
    fibl = fibl_rows(('Kosovo', 2019, 'Almonds', 7.0), ('Turkey', 2019, 'Spelt', 11.0), ('Turkey', 2019, 'Fruit, temperate, other', 13.0),
                     ('Turkey', 2019, 'Almonds', 5.0), ('Turkey', 2018, 'Spelt', 99.0))
    _, info = pc.clean_organic_areas(fibl, NO_RECENT, {'Almonds, in shell'}, harvested_rows(), 2019)
    assert info['left_out'].values.tolist() == [['Kosovo', 'no FAOSTAT crop series', 7.0], ['Türkiye', 'no FAOSTAT counterpart', 11.0],
                                                ['Türkiye', 'FAOSTAT aggregate', 13.0]]


def test_a_series_end_takes_its_one_neighbour_and_a_gap_is_not_checked():
    harvest = harvested_rows(('Spain', 'Apples', 2019, 1.0), ('Italy', 'Apples', 2019, 1.0), ('Greece', 'Apples', 2019, 1.0))
    areas = fibl_rows(('Spain', 2018, 'Apples', 2.0), ('Spain', 2019, 'Apples', 100.0),
                      ('Italy', 2019, 'Apples', 100.0), ('Italy', 2020, 'Apples', 5.0),
                      ('Greece', 2017, 'Apples', 2.0), ('Greece', 2019, 'Apples', 100.0), ('Greece', 2021, 'Apples', 2.0))
    out, _ = pc.clean_organic_areas(areas, NO_RECENT, {'Apples'}, harvest, 2019)
    assert out.set_index('area_fao')['organic_ha'].to_dict() == {'Greece': 100.0, 'Italy': 5.0, 'Spain': 2.0}


def test_reference_masks_separate_a_zero_from_a_missing_value():
    table = pd.DataFrame({'iso3_r250_label': ['A', 'B', 'C', 'D', 'E'], 'here': [1.0, 0.0, 2.0, np.nan, 0.0],
                          'there': [1.0, np.nan, np.nan, 3.0, 4.0]})
    masks = pc.reference_masks(table, 'here', 'there')
    assert masks == {'both': ['A'], 'zero_here': ['B'], 'value_here': ['C'], 'value_there': ['D', 'E']}


def coverage_rows():
    return value_table(
        [('A', 'Apples', 1.0), ('A', 'Pears', 2.0), ('B', 'Potatoes', 4.0), ('B', 'Pears', 8.0),
         ('C', 'Potatoes', 16.0), ('D', 'Apples', 32.0), ('E', 'Pears', 64.0), ('E', 'Potatoes', 128.0), ('F', 'Mystery', 256.0)],
        [('A', 'Apples', 1.0, 1.0, 'country'), ('B', 'Potatoes', 1.0, 1.0, 'country'), ('C', 'Potatoes', 1.0, 1.0, 'country'),
         ('E', 'Pears', 1.0, 1.0, 'country'), ('F', 'Mystery', 1.0, 1.0, 'country')]).assign(iso3_r250_label=lambda f: f['area_fao'])


def test_coverage_separates_a_complete_total_from_a_partial_one_and_a_zero_from_an_unknown():
    c = pc.coverage_by_country(coverage_rows())
    assert c['coverage'].to_dict() == {'A': 'partial_value', 'B': 'zero_with_missing', 'C': 'complete_zero', 'D': 'missing_inputs',
                                       'E': 'complete_value', 'F': 'missing_inputs'}
    assert c.loc['A', ['crops', 'crops_valued', 'crops_no_benefit', 'crops_missing']].tolist() == [2, 1, 0, 1]
    assert c.loc['A', 'area_missing_ha'] == 2.0 and c.loc['A', 'area_valued_ha'] == 1.0
    assert c.loc['E', 'crops_no_benefit'] == 1 and c.loc['E', 'crops_missing'] == 0 and c.loc['E', 'area_no_benefit_ha'] == 128.0


def test_a_crop_without_a_benefit_lacking_inputs_is_not_a_missing_contribution():
    rows = value_table([('A', 'Apples', 1.0), ('A', 'Potatoes', 5.0)], [('A', 'Apples', 1.0, 1.0, 'country')]).assign(iso3_r250_label='A')
    c = pc.coverage_by_country(rows).loc['A']
    assert rows['pest_control_value'].isna().tolist() == [False, True]
    assert c['coverage'] == 'complete_value' and c['crops_missing'] == 0 and c['crops_no_benefit'] == 1


def test_an_unclassified_crop_counts_as_a_missing_contribution():
    assert pc.coverage_by_country(coverage_rows()).loc['F', 'crops_missing'] == 1


def test_only_a_zero_beside_missing_crops_loses_its_value():
    rows = coverage_rows()
    table = pd.DataFrame({'iso3_r250_label': list('ABCDEFGH'), 'v': [5.0, 0.0, 0.0, np.nan, 6.0, np.nan, np.nan, np.nan]})
    out = pc.qualified_country_table(table, pc.coverage_by_country(rows), pd.Series({'G': 3.0}), ['v']).set_index('iso3_r250_label')
    assert out['v'].iloc[:3].tolist()[0] == 5.0 and pd.isna(out.loc['B', 'v']) and out.loc['C', 'v'] == 0.0 and out.loc['E', 'v'] == 6.0
    assert out.loc['G', 'pest_control_coverage'] == 'left_out_area_only' and out.loc['H', 'pest_control_coverage'] == 'no_organic_area'
    assert out.loc['G', 'pest_control_area_left_out_ha'] == 3.0 and out.loc['H', 'pest_control_crops'] == 0
    assert out.loc['B', 'pest_control_crops_missing'] == 1


def test_the_summary_splits_value_by_coverage_and_reports_the_area_share():
    table = pd.DataFrame({'iso3_r250_label': list('ABCDEFGH'), 'v': [5.0, 0.0, 0.0, np.nan, 6.0, np.nan, np.nan, np.nan]})
    out = pc.qualified_country_table(table, pc.coverage_by_country(coverage_rows()), pd.Series({'G': 3.0}), ['v'])
    info = {'total_ha': 1.0, 'no_fao_country_ha': 0.0, 'no_fao_crop_ha': 0.0, 'aggregate_ha': 0.0,
            'spikes': pd.DataFrame({'organic_ha': [9.0], 'replacement_ha': [4.0]})}
    stats = pc.coverage_summary(out, info, 'v').set_index('statistic')['value']
    assert stats['value_usd_partial_value'] == 5.0 and stats['value_usd_complete_value'] == 6.0
    assert stats['countries_missing_inputs'] == 2 and stats['countries_zero_with_missing'] == 1
    assert stats['crops_missing'] == out['pest_control_crops_missing'].sum() and stats['area_spike_correction_ha'] == 5.0
    assert stats['area_share_with_a_benefit_valued'] == stats['area_valued_ha'] / (stats['area_valued_ha'] + stats['area_missing_ha'])


def test_the_group_price_ratio_parameters_follow_the_delta_method():
    values = [1.5, 1.0, 1.1]
    params = pc.premium_parameters_by_group(PREMIUMS).loc[FRUIT]
    standard_error = np.std(values, ddof=1) / np.sqrt(3)
    assert np.isclose(params['log_mean'], np.log(np.mean(values)))
    assert np.isclose(params['log_se'], standard_error / np.mean(values))


def monte_carlo_rows():
    return value_table([('A', 'Apples', 10.0), ('A', 'Pears', 20.0), ('B', 'Pears', 30.0), ('B', 'Potatoes', 5.0)],
                       [('A', 'Apples', 2.0, 100.0, 'country'), ('A', 'Pears', 3.0, 200.0, 'country'),
                        ('B', 'Pears', 4.0, 300.0, 'country'), ('B', 'Potatoes', 9.0, 9.0, 'country')])


def premium_parameters():
    return pc.premium_parameters_by_group(PREMIUMS)


def test_with_no_uncertainty_the_draws_equal_the_central_value():
    rows = monte_carlo_rows()
    effect = {k: (v[0], 0.0) for k, v in EFFECT.items()}
    ratio = {k: (v[0], 0.0) for k, v in YIELD_RATIO.items()}
    params = premium_parameters().assign(log_se=0.0)
    labels, totals = pc.monte_carlo_by_country(rows, ratio, params, effect, 3, 1)
    central = rows.groupby('iso3_r250_label')['pest_control_value'].sum()
    assert labels.tolist() == ['AAA', 'BBB']
    assert np.allclose(totals, central.loc[labels].to_numpy())


def test_the_monte_carlo_mean_converges_on_the_exact_expectation():
    rows = monte_carlo_rows()
    labels, totals = pc.monte_carlo_by_country(rows, YIELD_RATIO, premium_parameters(), EFFECT, 200_000, 7)
    expected = pc.expected_value_by_country(rows, YIELD_RATIO, premium_parameters(), EFFECT).loc[labels].to_numpy()
    standard_error = totals.std(axis=0) / np.sqrt(totals.shape[0])
    assert (np.abs(totals.mean(axis=0) - expected) < 4 * standard_error).all()


def test_the_exact_expectation_is_the_hand_written_lognormal_mean():
    rows = value_table([('A', 'Apples', 1.0)], [('A', 'Apples', 1.0, 1.0, 'country')])
    expected = pc.expected_value_by_country(rows, YIELD_RATIO, premium_parameters(), EFFECT).loc['AAA']
    by_hand = (1.0 - np.exp(-0.28 + 0.12 ** 2 / 2)) * np.exp(-0.13 + 0.05 ** 2 / 2) * 1.5
    assert np.isclose(expected, by_hand)


def test_one_draw_per_group_and_category_is_shared_across_countries():
    # With the same crop and a group-mean price ratio in two countries, every simulation moves both
    # by the same factor, so the errors do not cancel when countries are summed.
    rows = value_table([('A', 'Pears', 1.0), ('B', 'Pears', 3.0)], [('A', 'Pears', 1.0, 1.0, 'country'), ('B', 'Pears', 1.0, 1.0, 'country')])
    _, totals = pc.monte_carlo_by_country(rows, YIELD_RATIO, premium_parameters(), EFFECT, 50, 3)
    assert np.allclose(totals[:, 1] / totals[:, 0], 3.0)


def test_groups_in_one_effect_category_share_the_benefit_draw():
    # Fruit and stimulants take the same category, so with the other draws fixed their ratio is constant.
    groups = {'Pears': FRUIT, 'Tea': 'Stimulant, spice and aromatic crops'}
    premiums = pd.DataFrame({'country': ['A', 'A'], 'item': ['Pears', 'Tea'], 'year': [2011, 2011],
                             'FAO_group': [FRUIT, 'Stimulant, spice and aromatic crops'], 'Org_premium': [1.2, 1.2]})
    rows = pc.pest_control_by_crop_country(
        pd.DataFrame([('A', 'Pears', 1.0), ('B', 'Tea', 1.0)], columns=['area_fao', 'item', 'organic_ha']),
        pd.DataFrame([('A', 'Pears', 1.0, 1.0, 'country'), ('B', 'Tea', 1.0, 1.0, 'country')],
                     columns=['area_fao', 'item', 'yield_t_ha', 'price_usd_t', 'price_level']),
        premiums, groups, NAMES, YIELD_RATIO, EFFECT, False, 2019)
    ratio = {k: (v[0], 0.0) for k, v in YIELD_RATIO.items()}
    params = pc.premium_parameters_by_group(premiums).assign(log_se=0.0)
    _, totals = pc.monte_carlo_by_country(rows, ratio, params, EFFECT, 50, 3)
    assert np.allclose(totals[:, 1] / totals[:, 0], np.exp(-0.16) / np.exp(-0.13))


def test_a_matched_price_ratio_is_fixed_across_draws():
    rows = value_table([('A', 'Apples', 1.0)], [('A', 'Apples', 1.0, 1.0, 'country')])
    ratio = {k: (v[0], 0.0) for k, v in YIELD_RATIO.items()}
    effect = {k: (v[0], 0.0) for k, v in EFFECT.items()}
    _, totals = pc.monte_carlo_by_country(rows, ratio, premium_parameters(), effect, 20, 5)
    assert np.allclose(totals, totals[0])


def test_a_country_of_zeros_stays_zero_and_an_unvalued_country_is_absent():
    rows = value_table([('A', 'Potatoes', 5.0), ('B', 'Apples', 5.0)], [('A', 'Potatoes', 9.0, 9.0, 'country')])
    labels, totals = pc.monte_carlo_by_country(rows, YIELD_RATIO, premium_parameters(), EFFECT, 10, 2)
    assert labels.tolist() == ['AAA'] and (totals == 0.0).all()
    expected = pc.expected_value_by_country(rows, YIELD_RATIO, premium_parameters(), EFFECT)
    assert expected.to_dict() == {'AAA': 0.0}


def test_a_benefit_draw_below_zero_is_kept_as_drawn():
    rows = value_table([('A', 'Wheat', 10.0)], [('A', 'Wheat', 1.0, 1.0, 'country')], include_cereals=True)
    _, totals = pc.monte_carlo_by_country(rows, YIELD_RATIO, premium_parameters(), EFFECT, 500, 2)
    assert (totals < 0).any() and (totals > 0).any()


def test_a_seed_fixes_the_draws_to_the_cent():
    rows = monte_carlo_rows()
    _, first = pc.monte_carlo_by_country(rows, YIELD_RATIO, premium_parameters(), EFFECT, 100, 4242)
    _, again = pc.monte_carlo_by_country(rows, YIELD_RATIO, premium_parameters(), EFFECT, 100, 4242)
    _, other = pc.monte_carlo_by_country(rows, YIELD_RATIO, premium_parameters(), EFFECT, 100, 4243)
    assert (first == again).all() and not (first == other).all()
    assert round(float(first[0, 0]), 2) == FROZEN_STREAM_FIRST_DRAW


FROZEN_STREAM_FIRST_DRAW = 3383.96


def staged_parameters():
    params = {}
    with open(os.path.join(os.path.dirname(pest_control_tasks.__file__), '..', 'input_template', 'es_parameters.csv'),
              encoding='utf-8-sig') as fh:
        for row in csv.reader(fh):
            if row and row[0] == 'pest_control':
                params[row[1]] = row[2]
    return params


def staged_base_data():
    import tempfile
    import hazelbean as hb
    from global_invest import utilities
    p = hb.ProjectFlow(project_dir=os.path.join(tempfile.mkdtemp(), 'anchors'))
    return p, utilities.service_data_dir(p, 'pest_control'), utilities.service_data_dir(p, 'pollination')


def staged_classification(base):
    return pd.read_csv(os.path.join(base, 'value_org_fb40d8c', 'FAO_classification.csv'), encoding='utf-8-sig')


def test_the_staged_classification_places_potatoes_in_roots_and_tubers():
    _, base, _ = staged_base_data()
    groups = staged_classification(base).set_index('item')['FAO_group']
    assert groups['Potatoes'].startswith('Edible roots and tubers')
    assert groups['Potatoes'] not in pc.EFFECT_CATEGORY_BY_GROUP


def test_every_mapped_crop_is_a_faostat_item_or_a_named_aggregate():
    _, base, _ = staged_base_data()
    cls = staged_classification(base)
    known = set(cls['item']) | set(cls['FAO_group']) | set(cls['FAO_class']) | set(cls['FAO_subclass'].dropna())
    mapped = {v for v in pc.FIBL_CROP_TO_FAO.values() if v is not None}
    assert mapped - known <= {'Fruit Primary', 'Pulses, Total', 'Vegetables Primary'}


def test_every_classified_group_is_valued_or_explicitly_without_a_benefit():
    _, base, _ = staged_base_data()
    groups = set(staged_classification(base)['FAO_group'])
    assert groups == set(pc.EFFECT_CATEGORY_BY_GROUP) | pc.NO_BENEFIT_GROUPS


def test_the_staged_matched_price_ratios_are_unique_after_the_country_recode():
    _, base, _ = staged_base_data()
    premiums = pd.read_csv(os.path.join(base, 'value_org_fb40d8c', 'org_premiums.csv'), encoding='utf-8-sig')
    recoded = premiums.assign(area_fao=premiums['country'].map(lambda c: pc.FIBL_COUNTRY_TO_FAO.get(c, c)))
    assert not recoded.dropna(subset=['area_fao']).duplicated(['area_fao', 'item', 'year']).any()


def test_every_valued_group_has_its_two_kinds_of_parameter():
    _, base, _ = staged_base_data()
    premiums = pd.read_csv(os.path.join(base, 'value_org_fb40d8c', 'org_premiums.csv'), encoding='utf-8-sig')
    params = pc.premium_parameters_by_group(premiums)
    for group in set(pc.EFFECT_CATEGORY_BY_GROUP) - {'Other oleaginous fruits'}:
        assert group in pc.YIELD_RATIO_KEY_BY_GROUP and group in params.index


def staged_table():
    """The first run's tables from the staged inputs, driven as the task drives them."""
    from global_invest import utilities
    p, base, pollination = staged_base_data()
    utilities.initialize_country_paths(p, simplified='30sec')
    params = staged_parameters()
    effect = {k: (float(params['pest_control_effect_lnrr_' + k]), float(params['pest_control_effect_lnrr_se_' + k]))
              for k in ['cereals', 'vegetables', 'others']}
    ratio = {k: (float(params['pest_control_yield_ratio_lnrr_' + k]), float(params['pest_control_yield_ratio_lnrr_se_' + k]))
             for k in pc.YIELD_RATIO_KEY_BY_GROUP.values()}
    value_org = os.path.join(base, 'value_org_fb40d8c')
    fao = pest_control_tasks.read_shared_fao_table(os.path.join(pollination, 'fao', 'fao_values_1993_2024.csv'),
                                                   os.path.join(pollination, 'fao', 'fao_production_1993_2024.csv'), 2019)
    items, harvested = pest_control_tasks.read_faostat_crop_items_and_area(os.path.join(base, 'faostat_qcl_2019.csv'), 2019)
    organic, info = pc.clean_organic_areas(
        pest_control_tasks.read_fibl_organic_areas(os.path.join(value_org, 'Area_org_FiBL.xlsx')),
        pest_control_tasks.read_recent_organic_areas(os.path.join(value_org, 'UK_org_2021_2022.csv')),
        items, harvested, 2019)
    names = utilities.collapse_countries_to_r250(p.df_countries)[['iso3_r250_label', 'iso3_r250_name']]
    name_to_iso3 = dict(zip(names['iso3_r250_name'], names['iso3_r250_label']))
    name_to_iso3.update(pest_control_tasks.SOURCE_NAME_TO_ISO3)
    name_to_iso3.update(dict(zip(fao['area_fao'], fao['iso3'])))
    groups = staged_classification(base).set_index('item', verify_integrity=True)['FAO_group'].to_dict()
    premiums = pd.read_csv(os.path.join(value_org, 'org_premiums.csv'), encoding='utf-8-sig')
    rows = pc.pest_control_by_crop_country(organic, fao, premiums, groups, name_to_iso3, ratio, effect, False, 2019)
    return rows, ratio, pc.premium_parameters_by_group(premiums), effect, info, params


def test_the_staged_inputs_give_the_first_run_figures_to_the_cent():
    """The staged inputs through the task's route, the yield ratios fitted on the exact Ponisio file,
    10,000 draws and the configured seed. These pin this code on the shared prices staged in base data,
    not a reproduction of the owner's current table, which is not yet held."""
    rows, ratio, premium, effect, info, params = staged_table()
    labels, totals = pc.monte_carlo_by_country(rows, ratio, premium, effect, int(params['pest_control_monte_carlo_draws']),
                                               int(params['pest_control_monte_carlo_seed']))
    world = totals.sum(axis=1)
    assert round(float(rows['pest_control_value'].sum()), 2) == 5_296_644_595.35
    assert round(float(pc.expected_value_by_country(rows, ratio, premium, effect).sum()), 2) == 5_181_246_896.82
    assert round(float(world.mean()), 2) == 5_195_473_666.75
    assert round(float(np.quantile(world, 0.025)), 2) == 1_389_762_095.71
    assert round(float(np.quantile(world, 0.975)), 2) == 8_402_623_749.17
    means = pd.Series(totals.mean(axis=0), index=labels)
    assert (means > 0).sum() == 124
    assert sorted(means[means == 0].index) == ['BTN', 'IRL', 'ISL', 'JPN', 'KHM', 'LAO', 'MLI', 'PRY', 'TJK', 'ZWE']
    assert (rows['benefit_status'] != 'unclassified').all() and len(rows) == 1461
    assert info['spikes'][['area_fao', 'item']].values.tolist() == [['Kyrgyzstan', 'Walnuts, in shell']]


def test_the_staged_inputs_correct_seven_records_across_all_years():
    # The additional documentation's count, on harvested areas from the shared FAOSTAT bulk file.
    _, base, pollination = staged_base_data()
    value_org = os.path.join(base, 'value_org_fb40d8c')
    bulk = pd.read_csv(os.path.join(pollination, 'fao', 'bulk', 'Production_Crops_Livestock_E_All_Data_(Normalized).csv'),
                       encoding='utf-8', usecols=['Area', 'Area Code', 'Item Code', 'Item', 'Element', 'Year', 'Value'], low_memory=False)
    bulk = bulk[(bulk['Item Code'] < 1700) & (bulk['Area Code'] < 5000) & (bulk['Area Code'] != 351)]
    harvested = bulk[bulk['Element'] == 'Area harvested'].dropna(subset=['Value'])[['Area', 'Item', 'Year', 'Value']]
    harvested.columns = ['area_fao', 'item', 'year', 'harvested_ha']
    fibl = pest_control_tasks.read_fibl_organic_areas(os.path.join(value_org, 'Area_org_FiBL.xlsx'))
    recent = pest_control_tasks.read_recent_organic_areas(os.path.join(value_org, 'UK_org_2021_2022.csv'))
    d = pd.concat([fibl, recent], ignore_index=True)
    d['area_fao'] = d['country'].map(lambda c: pc.FIBL_COUNTRY_TO_FAO.get(c, c))
    d['item'] = d['crop'].map(lambda c: pc.FIBL_CROP_TO_FAO.get(c, c))
    d = d.dropna(subset=['area_fao', 'item']).groupby(['area_fao', 'year', 'item'], as_index=False)['organic_ha'].sum()
    d = d[d['item'].isin(set(bulk['Item']))].assign(organic_ha=lambda f: f['organic_ha'].clip(lower=0.0))
    _, spikes = pc.correct_organic_area_spikes(d, harvested)
    assert len(spikes) == 7


def test_the_staged_inputs_qualify_the_totals_by_coverage():
    rows, _, _, _, info, _ = staged_table()
    c = pc.coverage_by_country(rows)
    assert c['coverage'].value_counts().to_dict() == {'complete_value': 80, 'partial_value': 44, 'complete_zero': 9,
                                                      'missing_inputs': 3, 'zero_with_missing': 1}
    assert c[c['coverage'] == 'zero_with_missing'].index.tolist() == ['PRY']
    assert sorted(c[c['coverage'] == 'missing_inputs'].index) == ['GIN', 'GMB', 'MUS']
    assert int(c['crops_missing'].sum()) == 118 and int(c['crops'].sum()) == 1461
    assert round(float(c['area_missing_ha'].sum()), 2) == round(float(rows.loc[rows['pest_control_value'].isna() & (rows['benefit_status'] != 'no_benefit'), 'organic_ha'].sum()), 2)
    # Every hectare FiBL reports in 2019 is valued, without a benefit, lacking an input, left out or removed as a spike.
    accounted = (c['area_valued_ha'].sum() + c['area_no_benefit_ha'].sum() + c['area_missing_ha'].sum() + info['left_out']['organic_ha'].sum()
                 + (info['spikes']['organic_ha'] - info['spikes']['replacement_ha']).sum())
    assert round(float(accounted), 2) == round(info['total_ha'], 2)


def one_way_rows(y):
    """Four clusters of three rows with equal sampling variances, where REML has the closed ANOVA form."""
    return np.array(y, dtype=float), np.full(12, 0.001), np.repeat(np.arange(4), 3), np.arange(12)


def test_the_reml_fit_equals_the_closed_form_of_a_balanced_design():
    y, v, cluster, observation = one_way_rows([0.10, 0.30, 0.20, 0.90, 1.10, 1.30, -0.50, -0.20, -0.30, 0.40, 0.70, 0.45])
    fit = pc.fit_nested_reml(y, v, [cluster, observation], ['study', 'observation'])
    means = y.reshape(4, 3).mean(axis=1)
    within = ((y.reshape(4, 3) - means[:, None]) ** 2).sum() / 8.0
    between = 3.0 * means.var(ddof=1)
    components = fit['levels'].set_index('level')['sigma2']
    assert fit['mean'] == pytest.approx(y.mean(), abs=1e-12)
    assert fit['se'] == pytest.approx(np.sqrt(between / 12.0), abs=1e-10)
    assert components['observation'] == pytest.approx(within - 0.001, abs=1e-10)
    assert components['study'] == pytest.approx((between - within) / 3.0, abs=1e-10)
    assert fit['ci_upper'] - fit['mean'] == pytest.approx(1.959963984540054 * fit['se'], abs=1e-12)
    assert fit['max_abs_gradient_free'] < pc.REML_GRADIENT_TOLERANCE and fit['n_starts_at_best'] == fit['n_starts']


def test_a_component_with_no_support_finishes_at_exactly_zero_with_a_positive_gradient():
    # The cluster means sit closer together than the rows within them allow.
    y, v, cluster, observation = one_way_rows([0.0, 0.5, 1.0, 0.1, 0.5, 0.9, 0.2, 0.5, 0.8, 1.0, 0.5, 0.0])
    fit = pc.fit_nested_reml(y, v, [cluster, observation], ['study', 'observation'])
    study = fit['levels'].set_index('level').loc['study']
    assert study['sigma2'] == 0.0 and study['status'] == 'boundary' and study['gradient'] > 0.0
    assert fit['mean'] == pytest.approx(y.mean(), abs=1e-12)
    assert fit['levels'].set_index('level').loc['observation', 'sigma2'] == pytest.approx(y.var(ddof=1) - 0.001, abs=1e-10)
    assert fit['se'] == pytest.approx(np.sqrt(y.var(ddof=1) / 12.0), abs=1e-10)


def test_rows_tighter_than_their_sampling_variances_leave_only_those_variances():
    y, v = np.array([0.10, 0.11, 0.09, 0.10]), np.array([0.5, 0.25, 0.5, 0.25])
    fit = pc.fit_nested_reml(y, v, [np.arange(4)], ['observation'])
    assert fit['levels']['sigma2'].iloc[0] == 0.0
    assert fit['mean'] == pytest.approx((y / v).sum() / (1.0 / v).sum(), abs=1e-14)
    assert fit['se'] == pytest.approx(np.sqrt(1.0 / (1.0 / v).sum()), abs=1e-14)


def test_a_level_with_one_cluster_is_fixed_at_zero_because_the_likelihood_cannot_see_it():
    y, v = np.array([-0.4, -0.1, -0.3, -0.2]), np.array([0.02, 0.05, 0.03, 0.04])
    one, observation = np.zeros(4), np.arange(4)
    blocks = pc._reml_blocks(one, [one, observation])
    without = pc._reml_terms(np.array([0.0, 0.01]), y, v, blocks)
    with_it = pc._reml_terms(np.array([1.0, 0.01]), y, v, blocks)
    assert with_it['objective'] == pytest.approx(without['objective'], abs=1e-12)
    assert with_it['mean'] == pytest.approx(without['mean'], abs=1e-12)
    assert with_it['se'] > 5.0 * without['se']
    fit = pc.fit_nested_reml(y, v, [one, one, observation], ['study', 'year', 'observation'])
    levels = fit['levels'].set_index('level')
    assert levels.loc[['study', 'year'], 'status'].tolist() == ['fixed_single_level'] * 2
    assert levels.loc[['study', 'year'], 'sigma2'].tolist() == [0.0, 0.0]


def test_levels_with_the_same_clusters_share_one_component():
    y, v, cluster, observation = one_way_rows([0.10, 0.30, 0.20, 0.90, 1.10, 1.30, -0.50, -0.20, -0.30, 0.40, 0.70, 0.45])
    two = pc.fit_nested_reml(y, v, [cluster, observation], ['study', 'observation'])
    three = pc.fit_nested_reml(y, v, [cluster, cluster * 10, observation], ['study', 'year', 'observation'])
    year = three['levels'].set_index('level').loc['year']
    assert year['status'] == 'shared_with_study' and np.isnan(year['sigma2'])
    assert three['mean'] == two['mean'] and three['se'] == two['se']


def test_the_reml_derivatives_equal_finite_differences():
    y = np.array([0.3, -0.2, 0.5, 0.1, -0.6, 0.2, 0.7, -0.1, 0.4])
    v = np.array([0.02, 0.05, 0.03, 0.04, 0.01, 0.06, 0.02, 0.03, 0.05])
    study, year, observation = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2]), np.array([0, 0, 1, 1, 2, 2, 3, 4, 5]), np.arange(9)
    blocks = pc._reml_blocks(study, [study, year, observation])
    theta = np.array([0.03, 0.02, 0.05])
    terms = pc._reml_terms(theta, y, v, blocks, second_order=True)
    h = 1e-5
    for j in range(3):
        up, down = theta.copy(), theta.copy()
        up[j] += h
        down[j] -= h
        at_up, at_down = pc._reml_terms(up, y, v, blocks), pc._reml_terms(down, y, v, blocks)
        assert terms['gradient'][j] == pytest.approx((at_up['objective'] - at_down['objective']) / (2 * h), rel=1e-6)
        assert terms['hessian'][:, j] == pytest.approx((at_up['gradient'] - at_down['gradient']) / (2 * h), rel=1e-6)


def test_the_effect_sizes_map_crops_fill_groups_and_keep_year_strings():
    observations = pd.DataFrame({'study': ['1', '1', '2', '3'], 'year': ['2001; 2002 ', '2001; 2002', '1990', '1'],
                                 'crop': ['barely', 'raisins', 'alfalfa', 'chard'], 'lnrr': [-0.1, -0.2, -0.3, -0.4], 'var_lnrr': [0.1, 0.2, 0.3, 0.4]})
    d = pc.yield_ratio_effect_sizes(observations, {'Barley': 'Cereals', 'Grapes': FRUIT})
    assert d['item'].tolist() == ['Barley', 'Grapes', 'Alfalfa', 'Chard']
    assert d['group'].tolist() == ['Cereals', FRUIT, pc.YIELD_RATIO_ITEM_GROUP_FILL['Alfalfa'], 'Vegetables']
    assert d['year'].tolist() == observations['year'].tolist() and d['observation'].tolist() == [1, 2, 3, 4]
    with pytest.raises(ValueError, match='crops with no FAO item'):
        pc.yield_ratio_effect_sizes(observations.assign(crop='quinoa'), {})
    with pytest.raises(ValueError, match='items with no FAO group'):
        pc.yield_ratio_effect_sizes(observations.head(1), {})


def test_the_ponisio_reader_handles_carriage_returns_accents_and_keeps_year_spaces(tmp_path):
    path = tmp_path / 'ponisio.csv'
    path.write_bytes(b'Study#,Author,Crop species,year-org,lnRR,var lnRR\r15,M\xfcller,maize,2001; 2002 ,-0.311587593,0.073293308\r16,Lee,oat,NA,0.5,0.1')
    d = pest_control_tasks.read_ponisio_observations(str(path))
    assert d.columns.tolist() == ['study', 'year', 'crop', 'lnrr', 'var_lnrr']
    assert d['year'].tolist() == ['2001; 2002 ', 'NA'] and d['lnrr'].tolist() == [-0.311587593, 0.5]


def test_the_exact_ponisio_file_refits_in_python_at_the_configured_coefficients():
    """The ten models from the file's rows. The tolerance is a drift guard, a few times the largest differences
    measured against the R fit (7.7e-9 on a mean, 2.5e-8 on a standard error). The coefficients stay configured."""
    _, base, _ = staged_base_data()
    observations = pest_control_tasks.read_ponisio_observations(os.path.join(base, 'value_org_fb40d8c', 'DataForSubmitting.csv'))
    groups = staged_classification(base).set_index('item', verify_integrity=True)['FAO_group'].to_dict()
    fit, diagnostics = pc.yield_ratios_by_group(pc.yield_ratio_effect_sizes(observations, groups))
    fit = fit.set_index('FAO_group')
    assert len(fit) == 10 and fit['n_effect_sizes'].drop('All crops').sum() == fit.loc['All crops', 'n_effect_sizes'] == 1071
    assert fit.loc['All crops', 'n_studies'] == 115
    assert round(1.0 - fit.loc['All crops', 'Y_difference_org'], 4) == 0.2087
    params = staged_parameters()
    for group, key in pc.YIELD_RATIO_KEY_BY_GROUP.items():
        assert fit.loc[group, 'lnRR_mean'] == pytest.approx(float(params['pest_control_yield_ratio_lnrr_' + key]), abs=1e-7)
        assert fit.loc[group, 'lnRR_se'] == pytest.approx(float(params['pest_control_yield_ratio_lnrr_se_' + key]), abs=1e-7)
    boundary = diagnostics[diagnostics['status'] == 'boundary']
    assert set(zip(boundary['FAO_group'].str[:6], boundary['level'])) == {('Cereal', 'year'), ('Fruit ', 'year'), ('Pulses', 'study'), ('Stimul', 'study'), ('Sugar ', 'observation')}
    assert (boundary['sigma2'] == 0.0).all() and (boundary['gradient'] > 0.0).all()
    sugar = diagnostics[diagnostics['FAO_group'] == 'Sugar crops'].set_index('level')
    assert sugar.loc[['study', 'year', 'item'], 'status'].tolist() == ['fixed_single_level'] * 3
    assert (diagnostics['max_abs_gradient_free'] < pc.REML_GRADIENT_TOLERANCE).all()
    assert (diagnostics['n_starts_at_best'] == diagnostics['n_starts']).all()


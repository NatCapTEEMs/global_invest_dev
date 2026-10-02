"""Unit tests for nature_view_amenity's valuation.

Every function is pinned on a hand-built frame, so the premium conventions, the outlier rule
and the missing-input rule are stated as executable facts, and the staged workbooks anchor the
joins.
"""
import numpy as np
import pandas as pd

from global_invest.nature_view_amenity import nature_view_amenity_functions as nv


def test_the_group_premium_is_the_relative_price_difference():
    df = pd.DataFrame({'scenic_price': [120.0], 'non_scenic_price': [100.0],
                       'scenic_availability': [3.0]})
    out = nv.summarize_groups(df)
    assert np.isclose(out['group_premium'].iloc[0], 0.2)
    assert np.isclose(out['log_price_ratio'].iloc[0], np.log(1.2))
    assert np.isclose(out['premium_component'].iloc[0], 3.0 * 20.0)


def test_the_outlier_rule_is_three_inclusive_iqrs_two_sided():
    # nine values spread 1..9: Q1 3, Q3 7, IQR 4 -> bounds [-9, 19]; 100 and -100 fall out.
    values = list(range(1, 10)) + [100.0, -100.0]
    mask = nv.flag_3iqr_outliers(values)
    assert mask.tolist() == [False] * 9 + [True, True]


def test_the_unit_price_difference_weights_by_scenic_availability():
    groups = pd.DataFrame({
        'iso3': ['AAA', 'AAA', 'AAA'],
        'scenic_availability': [2.0, 4.0, 1.0],
        'premium_component': [2.0 * 30.0, 4.0 * 15.0, 1.0 * 99.0],
        'group_premium': [0.3, 0.15, -0.5],
        'is_outlier': [False, False, False]})
    out = nv.country_unit_price_difference(groups)
    # the negative-premium group is dropped; (60 + 60) / (2 + 4)
    assert np.isclose(out['unit_price_difference'].iloc[0], 20.0)
    assert int(out['n_groups'].iloc[0]) == 2


def test_the_valuation_is_rooms_times_occupancy_times_share_times_premium():
    df = pd.DataFrame({'hotel_rooms': [1000.0, 500.0],
                       'scenic_room_share': [0.4, np.nan],
                       'unit_price_difference': [25.0, 10.0]})
    out = nv.nature_view_value(df, mean_occupancy=0.5)
    assert np.isclose(out['nature_view_amenity_gep'].iloc[0], 1000 * 0.5 * 365 * 0.4 * 25.0)
    assert pd.isna(out['nature_view_amenity_gep'].iloc[1])
    assert 'no room sample' in out['note'].iloc[1]


def test_the_staged_workbooks_carry_the_sample_and_the_anchor():
    import os, tempfile, hazelbean as hb
    from global_invest import utilities
    base = utilities.service_data_dir(
        hb.ProjectFlow(project_dir=os.path.join(tempfile.mkdtemp(), 'anchors')),
        'nature_view_amenity')
    mapping = pd.read_excel(os.path.join(base, 'Country_ISO_Mapping.xlsx'))
    reference = pd.read_excel(os.path.join(base, 'reference_results.xlsx'))
    un = pd.read_excel(os.path.join(base, 'UN_Tourism_2019_Hotel_Rooms_and_Occupancy.xlsx'),
                       sheet_name='Country Data')
    # every reference country resolves through the mapping, and the UN table matches its list
    name_to_iso3 = dict(zip(mapping['Country or Area'], mapping['ISO Alpha-3']))
    assert reference['Country or Area'].isin(name_to_iso3).all()
    assert set(un['ISO Alpha-3']) == set(mapping['ISO Alpha-3'])

# -*- coding: utf-8 -*-
"""Nature-view-amenity science: urban nature valued as the hotel room-price premium it earns.

Nothing here opens a file. The task layer reads the hotel workbooks, hands the frames in and
writes back what it gets, so every step can be pinned on a hand-built input in the test suite.

The valuation: within one hotel on one date, a room with a natural scenic view prices above a
comparable room without one, and that premium is the view's market price. Per country, the
annual value is hotel rooms times mean occupancy times 365 times the scenic-room share times
the unit price difference, with the premium identified from within-hotel comparable groups so
brand, location and star rating difference out.
"""
import numpy as np
import pandas as pd

# A group premium this close to zero is zero; the source pipeline's own tolerance.
ZERO_TOLERANCE = 1e-12

DAYS_PER_YEAR = 365


def summarize_groups(df):
    """Per-group premium statistics from the per-group price and availability aggregates.

    Args:
        df (pd.DataFrame): one row per comparable group, carrying `scenic_price` (mean price
            of the group's natural-view rooms), `non_scenic_price` (mean price of its
            comparable rooms without one) and `scenic_availability` (displayed availability
            of the natural-view rooms).

    Returns:
        pd.DataFrame: the input with `group_premium` (relative, zeroed within tolerance),
        `log_price_ratio` and `premium_component` (availability times the absolute price
        difference) added.
    """
    out = df.copy()
    out['group_premium'] = (out['scenic_price'] - out['non_scenic_price']) / out['non_scenic_price']
    out.loc[out['group_premium'].abs() <= ZERO_TOLERANCE, 'group_premium'] = 0.0
    out['log_price_ratio'] = np.log(out['scenic_price'] / out['non_scenic_price'])
    out['premium_component'] = out['scenic_availability'] * (out['scenic_price'] - out['non_scenic_price'])
    return out


def flag_3iqr_outliers(values):
    """Two-sided 3 IQR outlier mask with Excel-inclusive quartiles, the source's rule.

    Args:
        values: the log price ratios, one per group.

    Returns:
        np.ndarray of bool: True where the value lies outside [Q1 - 3 IQR, Q3 + 3 IQR].
    """
    arr = np.asarray(values, dtype=float)
    q1 = np.quantile(arr, 0.25, method='linear')
    q3 = np.quantile(arr, 0.75, method='linear')
    spread = q3 - q1
    return (arr < q1 - 3.0 * spread) | (arr > q3 + 3.0 * spread)


def country_unit_price_difference(groups_df):
    """The availability-weighted unit price difference per country, over the kept groups.

    Args:
        groups_df (pd.DataFrame): summarize_groups' output plus `iso3` and `is_outlier`;
            outliers and negative-premium groups are dropped here, the source's filter.

    Returns:
        pd.DataFrame: per iso3, `unit_price_difference` (USD per room-night) and the number
        of groups behind it.
    """
    kept = groups_df[~groups_df['is_outlier']
                     & (groups_df['group_premium'] >= -ZERO_TOLERANCE)]
    agg = kept.groupby('iso3').agg(
        component=('premium_component', 'sum'),
        scenic_availability=('scenic_availability', 'sum'),
        n_groups=('premium_component', 'size'))
    agg['unit_price_difference'] = agg['component'] / agg['scenic_availability']
    return agg.reset_index()[['iso3', 'unit_price_difference', 'n_groups']]


def nature_view_value(df, mean_occupancy):
    """The annual premium value per country: the valuation identity on the joined frame.

    Args:
        df (pd.DataFrame): one row per country, carrying `hotel_rooms`, `scenic_room_share`
            and `unit_price_difference`.
        mean_occupancy (float): the uniform mean room occupancy (fraction).

    Returns:
        pd.DataFrame: the input with `nature_view_amenity_gep` in USD added. A country
        missing any input carries NA there, never a silent zero, and the note column names
        which.
    """
    out = df.copy()
    out['nature_view_amenity_gep'] = (out['hotel_rooms'] * mean_occupancy * DAYS_PER_YEAR
                                      * out['scenic_room_share'] * out['unit_price_difference'])
    absences = {'hotel_rooms': 'no room inventory', 'scenic_room_share': 'no room sample',
                'unit_price_difference': 'no premium estimate'}
    out['note'] = out.apply(
        lambda row: '; '.join(absences[c] for c in absences if pd.isna(row[c])), axis=1)
    return out

# -*- coding: utf-8 -*-
"""Moisture-recycling science: the WAM2layers source-side diagnostic for rainfall regulation.

Nothing here opens a file. The task layer reads the WAM2layers sink-fraction matrix and the
water-supply country table, hands the frames in and writes back what it gets, so every step can
be pinned on a hand-built input in the test suite.

The service's headline is not computed here: it is the VMR crop value at the receiving country,
which the task layer reads from the water_supply run. This module supplies the diagnostic
beside it, a WAM-weighted source-side view that credits the land-sourced share of each
destination's value to the countries whose evaporation became its rain. The matrix columns sum
below one, so the diagnostic covers only that land-sourced part, not a redistribution of the
full headline. Its source-side and sink-side totals agree with each other by construction, and
neither may be added to the headline, agricultural water or crop provision.
"""
import numpy as np
import pandas as pd

# The matrix labels countries as m_<iso3>; the account's key is the bare iso3_r250_label.
MATRIX_LABEL_PREFIX = 'm_'


def moisture_matrix(df_raw):
    """The country-to-country sink-fraction matrix on bare iso3 labels.

    Args:
        df_raw (pd.DataFrame): the matrix as shipped -- first column the row labels, every
            other column one country, all labels carrying MATRIX_LABEL_PREFIX. Cell [i, j] is
            the fraction of column-country j's precipitation that evaporated from row-country
            i's land, so a column sums to j's terrestrial-origin share (below 1; the rest is
            ocean).

    Returns:
        pd.DataFrame: the same matrix indexed and columned by bare iso3 labels.

    Raises:
        NameError: if any cell falls outside [0, 1], which would mean the frame passed in is a
            volume table rather than the fraction matrix.
    """
    out = df_raw.set_index(df_raw.columns[0])
    out.index = [str(label).removeprefix(MATRIX_LABEL_PREFIX) for label in out.index]
    out.columns = [str(label).removeprefix(MATRIX_LABEL_PREFIX) for label in out.columns]
    out = out.astype('float64')
    if float(out.min().min()) < 0.0 or float(out.max().max()) > 1.0:
        raise NameError('moisture_matrix expected fractions in [0, 1]; got %r to %r. The frame '
                        'passed in looks like a volume table, not the sink-fraction matrix.'
                        % (float(out.min().min()), float(out.max().max())))
    return out


def untracked_destinations(matrix):
    """The destinations the moisture tracking could not resolve: their columns are all NaN.

    Args:
        matrix (pd.DataFrame): the sink-fraction matrix from moisture_matrix.

    Returns:
        list: the labels, sorted. These are islands and city-states below the tracking grid;
        the source-side diagnostic cannot place their value, and the caller reports it.
    """
    return sorted(matrix.columns[matrix.isna().all(axis=0)])


def terrestrial_precipitation_share(matrix):
    """Each country's terrestrial-origin share of precipitation: the matrix column sums.

    Args:
        matrix (pd.DataFrame): the sink-fraction matrix from moisture_matrix.

    Returns:
        pd.Series: destination label -> fraction of its precipitation evaporated from any
        tracked country's land. An untracked destination reads 0, matching the attribution
        it receives.
    """
    return matrix.fillna(0.0).sum(axis=0)


def reattributed_water_value(matrix, water_value, ecosystem_share):
    """The WAM-weighted source-side diagnostic: value credited to the countries whose land sent the rain.

    It is not a redistribution of the full headline. The matrix columns sum below one, so only
    each destination's land-sourced share is credited upstream.

    Args:
        matrix (pd.DataFrame): the sink-fraction matrix from moisture_matrix.
        water_value (pd.Series): destination label -> the receiving country's value, of which
            the land-sourced share is credited upstream. Destinations absent from the matrix
            are dropped; the caller reports them.
        ecosystem_share (float): the share of terrestrial evaporation credited to ecosystems.
            1 counts all of it; the transpiration-share decision lowers it.

    Returns:
        pd.DataFrame: one row per country appearing in the matrix, with
            moisture_recycling_gep -- the value credited to the country as a SOURCE (the task
                publishes it as moisture_recycling_source_attributed, never as the headline),
            moisture_recycling_gep_own_part -- the part credited for rain on its own crops,
            moisture_recycling_gep_export_part -- the part credited for rain on other
                countries' crops, and
            moisture_recycling_sink_side_value -- the destination's land-sourced value (the
                destination-side view of the same diagnostic).
        The source column and the sink-side column sum to the same total: one double sum
        computed two ways. That total is the land-sourced part of the input value, not all
        of it.
    """
    filled = matrix.fillna(0.0)
    value = water_value.reindex(filled.columns).fillna(0.0) * float(ecosystem_share)
    attributed = filled.to_numpy() @ value.to_numpy()
    if not (filled.index.is_unique and filled.columns.is_unique):
        raise ValueError('the sink-fraction matrix carries duplicated country labels; '
                         'the own/export split needs unique labels.')
    # The own-country fraction is the LABELLED diagonal -- cell [label, label] -- so a matrix
    # whose columns arrive in a different order still splits correctly; a row label with no
    # matching column has no own-column in the data and carries 0.0.
    own_fracs = pd.Series(0.0, index=filled.index)
    common = filled.index.intersection(filled.columns)
    own_fracs.loc[common] = [float(filled.at[label, label]) for label in common]
    own = (own_fracs * value.reindex(filled.index).fillna(0.0)).to_numpy()
    out = pd.DataFrame({
        'iso3_r250_label': list(matrix.index),
        'moisture_recycling_gep': attributed,
        'moisture_recycling_gep_own_part': own,
    })
    out['moisture_recycling_gep_export_part'] = (
        out['moisture_recycling_gep'] - out['moisture_recycling_gep_own_part'])
    sink_side = terrestrial_precipitation_share(matrix) * value
    out = out.merge(sink_side.rename('moisture_recycling_sink_side_value'),
                    left_on='iso3_r250_label', right_index=True, how='left')
    return out

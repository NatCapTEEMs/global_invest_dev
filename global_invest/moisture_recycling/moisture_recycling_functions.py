# -*- coding: utf-8 -*-
"""Moisture-recycling science: water value re-attributed to the countries whose land sent the rain.

Nothing here opens a file. The task layer reads the WAM2layers sink-fraction matrix and the
water-supply country table, hands the frames in and writes back what it gets, so every step can
be pinned on a hand-built input in the test suite.

The service is a RE-ATTRIBUTION, not new value: the water-supply account values water where it is
withdrawn, and this module moves the terrestrially-sourced share of that value upstream to the
countries whose evaporation became the rain. The source-side total therefore equals the sink-side
total by construction, and neither may be added to the water-supply row.
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
        their water value cannot be attributed to any source and the caller reports it.
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
    """Water value moved upstream to the countries whose land sent the rain.

    Args:
        matrix (pd.DataFrame): the sink-fraction matrix from moisture_matrix.
        water_value (pd.Series): destination label -> the water GEP being re-attributed.
            Destinations absent from the matrix are dropped; the caller reports them.
        ecosystem_share (float): the share of terrestrial evaporation credited to ecosystems.
            1 counts all of it; the transpiration-share decision lowers it.

    Returns:
        pd.DataFrame: one row per country appearing in the matrix, with
            moisture_recycling_gep -- the value attributed to the country as a SOURCE,
            moisture_recycling_gep_own_part -- the part serving its own water use,
            moisture_recycling_gep_export_part -- the part serving other countries, and
            moisture_recycling_sink_side_value -- the country's own water value that is
            terrestrially sourced (the destination-side view of the same money).
        The moisture_recycling_gep column and the sink-side column sum to the same total,
        which is the conservation the construction guarantees.
    """
    filled = matrix.fillna(0.0)
    value = water_value.reindex(filled.columns).fillna(0.0) * float(ecosystem_share)
    attributed = filled.to_numpy() @ value.to_numpy()
    own = np.diag(filled.to_numpy()) * value.reindex(filled.index).fillna(0.0).to_numpy() \
        if list(filled.index) == list(filled.columns) else np.zeros(len(filled.index))
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

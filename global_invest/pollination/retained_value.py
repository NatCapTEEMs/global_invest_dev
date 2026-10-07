"""Accounting for habitat-driven output change on retained cropland.

Inputs are aligned arrays of cell totals, NOT monetary densities. Convert densities
to totals with cell area before calling. Crop value is assumed uniform per baseline
cropland hectare within each valuation cell. This module does not alter production
exports until the raster-to-table integration has been independently checked.
"""
import numpy as np


def retained_value_change(dependent_value, retained_fraction, suff_base, suff_future):
    """Return dollar change; missing data remain missing rather than becoming zero."""
    value, fraction, base, future = np.broadcast_arrays(*[
        np.asarray(x, dtype=float) for x in
        (dependent_value, retained_fraction, suff_base, suff_future)])
    if np.any(np.isinf(value)) or np.any(value < 0):
        raise ValueError('Dependent crop value must be nonnegative and finite or missing')
    for name, array in [('retained fraction', fraction), ('base sufficiency', base),
                        ('future sufficiency', future)]:
        if np.any(np.isinf(array)) or np.any((array < 0) | (array > 1)):
            raise ValueError(name + ' must be in [0, 1] or missing')
    active = (value > 0) & (fraction > 0)
    if np.any(active & (np.isfinite(base) != np.isfinite(future))):
        raise ValueError('Base and future sufficiency coverage differs on retained value')
    result = value * fraction * (future - base)
    # No retained value contributes zero even if no sufficiency can be computed.
    return np.where((value == 0) | (fraction == 0), 0., result)


def output_share_pct(delta_value, baseline_crop_value):
    """Aggregate one sector-region, rejecting missing contributions/denominators."""
    delta = np.asarray(delta_value, dtype=float)
    crop = np.asarray(baseline_crop_value, dtype=float)
    if not np.all(np.isfinite(delta)) or not np.all(np.isfinite(crop)):
        raise ValueError('Missing data must be reconciled before aggregation')
    if np.any(crop < 0) or crop.sum() <= 0:
        raise ValueError('Baseline crop output must have positive total value')
    return float(100 * delta.sum() / crop.sum())


def annual_retained_rows(base_by_year, future_by_year, unpaired_base, scenario,
                         base_year, sectors):
    """Interpolate paired dollar levels before export; base-year change is zero.

    Series indices are (ENDW, REG). This preserves zero baseline provision cases:
    the economic numerator never divides by paired provision.
    """
    import pandas as pd
    years = sorted(base_by_year)
    if not years or years != sorted(future_by_year) or years[0] <= base_year:
        raise ValueError('Invalid retained-value anchor years')
    zones = unpaired_base.index
    for y in years:
        for series in (base_by_year[y], future_by_year[y]):
            if not series.index.equals(zones) or not np.isfinite(series.to_numpy()).all():
                raise ValueError('Missing or inconsistent retained-value zones')
            if (series < 0).any():
                raise ValueError('Negative retained provision')
    if not np.isfinite(unpaired_base.to_numpy()).all() or (unpaired_base < 0).any():
        raise ValueError('Invalid unpaired baseline')
    rows = []
    annual_years = np.arange(base_year, years[-1] + 1)
    for zone in zones:
        initial = float(unpaired_base.loc[zone])
        baselines = np.interp(annual_years, [base_year] + years,
                              [initial] + [float(base_by_year[y].loc[zone]) for y in years])
        futures = np.interp(annual_years, [base_year] + years,
                            [initial] + [float(future_by_year[y].loc[zone]) for y in years])
        for y, base, future in zip(annual_years, baselines, futures):
            delta = future - base
            for sector in sectors:
                rows.append(dict(ENDW=zone[0], REG=zone[1], ACTS=sector, scenario=scenario,
                    year=int(y), pollination_accounting='retained_value_v1',
                    delta_pollination_usd=delta, retained_pollination_future_usd=future,
                    retained_pollination_base_usd=base, value_usd_base=initial,
                    shock_pct=100*delta/initial if initial else 0.,
                    shock_pct_v3=100*delta/initial if initial else 0.))
    return pd.DataFrame(rows)

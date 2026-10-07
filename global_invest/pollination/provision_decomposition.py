"""Decompose the change in pollination provision into retained, lost and new cropland.

A DIAGNOSTIC. The fed measure is unchanged: retained_value_v1 continues to report the habitat effect
on cropland that is present in both years, and nothing here alters it. This exists to answer a
different question -- where does the total change in provision come from -- and the identity it
reconciles is

    total change = retained-land change
                 + provision on cropland that is NEW in the future year
                 - baseline provision on cropland that is LOST by the future year

Each term is a dollar flow on the valuation grid, so they sum. The first term is the fed measure.

WHAT VALUING NEW CROPLAND ASSUMES, stated rather than buried. The crop-value raster is a base-year
product: it values cropland where cropland was. Land that becomes cropland later therefore has no
crop value of its own anywhere in the inputs, and there is no way to derive one without an
assumption. The assumption offered here, and its name says so, is `local_base_rate`: new cropland in
a valuation cell earns the mean value per hectare of the cropland that cell already had. Where a
cell had NO baseline cropland, that rate does not exist, and the hectares are reported as
`unvalued_new_ha` rather than valued at zero -- zero is a claim that the land produces nothing, which
is a stronger statement than "we cannot say".

Coverage is kept consistent with the fed measure: a cell contributes only where the sufficiency
surfaces are finite in BOTH years, because a component computed on one year's coverage and differenced
against the other's is not a change.
"""

import numpy as np


def component_hectares(base_class, future_class, hectares, crop_class=2):
    """Retained, lost and new cropland hectares for one block of the fine land grid.

    Args:
        base_class, future_class: integer land-class arrays on one grid.
        hectares: hectares per fine cell, same shape.
        crop_class: the cropland class id (2 in seals7).

    Returns:
        dict of retained_ha, lost_ha, new_ha, base_cropland_ha, future_cropland_ha.

    Raises:
        ValueError: if the two dates do not share a shape.
    """
    base_class, future_class, hectares = (np.asarray(a) for a in (base_class, future_class, hectares))
    if base_class.shape != future_class.shape or base_class.shape != hectares.shape:
        raise ValueError('base, future and hectares must share one grid; got %s, %s, %s'
                         % (base_class.shape, future_class.shape, hectares.shape))
    was = base_class == crop_class
    now = future_class == crop_class
    retained = was & now
    return dict(retained_ha=float(np.sum(np.where(retained, hectares, 0.0))),
                lost_ha=float(np.sum(np.where(was & ~now, hectares, 0.0))),
                new_ha=float(np.sum(np.where(~was & now, hectares, 0.0))),
                base_cropland_ha=float(np.sum(np.where(was, hectares, 0.0))),
                future_cropland_ha=float(np.sum(np.where(now, hectares, 0.0))))


def decompose(dependent_value, retained_ha, lost_ha, new_ha, base_cropland_ha,
              suff_base, suff_future, new_cropland_valuation='local_base_rate'):
    """The three components and their reconciliation, per valuation cell.

    Args:
        dependent_value: base-year pollination-dependent crop value in the cell (whole cell).
        retained_ha, lost_ha, new_ha: hectares from component_hectares, same cell.
        base_cropland_ha: baseline cropland hectares in the cell; the denominator the value belongs to.
        suff_base, suff_future: pollination sufficiency in the two years, in [0, 1].
        new_cropland_valuation: 'local_base_rate' values new hectares at the cell's own baseline
            value per hectare; 'unvalued' declines to value them at all. Never zero.

    Returns:
        dict with retained_change, new_provision, lost_provision, total_change, unvalued_new_ha,
        and the identity residual, which is zero by construction and carried so a caller can assert it.

    Raises:
        ValueError: on impossible inputs, or an unknown valuation assumption.
    """
    values = [np.asarray(x, dtype=float) for x in
              (dependent_value, retained_ha, lost_ha, new_ha, base_cropland_ha, suff_base, suff_future)]
    value, retained, lost, new, base_ha, base_s, future_s = np.broadcast_arrays(*values)
    if np.any(value < 0) or np.any(~np.isfinite(value) & (value == value)):
        raise ValueError('Dependent crop value must be nonnegative')
    for name, array in (('base sufficiency', base_s), ('future sufficiency', future_s)):
        finite = np.isfinite(array)
        if np.any(finite & ((array < 0) | (array > 1))):
            raise ValueError(name + ' must lie in [0, 1] where present')
    for name, array in (('retained', retained), ('lost', lost), ('new', new)):
        if np.any(array < 0):
            raise ValueError(name + ' hectares must be nonnegative')
    if np.any(retained + lost > base_ha + 1e-6):
        raise ValueError('retained + lost exceeds baseline cropland: the components do not partition it')

    # One coverage rule for every component. A term computed where only one year has sufficiency
    # would be differenced against nothing, which is how a coverage gap becomes a fake signal.
    covered = np.isfinite(base_s) & np.isfinite(future_s)
    base_s = np.where(covered, base_s, 0.0)
    future_s = np.where(covered, future_s, 0.0)

    # Value per hectare of the cropland the cell actually had. Undefined where it had none.
    has_base = base_ha > 0
    rate = np.divide(value, base_ha, out=np.zeros_like(value), where=has_base)

    retained_change = np.where(covered, rate * retained * (future_s - base_s), 0.0)
    lost_provision = np.where(covered, rate * lost * base_s, 0.0)

    if new_cropland_valuation == 'local_base_rate':
        valued_new = covered & has_base
        new_provision = np.where(valued_new, rate * new * future_s, 0.0)
        unvalued_new_ha = np.where(valued_new, 0.0, new)
    elif new_cropland_valuation == 'unvalued':
        new_provision = np.zeros_like(value)
        unvalued_new_ha = new.copy()
    else:
        raise ValueError('unknown new-cropland valuation %r; use local_base_rate or unvalued'
                         % new_cropland_valuation)

    total = retained_change + new_provision - lost_provision
    # Area excluded for missing sufficiency, reported beside the components. Excluded area is not
    # absent land: it is land the decomposition could not speak about, and collapsing that into the
    # valued total would present a partial account as a complete one.
    excluded_ha = np.where(covered, 0.0, retained + lost + new)
    return dict(retained_change=retained_change, new_provision=new_provision,
                lost_provision=lost_provision, total_change=total,
                unvalued_new_ha=unvalued_new_ha, excluded_ha=excluded_ha,
                residual=total - (retained_change + new_provision - lost_provision),
                assumption=new_cropland_valuation)


def reconcile(components, tolerance=1e-9):
    """Assert the identity and summarise. Returns totals; raises if the identity does not hold."""
    residual = float(np.nansum(np.abs(components['residual'])))
    if residual > tolerance:
        raise ValueError('decomposition does not reconcile: residual %g exceeds %g' % (residual, tolerance))
    unvalued = float(np.nansum(components['unvalued_new_ha']))
    excluded = float(np.nansum(components.get('excluded_ha', 0.0)))
    # The quantity is only the TOTAL provision change when every hectare is valued and covered.
    # Where some is not, saying "total" would overstate what was measured, so the name changes with
    # the evidence rather than the caveat being left to a footnote nobody reads.
    complete = (unvalued == 0.0) and (excluded == 0.0)
    label = ('total provision change' if complete else 'provision change on valued support')
    return dict(retained_change=float(np.nansum(components['retained_change'])),
                new_provision=float(np.nansum(components['new_provision'])),
                lost_provision=float(np.nansum(components['lost_provision'])),
                total_change=float(np.nansum(components['total_change'])),
                quantity=label,
                unvalued_new_ha=unvalued,
                excluded_for_missing_sufficiency_ha=excluded,
                assumption=components['assumption'],
                caveat='; '.join(filter(None, [
                    ('%.1f ha of new cropland carry no defensible crop value and are reported as '
                     'area, not valued at zero' % unvalued) if unvalued else '',
                    ('%.1f ha excluded for missing sufficiency in one or both years' % excluded)
                    if excluded else ''])))

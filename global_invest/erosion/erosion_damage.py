"""Panagos-style area damage and physical erosion-prevention stress.

Pure arithmetic for the NGFS damage specification. Missing data remain missing;
the caller must aggregate severe and total cropland on the same valid support.
"""
import numpy as np


def productivity_level(severe_crop_ha, valid_crop_ha, alpha=0.08):
    """Negative productivity level in percent; zero cropland is undefined."""
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be a fraction in [0, 1]')
    severe, total = np.broadcast_arrays(np.asarray(severe_crop_ha, dtype=float),
                                      np.asarray(valid_crop_ha, dtype=float))
    if np.any(np.isinf(severe) | np.isinf(total)):
        raise ValueError('Areas must be finite or missing')
    valid = np.isfinite(severe) & np.isfinite(total)
    if np.any(valid & ((severe < 0) | (total < 0) | (severe > total))):
        raise ValueError('Require 0 <= severe cropland <= valid total cropland')
    result = np.full(total.shape, np.nan)
    np.divide(-100 * alpha * severe, total, out=result, where=valid & (total > 0))
    return result


def summarize_damage_areas(usle, potential, crop_fraction, hectares, zones,
                           common_support=None, threshold=11., provision_loss=.2):
    """Zonal areas and damage on aligned SDR rasters in t/ha/year.

    Zone 0 is outside the analysis domain. Missing soil data are reported as
    excluded cropland hectares, never recoded as zero erosion. Thresholding occurs
    before zonal averaging; a cell at exactly 11 is not classified as severe.
    """
    import pandas as pd
    loss, bare, crop, ha, ids = np.broadcast_arrays(*[np.asarray(x) for x in
        (usle, potential, crop_fraction, hectares, zones)])
    if np.any(np.isfinite(crop) & ((crop<0)|(crop>1))):
        raise ValueError('Cropland fraction outside [0,1]')
    if np.any(np.isfinite(ha)&(ha<=0)):
        raise ValueError('Cell hectares must be positive')
    if np.any(~np.isfinite(ids)) or np.any(ids<0) or np.any(ids!=np.floor(ids)):
        raise ValueError('Zone ids must be nonnegative integers')
    footprint=(ids>0)&np.isfinite(crop)&np.isfinite(ha)
    valid=footprint&np.isfinite(loss)&np.isfinite(bare)
    if common_support is not None:
        valid &= np.broadcast_to(np.asarray(common_support,dtype=bool),ids.shape)
    stress=stressed_soil_loss(np.where(valid,loss,np.nan),np.where(valid,bare,np.nan),provision_loss)
    weight=np.where(footprint,crop*ha,0.)
    def aggregate(mask):
        return np.bincount(ids.ravel().astype(int),weights=np.where(mask,weight,0.).ravel())
    total=aggregate(valid); severe=aggregate(valid&(loss>threshold))
    stressed=aggregate(valid&(stress>threshold)); excluded=aggregate(footprint&~valid)
    active=np.flatnonzero((total+excluded)>0)
    return pd.DataFrame(dict(zone_id=active,valid_crop_ha=total[active],
        severe_crop_ha=severe[active],stressed_severe_crop_ha=stressed[active],
        excluded_crop_ha=excluded[active],
        level=productivity_level(severe[active],total[active]),
        stressed_level=productivity_level(stressed[active],total[active])))


def stressed_soil_loss(usle, rkls, provision_loss=0.2):
    """Reduce on-site prevented soil loss, holding potential erosion fixed.

All quantities are t/ha/year. The caller applies the agreed onset and thresholds
the resulting soil loss, rather than scaling a signed productivity level.
"""
    if not 0 <= provision_loss <= 1:
        raise ValueError('provision_loss must be a fraction in [0, 1]')
    actual, potential = np.broadcast_arrays(np.asarray(usle, dtype=float),
                                            np.asarray(rkls, dtype=float))
    valid = np.isfinite(actual) & np.isfinite(potential)
    if np.any(np.isinf(actual) | np.isinf(potential)):
        raise ValueError('Soil loss must be finite or missing')
    if np.any(valid & ((actual < 0) | (potential < actual))):
        raise ValueError('Require 0 <= actual soil loss <= potential soil loss')
    return np.where(valid, actual + provision_loss * (potential - actual), np.nan)


def annual_damage_change(base_level, anchor_years, ordinary_levels, years,
                         base_year=2023, stressed_levels=None, stress_from=2030):
    """Interpolate negative productivity levels, then subtract the 2023 level.

    A physical stress begins at its onset, with no anticipation in preceding
    years. Stress onset must be an anchor so its physical level is known.
    """
    anchors = np.asarray(anchor_years, dtype=int)
    levels = np.asarray(ordinary_levels, dtype=float)
    dates = np.asarray(years, dtype=int)
    if (anchors.ndim != 1 or levels.shape != anchors.shape or anchors.size == 0
            or np.any(np.diff(anchors) <= 0) or np.any(anchors <= base_year)):
        raise ValueError('Need sorted unique anchor years after the base year and matching levels')
    if np.any(dates < base_year) or np.any(dates > anchors[-1]):
        raise ValueError('Requested dates are outside the trajectory')
    if not np.isfinite(base_level) or not np.all(np.isfinite(levels)):
        raise ValueError('Missing damage levels cannot define a trajectory')
    result = np.interp(dates, np.r_[base_year, anchors], np.r_[base_level, levels])
    if stressed_levels is not None:
        stress = np.asarray(stressed_levels, dtype=float)
        if stress.shape != levels.shape or not np.all(np.isfinite(stress)):
            raise ValueError('Need finite stressed levels at every anchor')
        if stress_from not in anchors:
            raise ValueError('Stress onset needs a physical anchor')
        if np.any(stress > levels + 1e-10):
            raise ValueError('Losing prevention must not improve productivity on the same support')
        active = dates >= stress_from
        result[active] = np.interp(dates[active], anchors, stress)
    return result - base_level

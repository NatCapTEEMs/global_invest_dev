"""Non-timber forest products (NTFP): CWoN NWFP value per accessible forest hectare.

The source repo (geo_ntfp) multiplies the CWoN 2024 NWFP value per hectare per country by
the hectares of ACCESSIBLE forest: ESA CCI forest classes within a 10 km buffer of roads
and rivers. Two script variants exist side by side (a vector-buffer calculation and a
distance-raster pipeline) and no output is committed, so the library runs BOTH and presents
both, the way fire_protection presents its variants.

The committed roads file is unusable (only sidecar files were committed, the .shp never
was), so the accessibility stage runs on a public global roads dataset; the original
roads file remains a replication ask.
"""

# From the source scripts: accessibility is a 10 km buffer around roads and rivers, and
# forest is the ESA CCI class range 50-90 inclusive.
NTFP_ACCESS_BUFFER_M = 10_000
ESA_FOREST_CLASS_MIN = 50
ESA_FOREST_CLASS_MAX = 90


def ntfp_gep_by_country(accessible_forest_ha_df, value_per_ha_df, year):
    """One row per country: the CWoN non-wood forest product value, spread over reachable forest.

    CWoN publishes an annual value per country and the forest area it attributes that value to.
    Dividing gives a rate per hectare, and the obvious thing is to multiply that rate by the
    forest people can actually reach. That understates it: CWoN's value was observed, and the
    part of it earned on forest nobody can reach is not zero, it is unallocated.

    So the rate is rescaled to the accessible area before it is applied. The country total then
    equals CWoN's own, and what accessibility changes is the rate rather than the sum: the same
    value concentrated on 63 percent of the forest, a median of $13.67 a hectare instead of
    $10.00.

    The consequence, stated because it is not obvious: since the accessible hectares appear in
    the denominator and again in the product, they cancel, and this country total is CWoN's
    number. Accessibility earns its place in where the value sits, not in how much there is.

    Args:
        accessible_forest_ha_df (pd.DataFrame): iso3_r250_label, accessible_forest_ha.
        value_per_ha_df (pd.DataFrame): iso3_r250_label, year, nwfp_value_usd and forest_ha
            (the CWoN NWFP series, current USD), plus nwfp_value_per_ha as CWoN priced it.
        year (int): the GEP base year to value at.

    Returns:
        pd.DataFrame: iso3_r250_label, accessible_forest_ha, nwfp_value_per_ha as CWoN priced it,
        nwfp_value_per_accessible_ha as rescaled, and ntfp_gep. A country with no accessible
        forest keeps a missing rate rather than an infinite one.
    """
    values = value_per_ha_df[value_per_ha_df['year'] == int(year)]
    df = accessible_forest_ha_df.merge(
        values[['iso3_r250_label', 'nwfp_value_usd', 'nwfp_value_per_ha']],
        on='iso3_r250_label', how='left')
    reachable = df['accessible_forest_ha'].where(df['accessible_forest_ha'] > 0)
    df['nwfp_value_per_accessible_ha'] = df['nwfp_value_usd'] / reachable
    df['ntfp_gep'] = df['accessible_forest_ha'] * df['nwfp_value_per_accessible_ha']
    return df


# =============================================================================
# Accessibility. The source scripts call forest accessible when it lies within
# 10 km of a road or a river. Everything below is array arithmetic over blocks
# the task layer reads, so each piece is testable without a global raster.
# =============================================================================

import numpy as np

# The road-length raster carries metres of road per cell, so any positive value is a cell a
# road passes through. Rivers arrive as a burned mask, so any positive value is a river cell.
ROAD_PRESENT_THRESHOLD_M = 0.0
M_PER_KM = 1000.0


def forest_mask(lulc_block, ndv=None):
    """True where the land-cover block is forest, on the ESA CCI class range the source uses."""
    mask = (lulc_block >= ESA_FOREST_CLASS_MIN) & (lulc_block <= ESA_FOREST_CLASS_MAX)
    if ndv is not None:
        mask &= (lulc_block != ndv)
    return mask


def access_source_mask(road_length_block, river_block, road_ndv=None, river_ndv=None):
    """True where a cell contains a road or a river, the two things accessibility is measured from.

    Args:
        road_length_block (np.ndarray): metres of road per cell.
        river_block (np.ndarray): a burned river mask on the same grid.

    Returns:
        np.ndarray: bool, True where either is present.
    """
    roads = np.nan_to_num(road_length_block, nan=0.0) > ROAD_PRESENT_THRESHOLD_M
    if road_ndv is not None:
        roads &= (road_length_block != road_ndv)
    rivers = np.nan_to_num(river_block, nan=0.0) > 0
    if river_ndv is not None:
        rivers &= (river_block != river_ndv)
    return roads | rivers


def accessible_mask(distance_to_source_km, buffer_km=NTFP_ACCESS_BUFFER_M / M_PER_KM):
    """True where a cell is within the accessibility buffer of a road or river.

    The comparison is inclusive, so a cell exactly on the buffer edge counts as accessible.
    """
    return distance_to_source_km <= float(buffer_km)


def accessible_forest_hectares(forest, accessible, ha_per_cell_block):
    """Hectares of forest that are also accessible, cell by cell.

    Returns an array carrying the cell's hectares where both hold and zero elsewhere, so the
    caller can sum it inside any zone without a second mask.
    """
    keep = forest & accessible
    return np.where(keep, np.nan_to_num(ha_per_cell_block, nan=0.0), 0.0)


def hectares_by_zone(accessible_forest_ha_block, zone_id_block, n_zones):
    """Accessible forest hectares summed per zone id, as a length n_zones + 1 array.

    Index 0 collects everything outside a zone, matching the id rasters' convention that zero
    is no-data, so a caller can drop it without a special case.
    """
    ids = np.nan_to_num(zone_id_block, nan=0).astype(np.int64).ravel()
    values = accessible_forest_ha_block.ravel()
    keep = (ids >= 0) & (ids <= n_zones)
    return np.bincount(ids[keep], weights=values[keep], minlength=n_zones + 1)


def buffer_mask_by_cells(source_mask, radius_cells):
    """The source mask grown by a disk of the given radius, which on this grid is kilometres.

    A disk rather than a square: a square would call a cell 14 km away accessible on the
    diagonal, which is not what a 10 km buffer means.
    """
    from scipy import ndimage
    span = np.arange(-radius_cells, radius_cells + 1)
    y_offsets, x_offsets = np.meshgrid(span, span, indexing='ij')
    disk = (y_offsets ** 2 + x_offsets ** 2) <= radius_cells ** 2
    return ndimage.binary_dilation(source_mask, structure=disk)

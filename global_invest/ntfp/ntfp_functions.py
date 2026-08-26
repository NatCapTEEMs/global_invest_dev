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


def nwfp_rate_long(price_wide_df):
    """The published rate table reshaped from one column per year to one row per country-year.

    The source ships the World Bank series as it comes, wide, with a year in every column
    heading. Everything downstream keys on country and year, so it is melted once here rather
    than indexed by a column name at each use.

    Args:
        price_wide_df (pd.DataFrame): iso3_r250_label plus one numeric column per year.

    Returns:
        pd.DataFrame: iso3_r250_label, year, nwfp_value_per_ha.
    """
    import pandas as pd

    years = [c for c in price_wide_df.columns if str(c).isdigit()]
    long = price_wide_df.melt(id_vars=['iso3_r250_label'], value_vars=years,
                              var_name='year', value_name='nwfp_value_per_ha')
    long['year'] = long['year'].astype(int)
    long['nwfp_value_per_ha'] = pd.to_numeric(long['nwfp_value_per_ha'], errors='coerce')
    return long.dropna(subset=['nwfp_value_per_ha'])


def ntfp_gep_by_country(accessible_forest_ha_df, value_per_ha_df, year):
    """One row per country: CWoN's non-wood forest product rate applied to reachable forest.

    CWoN publishes an annual value per hectare per country. That rate is applied to the forest
    people can actually reach, which is the forest within 10 km of a road or a river and green
    enough to yield a product.

    Args:
        accessible_forest_ha_df (pd.DataFrame): iso3_r250_label, accessible_forest_ha.
        value_per_ha_df (pd.DataFrame): iso3_r250_label, year, nwfp_value_per_ha.
        year (int): the GEP base year to value at.

    Returns:
        pd.DataFrame: iso3_r250_label, accessible_forest_ha, nwfp_value_per_ha and ntfp_gep.
    """
    values = value_per_ha_df[value_per_ha_df['year'] == int(year)]
    df = accessible_forest_ha_df.merge(
        values[['iso3_r250_label', 'nwfp_value_per_ha']], on='iso3_r250_label', how='left')
    df['ntfp_gep'] = df['accessible_forest_ha'] * df['nwfp_value_per_ha']
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

# The land-cover map calls a cell forest from its class alone, which lets bare and sparse cells
# into the mask. A five-year mean NDVI screens them out. The values are MODIS MOD13Q1 convention:
# int16 storing NDVI times 10,000, with -9999 for no data, and 0.2 is the floor below which a
# cell carries too little live vegetation to yield a harvestable product.
NDVI_MIN_THRESHOLD = 0.20
NDVI_SCALE_FACTOR = 0.0001
NDVI_NODATA = -9999


def forest_mask(lulc_block, ndv=None):
    """True where the land-cover block is forest, on the ESA CCI class range the source uses."""
    mask = (lulc_block >= ESA_FOREST_CLASS_MIN) & (lulc_block <= ESA_FOREST_CLASS_MAX)
    if ndv is not None:
        mask &= (lulc_block != ndv)
    return mask


def vegetated_forest_mask(forest, ndvi_block, ndvi_ndv=NDVI_NODATA,
                          scale=NDVI_SCALE_FACTOR, threshold=NDVI_MIN_THRESHOLD):
    """The forest mask narrowed to cells carrying enough live vegetation to yield a product.

    A cell whose NDVI is missing is dropped rather than kept. Absent evidence of vegetation is
    not evidence of it, and keeping those cells would credit a country with reachable forest the
    imagery cannot see.

    Args:
        forest (np.ndarray): boolean, the land-cover forest mask.
        ndvi_block (np.ndarray): the raw integer NDVI block, on the same grid.
        ndvi_ndv: the raw value standing for no data.
        scale (float): what the raw integer is multiplied by to give NDVI.
        threshold (float): the NDVI floor a cell must reach.

    Returns:
        np.ndarray: boolean, forest cells that also pass the NDVI floor.
    """
    # The comparison is made in the raster's own integer units rather than on scaled floats.
    # In float32 a raw 2000 scales to 0.19999999, so a cell sitting exactly on a 0.2 floor is
    # dropped by a rounding artifact rather than by its vegetation. Integers do not have that
    # problem, and it avoids scaling a global array to compare it against one number.
    raw_threshold = round(threshold / scale)
    return forest & (ndvi_block != ndvi_ndv) & (ndvi_block >= raw_threshold)


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

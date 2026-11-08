"""Non-timber forest products (NTFP): the NWFP value per hectare times FAO total forest.

The rate is IUCN's non-wood forest product value per hectare per country, which CWoN 2024
publishes as an average over all of a country's forest and multiplies by FAO total forest area
(the source report, PDF pages 61 and 63, printed 53 and 55, and `forest_nontimber.do` line 340).
The account does the same: the rate unadjusted, times FAO FRA 2020 total forest interpolated to
the base year by CWoN's own rule, in every country. A country FRA 2020 cannot give stays missing.

Three alternatives sit beside it as their own tables, never as the account's value: the rate
times the area CWoN publishes, the rate times all mapped ESA forest, and CWoN's divide-by-five for
three countries. The earlier valuation, the rate times accessible forest, is kept as a fourth
table.

The source repo (geo_ntfp) multiplies the rate by the hectares of ACCESSIBLE forest: ESA CCI
forest classes within a 10 km buffer of roads and rivers. The library grows that reach on the
account's own grid (see ntfp_tasks).

The roads layer is the configured `ntfp_roads_vector_path`
(`global_invest/ntfp/input/global_roads/grip4_global_roads.gpkg`), the GRIP4 roads (Meijer et al.
2018) merged from the seven regional geodatabases into one GeoPackage layer of 25,758,453 lines.
The registered source code reads `globalroads.shp` from its own base data. The copy staged under
that name holds 21,438,033 lines and no highway, primary or secondary road, so the library departs
from it. The authors' shared copy of `globalroads.*` is identical to it in all six files by
sha256, which identifies the file they shared, not the one their published run read.
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


def ntfp_gep_by_country(area_df, value_per_ha_df, year, area_column, value_column='ntfp_gep'):
    """One row per country: the non-wood forest product rate times a forest area.

    The account's value takes FAO total forest as the area. The same product over the area CWoN
    publishes, all mapped ESA forest, or accessible forest gives the alternatives kept beside it. A missing area or a
    missing rate gives a missing value, and a zero of either gives zero.

    Args:
        area_df (pd.DataFrame): iso3_r250_label and the area column, one row per country.
        value_per_ha_df (pd.DataFrame): iso3_r250_label, year, nwfp_value_per_ha.
        year (int): the GEP base year to value at.
        area_column (str): the hectares to multiply, e.g. fao_forest_ha.
        value_column (str): the name of the product column.

    Returns:
        pd.DataFrame: area_df's columns, nwfp_value_per_ha and value_column.
    """
    if not area_df['iso3_r250_label'].is_unique:
        raise ValueError('the area table repeats a country label')
    values = value_per_ha_df[value_per_ha_df['year'] == int(year)]
    if not values['iso3_r250_label'].is_unique:
        raise ValueError('the rate table repeats a country label in %d' % int(year))
    df = area_df.merge(
        values[['iso3_r250_label', 'nwfp_value_per_ha']], on='iso3_r250_label', how='left')
    df[value_column] = df[area_column] * df['nwfp_value_per_ha']
    return df


def divided_rates(value_per_ha_df, labels, divisor):
    """The rate table with the rate of each named country divided by a constant.

    CWoN divides the NWFP value per hectare by five for three countries
    (`forest_nontimber.do` line 305) and states no reason. The account keeps the published rate,
    and this gives the alternative as a sensitivity. A named country the table does not carry is
    a configuration error, not a silent no-op.

    Args:
        value_per_ha_df (pd.DataFrame): iso3_r250_label, year, nwfp_value_per_ha.
        labels (list): the iso3_r250_labels to divide.
        divisor (float): what their rate is divided by.

    Returns:
        pd.DataFrame: a copy with those rates divided, every other row unchanged.
    """
    absent = sorted(set(labels) - set(value_per_ha_df['iso3_r250_label']))
    if absent:
        raise ValueError('the rate table has no row for %s' % absent)
    out = value_per_ha_df.copy()
    named = out['iso3_r250_label'].isin(labels)
    out.loc[named, 'nwfp_value_per_ha'] = out.loc[named, 'nwfp_value_per_ha'] / float(divisor)
    return out


# =============================================================================
# FAO total forest. CWoN multiplies the rate by FAO total forest area, from FAO
# FRA 2020 interpolated by year. The account reads FRA 2020 directly and
# interpolates it the way CWoN does, in every country (fao_forest_ha_by_country).
# The area CWoN publishes is kept as a comparison (cwon_forest_ha_by_country):
# it equals FRA 2020 in 145 of its 151 countries, and the other six are FRA 2010
# extrapolated.
# =============================================================================

CWON_AREA_SERIES = 'Forest, non-timber ecosystem service total area (ha)'
CWON_MISSING = '..'
FRA_THOUSAND_HA = 1000.0
FAO_SOURCE_FRA = 'FAO FRA 2020, interpolated'


def cwon_forest_area(cwon_area_wide_df, year):
    """CWoN's published total forest area for one year, in hectares, keyed by its country code.

    The table is the package's `forest_nontimber_total_area.xlsx`: one row per country, a
    `YR<year>` column per year, and `..` where CWoN publishes no value. Any other text in the
    column is an error, so a changed export cannot turn into a missing value unnoticed.

    Args:
        cwon_area_wide_df (pd.DataFrame): countrycode, series and one YR<year> column per year.
        year (int): the year to read.

    Returns:
        pd.Series: hectares by countrycode, missing where CWoN publishes none.
    """
    import pandas as pd

    series = set(cwon_area_wide_df['series'].dropna())
    if series != {CWON_AREA_SERIES}:
        raise ValueError('expected the series %r in hectares, found %s' % (CWON_AREA_SERIES, sorted(series)))
    if not cwon_area_wide_df['countrycode'].is_unique:
        raise ValueError('the CWoN area table repeats a country code')
    text = cwon_area_wide_df['YR%d' % int(year)]
    missing = text.astype(str).str.strip() == CWON_MISSING
    area = pd.to_numeric(text.where(~missing), errors='coerce')
    unreadable = area.isna() & ~missing & text.notna()
    if unreadable.any():
        raise ValueError('the CWoN area column holds text other than %r: %s'
                         % (CWON_MISSING, sorted(text[unreadable].astype(str).unique())))
    return pd.Series(area.to_numpy(dtype=float), index=cwon_area_wide_df['countrycode'].to_numpy(),
                     name='cwon_forest_ha')


def fra_forest_area(fra_long_df, year):
    """FAO FRA 2020 forest area for one year, in hectares, by ISO3 code.

    FRA 2020 reports 1990, 2000, 2010, 2015 and 2020, in thousand hectares. A year between two
    reported years is interpolated linearly between the nearest one on each side, which is CWoN's
    rule (`forest_timber_depletion.do`, the `ipolate` at line 518). A year with no reported value
    on one side stays missing here: CWoN would extrapolate, and the FRA 2020 years bracket every
    year up to 2020.

    Args:
        fra_long_df (pd.DataFrame): iso3, year, forest, one row per country and reported year.
        year (int): the year wanted.

    Returns:
        pd.Series: hectares by iso3, missing where FRA cannot give the year.
    """
    import pandas as pd

    rows = fra_long_df[['iso3', 'year', 'forest']].dropna(subset=['iso3', 'forest'])
    if rows.duplicated(['iso3', 'year']).any():
        raise ValueError('FRA repeats a country-year')
    wide = rows.pivot(index='iso3', columns='year', values='forest')
    years = np.array(sorted(int(y) for y in wide.columns))
    out = {}
    for iso3, values in wide.iterrows():
        held = years[values.reindex(years).notna().to_numpy()]
        if int(year) in held:
            out[iso3] = float(values[int(year)]) * FRA_THOUSAND_HA
            continue
        below, above = held[held < int(year)], held[held > int(year)]
        if len(below) and len(above):
            y0, y1 = below.max(), above.min()
            v0, v1 = float(values[y0]), float(values[y1])
            out[iso3] = (v0 + (v1 - v0) * (int(year) - y0) / (y1 - y0)) * FRA_THOUSAND_HA
        else:
            out[iso3] = float('nan')
    return pd.Series(out, name='fra_forest_ha', dtype=float)


def area_on_country_keys(countries_df, area, column):
    """An area series keyed by ISO3 code, mapped onto the country keys as one named column."""
    if not countries_df['iso3_r250_label'].is_unique:
        raise ValueError('the country list repeats a label')
    out = countries_df[['iso3_r250_id', 'iso3_r250_label']].copy()
    out[column] = out['iso3_r250_label'].map(area)
    return out


def fao_forest_ha_by_country(countries_df, fra_area):
    """FAO total forest per country: FRA 2020 interpolated to the base year, in every country.

    FRA 2020 is the vintage CWoN's code ranks first (`forest_timber_depletion.do` line 511), read
    and interpolated by the same rule for every country. CWoN's published area differs from it in
    six countries (BOL, CIV, CZE, GBR, MKD, SWZ), where FRA 2020 rows went unmatched to a country
    code and FRA 2010's trend was extrapolated instead. That area is kept apart, as
    cwon_forest_ha_by_country. A country FRA 2020 cannot give stays missing. Nothing is filled
    from the land cover or from CWoN, and nothing is set to zero that FRA does not report as zero.

    Args:
        countries_df (pd.DataFrame): iso3_r250_id and iso3_r250_label, one row per country.
        fra_area (pd.Series): hectares by ISO3 code, from fra_forest_area.

    Returns:
        pd.DataFrame: iso3_r250_id, iso3_r250_label, fao_forest_ha and fao_forest_source.
    """
    out = area_on_country_keys(countries_df, fra_area, 'fao_forest_ha')
    out['fao_forest_source'] = np.where(out['fao_forest_ha'].notna(), FAO_SOURCE_FRA, None)
    return out


def cwon_forest_ha_by_country(countries_df, cwon_area):
    """The total forest area CWoN publishes per country, for the comparison with CWoN's own value.

    It is the area CWoN's NWFP value multiplies, so the rate times it gives CWoN's own value up to
    the rate file's printing precision, except in the three countries whose rate CWoN divides by
    five. Missing wherever CWoN publishes no area, with nothing filled.

    Args:
        countries_df (pd.DataFrame): iso3_r250_id and iso3_r250_label, one row per country.
        cwon_area (pd.Series): hectares by country code, from cwon_forest_area.

    Returns:
        pd.DataFrame: iso3_r250_id, iso3_r250_label and cwon_forest_ha.
    """
    return area_on_country_keys(countries_df, cwon_area, 'cwon_forest_ha')


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



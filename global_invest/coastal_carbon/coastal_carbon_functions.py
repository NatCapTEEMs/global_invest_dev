"""Coastal-carbon science: pure functions over arrays and frames.

Nothing here opens a file. The task layer reads the rasters and vectors, hands the arrays and
frames in, and writes back what it gets -- so every step below can be pinned on a hand-built
input in the test suite.

TWO SURFACES BY DESIGN. Every function up to and including `coastal_carbon_storage_value_frame`
works on the marine surface (eemarine_r566: one row per EEZ or coastal region, the
gep_regions_input_path cell in es_config). Only the last two collapse that surface to one row
per country: `eez_storage_value_by_iso3` keeps the marine `_EEZ` rows, and
`collapse_to_iso3_r250` joins them onto the canonical r264 row per country. Summing r264 rows
directly would count a split country once per sub-region -- see global_invest/utilities.py.

The valuation is storage-only, and identical in form to terrestrial carbon: mapped habitat
area x per-habitat carbon density = stock (Mg C), stock x the base-year rental social cost of
carbon = GEP ($).

Soil carbon is the stock in the configured top layer (coastal_carbon_soil_depth_cm). The
mangrove and salt marsh soil products and their latitude fallbacks are stated for the source
depth (coastal_carbon_soil_source_depth_cm), and the density functions take the ratio of the
two as `soil_depth_share`, a linear share of the profile.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

# The three habitats this service values, in the order every combined frame is written in.
COASTAL_ECOSYSTEMS = ('mangrove', 'salt_marsh', 'seagrass')

M2_PER_HA = 10000.0
# EPSG:6933 (NSIDC EASE-Grid 2.0 Global): the equal-area projection polygon extents are
# measured in. Areas measured in a geographic CRS are not areas.
EQUAL_AREA_EPSG = 6933
MG_C_PER_PG = 1e9
# The marine surface carries one row per EEZ plus land/territorial rows for the same coast;
# only the EEZ rows are valued, so the coast is counted once.
EEZ_LABEL_SUFFIX = '_EEZ'


# ============================================================================
# Per-pixel ecosystem carbon density
# ============================================================================

# Mangrove (Hamilton & Friess 2018 + IPCC 2014 zone BGB + Sanderman 2018 SOC)
MANGROVE_BGB_AGB_RATIO_TROPICAL_WET = 0.49
MANGROVE_BGB_AGB_RATIO_TROPICAL_DRY = 0.29
MANGROVE_BGB_AGB_RATIO_SUBTROPICAL = 0.96
MANGROVE_TROPICAL_LAT_THRESHOLD = 23.5
MANGROVE_TROPICAL_WET_PRECIP_THRESHOLD_MM = 2000.0
MANGROVE_AGB_TO_C = 0.48   # Hamilton & Friess 2018
MANGROVE_BGB_TO_C = 0.39   # Howard et al. 2014
# Hamilton & Friess 2018 EQ5: AGB (t/ha) falls linearly with distance from the equator.
MANGROVE_AGB_LAT_SLOPE_T_HA_PER_DEG = -6.4305
MANGROVE_AGB_LAT_INTERCEPT_T_HA = 271.747
# SOC used only when no SOC raster is supplied: Mg C/ha over the source depth (the top 1 m),
# poleward value first and then narrower latitude bands overriding it, so the equator ends up
# richest.
MANGROVE_FALLBACK_SOC_POLEWARD_MG_HA = 250.0
MANGROVE_FALLBACK_SOC_BANDS_MG_HA = ((25.0, 300.0), (20.0, 350.0), (15.0, 375.0), (10.0, 400.0))


def calculate_mangrove_density_array(latitude_arr, soil_depth_share, precipitation_arr=None,
                                     soc_arr=None):
    """
    Vectorised mangrove total carbon density (Mg C/ha).

    AGB: Hamilton & Friess (2018) EQ5 latitude regression
        AGB (t/ha) = max(0, -6.4305 * |lat| + 271.747)
    BGB: AGB x IPCC 2014 zone-specific ratio (0.49 tropical wet, 0.29 tropical
         dry, 0.96 subtropical). If precipitation_arr is None, all tropics use
         the wet ratio (0.49). Tropical/subtropical split at |lat| = 23.5 deg.
    SOC: from soc_arr (Sanderman 2018 raster, 0-100 cm) when provided; otherwise a
         latitude step function for the top 1 m, 400 down to 250 Mg C/ha by zone.
         Either is multiplied by soil_depth_share, the configured depth over the
         source depth (0.3 for 30 of 100 cm).
    """
    lat_abs = np.abs(latitude_arr).astype(np.float64)
    agb_t_per_ha = np.maximum(
        0.0, MANGROVE_AGB_LAT_SLOPE_T_HA_PER_DEG * lat_abs + MANGROVE_AGB_LAT_INTERCEPT_T_HA)

    is_tropical = lat_abs < MANGROVE_TROPICAL_LAT_THRESHOLD
    if precipitation_arr is not None:
        is_wet = precipitation_arr > MANGROVE_TROPICAL_WET_PRECIP_THRESHOLD_MM
        bgb_ratio = np.where(
            is_tropical & is_wet, MANGROVE_BGB_AGB_RATIO_TROPICAL_WET,
            np.where(is_tropical & ~is_wet, MANGROVE_BGB_AGB_RATIO_TROPICAL_DRY,
                     MANGROVE_BGB_AGB_RATIO_SUBTROPICAL),
        )
    else:
        bgb_ratio = np.where(
            is_tropical, MANGROVE_BGB_AGB_RATIO_TROPICAL_WET,
            MANGROVE_BGB_AGB_RATIO_SUBTROPICAL,
        )

    bgb_t_per_ha = agb_t_per_ha * bgb_ratio
    agb_c = agb_t_per_ha * MANGROVE_AGB_TO_C
    bgb_c = bgb_t_per_ha * MANGROVE_BGB_TO_C

    if soc_arr is not None:
        soc = soc_arr.astype(np.float64)
    else:
        soc = np.full_like(lat_abs, MANGROVE_FALLBACK_SOC_POLEWARD_MG_HA, dtype=np.float64)
        for lat_below, soc_mg_per_ha in MANGROVE_FALLBACK_SOC_BANDS_MG_HA:
            soc = np.where(lat_abs < lat_below, soc_mg_per_ha, soc)
    soc = soc * soil_depth_share

    return {
        'agb_c_mg_per_ha':   agb_c,
        'bgb_c_mg_per_ha':   bgb_c,
        'soil_c_mg_per_ha':  soc,
        'total_c_mg_per_ha': agb_c + bgb_c + soc,
    }


# Salt marsh (Chmura 2003 AGB + 2.5 BGB ratio + Maxwell 2024 MarSOC)
SALT_MARSH_AGB_MEDIAN_T_HA = 5.0
SALT_MARSH_BGB_AGB_RATIO = 2.5
SALT_MARSH_AGB_TO_C = 0.45
SALT_MARSH_BGB_TO_C = 0.41
SALT_MARSH_TROPICAL_BOOST_LAT = 25.0
SALT_MARSH_TROPICAL_BOOST_FACTOR = 1.2
# Same fallback convention as mangrove, also for the top 1 m: poleward value first, narrower
# bands override it. Salt marsh soils run the other way -- richest at high latitude.
SALT_MARSH_FALLBACK_SOC_POLEWARD_MG_HA = 350.0
SALT_MARSH_FALLBACK_SOC_BANDS_MG_HA = ((45.0, 250.0), (35.0, 220.0), (20.0, 180.0))


def calculate_salt_marsh_density_array(latitude_arr, soil_depth_share, soc_arr=None):
    """
    Vectorised salt marsh total carbon density (Mg C/ha).

    AGB: Chmura et al. 2003 median (5 t/ha) with 1.2x tropical boost (|lat| < 25).
    BGB: AGB x 2.5 (extensive root systems).
    SOC: from soc_arr (Maxwell 2024 MarSOC raster, 0-100 cm) when provided; otherwise a
         latitude step function for the top 1 m, 180 up to 350 Mg C/ha by zone. Either
         is multiplied by soil_depth_share, the configured depth over the source depth.
    """
    lat_abs = np.abs(latitude_arr).astype(np.float64)

    agb_t_per_ha = np.where(
        lat_abs < SALT_MARSH_TROPICAL_BOOST_LAT,
        SALT_MARSH_AGB_MEDIAN_T_HA * SALT_MARSH_TROPICAL_BOOST_FACTOR,
        SALT_MARSH_AGB_MEDIAN_T_HA,
    ).astype(np.float64)
    bgb_t_per_ha = agb_t_per_ha * SALT_MARSH_BGB_AGB_RATIO
    agb_c = agb_t_per_ha * SALT_MARSH_AGB_TO_C
    bgb_c = bgb_t_per_ha * SALT_MARSH_BGB_TO_C

    if soc_arr is not None:
        soc = soc_arr.astype(np.float64)
    else:
        soc = np.full_like(lat_abs, SALT_MARSH_FALLBACK_SOC_POLEWARD_MG_HA, dtype=np.float64)
        for lat_below, soc_mg_per_ha in SALT_MARSH_FALLBACK_SOC_BANDS_MG_HA:
            soc = np.where(lat_abs < lat_below, soc_mg_per_ha, soc)
    soc = soc * soil_depth_share

    return {
        'agb_c_mg_per_ha':   agb_c,
        'bgb_c_mg_per_ha':   bgb_c,
        'soil_c_mg_per_ha':  soc,
        'total_c_mg_per_ha': agb_c + bgb_c + soc,
    }


def seagrass_pool_densities(biomass_c_mg_per_ha, agb_fraction, soil_c_mg_per_ha):
    """Per-pool seagrass carbon density (Mg C/ha), one value for every mapped pixel.

    The extent is a presence map with no genus or meadow type, so one biomass density is split
    above and below ground by `agb_fraction`, and soil is one density for the configured depth.
    """
    agb = biomass_c_mg_per_ha * agb_fraction
    bgb = biomass_c_mg_per_ha * (1.0 - agb_fraction)
    return {
        'agb_c_mg_per_ha':   agb,
        'bgb_c_mg_per_ha':   bgb,
        'soil_c_mg_per_ha':  soil_c_mg_per_ha,
        'total_c_mg_per_ha': agb + bgb + soil_c_mg_per_ha,
    }


# ============================================================================
# Column conventions
# ============================================================================

def stock_columns(ecosystem):
    """The four per-pool stock columns this module writes for an ecosystem, in write order (Mg C)."""
    return [f'{ecosystem}_agb_c_total_mg', f'{ecosystem}_bgb_c_total_mg',
            f'{ecosystem}_soil_c_total_mg', f'{ecosystem}_total_c_stock_mg']


def stock_to_value_columns(ecosystem):
    """Stock column -> storage-value column, in write order; the last pair is the ecosystem total.

    The ecosystem's own total drops the pool word (`mangrove_storage_value`), because that is
    the column gep_calculation sums across ecosystems.
    """
    agb, bgb, soil, total = stock_columns(ecosystem)
    return {agb: f'{ecosystem}_agb_storage_value',
            bgb: f'{ecosystem}_bgb_storage_value',
            soil: f'{ecosystem}_soil_storage_value',
            total: f'{ecosystem}_storage_value'}


# ============================================================================
# Per-pixel stock, accumulated per region
# ============================================================================

def pixel_stock_sums(coverage_block, region_id_block, ha_per_cell_block, latitude_block,
                     density_func, n_region_ids, extras=None):
    """Per-region carbon stock (Mg C) contributed by one block of pixels.

    Density is evaluated at each pixel's latitude -- plus whatever extra per-pixel rasters the
    ecosystem's density function takes -- multiplied by the hectares of habitat in the pixel, and
    accumulated into the region the pixel falls in. A pixel counts only where the habitat covers
    some of it, a region id is present, and the cell has area. A NaN soil density (a nodata pixel
    inside an external SOC raster) contributes zero instead of poisoning its region's total.

    coverage_block is the share of each cell the habitat covers, in [0, 1]. A 0/1 mask is the
    special case where a cell the habitat merely touches counts whole, which for a habitat about
    one cell wide roughly doubles its extent.

    Returns a dict keyed 'agb', 'bgb', 'soil', 'total', each a float array of length
    n_region_ids indexed by region id.
    """
    valid = (coverage_block > 0) & (region_id_block > 0) & (ha_per_cell_block > 0)
    densities = density_func(latitude_block, **(extras or {}))

    ha_valid = ha_per_cell_block[valid] * coverage_block[valid]
    region_valid = region_id_block[valid].astype(np.int64)
    agb_pixels = ha_valid * densities['agb_c_mg_per_ha'][valid]
    bgb_pixels = ha_valid * densities['bgb_c_mg_per_ha'][valid]
    soil_pixels = ha_valid * np.nan_to_num(densities['soil_c_mg_per_ha'][valid], nan=0.0)

    return {
        'agb': np.bincount(region_valid, weights=agb_pixels, minlength=n_region_ids),
        'bgb': np.bincount(region_valid, weights=bgb_pixels, minlength=n_region_ids),
        'soil': np.bincount(region_valid, weights=soil_pixels, minlength=n_region_ids),
        'total': np.bincount(region_valid, weights=agb_pixels + bgb_pixels + soil_pixels,
                             minlength=n_region_ids),
    }


def stock_by_region_frame(region_ids, sums, ecosystem, id_col):
    """The accumulated per-region sums as a frame, one row per region id in the order given."""
    ids = [int(region_id) for region_id in region_ids]
    agb, bgb, soil, total = stock_columns(ecosystem)
    return pd.DataFrame({
        id_col: ids,
        agb: [sums['agb'][region_id] for region_id in ids],
        bgb: [sums['bgb'][region_id] for region_id in ids],
        soil: [sums['soil'][region_id] for region_id in ids],
        total: [sums['total'][region_id] for region_id in ids],
    })


# ============================================================================
# Habitat extent on the marine surface
# ============================================================================

def intersect_features_with_regions(gdf_features, gdf_regions, desc):
    """Clip a habitat layer to each region, keeping the region's attributes on every piece.

    Region by region rather than one bulk overlay, so a global run stays legible behind a
    progress bar. Two things keep it from being slow:

    The candidate set comes from the spatial index's `intersects` predicate rather than a
    bounding-box hit. Coastal habitat is long and thin while an EEZ region is huge, so a box
    test returns almost the whole layer; asking for real intersections drops about four fifths
    of it.

    A polygon that lies WHOLLY inside a region does not need clipping at all, only the region's
    attributes, and about nine in ten of them do. Only the boundary-straddling remainder goes
    through the overlay, which is the expensive part. On the largest regions that turns roughly
    165,000 overlay inputs into 18,000.

    Regions the habitat does not reach contribute no rows.
    """
    feature_sindex = gdf_features.sindex
    region_columns = [c for c in gdf_regions.columns if c != 'geometry']
    pieces = []
    for _, region in tqdm(gdf_regions.iterrows(), total=len(gdf_regions), desc=desc):
        touching = feature_sindex.query(region.geometry, predicate='intersects')
        if not len(touching):
            continue
        wholly_inside = set(feature_sindex.query(region.geometry, predicate='contains').tolist())

        inside_positions = [i for i in touching if i in wholly_inside]
        if inside_positions:
            whole = gdf_features.iloc[inside_positions].reset_index(drop=True)
            for column in region_columns:
                whole[column] = region[column]
            pieces.append(whole)

        cut_positions = [i for i in touching if i not in wholly_inside]
        if cut_positions:
            region_gdf = gpd.GeoDataFrame([region], crs=gdf_regions.crs)
            clipped = gpd.overlay(gdf_features.iloc[cut_positions], region_gdf,
                                  how='intersection')
            if not clipped.empty:
                pieces.append(clipped)

    if not pieces:
        raise ValueError(f'No intersections found ({desc}).')
    return gpd.GeoDataFrame(pd.concat(pieces, ignore_index=True), crs=gdf_regions.crs)


def add_equal_area_ha(gdf):
    """Polygon extent measured on the equal-area grid, in m2 and hectares.

    The returned frame is in the equal-area projection, which is what the polygon-level outputs
    are written in.
    """
    gdf = gdf.to_crs(epsg=EQUAL_AREA_EPSG)
    gdf['area_m2'] = gdf.geometry.area
    gdf['area_ha'] = gdf['area_m2'] / M2_PER_HA
    return gdf


def area_by_region(gdf_pieces, gdf_regions, id_col, how):
    """Hectares of habitat per region, carrying the region's attributes and geometry.

    `how` is the merge onto the region table: 'right' keeps every region, so a region with no
    habitat reports zero hectares; 'left' keeps only the regions the habitat reaches.
    """
    totals = gdf_pieces.groupby(id_col)['area_ha'].sum().reset_index()
    totals = totals.merge(gdf_regions, how=how, on=id_col)
    totals['area_ha'] = totals['area_ha'].fillna(0)
    # The merge on the region id brought each region's geometry; never attach it by row position.
    return gpd.GeoDataFrame(totals, geometry='geometry', crs=gdf_regions.crs)


def presence_area_by_zone(presence, zone_masks, pixel_area_ha):
    """Hectares of present pixels in each zone, a pixel belonging to every zone that holds its centre.

    `zone_masks` maps a zone id to a boolean array shaped like `presence`, True where the zone holds
    the pixel's centre. `pixel_area_ha` broadcasts to `presence`; on a geographic grid it is one
    value per row. Zones can overlap, so a pixel inside two zones counts in both zone totals.

    Returns a dict:
        area_ha: zone id -> hectares the zone holds (zones holding none are absent).
        shared_ha: zone id -> the part of that area another zone also holds.
        presence_ha: every present pixel, whether or not a zone holds it.
        unique_ha: every pixel some zone holds, counted once.
        overlap_counted_ha: what the zone totals add on top of unique_ha, each pixel's area once
            for every zone past the first.
    """
    pixel_area_ha = np.broadcast_to(pixel_area_ha, presence.shape)
    selected = {zone_id: presence & mask for zone_id, mask in zone_masks.items()}
    holders = np.zeros(presence.shape, dtype=np.int32)
    for held in selected.values():
        holders += held
    multiple = holders > 1
    area_ha, shared_ha = {}, {}
    for zone_id, held in selected.items():
        if held.any():
            area_ha[zone_id] = float(pixel_area_ha[held].sum())
            shared_ha[zone_id] = float(pixel_area_ha[held & multiple].sum())
    return {
        'area_ha': area_ha,
        'shared_ha': shared_ha,
        'presence_ha': float(pixel_area_ha[presence].sum()),
        'unique_ha': float(pixel_area_ha[holders > 0].sum()),
        'overlap_counted_ha': float((pixel_area_ha[multiple] * (holders[multiple] - 1)).sum()),
    }


def seagrass_stock_by_region(df_area, id_col, densities):
    """Per-region seagrass stock (Mg C): the region's hectares times each pool's density.

    One density applies everywhere (seagrass_pool_densities), so the stock is the area table
    scaled, one row per row of `df_area`. The total is the three pools added.
    """
    area_ha = df_area['area_ha'].to_numpy(dtype=np.float64)
    agb, bgb, soil, total = stock_columns('seagrass')
    df_stock = pd.DataFrame({
        id_col: df_area[id_col].to_numpy(),
        agb: area_ha * densities['agb_c_mg_per_ha'],
        bgb: area_ha * densities['bgb_c_mg_per_ha'],
        soil: area_ha * densities['soil_c_mg_per_ha'],
    })
    df_stock[total] = df_stock[agb] + df_stock[bgb] + df_stock[soil]
    return df_stock


# ============================================================================
# Physical stock -> GEP
# ============================================================================

def rental_price_for_year(df_prices, year, price_column):
    """The rental social cost of carbon a year's stock is valued at ($ per Mg C)."""
    return float(df_prices.loc[df_prices['year'] == year, price_column].iloc[0])


def apply_rental_price(df_stock, stock_to_value, rental_scc, year, price_column):
    """Physical to GEP: every pool's stock (Mg C) priced at the rental SCC ($ per Mg C).

    The price and the year travel with the values so the written table states what it was
    valued at rather than leaving it to the filename.
    """
    df = df_stock.copy()
    df['year'] = year
    df[price_column] = rental_scc
    for stock_col, value_col in stock_to_value.items():
        df[value_col] = df[stock_col] * rental_scc
    return df


def combine_ecosystem_areas_and_stocks(area_frames, stock_frames, df_regions, id_col):
    """One row per region: each habitat's area and per-pool stock, plus the two totals.

    `area_frames` and `stock_frames` map ecosystem -> frame. An ecosystem absent from a mapping
    contributes zero columns rather than dropping the region, which is how a run built without
    seagrass still reports a coastal total.
    """
    df_combined = None
    for ecosystem in COASTAL_ECOSYSTEMS:
        if ecosystem not in area_frames:
            continue
        area = area_frames[ecosystem][[id_col, 'area_ha']].rename(
            columns={'area_ha': f'{ecosystem}_area_ha'})
        df_combined = area if df_combined is None else df_combined.merge(
            area, on=id_col, how='outer')

    area_cols = [f'{ecosystem}_area_ha' for ecosystem in COASTAL_ECOSYSTEMS]
    for ecosystem in COASTAL_ECOSYSTEMS:
        if ecosystem not in area_frames:
            df_combined[f'{ecosystem}_area_ha'] = 0
    df_combined[area_cols] = df_combined[area_cols].fillna(0)
    df_combined['total_coastal_carbon_area_ha'] = df_combined[area_cols].sum(axis=1)

    for ecosystem in COASTAL_ECOSYSTEMS:
        cols = stock_columns(ecosystem)
        if ecosystem in stock_frames:
            df_combined = df_combined.merge(
                stock_frames[ecosystem][[id_col] + cols], on=id_col, how='left')
        else:
            for col in cols:
                df_combined[col] = 0
    all_stock_cols = [col for ecosystem in COASTAL_ECOSYSTEMS for col in stock_columns(ecosystem)]
    df_combined[all_stock_cols] = df_combined[all_stock_cols].fillna(0)
    df_combined['total_carbon_stock_mg'] = df_combined[
        [f'{ecosystem}_total_c_stock_mg' for ecosystem in COASTAL_ECOSYSTEMS]].sum(axis=1)

    return df_combined.merge(df_regions, on=id_col, how='left')


def coastal_carbon_storage_value_frame(df_areas, df_price, value_frames, id_col, base_year):
    """The marine-surface (r566) GEP frame: every habitat's storage value and their sum.

    The per-ecosystem storage_value tasks own stock x price; this merges their totals rather
    than recomputing them, so the same fact cannot drift between two places. A region absent
    from an ecosystem's table carries no value there, matching the zero stock in the combined
    table. Regions with no coastal habitat at all are dropped -- they are ocean rows, not
    zero-value countries.
    """
    df_gep = df_areas.copy()
    df_gep['year'] = base_year
    df_gep = df_gep.merge(df_price, on='year', how='left')

    for ecosystem in COASTAL_ECOSYSTEMS:
        value_col = f'{ecosystem}_storage_value'
        df_gep = df_gep.merge(value_frames[ecosystem][[id_col, value_col]], on=id_col, how='left')
        df_gep[value_col] = df_gep[value_col].fillna(0)
    df_gep['coastal_carbon_storage_value'] = df_gep[
        [f'{ecosystem}_storage_value' for ecosystem in COASTAL_ECOSYSTEMS]].sum(axis=1)

    return df_gep[df_gep['total_coastal_carbon_area_ha'] > 0]


def eez_storage_value_by_iso3(df_r566, region_label_col='eemarine_r566_label'):
    """The marine surface reduced to the rows that carry a country's coastal value.

    The r566 surface holds an `_EEZ` row per country AND land/territorial rows for the same
    coast, so summing every row counts habitat that the EEZ row already carries. Keeping only
    the `_EEZ` rows is a choice and the service's open question, not a cleanup step: the
    EEZ-only total and the all-rows total answer different questions. The module writes the
    EEZ-only figure. Summing the r566 table gives the all-rows figure.
    """
    is_eez = df_r566[region_label_col].astype(str).str.endswith(EEZ_LABEL_SUFFIX)
    return (df_r566[is_eez][['iso3_r250_label', 'coastal_carbon_storage_value']]
            .rename(columns={'coastal_carbon_storage_value': 'value'}))


def collapse_to_iso3_r250(df_value_by_iso3, df_r264):
    """One row per country: the marine values joined onto the canonical r264 row per country.

    The r264 correspondence splits large countries into sub-regions, so it is filtered to
    `ee_r264_label == iso3_r250_label` first -- without that, the join repeats a split country
    once per sub-region and the sum multiplies its value. Countries with no coastal value
    report zero rather than dropping out of the table.
    """
    canonical = df_r264[df_r264['ee_r264_label'] == df_r264['iso3_r250_label']]
    df_merged = canonical.merge(df_value_by_iso3, on='iso3_r250_label', how='left')
    df_merged['value'] = df_merged['value'].fillna(0)

    metadata_cols = [c for c in df_merged.columns if c not in ('iso3_r250_label', 'value')]
    agg = {'value': 'sum'}
    agg.update({c: 'first' for c in metadata_cols})
    return df_merged.groupby('iso3_r250_label', as_index=False).agg(agg)

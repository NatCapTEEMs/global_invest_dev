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
# External SOC products report the top 1 m; the stock convention here is the top 20 cm, taken
# as a linear share of the profile.
SOC_1M_TO_20CM_SCALAR = 0.2


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
# SOC used only when no SOC raster is supplied: Mg C/ha in the top 20 cm, poleward value first
# and then narrower latitude bands overriding it, so the equator ends up richest.
MANGROVE_FALLBACK_SOC_POLEWARD_MG_HA = 50.0
MANGROVE_FALLBACK_SOC_BANDS_MG_HA = ((25.0, 60.0), (20.0, 70.0), (15.0, 75.0), (10.0, 80.0))


def calculate_mangrove_density_array(latitude_arr, precipitation_arr=None,
                                     soc_arr=None):
    """
    Vectorised mangrove total carbon density (Mg C/ha).

    AGB: Hamilton & Friess (2018) EQ5 latitude regression
        AGB (t/ha) = max(0, -6.4305 * |lat| + 271.747)
    BGB: AGB x IPCC 2014 zone-specific ratio (0.49 tropical wet, 0.29 tropical
         dry, 0.96 subtropical). If precipitation_arr is None, all tropics use
         the wet ratio (0.49). Tropical/subtropical split at |lat| = 23.5 deg.
    SOC: from soc_arr (Sanderman 2018 raster) when provided; otherwise a
         latitude step function, 80 down to 50 Mg C/ha by zone -- the 400-to-250
         Mg C/ha profile for the top 1 m, scaled to the top 20 cm.
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
        soc = soc_arr.astype(np.float64) * SOC_1M_TO_20CM_SCALAR
    else:
        soc = np.full_like(lat_abs, MANGROVE_FALLBACK_SOC_POLEWARD_MG_HA, dtype=np.float64)
        for lat_below, soc_mg_per_ha in MANGROVE_FALLBACK_SOC_BANDS_MG_HA:
            soc = np.where(lat_abs < lat_below, soc_mg_per_ha, soc)

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
# Same fallback convention as mangrove: poleward value first, narrower bands override it. Salt
# marsh soils run the other way -- richest at high latitude.
SALT_MARSH_FALLBACK_SOC_POLEWARD_MG_HA = 70.0
SALT_MARSH_FALLBACK_SOC_BANDS_MG_HA = ((45.0, 50.0), (35.0, 44.0), (20.0, 36.0))


def calculate_salt_marsh_density_array(latitude_arr, soc_arr=None):
    """
    Vectorised salt marsh total carbon density (Mg C/ha).

    AGB: Chmura et al. 2003 median (5 t/ha) with 1.2x tropical boost (|lat| < 25).
    BGB: AGB x 2.5 (extensive root systems).
    SOC: from soc_arr (Maxwell 2024 MarSOC raster) when provided; otherwise a
         latitude step function, 36 up to 70 Mg C/ha by zone -- the 180-to-350
         Mg C/ha profile for the top 1 m, scaled to the top 20 cm.
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
        soc = soc_arr.astype(np.float64) * SOC_1M_TO_20CM_SCALAR
    else:
        soc = np.full_like(lat_abs, SALT_MARSH_FALLBACK_SOC_POLEWARD_MG_HA, dtype=np.float64)
        for lat_below, soc_mg_per_ha in SALT_MARSH_FALLBACK_SOC_BANDS_MG_HA:
            soc = np.where(lat_abs < lat_below, soc_mg_per_ha, soc)

    return {
        'agb_c_mg_per_ha':   agb_c,
        'bgb_c_mg_per_ha':   bgb_c,
        'soil_c_mg_per_ha':  soc,
        'total_c_mg_per_ha': agb_c + bgb_c + soc,
    }


# Seagrass (Gomis et al. 2025 genus-specific biomass + Fourqurean 2012 soil scaled to 1 m)
SEAGRASS_GENUS_TOTAL_BIOMASS_C_MG_HA = {
    # Persistent strategy
    'Posidonia':       9.78,
    'Phyllospadix':    3.70,
    'Enhalus':         3.70,
    'Cymodocea':       3.70,
    'Thalassia':       3.70,
    'Amphibolis':      3.70,
    'Thalassodendron': 3.70,
    # Opportunistic
    'Zostera':         0.94,
    'Syringodium':     0.94,
    'Halodule':        0.94,
    # Colonising
    'Ruppia':          0.33,
    'Halophila':       0.19,
}
SEAGRASS_DEFAULT_BIOMASS_C_MG_HA = 1.55  # Gomis 2025 global mean
SEAGRASS_BGB_FRACTION = 0.70
SEAGRASS_AGB_FRACTION = 0.30
SEAGRASS_SOIL_C_MG_HA_20CM = 22.0  # Fourqurean 2012 global mean for top 20 cm
SEAGRASS_NON_MARINE_GENERA = frozenset({
    'Valisneria', 'Trapa', 'Myriophyllum', 'Najas', 'Heteranthera',
    'Stuckenia', 'Utricularia',
})


def _normalize_seagrass_genus(genus_str):
    if not isinstance(genus_str, str):
        return None
    primary = genus_str.strip().split('|')[0].strip()
    if not primary or primary.lower() == 'not reported':
        return None
    return primary.capitalize()


def _seagrass_biomass_for_genus(genus_str):
    """Mg C/ha total biomass for a genus, or None if non-marine."""
    if isinstance(genus_str, str) and genus_str.strip().lower() == 'not reported':
        return SEAGRASS_DEFAULT_BIOMASS_C_MG_HA
    g = _normalize_seagrass_genus(genus_str)
    if g is None:
        return SEAGRASS_DEFAULT_BIOMASS_C_MG_HA
    if g in SEAGRASS_NON_MARINE_GENERA:
        return None
    return SEAGRASS_GENUS_TOTAL_BIOMASS_C_MG_HA.get(g, SEAGRASS_DEFAULT_BIOMASS_C_MG_HA)


def calculate_seagrass_pool_densities_array(genus_array):
    """
    Vectorised per-pool seagrass densities (Mg C/ha) for an array of genus strings.
    Non-marine genera (freshwater plants) are zeroed out across all pools so the
    caller can join by row index without dropping records. Soil density is a
    constant (Fourqurean 2012 global mean for top 20 cm).
    """
    n = len(genus_array)
    agb = np.zeros(n, dtype=np.float64)
    bgb = np.zeros(n, dtype=np.float64)
    soil = np.zeros(n, dtype=np.float64)
    for i, g in enumerate(genus_array):
        biomass = _seagrass_biomass_for_genus(g)
        if biomass is None:
            continue
        agb[i] = biomass * SEAGRASS_AGB_FRACTION
        bgb[i] = biomass * SEAGRASS_BGB_FRACTION
        soil[i] = SEAGRASS_SOIL_C_MG_HA_20CM
    return {
        'agb_c_mg_per_ha':   agb,
        'bgb_c_mg_per_ha':   bgb,
        'soil_c_mg_per_ha':  soil,
        'total_c_mg_per_ha': agb + bgb + soil,
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


def seagrass_stock_by_region(gdf_pieces, id_col):
    """Per-region seagrass stock (Mg C): each polygon's genus density times its hectares.

    Seagrass is the one habitat whose density is an attribute of the polygon rather than of the
    pixel, so it is aggregated from the clipped polygons instead of a streamed raster pass.
    """
    densities = calculate_seagrass_pool_densities_array(gdf_pieces['GENUS'].values)
    area_ha = gdf_pieces['area_ha'].values.astype(np.float64)
    agb, bgb, soil, total = stock_columns('seagrass')
    per_polygon = pd.DataFrame({
        id_col: gdf_pieces[id_col].values,
        agb: densities['agb_c_mg_per_ha'] * area_ha,
        bgb: densities['bgb_c_mg_per_ha'] * area_ha,
        soil: densities['soil_c_mg_per_ha'] * area_ha,
        total: densities['total_c_mg_per_ha'] * area_ha,
    })
    return per_polygon.groupby(id_col)[[agb, bgb, soil, total]].sum().reset_index()


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
    the `_EEZ` rows is the service's open question, not a cleanup step: it is the difference
    between the EEZ-only total and the all-rows total, and the two are reported as such.
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

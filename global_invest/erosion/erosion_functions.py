"""Erosion-control ES science helpers (sediment-retention shock).

STATIC helper (read_erosion_dependency): parse the frozen per-scenario dependency table
(raw_dependencies/erosion_prevention_dependency.csv). Scenario-name resolution is shared across services
in global_invest.utilities.resolve_raw_scenario. DYNAMIC helpers (#26): the SPAM->coefficient
crosswalk (load_erosion_yield_coefficients, get_erosion_yield_coefficient, SPAM_ALIAS_MAP) and the per-country
severe-threshold policy (build_severe_threshold_raster) used by the dynamic exposure and shock tasks.
"""
import numpy as np
import pandas as pd


def read_erosion_dependency(ero_path):
    """Load + normalize the erosion dependency table; return the df.

    Base extraction happens in the CALLER after resolving the configured base scenario through
    utilities.resolve_base_scenario (this function previously hardcoded 'baseline_ignore_damages'
    as the base, silently ignoring p.es_shock_base_scenario -- right only by spelling coincidence).
    """
    df = pd.read_csv(ero_path)
    df['scenario'] = df['scenario'].str.replace('_2050', '').str.replace('2023.0', 'baseline_2023')
    return df


# ---------------------------------------------------------------------------
# DYNAMIC helpers (#26): SPAM crop -> erosion-to-yield coefficient crosswalk.
# The 4-letter SPAM band codes are aliased to the coefficient table's crop names.
# ---------------------------------------------------------------------------
# SPAM2020 crop code -> candidate keys in the crop-coefficient table, tried in order.
# The FIRST alias of each entry is the EXACT FAO item name as it appears in
# elasticity_crops_fao_revised.csv; the looser stems after it are kept as fallbacks for other tables.
# This matters: the lookup is exact-match, so stems alone ("maize") never hit the FAO names
# ("Maize (corn)") and every crop silently took the 0.08 fallback -- which is the table MINIMUM, not
# its average (mean 0.163), so the miss biased the erosion shock low across the board.
# Six SPAM codes have NO counterpart in the table and correctly keep the default: grou (groundnut),
# ocer, orts, pige, vege and rest -- the n.e.c. aggregates FAO does not carry.
SPAM_ALIAS_MAP = {
    "whea": ["wheat"], "rice": ["rice"], "maiz": ["maize (corn)", "maize", "corn"],
    "barl": ["barley"], "sorg": ["sorghum"], "mill": ["millet", "small millet"],
    "pmil": ["millet", "pearl millet"], "pota": ["potatoes", "potato"],
    "cass": ["cassava, fresh", "cassava"], "soyb": ["soya beans", "soybean", "soy"],
    "grou": ["groundnut", "peanut"], "cott": ["seed cotton, unginned", "cotton"],
    "sugc": ["sugar cane", "sugarcane"], "bana": ["bananas", "banana"],
    "plnt": ["plantains and cooking bananas", "plantain"], "coco": ["cocoa beans", "cocoa"],
    "coff": ["coffee, green", "arabica coffee", "coffee"], "rcof": ["coffee, green", "robusta coffee"],
    "teas": ["tea leaves", "tea"], "toba": ["unmanufactured tobacco", "tobacco"],
    "toma": ["tomatoes", "tomato"],
    "onio": ["onions and shallots, dry (excluding dehydrated)", "onion"],
    "vege": ["vegetable", "other vegetables"], "sunf": ["sunflower seed", "sunflower"],
    "rape": ["rape or colza seed", "rapeseed", "canola"], "sesa": ["sesame seed", "sesame"],
    "citr": ["oranges", "citrus"], "lent": ["lentils, dry", "lentil"],
    "bean": ["beans, dry", "bean"], "chic": ["chick peas, dry", "chickpea"],
    "cowp": ["cow peas, dry", "cowpea"], "pige": ["peas, dry", "pigeon pea"], "yams": ["yams"],
    "swpo": ["sweet potatoes", "sweet potato"], "sugb": ["sugar beet", "sugarbeet"],
    "oilp": ["oil palm fruit", "oilpalm", "oil palm"], "cnut": ["coconuts, in shell", "coconut"],
    "ocer": ["other cereals"], "orts": ["other roots"],
    "opul": ["other pulses n.e.c.", "other pulses"], "ooil": ["castor oil seeds", "other oil crops"],
    "ofib": ["agave fibres, raw, n.e.c.", "other fibre crops"],
    "rubb": ["natural rubber in primary forms", "rubber"],
    "trof": ["other tropical fruits, n.e.c.", "other tropical fruit"],
    "temf": ["apples", "temperate fruit"], "rest": ["rest of crops"],
}


def load_erosion_yield_coefficients(elasticity_csv):
    """Return {crop_key (lowercased) -> erosion-to-yield coefficient in [0,1]} from the coefficient CSV.

    Accepts a crop-name column among crop/monfreda_crop/item/item_name plus an 'elasticity' column.
    """
    df = pd.read_csv(elasticity_csv, encoding='utf-8-sig')
    df.columns = [str(c).strip().lower() for c in df.columns]
    crop_col = next((c for c in ('crop', 'monfreda_crop', 'item', 'item_name') if c in df.columns), None)
    if crop_col is None or 'elasticity' not in df.columns:
        return {}
    df['elasticity'] = pd.to_numeric(df['elasticity'], errors='coerce').clip(0.0, 1.0)
    key = df[crop_col].astype(str).str.strip().str.lower()
    keep = key.ne('') & df['elasticity'].notna()
    df = df[keep].assign(__k=key[keep]).drop_duplicates('__k', keep='last')
    return dict(zip(df['__k'], df['elasticity']))


def get_erosion_yield_coefficient(crop_key, coef_map, fallback=0.08):
    """crop_key -> erosion-to-yield coefficient: direct hit, else SPAM alias, else the flat fallback."""
    k = str(crop_key).strip().lower()
    v = coef_map.get(k, np.nan)
    if np.isfinite(v):
        return float(np.clip(v, 0.0, 1.0))
    for alias in SPAM_ALIAS_MAP.get(k, []):
        v2 = coef_map.get(str(alias).strip().lower(), np.nan)
        if np.isfinite(v2):
            return float(np.clip(v2, 0.0, 1.0))
    return float(np.clip(fallback, 0.0, 1.0))


def build_seals7_biophysical_table(src_csv, out_csv):
    """Re-key a biophysical table from ESA lucodes onto SEALS7 classes, for InVEST SDR.

    SDR matches the table's `lucode` against the LULC raster's values, but the shipped table is keyed on
    ESA-CCI codes while our maps are SEALS7 (1-7), so SDR would match nothing. The table already carries
    a `seals_lucode` column, and usle_c/usle_p are CONSTANT within each SEALS class (verified: min == max
    for all 7), so the collapse is unambiguous -- no area weighting to choose. Returns the written path.
    """
    df = pd.read_csv(src_csv)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if 'seals_lucode' not in df.columns:
        raise ValueError('%s has no seals_lucode column, so it cannot be re-keyed onto SEALS7 classes; '
                         'supply an already-SEALS-keyed table via p.erosion_biophysical_table_path.'
                         % src_csv)
    df = df.dropna(subset=['seals_lucode'])
    out = (df.groupby(df['seals_lucode'].astype(int))[['usle_c', 'usle_p']].mean()
             .reset_index().rename(columns={'seals_lucode': 'lucode'}))
    out['description'] = ['seals7_class_%d' % c for c in out['lucode']]
    out.to_csv(out_csv, index=False)
    return out_csv


def repair_watersheds(src_path, out_path):
    """Repair self-intersecting watershed rings so InVEST SDR can finish.

    SDR's last step (_generate_report) unions the watershed polygons to test for overlap, and GEOS
    RAISES TopologyException on an invalid ring rather than warning -- so a single bad geometry kills a
    run whose rasters are already computed. HydroBASINS reprojected to an equal-area CRS carries ring
    self-intersections (1192 of 16397 in hybas_global_lev06_v1c), which is why this never showed up on a
    clipped AOI: the small subset happened to exclude them.

    make_valid (not buffer(0), which can silently drop slivers) clears all of them, leaves the union
    computable, and preserves total area. Returns out_path.
    """
    import geopandas as gpd

    gdf = gpd.read_file(src_path, engine='pyogrio')
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, 'geometry'] = gdf.loc[invalid, 'geometry'].make_valid()
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf.to_file(out_path, driver='GPKG')
    print('  erosion watersheds: repaired %d of %d invalid geometries -> %s'
          % (int(invalid.sum()), len(gdf), out_path))
    return out_path


def build_severe_threshold_raster(grid_da, country_boundary_path, dem_path=None,
                                  thresh_high=11.0, thresh_low=2.0,
                                  small_area_km2=50_000, low_elevation_mean_m=250,
                                  mask_below_sea=True, max_valid_elevation_m=9000.0):
    """Per-pixel soil-loss-tolerance threshold aligned to grid_da (the SES-11 policy).

    T = thresh_low for a country that is small-area (geometry area < small_area_km2) OR low-elevation
    (mean DEM elevation < low_elevation_mean_m); else thresh_high. grid_da must be on an equal-area
    grid (its CRS units are used for the km2 area). If dem_path is None the elevation rule is skipped.
    Returns a float32 array of shape grid_da.shape.
    """
    import geopandas as gpd
    import rioxarray as rxr
    from rasterio.features import rasterize
    from rasterio.enums import Resampling

    gdf = gpd.read_file(country_boundary_path, engine='pyogrio').to_crs(grid_da.rio.crs)
    gdf = gdf[gdf.geometry.notnull()].reset_index(drop=True)
    gdf['iid'] = range(1, len(gdf) + 1)
    shape, transform = grid_da.shape, grid_da.rio.transform()
    iso_id = rasterize([(g, int(i)) for g, i in zip(gdf.geometry, gdf['iid'])],
                       out_shape=shape, transform=transform, fill=0, dtype='int32')
    max_id = int(gdf['iid'].max())

    area_km2 = gdf.set_index('iid').geometry.area / 1e6      # equal-area CRS -> m2 -> km2
    iso_low = set(int(i) for i in area_km2[area_km2 < small_area_km2].index)

    if dem_path:
        dem = rxr.open_rasterio(dem_path, masked=True).squeeze().rio.reproject_match(
            grid_da, resampling=Resampling.bilinear)
        v = dem.values.astype('float64')
        if mask_below_sea:
            v[v < 0.0] = np.nan
        v[v > max_valid_elevation_m] = np.nan
        m = np.isfinite(v) & (iso_id > 0)
        s = np.bincount(iso_id[m], weights=v[m], minlength=max_id + 1).astype('float64')
        c = np.bincount(iso_id[m], minlength=max_id + 1).astype('float64')
        with np.errstate(invalid='ignore'):
            mean_elev = np.where(c > 0, s / c, np.nan)
        iso_low |= set(int(i) for i in range(1, max_id + 1)
                       if np.isfinite(mean_elev[i]) and mean_elev[i] < low_elevation_mean_m)

    thr = np.full(shape, float(thresh_high), dtype='float32')
    if iso_low:
        thr[np.isin(iso_id, np.fromiter(iso_low, dtype='int32'))] = float(thresh_low)
    return thr

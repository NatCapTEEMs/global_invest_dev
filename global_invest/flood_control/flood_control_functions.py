"""Flood-control (river flood mitigation) science: expected avoided damage per country.

The consortium drive's FINAL_GEP_FLOOD tree carries the full December 2025 pipeline (flood
hazard x JRC depth-damage x protection levels -> expected annual damage with and without
ecosystems) plus its committed global outputs and extensive method notes. The upstream
geospatial pipeline is taken as given. This file holds the read of the committed per-country
expected avoided damage (2019 USD), whose sum reproduces the pipeline's own global summary
exactly ($110.59bn, 148 countries non-zero of 236 assessed). That is the anchor rather than the
account's number: flood_control_functions recomputes the damage from the depth, damage-curve and
asset inputs, and that recompute is what the account reports ($112.78bn, matching the committed
table for 146 of the 148).

The FINAL_GEP_FLOOD pipeline's per-step science, ported as pure functions.

Source: base_data/global_invest/flood_regulation/input/code_dec_18_2025/ (the drive's
December 2025 code tree). The calculation, in the order the orchestrating notebook
(gep_running_file_dec_14_final.ipynb) runs it:

  step 2A  sda_step_2A_make_lulc_to_sda_mapping_esa300.py
           ESA-CCI 300m codes -> SDA mapping JSON (artif=190; crop=10/11/12/20/30/40;
           pasture=130; everything else non-SDA).
  step 2   sda_step2_build_sda_global.py / build_sda_from_esa300m.py
           SDA class rasters: economic-asset LULC codes, per-country intersected with
           the floodplain (depth > 0.1 m) per return period, or global without the
           floodplain intersect. Here: build_sda_class, build_depthbin_index.
  step 3   serviceflow_step3_spa_to_sda_ratio_global.py
           service_flow_frac = upstream SPA ratio on the SDA grid, zero outside SDA.
           Here: service_flow_from_spa. NOTE: the staged monetary pipeline (4B-4D) never
           multiplies by this fraction; the drive's global_upstream_spa_ratio.tif is a
           byte copy of the binary global_prr_spa.tif (notebook cell 6).
  step 4A  build_damage_table_USD2019.py
           JRC fractional depth-damage curves x per-country max damages x iso3->JRC-region
           map -> EUR2010/m2 depth-damage table, x 1.538344 -> USD2019, aggregated to SDA
           types. Here: build_canonical_damage_table, convert_damage_table_to_usd2019,
           aggregate_damage_table_to_sda, pivot_damage_table_wide.
  step 4B  flood_gep_step4b_pixel_damage_USD2019.py
           Per ISO3 x RP: pixel damage = curve(iso3, sda_type) interpolated at pixel
           depth x pixel area. Here: damage_curves_from_wide, interp_damage_per_m2,
           pixel_damage_totals. The final run used RPs 10, 20, 50, 500.
  step 4C  flood_gep_step4c_ead_USD2019_global.py
           EAD = trapezoid integral of D(p) over p = 1/RP in [0, 1], anchor (p=1, D=0),
           flat tail to p=0. Here: ead_from_rp_damages, ead_from_rp_stack.
  step 4D  flood_gep_step4d_export_global_USD2019.py + post_processing_code/
           Per-country EADs -> global country table (+ zero-fill and status), whose
           values ARE the committed country_avoided_damage_usd2019.csv (verified equal
           row-by-row; SDS remapped to the r250 label SSD). The "avoided damage" label
           is the analysis notebook's interpretation of the step-4D EAD column.
           Here: country_ead_table.

Raster inputs the drive names but we have NOT staged (paths as the scripts spell them,
on the author's MSI cluster under /users/3/damph002/GEP/flood/inputs/):
  lulc/lulc_esa_2019_int_reproj.tif                              (ESA-CCI 300m, 2019)
  floodplain_depth/aligned_to_lulc/JRC_flood_depth_rp{RP}y__matchLULC.tif
  sda/sda_esa300m_artif_crop_pasture.tif                         (regenerable from LULC)
  country_vector/country_boundary_r250_with_iso3.gpkg
  global_spa_ben/global_prr_spa.tif
  flood_damage/jrc_fractional_curves_long.csv
  flood_damage/jrc_max_damage_values_long_with_iso3.csv
  flood_damage/iso3_to_jrc_region.csv
The functions below take arrays/DataFrames, so they run as soon as those are staged.
"""
import numpy as np
import pandas as pd


def flood_control_gep_by_country(avoided_damage_df, countries_df):
    """The committed avoided-damage table joined onto the r250 country list, one row per
    country. Zeroes are the pipeline's own zeroes (assessed, no avoided damage); countries it
    never assessed stay NaN."""
    df = countries_df.merge(
        avoided_damage_df[['iso3_r250_label', 'avoided_damage_usd2019']].rename(
            columns={'avoided_damage_usd2019': 'flood_control_gep'}),
        on='iso3_r250_label', how='left')
    return df


# EUR2010 -> USD2019, the pipeline's committed factors (FRED DEXUSEU 2010 annual average;
# FRED GDPDEF 2019/2010 ratio; currency_conversion_factors_EUR2010_to_USD2019.ipynb).
FX_USD_PER_EUR_2010_AVG = 1.32609
INFLATOR_US_2010_TO_2019 = 1.160061
EUR2010_TO_USD2019 = FX_USD_PER_EUR_2010_AVG * INFLATOR_US_2010_TO_2019

# ESA-CCI code -> SDA membership (step 2A's JRC-style mapping). Codes not listed are
# non-SDA; the step-2A "ignore" list never overlaps the asset sets, so it is not needed
# to reproduce the class raster.
ESA_TO_SDA = {
    'artif': [190],
    'crop': [10, 11, 12, 20, 30, 40],
    'pasture': [130],
}

SDA_CODE = {'none': 0, 'artif': 1, 'crop': 2, 'pasture': 3, 'roads': 4}
SDA_CODE_TO_TYPE = {1: 'artif', 2: 'crop', 3: 'pasture'}

# Depth bins (m) of the JRC curves; the wide damage tables carry them as 0m, 0.5m, ... 6m.
DEPTH_BINS_M = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6]

# JRC landtype -> SDA type: cropland assets to crop; built/asset surfaces to artif.
LANDTYPE_TO_SDA = {
    'Agriculture': 'crop',
    'Residential buildings': 'artif',
    'Commercial buildings': 'artif',
    'Industrial buildings': 'artif',
    'Transport': 'artif',
    'Infrastructure - roads': 'artif',
}

_LANDTYPE_CANONICAL = {
    'residential': 'Residential buildings',
    'commercial': 'Commercial buildings',
    'industrial': 'Industrial buildings',
    'roads': 'Infrastructure - roads',
    'road': 'Infrastructure - roads',
    'infrastructure roads': 'Infrastructure - roads',
    'infrastructure - road': 'Infrastructure - roads',
    'transport': 'Transport',
    'transportation': 'Transport',
    'agriculture': 'Agriculture',
}


def normalize_landtype_label(landtype):
    """Map a short JRC landtype spelling ('Residential') to the fraction-curve label
    ('Residential buildings'); already-canonical labels pass through unchanged."""
    return _LANDTYPE_CANONICAL.get(str(landtype).strip().lower(), str(landtype).strip())


def fmt_depth_col(depth_m):
    """Numeric depth -> the wide-table column label: 1 -> '1m', 0.5 -> '0.5m'."""
    d = float(depth_m)
    return f'{int(d)}m' if abs(d - int(d)) < 1e-9 else f'{d}m'


def build_canonical_damage_table(fractional_df, max_damage_df, iso3_region_df):
    """Step 4A: the canonical EUR2010/m2 depth-damage table, long form.

    Args:
        fractional_df: JRC fractional curves [JRC_Region, LandType, depth_m, fraction]
            (jrc_fractional_curves_long.csv).
        max_damage_df: per-country maxima [iso3, LandType, max_damage_eur_m2]
            (jrc_max_damage_values_long_with_iso3.csv).
        iso3_region_df: [iso3, JRC_Region] (iso3_to_jrc_region.csv).

    Returns:
        DataFrame [iso3, landtype, depth_m, damage_per_m2] with damage_per_m2 =
        max_damage_eur_m2 x fraction. Countries without a JRC region and landtype/region
        combinations without a fraction curve drop out, as in the source script.
    """
    frac = fractional_df.copy()
    frac['LandType'] = frac['LandType'].map(normalize_landtype_label)
    maxd = max_damage_df.copy()
    maxd['LandType'] = maxd['LandType'].map(normalize_landtype_label)

    out = (maxd.merge(iso3_region_df, on='iso3', how='left')
               .dropna(subset=['JRC_Region'])
               .merge(frac, on=['JRC_Region', 'LandType'], how='left'))
    out['damage_per_m2'] = out['max_damage_eur_m2'] * out['fraction']
    out = out.rename(columns={'LandType': 'landtype'})
    return out.dropna(subset=['depth_m', 'damage_per_m2'])[
        ['iso3', 'landtype', 'depth_m', 'damage_per_m2']].reset_index(drop=True)


def convert_damage_table_to_usd2019(eur_long, combined_multiplier=EUR2010_TO_USD2019):
    """Step 4A: EUR2010/m2 -> USD2019/m2 (one multiplier for every row) + currency column."""
    out = eur_long.copy()
    out['damage_per_m2'] = out['damage_per_m2'] * float(combined_multiplier)
    out['currency'] = 'USD2019'
    return out


def aggregate_damage_table_to_sda(sector_long, set_pasture_equal_crop=True):
    """Step 4A: sector landtypes -> SDA types for the step-4B lookup.

    Landtypes map through LANDTYPE_TO_SDA (unmapped ones drop out of the SDA table; they
    stay in the sector outputs upstream). Because five landtypes map to artif, the wide
    pivot's mean makes the artif curve the average of the built-asset curves. The
    pipeline ran with pasture set equal to crop.

    Returns:
        DataFrame [iso3, sda_type, depth_m, damage_per_m2(, currency)].
    """
    out = sector_long.copy()
    out['sda_type'] = out['landtype'].map(lambda lt: LANDTYPE_TO_SDA.get(normalize_landtype_label(lt)))
    out = out.dropna(subset=['sda_type']).drop(columns=['landtype'])
    if set_pasture_equal_crop:
        pasture = out[out['sda_type'] == 'crop'].copy()
        pasture['sda_type'] = 'pasture'
        out = pd.concat([out, pasture], ignore_index=True)
    cols = ['iso3', 'sda_type', 'depth_m', 'damage_per_m2']
    if 'currency' in out.columns:
        cols.append('currency')
    return out[cols].reset_index(drop=True)


def pivot_damage_table_wide(long_df, group_cols):
    """Step 4A: long depth-damage rows -> wide depth columns ('0m', '0.5m', ...).

    Duplicate (group, depth) rows average (this is what folds the five built-asset
    landtypes into one artif curve); missing depths fill with 0.
    """
    tmp = long_df.dropna(subset=['depth_m']).copy()
    tmp['depth_col'] = tmp['depth_m'].map(fmt_depth_col)
    wide = tmp.pivot_table(index=group_cols, columns='depth_col', values='damage_per_m2',
                           aggfunc='mean', fill_value=0.0).reset_index()
    wide.columns = [str(c) for c in wide.columns]
    return wide


def build_sda_class(lulc, mapping=None, floodplain=None, roads=None, nodata=None,
                    include_pasture=True):
    """Step 2: LULC codes -> SDA class array (uint8).

    Codes: 0 none, 1 artif, 2 crop, 3 pasture, 4 roads. Written in priority order
    pasture -> crop -> artif -> roads, so a code in several sets keeps the highest
    priority and roads override everything (as in the source scripts).

    Args:
        lulc: 2D integer array of ESA-CCI codes. The drive's raster is
            lulc/lulc_esa_2019_int_reproj.tif (not staged).
        mapping: {'artif': [...], 'crop': [...], 'pasture': [...]}; default ESA_TO_SDA.
        floodplain: optional boolean array (depth > threshold); None reproduces the
            global SDA raster (no floodplain intersect), as build_sda_from_esa300m does.
        roads: optional binary road-mask array; roads == 1 inside the floodplain
            become class 4.
        nodata: LULC nodata value (255 for the drive's raster).
        include_pasture: the pipeline's step-2 run left pasture OFF (no
            --include-pasture); the global step-4B raster has it ON.
    """
    mapping = ESA_TO_SDA if mapping is None else mapping
    valid = np.ones(lulc.shape, dtype=bool) if nodata is None else (lulc != nodata)
    base = valid if floodplain is None else (valid & floodplain)

    sda = np.zeros(lulc.shape, dtype=np.uint8)
    if include_pasture and mapping.get('pasture'):
        sda[base & np.isin(lulc, mapping['pasture'])] = SDA_CODE['pasture']
    if mapping.get('crop'):
        sda[base & np.isin(lulc, mapping['crop'])] = SDA_CODE['crop']
    if mapping.get('artif'):
        sda[base & np.isin(lulc, mapping['artif'])] = SDA_CODE['artif']
    if roads is not None:
        road_base = (roads == 1) if floodplain is None else ((roads == 1) & floodplain)
        sda[road_base] = SDA_CODE['roads']
    return sda


def build_depthbin_index(depth_m, nodata_mask, max_depth=6.0):
    """Step 2: depth (m) -> 0.5 m bin index (int16) for later monetary joins.

    Bin edges are 0, 0.5, ..., max_depth; negatives clamp to 0, values above max_depth
    to the last bin, nodata pixels to -1.
    """
    edges = np.arange(0, max_depth + 0.5, 0.5)
    d = depth_m.astype('float32').copy()
    d[~np.isfinite(d)] = np.nan
    d[(d < 0) & np.isfinite(d)] = 0.0
    d[(d > max_depth) & np.isfinite(d)] = max_depth
    idx = np.searchsorted(edges, d, side='left').astype(np.int16)
    idx[nodata_mask] = -1
    return idx


def service_flow_from_spa(spa_ratio, sda_mask, nodata=None):
    """Step 3: the service-flow fraction on the SDA grid.

    The upstream SPA ratio (the drive's global_spa_ben/global_upstream_spa_ratio.tif,
    reprojected onto the SDA grid with average resampling before this function) is
    clipped to [0, 1] and zeroed outside the SDA mask -- zero, not NaN, so the monetary
    step can multiply directly.
    """
    ratio = spa_ratio.astype(np.float32, copy=True)
    if nodata is not None:
        ratio = np.where(ratio == nodata, np.nan, ratio)
    ratio = np.clip(ratio, 0.0, 1.0)
    return np.where(sda_mask, ratio, np.float32(0.0))


def damage_curves_from_wide(wide_df, depth_bins_m=DEPTH_BINS_M):
    """Step 4B: the wide SDA damage table -> {(iso3, sda_type): (depths, damages)} arrays.

    wide_df is damage_functions_sda_depth_USD2019_wide.csv: columns iso3, sda_type and
    one column per depth bin ('0m', '0.5m', ..., '6m').
    """
    xs = np.array(depth_bins_m, dtype='float32')
    depth_cols = [fmt_depth_col(d) for d in depth_bins_m]
    curves = {}
    for _, r in wide_df.iterrows():
        ys = np.array([float(r[c]) for c in depth_cols], dtype='float32')
        curves[(str(r['iso3']).strip(), str(r['sda_type']).strip())] = (xs, ys)
    return curves


def interp_damage_per_m2(depth_m, xs, ys):
    """Step 4B: damage (USD2019/m2) at each depth by linear interpolation, depths
    clamped to the curve's [min, max]."""
    d = np.clip(depth_m, xs.min(), xs.max())
    return np.interp(d, xs, ys).astype('float32')


def pixel_damage_totals(depth, sda, curves, iso3, pixel_area_m2, depth_nodata=None):
    """Step 4B: per-pixel damages for one country x return period.

    Only pixels with finite depth > 0 count; each SDA type looks up its (iso3, sda_type)
    curve; damage = interpolated USD2019/m2 x pixel area. An SDA type without a curve
    contributes zero (the source script's behavior when 4A did not create it). The
    depth and SDA arrays come from the drive's aligned rasters (JRC_flood_depth_rp{RP}y_
    _matchLULC.tif and sda_esa300m_artif_crop_pasture.tif, not staged), clipped to the
    country geometry.

    Returns:
        (damage_raster, totals_by_sda_type, total): float32 USD2019-per-pixel array,
        {sda_type: total}, and the all-type total.
    """
    depth = depth.astype('float32')
    valid = np.isfinite(depth)
    if depth_nodata is not None:
        valid &= (depth != depth_nodata)
    valid &= (depth > 0)

    damage_raster = np.zeros_like(depth, dtype='float32')
    totals = {t: 0.0 for t in set(SDA_CODE_TO_TYPE.values())}
    for code, sda_type in SDA_CODE_TO_TYPE.items():
        m = valid & (sda == code)
        if not np.any(m) or (iso3, sda_type) not in curves:
            continue
        xs, ys = curves[(iso3, sda_type)]
        dmg = interp_damage_per_m2(depth[m], xs, ys) * pixel_area_m2
        totals[sda_type] += float(dmg.sum())
        damage_raster[m] = dmg
    return damage_raster, totals, float(sum(totals.values()))


def ead_from_rp_damages(rp, damage, add_p1_zero=True, tail_mode='flat',
                        enforce_monotone=False):
    """Step 4C: expected annual damage from (return period, damage) points.

    EAD = trapezoid integral of D(p) over p = 1/RP in [0, 1]. Duplicate RPs average;
    duplicate p values keep the max damage. The pipeline's defaults: an anchor
    (p=1, D=0) -- the depth threshold + SDA masking treat frequent shallow events as
    zero damage -- and a flat tail holding D(RPmax) from p = 1/RPmax down to p = 0
    ('zero' sets the tail to 0 instead).

    Returns:
        (ead, points_df): the EAD and the integration points [p, rp, damage].
    """
    if tail_mode not in {'flat', 'zero'}:
        raise ValueError("tail_mode must be 'flat' or 'zero'")
    df = pd.DataFrame({'rp': np.asarray(rp, dtype=float),
                       'damage': np.asarray(damage, dtype=float)})
    df = df.dropna()
    df = df[(df['rp'] > 0) & (df['damage'] >= 0)]
    if df.empty:
        return 0.0, pd.DataFrame(columns=['p', 'rp', 'damage'])

    df = df.groupby('rp', as_index=False)['damage'].mean().sort_values('rp')
    if enforce_monotone:
        df['damage'] = np.maximum.accumulate(df['damage'].to_numpy())
    df['p'] = 1.0 / df['rp']

    pts = df[['p', 'rp', 'damage']]
    if add_p1_zero:
        pts = pd.concat([pd.DataFrame({'p': [1.0], 'rp': [1.0], 'damage': [0.0]}), pts],
                        ignore_index=True)
    d_tail = float(df['damage'].iloc[-1]) if tail_mode == 'flat' else 0.0
    pts = pd.concat([pts, pd.DataFrame({'p': [0.0], 'rp': [np.inf], 'damage': [d_tail]})],
                    ignore_index=True)

    grid = pts.groupby('p', as_index=False)['damage'].max().sort_values('p')
    ead = float(np.trapezoid(grid['damage'].to_numpy(), grid['p'].to_numpy()))
    return ead, pts.sort_values('p', ascending=False).reset_index(drop=True)


def ead_from_rp_stack(damage_stack, rps, add_p1_zero=True, tail_mode='flat'):
    """Step 4D (raster option): per-pixel EAD from a stack of per-RP damage arrays.

    Same integral and boundary treatment as ead_from_rp_damages, vectorized over pixels.
    damage_stack is (n_rp, ...) aligned arrays ordered like rps; nodata must already be
    zeroed (the source treats nodata as zero damage, conservatively).
    """
    if tail_mode not in {'flat', 'zero'}:
        raise ValueError("tail_mode must be 'flat' or 'zero'")
    order = np.argsort(rps)
    y = np.asarray(damage_stack, dtype='float64')[order]
    p = 1.0 / np.asarray(rps, dtype='float64')[order]

    if add_p1_zero:
        p = np.concatenate([[1.0], p])
        y = np.concatenate([np.zeros_like(y[:1]), y], axis=0)
    p = np.concatenate([p, [0.0]])
    tail = y[-1:] if tail_mode == 'flat' else np.zeros_like(y[-1:])
    y = np.concatenate([y, tail], axis=0)

    return np.trapezoid(y[::-1], p[::-1], axis=0).astype('float32')


def country_ead_table(country_df, admin0_df, fill_missing_zero=True):
    """Step 4D: per-country EADs -> the global country table.

    Args:
        country_df: [iso3, ead_usd2019], one row per assessed country (from the
            per-country step-4C files).
        admin0_df: the Admin0 attribute table [iso3, ...enrichment columns...]
            (country_boundary_r250_with_iso3.gpkg, not staged).
        fill_missing_zero: keep every Admin0 iso3; countries without a step-4C result
            get ead_usd2019 = 0 and status 'missing_step4c' (assessed countries 'ok').
            This status is what the committed table's zero/NaN distinction rests on.

    Returns:
        DataFrame [iso3, <admin0 columns>, ead_usd2019, status] sorted by iso3.
    """
    country = country_df.drop_duplicates(subset=['iso3'], keep='last').copy()
    if fill_missing_zero:
        out = admin0_df.merge(country, on='iso3', how='left')
        out['status'] = np.where(out['ead_usd2019'].isna(), 'missing_step4c', 'ok')
        out['ead_usd2019'] = out['ead_usd2019'].fillna(0.0)
    else:
        out = country.merge(admin0_df, on='iso3', how='left')
        out['status'] = 'ok'
    cols = ['iso3'] + [c for c in admin0_df.columns if c != 'iso3'] + ['ead_usd2019', 'status']
    return out[cols].sort_values('iso3').reset_index(drop=True)

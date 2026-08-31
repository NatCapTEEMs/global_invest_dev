"""Erosion-control ES-shock tasks (static + dynamic), on the add_<es>_tasks seam like carbon/pollination.

METHOD SOURCE: the prevention-share erosion valuation in `global_erosion_gep`. That is where the
science comes from -- the on-farm share AE/(AE+USLE), the upstream share, the union
1 - (1-onfarm)(1-upstream) that avoids double-counting, the restriction to cropland AND
severely-eroding pixels, the per-country SES-11 soil-loss-tolerance policy, and the aggregation
weighted by crop production with per-crop erosion-yield elasticities. The biophysics below that is
InVEST SDR plus pygeoprocessing's D8 routing.

What differs here: that version runs one static 2019 map, so this recomputes the whole calculation PER
SCENARIO AND YEAR on the SEALS maps and aggregates to r50xAEZ rather than to countries -- and, because
scenarios exist here and not there, holds the severe-pixel set FIXED to the base scenario (a set that
moves between scenarios would make part of the shock a change in WHICH pixels are averaged rather than
a change in protection; see level_service_threshold).

STATIC (erosion_shock_static): read raw_dependencies/erosion_prevention_dependency.csv, subtract
the baseline reference, linearly ramp 0 -> the scenario value over the horizon, apply to the 8
erosion-affected crop sectors -> erosion_interpolated.csv. UNCAPPED here -- the cap is applied
later on the COMBINED value in build_combined_afeall_cc_es.

DYNAMIC (#26; erosion_sdr -> upstream -> exposure -> shock): recompute the shock from our SEALS
maps via InVEST SDR -> D8 upstream -> prevention shares -> per-zone crop-productivity shock, by THREE
methods reported side by side (A = 'damage', thresholded/area; B = 'service', threshold-free and
magnitude-weighted with a per-crop coefficient; B-thresholded = 'service_threshold', B restricted to a
FIXED severe-pixel set and the DEFAULT; see erosion_shock). add_erosion_tasks (erosion_initialize) dispatches static vs dynamic on p.dynamic_es.
"""
from __future__ import annotations
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import time
import json
import logging
import warnings
from datetime import datetime
import geopandas as gpd
import rasterio
import rasterio.features
import rioxarray as rxr
import xarray as xr
from rasterio.enums import Resampling
from rasterio.crs import CRS as rioCRS
from osgeo import gdal
import pandas as pd
import hazelbean as hb

from global_invest import utilities
from global_invest.erosion import erosion_functions as ef

# 8 crop sectors whose productivity depends on erosion control (sediment retention).
# SPAM2020 crop code -> GTAP crop sector. Lets method B report a DIFFERENT shock per sector instead of
# one zone number copied across all eight: each sector aggregates only its own crops, so a zone that
# erodes mainly under cereals sends that signal to GRO rather than to V_F.
#
# Built from the GTAP sector definitions in base_data/gtappy/aggregation_mappings/: PDR "Paddy rice",
# WHT "Wheat", GRO "Cereal grains nec", V_F "Vegetables, fruit, nuts", OSD "Oil seeds", C_B "Sugar cane,
# sugar beet", PFB "Plant-based fibers", OCR "Crops nec". These eight are IDENTICAL across the s65, c26
# and a24 aggregations (verified in gtapv7_s65_correspondence.csv, which carries all three label
# columns), so this map holds for our v12_s26_r50 runs and would survive a change of aggregation.
# CPC separates starchy roots and tubers (015) and pulses (017) from vegetables (012) and fruit (013),
# while GTAP's V_F names only the latter two, so those groups need placing: roots and tubers
# (cass/pota/swpo/yams/orts) go to V_F, pulses (bean/lent/cowp/pige/chic/opul) to OCR. This shifts
# shock between V_F and OCR only; every crop lands in some sector either way, so no total changes.
# Crops absent fall back to OCR. The map itself is the erosion_crop_to_sector row in
# es_parameters (shipped default; a consumer overrides p.erosion_crop_to_sector).
# SEALS7 cropland class, the cropland definition method A weights by.
CROPLAND_SEALS7_CLASS = 2
# The erosion -> yield bridge: the fraction of yield lost per unit of erosion exposure. Converting
# biophysical erosion into an economic productivity shock requires such a coefficient, so all three
# methods rest on one. Method A applies this flat value to its thresholded area share. The service
# methods read a
# per-crop coefficient from elasticity_crops_fao_revised.csv (see alpha_for in erosion_shock) and
# falls back here only when neither the crop nor its sector has a value.
# The SES-11 threshold policy and analysis frame (SES-11 = the erosion author's run-series tag;
# the 11 is the 11 t/ha/yr severe threshold, a standard tolerable-soil-loss benchmark -- the
# expansion of 'SES' is the author's naming, to confirm at submission): METHOD CONSTANTS defining
# the published science
# (provisional, the erosion author's to bless) -- in code so a change costs a reviewed commit, not
# an input/-copy edit. getattr hooks below allow a deliberate consumer override.


# ---------------------------------------------------------------------------
# DYNAMIC path (#26): recompute the shock from our SEALS maps instead of the frozen table. Reached only
# when 'erosion' is in p.dynamic_es. Heavy imports (natcap.invest.sdr, pygeoprocessing.routing) go
# inside the functions so module import stays light.
# ---------------------------------------------------------------------------




# =============================================================================
# SDR RUN
# =============================================================================
def run_invest_sdr(paths):
    _setup_sdr_environment()
    ef._assert_exists(paths.input.biophysical_table, "biophysical_table_path")
    ef._assert_exists(paths.input.dem, "dem_path")
    ef._assert_exists(paths.input.lulc, "lulc_path")
    ef._assert_exists(paths.input.erodibility, "erodibility_path")
    ef._assert_exists(paths.input.erosivity, "erosivity_path")
    ef._assert_exists(paths.input.watersheds, "watersheds_path (MERGED raw)")

    dem_wkt = _dem_wkt(paths.input.dem)

    hb.log("[prep] Sanitizing MERGED watersheds for SDR report/zonal stats …")
    ws_sanitized = sanitize_watersheds_for_report(
        watersheds_in=paths.input.watersheds,
        watersheds_out=paths.output.watersheds_sanitized,
        target_wkt=dem_wkt,
        layer=p.erosion_watersheds_sanitized_layer,
    )
    hb.log("[prep] Watersheds sanitized:", ws_sanitized)

    args = ef.build_args(p, ws_sanitized)

    hb.log("\n" + "=" * 78)
    hb.log("[run] Starting InVEST SDR …")
    hb.log("[run] workspace_dir :", args["workspace_dir"])
    hb.log("[run] results_suffix:", args["results_suffix"])
    hb.log("[run] flow_dir_algorithm:", args["flow_dir_algorithm"])
    hb.log("[run] threshold_flow_accumulation:", args["threshold_flow_accumulation"])
    hb.log("[run] params: k_param=%.3f, sdr_max=%.3f, ic_0_param=%.3f, l_max=%.3f"
          % (args["k_param"], args["sdr_max"], args["ic_0_param"], args["l_max"]))
    hb.log("[run] n_workers:", args.get("n_workers", "(not set)"))
    hb.log("=" * 78 + "\n")

    import natcap.invest.sdr.sdr
    file_registry = natcap.invest.sdr.sdr.execute(args)

    hb.log("\n[done] InVEST SDR finished.")
    hb.log("[done] Results in:", args["workspace_dir"])
    hb.log("[done] MERGED watersheds used (raw):", paths.input.watersheds)
    hb.log("[done] Sanitized watersheds used   :", ws_sanitized)
    return args, file_registry


def compute_country_mean_elevation(
    usle_like_da: xr.DataArray,
    iso_id_raster: np.ndarray,
    iso_lut: pd.DataFrame,
    elev_path: str | None
) -> dict[int, float]:
    if elev_path is None or not hb.path_exists(elev_path):
        warnings.warn("No DEM provided — elevation rule will be skipped.")
        return {}

    dem_native = open_raster_1band(elev_path)
    ef._ensure_crs(dem_native, "DEM")
    dem = dem_native.rio.reproject_match(usle_like_da, resampling=ef.RESAMPLE_DEM)

    vals = dem.values.astype("float64", copy=True)
    ids  = iso_id_raster.astype("int32", copy=False)

    nodata = dem.rio.nodata
    if nodata is not None and np.isfinite(nodata):
        vals[vals == float(nodata)] = np.nan
    vals[~np.isfinite(vals)] = np.nan

    if p.erosion_dem_mask_below_sea_level:
        vals[vals < 0.0] = np.nan
    if p.erosion_dem_max_valid_elev_m is not None and np.isfinite(p.erosion_dem_max_valid_elev_m):
        vals[vals > float(p.erosion_dem_max_valid_elev_m)] = np.nan

    m = np.isfinite(vals) & (ids > 0)
    max_id = int(iso_lut["iso_id"].max())

    if not np.any(m):
        warnings.warn("DEM masking removed all elevation samples; skipping elevation rule.")
        return {}

    sum_elev = np.bincount(ids[m], weights=vals[m], minlength=max_id + 1).astype("float64")
    cnt_elev = np.bincount(ids[m], minlength=max_id + 1).astype("float64")
    mean_elev = np.divide(sum_elev, cnt_elev, out=np.full_like(sum_elev, np.nan), where=cnt_elev > 0)

    return {i: float(mean_elev[i]) for i in range(1, max_id + 1)}


# ==========================================================
# 8) Valuation: elasticity-weighted shock + GEP (Option A)
# ==========================================================
def compute_country_gep_from_country_crop(
    paths,
    df_country_crop_component: pd.DataFrame,
    fao_iso3_csv: str,
    prices_full_csv: str,
    base_year: int,
    gdp_current_2019_csv: str,
    component: str,
) -> pd.DataFrame:
    """Country protected production, shock, value and GDP share, for one prevention component.

    Reads the crop gross production value and GDP tables, then hands the frames to
    `erosion_functions`, which holds the arithmetic.
    """
    df_shock = ef.country_erosion_shock(df_country_crop_component, p.erosion_min_shock_floor)
    df_crop_gpv = load_fao_gpv_iso3_const2019_with_fallback(
        paths, fao_iso3_csv, prices_full_csv, base_year=base_year)
    df_gdp = load_wb_gdp_current_2019(gdp_current_2019_csv)
    return ef.country_gep(df_shock, df_crop_gpv, df_gdp, component)


def read_erosion_dependency(ero_path):
    """Load + normalize the erosion dependency table; return the df.

    Base extraction happens in the CALLER after resolving the configured base scenario through
    utilities.resolve_base_scenario (this function previously hardcoded 'baseline_ignore_damages'
    as the base, silently ignoring p.es_shock_base_scenario -- right only by spelling coincidence).
    """
    df = hb.df_read(str(ero_path))
    df['scenario'] = df['scenario'].str.replace('_2050', '').str.replace('2023.0', 'baseline_2023')
    return df


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


def build_seals7_biophysical_table(src_csv, out_csv):
    """Re-key a biophysical table from ESA lucodes onto SEALS7 classes, for InVEST SDR.

    SDR matches the table's `lucode` against the LULC raster's values, but the shipped table is keyed on
    ESA-CCI codes while our maps are SEALS7 (1-7), so SDR would match nothing. The table already carries
    a `seals_lucode` column, and usle_c/usle_p are CONSTANT within each SEALS class (verified: min == max
    for all 7), so the collapse is unambiguous -- no area weighting to choose. Returns the written path.
    """
    df = hb.df_read(str(src_csv))
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
    hb.log('  erosion watersheds: repaired %d of %d invalid geometries -> %s'
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


def open_raster_1band(path: str) -> xr.DataArray:
    """Open a single-band raster as a 2D DataArray (masked)."""
    return rxr.open_rasterio(path, masked=True).squeeze()


def plot_raster_global(tif_path: str, title: str, out_png: str, downsample_factor: int = 6):
    utilities.assert_exists(tif_path)
    da = rxr.open_rasterio(tif_path, masked=True).squeeze()

    if downsample_factor and downsample_factor > 1:
        da = da.isel(
            y=slice(None, None, downsample_factor),
            x=slice(None, None, downsample_factor),
        )

    arr = da.values.astype("float32", copy=False)
    arr = np.where(np.isfinite(arr), arr, np.nan)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(arr, interpolation="nearest")
    ax.set_title(title, fontsize=16, pad=12)
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Share (0–1)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    utilities.savefig(out_png, dpi=300)




def _setup_sdr_environment():
    """PROJ/GDAL env, gdal exceptions and InVEST-style logging. Called by run_invest_sdr, never at
    import: importing global_invest.erosion must not mutate process-wide state (root logging
    config, GDAL error behaviour). In the source repo this ran at module import."""
    import natcap.invest.utils
    ef._set_proj_gdal_env()
    gdal.UseExceptions()
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt=natcap.invest.utils.LOG_FMT, datefmt="%m/%d/%Y %H:%M:%S ")
    handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[handler])


def _dem_wkt(dem_path: str) -> str:
    """Return DEM CRS as WKT. We trust DEM CRS as the canonical target CRS. (hazelbean raster-info;
    swap verified by the bit-identical section-A rerun.)"""
    wkt = hb.get_raster_info_hb(str(dem_path))['projection']
    if not wkt:
        raise ValueError(f"DEM has no CRS: {dem_path}")
    return wkt


def sanitize_watersheds_for_report(
    watersheds_in: str,
    watersheds_out: str,
    target_wkt: str,
    layer: str = "watersheds",
) -> str:
    """
    Creates a minimal watersheds GeoPackage for InVEST SDR reporting/zonal stats.

    NOTE:
    - SDR does NOT need HydroBASINS IDs; it only needs polygons.
    - We keep only geometry + a safe integer ws_id, dropping everything else.
    - Reproject to DEM CRS (clean WKT) to avoid CRS parse failures.
    - Attempt to fix invalid geometries via buffer(0) (common in global vectors).
    """
    if not hb.path_exists(watersheds_in):
        raise FileNotFoundError(f"Missing watersheds: {watersheds_in}")

    gdf = gpd.read_file(watersheds_in)
    if gdf.empty:
        raise ValueError(f"Watersheds are empty: {watersheds_in}")

    # Fix invalid geometries defensively
    try:
        bad = ~gdf.geometry.is_valid
        if bad.any():
            ef.LOGGER.warning("Watersheds: fixing %d invalid geometries via buffer(0).", int(bad.sum()))
            gdf.loc[bad, "geometry"] = gdf.loc[bad, "geometry"].buffer(0)
    except Exception:
        ef.LOGGER.warning("Watersheds: geometry validity check/fix skipped (non-fatal).")

    # Reproject to DEM CRS using WKT -> CRS
    gdf = gdf.to_crs(target_wkt)

    # Keep only geometry + safe ID
    gdf_out = gdf[["geometry"]].copy()
    gdf_out["ws_id"] = range(1, len(gdf_out) + 1)

    # Write a fresh GPKG
    if hb.path_exists(watersheds_out):
        os.remove(watersheds_out)

    gdf_out.to_file(watersheds_out, layer=layer, driver="GPKG")
    return watersheds_out


def accumulate_upstream_prevention_share(dem_path, avoided_path, potential_path, work_dir,
                                         output_path):
    """Route the two erosion rasters down the drainage network and write the upstream share.

    A field is protected not only by its own cover but by everything upslope of it, so the share
    upstream cover prevents is the flow-accumulated avoided erosion over the flow-accumulated
    potential erosion. Both are accumulated on the same D8 network, so the pixel area cancels and
    the result is a share the on-farm one can be combined with.

    This is the layer the valuation used to read from the source repo's cluster workspace. It is
    computed here instead, from the DEM and the SDR outputs, which is what lets the account run
    without that workspace.

    Args:
        dem_path (str): elevation, resampled here to the erosion rasters' grid.
        avoided_path (str): avoided erosion, t/ha/yr.
        potential_path (str): potential (bare-soil) erosion, t/ha/yr.
        work_dir (str): where the filled DEM, flow direction and accumulation rasters go.
        output_path (str): where to write the share.

    Returns:
        str: `output_path`.
    """
    import pygeoprocessing as pgp
    import pygeoprocessing.routing as routing

    os.makedirs(work_dir, exist_ok=True)
    info = pgp.get_raster_info(avoided_path)
    pixel_size, geotransform, wkt = (
        info['pixel_size'], info['geotransform'], info['projection_wkt'])
    origin = (geotransform[0], geotransform[3])

    def _weight(source_path, destination_path):
        # Flow accumulation sums its weight raster, so nodata and negatives have to become zero
        # first: left as they are, a nodata sentinel would be routed downstream as if it were mass.
        source_ndv = pgp.get_raster_info(source_path)['nodata'][0]
        array = hb.as_array(source_path).astype('float64')
        array = np.where(np.isfinite(array) & (array != source_ndv) & (np.abs(array) < 1e30),
                         np.maximum(array, 0.0), 0.0)
        pgp.numpy_array_to_raster(array.astype('float32'), -1.0, pixel_size, origin, wkt,
                                  destination_path)
        return destination_path

    dem_on_grid = os.path.join(work_dir, 'dem_grid.tif')
    hb.resample_to_match(dem_path, avoided_path, dem_on_grid, resample_method='bilinear')
    avoided_weight = _weight(avoided_path, os.path.join(work_dir, 'avoided_w.tif'))
    potential_weight = _weight(potential_path, os.path.join(work_dir, 'rkls_w.tif'))

    filled = os.path.join(work_dir, 'filled.tif')
    flow_direction = os.path.join(work_dir, 'fdir.tif')
    accumulated_avoided = os.path.join(work_dir, 'acc_avoided.tif')
    accumulated_potential = os.path.join(work_dir, 'acc_rkls.tif')
    routing.fill_pits((dem_on_grid, 1), filled)
    routing.flow_dir_d8((filled, 1), flow_direction)
    routing.flow_accumulation_d8((flow_direction, 1), accumulated_avoided,
                                 weight_raster_path_band=(avoided_weight, 1))
    routing.flow_accumulation_d8((flow_direction, 1), accumulated_potential,
                                 weight_raster_path_band=(potential_weight, 1))

    share = ef.upstream_prevention_share(
        hb.as_array(accumulated_avoided), hb.as_array(accumulated_potential))
    pgp.numpy_array_to_raster(share, -9999.0, pixel_size, origin, wkt, output_path)
    return output_path


def load_elasticity_map(elasticity_csv: str, fallback_value: float) -> tuple[dict, pd.DataFrame]:
    utilities.assert_exists(elasticity_csv, "Provide elasticity CSV in inputs/.")
    df = pd.read_csv(elasticity_csv, encoding="utf-8-sig")
    df = utilities.normalize_columns(df)

    crop_col = None
    for cand in ["crop", "monfreda_crop", "item", "item_name"]:
        if cand in df.columns:
            crop_col = cand
            break

    if crop_col is None:
        warnings.warn(
            "No crop-name column found in elasticity CSV. "
            "Accepted: 'crop'|'monfreda_crop'|'item'|'item_name'. "
            f"All crops will fall back to {fallback_value:.2%}."
        )
        return {}, df

    if "elasticity" not in df.columns:
        raise ValueError("Elasticity CSV must contain 'elasticity'.")

    df["elasticity"] = pd.to_numeric(df["elasticity"], errors="coerce").clip(0.0, 1.0)
    df["__key__"] = df[crop_col].astype(str).str.strip().str.lower()
    df = df[df["__key__"].ne("") & df["elasticity"].notna()].drop_duplicates(subset="__key__", keep="last")
    mapping = dict(zip(df["__key__"], df["elasticity"]))

    audit = df.rename(columns={crop_col: "crop_key"})[["crop_key", "elasticity"]].copy()
    return mapping, audit


# ==============================
# 4) Countries utilities
# ==============================
def load_countries_iso3_alpha(paths, in_crs: rioCRS):
    utilities.assert_exists(paths.input.country_boundary, "Provide boundary with ISO3 column.")
    gdf = gpd.read_file(str(paths.input.country_boundary))
    gdf = gdf[gdf.geometry.notnull()].copy()

    if gdf.crs is None:
        if p.erosion_boundary_source_epsg is not None:
            gdf = gdf.set_crs(p.erosion_boundary_source_epsg)
            warnings.warn(f"Boundary had no CRS; set to EPSG:{p.erosion_boundary_source_epsg}.")
        else:
            minx, miny, maxx, maxy = gdf.total_bounds
            looks_like_wgs84 = (-180.01 <= minx <= 180.01 and -180.01 <= maxx <= 180.01 and
                                -90.01  <= miny <=  90.01 and  -90.01  <= maxy <=  90.01)
            if looks_like_wgs84:
                gdf = gdf.set_crs(4326)
                warnings.warn("Boundary had no CRS; heuristically set to EPSG:4326 (WGS84).")
            else:
                raise ValueError(
                    "Boundary file has no CRS and does not look like lon/lat. "
                    "Set p.erosion_boundary_source_epsg to the file’s source CRS."
                )

    iso_col = None
    # The devstack's own label comes first. A boundary file may carry Natural Earth's columns
    # beside it, and those disagree: in ee_r264_correspondence South Sudan is SSD under
    # iso3_r250_label and SDS under adm0_a3, sov_a3 and su_a3. Preferring adm0_a3 gave the country
    # a code the FAO tables do not use, so its crop GPV came back empty and its GEP was zero while
    # the author's run values it at $3.7M. Only ee_r264_correspondence carries both, so ee_r250
    # never showed it.
    for cand in ["iso3_r250_label", "iso3", "ISO3", "iso_a3", "adm0_a3", "ADM0_A3"]:
        if cand in gdf.columns:
            iso_col = cand
            break
    if not iso_col:
        raise ValueError("Boundary file must contain an ISO3 column (e.g., 'iso3').")
    gdf = gdf.rename(columns={iso_col: "ISO3"})
    gdf["ISO3"] = gdf["ISO3"].astype(str).str.upper()

    name_col = None
    for cand in ["country_name","NAME_EN","ADMIN","NAME_LONG","NAME",
                 "COUNTRY","NAME_0","ADM0_NAME","GEOUNIT","iso3_r250_name"]:
        if cand in gdf.columns:
            name_col = cand
            break
    gdf["country_name"] = gdf[name_col].astype(str) if name_col else gdf["ISO3"]

    gdf = gdf.to_crs(in_crs)
    return gdf[["ISO3","country_name","geometry"]]


def rasterize_iso3(gdf: gpd.GeoDataFrame, like_da: xr.DataArray):
    gdf = gdf.copy()
    gdf["ISO3"] = gdf["ISO3"].astype(str).str.upper()
    lut = (
        gdf[["ISO3"]].drop_duplicates().reset_index(drop=True)
        .assign(iso_id=lambda d: np.arange(1, len(d) + 1, dtype=np.int32))
    )
    gdf = gdf.merge(lut, on="ISO3", how="left")

    arr = rasterio.features.rasterize(
        shapes=zip(gdf.geometry, gdf["iso_id"].astype(int)),
        out_shape=(like_da.rio.height, like_da.rio.width),
        transform=like_da.rio.transform(),
        fill=0,
        dtype="int32",
        all_touched=p.erosion_rasterize_all_touched,
    )
    return arr, lut


def load_fao_prices_full(path: str) -> pd.DataFrame:
    utilities.assert_exists(path, "Provide prices CSV for GPV fallback.")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = utilities.normalize_columns(df)

    if "quantity_tons" not in df.columns and "crop_quantity_fao" in df.columns:
        df = df.rename(columns={"crop_quantity_fao": "quantity_tons"})
    if "item_name" not in df.columns and "item" in df.columns:
        df = df.rename(columns={"item": "item_name"})
    if "iso3" not in df.columns:
        raise ValueError("Prices CSV must contain 'iso3' (or '*' rows for global).")

    for need in ["iso3", "price_usd_per_ton", "quantity_tons"]:
        if need not in df.columns:
            raise ValueError(f"Prices CSV missing '{need}' column.")

    df["iso3"] = df["iso3"].astype(str).str.upper()
    df["price_usd_per_ton"] = pd.to_numeric(df["price_usd_per_ton"], errors="coerce")
    df["quantity_tons"] = pd.to_numeric(df["quantity_tons"], errors="coerce")
    return df


def load_fao_gpv_iso3_const2019_with_fallback(paths, 
    fao_csv_iso3: str,
    prices_full_csv: str,
    base_year: int = 2019
) -> pd.DataFrame:
    utilities.assert_exists(fao_csv_iso3, "Provide iso3-based FAO file (faostat_gpv_2019_iso3.csv).")
    base = pd.read_csv(fao_csv_iso3, encoding="utf-8-sig")
    base = utilities.normalize_columns(base)

    needed = {"iso3", "year", "unit", "value", "element"}
    miss = needed - set(base.columns)
    if miss:
        raise ValueError(f"FAO CSV missing required columns {miss}. Found: {list(base.columns)}")

    base = base[base["year"].astype(str) == str(base_year)].copy()
    el_ok = base["element"].astype(str).str.lower().str.contains("gross production value")
    usd_ok = base["unit"].astype(str).str.lower().str.contains("1000") & base["unit"].astype(str).str.upper().str.contains("USD")
    base = base[el_ok & usd_ok].copy()
    if p.erosion_crops_only:
        base = ef._filter_crops_only(base)

    base["iso3"] = base["iso3"].astype(str).str.upper()
    base["value_thousand_usd"] = pd.to_numeric(base["value"], errors="coerce")
    fao_out = base.groupby("iso3", as_index=False)["value_thousand_usd"].sum()
    fao_out["crop_gpv_const2019_2019"] = fao_out["value_thousand_usd"] * 1000.0
    fao_out = fao_out[["iso3","crop_gpv_const2019_2019"]]

    prices = load_fao_prices_full(prices_full_csv)
    key = "item_code_fao" if "item_code_fao" in prices.columns else "item_name"

    if (prices["iso3"] == "*").any():
        global_price = (
            prices[prices["iso3"] == "*"]
            .groupby(key, as_index=False)["price_usd_per_ton"]
            .mean()
            .rename(columns={"price_usd_per_ton":"price_usd_per_ton_global"})
        )
    else:
        global_price = (
            prices.groupby(key, as_index=False)["price_usd_per_ton"]
            .mean()
            .rename(columns={"price_usd_per_ton":"price_usd_per_ton_global"})
        )

    dfp = prices[prices["iso3"] != "*"].copy()
    dfp = dfp.merge(global_price, on=key, how="left")
    dfp["price_used"] = dfp["price_usd_per_ton"]
    dfp.loc[dfp["price_used"].isna() | (dfp["price_used"] <= 0), "price_used"] = dfp["price_usd_per_ton_global"]
    dfp["fallback_value_usd"] = (dfp["price_used"].clip(lower=0.0) * dfp["quantity_tons"].clip(lower=0.0))

    gpv_fb = dfp.groupby("iso3", as_index=False)["fallback_value_usd"].sum()
    gpv_fb = gpv_fb.rename(columns={"fallback_value_usd":"crop_gpv_const2019_2019_fallback"})

    out = fao_out.merge(gpv_fb, on="iso3", how="outer")
    out["crop_gpv_const2019_2019"] = out["crop_gpv_const2019_2019"].fillna(out["crop_gpv_const2019_2019_fallback"])

    # extra diagnostic (publication transparency)
    out_diag = out.merge(gpv_fb, on="iso3", how="left")
    out_diag.to_csv(paths.output.gpv_fallback_diagnostic, index=False)

    return out[["iso3","crop_gpv_const2019_2019"]]


# ------------------------------------------------------
# 6) World Bank GDP loader
# ------------------------------------------------------
def _write_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def load_wb_gdp_current_2019(gdp_csv: str) -> pd.DataFrame:
    if not hb.path_exists(gdp_csv):
        raise NameError(
            'erosion has no World Bank GDP table at %s. es_parameters carries the path and its '
            'source url, so the shared download task stages it into base data. Fetching it here '
            'per run would make the valuation depend on the day it ran, since the World Bank '
            'revises NY.GDP.MKTP.CD.' % gdp_csv)
    df = utilities.normalize_columns(pd.read_csv(gdp_csv, encoding="utf-8-sig"))
    if "iso3" not in df.columns:
        raise ValueError("GDP CSV must contain column 'iso3'.")
    df["iso3"] = df["iso3"].astype(str).str.upper()
    col = "gdp_const2019_2019"
    if "gdp_current_2019" in df.columns:
        out = df[["iso3", "gdp_current_2019"]].copy()
        out[col] = pd.to_numeric(out["gdp_current_2019"], errors="coerce")
        return out[["iso3", col]]
    if col in df.columns:
        out = df[["iso3", col]].copy()
        out[col] = pd.to_numeric(out[col], errors="coerce")
        return out
    raise NameError(
        'the GDP table at %s carries neither gdp_current_2019 nor %s, only %s. Re-downloading it '
        'here would replace a staged input with whatever the World Bank serves today, so this '
        'stops instead.' % (gdp_csv, col, sorted(df.columns)[:6]))


# ==========================================================
# 7) CORE: SPAM production aggregation given a PS raster
# ==========================================================
def aggregate_country_crop_production(paths, spam_aliases,
    ps_arr01: np.ndarray,                        # 2D array aligned to USLE grid
    usle_like: xr.DataArray,                     # template for coords/dims
    iso_id_raster: np.ndarray,                   # int32 country ids aligned to grid
    id2iso: dict[int, str],
    bandmap: pd.DataFrame,
    elast_map: dict,
    max_id: int,
    out_prefix: str,
) -> pd.DataFrame:
    """
    Computes (for each country & crop):
      - protected_production_tons = PS * yield * harvested_area
      - total_production_tons = yield * harvested_area
      - share_protected_production = protected/total
      - elasticity_used
    """
    rows = []

    with rasterio.open(paths.input.yield_stack) as ds_y, rasterio.open(paths.input.area_stack) as ds_a:
        for _, row in bandmap.iterrows():
            b = int(row["band"])
            crop_raw = str(row["crop"]).strip()
            crop_key = crop_raw.lower()

            if b < 1 or b > ds_y.count or b > ds_a.count:
                continue

            elast = ef.get_elasticity_for_crop(crop_key, elast_map,
                                               p.erosion_yield_reduction_for_shock,
                                               spam_aliases)

            y_native = rxr.open_rasterio(ds_y.name, masked=True).sel(band=b).squeeze()
            a_native = rxr.open_rasterio(ds_a.name, masked=True).sel(band=b).squeeze()

            y_tgt  = y_native.rio.reproject_match(usle_like, resampling=ef.RESAMPLE_YIELD).fillna(0.0)
            ha_tgt = a_native.clip(min=0.0).fillna(0.0).rio.reproject_match(usle_like, resampling=ef.RESAMPLE_AREA).fillna(0.0)

            totalprod = (y_tgt * ha_tgt).fillna(0.0)
            protprod  = xr.DataArray(ps_arr01, coords=usle_like.coords, dims=usle_like.dims) * y_tgt * ha_tgt
            protprod  = protprod.fillna(0.0)

            tp_vals = totalprod.values.astype("float64", copy=False)
            pp_vals = protprod.values.astype("float64", copy=False)
            ids = iso_id_raster.astype("int32", copy=False)

            m_tp = np.isfinite(tp_vals) & (ids > 0)
            m_pp = np.isfinite(pp_vals) & (ids > 0)

            sums_tp = np.bincount(ids[m_tp], weights=tp_vals[m_tp], minlength=max_id + 1)
            sums_pp = np.bincount(ids[m_pp], weights=pp_vals[m_pp], minlength=max_id + 1)

            nz = np.flatnonzero((sums_tp + sums_pp) > 0)
            nz = nz[nz > 0]
            for iso_id in nz:
                iso_code = id2iso.get(int(iso_id))
                if iso_code is None:
                    continue
                protected_tons = float(sums_pp[iso_id])
                total_tons     = float(sums_tp[iso_id])
                share = (protected_tons / total_tons) if total_tons > 1e-12 else np.nan
                rows.append({
                    "component": out_prefix,  # onfarm / upstream / combined
                    "ISO3": iso_code,
                    "crop": crop_raw,
                    "elasticity_used": elast,
                    "protected_production_tons": protected_tons,
                    "total_production_tons": total_tons,
                    "share_protected_production": float(np.clip(share, 0.0, 1.0)) if pd.notna(share) else np.nan,
                })

    return pd.DataFrame(rows)


# ==========================================================
# 9) BIOPHYSICAL — compute PS_onfarm, load UPS, build PS_eff
#     then compute onfarm/upstream/combined in parallel
# ==========================================================
def run_biophysical_decomposed(paths):
    # ---- Required inputs
    utilities.assert_exists(paths.input.usle,  "Expected USLE raster.")
    utilities.assert_exists(paths.input.avoided_erosion, "Expected avoided_erosion raster.")
    utilities.assert_exists(paths.output.upstream_prevention_share,   "Expected upstream_prevention_share.tif from upstream workflow.")
    utilities.assert_exists(paths.input.yield_stack, "Missing SPAM yield stack.")
    utilities.assert_exists(paths.input.area_stack,  "Missing SPAM harvested area stack.")
    utilities.assert_exists(paths.input.bandmap, "Missing SPAM band map CSV.")
    utilities.assert_exists(paths.input.elasticity, "Missing elasticity table.")
    utilities.assert_exists(paths.input.country_boundary, "Missing country boundary GPKG.")

    analysis_crs = rioCRS.from_epsg(p.erosion_analysis_epsg)

    # ---- Load native erosion layers
    usle_native = open_raster_1band(paths.input.usle)
    avo_native  = open_raster_1band(paths.input.avoided_erosion)
    ups_native  = open_raster_1band(paths.output.upstream_prevention_share)

    ef._ensure_crs(usle_native, "USLE")
    ef._ensure_crs(avo_native,  "AVOID")
    ef._ensure_crs(ups_native,  "UPS")

    # ---- Reproject to equal-area CRS
    usle = ef.reproject_to_analysis_grid(usle_native, analysis_crs, ef.RESAMPLE_USLE_AVOID)
    avo  = ef.reproject_to_analysis_grid(avo_native,  analysis_crs, ef.RESAMPLE_USLE_AVOID)
    ups  = ef.reproject_to_analysis_grid(ups_native,  analysis_crs, ef.RESAMPLE_UPS)

    # ---- Force exact alignment to USLE grid
    avo  = avo.rio.reproject_match(usle, resampling=ef.RESAMPLE_USLE_AVOID)
    ups  = ups.rio.reproject_match(usle, resampling=ef.RESAMPLE_UPS)

    usle = ef._clean_nonneg(usle)
    avo  = ef._clean_nonneg(avo)
    ups_vals = ef._clip01_arr(ups.values)

    # ---- Countries raster
    gdf_countries = load_countries_iso3_alpha(paths, usle.rio.crs)
    gdf_countries["area_km2"] = ef.compute_country_areas_km2(gdf_countries)
    iso_id_raster, iso_lut = rasterize_iso3(gdf_countries, usle)
    max_id = int(iso_lut["iso_id"].max())
    id2iso = dict(zip(iso_lut["iso_id"].to_numpy(), iso_lut["ISO3"].to_numpy()))
    name_by_iso = dict(zip(gdf_countries["ISO3"], gdf_countries["country_name"]))

    # ---- Threshold policy (optional DEM)
    mean_elev_by_id = compute_country_mean_elevation(
        usle, iso_id_raster, iso_lut,
        paths.input.dem if (paths.input.dem and hb.path_exists(paths.input.dem)) else None
    )

    # A country's area is the sum of the sub-regions the boundary file splits it into, not any
    # one of them. r264 splits six countries out into territories -- CHN and IND into six rows
    # each, FRA, GBR, PAK and TUR into two -- so deciding on a single sub-region's area could let
    # a country qualify as small on the strength of an island and take the low soil-loss
    # tolerance, which enlarges the domain the severity threshold defines. Measured on that file,
    # the countries whose small/large status can turn on this are CHN, FRA, GBR, IND and TUR.
    # r250, which the run now reads, carries one row per country, so it cannot arise there at all.
    df_country_area = (gdf_countries[["ISO3", "area_km2"]].rename(columns={"ISO3": "iso3"})
                       .groupby("iso3", as_index=False)["area_km2"].sum(min_count=1))
    df_threshold = ef.country_threshold_policy(
        iso_lut.rename(columns={"ISO3": "iso3"})
               .merge(df_country_area, on="iso3", how="left")
               .assign(mean_elevation_m=lambda d: [mean_elev_by_id.get(int(i), np.nan)
                                                   for i in d["iso_id"]]),
        p.erosion_threshold_high_t_ha_yr, p.erosion_threshold_low_t_ha_yr, p.erosion_small_country_area_km2, p.erosion_low_elevation_mean_m)
    if df_threshold["iso3"].duplicated().any():
        raise ValueError("the threshold policy has more than one row for a country, so the "
                         "per-country threshold raster would depend on row order.")

    threshold_by_id = np.full(max_id + 1, p.erosion_threshold_high_t_ha_yr, dtype="float32")
    id_by_iso = dict(zip(iso_lut["ISO3"], iso_lut["iso_id"].astype(int)))
    for iso3, threshold in zip(df_threshold["iso3"], df_threshold["threshold_t_ha_yr"]):
        threshold_by_id[id_by_iso[iso3]] = threshold
    threshold_map = threshold_by_id[iso_id_raster.astype("int32")]

    severe = (usle.values > threshold_map) if p.erosion_apply_severe_filter else np.ones_like(usle.values, dtype=bool)

    df_threshold.insert(1, "country_name", [name_by_iso.get(i, i) for i in df_threshold["iso3"]])
    df_threshold.rename(columns={"iso3": "ISO3"}).to_csv(paths.output.threshold_policy, index=False)

    # ---- Bandmap + elasticity
    bandmap = hb.df_read(str(paths.input.bandmap))
    bandmap = utilities.normalize_columns(bandmap)
    if "band" not in bandmap.columns or "crop" not in bandmap.columns:
        raise ValueError("paths.input.bandmap must have columns: 'band', 'crop'.")
    bandmap["crop"] = bandmap["crop"].astype(str).str.strip()
    bandmap["band"] = pd.to_numeric(bandmap["band"], errors="coerce").astype("Int64")

    elast_map, elast_audit = load_elasticity_map(paths.input.elasticity, fallback_value=p.erosion_yield_reduction_for_shock)
    elast_audit.to_csv(paths.output.elasticity_audit, index=False)

    # ---- Cropland mask built from SPAM area (union across bands) + area conservation audit
    cropland_mask = None
    area_conservation_rows = []

    with rasterio.open(paths.input.area_stack) as ds_a:
        for _, row in bandmap.iterrows():
            b = int(row["band"])
            if b < 1 or b > ds_a.count:
                continue
            a_native = rxr.open_rasterio(ds_a.name, masked=True).sel(band=b).squeeze()
            if a_native.rio.crs is None:
                raise ValueError(f"Area stack band {b} has no CRS. Fix the stack metadata.")
            a_pos  = a_native.clip(min=0.0).fillna(0.0)
            ha_tgt = a_pos.rio.reproject_match(usle, resampling=ef.RESAMPLE_AREA).fillna(0.0)

            tot_src = float(np.nansum(a_pos.values))
            tot_tgt = float(np.nansum(ha_tgt.values))
            ratio = (tot_tgt / tot_src) if tot_src > 0 else np.nan
            area_conservation_rows.append({
                "band": b,
                "crop": str(row["crop"]).strip(),
                "total_area_ha_src": tot_src,
                "total_area_ha_tgt": tot_tgt,
                "ratio_tgt_over_src": ratio
            })

            band_mask = (ha_tgt > 0)
            cropland_mask = band_mask if cropland_mask is None else (cropland_mask | band_mask)

    pd.DataFrame(area_conservation_rows).to_csv(paths.output.area_conservation_audit, index=False)

    cm = cropland_mask.values.astype(bool)

    # ---- Prevention shares, on cropland where soil loss is severe (see erosion_functions)
    ps_onfarm = ef.restrict_to_valued_pixels(
        ef.onfarm_prevention_share(avo.values, usle.values), cm, severe).astype("float32")
    ps_upstream = ef.restrict_to_valued_pixels(ups_vals, cm, severe).astype("float32")
    ps_combined = ef.combined_prevention_share(ps_onfarm, ps_upstream).astype("float32")

    # ---- Save PS rasters for transparency
    ef._write_share(paths.output.prevention_share_onfarm, usle, ps_onfarm)
    ef._write_share(paths.output.prevention_share_upstream, usle, ps_upstream)
    ef._write_share(paths.output.prevention_share_combined, usle, ps_combined)

    # ---- Country PS diagnostics (means on cropland&severe)
    diag_mask = cm & severe & (iso_id_raster > 0)
    ids_1d = iso_id_raster[diag_mask].astype("int32", copy=False)

    mean_onfarm = ef._bincount_weighted_mean(ids_1d, ps_onfarm[diag_mask], max_id)
    mean_up     = ef._bincount_weighted_mean(ids_1d, ps_upstream[diag_mask], max_id)
    mean_comb   = ef._bincount_weighted_mean(ids_1d, ps_combined[diag_mask], max_id)

    diag_rows = []
    for i in range(1, max_id + 1):
        iso = id2iso.get(i)
        if not iso:
            continue
        diag_rows.append({
            "ISO3": iso,
            "country_name": name_by_iso.get(iso, iso),
            "mean_ps_onfarm_cropland_severe": float(mean_onfarm[i]),
            "mean_ps_upstream_cropland_severe": float(mean_up[i]),
            "mean_ps_combined_cropland_severe": float(mean_comb[i]),
        })
    df_diag = pd.DataFrame(diag_rows)

    # Optional: include upstream LULC attribution shares as country means (diagnostics only)
    if p.erosion_use_upslope_lulc_diagnostics:
        def _load_attr(p, nm: str) -> np.ndarray | None:
            if not hb.path_exists(p):
                warnings.warn(f"[ATTR] Missing {nm}: {p}")
                return None
            da0 = open_raster_1band(p)
            ef._ensure_crs(da0, nm)
            da1 = ef.reproject_to_analysis_grid(da0, analysis_crs, Resampling.average).rio.reproject_match(usle, resampling=Resampling.average)
            return ef._clip01_arr(da1.values)

        attrs = {
            "upslope_forest_share": _load_attr(paths.output.upslope_forest_share, "upslope_forest_share"),
            "upslope_grass_share":  _load_attr(paths.output.upslope_grass_share,  "upslope_grass_share"),
            "upslope_cropland_share": _load_attr(paths.output.upslope_cropland_share, "upslope_cropland_share"),
            "upslope_bare_share":   _load_attr(paths.output.upslope_bare_share,   "upslope_bare_share"),
        }
        for nm, arr in attrs.items():
            if arr is None:
                continue
            mean_attr = ef._bincount_weighted_mean(ids_1d, arr[diag_mask], max_id)
            df_diag[f"mean_{nm}_cropland_severe"] = [float(mean_attr[i]) for i in range(1, max_id + 1) if id2iso.get(i)]

    df_diag.to_csv(paths.output.country_ps_diagnostics, index=False)

    # ---- Soil retained on cropland (tons): AE rate * pixel area, cropland mask (independent of decomposition)
    px_ha = ef.pixel_area_hectares(usle)
    soil_retained_tons = xr.where(cropland_mask, (avo * px_ha), 0.0).fillna(0.0)
    vals = soil_retained_tons.values.astype("float64")
    ids_all = iso_id_raster.astype("int32")
    m = np.isfinite(vals) & (ids_all > 0)
    soil_sums = np.bincount(ids_all[m], weights=vals[m], minlength=max_id + 1)

    df_soil = pd.DataFrame({
        "ISO3": [id2iso[i] for i in range(1, max_id + 1) if i in id2iso],
        "soil_retained_cropland_tons": [float(soil_sums[i]) for i in range(1, max_id + 1) if i in id2iso]
    })

    # ---- Compute country-crop protected production for each component
    spam_aliases = {k: v.split(';') for k, v in
                    utilities.read_lookup(p.erosion_spam_alias_path,
                                          'spam_label', 'aliases').items()}
    df_cc_onfarm   = aggregate_country_crop_production(paths, spam_aliases, ps_onfarm,   usle, iso_id_raster, id2iso, bandmap, elast_map, max_id, "onfarm")
    df_cc_upstream = aggregate_country_crop_production(paths, spam_aliases, ps_upstream, usle, iso_id_raster, id2iso, bandmap, elast_map, max_id, "upstream")
    df_cc_combined = aggregate_country_crop_production(paths, spam_aliases, ps_combined, usle, iso_id_raster, id2iso, bandmap, elast_map, max_id, "combined")

    # ---- Save per-country-crop (long form; publication transparency)
    df_country_crop_long = pd.concat([df_cc_onfarm, df_cc_upstream, df_cc_combined], ignore_index=True)
    df_country_crop_long.to_csv(paths.output.country_crop_protected_production_long, index=False)

    # Optional: also write 3 separate files (handy for reviewers)
    df_cc_onfarm.to_csv(os.path.join(paths.output.directory, "country_crop_protected_production_onfarm.csv"), index=False)
    df_cc_upstream.to_csv(os.path.join(paths.output.directory, "country_crop_protected_production_upstream.csv"), index=False)
    df_cc_combined.to_csv(os.path.join(paths.output.directory, "country_crop_protected_production_combined.csv"), index=False)

    # Country name master
    country_master = gdf_countries[["ISO3","country_name"]].drop_duplicates().reset_index(drop=True)

    return {
        "usle_template": usle,
        "iso_id_raster": iso_id_raster,
        "iso_lut": iso_lut,
        "id2iso": id2iso,
        "country_master": country_master,
        "df_soil": df_soil,
        "df_diag": df_diag,
        "df_country_crop": {
            "onfarm": df_cc_onfarm,
            "upstream": df_cc_upstream,
            "combined": df_cc_combined,
        }
    }


# ==============================================
# 10) INTEGRATE + WRITE (decomposed outputs)
# ==============================================
def integrate_and_write(paths):
    t0 = time.time()

    # ---- Run biophysical + produce country-crop tables per component
    pack = run_biophysical_decomposed(paths)
    country_master = pack["country_master"]
    df_soil = pack["df_soil"]
    df_diag = pack["df_diag"]
    dcc = pack["df_country_crop"]

    # ---- Compute valuation per component
    df_gep_onfarm = compute_country_gep_from_country_crop(
        paths, dcc["onfarm"], paths.input.fao_gpv, paths.input.fao_prices, p.erosion_base_year, paths.input.gdp, "onfarm"
    )
    df_gep_upstream = compute_country_gep_from_country_crop(
        paths, dcc["upstream"], paths.input.fao_gpv, paths.input.fao_prices, p.erosion_base_year, paths.input.gdp, "upstream"
    )
    df_gep_combined = compute_country_gep_from_country_crop(
        paths, dcc["combined"], paths.input.fao_gpv, paths.input.fao_prices, p.erosion_base_year, paths.input.gdp, "combined"
    )

    # ---- Pivot to wide (one row per country) for integrated_country_gep.csv
    def _pivot_component(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        x = df.copy()
        x = x.rename(columns={
            "iso3": "iso3",
            "protected_production_tons": f"protected_production_tons_{prefix}",
            "total_production_tons": f"total_production_tons_{prefix}",
            "share_protected_production": f"share_protected_production_{prefix}",
            "erosion_shock_share": f"erosion_shock_share_{prefix}",
            "gep_const2019_usd": f"gep_const2019_usd_{prefix}",
            "gdp_loss_pct": f"gdp_loss_pct_{prefix}",
        })
        keep = ["iso3",
                f"protected_production_tons_{prefix}",
                f"total_production_tons_{prefix}",
                f"share_protected_production_{prefix}",
                f"erosion_shock_share_{prefix}",
                "crop_gpv_const2019_2019",
                "gdp_const2019_2019",
                f"gep_const2019_usd_{prefix}",
                f"gdp_loss_pct_{prefix}"]
        return x[keep]

    w_on  = _pivot_component(df_gep_onfarm, "onfarm")
    w_up  = _pivot_component(df_gep_upstream, "upstream")
    w_com = _pivot_component(df_gep_combined, "combined")

    # ---- Merge wide tables + add soil retained + add diagnostics
    out = country_master.rename(columns={"ISO3":"iso3"}).merge(w_on, on="iso3", how="left")
    out = out.merge(w_up.drop(columns=["crop_gpv_const2019_2019","gdp_const2019_2019"], errors="ignore"), on="iso3", how="left")
    out = out.merge(w_com.drop(columns=["crop_gpv_const2019_2019","gdp_const2019_2019"], errors="ignore"), on="iso3", how="left")

    out = out.merge(df_soil.rename(columns={"ISO3":"iso3"}), on="iso3", how="left")
    out = out.merge(df_diag.rename(columns={"ISO3":"iso3"}), on=["iso3","country_name"], how="left")

    # ---- Optional: “incremental” attribution (not additive; shown for interpretation only)
    #   incremental_upstream = combined - onfarm
    #   incremental_onfarm   = combined - upstream
    out["gep_incremental_upstream_usd"] = out["gep_const2019_usd_combined"] - out["gep_const2019_usd_onfarm"]
    out["gep_incremental_onfarm_usd"]   = out["gep_const2019_usd_combined"] - out["gep_const2019_usd_upstream"]

    # ---- Write final integrated file
    out_cols_front = ["iso3","country_name",
                      "crop_gpv_const2019_2019","gdp_const2019_2019",
                      "soil_retained_cropland_tons"]
    # keep everything else after
    out = out[out_cols_front + [c for c in out.columns if c not in out_cols_front]]
    out = out.sort_values(["country_name","iso3"], na_position="last")
    out.to_csv(paths.output.integrated_country_gep, index=False)

    # ---- Also write a “long” valuation table (nice for figures/tables)
    df_gep_long = pd.concat([df_gep_onfarm, df_gep_upstream, df_gep_combined], ignore_index=True)
    df_gep_long.to_csv(paths.output.country_gep_decomposition_long, index=False)

    # ---- Manifest + run metadata
    manifest = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "scenario_name": p.erosion_scenario_name,
        "run_tag": p.erosion_run_tag,
        "analysis_epsg": p.erosion_analysis_epsg,
        "apply_severe_filter": p.erosion_apply_severe_filter,
        "threshold_high_t_ha_yr": p.erosion_threshold_high_t_ha_yr,
        "threshold_low_t_ha_yr": p.erosion_threshold_low_t_ha_yr,
        "small_country_area_km2": p.erosion_small_country_area_km2,
        "low_elevation_mean_m": p.erosion_low_elevation_mean_m,
        "definitions": {
            "PS_onfarm": "AE/(AE+USLE) on cropland pixels (and severe if enabled); else 0",
            "UPS": "upstream_prevention_share evaluated at pixel j (and restricted to cropland & severe); else 0",
            "PS_combined": "1 - (1-PS_onfarm)*(1-UPS) (union-of-protection; avoids double counting)",
        },
        "inputs": {
            "usle_path": str(paths.input.usle),
            "avoided_path": str(paths.input.avoided_erosion),
            "ups_path": str(paths.output.upstream_prevention_share),
            "boundary_gpkg": str(paths.input.country_boundary),
            "dem_path_for_thresholds": str(paths.input.dem),
            "yield_stack": str(paths.input.yield_stack),
            "area_stack": str(paths.input.area_stack),
            "bandmap_csv": str(paths.input.bandmap),
            "elasticity_csv": str(paths.input.elasticity),
            "fao_gpv_iso3_csv": str(paths.input.fao_gpv),
            "fao_prices_csv": str(paths.input.fao_prices),
            "worldbank_gdp_csv": str(paths.input.gdp),
        },
        "outputs": {
            "integrated_country_gep": str(paths.output.integrated_country_gep),
            "country_crop_protected_production_long": str(paths.output.country_crop_protected_production_long),
            "country_gep_decomposition_long": str(paths.output.country_gep_decomposition_long),
            "country_ps_diagnostics": str(paths.output.country_ps_diagnostics),
            "ps_onfarm_raster": str(paths.output.prevention_share_onfarm),
            "ps_upstream_raster": str(paths.output.prevention_share_upstream),
            "ps_combined_raster": str(paths.output.prevention_share_combined),
            "area_conservation_audit": str(paths.output.area_conservation_audit),
            "elasticity_audit": str(paths.output.elasticity_audit),
            "gpv_fallback_diagnostic": str(paths.output.gpv_fallback_diagnostic),
            "threshold_policy": str(paths.output.threshold_policy),
        },
        "elapsed_minutes": round((time.time() - t0) / 60.0, 3),
    }
    with open(paths.output.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    run_metadata_text = f"""
============================================================
Integrated GEP run — {p.erosion_scenario_name}
Run tag: {p.erosion_run_tag}
Timestamp (UTC): {manifest['timestamp_utc']}

Analysis CRS:
  EPSG:{p.erosion_analysis_epsg} (equal-area required for tons conversions)
  GDAL {gdal.__version__}, and the settings that decide how a stored CRS is read:
{chr(10).join(f'    {name}: {value}' for name, value in ef.proj_gdal_env().items())}

Severe erosion definition:
  severe = usle_actual > threshold(country)
  Default threshold (high): {p.erosion_threshold_high_t_ha_yr:g} t/ha/yr
  Low threshold (special cases): {p.erosion_threshold_low_t_ha_yr:g} t/ha/yr

Decomposed prevention shares (cropland-only; severe-only if enabled):
  On-farm:   PS_onfarm = AE/(AE+USLE)
  Upstream:  PS_upstream = UPS(j)
  Combined:  PS_combined = 1 - (1-PS_onfarm)(1-PS_upstream)

Primary output:
  {paths.output.integrated_country_gep}

Also written:
  - country_crop_protected_production_long.csv (country-crop, all components)
  - country_gep_decomposition_long.csv (country totals, all components)
  - country_ps_diagnostics.csv (country mean PS values on cropland&severe)
  - gpv_fallback_diagnostic.csv
  - area_conservation_audit.csv
  - elasticity_audit.csv

Elapsed minutes: {manifest['elapsed_minutes']}
============================================================
"""
    with open(os.path.join(paths.output.directory, "run_metadata.txt"), "w", encoding="utf-8") as f:
        f.write(run_metadata_text)

    hb.log(f"Done → {paths.output.integrated_country_gep}")
    hb.log(f"Manifest → {paths.output.manifest}")


def load_world_boundary_prefer_run(paths) -> gpd.GeoDataFrame:
    if hb.path_exists(paths.input.country_boundary):
        world = gpd.read_file(paths.input.country_boundary)
        iso_col = utilities.pick_iso3_column(world)
        if not iso_col:
            raise ValueError(f"Boundary has no ISO3 column. Columns: {list(world.columns)}")
        world = world.rename(columns={iso_col: "iso3"})
        world["iso3"] = world["iso3"].astype(str).str.upper()

        name_col = utilities.pick_name_column(world)
        if name_col and name_col != "country_name":
            world = world.rename(columns={name_col: "country_name"})
        if "country_name" not in world.columns:
            world["country_name"] = world["iso3"]

        world = world[world.geometry.notna()].copy()
        return world[["iso3", "country_name", "geometry"]]

    raise FileNotFoundError(f"Boundary GPKG not found: {paths.input.country_boundary}")


def generate_all_maps_and_figures(paths):
    """Driver that produces every map/figure/CSV described in the module docstring below (originally a flat script)."""
    # =============================================================================
    # 2) LOAD DATA
    # =============================================================================
    utilities.assert_exists(paths.output.integrated_country_gep, "Run the latest integrated pipeline first.")
    df = hb.df_read(str(paths.output.integrated_country_gep))
    df.columns = [c.strip() for c in df.columns]
    
    if "ISO3" in df.columns and "iso3" not in df.columns:
        df = df.rename(columns={"ISO3": "iso3"})
    df["iso3"] = df["iso3"].astype(str).str.upper()
    
    NUM_COLS = [
        "crop_gpv_const2019_2019", "gdp_const2019_2019", "soil_retained_cropland_tons",
        "protected_production_tons_onfarm", "total_production_tons_onfarm", "share_protected_production_onfarm",
        "erosion_shock_share_onfarm", "gep_const2019_usd_onfarm", "gdp_loss_pct_onfarm",
        "protected_production_tons_upstream", "total_production_tons_upstream", "share_protected_production_upstream",
        "erosion_shock_share_upstream", "gep_const2019_usd_upstream", "gdp_loss_pct_upstream",
        "protected_production_tons_combined", "total_production_tons_combined", "share_protected_production_combined",
        "erosion_shock_share_combined", "gep_const2019_usd_combined", "gdp_loss_pct_combined",
        "mean_ps_onfarm_cropland_severe", "mean_ps_upstream_cropland_severe", "mean_ps_combined_cropland_severe",
        "gep_incremental_upstream_usd", "gep_incremental_onfarm_usd",
    ]
    df = utilities.to_num(df, NUM_COLS)
    
    if {"gep_const2019_usd_onfarm", "gep_const2019_usd_upstream", "gep_const2019_usd_combined"}.issubset(df.columns):
        df["gep_const2019_usd_overlap"] = (
            df["gep_const2019_usd_onfarm"].fillna(0.0)
            + df["gep_const2019_usd_upstream"].fillna(0.0)
            - df["gep_const2019_usd_combined"].fillna(0.0)
        )
        sum_components = df["gep_const2019_usd_onfarm"].fillna(0.0) + df["gep_const2019_usd_upstream"].fillna(0.0)
        df["overlap_pct_of_sum_components"] = (
            100.0 * df["gep_const2019_usd_overlap"] / sum_components.where(sum_components > 0)
        ).where(sum_components > 0)
    else:
        df["gep_const2019_usd_overlap"] = np.nan
        df["overlap_pct_of_sum_components"] = np.nan
    
    for c in [
        "crop_gpv_const2019_2019",
        "gdp_const2019_2019",
        "gep_const2019_usd_onfarm",
        "gep_const2019_usd_upstream",
        "gep_const2019_usd_combined",
        "gep_const2019_usd_overlap",
        "gep_incremental_upstream_usd",
        "gep_incremental_onfarm_usd",
    ]:
        if c in df.columns:
            df[f"{c}_million"] = df[c] / p.erosion_usd_to_millions
    
    if "country_name" not in df.columns:
        df["country_name"] = df["iso3"]
    
    if hb.path_exists(paths.output.country_crop_long):
        df_crop_long = hb.df_read(str(paths.output.country_crop_long))
        df_crop_long.columns = [c.strip() for c in df_crop_long.columns]
        if "ISO3" in df_crop_long.columns and "iso3" not in df_crop_long.columns:
            df_crop_long = df_crop_long.rename(columns={"ISO3": "iso3"})
    else:
        df_crop_long = None
    
    df.to_csv(os.path.join(paths.output.figure_directory, "integrated_country_gep_plus_overlap.csv"), index=False)
    
    
    # =============================================================================
    # 3) WORLD GEOMETRY
    # =============================================================================
    world = load_world_boundary_prefer_run(paths)
    world["iso3"] = world["iso3"].astype(str).str.upper()
    g = world.merge(df, on="iso3", how="left")
    
    
    # =============================================================================
    # 4) BAR FIGURES
    # =============================================================================
    
    # 4.1 Top countries: Combined GEP
    col = "gep_const2019_usd_combined"
    if col in df.columns:
        top = utilities.top_n(df, col, p.erosion_top_n).copy()
        top["label"] = top["country_name"].fillna(top["iso3"])
        top = top.sort_values(col, ascending=True)
    
        plt.figure(figsize=(12, 10))
        plt.barh(top["label"], top[f"{col}_million"])
        plt.xlabel(f"Combined GEP ({p.erosion_money_unit_label})", fontsize=12)
        plt.title(f"Top {p.erosion_top_n}: Combined GEP from severe erosion protection", fontsize=16, pad=12)
        plt.grid(axis="x", alpha=0.25)
        utilities.savefig(os.path.join(paths.output.figure_directory, "fig1_top20_combined_gep_2019usd_million.png"), dpi=300)
    
    # 4.2 Decomposition to combined
    if {"gep_const2019_usd_onfarm_million", "gep_const2019_usd_combined_million"}.issubset(df.columns):
        top2 = utilities.top_n(df, "gep_const2019_usd_combined", p.erosion_top_n).copy()
        top2["label"] = top2["country_name"].fillna(top2["iso3"])
        top2 = top2.sort_values("gep_const2019_usd_combined", ascending=True)
    
        on = top2["gep_const2019_usd_onfarm_million"].fillna(0.0)
        comb = top2["gep_const2019_usd_combined_million"].fillna(0.0)
        incr_up = (comb - on).clip(lower=0.0)
    
        plt.figure(figsize=(12, 10))
        plt.barh(top2["label"], on, label="On-farm protection (standalone)")
        plt.barh(top2["label"], incr_up, left=on, label="Incremental upstream protection (given on-farm)")
        plt.xlabel(f"GEP ({p.erosion_money_unit_label})", fontsize=12)
        plt.title(f"Top {p.erosion_top_n}: Decomposition summing to Combined GEP", fontsize=16, pad=12)
        plt.grid(axis="x", alpha=0.25)
        plt.legend(loc="lower right", frameon=True)
        utilities.savefig(os.path.join(paths.output.figure_directory, "fig2_top20_decomposition_to_combined_2019usd_million.png"), dpi=300)
    
    # 4.3 Top overlap percent
    if "overlap_pct_of_sum_components" in df.columns:
        top_ov = utilities.top_n(df, "gep_const2019_usd_overlap", p.erosion_top_n).copy()
        top_ov["label"] = top_ov["country_name"].fillna(top_ov["iso3"])
        top_ov = top_ov.sort_values("gep_const2019_usd_overlap", ascending=True)
    
        plt.figure(figsize=(12, 10))
        plt.barh(top_ov["label"], top_ov["overlap_pct_of_sum_components"])
        plt.xlabel("Overlap as % of (On-farm + Upstream)", fontsize=12)
        plt.title(f"Top {p.erosion_top_n}: Overlap removed by union-of-protection", fontsize=16, pad=12)
        plt.grid(axis="x", alpha=0.25)
        utilities.savefig(os.path.join(paths.output.figure_directory, "fig3_top20_overlap_pct_of_sum.png"), dpi=300)
    
    # 4.4 Top overlap absolute
    if "gep_const2019_usd_overlap_million" in df.columns:
        top_ov_abs = utilities.top_n(df, "gep_const2019_usd_overlap", p.erosion_top_n).copy()
        top_ov_abs["label"] = top_ov_abs["country_name"].fillna(top_ov_abs["iso3"])
        top_ov_abs = top_ov_abs.sort_values("gep_const2019_usd_overlap", ascending=True)
    
        plt.figure(figsize=(12, 10))
        plt.barh(top_ov_abs["label"], top_ov_abs["gep_const2019_usd_overlap_million"])
        plt.xlabel(f"Overlap removed ({p.erosion_money_unit_label})", fontsize=12)
        plt.title(f"Top {p.erosion_top_n}: Overlap removed in absolute terms", fontsize=16, pad=12)
        plt.grid(axis="x", alpha=0.25)
        utilities.savefig(os.path.join(paths.output.figure_directory, "fig4_top20_overlap_removed_2019usd_million.png"), dpi=300)
    
    # 4.5 Macro exposure
    if "gdp_loss_pct_combined" in df.columns:
        top_gdp = utilities.top_n(df, "gdp_loss_pct_combined", p.erosion_top_n).copy()
        top_gdp["label"] = top_gdp["country_name"].fillna(top_gdp["iso3"])
        top_gdp = top_gdp.sort_values("gdp_loss_pct_combined", ascending=True)
    
        plt.figure(figsize=(12, 10))
        plt.barh(top_gdp["label"], top_gdp["gdp_loss_pct_combined"])
        plt.xlabel("Combined GEP as % of GDP", fontsize=12)
        plt.title(f"Top {p.erosion_top_n}: Macro exposure (Combined GEP / GDP)", fontsize=16, pad=12)
        plt.grid(axis="x", alpha=0.25)
        utilities.savefig(os.path.join(paths.output.figure_directory, "fig5_top20_gdp_loss_pct_combined.png"), dpi=300)
    
    # 4.6 Top countries by combined protected production
    if "protected_production_tons_combined" in df.columns:
        top_prot = utilities.top_n(df, "protected_production_tons_combined", p.erosion_top_n).copy()
        top_prot["label"] = top_prot["country_name"].fillna(top_prot["iso3"])
        top_prot = top_prot.sort_values("protected_production_tons_combined", ascending=True)
    
        plt.figure(figsize=(12, 10))
        plt.barh(top_prot["label"], top_prot["protected_production_tons_combined"])
        plt.xlabel("Protected production (tons)", fontsize=12)
        plt.title(f"Top {p.erosion_top_n}: Countries by protected production (combined)", fontsize=16, pad=12)
        plt.grid(axis="x", alpha=0.25)
        utilities.savefig(os.path.join(paths.output.figure_directory, "fig6_top20_protected_production_tons_combined.png"), dpi=300)
    
    # 4.7 Top countries by crop GPV
    if "crop_gpv_const2019_2019_million" in df.columns:
        top_gpv = utilities.top_n(df, "crop_gpv_const2019_2019", p.erosion_top_n).copy()
        top_gpv["label"] = top_gpv["country_name"].fillna(top_gpv["iso3"])
        top_gpv = top_gpv.sort_values("crop_gpv_const2019_2019", ascending=True)
    
        plt.figure(figsize=(12, 10))
        plt.barh(top_gpv["label"], top_gpv["crop_gpv_const2019_2019_million"])
        plt.xlabel(f"Crop production value ({p.erosion_money_unit_label})", fontsize=12)
        plt.title(f"Top {p.erosion_top_n}: Countries by crop production value", fontsize=16, pad=12)
        plt.grid(axis="x", alpha=0.25)
        utilities.savefig(os.path.join(paths.output.figure_directory, "fig7_top20_crop_gpv_2019usd_million.png"), dpi=300)
    
    # 4.8 On-farm vs upstream standalone
    if {"gep_const2019_usd_onfarm_million", "gep_const2019_usd_upstream_million"}.issubset(df.columns):
        top_cmp = utilities.top_n(df, "gep_const2019_usd_combined", p.erosion_top_n).copy()
        top_cmp["label"] = top_cmp["country_name"].fillna(top_cmp["iso3"])
        top_cmp = top_cmp.sort_values("gep_const2019_usd_combined", ascending=True)
    
        y = np.arange(len(top_cmp))
        h = 0.38
    
        plt.figure(figsize=(12, 10))
        plt.barh(y - h/2, top_cmp["gep_const2019_usd_onfarm_million"].fillna(0.0), height=h, label="On-farm")
        plt.barh(y + h/2, top_cmp["gep_const2019_usd_upstream_million"].fillna(0.0), height=h, label="Upstream")
        plt.yticks(y, top_cmp["label"])
        plt.xlabel(f"GEP ({p.erosion_money_unit_label})", fontsize=12)
        plt.title(f"Top {p.erosion_top_n}: Standalone On-farm vs Upstream GEP", fontsize=16, pad=12)
        plt.grid(axis="x", alpha=0.25)
        plt.legend(frameon=True)
        utilities.savefig(os.path.join(paths.output.figure_directory, "fig8_top20_onfarm_vs_upstream_2019usd_million.png"), dpi=300)
    
    
    # =============================================================================
    # 5) HISTOGRAMS
    # =============================================================================
    
    if "share_protected_production_combined" in df.columns:
        m = np.isfinite(df["share_protected_production_combined"])
        plt.figure(figsize=(9, 6))
        plt.hist(df.loc[m, "share_protected_production_combined"].clip(lower=0, upper=1), bins=30)
        plt.xlabel("Share of protected production (combined)", fontsize=12)
        plt.ylabel("Number of countries", fontsize=12)
        plt.title("Distribution of share of protected production (combined)", fontsize=16, pad=12)
        plt.grid(alpha=0.25)
        utilities.savefig(os.path.join(paths.output.figure_directory, "hist_share_protected_production_combined.png"), dpi=300)
    
    if "erosion_shock_share_combined" in df.columns:
        m = np.isfinite(df["erosion_shock_share_combined"])
        plt.figure(figsize=(9, 6))
        plt.hist(df.loc[m, "erosion_shock_share_combined"].clip(lower=0), bins=30)
        plt.xlabel("Erosion shock share (combined)", fontsize=12)
        plt.ylabel("Number of countries", fontsize=12)
        plt.title("Distribution of erosion shock shares (combined)", fontsize=16, pad=12)
        plt.grid(alpha=0.25)
        utilities.savefig(os.path.join(paths.output.figure_directory, "hist_erosion_shock_share_combined.png"), dpi=300)
    
    if "overlap_pct_of_sum_components" in df.columns:
        m = np.isfinite(df["overlap_pct_of_sum_components"])
        plt.figure(figsize=(9, 6))
        plt.hist(df.loc[m, "overlap_pct_of_sum_components"], bins=30)
        plt.xlabel("Overlap as % of (On-farm + Upstream)", fontsize=12)
        plt.ylabel("Number of countries", fontsize=12)
        plt.title("Distribution of overlap removed by union-of-protection", fontsize=16, pad=12)
        plt.grid(alpha=0.25)
        utilities.savefig(os.path.join(paths.output.figure_directory, "hist_overlap_pct_of_sum.png"), dpi=300)
    
    
    # =============================================================================
    # 6) SCATTERS
    # =============================================================================
    
    # 6.1 Combined GEP vs Crop GPV
    if {"crop_gpv_const2019_2019_million", "gep_const2019_usd_combined_million"}.issubset(df.columns):
        m = (
            np.isfinite(df["crop_gpv_const2019_2019"]) &
            np.isfinite(df["gep_const2019_usd_combined"]) &
            (df["crop_gpv_const2019_2019"] > 0) &
            (df["gep_const2019_usd_combined"] > 0)
        )
        d = df[m].copy()
    
        plt.figure(figsize=(9, 7))
        plt.scatter(
            d["crop_gpv_const2019_2019_million"],
            d["gep_const2019_usd_combined_million"],
            s=18
        )
        plt.xlabel(f"Crop GPV ({p.erosion_money_unit_label})", fontsize=12)
        plt.ylabel(f"Combined GEP ({p.erosion_money_unit_label})", fontsize=12)
        plt.title("Combined GEP vs Crop GPV (log-log)", fontsize=16, pad=12)
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(alpha=0.25)
        utilities.savefig(os.path.join(paths.output.figure_directory, "scatter_combined_gep_vs_crop_gpv_loglog_2019usd_million.png"), dpi=300)
    
    # 6.2 Combined GEP vs GDP with labels
    if {"gdp_const2019_2019", "gep_const2019_usd_combined"}.issubset(df.columns):
        d = df.copy()
        mask = (
            np.isfinite(d["gdp_const2019_2019"]) &
            np.isfinite(d["gep_const2019_usd_combined"]) &
            (d["gdp_const2019_2019"] > 0) &
            (d["gep_const2019_usd_combined"] > 0)
        )
        d = d.loc[mask].copy()
    
        if len(d) > 0:
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.scatter(
                d["gdp_const2019_2019"],
                d["gep_const2019_usd_combined"],
                s=28,
                alpha=0.75,
                edgecolors="white",
                linewidths=0.3
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("GDP 2019 (USD, log scale)")
            ax.set_ylabel("Combined GEP (USD, log scale)")
            ax.set_title("Combined GEP vs GDP (log-log)")
    
            label_subset = d.sort_values("gep_const2019_usd_combined", ascending=False).head(p.erosion_top_n_labels)
            for _, r in label_subset.iterrows():
                ax.text(
                    r["gdp_const2019_2019"] * 1.03,
                    r["gep_const2019_usd_combined"] * 1.03,
                    str(r["country_name"])[:18],
                    fontsize=7,
                    color="gray",
                    alpha=0.85
                )
    
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            diag_min = max(xmin, ymin)
            diag_max = min(xmax, ymax)
            if diag_max > diag_min:
                ax.plot([diag_min, diag_max], [diag_min, diag_max], linestyle="--", linewidth=0.8, color="black", alpha=0.4)
    
            utilities.savefig(os.path.join(paths.output.figure_directory, "scatter_combined_gep_vs_gdp_log_countrynames.png"), dpi=300)
    
    # 6.3 Income group scatter plots
    
    
    if {"gdp_const2019_2019", "gep_const2019_usd_combined"}.issubset(df.columns):
        d0, income_order = utilities.attach_income_group(df.copy(), p.df_countries)
        n_unlabelled = int(d0["income_group"].isna().sum())
        if n_unlabelled:
            hb.log("%d of %d countries have no income group and are left out of the "
                   "income-group figures." % (n_unlabelled, len(d0)))
        d0 = d0.dropna(subset=["income_group"]).copy()
    
        mask = (
            np.isfinite(d0["gdp_const2019_2019"]) &
            np.isfinite(d0["gep_const2019_usd_combined"]) &
            (d0["gdp_const2019_2019"] > 0) &
            (d0["gep_const2019_usd_combined"] > 0)
        )
        d = d0.loc[mask].copy()
    
        if len(d) > 0:
            order = income_order
            income_colors = utilities.income_group_colors(order)
    
            # Log-log
            fig, ax = plt.subplots(figsize=(10, 7))
            for group in order:
                subset = d[d["income_group"] == group]
                if subset.empty:
                    continue
                ax.scatter(
                    subset["gdp_const2019_2019"],
                    subset["gep_const2019_usd_combined"],
                    s=30,
                    alpha=0.78,
                    edgecolors="white",
                    linewidths=0.4,
                    label=group,
                    color=income_colors.get(group, "gray")
                )
    
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("GDP 2019 (USD, log scale)")
            ax.set_ylabel("Combined GEP (USD, log scale)")
            ax.set_title("Combined GEP vs GDP (log-log), by income group")
    
            label_subset = d.sort_values("gep_const2019_usd_combined", ascending=False).head(p.erosion_top_n_labels)
            for _, r in label_subset.iterrows():
                ax.text(
                    r["gdp_const2019_2019"] * 1.03,
                    r["gep_const2019_usd_combined"] * 1.03,
                    str(r["country_name"])[:18],
                    fontsize=7, color="gray", alpha=0.85
                )
    
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            diag_min = max(xmin, ymin)
            diag_max = min(xmax, ymax)
            if diag_max > diag_min:
                ax.plot([diag_min, diag_max], [diag_min, diag_max], "k--", lw=0.8, alpha=0.4)
    
            ax.legend(title="Income Group", fontsize=8, title_fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
            plt.tight_layout()
            utilities.savefig(os.path.join(paths.output.figure_directory, "scatter_combined_gep_vs_gdp_log_income_groups.png"), dpi=300, bbox_inches="tight")
            plt.close()
    
            # Linear capped
            CAP_AT_PCTL = 99
            x_cap = np.nanpercentile(d["gdp_const2019_2019"], CAP_AT_PCTL)
            y_cap = np.nanpercentile(d["gep_const2019_usd_combined"], CAP_AT_PCTL)
    
            fig, ax = plt.subplots(figsize=(12, 7))
            for group in order:
                subset = d[d["income_group"] == group]
                if subset.empty:
                    continue
                ax.scatter(
                    subset["gdp_const2019_2019"],
                    subset["gep_const2019_usd_combined"],
                    s=30,
                    alpha=0.78,
                    edgecolors="white",
                    linewidths=0.4,
                    label=group,
                    color=income_colors.get(group, "gray")
                )
    
            ax.set_xlabel("GDP 2019 (USD, linear)")
            ax.set_ylabel("Combined GEP (USD, linear)")
            ax.set_title(f"Combined GEP vs GDP (linear), by income group (axes capped at p{CAP_AT_PCTL})")
            ax.set_xlim(0, x_cap)
            ax.set_ylim(0, y_cap)
    
            label_subset = d.sort_values("gep_const2019_usd_combined", ascending=False).head(p.erosion_top_n_labels)
            for _, r in label_subset.iterrows():
                if r["gdp_const2019_2019"] <= x_cap and r["gep_const2019_usd_combined"] <= y_cap:
                    ax.text(
                        r["gdp_const2019_2019"] * 1.01,
                        r["gep_const2019_usd_combined"] * 1.01,
                        str(r["country_name"])[:18],
                        fontsize=7, color="gray", alpha=0.85
                    )
    
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            diag_min = max(xmin, ymin)
            diag_max = min(xmax, ymax)
            if diag_max > diag_min:
                ax.plot([diag_min, diag_max], [diag_min, diag_max], "k--", lw=0.8, alpha=0.4)
    
            ax.legend(title="Income Group", fontsize=8, title_fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
            fig.subplots_adjust(right=0.78)
            utilities.savefig(os.path.join(paths.output.figure_directory, "scatter_combined_gep_vs_gdp_linear_income_groups.png"), dpi=300, bbox_inches="tight")
            plt.close()
    
    
    # =============================================================================
    # 7) GLOBAL CROP-LEVEL FIGURE
    # =============================================================================
    if df_crop_long is not None:
        needed = {"component", "crop", "protected_production_tons"}
        if needed.issubset(df_crop_long.columns):
            dcc = df_crop_long.copy()
            dcc["protected_production_tons"] = pd.to_numeric(dcc["protected_production_tons"], errors="coerce")
    
            dcc_comb = dcc[dcc["component"].astype(str).str.lower() == "combined"].copy()
            top_crop = (
                dcc_comb.groupby("crop", as_index=False)["protected_production_tons"]
                .sum()
                .sort_values("protected_production_tons", ascending=False)
                .head(p.erosion_top_n)
                .copy()
            )
    
            if len(top_crop) > 0:
                top_crop = top_crop.sort_values("protected_production_tons", ascending=True)
    
                plt.figure(figsize=(11, 8))
                plt.barh(top_crop["crop"], top_crop["protected_production_tons"])
                plt.xlabel("Protected production (tons)")
                plt.title(f"Top {p.erosion_top_n} crops by nature protected production (combined)", fontsize=16, pad=12)
                plt.grid(axis="x", alpha=0.25)
                utilities.savefig(os.path.join(paths.output.figure_directory, "bar_top20_crops_protected_tons_combined.png"), dpi=300)
    
    
    # =============================================================================
    # 8) CHOROPLETH MAPS
    # =============================================================================
    
    # Monetary maps
    utilities.plot_publication_choropleth_categorical(
        g, "gep_const2019_usd_combined",
        "Combined GEP from severe erosion protection",
        os.path.join(paths.output.figure_directory, "map1_country_combined_gep_5class_2019usd_million.png"),
        f"Combined GEP ({p.erosion_money_unit_label})",
        scheme="fisher_jenks", k=p.erosion_map_k_classes, value_unit="usd_millions", label_format="usd_millions"
    )
    
    utilities.plot_publication_choropleth_categorical(
        g, "gep_const2019_usd_onfarm",
        "On-farm GEP from severe erosion protection",
        os.path.join(paths.output.figure_directory, "map2_country_onfarm_gep_5class_2019usd_million.png"),
        f"On-farm GEP ({p.erosion_money_unit_label})",
        scheme="fisher_jenks", k=p.erosion_map_k_classes, value_unit="usd_millions", label_format="usd_millions"
    )
    
    utilities.plot_publication_choropleth_categorical(
        g, "gep_const2019_usd_upstream",
        "Upstream GEP from severe erosion protection",
        os.path.join(paths.output.figure_directory, "map3_country_upstream_gep_5class_2019usd_million.png"),
        f"Upstream GEP ({p.erosion_money_unit_label})",
        scheme="fisher_jenks", k=p.erosion_map_k_classes, value_unit="usd_millions", label_format="usd_millions"
    )
    
    utilities.plot_publication_choropleth_categorical(
        g, "gep_const2019_usd_overlap",
        "Overlap removed = On-farm + Upstream - Combined",
        os.path.join(paths.output.figure_directory, "map4_country_overlap_5class_2019usd_million.png"),
        f"Overlap ({p.erosion_money_unit_label})",
        scheme="fisher_jenks", k=p.erosion_map_k_classes, value_unit="usd_millions", label_format="usd_millions"
    )
    
    utilities.plot_publication_choropleth_categorical(
        g, "crop_gpv_const2019_2019",
        "Total crop production value (FAO 2019)",
        os.path.join(paths.output.figure_directory, "map5_country_crop_gpv_5class_2019usd_million.png"),
        f"Crop GPV ({p.erosion_money_unit_label})",
        scheme="fisher_jenks", k=p.erosion_map_k_classes, value_unit="usd_millions", label_format="usd_millions"
    )
    
    # Shares / percentages
    utilities.plot_publication_choropleth_categorical(
        g, "overlap_pct_of_sum_components",
        "Overlap as % of (On-farm + Upstream)",
        os.path.join(paths.output.figure_directory, "map6_country_overlap_pct_5class.png"),
        "Overlap (% of On-farm + Upstream)",
        scheme="equal_interval", k=p.erosion_map_k_classes, value_unit="raw", label_format="percent"
    )
    
    utilities.plot_publication_choropleth_categorical(
        g, "gdp_loss_pct_combined",
        "Combined GEP as % of GDP (indicative macro exposure)",
        os.path.join(paths.output.figure_directory, "map7_country_gdp_loss_pct_combined_5class.png"),
        "Combined GEP / GDP (%)",
        scheme="equal_interval", k=p.erosion_map_k_classes, value_unit="raw", label_format="percent"
    )
    
    utilities.plot_publication_choropleth_categorical(
        g, "share_protected_production_combined",
        "Share of protected production (combined)",
        os.path.join(paths.output.figure_directory, "map8_country_share_protected_combined_5class.png"),
        "Share protected production",
        scheme="equal_interval", k=p.erosion_map_k_classes, value_unit="raw", label_format="percent"
    )
    
    utilities.plot_publication_choropleth_categorical(
        g, "share_protected_production_onfarm",
        "Share of protected production (on-farm)",
        os.path.join(paths.output.figure_directory, "map9_country_share_protected_onfarm_5class.png"),
        "Share protected production",
        scheme="equal_interval", k=p.erosion_map_k_classes, value_unit="raw", label_format="percent"
    )
    
    utilities.plot_publication_choropleth_categorical(
        g, "share_protected_production_upstream",
        "Share of protected production (upstream)",
        os.path.join(paths.output.figure_directory, "map10_country_share_protected_upstream_5class.png"),
        "Share protected production",
        scheme="equal_interval", k=p.erosion_map_k_classes, value_unit="raw", label_format="percent"
    )
    
    utilities.plot_publication_choropleth_categorical(
        g, "erosion_shock_share_combined",
        "Erosion shock share (combined)",
        os.path.join(paths.output.figure_directory, "map11_country_erosion_shock_share_combined_5class.png"),
        "Shock share",
        scheme="equal_interval", k=p.erosion_map_k_classes, value_unit="raw", label_format="percent"
    )
    
    # Mean PS maps
    utilities.plot_publication_choropleth_categorical(
        g, "mean_ps_onfarm_cropland_severe",
        "Mean prevention share on cropland severe pixels (on-farm)",
        os.path.join(paths.output.figure_directory, "map12_country_mean_ps_onfarm_5class.png"),
        "Mean PS_onfarm (0–1)",
        scheme="equal_interval", k=p.erosion_map_k_classes, value_unit="raw", label_format="percent"
    )
    
    utilities.plot_publication_choropleth_categorical(
        g, "mean_ps_upstream_cropland_severe",
        "Mean prevention share on cropland severe pixels (upstream)",
        os.path.join(paths.output.figure_directory, "map13_country_mean_ps_upstream_5class.png"),
        "Mean PS_upstream (0–1)",
        scheme="equal_interval", k=p.erosion_map_k_classes, value_unit="raw", label_format="percent"
    )
    
    utilities.plot_publication_choropleth_categorical(
        g, "mean_ps_combined_cropland_severe",
        "Mean prevention share on cropland severe pixels (combined)",
        os.path.join(paths.output.figure_directory, "map14_country_mean_ps_combined_5class.png"),
        "Mean PS_combined (0–1)",
        scheme="equal_interval", k=p.erosion_map_k_classes, value_unit="raw", label_format="percent"
    )
    
    # Log10 combined GEP map
    if "gep_const2019_usd_combined" in g.columns:
        g_log = g.copy()
        g_log["log10_gep_million_usd_combined"] = np.log10(
            (pd.to_numeric(g_log["gep_const2019_usd_combined"], errors="coerce") / p.erosion_usd_to_millions)
            .where(pd.to_numeric(g_log["gep_const2019_usd_combined"], errors="coerce") > 0)
        )
        utilities.plot_publication_choropleth_categorical(
            g_log, "log10_gep_million_usd_combined",
            "Combined GEP (log10 USD million)",
            os.path.join(paths.output.figure_directory, "map15_country_log10_combined_gep_5class.png"),
            "log10(USD million)",
            scheme="fisher_jenks", k=p.erosion_map_k_classes, value_unit="raw", label_format="percent"
        )
    
    
    # =============================================================================
    # 9) RASTER PREVIEWS
    # =============================================================================
    if hb.path_exists(paths.output.prevention_share_onfarm):
        plot_raster_global(
            paths.output.prevention_share_onfarm,
            "PS_onfarm on cropland & severe",
            os.path.join(paths.output.figure_directory, "raster1_ps_onfarm_cropland_severe.png"),
            downsample_factor=p.erosion_raster_downsample_factor,
        )
    
    if hb.path_exists(paths.output.prevention_share_upstream):
        plot_raster_global(
            paths.output.prevention_share_upstream,
            "PS_upstream on cropland & severe",
            os.path.join(paths.output.figure_directory, "raster2_ps_upstream_cropland_severe.png"),
            downsample_factor=p.erosion_raster_downsample_factor,
        )
    
    if hb.path_exists(paths.output.prevention_share_combined):
        plot_raster_global(
            paths.output.prevention_share_combined,
            "PS_combined (union-of-protection) on cropland & severe",
            os.path.join(paths.output.figure_directory, "raster3_ps_combined_union_cropland_severe.png"),
            downsample_factor=p.erosion_raster_downsample_factor,
        )
    
    
    # =============================================================================
    # 10) SUMMARY
    # =============================================================================
    hb.log(f"✅ Done. Figures saved to: {paths.output.figure_directory}")
    hb.log("Created files:")
    for fp in sorted(glob.glob(os.path.join(paths.output.figure_directory, "*"))):
        if os.path.splitext(fp)[1].lower() in {".png", ".csv"}:
            hb.log(" -", os.path.basename(fp))


# The files Section B writes. Each name is spelled once, so a rename cannot be applied to the
# writer and missed in the manifest that claims to record what ran.

def publish_inputs(p):
    """Every task's first line: erosion's es_config row plus its es_parameters block (the SDR
    data references, the SES-11 threshold policy, the crop-sector export map, the blank
    erosion_gep_root machine key the configure_* functions read) and the results registry.
    Defaults layer throughout: anything the caller set wins."""
    utilities.hydrate_es_config(p, 'erosion', log=hb.log)
    utilities.hydrate_es_parameters(p, 'erosion', log=hb.log)
    # The country table every service collapses onto at the end. gep_calculation reads it through
    # utilities.country_attributes, which expects p.df_countries, so it is published here with the
    # rest of the inputs exactly as the other twenty services do.
    utilities.initialize_country_paths(p)
    # Derived from the DEM row (caller wins): configure_* in erosion_functions reads these, and
    # their own fallbacks point at the author's cluster layout -- without these lines a local
    # Section-A run would silently reach for MSI paths.
    if getattr(p, 'erosion_sdr_input_dir', None) is None:
        p.erosion_sdr_input_dir = os.path.dirname(p.erosion_dem_path)
    if getattr(p, 'erosion_elevation_path', None) is None:
        p.erosion_elevation_path = p.erosion_dem_path
    # Optional: a blank es_parameters cell means the run does not use one, and the readers branch
    # on None rather than on a value that happens not to resolve.
    for optional in ('erosion_drainage_path', 'erosion_boundary_source_epsg'):
        setattr(p, optional, getattr(p, optional, None) or None)
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def erosion_sdr(p):
    """DYNAMIC step 1: per (scenario, anchor year), resample the SEALS map to the erosion analysis
    grid (erosion_native_resolution row: false -> 6.45 km reference grid, true -> native SEALS 300 m) and run
    InVEST SDR. Outputs per map, in p.cur_dir/<scn>_<yr>/: usle_<scn>_<yr>.tif (actual erosion) and
    rkls_<scn>_<yr>.tif (potential/bare soil); avoided = rkls - usle is formed downstream.

    Caller sets on p: scenario_lulc_paths {scenario: {year: seals_lulc_path}}; erosion_dem_path,
    erosion_erosivity_path, erosion_erodibility_path, erosion_watersheds_path,
    erosion_biophysical_table_path (SEALS7 lucode -> usle_c/usle_p); erosion_analysis_grid_path
    (6.45 km reference raster; local only). SDR knobs via p.erosion_sdr_params (defaults below).
    """
    publish_inputs(p)
    # Published before the run_this guard: a skipped task (skip_existing with the dir already there)
    # still needs downstream to find these outputs, and a task body that returns early would leave
    # the attribute unset and fail the next task with an AttributeError.
    p.erosion_sdr_dir = p.cur_dir      # downstream tasks read usle_/rkls_/avoided_erosion_ from here
    if not p.run_this:
        return
    import hazelbean as hb
    from natcap.invest.sdr import sdr

    # Build scenario_lulc_paths from a template if the caller didn't (mirrors carbon/pollination).
    # p.es_lulc_path_template uses {scenario} and {year}; include the base scenario for differencing.
    if not getattr(p, 'scenario_lulc_paths', None) and getattr(p, 'es_lulc_path_template', None):
        tmpl = p.es_lulc_path_template
        years = [int(y) for y in getattr(p, 'es_shock_years', [])]
        scens = list(getattr(p, 'es_shock_scenarios', []))
        base = utilities.required_base_scenario(p, 'erosion')
        if base not in scens:
            scens = scens + [base]
        p.scenario_lulc_paths = {}
        for scn in scens:
            yr_map = {y: sorted(glob.glob(tmpl.format(scenario=scn, year=y)))[0]
                      for y in years if glob.glob(tmpl.format(scenario=scn, year=y))}
            if yr_map:
                p.scenario_lulc_paths[scn] = yr_map

    # analysis grid: downsample to a 6.45 km reference for local; run at native SEALS res on the cluster
    native = bool(p.erosion_native_resolution)   # false -> the 6.45 km analysis grid; true -> native SEALS 300 m
    grid_ref = None if native else p.get_path(p.erosion_analysis_grid_path)
    dem         = p.get_path(p.erosion_dem_path)
    erosivity   = p.get_path(p.erosion_erosivity_path)
    erodibility = p.get_path(p.erosion_erodibility_path)
    # Repaired once per run and cached: SDR's report step unions these and GEOS raises on an invalid
    # ring, so a bad geometry kills the run AFTER the rasters are already computed.
    watersheds = os.path.join(p.cur_dir, 'watersheds_valid.gpkg')
    if not hb.path_exists(watersheds):
        repair_watersheds(p.get_path(p.erosion_watersheds_path), watersheds)
    # SDR matches the biophysical table's lucode against the LULC values, and our maps are SEALS7 while
    # the shipped table is keyed on ESA codes -- so re-key it (once) rather than matching nothing.
    biophysical = build_seals7_biophysical_table(
        p.get_path(p.erosion_biophysical_table_path),
        os.path.join(p.cur_dir, 'biophysical_table_seals7.csv'))
    # The same rows Section A reads, so the two paths cannot drift apart again.
    sdr_params = dict(threshold_flow_accumulation=p.erosion_sdr_threshold_flow_accumulation,
                      k_param=p.erosion_sdr_k_param, sdr_max=p.erosion_sdr_max,
                      ic_0_param=p.erosion_sdr_ic_0_param, l_max=p.erosion_sdr_l_max,
                      flow_dir_algorithm=p.erosion_sdr_flow_dir_algorithm,
                      n_workers=p.erosion_sdr_n_workers)
    sdr_params.update(getattr(p, 'erosion_sdr_params', {}))

    n = 0
    for scenario, by_year in p.scenario_lulc_paths.items():
        for year, lulc in by_year.items():
            lulc = p.get_path(lulc)
            if native:
                lulc_grid = lulc
            else:
                lulc_grid = os.path.join(p.cur_dir, 'lulc_%s_%d_grid.tif' % (scenario, year))
                if not hb.path_exists(lulc_grid):  # categorical LULC -> mode
                    hb.resample_to_match(lulc, grid_ref, lulc_grid, resample_method='mode')
            suffix = '%s_%d' % (scenario, year)
            sdr.execute(dict(workspace_dir=os.path.join(p.cur_dir, suffix), results_suffix=suffix,
                             dem_path=dem, erosivity_path=erosivity, erodibility_path=erodibility,
                             lulc_path=lulc_grid, watersheds_path=watersheds,
                             biophysical_table_path=biophysical, **sdr_params))
            n += 1
    hb.log('  erosion SDR: %d scenario x year maps (%s grid) -> usle_/rkls_ in %s'
          % (n, 'native SEALS' if native else '6.45 km', p.cur_dir))
    return True


def erosion_upstream(p):
    """DYNAMIC step 2: per (scenario, year), upstream prevention share = acc(avoided) / acc(rkls),
    D8 flow-accumulation of avoided-mass over potential-mass (the pixel-area weight cancels in the
    ratio). Recomputed per scenario because the upslope land cover changes. Reads the SDR outputs
    from p.erosion_sdr_dir; writes upstream_<scn>_<yr>.tif to p.cur_dir. Uses pygeoprocessing.routing
    (fill_pits -> flow_dir_d8 -> flow_accumulation_d8), verified equal to a hand-written D8.

    Caller sets on p: erosion_dem_path (aligned to the SDR grid here) + the step-1 outputs.
    """
    publish_inputs(p)
    # Published before the run_this guard, as in steps 1 and 3: a skipped task must still tell the
    # exposure task where its rasters are, or a resumed run dies on an AttributeError.
    p.erosion_upstream_dir = p.cur_dir
    if not p.run_this:
        return
    dem = p.get_path(p.erosion_dem_path)

    n = 0
    for scenario, by_year in p.scenario_lulc_paths.items():
        for year in by_year:
            suffix = '%s_%d' % (scenario, year)
            sdr_dir = os.path.join(p.erosion_sdr_dir, suffix)
            accumulate_upstream_prevention_share(
                dem,
                os.path.join(sdr_dir, 'avoided_erosion_%s.tif' % suffix),
                os.path.join(sdr_dir, 'rkls_%s.tif' % suffix),
                os.path.join(p.cur_dir, suffix),
                os.path.join(p.cur_dir, 'upstream_%s.tif' % suffix))
            n += 1
    hb.log('  erosion upstream: %d maps -> upstream_<scn>_<yr>.tif in %s' % (n, p.cur_dir))
    return True


def erosion_exposure(p):
    """DYNAMIC step 3: per (scenario, year), turn the SDR outputs into the pixel fields the level
    functions consume, on the equal-area analysis grid.

    Reads usle/avoided (p.erosion_sdr_dir) and upstream (p.erosion_upstream_dir). On-farm PS =
    avoided/(avoided+usle), which is identically 1 - USLE/RKLS; combined = 1 - (1-onfarm)(1-upstream),
    the serial-filter union (a tonne must escape both on-site and downslope retention to be lost).
    Writes six rasters:
      ps_gated              combined, zeroed off severe pixels    -> B-thresholded (the default)
      ps_continuous         combined across all land              -> B
      rkls_grid             potential (bare-soil) erosion         -> the service methods' weight
      cropland_frac         SEALS cropland fraction               -> method A denominator
      severe_cropland_frac  the same, zeroed off severe pixels    -> method A numerator
      severe_mask           the severe gate itself, so a level function can restrict BOTH halves of
                            a ratio to it. Gating only the numerator measures how much severe erosion
                            a zone HAS rather than how well it is protected.

    No cropland restriction is applied here. It is deferred to the shock task, where SPAM production
    is zero off cropland and multiplies through every term, so non-cropland drops out on its own and
    a binary mask would add nothing.

    The severe threshold follows the per-country SES-11 policy (T=2 for small-area <50,000 km2 or
    low-elevation <250 m countries, else 11) when p.erosion_country_boundary_path (+ p.erosion_dem_path
    for the elevation rule) is set; otherwise flat p.erosion_severe_threshold_t_ha.
    """
    publish_inputs(p)
    # Published before the run_this guard, for the same reason as step 1: a skipped task must still
    # tell the shock task where its rasters are.
    p.erosion_exposure_dir = p.cur_dir
    if not p.run_this:
        return
    import numpy as np
    import rioxarray as rxr
    import pygeoprocessing as pgp
    from osgeo import gdal
    from rasterio.crs import CRS as rioCRS
    from rasterio.enums import Resampling
    from global_invest.erosion import erosion_functions
    from global_invest.erosion import erosion_functions as ef

    thresh_high = float(p.erosion_threshold_high_t_ha_yr)
    analysis_crs = rioCRS.from_epsg(int(p.erosion_analysis_epsg))

    def _to_grid_da(da, template=None):    # reproject an open DataArray to the equal-area analysis grid
        da = da.rio.reproject(analysis_crs, resampling=Resampling.average)
        return da if template is None else da.rio.reproject_match(template, resampling=Resampling.average)

    def _to_grid(path, template=None):
        return _to_grid_da(rxr.open_rasterio(path, masked=True).squeeze(), template)

    n = 0
    for scenario, by_year in p.scenario_lulc_paths.items():
        for year in by_year:
            suffix = '%s_%d' % (scenario, year)
            sdr_dir = os.path.join(p.erosion_sdr_dir, suffix)
            # PS is computed ON the analysis grid (reproject usle/avoided/ups FIRST, then PS) -- the
            # order matters because PS is nonlinear; computing it on the native grid then reprojecting
            # biases the shock high.
            usle = _to_grid(os.path.join(sdr_dir, 'usle_%s.tif' % suffix))
            avoided = _to_grid(os.path.join(sdr_dir, 'avoided_erosion_%s.tif' % suffix), usle)
            ups = _to_grid(os.path.join(p.erosion_upstream_dir, 'upstream_%s.tif' % suffix), usle)
            usle_v = np.nan_to_num(np.maximum(usle.values, 0.0))
            avoided_v = np.nan_to_num(np.maximum(avoided.values, 0.0))
            ups_v = np.clip(np.nan_to_num(ups.values), 0.0, 1.0)

            # per-country severe threshold (SES-11 policy: T=2 for small-area/low-elevation countries,
            # else thresh_high). Computed once on the analysis grid (same across scenarios), cached.
            thr = getattr(p, '_erosion_threshold_raster', None)
            if thr is None:
                cb = getattr(p, 'erosion_country_boundary_path', None)
                if cb:
                    thr = build_severe_threshold_raster(
                        usle, p.get_path(cb),
                        p.get_path(p.erosion_dem_path) if getattr(p, 'erosion_dem_path', None) else None,
                        thresh_high=thresh_high,
                        thresh_low=float(p.erosion_threshold_low_t_ha_yr),
                        small_area_km2=float(p.erosion_small_country_area_km2),
                        low_elevation_mean_m=float(p.erosion_low_elevation_mean_m))
                else:
                    thr = thresh_high        # flat fallback when no country boundary is provided
                p._erosion_threshold_raster = thr
            mask = usle_v > thr              # severe pixels (per-country T); cropland from SPAM in the shock task
            with np.errstate(invalid='ignore', divide='ignore'):
                onfarm = np.where(mask & (avoided_v + usle_v > 0), avoided_v / (avoided_v + usle_v), 0.0)
            combined = np.where(mask, 1.0 - (1.0 - onfarm) * (1.0 - ups_v), 0.0)

            # Method B needs the same prevention WITHOUT the severe gate. avoided/(avoided+usle) is
            # identically avoided/rkls = 1 - USLE/RKLS, so this is the continuous prevention fraction the
            # A/B framing calls for; rkls is kept as B's magnitude weight, so that preventing most of a
            # negligible erosion rate earns almost no credit.
            with np.errstate(invalid='ignore', divide='ignore'):
                onfarm_cont = np.where(avoided_v + usle_v > 0, avoided_v / (avoided_v + usle_v), 0.0)
            continuous = 1.0 - (1.0 - onfarm_cont) * (1.0 - ups_v)
            rkls_v = avoided_v + usle_v      # potential (bare-soil) erosion

            # Method A weights by CROPLAND AREA (SEALS7 class 2), not SPAM production: p_crop is the
            # severe share of a zone's cropland. Averaging a 0/1 cropland mask onto the analysis grid
            # gives the cropland fraction per cell, and the equal-area grid makes cell area cancel.
            #
            # Done BLOCK-WISE, never in memory: a global 300 m SEALS map is ~8.4e9 pixels, so building
            # the mask as a float32 array would need ~34 GB and gets the run OOM-killed. raster_calculator
            # streams it by block to a compressed byte raster, then the average-resample coarsens it.
            lulc_native = p.get_path(by_year[year])
            crop_mask = os.path.join(p.cur_dir, 'cropland_mask_%s.tif' % suffix)
            if not hb.path_exists(crop_mask):
                _nodata = pgp.get_raster_info(lulc_native)['nodata'][0]
                pgp.raster_calculator(
                    [(lulc_native, 1)],
                    lambda a: (a == CROPLAND_SEALS7_CLASS).astype('uint8'),
                    crop_mask, gdal.GDT_Byte, 255,
                    raster_driver_creation_tuple=('GTIFF', (
                        'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=DEFLATE', 'PREDICTOR=2')))
            cropfrac = np.nan_to_num(_to_grid(crop_mask, usle).values)

            tr = usle.rio.transform(); px = usle.rio.resolution()

            def _write(arr, name):
                pgp.numpy_array_to_raster(arr.astype('float32'), -9999.0, (px[0], px[1]),
                                          (tr.c, tr.f), usle.rio.crs.to_wkt(),
                                          os.path.join(p.cur_dir, '%s_%s.tif' % (name, suffix)))

            _write(combined, 'ps_gated')            # threshold-gated (original candidate)
            _write(continuous, 'ps_continuous')        # threshold-free (method B)
            _write(rkls_v, 'rkls_grid')                # method B magnitude weight
            _write(cropfrac, 'cropland_frac')          # method A denominator
            _write(np.where(mask, cropfrac, 0.0), 'severe_cropland_frac')   # method A numerator

            # The severe gate itself, so a level function can restrict BOTH halves of a ratio to it.
            # A severe pixel can legitimately have zero protection, so this cannot be recovered by
            # testing ps_gated > 0.
            _write(mask.astype('float32'), 'severe_mask')
            n += 1
    per_country = getattr(p, 'erosion_country_boundary_path', None) is not None
    hb.log('  erosion prevention: %d maps -> ps_gated_ on EPSG:%d (severe T=%s)'
          % (n, analysis_crs.to_epsg(), 'per-country 11/2' if per_country else '%.1f flat' % thresh_high))
    return True


def erosion_paths(p):
    """Every path erosion needs, resolved from the project rather than hardcoded.

    The module used to carry these as constants built from a `ROOT` that named someone else's
    machine, so three tasks in the tree could only run on the machine the code was written on.
    Inputs now resolve through `p.get_path` against base data and outputs land under the task's
    own directory, which is what lets the same code run anywhere.

    Args:
        p (ProjectFlow): the project, after publish_inputs has set the es_parameters values.

    Returns:
        SimpleNamespace: `.input` for what the run reads and `.output` for what it writes. Every
        value is a path string, joined with os.path.join like the rest of the library.
    """
    from types import SimpleNamespace
    # Every path below names something prevention_shares produces, so the directory is that
    # task's, not the caller's. prevention_shares publishes it as erosion_gep_output_dir before
    # its own run_this check, so a later task sees it whether that task ran or skipped. Reading
    # p.cur_dir here instead sends a consumer looking in its own directory, where a producer
    # never wrote: gep_calculation raised on a missing integrated_country_gep.csv that had been
    # written seven hours earlier, one directory across.
    out_dir = str(getattr(p, 'erosion_gep_output_dir', None) or p.cur_dir)
    hb.create_directories([str(out_dir)])
    # The figures are the other way round. Every one of them is written by
    # generate_all_maps_and_figures and read by nothing, so they belong to the task that draws
    # them rather than to the task whose numbers they draw. Hanging them off out_dir put
    # maps_and_figures' output inside prevention_shares' directory.
    figure_dir = os.path.join(str(p.cur_dir), 'figures')
    hb.create_directories([str(figure_dir)])

    def published(attribute, default_name):
        """Where an earlier task said it wrote a file, or this task's directory if none did.

        Several of these are produced by one task and read by another: upstream_prevention_share
        is written by its own task and consumed by prevention_shares. The producer publishes the
        path on p before its run_this guard, so the consumer must read that rather than assume the
        file sits in its own directory, which is what made the first run fail.
        """
        value = getattr(p, attribute, None)
        return str(value) if value else os.path.join(out_dir, default_name)

    return SimpleNamespace(
        input=SimpleNamespace(
            biophysical_table=str(p.get_path(p.erosion_biophysical_table_path)),
            dem=str(p.get_path(p.erosion_dem_path)),
            lulc=str(p.get_path(p.erosion_lulc_path)),
            erodibility=str(p.get_path(p.erosion_erodibility_path)),
            erosivity=str(p.get_path(p.erosion_erosivity_path)),
            watersheds=str(p.get_path(p.erosion_watersheds_path)),
            # Section A writes these and Section B reads them, so they are resolved the way the
            # upstream share is: whatever invest_sdr published, without requiring it to exist
            # yet. Resolving them through get_path made invest_sdr demand the file it was about
            # to create, which is why the first cluster run died before doing any work.
            usle=published('erosion_usle_path', 'usle.tif'),
            avoided_erosion=published('erosion_avoided_erosion_path', 'avoided_erosion.tif'),
            country_boundary=str(p.get_path(p.erosion_country_boundary_path)),
            yield_stack=str(p.get_path(p.erosion_yield_stack_path)),
            area_stack=str(p.get_path(p.erosion_area_stack_path)),
            bandmap=str(p.get_path(p.erosion_bandmap_csv_path)),
            elasticity=str(p.get_path(p.erosion_elasticity_csv_path)),
            fao_gpv=str(p.get_path(p.erosion_fao_gpv_iso3_csv_path)),
            fao_prices=str(p.get_path(p.erosion_fao_prices_csv_path)),
            gdp=str(p.get_path(p.erosion_gdp_csv_path)),
        ),
        output=SimpleNamespace(
            directory=out_dir,
            figure_directory=figure_dir,
            watersheds_sanitized=published('erosion_watersheds_sanitized_path',
                                           'watersheds_sanitized.gpkg'),
            upstream_prevention_share=published('erosion_upstream_prevention_share_path',
                                                'upstream_prevention_share.tif'),
            upslope_forest_share=published('erosion_upslope_forest_share_path',
                                           'upslope_forest_share.tif'),
            upslope_grass_share=published('erosion_upslope_grass_share_path',
                                          'upslope_grass_share.tif'),
            upslope_cropland_share=published('erosion_upslope_cropland_share_path',
                                             'upslope_cropland_share.tif'),
            upslope_bare_share=published('erosion_upslope_bare_share_path',
                                         'upslope_bare_share.tif'),
            area_conservation_audit=os.path.join(out_dir, "area_conservation_audit.csv"),
            country_gep_decomposition_long=os.path.join(out_dir, "country_gep_decomposition_long.csv"),
            country_ps_diagnostics=os.path.join(out_dir, "country_ps_diagnostics.csv"),
            elasticity_audit=os.path.join(out_dir, "elasticity_audit.csv"),
            erosion_interpolated=os.path.join(out_dir, "erosion_interpolated.csv"),
            gpv_fallback_diagnostic=os.path.join(out_dir, "gpv_fallback_diagnostic.csv"),
            manifest=os.path.join(out_dir, "manifest.json"),
            threshold_policy=os.path.join(out_dir, "threshold_policy.csv"),
            prevention_share_onfarm=os.path.join(out_dir, "ps_onfarm_cropland_severe.tif"),
            prevention_share_upstream=os.path.join(out_dir, "ps_upstream_cropland_severe.tif"),
            prevention_share_combined=os.path.join(out_dir, "ps_combined_union_cropland_severe.tif"),
            integrated_country_gep=os.path.join(out_dir, "integrated_country_gep.csv"),
            country_crop_long=os.path.join(out_dir, "country_crop_protected_production_long.csv"),
        ))


def resample_band_to_match(stack_path, band, match_path, working_dir, resample_method, src_ndv=-9999.0):
    """One band of a multi-band stack, on the grid of `match_path`, through hazelbean.

    This replaces a rioxarray `reproject_match` per band. The two are byte-identical on all
    24,669,056 pixels of the real SPAM stacks, for both the yield band (average) and the area
    band (sum), and hazelbean does it in about 2 seconds against 22.

    Two things have to be right or the result is wrong rather than merely different, and both
    were wrong on an earlier attempt at this switch:

    The method. Yield resamples by `average` and harvested area by `sum`, because a yield is a
    rate that survives being averaged over a coarser cell and an area is a quantity that has to
    be added up. Using `average` for area understated it by four orders of magnitude, and
    `bilinear` for yield moved it 5 percent, neither of which announces itself.

    The nodata. `resample_to_match` defaults the output nodata to the float32 maximum rather
    than carrying the source's, so the -9999 sentinel is resampled as though it were data. With
    `sum` that accumulates: the harvested-area total came out negative. Declaring `src_ndv` and
    `ndv` fixes it, so they are passed explicitly rather than left to a default.

    Args:
        stack_path (str): the multi-band source.
        band (int): 1-based band index.
        match_path (str): a raster whose grid the output should take.
        working_dir (str): where the single-band extract and the resampled file are written.
        resample_method (str): 'average' for a rate, 'sum' for a quantity.
        src_ndv (float): the source's nodata value.

    Returns:
        np.ndarray: float64, with nodata as 0.0, matching what the rioxarray path produced
        after its `.fillna(0.0)`.
    """
    import rasterio
    hb.create_directories([str(working_dir)])
    extracted = os.path.join(working_dir, 'band_%d_src.tif' % band)
    resampled = os.path.join(working_dir, 'band_%d_%s.tif' % (band, resample_method))
    with rasterio.open(stack_path) as src:
        profile = src.profile
        profile.update(count=1)
        with rasterio.open(extracted, 'w', **profile) as dst:
            dst.write(src.read(band), 1)
    hb.resample_to_match(extracted, match_path, resampled, resample_method=resample_method,
                         output_data_type=6, src_ndv=src_ndv, ndv=src_ndv)
    with rasterio.open(resampled) as src:
        array = src.read(1).astype('float64')
        array[array == src.nodata] = 0.0
    hb.remove_path(extracted)
    return array


def erosion_shock(p):
    """DYNAMIC step 4: per-ee_r50_aez18 crop-productivity LEVELS by three methods, reported side by side.

    All share the SDR front-end (USLE, RKLS, avoided) and all bridge erosion to yield with the same
    coefficient, so they differ only in how erosion exposure is measured:
      A ("damage")                    level = -100 * alpha * p_crop, where p_crop is the severe share
        of the zone's cropland AREA and severe = USLE > the per-country T (2/11). Thresholded, binary,
        on-farm only, one flat alpha, and necessarily uniform across erosion_shock_acts.
      B ("service", threshold-free)   level = +100 * mean over crops of alpha_crop * the prevention
        share, prevention = prevented tonnes / potential tonnes including the upstream D8 term.
        Continuous and per-crop, but composed across ALL land, which saturates it: the union
        1-(1-onfarm)(1-upstream) sits near 1 over the ~98% of land that is not severely eroding, so
        the level reaches a median of 0.9988 with about half of pixels pinned at the ceiling, where
        no improvement in land cover can register.
      B-thresholded ("service_threshold", the DEFAULT)   B confined to SEVERE pixels, with that set
        taken from the base scenario and held FIXED, so the shock measures protection change and not a
        change of population. Verified in-task on the full ZAF run: min exactly 0.0, max +1.30%, no
        negative outliers, 63 zone-erosion_shock_acts responding at 2050 with a mean of +0.157% -- roughly 2.5x
        unthresholded B, which tops out at +0.52% on the same run. A scenario-VARYING set put -18%
        into a paddy-rice zone; fixing it removed that entirely. Matches how the published account of
        this method builds it.
    A is signed negative (damage borne) and B positive (protection delivered), but BOTH increase with
    better land condition, so they are positively correlated by construction and neither is a sign
    flip of the other. They are differently shaped functions of the erosion field, not offsets of one
    another, so their difference does not cancel.
    A fourth PRESERVED level (a prevention share behind A's severe gate, weighted by production alone)
    is emitted for comparison and never fed to GTAP. Its numerator is gated while its denominator is
    not, which makes it track erosion PREVALENCE rather than protection, and inverts its orientation.
    p.erosion_method ('damage'|'service'|'service_threshold', default 'service_threshold') selects
    which becomes shock_pct, the column GTAP consumes.

    Each level is differenced ABSOLUTELY against the contemporaneous baseline (the level is already a %
    of crop productivity) and ramped 0 at es_shock_base_year through the anchors. Writes the 8-sector per-zone
    CSV at p.erosion_shock_output_path: the shared ENDW, ACTS, REG, scenario, year, shock_pct,
    shock_pct_contemp, shock_pct_fixedbase plus shock_pct_damage, shock_pct_service and
    shock_pct_service_threshold.

    Caller sets on p: scenario_lulc_paths (incl. the base scenario), es_shock_years (anchors),
    es_shock_base_year, es_shock_end_year; erosion_exposure_dir (set by step 3);
    region_boundary_path (ee_r50_aez18 correspondence gpkg with ee_r50_aez18_id, aez18_id,
    gtapv7_r50_label); erosion_yield_stack_path, erosion_area_stack_path, erosion_bandmap_csv_path,
    erosion_elasticity_csv_path; base scenario via es_shock_base_scenario. Optional: erosion_alpha,
    erosion_method.
    """
    spam_aliases = {k: v.split(';') for k, v in
                    utilities.read_lookup(p.erosion_spam_alias_path,
                                          'spam_label', 'aliases').items()}
    publish_inputs(p)
    # Default into the es_shocks parent dir. Runtime, not build time: p.es_shock_dir is
    # published by that task, which ProjectFlow runs before this one.
    if not getattr(p, 'erosion_shock_output_path', None):
        p.erosion_shock_output_path = os.path.join(getattr(p, 'es_shock_dir', None) or p.project_dir,
                                                   'erosion_interpolated.csv')
    if not p.run_this:
        return
    import numpy as np, pandas as pd, rioxarray as rxr, geopandas as gpd
    from rasterio.features import rasterize as rio_rasterize
    from global_invest.erosion import erosion_functions
    from global_invest.erosion import erosion_functions as ef

    es_shock_base_year = int(p.es_shock_base_year); es_shock_end_year = int(p.es_shock_end_year)
    base_scenario = utilities.required_base_scenario(p, 'erosion')
    fallback_coef = float(p.erosion_yield_coefficient_fallback)
    alpha = float(p.erosion_alpha)
    # Method B lets the erosion->yield coefficient vary by crop; A applies the flat alpha to all.
    # SOURCE = elasticity_crops_fao_revised.csv (already loaded above as coef_map). Despite the column
    # name, that table holds erosion-to-yield sensitivities, not price responses: its references are all
    # soil-erosion-and-yield literature (Lal 1998, Borrelli 2020, Panagos 2018, Oldfield 2019) and its
    # categories are a qualitative ranking rather than estimated coefficients. It is the crop-specific
    # version of alpha: yield lost per unit of erosion exposure (greens 0.5, cereals 0.3, default 0.08).
    # p.erosion_alpha_by_crop overrides per SPAM code.
    alpha_by_crop = dict(getattr(p, 'erosion_alpha_by_crop', {}) or {})

    # Six SPAM codes (grou, ocer, orts, pige, rest, vege) have no counterpart in the table -- they are
    # n.e.c. aggregates FAO does not carry. Falling back to the flat alpha would give "other cereals"
    # 0.08 when every named cereal in the table is 0.30, and the table-wide mean would give it 0.163;
    # its own SECTOR's mean (0.30, from barley/maize/millet) is the better estimate. So an unmatched
    # crop inherits the mean of the crops that DID match in its GTAP sector, and only a sector with no
    # matches at all falls back to the flat alpha.
    _MISS = object()

    def _table_alpha(crop):
        for key in [crop] + list(ef.SPAM_ALIAS_MAP.get(crop, [])):
            if str(key).strip().lower() in coef_map:
                return ef.get_erosion_yield_coefficient(crop, coef_map, alpha, spam_aliases)
        return _MISS

    def alpha_for(crop):
        if crop in alpha_by_crop:
            return float(alpha_by_crop[crop])
        v = _matched.get(crop, _table_alpha(crop))
        return _sector_mean[crop_sector(crop)] if v is _MISS else v

    crop_to_sector = dict(p.erosion_crop_to_sector)

    def crop_sector(crop):
        return crop_to_sector.get(crop, 'OCR')     # unmapped SPAM codes fall to other crops

    erosion_shock_acts = tuple(p.erosion_shock_acts)

    yield_stack = p.get_path(p.erosion_yield_stack_path)
    area_stack = p.get_path(p.erosion_area_stack_path)
    bandmap = hb.df_read(p.get_path(p.erosion_bandmap_csv_path))
    bcol = next(c for c in bandmap.columns if 'band' in c.lower())
    crcol = next(c for c in bandmap.columns if c.lower() in ('crop', 'crop_name', 'name'))
    coef_map = load_erosion_yield_coefficients(p.get_path(p.erosion_elasticity_csv_path))
    zones = gpd.read_file(p.get_path(p.region_boundary_path), engine='pyogrio')
    zid_col = next(c for c in zones.columns if c.lower() == 'ee_r50_aez18_id')
    aez_col = next(c for c in zones.columns if c.lower() == 'aez18_id')
    reg_col = next(c for c in zones.columns if c.lower() == 'gtapv7_r50_label')
    labels = {int(r[zid_col]): ('AEZ%d' % int(r[aez_col]), r[reg_col]) for _, r in zones.iterrows()}

    anchor_years = sorted(int(y) for y in getattr(p, 'es_shock_years', []) if int(y) > es_shock_base_year) or [es_shock_end_year]
    # The base scenario is EMITTED too, not just used as the reference. Its rows are the self-difference
    # (base - base), so they are identically 0 -- which is what carbon and pollination already write. GTAP
    # is indifferent (a missing row and an explicit zero both mean "no shock"), but writing it keeps the
    # four services on one shape and makes "B_y == 0 for the ignore-dependencies baseline" a check that can
    # actually be run against the CSV rather than inferred from an absence.
    scenarios = list(p.scenario_lulc_paths)

    # Precompute ONCE (all ps_gated rasters share the analysis grid): rasterize the zones and reproject
    # each SPAM crop's production to that grid, plus its zone totals (ps-independent, so constant across
    # scenarios). zone_level then only reads ps and does the ps-weighted bincount -- no per-(scenario,year)
    # SPAM reproject or zone re-rasterize.
    def _ps_path(scn, yr): return os.path.join(p.erosion_exposure_dir, 'ps_gated_%s_%d.tif' % (scn, yr))
    _ref = rxr.open_rasterio(_ps_path(base_scenario, anchor_years[0]), masked=True).squeeze()   # any ps: shared grid
    zr = zones.to_crs(_ref.rio.crs)
    zone_id = rio_rasterize([(g, int(z)) for g, z in zip(zr.geometry, zr[zid_col])],
                            out_shape=_ref.shape, transform=_ref.rio.transform(), fill=0, dtype='int32')
    max_id = int(zone_id.max())
    _ref_path = _ps_path(base_scenario, anchor_years[0])
    _resample_dir = os.path.join(p.cur_dir, 'resampled_spam_bands')
    with rasterio.open(yield_stack) as _sy:
        nb = _sy.count
    crop_prod = []                     # [(spam_crop, production_array float64, coefficient)] per SPAM crop, on the grid
    tot = np.zeros(max_id + 1)         # total production per zone (ps-independent -> constant)
    for _, r in bandmap.iterrows():
        b = int(r[bcol])
        if b < 1 or b > nb:
            continue
        elast = ef.get_erosion_yield_coefficient(str(r[crcol]).strip().lower(), coef_map,
                                                 fallback_coef, spam_aliases)
        # Yield averages and area sums: see resample_band_to_match for why the pair is not
        # interchangeable. Byte-identical to the rioxarray path this replaced, and ~11x faster.
        y = resample_band_to_match(yield_stack, b, _ref_path, _resample_dir, 'average')
        ha = np.clip(resample_band_to_match(area_stack, b, _ref_path, _resample_dir, 'sum'), 0.0, None)
        prod = y * ha
        crop_prod.append((str(r[crcol]).strip().lower(), prod, elast))
        m = np.isfinite(prod) & (zone_id > 0)
        tot += np.bincount(zone_id[m], weights=prod[m], minlength=max_id + 1)

    _matched = {c: _table_alpha(c) for c in {cr for cr, _, _ in crop_prod}}
    _sector_mean = {}
    for s in erosion_shock_acts:
        vals = [v for c, v in _matched.items() if v is not _MISS and crop_sector(c) == s]
        _sector_mean[s] = sum(vals) / len(vals) if vals else alpha


    def _grid(name, scn, yr):
        path = os.path.join(p.erosion_exposure_dir, '%s_%s_%d.tif' % (name, scn, yr))
        return np.nan_to_num(rxr.open_rasterio(path, masked=True).squeeze().values)

    def _zonal(weights):
        """sum a per-pixel weight into zones -> array indexed by zone id."""
        m = np.isfinite(weights) & (zone_id > 0)
        return np.bincount(zone_id[m], weights=weights[m], minlength=max_id + 1)

    def _series(num, den):
        with np.errstate(invalid='ignore', divide='ignore'):
            lvl = np.where(den > 0, num / den, np.nan)
        return pd.Series({int(i): lvl[i] for i in range(1, max_id + 1) if den[i] > 0})

    def level_damage(scn, yr):
        """METHOD A ("damage") -- the documented GTAP method behind the paper's frozen
        numbers. p_crop = severe share of the zone's cropland AREA (USLE > the per-country T of 2/11,
        cropland = SEALS7 class 2); level = -100*alpha*p_crop. Binary threshold, flat alpha, no off-site
        routing. UNIFORM across the GTAP crop sectors by construction: it is measured from LAND COVER, which
        carries no crop detail, so A cannot distinguish wheat land from vegetable land."""
        p_crop = _series(_zonal(_grid('severe_cropland_frac', scn, yr)),
                         _zonal(_grid('cropland_frac', scn, yr)))
        lvl = -100.0 * alpha * p_crop
        return {s: lvl for s in erosion_shock_acts}

    def _service_level(ps, rkls):
        """Production-weighted prevention level per GTAP sector, shared by B and B-thresholded.

        Per crop, the prevention share is prevented tonnes over potential tonnes, so a pixel with
        negligible erosion cannot earn credit for preventing almost nothing. Each crop's share is
        bridged to yield by its OWN alpha and averaged across crops by production. The two callers
        differ only in the ps field and in whether rkls is restricted to severe pixels."""
        per_sector = {s: [np.zeros(max_id + 1), np.zeros(max_id + 1)] for s in erosion_shock_acts}
        all_num = np.zeros(max_id + 1)
        for crop, prod, _elast in crop_prod:
            potential = _zonal(rkls * prod)
            prevented = _zonal(ps * rkls * prod)
            at_stake = potential > 0
            share = np.where(at_stake, prevented / np.where(at_stake, potential, 1.0), 0.0)
            # A zone-crop with nothing at stake carries NO WEIGHT, rather than scoring a share of 0.
            # Zero would read as "no protection delivered" when it means "no erosion to protect
            # against", so under a severe threshold a zone that stops eroding between baseline and
            # scenario would look like its protection collapsed. That produced a -28% shock for paddy
            # rice in South Africa, a zone with almost no rice and no severe pixels in the scenario.
            # Dropping it from both sides of the ratio lets it fall back to the all-crop level below.
            weight = _zonal(prod) * at_stake
            contribution = weight * alpha_for(crop) * share
            all_num += contribution
            sector = crop_sector(crop)
            if sector in per_sector:
                per_sector[sector][0] += contribution
                per_sector[sector][1] += weight
        # A zone growing none of a sector's crops has no sector-specific signal, so fall back to the
        # all-crop level there rather than emitting NaN into the GTAP shock.
        all_crop = 100.0 * _series(all_num, tot)
        out = {}
        for s, (num, den) in per_sector.items():
            lvl = 100.0 * _series(num, den)
            out[s] = lvl.reindex(all_crop.index).fillna(all_crop)
        return out

    def level_service(scn, yr):
        """METHOD B ("service") -- threshold-free. Credits the continuous prevention share across ALL
        land, which saturates it (median 0.9988, about half of pixels pinned at full protection), so
        it is reported for comparison and no longer feeds GTAP. Signed positive as a service
        delivered, but it still INCREASES with better land condition exactly as A does."""
        return _service_level(np.clip(_grid('ps_continuous', scn, yr), 0.0, 1.0),
                              _grid('rkls_grid', scn, yr))

    def level_service_threshold(scn, yr):
        """METHOD B THRESHOLDED -- B confined to severely eroding pixels, with the severe set taken
        from the BASE scenario and held FIXED. THE DEFAULT (see p.erosion_method).

        Verified inside this task by the full ZAF pipeline run: shock_pct, the column
        build_combined_afeall consumes, comes out identical to shock_pct_service_threshold with a
        minimum of exactly 0.0 and a maximum of +1.30%, i.e. no negative outliers at all. At 2050,
        63 zone-erosion_shock_acts respond with a mean of +0.157%. Unthresholded B on the same run tops out at
        +0.52%, so the threshold carries roughly 2.5x the signal.

        Restricting to severe pixels is what stops B saturating, since the union sits near 1 across
        the ~98% of land that is not eroding. But a scenario-VARYING severe set makes the shock partly
        a change of population rather than of protection: the two levels then average over different
        pixels, and in a zone with only a handful of severe pixels one entering or leaving swings the
        average. That put -18% into a paddy-rice zone in South Africa. Holding the set fixed makes the
        difference measure protection alone, and also makes `potential` identical across scenarios,
        because RKLS carries no cover factor and so does not vary with land use.

        ps_continuous is read here rather than ps_gated: ps_gated is masked to each scenario's OWN
        severe set, which is exactly what is being held fixed. rkls carries the same fixed mask, so
        numerator and denominator are restricted together and this stays a prevention share where
        higher means better. Gating only the numerator would instead measure how much severe erosion a
        zone HAS. No cropland term is needed: every sum carries prod as a factor, so a pixel with no
        production contributes nothing regardless."""
        keep = _grid('severe_mask', base_scenario, yr) > 0.5
        return _service_level(np.where(keep, np.clip(_grid('ps_continuous', scn, yr), 0.0, 1.0), 0.0),
                              np.where(keep, _grid('rkls_grid', scn, yr), 0.0))

    LEVELS = {'damage': level_damage, 'service': level_service,
              'service_threshold': level_service_threshold}
    primary = str(p.erosion_method).lower()
    if primary not in ('damage', 'service', 'service_threshold'):
        raise ValueError("p.erosion_method must be 'damage' (the deck's Method A), 'service' "
                         "(Method B) or 'service_threshold' (B restricted to severely eroding "
                         "pixels before compositing), got %r. All three are computed and "
                         "reported side by side; this only selects which becomes shock_pct." % primary)

    base_map = p.scenario_lulc_paths.get(base_scenario, {})
    all_years = list(range(es_shock_base_year, es_shock_end_year + 1))
    anchors_x = [es_shock_base_year] + anchor_years

    def annual(scn_by_year, base_by_year, sector, zid):
        """ABSOLUTE difference of the % levels for one sector (the level IS a % of crop productivity, so
        differencing it gives the % productivity change), ramped 0 at es_shock_base_year through the anchors."""
        a = [scn_by_year[y][sector].get(zid, np.nan) - base_by_year[y][sector].get(zid, np.nan)
             for y in anchor_years]
        return np.interp(all_years, anchors_x, [0.0] + a)

    # one pass per method: {sector: level Series} at each anchor, for the baseline and every scenario
    by_method = {}
    for name, fn in LEVELS.items():
        base_by_year = {y: fn(base_scenario, y) for y in anchor_years}
        base_at_base = fn(base_scenario, es_shock_base_year) if es_shock_base_year in base_map else None
        by_method[name] = (base_by_year, base_at_base,
                           {scn: {y: fn(scn, y) for y in anchor_years} for scn in scenarios})

    rows = []
    for scn in scenarios:
        anchor_levels = by_method[primary][2][scn]
        zids = sorted(set().union(*[set(lv[erosion_shock_acts[0]].index) for lv in anchor_levels.values()]))
        for zid in zids:
            if zid not in labels:
                continue
            endw, reg = labels[zid]
            for sector in erosion_shock_acts:
                series = {name: annual(base_and_scn[2][scn], base_and_scn[0], sector, zid)
                          for name, base_and_scn in by_method.items()}
                base_by_year, base_at_base, scn_levels = by_method[primary]
                if base_at_base is not None:
                    f = [scn_levels[scn][y][sector].get(zid, np.nan) - base_at_base[sector].get(zid, np.nan)
                         for y in anchor_years]
                    annual_f = np.interp(all_years, anchors_x, [0.0] + f)
                else:
                    annual_f = [np.nan] * len(all_years)
                for i, yr in enumerate(all_years):
                    rows.append({'ENDW': endw, 'ACTS': sector, 'REG': reg, 'scenario': scn, 'year': yr,
                                 'shock_pct': series[primary][i],
                                 # Equal to shock_pct by construction here: erosion's level is already
                                 # a % of crop productivity, so the shock is an absolute difference with
                                 # no denominator to vary. Emitted anyway because carbon and pollination
                                 # carry the contemp/fixedbase pair and the viz gates a figure on both.
                                 'shock_pct_contemp': series[primary][i],
                                 'shock_pct_fixedbase': annual_f[i],
                                 'shock_pct_damage': series['damage'][i],
                                 'shock_pct_service': series['service'][i],
                                 'shock_pct_service_threshold': series['service_threshold'][i]})

    out = pd.DataFrame(rows)
    utilities.assert_shock_table_sound(out, scenarios, 'erosion')
    out.to_csv(p.erosion_shock_output_path, index=False)
    end = out[out['year'] == es_shock_end_year]
    hb.log('  erosion shock (dynamic): %d rows, %d scenarios, %d anchors, alpha=%.3f, primary=%s'
          % (len(out), len(scenarios), len(anchor_years), alpha, primary.upper()))
    hb.log('     mean shock @%d   A: %+.4f%%   B: %+.4f%%   B-thresholded: %+.4f%%'
          % (es_shock_end_year, end['shock_pct_damage'].mean(), end['shock_pct_service'].mean(),
             end['shock_pct_service_threshold'].mean()))
    return True

def erosion_shock_static(p):
    """Static per-scenario erosion shock -> 8 crop erosion_shock_acts, linear ramp 0->es_shock_end_year.

    Caller sets on p before calling: es_shock_scenarios, es_shock_base_year,
    es_shock_end_year, erosion_shock_output_path. Dependency csv defaults to
    input_dir/raw_dependencies/erosion_prevention_dependency.csv (override p.erosion_dependency_path);
    scenario->raw name via p.erosion_scenario_map (default: identity -- each scenario maps to its own
    name; a scenario the table labels differently is warned about loudly and skipped rather than
    silently zeroed, so set the map for those).
    """
    publish_inputs(p)
    # Default into the es_shocks parent dir. Runtime, not build time: p.es_shock_dir is
    # published by that task, which ProjectFlow runs before this one.
    if not getattr(p, 'erosion_shock_output_path', None):
        p.erosion_shock_output_path = os.path.join(getattr(p, 'es_shock_dir', None) or p.project_dir,
                                                   'erosion_interpolated.csv')
    if not p.run_this:
        return

    es_shock_base_year = int(p.es_shock_base_year)
    es_shock_end_year = int(p.es_shock_end_year)
    n_years = es_shock_end_year - es_shock_base_year
    erosion_scenario_map = getattr(p, 'erosion_scenario_map', {})
    es_shock_scenarios = list(p.es_shock_scenarios)
    erosion_shock_acts = tuple(p.erosion_shock_acts)   # GTAP crop sectors

    ero_path = getattr(p, 'erosion_dependency_path', None) or os.path.join(
        p.input_dir, 'raw_dependencies', 'erosion_prevention_dependency.csv')
    if not hb.path_exists(ero_path):
        raise NameError(
            'erosion shock: no dependency table at %s. This used to print and return, which left '
            'the consumer with no erosion shock and nothing in the run that failed -- the same '
            'shape as a scenario silently zeroed, which the loop below refuses to do. Set '
            'p.erosion_dependency_path, or stage the file under input_dir/raw_dependencies/.'
            % ero_path)

    df = read_erosion_dependency(ero_path)
    # Resolve the configured base through the candidate mechanism (fatal if absent) -- the erosion
    # table spells the nature-off baseline 'baseline_ignore_damages' while the shared config may say
    # 'baseline_ignore_dependencies'; the two spellings are mutual aliases by default
    # (utilities.NATURE_OFF_SPELLINGS), so no consumer map is needed for this.
    base_scenario = utilities.required_base_scenario(p, 'erosion')
    raw_base = utilities.resolve_base_scenario(df['scenario'].values, erosion_scenario_map, base_scenario, 'erosion')
    base_vals = df[df['scenario'] == raw_base].set_index(
        ['aez18_id', 'gtapv7_r50_label'])['value'].astype(float).fillna(0)

    rows = []
    for our_scn in es_shock_scenarios:
        raw_scn = utilities.resolve_raw_scenario(df['scenario'].values, erosion_scenario_map, our_scn, 'erosion')
        if raw_scn is None:
            continue
        scn_vals = df[df['scenario'] == raw_scn].set_index(
            ['aez18_id', 'gtapv7_r50_label'])['value'].astype(float).fillna(0)
        common = scn_vals.index.intersection(base_vals.index)
        shock = scn_vals.loc[common] - base_vals.loc[common]
        for year in range(es_shock_base_year, es_shock_end_year + 1):
            frac = (year - es_shock_base_year) / n_years
            for (aez_id, reg), val in shock.items():
                endw = 'AEZ%d' % int(aez_id)
                for sector in erosion_shock_acts:
                    rows.append({'ENDW': endw, 'ACTS': sector, 'REG': reg,
                                 'scenario': our_scn, 'year': year, 'shock_pct': val * frac})

    out = pd.DataFrame(rows)
    utilities.assert_shock_table_sound(out, es_shock_scenarios, 'erosion')
    out.to_csv(p.erosion_shock_output_path, index=False)
    nz = out[(out['year'] == es_shock_end_year) & (out['shock_pct'] != 0)] if len(out) else out
    hb.log('  erosion shock: %d rows, %d scenarios, %d nonzero @%d (static, uncapped) -> %s'
          % (len(out), out['scenario'].nunique() if len(out) else 0, len(nz), es_shock_end_year,
             p.erosion_shock_output_path))
    return True


# =============================================================================
# GEP valuation tasks (folded from global_erosion_gep): InVEST SDR -> prevention
# shares -> per-country GEP -> maps/figures. The ES-shock tasks above and this
# valuation are separate consumers of the same erosion science.
# =============================================================================


def invest_sdr(p):
    """Section A: run InVEST SDR to produce the erosion rasters (USLE, avoided erosion) that
    Section B consumes. ProjectFlow-idiomatic: outputs default into THIS task's dir (caller may
    override erosion_sdr_output_dir), the paths Section B reads are PUBLISHED on p before the
    run_this guard (so a skipped rerun still feeds downstream), and the builders register this
    with skip_existing=1 -- delete the task dir to force a rerun.
    """
    publish_inputs(p)
    if not getattr(p, 'erosion_sdr_output_dir', None):
        p.erosion_sdr_output_dir = p.cur_dir
    if not getattr(p, 'erosion_watersheds_sanitized_path', None):
        p.erosion_watersheds_sanitized_path = os.path.join(p.erosion_sdr_output_dir, 'wshed_sanitized.gpkg')
    # Publish Section A's outputs so Section B reads them, but ONLY when this task is producing
    # them or has already produced them. Unconditionally it overwrote the erosion_usle_path and
    # erosion_avoided_erosion_path rows, which point at the author's staged rasters -- and those
    # rasters are what the published number is computed from, so the account was being pointed at
    # a file this task had not written.
    _sfx = p.erosion_sdr_results_suffix
    usle = os.path.join(p.erosion_sdr_output_dir, f'usle_{_sfx}.tif')
    avoided = os.path.join(p.erosion_sdr_output_dir, f'avoided_erosion_{_sfx}.tif')
    if p.run_this or hb.path_exists(usle):
        p.erosion_usle_path = usle
        p.erosion_avoided_erosion_path = avoided
    if not p.run_this:
        return
    hb.create_directories(p.erosion_sdr_output_dir)
    paths = erosion_paths(p)
    p.erosion_sdr_args, p.erosion_sdr_file_registry = run_invest_sdr(paths)
    return True


def upstream_prevention_share(p):
    """The share of soil loss that upslope land cover prevents, routed from the SDR outputs.

    Section A gives actual erosion and avoided erosion, so potential (bare-soil) erosion is their
    sum, by the definition of avoided. Accumulating avoided and potential down the same D8 network
    gives the upstream share that Section B combines with the on-farm one.

    The source repo read this layer out of its own cluster workspace, which is why the valuation
    used to need that workspace. Computing it here from the DEM and Section A's own outputs is
    what lets the account run wherever the SDR does.
    """
    publish_inputs(p)
    # Published before the run_this guard: Section B reads this path off p whether or not this
    # task ran, the same way it reads Section A's.
    #
    # A path configured in es_parameters wins, so the account can read the author's own layer
    # rather than our rebuild of it. Until 2026-08-27 this line overwrote the configured value
    # unconditionally, which made that row inert: it could be set to anything and the task's own
    # output was used regardless. Where nothing is configured, or the configured file is absent,
    # the task owns the path -- that fallback is what lets the account run on a machine that
    # cannot reach the cluster the layer came from.
    built = os.path.join(p.cur_dir, 'upstream_prevention_share.tif')
    configured = getattr(p, 'erosion_upstream_prevention_share_path', None)
    if configured and os.path.abspath(str(configured)) != os.path.abspath(built) \
            and hb.path_exists(configured):
        hb.log('upstream_prevention_share: reading the configured layer at %s' % configured)
    else:
        p.erosion_upstream_prevention_share_path = built
    if not p.run_this:
        return
    if hb.path_exists(p.erosion_upstream_prevention_share_path):
        hb.log('upstream_prevention_share.tif already exists. Skipping the D8 routing.')
        return True
    import numpy as np
    import pygeoprocessing as pgp

    work_dir = os.path.join(p.cur_dir, 'routing')
    os.makedirs(work_dir, exist_ok=True)

    # Potential erosion, as avoided + actual. InVEST writes rkls to its intermediate directory,
    # but forming it from the two rasters Section A publishes keeps this task independent of
    # InVEST's internal layout, and is exact rather than approximate.
    info = pgp.get_raster_info(p.erosion_avoided_erosion_path)
    avoided = hb.as_array(p.erosion_avoided_erosion_path).astype('float64')
    actual = hb.as_array(p.erosion_usle_path).astype('float64')
    avoided_ndv = info['nodata'][0]
    actual_ndv = pgp.get_raster_info(p.erosion_usle_path)['nodata'][0]
    valid = (np.isfinite(avoided) & (avoided != avoided_ndv)
             & np.isfinite(actual) & (actual != actual_ndv))
    potential_path = os.path.join(work_dir, 'rkls.tif')
    pgp.numpy_array_to_raster(
        np.where(valid, avoided + actual, -1.0).astype('float32'), -1.0, info['pixel_size'],
        (info['geotransform'][0], info['geotransform'][3]), info['projection_wkt'], potential_path)

    accumulate_upstream_prevention_share(
        p.get_path(p.erosion_dem_path), p.erosion_avoided_erosion_path, potential_path,
        work_dir, p.erosion_upstream_prevention_share_path)
    return True


def prevention_shares(p):
    """Section B: combine on-farm (AE/(AE+USLE)) and upstream prevention shares into the
    union-of-protection PS_combined, then country-crop protected production and the GEP valuation
    (onfarm / upstream / combined) -> integrated_country_gep.csv + the PS rasters the maps task
    reads. ProjectFlow-idiomatic: outputs default into THIS task's dir via erosion_gep_output_dir
    (the same attr configure_maps calculations on), USLE/avoided arrive from invest_sdr's
    published attrs, and the registered result is the skip check (like every gep_calculation).
    """
    publish_inputs(p)
    if not getattr(p, 'erosion_gep_output_dir', None):
        p.erosion_gep_output_dir = p.cur_dir
    service_results = p.results.setdefault('erosion', {})
    service_results['integrated_country_gep'] = os.path.join(
        p.erosion_gep_output_dir, 'integrated_country_gep.csv')
    if not p.run_this:
        return
    if hb.path_all_exist(list(service_results.values())):
        hb.log("%s already exists. Skipping prevention-share calculation for erosion."
               % os.path.basename(service_results['integrated_country_gep']))
        return True
    hb.create_directories(p.erosion_gep_output_dir)
    paths = erosion_paths(p)
    integrate_and_write(paths)
    return True


def gep_result(p):
    """Render the results report via utilities.render_service_results, as every service does."""
    publish_inputs(p)
    utilities.render_service_results(p)


def maps_and_figures(p):
    """Section C: publication-ready choropleths, raster previews and charts from Section B's
    outputs (found via the shared erosion_gep_output_dir attr). Figures default into THIS task's
    dir; skip_existing at registration."""
    publish_inputs(p)
    if not getattr(p, 'erosion_figures_dir', None):
        p.erosion_figures_dir = p.cur_dir
    if not p.run_this:
        return
    hb.create_directories(p.erosion_figures_dir)
    paths = erosion_paths(p)
    generate_all_maps_and_figures(paths)
    return True


def gep_calculation(p):
    """One row per country, under the key every other service writes.

    The integrated pipeline writes integrated_country_gep.csv, which is wide: one row per
    country carrying the on-farm, upstream and combined components side by side. The account
    reads gep_by_country_base_year.csv with a single <service>_gep column, the same shape as the
    other twenty services, so this turns one into the other. Without it erosion's number is in
    the project directory but not in the place anything downstream looks.

    The combined component is the account's, because the service is what on-farm cover and
    everything upslope prevent together, not either alone.
    """
    publish_inputs(p)
    service_results, already_done = utilities.begin_gep_calculation(p, 'erosion')
    if not p.run_this or already_done:
        return

    paths = erosion_paths(p)
    if not hb.path_exists(str(paths.output.integrated_country_gep)):
        raise NameError('erosion has no integrated_country_gep.csv at %s. prevention_shares '
                        'writes it, so run that first.' % paths.output.integrated_country_gep)

    df = hb.df_read(str(paths.output.integrated_country_gep))
    column = 'gep_const2019_usd_combined'
    if column not in df.columns:
        raise NameError('integrated_country_gep.csv carries no %s column, only %s'
                        % (column, sorted(df.columns)[:8]))

    # The integrated table is one row per territory, not per country: France carries seven rows,
    # Australia three, and eleven codes repeat in all. Summing to the code first is what makes the
    # join one-to-one; without it the merge widens the 250-country table to 268 rows, reports a
    # country count that counts territories, and hands any downstream join France seven times.
    # The total is unchanged either way, because every territory still contributes exactly once.
    values = (df[['iso3', column]]
              .rename(columns={'iso3': 'iso3_r250_label', column: 'erosion_gep'})
              .groupby('iso3_r250_label', as_index=False)['erosion_gep'].sum(min_count=1))
    df_gep = utilities.country_attributes(p).merge(values, on='iso3_r250_label', how='left')
    df_gep['year'] = int(p.gep_base_year)
    hb.df_write(df_gep, service_results['gep_by_country_base_year'])
    hb.log('Total erosion GEP for base year %d: %s over %d countries'
           % (int(p.gep_base_year), format(df_gep['erosion_gep'].sum(), ',.2f'),
              int(df_gep['erosion_gep'].notna().sum())))
    return True

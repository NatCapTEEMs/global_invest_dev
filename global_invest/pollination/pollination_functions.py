"""Thin driver over William Sidemo-Holm's crop_benefits pollination science.

The science (300 m sufficiency, 5 km valuation, PNAS diff) is imported unchanged from
crop_benefits; here we only run it per scenario on our SEALS maps and zonal-aggregate the diff
to GTAP r50xAEZ regions, reproducing the % change definition in his pnas_ngfs pipeline
(crop_benefits/project_flow/pnas_ngfs.py, which is not part of the installed package).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize

from crop_benefits.pollination.sufficiency_poll import (
    run_pollination_sufficiency_300m,
    run_pollination_sufficiency_300m_stable_ag,
    run_pollination_sufficiency_5km,
)
from crop_benefits.pollination.sufficiency_value import run_pollination_valuation_5km
from crop_benefits.pollination.sufficiency_diff import run_pollination_diff_5km_pnas
from crop_benefits.raster.spatial import build_area_km2_raster
from crop_benefits.config import load_config

BASELINE_LABEL = '2023_pnas'


def configure_crop_benefits(p, target_year):
    """crop_benefits Config pointed at OUR base_data + task dir -- from p, no hardcoded machine paths.

    validate=False: our chain skips the FAO/CropGrids tabular stages, so the ONLY external input it
    needs is the precomputed baseline pollination-value raster under base_data/crop_benefits/. The 5 km
    resample template only has to define the target grid, and run_pollination_valuation_5km requires
    sufficiency and value to share that grid -- so country_raster points at the value raster itself,
    which removes the separate country-raster dependency. Sufficiency outputs go to the task dir.
    """
    cfg = load_config(validate=False)
    cb = p.get_path('crop_benefits')
    # poll_value_global_<year>usd.tif is a PRECOMPUTED baseline input, not built here. It is the
    # output of the crop_benefits raster pipeline, run once over Monfreda yields x CropGrids area x
    # FAO producer prices x pollination-dependence ratios:
    #   crop_benefits: scripts/pipelines/run_fao_pipeline.py  then
    #                  scripts/pipelines/run_raster_pipeline.py --step poll_value
    #                  (src/crop_benefits/raster/pollination_value.py::run_pollination_value)
    # This fold reuses that raster and skips the FAO/CropGrids tabular stages (validate=False).
    cfg.paths.country_raster = Path(os.path.join(cb, 'poll_value_global_%dusd.tif' % int(target_year)))
    cfg.outputs.pollination_value_2020 = Path(cb)
    cfg.outputs.pollination_sufficiency = Path(p.cur_dir)
    return cfg


def baseline_denominator(cfg, baseline_lulc_path, target_year):
    """Unpaired 2023 pollination value (the % change denominator), computed once."""
    run_pollination_sufficiency_300m(cfg, lulc_path=baseline_lulc_path, scenario=BASELINE_LABEL)
    run_pollination_sufficiency_5km(cfg, scenario=BASELINE_LABEL)
    run_pollination_valuation_5km(cfg, scenario=BASELINE_LABEL, target_year=target_year)
    return cfg.outputs.pollination_sufficiency / f'value_pollination_sufficiency_{BASELINE_LABEL}_5km.tif'


def scenario_region_pct_change(cfg, scenario, lulc_path, baseline_lulc_path,
                               denominator_path, correspondence_gpkg, target_year):
    """Per-region % change of pollination value, scenario vs 2023 baseline (stable cropland only).

    Returns (pct_change, baseline_value_usd); see _zonal_pct_change for why the level comes free.
    """
    stab, b_stab = f'{scenario}_stab', f'{BASELINE_LABEL}_stab_{scenario}'
    for suff_scen, lulc, other in [(stab, lulc_path, baseline_lulc_path),
                                   (b_stab, baseline_lulc_path, lulc_path)]:
        run_pollination_sufficiency_300m_stable_ag(cfg, lulc_path=lulc, other_lulc_path=other, scenario=suff_scen)
        run_pollination_sufficiency_5km(cfg, scenario=suff_scen)
        run_pollination_valuation_5km(cfg, scenario=suff_scen, target_year=target_year)

    suff_dir = cfg.outputs.pollination_sufficiency
    diff_path = run_pollination_diff_5km_pnas(
        cfg, scenario=scenario, baseline_scenario=BASELINE_LABEL,
        scenario_value_path=suff_dir / f'value_pollination_sufficiency_{stab}_5km.tif',
        baseline_value_path=suff_dir / f'value_pollination_sufficiency_{b_stab}_5km.tif')
    return _zonal_pct_change(diff_path, denominator_path, correspondence_gpkg)


def _zonal_pct_change(diff_path, denominator_path, correspondence_gpkg, region_id_field='ee_r50_aez18_id'):
    """Per r50xAEZ zone, keyed to (ENDW, REG): the % change AND the absolute baseline value.

    Returns (pct_change, baseline_value_usd) as two aligned Series.

    The second is the GEP quantity and costs nothing to emit: it is the DENOMINATOR of the first
    (sum of the baseline pollination-value raster x pixel area over the zone, in target-year USD),
    so it is already computed here and was previously discarded. GEP wants the level, GTAP wants
    the ratio; returning both means one task serves both rather than two chains recomputing the
    same rasters. See pollination_shock, which writes it as `value_usd_base`.
    """
    gdf = gpd.read_file(correspondence_gpkg, engine='pyogrio')
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    gdf[region_id_field] = gdf[region_id_field].astype(int)

    with rasterio.open(denominator_path) as src:
        base = src.read(1).astype(np.float64)
        base[base == src.nodata] = np.nan
        transform, shape, meta = src.transform, src.shape, src.meta.copy()
    with rasterio.open(diff_path) as src:
        diff = src.read(1).astype(np.float64)
        diff[diff == src.nodata] = np.nan

    zones = rasterize(
        ((g, int(r)) for g, r in zip(gdf.geometry, gdf[region_id_field]) if g is not None and not g.is_empty),
        out_shape=shape, transform=transform, fill=0, dtype=np.int32)
    area = build_area_km2_raster(meta)

    out, level = {}, {}
    for rid, sub in gdf.drop_duplicates(region_id_field).set_index(region_id_field).iterrows():
        mask = zones == rid
        denom = np.nansum(base[mask] * area[mask]) if mask.any() else 0.0
        if not denom:
            continue
        key = (f"AEZ{int(sub['aez18_id'])}", sub['gtapv7_r50_label'])
        out[key] = np.nansum(diff[mask] * area[mask]) / denom * 100.0
        level[key] = denom
    return pd.Series(out), pd.Series(level)

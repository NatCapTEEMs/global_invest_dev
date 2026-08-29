"""Dynamic pollination ES-shock task (V_F, OSD).

Runs the pollination sufficiency and value calculation on our SEALS 300 m maps at EACH SEALS
anchor year (seals_years), then piecewise-linearly interpolates the shock to annual values. Writes
pollination_interpolated.csv -- the file build_combined_afeall_cc_es reads -- into the shared ES-shock
directory (p.es_shock_dir). Grafted by consumers via add_pollination_tasks (dispatch on p.dynamic_es).
"""
from __future__ import annotations
import os
import gc
import logging
import math
from typing import Any, Dict, Set, Tuple
import rasterio
from joblib import Parallel, delayed
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window
from scipy.ndimage import convolve
from tqdm import tqdm
import glob
import numpy as np
import pandas as pd
import hazelbean as hb
from global_invest import utilities
from global_invest.pollination import pollination_functions as pf

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------------------------
# VENDORED_FROM: wsidemoholm/crop_benefits @ 80a23b0 (2026-07-08). Upstream moved to
# 51bcceb (2026-08-28); those commits refactor sufficiency_poll and sufficiency_value
# without changing the arithmetic for an ESA call. Record the commit when re-vendoring:
# without it nobody can answer "are we current?" without diffing the repo by hand.
# Vendored from crop_benefits: the sufficiency and value raster steps. They stream global
# rasters window by window, so they open and write files and belong here rather than beside
# the arithmetic.
# ---------------------------------------------------------------------------------------------


FAO_MEDIAN_PRICES_REF_PATH = os.path.join('fao', 'median_prices')

_RADIUS_METERS = 2000.0
_THRESHOLD = 0.30
_COS_LAT_FLOOR = 0.1
_MAX_RX = 2048
_MAX_RY = 1024
_METERS_PER_DEG_LAT = 111320.0
_AG_CLASSES = {10, 11, 12, 20, 30, 40}
_NAT_CLASSES = {50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100, 110, 120, 121, 122, 130, 140, 150, 152, 153}
_SEALS_AG_CLASSES = {2}
_SEALS_NAT_CLASSES = {3, 4, 5}


# ---------------------------------------------------------------------------------------------
# Vendored from crop_benefits: the FAO price path, the parts that download and write.
# ---------------------------------------------------------------------------------------------



# Helpers the vendored raster steps use, from the same source modules.

def _make_elliptical_kernel(ry: int, rx: int) -> np.ndarray:
    """Create a binary elliptical kernel of half-axes (ry, rx) pixels."""
    ry = max(1, int(ry))
    rx = max(1, int(rx))
    size_y = ry * 2 + 1
    size_x = rx * 2 + 1
    yy, xx = np.ogrid[:size_y, :size_x]
    cy, cx = ry, rx
    dy = (yy - cy) / float(ry)
    dx = (xx - cx) / float(rx)
    dist = dy * dy + dx * dx
    return (dist <= 1.0).astype(np.uint8)


def _window_expand_aniso(window: Window, pad_y: int, pad_x: int, height: int, width: int) -> Window:
    """Expand a rasterio Window by (pad_y, pad_x) pixels, clamped to raster bounds."""
    r0 = max(0, int(window.row_off) - int(pad_y))
    c0 = max(0, int(window.col_off) - int(pad_x))
    r1 = min(height, int(window.row_off + window.height) + int(pad_y))
    c1 = min(width, int(window.col_off + window.width) + int(pad_x))
    return Window(c0, r0, c1 - c0, r1 - r0)


def _tile_mid_lat(bounds_top: float, transform_e: float, row0: int, h: int) -> float:
    """Mid-tile centroid latitude."""
    mid_row = row0 + (h / 2.0)
    return bounds_top + (mid_row + 0.5) * transform_e


def _compute_radii_pixels(lat_deg: float, pixel_lat_deg: float, pixel_lon_deg: float) -> Tuple[int, int]:
    """Convert the ~2 km foraging radius to pixel counts at a given latitude."""
    cos_lat = abs(math.cos(math.radians(lat_deg)))
    cos_lat = max(cos_lat, _COS_LAT_FLOOR)

    radius_deg_lat = _RADIUS_METERS / _METERS_PER_DEG_LAT
    radius_deg_lon = _RADIUS_METERS / (_METERS_PER_DEG_LAT * cos_lat)

    ry = int(math.ceil(radius_deg_lat / pixel_lat_deg))
    rx = int(math.ceil(radius_deg_lon / pixel_lon_deg))

    ry = max(1, min(ry, _MAX_RY))
    rx = max(1, min(rx, _MAX_RX))
    return ry, rx


def save_parquet(df: pd.DataFrame, path: str | str) -> str:
    """Save DataFrame as parquet and log the result."""
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved %d rows -> %s", len(df), path)
    return path


def baseline_denominator(cfg, baseline_lulc_path, target_year):
    """Unpaired 2023 pollination value (the % change denominator), computed once."""
    run_pollination_sufficiency_300m(cfg, lulc_path=baseline_lulc_path, scenario=pf.BASELINE_LABEL)
    run_pollination_sufficiency_5km(cfg, scenario=pf.BASELINE_LABEL)
    run_pollination_valuation_5km(cfg, scenario=pf.BASELINE_LABEL, target_year=target_year)
    return cfg.output_dir / f'value_pollination_sufficiency_{pf.BASELINE_LABEL}_5km.tif'


def scenario_diff_raster(cfg, scenario, lulc_path, baseline_lulc_path, target_year):
    """The 5 km scenario-minus-baseline value raster, on cropland stable across the two maps.

    Returns the path crop_benefits wrote it to; the task reads it and hands the array to
    zonal_pct_change.
    """
    stab, b_stab = f'{scenario}_stab', f'{pf.BASELINE_LABEL}_stab_{scenario}'
    for suff_scen, lulc, other in [(stab, lulc_path, baseline_lulc_path),
                                   (b_stab, baseline_lulc_path, lulc_path)]:
        run_pollination_sufficiency_300m_stable_ag(cfg, lulc_path=lulc, other_lulc_path=other, scenario=suff_scen)
        run_pollination_sufficiency_5km(cfg, scenario=suff_scen)
        run_pollination_valuation_5km(cfg, scenario=suff_scen, target_year=target_year)

    suff_dir = cfg.output_dir
    return run_pollination_diff_5km_pnas(
        cfg, scenario=scenario, baseline_scenario=pf.BASELINE_LABEL,
        scenario_value_path=suff_dir / f'value_pollination_sufficiency_{stab}_5km.tif',
        baseline_value_path=suff_dir / f'value_pollination_sufficiency_{b_stab}_5km.tif')


def _add_geographic_and_classification_data(
    df: pd.DataFrame,
    classif: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Normalise columns, add ISO3 and FAO group."""

    # Column rename
    df = df.rename(columns={
        "Area Code (M49)": "area_code_m49",
        "Area": "area_fao",
        "Item Code": "item_code_fao",
        "Item": "item_fao",
        "Element": "element",
        "Year": "year",
        "Value": "value",
    })

    # Normalise M49
    df["area_code_m49"] = (
        df["area_code_m49"]
        .astype(str)
        .str.replace(r"[^0-9]", "", regex=True)
        .str.zfill(3)
    )
    df["item_code_fao"] = df["item_code_fao"].astype(int)
    df["year"] = df["year"].astype(int)

    # Load M49 crosswalk and join
    cw, _ = load_m49_iso3(cfg)
    df = df.merge(
        cw[["area_code_m49", "iso3", "region_fao", "subregion_fao"]],
        on="area_code_m49",
        how="left",
        validate="m:1",
    )

    n_before = len(df)
    df = df.dropna(subset=["iso3"])
    logger.info(
        "Dropped %d rows without ISO3 mapping", n_before - len(df)
    )

    # Add FAO group
    df = df.merge(
        classif[["item_code_fao", "group_fao"]],
        on="item_code_fao",
        how="left",
        validate="m:1",
    )

    return df


def run_fao_production(cfg: Config) -> str:
    """
    Execute the full FAO production pipeline.

    Returns the path to the output parquet file.
    """
    logger.info("=== FAO Production Pipeline ===")

    df, classif = _read_and_filter_fao_production(cfg)
    df = pf._convert_yield_units(df)
    df = _add_geographic_and_classification_data(df, classif, cfg)

    outdir = cfg.outputs.fao_production
    pq_path = _save_production_outputs(df, outdir)

    logger.info("=== FAO Production Pipeline COMPLETE ===")
    return pq_path


def run_fao_prices(cfg: Config) -> str:
    """
    Execute the full FAO producer-price pipeline.

    Returns the path to the output parquet file.
    """
    logger.info("=== FAO Prices Pipeline ===")

    years = list(range(cfg.run.fao_start_year, cfg.run.fao_end_year + 1))

    # Steps 1-3
    pp_raw = _read_fao_prices(cfg, years)
    pp_wide = pf._reshape_prices(pp_raw)

    # Steps 4-7
    pp3 = pf._reconstruct_slc_lcu(pp_wide)

    # Steps 9-12
    pp3, fx = _add_iso3_and_fx(pp3, cfg, years)

    # Steps 13-15
    pp_usd = pf._build_usd_with_qc(pp3, fx)

    # Step 16
    outdir = cfg.outputs.fao_prices
    pq_path = _save_price_outputs(pp_usd, outdir)

    logger.info("=== FAO Prices Pipeline COMPLETE ===")
    return pq_path


def _read_and_filter_fao_production(
    cfg: Config,
) -> pd.DataFrame:
    """Read the staged FAOSTAT production bulk, filter years / elements / items."""

    # Load classification to get valid item codes for items to keep
    classif = load_fao_classification(cfg)
    classif = (
        classif[["item_code_fao", "item_fao", "group_fao"]]
        .dropna(subset=["item_code_fao"])
        .copy()
    )
    classif["item_code_fao"] = classif["item_code_fao"].astype(int)
    item_codes = set(classif["item_code_fao"].unique()) - pf._EXCLUDE_ITEM_CODES
    logger.info("Valid FAO crop item codes: %d", len(item_codes))

    # Year range from config
    years = set(range(cfg.run.fao_start_year, cfg.run.fao_end_year + 1))

    # The staged bulk, put in base data by the shared download task rather than pulled here.
    logger.info("Reading staged FAOSTAT production bulk from %s", cfg.paths.fao_production_bulk_path)
    if not hb.path_exists(cfg.paths.fao_production_bulk_path):
        raise NameError(
            'pollination has no FAOSTAT production bulk at %s. es_parameters carries its url and '
            'archive member, so the shared download task stages it.'
            % cfg.paths.fao_production_bulk_path)
    with open(cfg.paths.fao_production_bulk_path, 'rb') as f:
            df = pd.read_csv(
                f,
                usecols=[
                    "Area Code (M49)", "Area", "Item Code",
                    "Item", "Element", "Year", "Value",
                ],
                low_memory=False,
            )

    logger.info("Bulk data loaded: %d rows", len(df))

    # Numeric coercion and filtering
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Item Code"] = pd.to_numeric(df["Item Code"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    df = df[
        df["Year"].isin(years)
        & df["Element"].isin(pf._ELEMENTS_KEEP)
        & df["Item Code"].isin(item_codes)
        & (df["Value"] > 0)
    ].copy()

    logger.info("Rows after filtering: %d", len(df))
    return df, classif


def _save_production_outputs(
    df: pd.DataFrame,
    outdir: str,
) -> str:
    """Save final production dataset in standard column order."""
    os.makedirs(outdir, exist_ok=True)

    col_order = [
        "area_code_m49", "iso3", "region_fao", "subregion_fao",
        "area_fao", "item_code_fao", "item_fao", "group_fao",
        "element", "year", "value",
    ]
    df = df[col_order]

    csv_out = os.path.join(outdir, "fao_production_1993_2024.csv")
    pq_out = os.path.join(outdir, "fao_production_1993_2024.parquet")

    save_csv(df, csv_out)
    save_parquet(df, pq_out)

    logger.info("Final dataset: %d rows", len(df))
    return pq_out


def _read_fao_prices(cfg: Config, years: list[int]) -> pd.DataFrame:
    """Read the staged FAOSTAT producer-price bulk and return annual rows."""
    logger.info("=== 1) READING STAGED FAOSTAT PRODUCER PRICES ===")

    if not hb.path_exists(cfg.paths.fao_prices_bulk_path):
        raise NameError(
            'pollination has no FAOSTAT producer-price bulk at %s. es_parameters carries its url '
            'and archive member, so the shared download task stages it.'
            % cfg.paths.fao_prices_bulk_path)
    logger.info("Reading bulk file: %s", cfg.paths.fao_prices_bulk_path)
    pp_raw = pd.read_csv(cfg.paths.fao_prices_bulk_path, encoding="latin1", low_memory=False)

    # Standardize column names
    pp_raw.columns = [c.replace(" ", "_").lower() for c in pp_raw.columns]
    pp_raw = pp_raw.rename(columns={
        "area_code_(m49)": "Area Code (M49)",
        "area": "Area", "item": "Item", "item_code": "Item Code",
        "element": "Element", "year": "Year", "value": "Value",
        "months": "Months",
    })

    pp_raw = pp_raw[pp_raw["Months"] == "Annual value"].copy()
    pp_raw = pp_raw[pp_raw["Year"].isin(years)].copy()

    # Type cleanup
    pp_raw["Area Code (M49)"] = (
        pp_raw["Area Code (M49)"]
        .astype(str).str.replace("'", "", regex=False)
        .str.strip().str.zfill(3)
    )
    pp_raw["Value"] = pd.to_numeric(pp_raw["Value"], errors="coerce")

    # Text fixes
    pp_raw["Item"] = (
        pp_raw["Item"].astype(str)
        .str.replace("MatÃ© leaves", "Mate leaves", regex=False)
    )
    pp_raw["Area"] = (
        pp_raw["Area"].astype(str)
        .str.replace("CÃ´te d'Ivoire", "Ivory Coast", regex=False)
        .str.replace("RÃ©union", "Reunion", regex=False)
        .str.replace("TÃ¼rkiye", "Turkey", regex=False)
    )

    pf._log_df("pp_raw (annual PP, bulk)", pp_raw)
    return pp_raw


def world_bank_fx(fx_path):
    """Local currency per USD per country and year, from the staged World Bank file.

    Read rather than fetched, for the same reason the FAOSTAT bulks are. The World Bank revises
    PA.NUS.FCRF, so pulling it per run would make the pollination total depend on the day the
    price stage ran with nothing recording which rates were used, and a compute node without
    outbound network could not run the stage at all. es_parameters carries the indicator url, so
    the shared download task stages the file; nothing here opens a socket.

    Args:
        fx_path (str): the staged rates, iso3 and year and lcu_per_usd.

    Returns:
        pd.DataFrame: iso3, year, lcu_per_usd, fx_source.

    Raises:
        NameError: when the file is absent. Falling through to IMF-only rates would change every
            price without saying so, which is what happened when an empty fetch was not an error.
    """
    if not hb.path_exists(fx_path):
        raise NameError(
            'pollination has no World Bank exchange rates at %s. es_parameters carries the '
            'indicator and its url, so the shared download task stages it. Continuing without '
            'them would silently price everything off IMF rates instead.' % fx_path)
    staged = pd.read_csv(fx_path, encoding='utf-8-sig')
    return staged[['iso3', 'year', 'lcu_per_usd']].assign(fx_source='wb')


def _add_iso3_and_fx(
    pp3: pd.DataFrame,
    cfg: Config,
    years: list[int],
) -> pd.DataFrame:
    """Add ISO3, read WB and IMF FX, combine with inheritance."""

    # 9) ISO3 via crosswalk
    logger.info("=== 9) ADDING ISO3 ===")
    cw, _ = load_m49_iso3(cfg)
    pp3["Area Code (M49)"] = pp3["Area Code (M49)"].astype(int).astype(str).str.zfill(3)

    pp3 = pp3.merge(
        cw[["area_code_m49", "iso3", "region_fao", "subregion_fao"]],
        left_on="Area Code (M49)", right_on="area_code_m49",
        how="left", validate="m:1",
    ).drop(columns=["area_code_m49"])

    logger.info("Unique ISO3: %d", pp3["iso3"].nunique())

    # 10) World Bank FX, from the staged file beside the IMF one
    logger.info("=== 10) READING WORLD BANK FX ===")
    wb_fx = world_bank_fx(cfg.paths.fx_lcu_per_usd_path)
    wb_fx = wb_fx.dropna(subset=["iso3", "lcu_per_usd"]).copy()
    pf._log_df("wb_fx", wb_fx)

    # 11) IMF FX (local file)
    logger.info("=== 11) LOADING IMF FX ===")
    # three levels up from the crosswalk file, then the currencies tree beside it
    imf_fx_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(cfg.paths.crosswalk_m49_iso3))),
        "currencies", "imf")
    imf_files = sorted(glob.glob(os.path.join(imf_fx_path, "dataset_*.csv")))
    if not imf_files:
        logger.warning("No IMF FX file found in %s; skipping IMF FX", imf_fx_path)
        imf_fx = pd.DataFrame(columns=["iso3", "year", "lcu_per_usd", "fx_source"])
    else:
        imf_raw = pd.read_csv(imf_files[0])
        imf_fx = imf_raw.query(
            "INDICATOR == 'Domestic currency per US Dollar' and "
            "TYPE_OF_TRANSFORMATION == 'Period average' and "
            "FREQUENCY == 'Annual'"
        )
        year_cols = [c for c in imf_fx.columns if c.isdigit() and len(c) == 4]
        imf_fx = (
            imf_fx.melt(
                id_vars=["COUNTRY"], value_vars=year_cols,
                var_name="year", value_name="lcu_per_usd",
            )
            .assign(
                year=lambda d: d["year"].astype(int),
                country=lambda d: d["COUNTRY"].astype(str),
                fx_source="imf",
            )
            .loc[:, ["country", "year", "lcu_per_usd", "fx_source"]]
            .dropna(subset=["country", "lcu_per_usd"])
        )

        # 11b) Harmonise IMF country names
        imf_fx["country"] = pf._recode_country(imf_fx["country"], pf._IMF_RECODE)

        # 11c) Map IMF to ISO3
        country_to_iso3 = pp3[["country", "iso3"]].dropna().drop_duplicates()
        imf_fx = imf_fx.merge(country_to_iso3, on="country", how="left")
        imf_fx = imf_fx.dropna(subset=["iso3"]).copy()

    pf._log_df("imf_fx", imf_fx)

    # 12) Combine FX (WB priority)
    logger.info("=== 12) COMBINING FX (WB PRIORITY) ===")
    fx = (
        pd.concat([wb_fx, imf_fx[["iso3", "year", "lcu_per_usd", "fx_source"]]], ignore_index=True)
        .sort_values(
            ["iso3", "year", "fx_source"],
            key=lambda s: s.map({"wb": 0, "imf": 1}).fillna(2) if s.name == "fx_source" else s,
        )
        .drop_duplicates(subset=["iso3", "year"], keep="first")
    )

    # 12b) FX inheritance
    logger.info("=== 12b) FX INHERITANCE ===")
    inherit_rows = []
    for child, parent in pf._FX_INHERIT_ISO3.items():
        parent_frame = fx.loc[fx["iso3"] == parent].copy()
        if parent_frame.empty:
            continue
        parent_frame["iso3"] = child
        inherit_rows.append(parent_frame)
    if inherit_rows:
        fx = pd.concat([fx] + inherit_rows, ignore_index=True)
        fx = (
            fx.sort_values(
                ["iso3", "year", "fx_source"],
                key=lambda s: s.map({"wb": 0, "imf": 1}).fillna(2) if s.name == "fx_source" else s,
            )
            .drop_duplicates(subset=["iso3", "year"], keep="first")
        )
    pf._log_df("fx (after inheritance)", fx)

    return pp3, fx


def _save_price_outputs(
    pp_usd: pd.DataFrame,
    outdir: str,
) -> str:
    """Rename columns and save final price panel."""
    os.makedirs(outdir, exist_ok=True)

    pp_usd = pp_usd.rename(columns={
        "country": "area_fao",
        "Item": "item_fao",
        "Item Code": "item_code_fao",
        "Area Code (M49)": "area_code_m49",
        "usd_filled": "price_usd_tonne",
    })

    final_cols = [
        "area_code_m49", "iso3", "region_fao", "subregion_fao",
        "area_fao", "item_code_fao", "item_fao", "year",
        "price_usd_tonne", "usd_source",
        "usd_tonne_obs", "usd_fx_implied", "fx_source",
        "slc_filled", "slc_source", "lcu_filled", "lcu_source",
        "price_index",
    ]

    pp_final = pp_usd[final_cols].copy()

    csv_out = os.path.join(outdir, "fao_prices_1993_2024.csv")
    pq_out = os.path.join(outdir, "fao_prices_1993_2024.parquet")

    save_csv(pp_final, csv_out)
    save_parquet(pp_final, pq_out)

    logger.info("Final price panel: %d rows", len(pp_final))
    return pq_out


def _load_production_and_prices(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the production and price parquets from earlier pipeline steps."""
    prod_dir = cfg.outputs.fao_production
    prod_pq = os.path.join(prod_dir, "fao_production_1993_2024.parquet")
    logger.info("Loading production: %s", prod_pq)
    prod = pd.read_parquet(prod_pq)

    price_dir = cfg.outputs.fao_prices
    price_pq = os.path.join(price_dir, "fao_prices_1993_2024.parquet")
    logger.info("Loading prices: %s", price_pq)
    prices = pd.read_parquet(price_pq)

    return prod, prices


def _compute_median_prices(
    prices: pd.DataFrame,
    price_years: list[int],
    outdir: str,
    cw: pd.DataFrame,
) -> pd.DataFrame:
    """Compute median price per hierarchical level over the price window."""
    logger.info("Computing median prices over years %s", price_years)

    if "region_fao" not in prices.columns:
        prices = prices.merge(cw[["area_code_m49", "region_fao", "subregion_fao"]], on="area_code_m49", how="left")

    parent_frame = prices[
        prices["year"].isin(price_years)
        & prices["price_usd_tonne"].notna()
        & (prices["price_usd_tonne"] > 0)
    ].copy()

    price_country = (
        parent_frame.groupby(
            ["area_code_m49", "iso3", "region_fao", "subregion_fao", "item_code_fao", "item_fao"],
            as_index=False, dropna=False
        )
        .agg(
            price_median_usd_tonne=("price_usd_tonne", "median"),
            years_used=("year", "nunique"),
        )
    )
    price_country["agg_level"] = "country"

    price_subregion = (
        price_country.dropna(subset=["subregion_fao"])
        .groupby(["subregion_fao", "item_code_fao"], as_index=False)
        .agg(
            item_fao=("item_fao", "first"),
            price_median_usd_tonne=("price_median_usd_tonne", "median"),
            years_used=("years_used", "mean"),
        )
    )
    price_subregion["agg_level"] = "subregion"

    price_region = (
        price_country.dropna(subset=["region_fao"])
        .groupby(["region_fao", "item_code_fao"], as_index=False)
        .agg(
            item_fao=("item_fao", "first"),
            price_median_usd_tonne=("price_median_usd_tonne", "median"),
            years_used=("years_used", "mean"),
        )
    )
    price_region["agg_level"] = "region"

    price_world = (
        price_country.groupby("item_code_fao", as_index=False)
        .agg(
            item_fao=("item_fao", "first"),
            price_median_usd_tonne=("price_median_usd_tonne", "median"),
            years_used=("years_used", "mean"),
        )
    )
    price_world["agg_level"] = "world"
    price_world["area_code_m49"] = "999"
    price_world["iso3"] = "WOR"
    price_world["item_code_fao"] = price_world["item_code_fao"].astype(int)

    median_prices = pd.concat([price_country, price_subregion, price_region, price_world], ignore_index=True)

    os.makedirs(outdir, exist_ok=True)
    yr_start, yr_end = min(price_years), max(price_years)
    pq_out = outdir / f"price_median_usd_tonne_{yr_start}_{yr_end}.parquet"
    save_parquet(median_prices, pq_out)

    logger.info("Median prices: %d rows", len(median_prices))
    return median_prices


def _merge_production_value(
    prod: pd.DataFrame,
    price_country: pd.DataFrame,
    price_subregion: pd.DataFrame,
    price_region: pd.DataFrame,
    price_world: pd.DataFrame,
    outdir: str,
) -> str:
    """Merge production with annual prices (country -> subregion -> region -> world fallback)."""
    logger.info("Merging production × annual prices (Hierarchical)")

    prod_q = prod[prod["element"] == "Production"].copy()
    if "value" in prod_q.columns:
        prod_q = prod_q.rename(columns={"value": "total_production_tonnes"})

    # 1. Merge country prices
    merged = prod_q.merge(
        price_country[["year", "area_code_m49", "item_code_fao", "price_usd_tonne"]],
        on=["year", "area_code_m49", "item_code_fao"],
        how="left",
    )

    # 2. Merge subregion prices
    if "subregion_fao" in merged.columns:
        merged = merged.merge(
            price_subregion[["year", "subregion_fao", "item_code_fao", "subreg_price_usd_tonne"]],
            on=["year", "subregion_fao", "item_code_fao"],
            how="left"
        )
    else:
        merged["subreg_price_usd_tonne"] = np.nan

    # 3. Merge region prices
    if "region_fao" in merged.columns:
        merged = merged.merge(
            price_region[["year", "region_fao", "item_code_fao", "reg_price_usd_tonne"]],
            on=["year", "region_fao", "item_code_fao"],
            how="left"
        )
    else:
        merged["reg_price_usd_tonne"] = np.nan

    # 4. Merge world prices
    merged = merged.merge(
        price_world[["year", "item_code_fao", "world_price_usd_tonne"]],
        on=["year", "item_code_fao"],
        how="left"
    )

    # Apply fallbacks
    merged["price_source_agg_level"] = "country"
    
    # Subregion fallback
    miss_ctry = merged["price_usd_tonne"].isna()
    merged.loc[miss_ctry, "price_usd_tonne"] = merged.loc[miss_ctry, "subreg_price_usd_tonne"]
    merged.loc[miss_ctry & merged["subreg_price_usd_tonne"].notna(), "price_source_agg_level"] = "subregion"
    
    # Region fallback
    miss_subreg = merged["price_usd_tonne"].isna()
    merged.loc[miss_subreg, "price_usd_tonne"] = merged.loc[miss_subreg, "reg_price_usd_tonne"]
    merged.loc[miss_subreg & merged["reg_price_usd_tonne"].notna(), "price_source_agg_level"] = "region"
    
    # World fallback
    miss_reg = merged["price_usd_tonne"].isna()
    merged.loc[miss_reg, "price_usd_tonne"] = merged.loc[miss_reg, "world_price_usd_tonne"]
    merged.loc[miss_reg & merged["world_price_usd_tonne"].notna(), "price_source_agg_level"] = "world"
    
    # Missing everywhere
    merged.loc[merged["price_usd_tonne"].isna(), "price_source_agg_level"] = "missing"

    merged["value_usd"] = merged["total_production_tonnes"] * merged["price_usd_tonne"]
    
    # Cleanup
    merged = merged.drop(columns=["subreg_price_usd_tonne", "reg_price_usd_tonne", "world_price_usd_tonne"])

    logger.info(
        "Price sources: %s",
        merged.groupby("price_source_agg_level").size().to_dict()
    )
    logger.info("Rows with value_usd: %d", merged["value_usd"].notna().sum())

    os.makedirs(outdir, exist_ok=True)
    
    # Drop element column if present (it's always 'Production')
    if "element" in merged.columns:
        merged = merged.drop(columns=["element"])

    csv_out = os.path.join(outdir, "fao_values_1993_2024.csv")
    pq_out = os.path.join(outdir, "fao_values_1993_2024.parquet")

    save_csv(merged, csv_out)
    save_parquet(merged, pq_out)

    logger.info("Production value dataset: %d rows", len(merged))
    return pq_out


def run_fao_values(cfg: Config) -> str:
    """
    Compute total production value by merging prices and production.

    Returns the path to the output parquet file.
    """
    logger.info("=== FAO Values Pipeline ===")

    prod, prices = _load_production_and_prices(cfg)

    # Side-effect: generate smoothed median prices for rasters (2018-2022)
    cw = pd.read_csv(cfg.paths.crosswalk_m49_iso3)
    cw["area_code_m49"] = cw["area_code_m49"].astype(str).str.zfill(3)
    
    _ = _compute_median_prices(
        prices,
        cfg.run.price_years,
        cfg.outputs.fao_median_prices,
        cw,
    )

    # Main pipeline: use annual prices
    price_country, price_subregion, price_region, price_world = pf._compute_annual_prices(prices, cw)

    pq_path = _merge_production_value(
        prod, price_country, price_subregion, price_region, price_world, cfg.outputs.fao_values,
    )

    logger.info("=== FAO Values Pipeline COMPLETE ===")
    return pq_path


def load_m49_iso3(cfg: Config) -> Tuple[pd.DataFrame, Set[str]]:
    """
    Load the M49 -> ISO3 crosswalk and return *(df, valid_m49)*.

    The "area_code_m49" column is zero-filled to 3 chars to match the
    convention used throughout the FAO and raster pipelines.

    Parameters
    ----------
    cfg : Config

    Returns
    -------
    df : pd.DataFrame
        Columns include "area_code_m49", "iso3", "area_fao",
        "region_fao", "subregion_fao".
    valid_m49 : set[str]
        The set of valid 3-digit M49 strings.
    """
    path = cfg.paths.crosswalk_m49_iso3
    logger.info("Loading M49↔ISO3 crosswalk: %s", path)

    df = pd.read_csv(path)

    # Normalise M49 key
    df["area_code_m49"] = (
        df["area_code_m49"].astype(int).astype(str).str.zfill(3)
    )
    df["iso3"] = df["iso3"].astype(str)

    valid_m49 = set(df["area_code_m49"])
    logger.info("Valid M49 codes: %d", len(valid_m49))

    return df, valid_m49


def load_fao_classification(cfg: Config) -> pd.DataFrame:
    """
    Load the FAO crop classification table.

    Parameters
    ----------
    cfg : Config

    Returns
    -------
    pd.DataFrame
    """
    path = cfg.paths.fao_classification
    logger.info("Loading FAO classification: %s", path)
    return pd.read_csv(path)


def load_fao_cropgrids(cfg: Config) -> pd.DataFrame:
    """
    Load the crosswalk that maps CropGrids crop names to FAO item codes.

    Returns a DataFrame with columns "cropgrids_2024" and
    "item_code_fao" (as string), plus "item_fao".
    """
    path = cfg.paths.crosswalk_fao_cropgrids
    logger.info("Loading FAO↔CropGrids crosswalk: %s", path)

    df = pd.read_csv(path)

    df["item_code_fao"] = (
        df["item_code_fao"].astype("Int64").astype(str)
    )
    df["cropgrids_2024"] = df["cropgrids_2024"].astype(str)

    return df

def fao_median_prices(p):
    """Per-crop median producer prices in USD, downloaded and built here rather than taken as given.

    The pollination value raster is production times price times each crop's dependence on
    pollinators. The price half used to arrive as a finished table. This runs the path that makes
    it: FAOSTAT production and producer prices are downloaded, local currency is reconstructed
    where FAOSTAT reports only the discontinued series, USD is built against World Bank exchange
    rates, and a median is taken over the price years.

    Registered with skip_existing=1: it downloads several hundred megabytes from FAOSTAT and is
    deterministic once it has, the same reason erosion's SDR step skips.
    """
    publish_inputs(p)
    # A base-data-generating task: the prices are an input other services and other machines will
    # want, not a per-run result, so they are written under base data at a stable relative path.
    # get_path finds them there on any later run, on any machine that has synced it.
    p.fao_median_prices_dir = p.get_path(FAO_MEDIAN_PRICES_REF_PATH,
                                         possible_dirs=[p.base_data_dir],
                                         raise_error_if_fail=False)
    if not p.run_this:
        return
    if hb.path_exists(p.fao_median_prices_dir) and os.listdir(p.fao_median_prices_dir):
        hb.log('FAO median prices already in base data at %s. Skipping the download.'
               % p.fao_median_prices_dir)
        return True
    hb.create_directories(p.fao_median_prices_dir)
    settings = pf.FaoPriceSettings(
        crosswalk_m49_iso3_path=str(p.get_path(p.pollination_crosswalk_m49_iso3_path)),
        fao_classification_path=str(p.get_path(p.pollination_fao_classification_path)),
        crosswalk_fao_cropgrids_path=str(p.get_path(p.pollination_crosswalk_fao_cropgrids_path)),
        fao_production_bulk_path=str(p.pollination_fao_production_path),
        fao_prices_bulk_path=str(p.pollination_fao_prices_path),
        fx_lcu_per_usd_path=str(p.pollination_fx_path),
        output_dir=str(os.path.dirname(p.fao_median_prices_dir)))
    run_fao_production(settings)
    run_fao_prices(settings)
    run_fao_values(settings)
    hb.log('FAO median prices written to %s' % p.fao_median_prices_dir)
    return True


def read_raster(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read a single-band GeoTIFF.

    Returns
    -------
    data : np.ndarray
        2-D float32 array.
    meta : dict
        Rasterio profile (used to write matching outputs).
    """
    path = str(path)
    logger.debug("Reading raster: %s", path)

    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        meta = src.meta.copy()

    return data, meta


def save_csv(df: pd.DataFrame, path: str | str) -> str:
    """Save DataFrame as CSV and log the result."""
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %d rows -> %s", len(df), path)
    return path


def _process_tile(
    lulc_path: str | str,
    window: Window,
    height: int,
    width: int,
    bounds_top: float,
    transform_e: float,
    pixel_lat_deg: float,
    pixel_lon_deg: float,
    ag_classes: set[int],
    nat_classes: set[int],
    fill_non_ag: float | None = None,
    stable_ag_lulc_path: str | None = None,
) -> Tuple[Window, np.ndarray, int] | None:
    """Worker function to process a single tile.

    Parameters
    ----------
    fill_non_ag:
        When ``None`` (default) non-agricultural pixels are written as NaN
        (excluded from the 5 km average).  When set to a float (typically
        ``0.0``) every valid (non-nodata) non-agricultural pixel receives that
        value, so the 5 km average is weighted by the fraction of agricultural
        area in the cell — matching the methodology used in the original
        InVEST/GLOBIO pollination repos.
    stable_ag_lulc_path:
        When provided, the agricultural mask is restricted to pixels that are
        agricultural in *both* ``lulc_path`` and this path (intersection).
        Nature counts still come from ``lulc_path`` only.  Non-stable-ag pixels
        receive NaN regardless of ``fill_non_ag``.  Used by
        :func:`run_pollination_sufficiency_300m_stable_ag`.
    """
    h = window.height
    w = window.width

    # Open LULC locally in the worker process
    with rasterio.open(lulc_path) as src:
        lulc_nodata = src.nodata
        # Read core LULC
        lulc_core = src.read(1, window=window)

        # Identify AG pixels — optionally intersected with other-period LULC
        ag_mask = np.isin(lulc_core, list(ag_classes))
        if stable_ag_lulc_path is not None:
            with rasterio.open(stable_ag_lulc_path) as src2:
                lulc_other = src2.read(1, window=window)
            ag_mask = ag_mask & np.isin(lulc_other, list(ag_classes))

        if not ag_mask.any():
            if fill_non_ag is None:
                return None
            # fill_non_ag mode: return zeros for all valid (non-nodata) pixels
            # so they are included in the 5 km average and pull it toward zero.
            out_core = np.full((h, w), np.nan, dtype=np.float32)
            if lulc_nodata is not None:
                valid = lulc_core != int(lulc_nodata)
            else:
                valid = np.ones((h, w), dtype=bool)
            out_core[valid] = fill_non_ag
            return window, out_core, 0

        # Calculate kernel for this tile latitude
        # Window off is row_off, col_off
        r0 = int(window.row_off)
        
        lat_mid = _tile_mid_lat(bounds_top, transform_e, r0, h)
        ry, rx = _compute_radii_pixels(lat_mid, pixel_lat_deg, pixel_lon_deg)
        kernel = _make_elliptical_kernel(ry, rx)
        kernel_sum = int(kernel.sum())
        
        # Expand window for convolution context
        nat_win = _window_expand_aniso(window, pad_y=ry, pad_x=rx, height=height, width=width)
        lulc_nat = src.read(1, window=nat_win)
        nat_mask = np.isin(lulc_nat, list(nat_classes)).astype(np.uint8)
        
        # Convolve
        counts = convolve(nat_mask, weights=kernel, mode="constant", cval=0.0).astype(np.int32)
        
        # Extract core
        nat_row0 = int(nat_win.row_off)
        nat_col0 = int(nat_win.col_off)
        r_off = int(r0 - nat_row0)
        c0 = int(window.col_off)
        c_off = int(c0 - nat_col0)
        
        counts_core = counts[r_off:r_off + h, c_off:c_off + w]
        
        # Calc sufficiency
        denom = counts_core[ag_mask]
        valid = denom > 0
        
        frac = np.zeros_like(denom, dtype=np.float32)
        if valid.any():
            frac[valid] = denom[valid].astype(np.float32) / float(kernel_sum)
            
        suff_vals = np.minimum(frac / np.float32(_THRESHOLD), 1.0)
        
        # Write back
        out_core = np.full((h, w), np.nan, dtype=np.float32)

        if fill_non_ag is not None:
            # Fill valid non-ag pixels so they are included in the 5 km average.
            if lulc_nodata is not None:
                valid_pixels = lulc_core != int(lulc_nodata)
            else:
                valid_pixels = np.ones((h, w), dtype=bool)
            out_core[valid_pixels & ~ag_mask] = fill_non_ag

        out_core_ag = out_core[ag_mask]
        out_core_ag[:] = suff_vals
        out_core[ag_mask] = out_core_ag

        ag_count = int(ag_mask.sum())

        return window, out_core, ag_count


def run_pollination_sufficiency_300m(
    cfg: pf.SufficiencySettings,
    lulc_path: str | None = None,
    scenario: str = "2020",
    fill_non_ag: float | None = None,
    stable_ag_lulc_path: str | None = None,
) -> str:
    """Compute 300 m pollination sufficiency from ESA-CCI LULC or custom LULC map.

    For every agricultural pixel, counts natural-habitat neighbours within
    a latitude-adjusted elliptical kernel and normalises against _THRESHOLD.
    Returns the path to the output GeoTIFF.

    Parameters
    ----------
    fill_non_ag:
        Passed through to :func:`_process_tile`.  ``None`` (default) leaves
        non-agricultural pixels as NaN so the 5 km average is computed only
        over agricultural sub-pixels.  Pass ``0.0`` to use the original-repo
        approach where non-agricultural pixels contribute zero to the average,
        scaling the 5 km sufficiency by the agricultural area fraction.
    """
    lulc_path = lulc_path or cfg.lulc_path
    if not lulc_path or str(lulc_path) == ".":
        raise ValueError("lulc_esa path not defined in config (and no override provided)")
        
    out_dir = cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / f"pollination_sufficiency_{scenario}_300m.tif"
    
    logger.info("Computing 300m Pollination Sufficiency...")
    logger.info("Input LULC: %s", lulc_path)
    logger.info("Output: %s", out_path)
    
    # 0. Interpret LULC mapping
    if scenario == "2020":
        # ESA-CCI default classes
        ag_classes = _AG_CLASSES
        nat_classes = _NAT_CLASSES
        logger.info("Using default ESA-CCI LULC classes.")
    else:
        # SEALS future scenario classes
        # 1 = urban, 2 = cropland, 3 = grassland, 4 = forest, 5 = non-forest natural
        # 6 = water, 7 = barren land
        ag_classes = _SEALS_AG_CLASSES
        nat_classes = _SEALS_NAT_CLASSES
        logger.info("Using SEALS LULC classes.")
    
    if hb.path_exists(out_path):
        logger.info("Output already exists, skipping computation.")
        return out_path

    # 1. Open / Profile
    with rasterio.open(lulc_path) as src:
        profile = src.profile.copy()
        transform = src.transform
        bounds_top = src.bounds.top
        height, width = src.shape
        pixel_lat_deg = abs(transform.e)
        pixel_lon_deg = abs(transform.a)
        
        # Prepare output profile
        out_profile = profile.copy()
        out_profile.update(
            dtype="float32",
            nodata=np.nan,
            **pf.get_compression_profile(cfg, "continuous")
        )
        
    # 2. Init output with NaNs
    logger.info("Initializing output with NaNs...")
    with rasterio.open(out_path, "w", **out_profile) as dst:
        pass

    # 3. Compute Parallel
    tile_size = cfg.tile_size
    n_tiles_row = math.ceil(height / tile_size)
    n_tiles_col = math.ceil(width / tile_size)
    
    # Prepare Windows
    windows = []
    for tr in range(n_tiles_row):
        for tc in range(n_tiles_col):
            r0 = tr * tile_size
            c0 = tc * tile_size
            h = min(tile_size, height - r0)
            w = min(tile_size, width - c0)
            windows.append(Window(c0, r0, w, h))
            
    ag_pixels = 0
    
    # Generator for parallel results
    # n_jobs=-1 uses all cores. 
    # use return_as="generator" to yield results as they complete
    
    logger.info("Processing %d tiles with parallel execution...", len(windows))
    
    # Scan all agricultural pixels to measure the proportion of natural habitat within foraging range, creating a 300m sufficiency index grid
    _stable_ag_str = str(stable_ag_lulc_path) if stable_ag_lulc_path is not None else None
    results = Parallel(n_jobs=cfg.n_workers, return_as="generator")(
        delayed(_process_tile)(
            str(lulc_path),
            w,
            height,
            width,
            bounds_top,
            transform.e,
            pixel_lat_deg,
            pixel_lon_deg,
            ag_classes,
            nat_classes,
            fill_non_ag,
            _stable_ag_str,
        ) for w in windows
    )
    
    with rasterio.open(out_path, "w", **out_profile) as dst:
        with tqdm(total=len(windows), desc="Processing Tiles") as pbar:
            for res in results:
                if res is not None:
                    win, data, count = res
                    dst.write(data, 1, window=win)
                    ag_pixels += count

                
                pbar.update(1)
                    
    logger.info("Completed 300m sufficiency. Ag pixels processed: %d", ag_pixels)
    gc.collect()
    return out_path


def run_pollination_sufficiency_5km(cfg: pf.SufficiencySettings, input_300m: str | None = None, scenario: str = "2020") -> str:
    """Resample 300 m sufficiency to 5 km using average resampling.

    Matches the grid of "country_raster" (5 km template).
    Returns the path to the 5 km output GeoTIFF.
    """
    if input_300m is None:
        input_300m = cfg.output_dir / f"pollination_sufficiency_{scenario}_300m.tif"
        
    out_dir = cfg.output_dir
    out_path = out_dir / f"pollination_sufficiency_{scenario}_5km.tif"
    
    # Template target: We need a 5km grid.
    # The config has 'country_raster' which should be 5km.
    template_path = cfg.country_raster_path
    
    logger.info("Resampling sufficiency to 5km...")
    logger.info("Input: %s", input_300m)
    logger.info("Template: %s", template_path)
    logger.info("Output: %s", out_path)
    
    if hb.path_exists(out_path):
        logger.info("Output already exists, skipping resampling.")
        return out_path

    if not hb.path_exists(input_300m):
        raise FileNotFoundError(f"300m sufficiency raster not found at {input_300m}")
        
    with rasterio.open(template_path) as tgt:
        tgt_profile = tgt.profile.copy()
        tgt_transform = tgt.transform
        tgt_crs = tgt.crs
        tgt_shape = (tgt.height, tgt.width)
        
    tgt_profile.update(
        dtype="float32",
        nodata=np.nan,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        predictor=2,
        bigtiff="if_safer"
    )

    with rasterio.open(input_300m) as src:
        # Downsample the highly detailed 300m sufficiency map to a generalized 5km grid utilizing averaging to match crop production scales
        # Reproject using Average resampling (ignoring nodata)
        # This gives average sufficiency of ARABLE land in the cell
        data_5km = np.empty(tgt_shape, dtype=np.float32)
        
        reproject(
            source=rasterio.band(src, 1),
            destination=data_5km,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=tgt_transform,
            dst_crs=tgt_crs,
            dst_nodata=np.nan,
            resampling=Resampling.average
        )
        
    with rasterio.open(out_path, "w", **tgt_profile) as dst:
        dst.write(data_5km, 1)
        
    logger.info("Created 5km sufficiency raster.")
    return out_path


def run_pollination_sufficiency_300m_stable_ag(
    cfg: pf.SufficiencySettings,
    lulc_path: str,
    other_lulc_path: str,
    scenario: str,
) -> str:
    """Compute 300 m sufficiency restricted to pixels agricultural in both periods.

    A thin wrapper around :func:`run_pollination_sufficiency_300m` that
    restricts the agricultural mask to the *intersection* of the two LULC
    rasters: a pixel is treated as agricultural only if it carries an
    agricultural code in **both** ``lulc_path`` (the primary LULC used for
    nature counting) and ``other_lulc_path`` (the other-period LULC).

    - Nature neighbours are counted from ``lulc_path`` only (correct: each
      period's sufficiency reflects that period's surrounding landscape).
    - Non-stable-ag pixels receive NaN and are excluded from the 5 km average.

    This closes the sub-pixel gap in the PNAS replication: by restricting
    both the baseline and scenario sufficiency rasters to the same stable-ag
    mask before averaging to 5 km, the 5 km diff captures only changes in
    sufficiency on continuously farmed land — exactly as the original
    ``pollination_shock()`` function did at 300 m.

    Parameters
    ----------
    lulc_path:
        Primary LULC raster.  Used for both ag classification and nature
        counting.
    other_lulc_path:
        The other period's LULC raster.  Only its ag mask is used (nature
        is NOT counted from here).
    scenario:
        Output label; determines the output filename.

    Returns
    -------
    Path
        Path to ``pollination_sufficiency_{scenario}_300m.tif``.
    """
    return run_pollination_sufficiency_300m(
        cfg,
        lulc_path=lulc_path,
        scenario=scenario,
        fill_non_ag=None,           # NaN for non-stable-ag (exclude from average)
        stable_ag_lulc_path=other_lulc_path,
    )


def run_pollination_valuation_5km(cfg: pf.SufficiencySettings, scenario: str = "2020", target_year: int = 2024) -> str:
    """Compute 5 km pollination-value rasters (Value × Sufficiency).

    Uses the pre-calculated global pollination value raster (from build_pollination_value.py)
    and multiplies it by the 5 km sufficiency index.
    """
    suff_path = cfg.output_dir / f"pollination_sufficiency_{scenario}_5km.tif"

    # Input: Pre-calculated global pollination value
    poll_value_path = cfg.value_raster_dir / f"poll_value_global_{target_year}usd.tif"

    out_dir = cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)
    
    out_total = out_dir / f"value_pollination_sufficiency_{scenario}_5km.tif"

    logger.info("Computing 5km Pollination Value (weighted by sufficiency, %d USD)...", target_year)
    logger.info("  Pollination Value Input: %s", poll_value_path)
    logger.info("  Sufficiency Input      : %s", suff_path)
    
    if hb.path_exists(out_total):
        logger.info("Output already exists, skipping valuation.")
        return out_total

    if not hb.path_exists(poll_value_path):
        raise FileNotFoundError(f"Global pollination value raster missing: {poll_value_path}")
    if not hb.path_exists(suff_path):
        raise FileNotFoundError(f"Sufficiency raster missing: {suff_path}")

    # Load both rasters
    with rasterio.open(poll_value_path) as src_val, rasterio.open(suff_path) as src_suff:
        # Safety checks
        if src_val.shape != src_suff.shape:
             raise RuntimeError(f"Shape mismatch: Val {src_val.shape} vs Suff {src_suff.shape}")
        
        # Read and mask
        val_data = src_val.read(1).astype(np.float32)
        suff_data = src_suff.read(1).astype(np.float32)
        
        # Handle nodata
        val_nodata = src_val.nodata
        suff_nodata = src_suff.nodata
        
        if val_nodata is not None:
             val_data[val_data == val_nodata] = np.nan
        if suff_nodata is not None:
             suff_data[suff_data == suff_nodata] = np.nan
             
        # Calculate: Intersect baseline pollination values with the corresponding sufficiency index (Value * Sufficiency)
        
        result = val_data * suff_data
        
        # Output profile from source
        profile = src_val.profile.copy()
        profile.update(dtype="float32", nodata=-9999.0, compress="lzw")
        
        # Fill NaN with nodata for output
        out_data = np.nan_to_num(result, nan=-9999.0)
        
        with rasterio.open(out_total, "w", **profile) as dst:
             dst.write(out_data, 1)
             
    logger.info("Saved sufficiency-weighted value: %s", out_total)
    return out_total


def redistribution_value_to_300m(cfg: pf.SufficiencySettings, scenario: str = "2020") -> str:
    """Redistribute 5 km pollination value back to 300 m pixels.

    Each 300 m pixel receives a share of its parent 5 km cell's value
    proportional to its sufficiency weight (suff_i / Σ suff).
    """
    large_path = cfg.output_dir / f"value_pollination_sufficiency_{scenario}_5km.tif"
    fine_path = cfg.output_dir / f"pollination_sufficiency_{scenario}_300m.tif"
    out_path = cfg.output_dir / f"value_pollination_sufficiency_{scenario}_300m.tif"

    logger.info("Redistributing value to 300m...")

    if hb.path_exists(out_path):
        logger.info("Output already exists, skipping redistribution.")
        return out_path

    if not hb.path_exists(large_path):
        raise FileNotFoundError(f"5km value raster missing: {large_path}")
    
    with rasterio.open(large_path) as large, rasterio.open(fine_path) as fine:
        large_vals = large.read(1).astype(np.float64)
        if large.nodata is not None:
             large_vals = np.where(large_vals == large.nodata, 0, large_vals)
        else:
             large_vals = np.nan_to_num(large_vals, nan=0.0)

        # Area ratio: large pixel area / fine pixel area (in degree² units; cos(lat) cancels).
        # Multiplying by this converts the redistributed fraction back to USD/km² so that
        # summing val_300m * area_300m recovers the same total as summing val_5km * area_5km.
        area_ratio = (abs(large.transform.a) * abs(large.transform.e)) / (
            abs(fine.transform.a) * abs(fine.transform.e)
        )
        logger.info("  Area ratio (5km/300m pixel): %.2f", area_ratio)

        profile = fine.profile.copy()
        profile.update(dtype="float32", nodata=0.0, **pf.get_compression_profile(cfg, "continuous"))

        block_size = cfg.tile_size

        with rasterio.open(out_path, "w", **profile) as dst:
             n_blocks = math.ceil(fine.height / block_size) * math.ceil(fine.width / block_size)

             # Downscale the 5km aggregated pollination value back into smaller 300m sections proportionally based on local weights
             with tqdm(total=n_blocks, desc="300m Redistribution") as pbar:
                for br in range(0, fine.height, block_size):
                    bh = min(block_size, fine.height - br)
                    for bc in range(0, fine.width, block_size):
                        bw = min(block_size, fine.width - bc)
                        window = Window(bc, br, bw, bh)

                        suff = fine.read(1, window=window).astype(np.float64)
                        mask = suff > 0
                        weights = suff[mask]

                        if weights.size == 0:
                            dst.write(np.zeros((bh, bw), dtype=np.float32), 1, window=window)
                            pbar.update(1)
                            continue

                        # Centroids
                        fine_r, fine_c = np.where(mask)
                        # Transform to coords
                        xs, ys = rasterio.transform.xy(fine.transform, br + fine_r, bc + fine_c, offset='center')
                        xs = np.array(xs)
                        ys = np.array(ys)

                        # Parent indices
                        # row = (y - top) / dy, col = (x - left) / dx
                        parent_c = np.floor((xs - large.transform.c) / large.transform.a).astype(int)
                        # large.transform.e is usually negative
                        parent_r = np.floor((ys - large.transform.f) / large.transform.e).astype(int)

                        parent_r = np.clip(parent_r, 0, large.height - 1)
                        parent_c = np.clip(parent_c, 0, large.width - 1)

                        # Vectorized attribution
                        # We need to group by parent pixel to calculate sum_weights per parent
                        flat_idx = parent_r * large.width + parent_c

                        unique_idx, input_idx = np.unique(flat_idx, return_inverse=True)
                        parent_vals = large_vals.flatten()[unique_idx]

                        sum_weights = np.zeros(unique_idx.size, dtype=np.float64)
                        np.add.at(sum_weights, input_idx, weights)

                        # Redistributions
                        # val_i = (weight_i / sum_weights_parent) * val_parent * area_ratio
                        # area_ratio converts the fractional density back to USD/km² so that
                        # Σ(val_300m * area_300m) == Σ(val_5km * area_5km).
                        # Avoid div by zero
                        divisor = sum_weights[input_idx]
                        valid_div = divisor > 0

                        res = np.zeros_like(weights)
                        if valid_div.any():
                             res[valid_div] = (weights[valid_div] / divisor[valid_div]) * parent_vals[input_idx][valid_div] * area_ratio

                        out_block = np.zeros((bh, bw), dtype=np.float32)
                        out_block[mask] = res.astype(np.float32)

                        dst.write(out_block, 1, window=window)
                        pbar.update(1)
                        
    return out_path


def mask_protected_areas_300m(cfg: pf.SufficiencySettings, scenario: str = "2020") -> str:
    """
    Mask pollination value inside protected areas (PA == 1 -> value = 0).

    Reads the 300m pollination-value raster and a binary PA raster,
    zeroes all pixels that fall inside a protected area, and writes
    a masked raster together with a summary CSV.

    Returns
    -------
    Path
        Path to the masked output raster.
    """
    value_path = (
        cfg.output_dir
        / f"value_pollination_sufficiency_{scenario}_300m.tif"
    )
    pa_path = cfg.pa_raster_300m_path
    out_raster = (
        cfg.output_dir
        / f"value_pollination_sufficiency_{scenario}_300m_no_agri_in_PA.tif"
    )
    out_csv = (
        cfg.output_dir
        / f"summary_300m_no_agri_in_PA_{scenario}.csv"
    )

    logger.info("Masking PA pixels from 300m pollination value...")
    logger.info("Value input : %s", value_path)
    logger.info("PA raster   : %s", pa_path)

    if hb.path_exists(out_raster):
        logger.info("Output already exists, skipping PA masking.")
        return out_raster

    if not hb.path_exists(value_path):
        raise FileNotFoundError(f"300m value raster missing: {value_path}")
    if not hb.path_exists(pa_path):
        raise FileNotFoundError(f"PA raster missing: {pa_path}")

    with rasterio.open(value_path) as val_src, rasterio.open(pa_path) as pa_src:
        # Safety checks
        if val_src.transform != pa_src.transform:
            raise RuntimeError("Transforms do not match")
        if val_src.crs != pa_src.crs:
            raise RuntimeError("CRS does not match")
        if val_src.width != pa_src.width or val_src.height != pa_src.height:
            raise RuntimeError("Raster shapes do not match")

        profile = val_src.profile.copy()
        profile.update(dtype="float32", nodata=0, **pf.get_compression_profile(cfg, "continuous"))

        transform = val_src.transform
        pixel_lat_deg = abs(transform.e)
        pixel_lon_deg = abs(transform.a)
        bounds_top = val_src.bounds.top

        total_before = 0.0
        total_after = 0.0

        with rasterio.open(out_raster, "w", **profile) as dst:
            # We process block by block to avoid loading full raster
            # For each block, exclude or zero-out values located strictly inside designated protected Ecological Boundaries
            # Force block processing based on output structure
            for _, window in tqdm(list(dst.block_windows(1)), desc="Masking PA"):
                # Read
                value = val_src.read(1, window=window).astype(np.float64)
                pa = pa_src.read(1, window=window)

                # Check if we have data
                if value.size == 0:
                    continue

                # Normalize nodata -> NaN
                if val_src.nodata is not None:
                    value[value == val_src.nodata] = np.nan

                # Cell area per ROW, WGS84, so this diagnostic measures a cell the way the
                # value raster and the rest of the account do. The vendored version used a flat
                # 111.32 km per degree AND one mid-tile latitude for the whole tile height, so a
                # tall tile got a single area for rows hundreds of kilometres apart.
                row_off = window.row_off
                tile_h = window.height
                row_lats = bounds_top - (np.arange(row_off, row_off + tile_h) + 0.5) * pixel_lat_deg
                area_km2 = np.asarray(hb.get_area_of_pixel_column_from_center_lats(
                    pixel_lat_deg, row_lats.astype('float64')))[:, None] / 1e6

                # Sum before (USD/km² * km² = USD, ignoring NaNs)
                total_before += float(np.nansum(value * area_km2))

                # Mask: PA == 1 -> 0
                # Assuming PA=1 is protected
                if pa_src.nodata is not None:
                    pass

                value[pa == 1] = 0.0

                # Sum after
                total_after += float(np.nansum(value * area_km2))

                # Write masked raster (NaN -> 0, nodata = 0)
                out_data = np.nan_to_num(value, nan=0.0).astype(np.float32)
                dst.write(out_data, 1, window=window)

    removed = total_before - total_after
    pct_removed = (removed / total_before * 100) if total_before > 0 else 0.0

    # Summary CSV
    summary = pd.DataFrame([{
        "total_value_before_mask": round(total_before, 2),
        "total_value_after_mask": round(total_after, 2),
        "value_removed_by_PA": round(removed, 2),
        "percent_removed": round(pct_removed, 4),
    }])
    save_csv(summary, out_csv)

    logger.info("PA masking complete.")
    logger.info("  Before: %.2f  After: %.2f  Removed: %.4f%%",
                total_before, total_after, pct_removed)
    logger.info("Output:  %s", out_raster)
    logger.info("Summary: %s", out_csv)

    return out_raster


def summarize_run(cfg: pf.SufficiencySettings, scenario: str = "2020") -> None:
    """Summarize total values of all generated valuation rasters."""
    logger = logging.getLogger("poll_suff_pipeline")
    out_dir = cfg.output_dir
    summary_path = out_dir / f"pollination_sufficiency_valuation_totals_{scenario}.csv"
    
    # Rasters to check (Name -> Description)
    targets = {
        f"value_pollination_sufficiency_{scenario}_5km.tif": "5km Valuation (ag_value * sufficiency)",
        f"value_pollination_sufficiency_{scenario}_300m.tif": "300m Valuation (redistributed)",
        f"value_pollination_sufficiency_{scenario}_300m_no_agri_in_PA.tif": "300m Valuation (Masked: No Ag in PA)",
        f"nature_value_pollination_{scenario}_300m.tif": "Nature Attribution (Value provided by Nature pixels)",
        f"pa_value_pollination_{scenario}_300m.tif": "PA Analysis (Value provided by PA pixels)"
    }
    
    rows = []
    
    for fname, desc in targets.items():
        path = out_dir / fname
        if not hb.path_exists(path):
            continue
            
        try:
            arr, meta = read_raster(path)
            
            # Mask nodata
            nodata = meta.get("nodata")
            if nodata is not None:
                arr[arr == nodata] = np.nan
            
            arr[arr < 0] = np.nan

            arr = arr.astype(np.float64) # Ensure precision for sum
          
            area_km2 = pf.build_area_km2_raster(meta)
            mass = pf.convert_density_to_mass(arr, area_km2)
            total_usd = float(np.nansum(mass))
            
            rows.append({
                "raster_filename": fname,
                "description": desc,
                "total_value_usd2024": total_usd,
                "path": str(path)
            })
            logger.info(f"Summary: {fname} = {total_usd/1e9:.3f} B USD")
            
        except Exception as e:
            logger.error(f"Failed to summarize {fname}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(summary_path, index=False)
        logger.info(f"Saved valuation totals to: {summary_path}")


def run_pollination_diff_5km_pnas(
    cfg: pf.SufficiencySettings,
    scenario: str,
    baseline_scenario: str = "2023_pnas",
    scenario_value_path: str | None = None,
    baseline_value_path: str | None = None,
) -> str:
    """Compute a 5 km difference raster restricted to stable cropland.

    Replicates the behaviour of ``pollination_shock()`` in the original
    InVEST/GLOBIO repos: only changes in pollination sufficiency on
    continuously farmed land are captured.

    Two modes:

    **5 km mask mode** (default when ``scenario_value_path`` is ``None``):
        Uses the pre-built stable-ag mask at 5 km to zero out cells where
        land converted.  Fast but approximate — partial conversions within a
        5 km cell are not handled at sub-pixel resolution.

    **Exact sub-pixel mode** (when explicit paths are provided):
        ``scenario_value_path`` and ``baseline_value_path`` point to value
        rasters already computed from stable-ag-restricted sufficiency (via
        :func:`run_pollination_sufficiency_300m_stable_ag`).
        The 5 km averages in those rasters already exclude converted sub-pixels,
        so no additional masking is needed — a plain subtraction is sufficient.
        This exactly matches the 300 m pixel-level masking of the original repos.

    Parameters
    ----------
    scenario:
        Label of the future LULC scenario.  Also determines the output filename.
    baseline_scenario:
        Label of the baseline scenario.  Also determines the output filename.
    scenario_value_path:
        Override the scenario value raster path (exact sub-pixel mode).
    baseline_value_path:
        Override the baseline value raster path (exact sub-pixel mode).

    Returns
    -------
    Path
        ``diff_value_pollination_sufficiency_{scenario}_vs_{baseline_scenario}_5km.tif``
    """
    out_dir  = cfg.output_dir
    out_path = out_dir / f"diff_value_pollination_sufficiency_{scenario}_vs_{baseline_scenario}_5km.tif"

    logger.info(
        "Computing PNAS-style difference raster (stable ag only): %s − %s",
        scenario, baseline_scenario,
    )

    if hb.path_exists(out_path):
        logger.info("Output already exists, skipping: %s", out_path)
        return out_path

    # Resolve input paths
    s_path = scenario_value_path  or (out_dir / f"value_pollination_sufficiency_{scenario}_5km.tif")
    b_path = baseline_value_path  or (out_dir / f"value_pollination_sufficiency_{baseline_scenario}_5km.tif")
    exact_mode = scenario_value_path is not None

    required = [s_path, b_path]
    if not exact_mode:
        mask_path = out_dir / f"stable_ag_mask_{scenario}_vs_{baseline_scenario}_5km.tif"
        required.append(mask_path)

    for p in required:
        if not hb.path_exists(p):
            raise FileNotFoundError(f"Required input missing: {p}")

    with rasterio.open(s_path) as src_s, rasterio.open(b_path) as src_b:
        if src_s.shape != src_b.shape:
            raise RuntimeError(
                f"Shape mismatch: scenario {src_s.shape} vs baseline {src_b.shape}"
            )
        scen_data = src_s.read(1).astype(np.float32)
        base_data = src_b.read(1).astype(np.float32)
        if src_s.nodata is not None:
            scen_data[scen_data == src_s.nodata] = np.nan
        if src_b.nodata is not None:
            base_data[base_data == src_b.nodata] = np.nan

        diff = scen_data - base_data
        profile = src_s.profile.copy()

    if not exact_mode:
        # 5 km mask mode: zero out cells not agricultural in both periods
        with rasterio.open(mask_path) as src_m:
            stable = src_m.read(1).astype(np.float32)
            if src_m.nodata is not None:
                stable[stable == src_m.nodata] = np.nan
        diff = np.where(stable == 1.0, diff, 0.0).astype(np.float32)
    # exact mode: no masking needed — converted sub-pixels already excluded
    # from the stable-ag sufficiency averages that produced s_path / b_path.

    profile.update(dtype="float32", nodata=-9999.0, compress="lzw")
    out_data = np.nan_to_num(diff, nan=-9999.0)

    os.makedirs(out_dir, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out_data, 1)

    logger.info("Saved PNAS-style difference raster: %s", out_path)
    return out_path


def publish_inputs(p):
    """Every GEP task's first line: the pollination es_config row (defaults layer -- a caller-set
    value wins) plus the shared country references and the results registry. gep_base_year and
    the value raster (gep_quantity_input_path) are a PAIR defaulting to 2023 -- the source
    rasters on hand; the GEP manuscript's base year is 2019, so regenerate
    poll_value_global_2019usd.tif with the recipe in pollination_sufficiency and update BOTH cells (the row
    names the year twice) before quoting a manuscript-aligned number."""
    utilities.hydrate_es_config(p, 'pollination', log=hb.log)
    utilities.hydrate_es_parameters(p, 'pollination', log=hb.log)
    utilities.initialize_country_paths(p)
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def _read_masked(raster_path):
    """A single-band raster as float64 with its nodata read as NaN, and its metadata beside it."""
    import rasterio
    with rasterio.open(raster_path) as src:
        arr = src.read(1).astype(np.float64)
        arr[arr == src.nodata] = np.nan
        return arr, src.meta.copy()


def _zonal_context(denominator_path, correspondence_gpkg):
    """The fixed side of every zonal percent change, read once.

    The baseline value raster defines the grid, so the pixel-area raster and the burned zone ids
    are built on it. All three plus the zone labels are handed to pf.zonal_pct_change with each
    scenario's difference array.
    """
    import geopandas as gpd
    from rasterio.features import rasterize

    gdf = gpd.read_file(correspondence_gpkg, engine='pyogrio')
    if gdf.crs is None or gdf.crs.to_epsg() != pf.LATLON_EPSG:
        gdf = gdf.to_crs(pf.LATLON_EPSG)
    gdf[pf.REGION_ID_FIELD] = gdf[pf.REGION_ID_FIELD].astype(int)

    baseline, meta = _read_masked(denominator_path)
    zones = rasterize(
        ((g, int(r)) for g, r in zip(gdf.geometry, gdf[pf.REGION_ID_FIELD]) if g is not None and not g.is_empty),
        out_shape=(meta['height'], meta['width']), transform=meta['transform'],
        fill=pf.NO_ZONE_ID, dtype=np.int32)
    return baseline, pf.build_area_km2_raster(meta), zones, pf.zone_labels_from_boundary(gdf)


def pollination_shock(p):
    """Per-scenario 300 m LULC at each SEALS anchor year -> V_F/OSD shock, piecewise-interp to annual.

    Caller sets on p: es_shock_years (SEALS anchor years, from seals_years),
    es_shock_base_year, es_shock_scenarios, es_lulc_path_template
    ({scenario}/{year}) or scenario_lulc_paths, pollination_base_year_lulc_path (or the shared es_base_year_lulc_path),
    pollination_shock_output_path. Optional: es_shock_base_scenario, pollination_shock_acts,
    region_boundary_path.
    """
    # Default into the es_shocks parent dir. Runtime, not build time: p.es_shock_dir is
    # published by that task, which ProjectFlow runs before this one.
    if not getattr(p, 'pollination_shock_output_path', None):
        p.pollination_shock_output_path = os.path.join(getattr(p, 'es_shock_dir', None) or p.project_dir, 'pollination_interpolated.csv')
    if not p.run_this:
        return
    # Export keys and boundary are es_parameters shipped defaults; a consumer's own tables win.
    utilities.hydrate_es_parameters(p, 'pollination', log=hb.log)
    base_scenario      = utilities.required_base_scenario(p, 'pollination')
    es_shock_base_year = int(p.es_shock_base_year)
    es_shock_scenarios = list(p.es_shock_scenarios)      # was read unbound: this task raised NameError
    anchor_years = sorted(y for y in map(int, p.es_shock_years) if y > es_shock_base_year)

    cfg = pf.configure_sufficiency(p, es_shock_base_year)              # settings -> our base_data + task dir

    if not getattr(p, 'scenario_lulc_paths', None):
        tmpl = p.es_lulc_path_template
        p.scenario_lulc_paths = {s: {y: glob.glob(tmpl.format(scenario=s, year=y))[0]
                                     for y in anchor_years if glob.glob(tmpl.format(scenario=s, year=y))}
                                 for s in [base_scenario] + es_shock_scenarios}

    # base-year SEALS7 map: per-ES override, else the ES-shared attr. NEVER p.base_year_lulc_path, which
    # SEALS OWNS and overwrites at runtime with its raw-ESA source (a raw-ESA base map would make the
    # SEALS7-keyed sufficiency lookup produce garbage). This base-year value IS pollination's primary
    # denominator (it feeds both fixedbase and contemp), so a missing one is fatal.
    base_map = getattr(p, 'pollination_base_year_lulc_path', None) or getattr(p, 'es_base_year_lulc_path', None)
    if not base_map:
        raise ValueError('pollination base-year LULC not set: point p.es_base_year_lulc_path (or '
                         'p.pollination_base_year_lulc_path) at the SEALS7 base-year map.')
    # The denominator (unpaired 2023 value) is year- and scenario-independent, so the fixed side of
    # the zonal step is built once.
    denominator_path = baseline_denominator(cfg, base_map, es_shock_base_year)
    baseline_arr, area_arr, zones_arr, zone_labels = _zonal_context(denominator_path, p.region_boundary_path)

    # value[scenario][year] = per-zone % change of that scenario's year-map vs the 2023 baseline (stable
    # ag). level_usd = the denominator of that % change, the per-zone absolute baseline value in base-year
    # USD, emitted so the GEP chain can consume this task instead of rerunning the same rasters.
    value, level_usd = {}, None
    for year in anchor_years:
        for scen in [base_scenario] + es_shock_scenarios:
            diff_arr, _ = _read_masked(scenario_diff_raster(
                cfg, scenario=f'{scen}_{year}', lulc_path=p.scenario_lulc_paths[scen][year],
                baseline_lulc_path=base_map, target_year=es_shock_base_year))
            pct, level = pf.zonal_pct_change(diff_arr, baseline_arr, area_arr, zones_arr, zone_labels)
            value.setdefault(scen, {})[year] = pct
            if level_usd is None:
                level_usd = level

    # The shock numerator is scenario minus nature-off baseline at each anchor. anchor_shock_tables
    # puts it over the two denominators and dynamic_shock_rows expands those to annual rows.
    rows = []
    for scen in es_shock_scenarios:
        anchor_shock, anchor_contemp = pf.anchor_shock_tables(
            {y: value[scen][y] for y in anchor_years},
            {y: value[base_scenario][y] for y in anchor_years})
        rows += pf.dynamic_shock_rows(anchor_shock, anchor_contemp, level_usd, scen,
                                      p.pollination_shock_acts, es_shock_base_year)

    out = pd.DataFrame(rows)
    utilities.assert_shock_table_sound(out, es_shock_scenarios, 'pollination')
    out.to_csv(p.pollination_shock_output_path, index=False)
    print('  pollination shock: %d rows, %d scenarios (shock_pct=shock_pct_contemp=/baseline-year value, shock_pct_fixedbase=/2023 value) -> %s'
          % (len(out), out['scenario'].nunique() if rows else 0, p.pollination_shock_output_path))
    # value_usd_base is the GEP hand-off, not read by GTAP (build_combined_afeall takes shock_pct only).
    if level_usd is not None and len(level_usd):
        print('  pollination value (GEP): %d zones, total %.4g base-year USD -> column value_usd_base'
              % (len(level_usd), float(level_usd.sum())))
    return True


def pollination_shock_static(p):
    """Static per-scenario pollination shock -> V_F/OSD, linear ramp 0->es_shock_end_year, from the frozen table.

    The fallback add_pollination_tasks selects when <2 SEALS map years exist. READS
    input_dir/raw_dependencies/pollination_dependency.csv (override p.pollination_dependency_path) and
    subtracts the baseline_ignore_damages row (the frozen table's nature-off baseline; == ignore
    dependencies, just the old label kept in this CSV), ramping that difference linearly from 0 at
    es_shock_base_year. NEVER writes back to raw_dependencies -- output goes to p.pollination_shock_output_path
    (pollination_interpolated.csv), the same file the dynamic task writes. Caller sets:
    es_shock_base_year, es_shock_end_year, es_shock_scenarios,
    pollination_shock_output_path; scenario->raw via p.pollination_scenario_map (default: identity --
    each scenario maps to its own name; a scenario the table labels differently is warned about loudly
    and skipped rather than silently zeroed, so set the map for those);
    sectors via p.pollination_shock_acts (default ('V_F', 'OSD'), matching the dynamic task).
    """
    # Default into the es_shocks parent dir. Runtime, not build time: p.es_shock_dir is
    # published by that task, which ProjectFlow runs before this one.
    if not getattr(p, 'pollination_shock_output_path', None):
        p.pollination_shock_output_path = os.path.join(getattr(p, 'es_shock_dir', None) or p.project_dir, 'pollination_interpolated.csv')
    if not p.run_this:
        return

    utilities.hydrate_es_parameters(p, 'pollination', log=hb.log)   # shipped defaults; caller wins
    es_shock_base_year = int(p.es_shock_base_year)
    es_shock_end_year = int(p.es_shock_end_year)
    pollination_scenario_map = getattr(p, 'pollination_scenario_map', {})
    es_shock_scenarios = list(p.es_shock_scenarios)

    poll_path = getattr(p, 'pollination_dependency_path', None) or os.path.join(
        p.input_dir, 'raw_dependencies', 'pollination_dependency.csv')
    if not hb.path_exists(poll_path):
        print('  pollination shock: dependency csv not found (%s) -- skipping' % poll_path)
        return

    df = hb.df_read(poll_path)
    # Normalise the one year-suffixed label so pollination's table presents the same scenario names as
    # the other services (carbon is plain everywhere, erosion strips '_2050' on read). net_zero_2050 is
    # the only alias here, so target it explicitly rather than a blunt '_2050' strip that would silently
    # mangle any future label carrying that suffix. Keeps the consumer's pollination_scenario_map at identity.
    df['scenario'] = df['scenario'].replace({'net_zero_2050': 'net_zero'})
    df = df[df['ENDW'] != 'AEZ0']  # AEZ0 not valid in GTAP
    # The base resolves through the candidate mechanism (fatal if absent): the table's spelling
    # may differ from the configured name (nature-off aliasing in utilities).
    base_scenario = utilities.required_base_scenario(p, 'pollination')
    raw_base = utilities.resolve_base_scenario(df['scenario'].values, pollination_scenario_map, base_scenario, 'pollination')
    base = df[df['scenario'] == raw_base].set_index(['ENDW', 'REG'])['value'].astype(float)

    rows = []
    for our_scn in es_shock_scenarios:
        raw_scn = utilities.resolve_raw_scenario(df['scenario'].values, pollination_scenario_map, our_scn, 'pollination')
        if raw_scn is None:
            continue
        scn_vals = df[df['scenario'] == raw_scn].set_index(['ENDW', 'REG'])['value'].astype(float)
        rows += pf.static_shock_rows(base, scn_vals, our_scn, p.pollination_shock_acts,
                                     es_shock_base_year, es_shock_end_year)

    out = pd.DataFrame(rows)
    utilities.assert_shock_table_sound(out, es_shock_scenarios, 'pollination')
    out.to_csv(p.pollination_shock_output_path, index=False)
    nz = out[(out['year'] == es_shock_end_year) & (out['shock_pct'] != 0)] if len(out) else out
    print('  pollination shock: %d rows, %d scenarios, %d nonzero @%d (static, uncapped) -> %s'
          % (len(out), out['scenario'].nunique() if len(out) else 0, len(nz), es_shock_end_year,
             p.pollination_shock_output_path))
    return True


# =============================================================================
# GEP valuation tasks. The ES shock above and the valuation below are separate
# consumers of the same crop_benefits science; neither depends on the other.
# The value raster (poll_value_global_<year>usd.tif, a crop_benefits output) is
# USD per cell with crop prices and pollination dependence already embedded
# upstream, so lambda = 1 and no price join happens here: quantity IS value.
# =============================================================================

def pollination_value_by_region(p):
    """GEP quantity stage: per-region sum of the pollination value raster (USD per cell) on the
    service's aggregation surface (the gep_regions cells)."""
    publish_inputs(p)
    p.pollination_value_by_region_path = os.path.join(p.cur_dir, "pollination_value_by_region.csv")
    if not p.run_this:
        return
    # Our own raster, in USD per cell, so this sum is a sum of money. It used to be
    # p.gep_quantity_input_path, a density from elsewhere, which made the total 26x too small.
    utilities.summarize_raster_by_region(
        value_raster_path=p.pollination_value_raster_path,
        region_boundary_path=p.gep_regions_input_path,
        out_path=p.pollination_value_by_region_path,
        year=p.gep_base_year, id_column=p.gep_regions_id_col)
    # Full-scale conservation: the region sums must add up to the raster's own total
    # (verified at 100.0000% on the real raster when this check was added).
    df_regions = hb.df_read(p.pollination_value_by_region_path)
    utilities.assert_zonal_conservation(df_regions['total'].sum(),
                                        p.pollination_value_raster_path, 'pollination')
    return True


def gep_calculation(p):
    """GEP valuation for pollination: r264 region values -> ONE row per country (r250)."""
    publish_inputs(p)
    service_results, already_done = utilities.begin_gep_calculation(p, 'pollination')
    if already_done:
        return

    # 1. Per-region (r264) USD value -> one row per COUNTRY (r250), written as the per-country CSV
    #    that is the source of truth for every sum. Summing the r264-expanded table instead would
    #    double-count split countries (see utilities docstring).
    df_q264 = hb.df_read(p.pollination_value_by_region_path)
    df_gep = pf.collapse_regions_to_countries(df_q264)
    hb.df_write(df_gep, service_results['gep_by_country_base_year'])

    # 2. Map only: r264-expanded, each sub-region carries its country's value, never summed
    #    (carbon template; the repo-wide map convention is the open flag-3 decision).
    df_regions = pf.expand_country_values_to_regions(df_q264, df_gep)
    gdf = hb.df_merge(p.gdf_countries_simplified, df_regions, how='outer',
                      left_on='ee_r264_id', right_on='ee_r264_id')
    gdf.to_file(service_results['gep_by_country_base_year'].replace('.csv', '.gpkg'), driver='GPKG')

    # 3. National total = sum over the one-row-per-country table.
    value_gep_base_year = df_gep['pollination_gep'].sum()
    hb.log(f"Total pollination GEP for base year {p.gep_base_year}: {value_gep_base_year}")
    return value_gep_base_year


def gep_load_results(p):
    """Load GEP results from a PRIOR calculation run so the report renders without recomputing.
    Fails loudly if absent (run run_pollination.py first). The results-only entry point."""
    publish_inputs(p)
    result_path = os.path.join(p.intermediate_dir, 'gep_calculation', 'gep_by_country_base_year.csv')
    if not hb.path_exists(result_path):
        raise FileNotFoundError(
            f"pollination GEP results not found at {result_path}. "
            f"Run the calculation first (run_pollination.py), then re-run results.")
    p.results.setdefault('pollination', {})
    p.results['pollination']['gep_by_country_base_year'] = result_path


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)


# =============================================================================
# The pollination value raster, built here rather than taken as given.
# =============================================================================

# The nodata value the source pipeline writes into its production and value rasters. Kept
# identical so a raster of ours and one of theirs can be compared without a conversion step.
NODATA_OUT = -9999.0


def write_raster(path, data, meta, nodata=None):
    """Write a single-band float32 GeoTIFF, creating the parent directory if needed.

    Args:
        path (Path): where to write.
        data (np.ndarray): the 2-D array.
        meta (dict): a rasterio profile, normally the one read_raster returned.
        nodata (float): overrides the profile's nodata when given.

    Returns:
        Path: the path written.
    """
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    profile = dict(meta)
    profile.update(driver='GTiff', dtype='float32', count=1,
                   compress='deflate', predictor=2, tiled=True, zlevel=6)
    if nodata is not None:
        profile['nodata'] = nodata
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data.astype(np.float32), 1)
    return path


CROP_PRODUCTION_RASTER_REF_PATH = os.path.join('crops', 'cropgrids', 'production_2020')
# The files, not their directories: get_path searches the task's own directory first, so a
# directory reference resolved into intermediate/pollination/ rather than into base data.
POLLINATION_DEPENDENCE_REF_PATH = os.path.join('fao', 'pollination',
                                               'pollination_1993_2024.parquet')
FAO_MEDIAN_PRICES_FILE_REF_PATH = os.path.join('fao', 'median_prices',
                                               'price_median_usd_tonne_2018_2022.parquet')
CROPGRIDS_COUNTRY_RASTER_REF_PATH = os.path.join('crops', 'cropgrids', 'country_m49_cropgrids_grid.tif')
COFFEE_ARABICA_ROBUSTA_REF_PATH = os.path.join('pollination', 'coffee_types_distribution', 'prop_arabica_robusta.csv')
CROPGRIDS_CROSSWALK_REF_PATH = os.path.join('fao', 'crosswalks', 'crosswalk_fao_cropgrids.csv')


def world_prices_by_item(df_prices):
    """One producer price per FAO item, taken from the world rows explicitly.

    The source loader built this dictionary from every row of the price table and let
    duplicate keys overwrite each other, so the price each crop ended up with was whichever
    row came last. It happened to be a world row, because the file is written in that order,
    so the numbers were right; they were right by row order rather than by construction, and
    the EE spec's rule against relying on position is exactly this case. Our regenerated
    table has 11,601 rows where theirs had 11,534, so the ordering is not something to
    inherit on trust.

    Args:
        df_prices (pd.DataFrame): the median price table, with agg_level, item_code_fao and
            price_median_usd_tonne.

    Returns:
        dict: FAO item code (int) to price in USD per tonne.
    """
    world = df_prices[df_prices['agg_level'] == 'world']
    if world.empty:
        raise ValueError('The price table carries no world rows, so there is no single price '
                         'per crop to value a global raster at.')
    codes = pd.to_numeric(world['item_code_fao'], errors='coerce')
    return dict(zip(codes.astype('Int64'), world['price_median_usd_tonne'].astype(float)))


def pollination_dependence_by_item(df_dependence):
    """One pollination dependence ratio per FAO item.

    The ratio is a crop property, not a country one, so the table's country-year rows all
    carry the same value for a given crop and collapsing them is safe. It is checked rather
    than assumed: a crop whose rows disagree would mean the ratio is not what we think.

    Args:
        df_dependence (pd.DataFrame): the FAO pollination table, with item_code_fao and poll_dep.

    Returns:
        dict: FAO item code (int) to dependence ratio.
    """
    df = df_dependence[['item_code_fao', 'poll_dep']].dropna(subset=['item_code_fao'])
    spread = df.groupby('item_code_fao')['poll_dep'].nunique(dropna=True)
    unexpected = [c for c in spread[spread > 1].index if int(c) != pf.COFFEE_ITEM_CODE_FAO]
    if unexpected:
        raise ValueError('Pollination dependence varies within FAO items %s. Coffee is the one '
                         'known case, where item 656 covers arabica and robusta and the task '
                         'blends them by country; another split means a crop we have not '
                         'looked at, so it is not collapsed silently.' % unexpected[:5])
    # Coffee keeps arabica's ratio here and is overridden per country in the task, so a caller
    # that forgets the override gets the source pipeline's number rather than a wrong new one.
    df = df.sort_values('poll_dep')
    collapsed = df.drop_duplicates('item_code_fao')
    codes = pd.to_numeric(collapsed['item_code_fao'], errors='coerce')
    return dict(zip(codes.astype('Int64'), collapsed['poll_dep'].fillna(0.0).astype(float)))


def write_source_provenance(raster_path, out_path):
    """Record which file the GEP value came from, so a stale copy is visible rather than silent."""
    import hashlib
    import pandas as pd
    digest = hashlib.sha256()
    with open(raster_path, 'rb') as raster_file:
        for chunk in iter(lambda: raster_file.read(1 << 20), b''):
            digest.update(chunk)
    stats = os.stat(raster_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame([{
        'source_raster': os.path.basename(raster_path),
        'source_raster_path': raster_path,
        'bytes': stats.st_size,
        'modified_utc': pd.Timestamp(stats.st_mtime, unit='s', tz='UTC').isoformat(),
        'sha256': digest.hexdigest(),
    }]).to_csv(out_path, index=False, encoding='utf-8-sig')
    return out_path


def pollination_source_value_raster(p):
    """Make sure the source author's value raster for the GEP base year is in base_data.

    His published rasters cover only some price years, so a GEP base year he has not released has
    to be generated by running his pipeline at that year. Doing that by hand is how a stale file
    gets staged: on 2026-08-28 a raster built from a substituted price table sat in base_data for
    45 minutes after the real one had been rebuilt, and nothing would have caught it because the
    total was only 0.1 percent out. So the staging is a task, and it records what it staged.

    Records what is staged and raises when it is absent; it does not build anything. Generating a
    year the author has not released is a manual procedure: docs/runbook_pollination_value_raster.md.
    """
    publish_inputs(p)
    year = int(p.gep_base_year)
    file_name = 'poll_value_global_%dusd.tif' % year
    p.pollination_source_value_raster_path = os.path.join(
        str(p.get_path(pf.SOURCE_VALUE_RASTER_DIR_REF_PATH)), file_name)
    p.pollination_source_provenance_path = os.path.join(p.cur_dir, 'source_raster_provenance.csv')
    if not p.run_this:
        return

    if hb.path_exists(p.pollination_source_value_raster_path):
        hb.log('Source value raster present: %s' % file_name)
        write_source_provenance(p.pollination_source_value_raster_path,
                                   p.pollination_source_provenance_path)
        return True

    available = pf.available_source_value_years(p)
    raise NameError(
        'No %s in %s. The GEP pollination value is the source author\'s raster, and he publishes '
        'only some price years — this directory has %s. Generating %d means running his pipeline '
        'at that year: see docs/runbook_pollination_value_raster.md, which is six commands and '
        'about half an hour. It is deliberately not automated here, because doing so would put the '
        'execution of his pipeline inside ours.'
        % (file_name, os.path.dirname(p.pollination_source_value_raster_path),
           ', '.join(str(y) for y in available) if available else 'none', year))


def pollination_value_raster(p):
    """Read the source author's pollination value raster and convert it to USD in the cell.

    The science here is the author's, not ours. His raster is production times price times
    pollination dependence, built from CropGrids harvested area, a Monfreda within-country yield
    pattern and FAO calibration, with coffee split into arabica and robusta by country. We do not
    rebuild any of that; we read what he publishes and fix only the one thing that was actually
    wrong on our side, which was the units.

    His file is a DENSITY, USD per square kilometre, stated as such in his repo's
    methods_overview.md and confirmed by his own summary CSV, which area-weights before totalling.
    The GEP path used to sum it directly, giving $18.28bn where the same raster carries $476.29bn
    area-weighted. So the fix is to multiply by cell area, on the shared WGS84 pyramid like every
    other service, and to deflate to the GEP base year when his file is stamped in another year's
    dollars.

    `pollination_value_raster_rebuilt` is our own construction of the same quantity and is kept as
    a cross-check, not as the GEP path.
    """
    publish_inputs(p)
    year = int(p.gep_base_year)
    p.pollination_value_raster_path = os.path.join(
        p.cur_dir, 'poll_value_per_cell_%dusd.tif' % year)
    p.pollination_value_summary_path = os.path.join(
        p.cur_dir, 'poll_value_summary_%dusd.csv' % year)
    if not p.run_this:
        return

    if hb.path_all_exist([p.pollination_value_raster_path, p.pollination_value_summary_path]):
        hb.log('Pollination value raster already built. Skipping.')
        return True

    # Prefer his raster for the GEP base year itself; fall back to the nearest year he publishes
    # and deflate, which is exact because the deflator is a scalar on a density.
    source_path, source_year = pf.find_source_value_raster(p, year)
    deflator = pf.usd_deflator(source_year, year)
    hb.log('Pollination value raster from the source author: %s (%d USD), deflator to %d is %.4f'
           % (os.path.basename(source_path), source_year, year, deflator))

    density, meta = read_raster(str(source_path))
    density = np.where(np.isfinite(density) & (density != meta.get('nodata')), density, np.nan)
    area_km2 = pf.build_area_km2_raster(meta)
    value_per_cell = pf.value_density_to_per_cell(density * deflator, area_km2)

    out_meta = dict(meta)
    out_meta.update(dtype='float32', nodata=NODATA_OUT, count=1)
    write_raster(str(p.pollination_value_raster_path),
                 np.where(np.isfinite(value_per_cell), value_per_cell, NODATA_OUT).astype('float32'),
                 out_meta, nodata=NODATA_OUT)

    total = float(np.nansum(value_per_cell))
    hb.df_write(pd.DataFrame([{
        'source_raster': os.path.basename(source_path),
        'source_year_usd': source_year,
        'gep_base_year': year,
        'deflator_applied': deflator,
        'total_pollination_value_usd': total,
    }]), p.pollination_value_summary_path)

    hb.log('Pollination value raster: %.2f bn USD at %d prices, from the author\'s raster.'
           % (total / 1e9, year))
    return True


def pollination_value_raster_rebuilt(p):
    """Rebuild the pollination value raster from source: production times price times dependence.

    ⚠ NOT the GEP path. `pollination_value_raster` reads the source author's raster instead, because
    the construction below is a different scientific method from his (we take CropGrids production
    directly; he goes harvested area times a Monfreda within-country yield pattern times FAO
    calibration). Under the rule that infrastructure is ours and the science is the author's, that
    was not ours to change. This is kept as an independent cross-check: it agrees with his national
    total to about one percent, which is a useful confirmation that his raster is being consumed
    correctly, and it is the only thing here that would catch a units error in that consumption.

    The GEP valuation used to sum a raster somebody else produced. This makes it, from the
    CropGrids production rasters, the FAO world producer price we now build ourselves, and
    each crop's dependence on animal pollination.

    It writes USD in the cell, not USD per square kilometre. The source pipeline wrote a
    density and its own summary CSV multiplied by cell area before totalling, but the GEP
    path summed the file directly, so the published figure was a sum of densities: $18.28bn
    where the same raster carries $476bn. Writing per-cell values here means the zonal sum
    downstream is a sum of money, and the two conventions can no longer be confused, because
    the units are in the file name.

    Registered with skip_existing=1: it reads 158 crop rasters and is deterministic.
    """
    publish_inputs(p)
    year = int(p.gep_base_year)
    p.pollination_value_raster_rebuilt_path = os.path.join(
        p.cur_dir, 'poll_value_per_cell_%dusd.tif' % year)
    p.pollination_value_summary_rebuilt_path = os.path.join(
        p.cur_dir, 'poll_value_summary_%dusd.csv' % year)
    if not p.run_this:
        return

    if hb.path_all_exist([p.pollination_value_raster_rebuilt_path, p.pollination_value_summary_rebuilt_path]):
        hb.log('Pollination value raster already built. Skipping.')
        return True

    production_dir = p.get_path(CROP_PRODUCTION_RASTER_REF_PATH)
    crosswalk = hb.df_read(p.get_path(CROPGRIDS_CROSSWALK_REF_PATH))
    # pd.read_parquet, not hb.df_read: df_read is a CSV reader and reports a parquet as an
    # encoding failure, naming every text encoding it tried.
    prices = world_prices_by_item(pd.read_parquet(
        p.get_path(FAO_MEDIAN_PRICES_FILE_REF_PATH)))
    dependence = pollination_dependence_by_item(pd.read_parquet(
        p.get_path(POLLINATION_DEPENDENCE_REF_PATH)))

    # The prices are medians over 2018-2022 in the dollars of their own years and the
    # production rasters are dated 2020, so both are brought to the GEP base year.
    deflator = pf.usd_deflator(pf.PRODUCTION_RASTER_YEAR, year)
    hb.log('Pricing at %d USD: deflator from %d is %.4f'
           % (year, pf.PRODUCTION_RASTER_YEAR, deflator))

    country_ids, _ = read_raster(str(p.get_path(CROPGRIDS_COUNTRY_RASTER_REF_PATH)))
    country_ids = country_ids.astype('int32')
    coffee_by_country = pf.coffee_dependence_by_country(
        hb.df_read(p.get_path(COFFEE_ARABICA_ROBUSTA_REF_PATH)))
    hb.log('Coffee: blending arabica and robusta dependence over %d countries.'
           % len(coffee_by_country))

    total_pollination_density = None
    total_crop_density = None
    reference_meta = None
    summary_rows = []
    skipped = {'no_item_code': [], 'no_raster': [], 'no_price': []}

    for crop_name in sorted(crosswalk['cropgrids_2024'].dropna().unique()):
        row = crosswalk[crosswalk['cropgrids_2024'] == crop_name].iloc[0]
        item_code = pd.to_numeric(row['item_code_fao'], errors='coerce')
        if pd.isna(item_code):
            skipped['no_item_code'].append(crop_name)
            continue
        item_code = int(item_code)
        raster_path = os.path.join(production_dir, 'production_%s_2020.tif' % crop_name)
        if not hb.path_exists(raster_path):
            skipped['no_raster'].append(crop_name)
            continue
        price = prices.get(item_code)
        if price is None or not np.isfinite(price):
            skipped['no_price'].append(crop_name)
            continue

        production_density, meta = read_raster(str(raster_path))
        ratio = float(dependence.get(item_code, 0.0))
        if item_code == pf.COFFEE_ITEM_CODE_FAO:
            # One item code, two plants: the ratio has to vary by what each country grows.
            ratio = pf.dependence_raster_from_country_lookup(
                country_ids, coffee_by_country, pf.COFFEE_DEPENDENCE['arabica'])
        pollination_density, crop_density = pf.crop_pollination_value_density(
            production_density, float(price) * deflator, ratio)

        if reference_meta is None:
            reference_meta = meta
            area_km2 = pf.build_area_km2_raster(meta)
            total_pollination_density = np.zeros(pollination_density.shape, dtype='float64')
            total_crop_density = np.zeros(crop_density.shape, dtype='float64')
            covered = np.zeros(pollination_density.shape, dtype=bool)

        valid = np.isfinite(pollination_density)
        total_pollination_density[valid] += pollination_density[valid]
        total_crop_density[valid] += crop_density[valid]
        covered |= valid

        summary_rows.append({
            'cropgrids_crop': crop_name, 'item_code_fao': item_code,
            'item_fao': row.get('item_fao'),
            'price_usd_per_tonne': float(price) * deflator,
            'pollination_dependence': ratio,
            'crop_value_usd': float(np.nansum(
                pf.value_density_to_per_cell(crop_density, area_km2))),
            'pollination_value_usd': float(np.nansum(
                pf.value_density_to_per_cell(pollination_density, area_km2))),
        })

    if reference_meta is None:
        raise RuntimeError('No crop produced a value raster, so there is nothing to write. '
                           'Production rasters were looked for in %s' % production_dir)

    # A cell no crop covers is not a cell worth zero, so it stays nodata.
    total_pollination_density[~covered] = np.nan
    value_per_cell = pf.value_density_to_per_cell(total_pollination_density, area_km2)

    out_meta = dict(reference_meta)
    out_meta.update(dtype='float32', nodata=NODATA_OUT, count=1)
    write_raster(str(p.pollination_value_raster_rebuilt_path),
                 np.where(np.isfinite(value_per_cell), value_per_cell, NODATA_OUT).astype('float32'),
                 out_meta, nodata=NODATA_OUT)

    df_summary = pd.DataFrame(summary_rows)
    hb.df_write(df_summary, p.pollination_value_summary_rebuilt_path)

    total = float(np.nansum(value_per_cell))
    hb.log('Pollination value raster: %d crops valued, %.2f bn USD at %d prices.'
           % (len(summary_rows), total / 1e9, year))
    for reason, crops in skipped.items():
        if crops:
            hb.log('  skipped (%s), %d: %s' % (reason, len(crops), ', '.join(crops[:8])))
    return True


def pollination_value_independence_check(p):
    """Compare our own construction of the value raster against the author's, and record the gap.

    Our construction and the author's share no code and only some data: he goes CropGrids harvested
    area times a Monfreda yield pattern times FAO calibration, we take CropGrids production directly.
    Agreement is therefore evidence about the method rather than about the port.

    ⚠ Reports only. The GEP total is the author's raster, not this one.
    """
    publish_inputs(p)
    p.pollination_independence_path = os.path.join(p.cur_dir, 'value_raster_independence.csv')
    if not p.run_this:
        return

    if hb.path_exists(p.pollination_independence_path):
        hb.log('Independence check already computed. Skipping.')
        return True

    ours_path = getattr(p, 'pollination_value_raster_rebuilt_path', None)
    theirs_path = getattr(p, 'pollination_value_raster_path', None)
    if not (ours_path and hb.path_exists(ours_path) and theirs_path and hb.path_exists(theirs_path)):
        hb.log('Independence check needs both rasters and one is absent; skipping rather than '
               'reporting a comparison against nothing.')
        return

    ours, _ = read_raster(str(ours_path))
    theirs, _ = read_raster(str(theirs_path))
    ours = np.where(np.isfinite(ours) & (ours != NODATA_OUT), ours, np.nan)
    theirs = np.where(np.isfinite(theirs) & (theirs != NODATA_OUT), theirs, np.nan)

    ours_total, theirs_total = float(np.nansum(ours)), float(np.nansum(theirs))
    both = np.isfinite(ours) & np.isfinite(theirs)
    correlation = float(np.corrcoef(ours[both], theirs[both])[0, 1]) if both.sum() > 1 else float('nan')

    hb.df_write(pd.DataFrame([{
        'ours_independent_usd': ours_total,
        'author_raster_usd': theirs_total,
        'ratio_ours_over_author': ours_total / theirs_total if theirs_total else float('nan'),
        'pct_difference': (ours_total / theirs_total - 1.0) * 100.0 if theirs_total else float('nan'),
        'cells_in_both': int(both.sum()),
        'correlation_where_both': correlation,
    }]), p.pollination_independence_path)

    hb.log('Independence check: ours $%.2fbn against the author\'s $%.2fbn, %+.2f percent, '
           'correlation %.4f over %d cells.'
           % (ours_total / 1e9, theirs_total / 1e9,
              (ours_total / theirs_total - 1.0) * 100.0 if theirs_total else float('nan'),
              correlation, int(both.sum())))
    return True

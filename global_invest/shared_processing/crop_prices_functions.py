"""FAO crop prices: the steps that hold no file handling, behind the shared FAOSTAT tables.

crop_prices_tasks reads the staged bulks and exchange rates, calls these steps and writes the production, price, value
and median-price tables. Crop provision and pollination read those tables. The tables keep their place under
global_invest/pollination/fao in base data and their pollination_ settings in es_parameters. Nothing here opens a file.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FaoPriceSettings:
    """What the FAO price steps need, in place of the crop_benefits Config the source module reads.

    The prices are a base-data input rather than a per-run result: the pipeline reads the FAOSTAT
    production and producer-price bulks, reconstructs local currency where FAOSTAT reports only the
    old series, converts to USD against World Bank exchange rates, and writes a median price per
    crop over `price_years`. The pollination value raster is built against that table.

    The two FAOSTAT bulks are staged into base data rather than pulled on each run. FAOSTAT
    revises them, so a fresh pull would make the price table depend on the day it was built, and
    a compute node without outbound network could not run the stage at all. es_parameters carries
    a url and an archive member for each, so the shared download task can fetch them once.

    Attributes:
        crosswalk_m49_iso3_path (str): FAO M49 area codes to ISO3.
        fao_classification_path (str): the FAO item classification.
        crosswalk_fao_cropgrids_path (str): FAO items to CropGrids crop names.
        fao_production_bulk_path (str): the staged FAOSTAT production and yield bulk CSV.
        fao_prices_bulk_path (str): the staged FAOSTAT producer-price bulk CSV.
        fx_lcu_per_usd_path (str): the staged World Bank exchange rates.
        output_dir (str): where the production, price, value and median-price tables are written.
        fao_start_year (int): first year of FAOSTAT data to keep.
        fao_end_year (int): last year.
        price_years (tuple): the years the median price is taken over.
    """
    crosswalk_m49_iso3_path: str
    fao_classification_path: str
    crosswalk_fao_cropgrids_path: str
    fao_production_bulk_path: str
    fao_prices_bulk_path: str
    fx_lcu_per_usd_path: str
    output_dir: str
    # No defaults on the three windows: es_parameters is the only place they are set, so a run
    # at a different base year cannot silently inherit one built for another.
    fao_start_year: int
    fao_end_year: int
    price_years: tuple
    # Tables the price path needs, read from CSVs beside the service rather than held as
    # dictionaries in this module: FAO aggregate codes to leave out, countries with no
    # exchange rate of their own, and IMF country names in the FAO spelling.
    excluded_item_codes: set
    fx_inherit_iso3: dict
    imf_fao_names: dict
    # Price quality-control tolerances. The QC window starts at fao_start_year, so the panel's
    # span and the QC window cannot disagree.
    tol_near_equal_lcu_slc: float
    anchor_max_year_dist: int
    anchor_max_times_off: float
    global_median_max_times_off: float
    usd_copy_of_lcu_tol: float
    usd_copy_of_lcu_min_fx: float
    obs_global_median_max_times_off: float
    obs_global_median_min_countries: int
    qc_min_overlap: int
    qc_bad_median_times_off: float
    qc_bad_share_over_3x: float


# ---------------------------------------------------------------------------------------------
# The FAO price path, the parts that hold no file handling. These build the per-crop median
# producer prices the pollination value raster is priced at.
# ---------------------------------------------------------------------------------------------

# Two bulks, two element vocabularies. Upstream keeps these in separate modules, where both are
# called _ELEMENTS_KEEP; in one module they need distinct names, or the price list overwrites the
# production list and the production filter keeps price elements, leaving the panel empty.
_PRODUCTION_ELEMENTS_KEEP = {"Yield", "Production"}


_PRICE_ELEMENTS_KEEP = [
    "Producer Price (USD/tonne)",
    "Producer Price (SLC/tonne)",
    "Producer Price (LCU/tonne)",
    "Producer Price Index (2014-2016 = 100)",
]



def _convert_yield_units(df: pd.DataFrame) -> pd.DataFrame:
    """Convert yield from kg/ha -> t/ha."""
    mask = df["Element"] == "Yield"
    df.loc[mask, "Value"] = df.loc[mask, "Value"] / 1000
    logger.info("Converted %d yield rows from kg/ha to t/ha", mask.sum())
    return df


def _recode_country(series: pd.Series, mapping: dict) -> pd.Series:
    return series.map(mapping).fillna(series)


def _log_df(name: str, df: pd.DataFrame) -> None:
    logger.info("%s: %d rows, %d cols", name, df.shape[0], df.shape[1])


def _reshape_prices(pp_raw: pd.DataFrame) -> pd.DataFrame:
    """Keep relevant elements and pivot to wide format."""
    logger.info("=== 2-3) FILTER ELEMENTS + RESHAPE TO WIDE ===")

    pp = pp_raw[pp_raw["Element"].isin(_PRICE_ELEMENTS_KEEP)].copy()

    pp_wide = (
        pp.pivot_table(
            index=["Area", "Area Code (M49)", "Item", "Item Code", "Year"],
            columns="Element", values="Value", aggfunc="first",
        )
        .reset_index()
        .rename(columns={
            "Area": "country", "Year": "year",
            "Producer Price (USD/tonne)": "usd_tonne_obs",
            "Producer Price (SLC/tonne)": "slc_tonne",
            "Producer Price (LCU/tonne)": "lcu_tonne",
            "Producer Price Index (2014-2016 = 100)": "price_index",
        })
    )
    _log_df("pp_wide", pp_wide)
    return pp_wide


def _reconstruct_slc_lcu(pp_wide: pd.DataFrame, cfg) -> pd.DataFrame:
    """Steps 4-7: base SLC, fill SLC, LCU/SLC ratios, fill LCU."""

    # 4) Base SLC at index = 100
    logger.info("=== 4) ESTIMATING BASE SLC AT INDEX=100 ===")
    slc_base = (
        pp_wide.dropna(subset=["slc_tonne", "price_index"])
        .query("price_index > 0")
        .assign(base_slc_implied=lambda d: d["slc_tonne"] * 100 / d["price_index"])
        .groupby(["country", "Item"], as_index=False)
        .agg(
            slc_at_100_index=("base_slc_implied", "median"),
            n_years_base=("base_slc_implied", "size"),
        )
    )

    # 5) Fill SLC
    logger.info("=== 5) FILLING SLC USING INDEX ===")
    pp2 = pp_wide.merge(slc_base, on=["country", "Item"], how="left")

    pp2["slc_filled"] = np.where(
        pp2["slc_tonne"].notna(), pp2["slc_tonne"],
        np.where(
            pp2["price_index"].notna() & pp2["slc_at_100_index"].notna(),
            pp2["slc_at_100_index"] * pp2["price_index"] / 100, np.nan,
        ),
    )
    pp2["slc_source"] = np.select(
        [pp2["slc_tonne"].notna(),
         pp2["slc_tonne"].isna() & pp2["price_index"].notna() & pp2["slc_at_100_index"].notna()],
        ["observed", "index_imputed"], default="missing",
    )

    # 6) LCU/SLC ratios
    logger.info("=== 6) ESTIMATING LCU/SLC RATIOS ===")
    lcu_slc_year = (
        pp2.dropna(subset=["lcu_tonne", "slc_tonne"])
        .query("slc_tonne > 0")
        .assign(ratio=lambda d: d["lcu_tonne"] / d["slc_tonne"])
        .groupby(["country", "year"], as_index=False)
        .agg(lcu_per_slc_year_country=("ratio", "median"))
    )
    lcu_slc_country = (
        lcu_slc_year.groupby("country", as_index=False)
        .agg(lcu_per_slc_country=("lcu_per_slc_year_country", "median"))
    )

    # 7) Fill LCU
    logger.info("=== 7) FILLING LCU USING SLC + RATIOS ===")
    pp3 = (
        pp2.merge(lcu_slc_year, on=["country", "year"], how="left")
        .merge(lcu_slc_country, on="country", how="left")
    )

    pp3["lcu_filled"] = np.select(
        [
            pp3["lcu_tonne"].notna(),
            pp3["lcu_tonne"].isna() & pp3["slc_filled"].notna() & pp3["lcu_per_slc_year_country"].notna(),
            pp3["lcu_tonne"].isna() & pp3["slc_filled"].notna()
            & pp3["lcu_per_slc_year_country"].isna() & pp3["lcu_per_slc_country"].notna()
            & ((pp3["lcu_per_slc_country"] - 1).abs() <= cfg.tol_near_equal_lcu_slc),
            pp3["lcu_tonne"].isna() & pp3["slc_filled"].notna()
            & pp3["lcu_per_slc_year_country"].isna() & pp3["lcu_per_slc_country"].notna(),
            pp3["lcu_tonne"].isna() & pp3["slc_filled"].notna(),
        ],
        [
            pp3["lcu_tonne"],
            pp3["slc_filled"] * pp3["lcu_per_slc_year_country"],
            pp3["slc_filled"],
            pp3["slc_filled"] * pp3["lcu_per_slc_country"],
            pp3["slc_filled"],
        ],
        default=np.nan,
    )

    pp3["lcu_source"] = np.select(
        [
            pp3["lcu_tonne"].notna(),
            pp3["lcu_tonne"].isna() & pp3["slc_filled"].notna() & pp3["lcu_per_slc_year_country"].notna(),
            pp3["lcu_tonne"].isna() & pp3["slc_filled"].notna()
            & pp3["lcu_per_slc_year_country"].isna() & pp3["lcu_per_slc_country"].notna()
            & ((pp3["lcu_per_slc_country"] - 1).abs() <= cfg.tol_near_equal_lcu_slc),
            pp3["lcu_tonne"].isna() & pp3["slc_filled"].notna()
            & pp3["lcu_per_slc_year_country"].isna() & pp3["lcu_per_slc_country"].notna(),
            pp3["lcu_tonne"].isna() & pp3["slc_filled"].notna(),
        ],
        [
            "observed", "slc_filled_country_year_factor",
            "slc_filled_country_factor_near_equal",
            "slc_filled_country_factor", "slc_filled_assume_equal",
        ],
        default="missing",
    )

    logger.info("LCU source counts:\n%s", pp3["lcu_source"].value_counts())
    return pp3


def _build_usd_with_qc(
    pp3: pd.DataFrame,
    fx: pd.DataFrame,
    cfg,
) -> pd.DataFrame:
    """Merge FX, compute USD, drop USD copies of local currency, apply the QC filters."""

    # 14) Merge FX
    logger.info("=== 14) MERGING FX INTO PRICE PANEL ===")
    pp_usd = pp3.merge(fx, on=["iso3", "year"], how="left")

    # 15) FX-implied USD
    logger.info("=== 15) COMPUTING FX-IMPLIED USD ===")
    pp_usd["usd_fx_implied"] = np.where(
        pp_usd["lcu_filled"].notna() & pp_usd["lcu_per_usd"].notna() & (pp_usd["lcu_per_usd"] > 0),
        pp_usd["lcu_filled"] / pp_usd["lcu_per_usd"], np.nan,
    )
    pp_usd["usd_filled"] = pp_usd["usd_tonne_obs"].combine_first(pp_usd["usd_fx_implied"])
    pp_usd["usd_source"] = np.select(
        [pp_usd["usd_tonne_obs"].notna(),
         pp_usd["usd_tonne_obs"].isna() & pp_usd["usd_fx_implied"].notna()],
        ["observed", "filled"], default="missing",
    )

    # 15) A FAO "USD" price can be a copy of the local-currency price. Equal while the exchange rate
    # is not about 1, it is unknown which one is right, so the price goes missing and the subregion,
    # region or world fallback applies. The USD value leaves usd_tonne_obs too, so it is no observed
    # anchor, no part of the global medians and no part of the FX country check.
    pp_usd["usd_copy_of_lcu"] = (
        (pp_usd["usd_source"] == "observed")
        & pp_usd["lcu_tonne"].notna() & (pp_usd["lcu_tonne"] > 0)
        & ((pp_usd["usd_tonne_obs"] / pp_usd["lcu_tonne"] - 1).abs() < cfg.usd_copy_of_lcu_tol)
        & pp_usd["lcu_per_usd"].notna() & (pp_usd["lcu_per_usd"] > 0)
        & (np.log(pp_usd["lcu_per_usd"]).abs() > np.log(cfg.usd_copy_of_lcu_min_fx))
    )
    copy_mask = pp_usd["usd_copy_of_lcu"]
    logger.warning("USD equal to LCU (copy), set to missing: %d rows", copy_mask.sum())
    pp_usd.loc[copy_mask, ["usd_tonne_obs", "usd_filled", "usd_source"]] = [
        np.nan, np.nan, "invalid_usd_copy_of_lcu",
    ]

    # 15a) Invalidate USD from bad LCU sources
    logger.info("=== 15a) INVALIDATING USD FROM BAD LCU SOURCES ===")
    bad_lcu = {"slc_filled_country_factor", "missing", "slc_filled_assume_equal"}
    mask_bad = (pp_usd["usd_source"] == "filled") & pp_usd["lcu_source"].isin(bad_lcu)
    pp_usd.loc[mask_bad, ["usd_fx_implied", "usd_filled", "usd_source"]] = [
        np.nan, np.nan, "invalid_currency_mapping",
    ]

    # 15b) QC: closest observed USD anchor
    logger.info("=== 15b) QC: CLOSEST OBSERVED USD ANCHOR ===")
    obs_anchor = (
        pp_usd.loc[pp_usd["usd_tonne_obs"].notna(), ["iso3", "Item", "year", "usd_tonne_obs"]]
        .rename(columns={"year": "anchor_year", "usd_tonne_obs": "anchor_usd"})
    )
    tmp = pp_usd.merge(obs_anchor, on=["iso3", "Item"], how="left")
    tmp["anchor_year_dist"] = (tmp["year"] - tmp["anchor_year"]).abs()
    tmp = tmp.sort_values("anchor_year_dist")
    closest_idx = tmp.groupby(["iso3", "Item", "year"])["anchor_year_dist"].idxmin().dropna().astype(int)
    nearest = tmp.loc[closest_idx, ["iso3", "Item", "year", "anchor_year", "anchor_usd", "anchor_year_dist"]].copy()
    pp_usd = pp_usd.merge(nearest, on=["iso3", "Item", "year"], how="left")

    use_anchor = (
        (pp_usd["usd_source"] == "filled")
        & (pp_usd["anchor_year_dist"] <= cfg.anchor_max_year_dist)
        & pp_usd["anchor_usd"].notna() & (pp_usd["anchor_usd"] > 0)
        & pp_usd["usd_fx_implied"].notna() & (pp_usd["usd_fx_implied"] > 0)
    )
    pp_usd.loc[use_anchor, "times_off_closest_obs"] = np.maximum(
        pp_usd.loc[use_anchor, "usd_fx_implied"] / pp_usd.loc[use_anchor, "anchor_usd"],
        pp_usd.loc[use_anchor, "anchor_usd"] / pp_usd.loc[use_anchor, "usd_fx_implied"],
    )
    bad_anchor = use_anchor & (pp_usd["times_off_closest_obs"] > cfg.anchor_max_times_off)
    logger.warning("Anchor QC dropping %d rows", bad_anchor.sum())
    pp_usd.loc[bad_anchor, ["usd_fx_implied", "usd_filled", "usd_source"]] = [
        np.nan, np.nan, "implausible_vs_closest_observed_10x",
    ]

    # 15c) QC: global item-year median band
    logger.info("=== 15c) QC: GLOBAL ITEM-YEAR MEDIAN BAND ===")
    global_med = (
        pp_usd.loc[pp_usd["usd_tonne_obs"].notna()]
        .groupby(["Item", "year"])
        .agg(global_median_usd=("usd_tonne_obs", "median"), n_obs=("usd_tonne_obs", "size"))
        .reset_index()
    )
    pp_usd = pp_usd.merge(global_med, on=["Item", "year"], how="left")
    use_global = (
        (pp_usd["usd_source"] == "filled")
        & pp_usd["global_median_usd"].notna() & (pp_usd["global_median_usd"] > 0)
        & pp_usd["usd_fx_implied"].notna() & (pp_usd["usd_fx_implied"] > 0)
    )
    pp_usd.loc[use_global, "global_ratio"] = (
        pp_usd.loc[use_global, "usd_fx_implied"] / pp_usd.loc[use_global, "global_median_usd"]
    )
    bad_global = use_global & (
        (pp_usd["global_ratio"] > cfg.global_median_max_times_off)
        | (pp_usd["global_ratio"] < 1 / cfg.global_median_max_times_off)
    )
    logger.warning("Global median QC dropping %d rows", bad_global.sum())
    pp_usd.loc[bad_global, ["usd_fx_implied", "usd_filled", "usd_source"]] = [
        np.nan, np.nan, "implausible_vs_global_median_10x",
    ]

    # 15c-obs) The same check on observed (FAO-reported) USD prices, at its own threshold and only
    # when enough countries report the crop and year. The observed price stays in usd_tonne_obs, so
    # it remains an anchor and a part of the global medians and the FX country check. Only usd_filled
    # goes missing.
    bad_obs_global = (
        (pp_usd["usd_source"] == "observed")
        & pp_usd["global_median_usd"].notna() & (pp_usd["global_median_usd"] > 0)
        & (pp_usd["n_obs"] >= cfg.obs_global_median_min_countries)
        & pp_usd["usd_tonne_obs"].notna() & (pp_usd["usd_tonne_obs"] > 0)
        & (
            (pp_usd["usd_tonne_obs"] > pp_usd["global_median_usd"] * cfg.obs_global_median_max_times_off)
            | (pp_usd["usd_tonne_obs"] < pp_usd["global_median_usd"] / cfg.obs_global_median_max_times_off)
        )
    )
    logger.warning("Observed USD global median QC dropping %d rows", bad_obs_global.sum())
    pp_usd.loc[bad_obs_global, ["usd_filled", "usd_source"]] = [
        np.nan, "implausible_observed_vs_global_median_%gx" % cfg.obs_global_median_max_times_off,
    ]

    # 15d) QC: FX reliability by country
    logger.info("=== 15d) QC: FX RELIABILITY BY COUNTRY ===")
    qc = (
        pp_usd.loc[
            (pp_usd["year"] >= cfg.fao_start_year)
            & pp_usd["usd_tonne_obs"].notna() & pp_usd["usd_fx_implied"].notna()
            & (pp_usd["usd_tonne_obs"] > 0) & (pp_usd["usd_fx_implied"] > 0),
            ["iso3", "usd_tonne_obs", "usd_fx_implied"],
        ].copy()
    )
    qc["times_off"] = np.maximum(
        qc["usd_tonne_obs"] / qc["usd_fx_implied"],
        qc["usd_fx_implied"] / qc["usd_tonne_obs"],
    )
    country_qc = (
        qc.groupby("iso3").agg(
            n_overlap=("times_off", "size"),
            median_times_off=("times_off", "median"),
            share_over_3x=("times_off", lambda s: (s > 3).mean()),
        ).reset_index()
    )
    bad_fx_countries = set(
        country_qc.loc[
            (country_qc["n_overlap"] >= cfg.qc_min_overlap)
            & (
                (country_qc["median_times_off"] > cfg.qc_bad_median_times_off)
                | (country_qc["share_over_3x"] > cfg.qc_bad_share_over_3x)
            ),
            "iso3",
        ]
    )
    logger.warning("Bad FX countries: %d", len(bad_fx_countries))
    bad_fx_mask = (pp_usd["usd_source"] == "filled") & pp_usd["iso3"].isin(bad_fx_countries)
    pp_usd.loc[bad_fx_mask, ["usd_fx_implied", "usd_filled", "usd_source"]] = [
        np.nan, np.nan, "invalid_fx_country",
    ]

    # 15e) Re-attach country column
    country_lookup = pp3[["iso3", "region_fao", "subregion_fao", "year", "Item", "country"]].drop_duplicates()
    pp_usd = pp_usd.merge(country_lookup, on=["iso3", "year", "Item"], how="left", suffixes=("", "_from_pp3"))
    if "country_from_pp3" in pp_usd.columns:
        pp_usd["country"] = pp_usd["country"].combine_first(pp_usd["country_from_pp3"])
        pp_usd = pp_usd.drop(columns=["country_from_pp3"])

    logger.info("USD source counts (final):\n%s", pp_usd["usd_source"].value_counts())
    return pp_usd


def _compute_annual_prices(prices: pd.DataFrame, cw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute annual prices hierarchically: country, subregion, region, world.
    """
    if "region_fao" not in prices.columns:
        prices = prices.merge(cw[["area_code_m49", "region_fao", "subregion_fao"]], on="area_code_m49", how="left")

    pf = prices[
        prices["price_usd_tonne"].notna()
        & (prices["price_usd_tonne"] > 0)
    ].copy()

    # 1. Annual country prices
    price_country = (
        pf.groupby(
            ["year", "area_code_m49", "region_fao", "subregion_fao", "item_code_fao"],
            as_index=False, dropna=False
        )
        .agg(
            price_usd_tonne=("price_usd_tonne", "mean"),
            n_obs=("price_usd_tonne", "size"),
        )
    )
    
    # 2. Annual subregion median prices
    price_subregion = (
        price_country.dropna(subset=["subregion_fao"])
        .groupby(["year", "subregion_fao", "item_code_fao"], as_index=False)
        .agg(
            subreg_price_usd_tonne=("price_usd_tonne", "median")
        )
    )

    # 3. Annual region median prices
    price_region = (
        price_country.dropna(subset=["region_fao"])
        .groupby(["year", "region_fao", "item_code_fao"], as_index=False)
        .agg(
            reg_price_usd_tonne=("price_usd_tonne", "median")
        )
    )
    
    # 4. Annual world median prices
    price_world = (
        price_country.groupby(["year", "item_code_fao"], as_index=False)
        .agg(
            world_price_usd_tonne=("price_usd_tonne", "median"),
            n_countries_price=("area_code_m49", "nunique"),
        )
    )
    
    return price_country, price_subregion, price_region, price_world

"""FAO crop prices: the steps that read and write, behind the shared FAOSTAT tables.

run_fao_production, run_fao_prices and run_fao_values build the production, price, value and median-price tables from
the staged FAOSTAT bulks and exchange rates. fao_median_prices runs the three as a base-data-generating task, which any
service's tree can register and pollination's does. Crop provision and pollination read the tables. The tables keep their
place under global_invest/pollination/fao in base data and their pollination_ settings in es_parameters, which
publish_inputs hydrates.
The arithmetic is in crop_prices_functions. This module imports no service, so any service folder can import it.
"""
from __future__ import annotations
import glob
import logging
import os
from typing import Set, Tuple
import numpy as np
import pandas as pd
import hazelbean as hb
from global_invest import utilities
from global_invest.shared_processing import crop_prices_functions as cpf

logger = logging.getLogger(__name__)

FAO_MEDIAN_PRICES_REF_PATH = os.path.join('global_invest', 'pollination', 'fao', 'median_prices')


def save_parquet(df: pd.DataFrame, path: str | str) -> str:
    """Save DataFrame as parquet and log the result."""
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved %d rows -> %s", len(df), path)
    return path


def _add_geographic_and_classification_data(
    df: pd.DataFrame,
    classif: pd.DataFrame,
    cfg: cpf.FaoPriceSettings,
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


def run_fao_production(cfg: cpf.FaoPriceSettings) -> str:
    """
    Execute the full FAO production pipeline.

    Returns the path to the output parquet file.
    """
    logger.info("=== FAO Production Pipeline ===")

    df, classif = _read_and_filter_fao_production(cfg)
    df = cpf._convert_yield_units(df)
    df = _add_geographic_and_classification_data(df, classif, cfg)

    outdir = cfg.output_dir
    pq_path = _save_production_outputs(df, outdir, cfg)

    logger.info("=== FAO Production Pipeline COMPLETE ===")
    return pq_path


def run_fao_prices(cfg: cpf.FaoPriceSettings) -> str:
    """
    Execute the full FAO producer-price pipeline.

    Returns the path to the output parquet file.
    """
    logger.info("=== FAO Prices Pipeline ===")

    years = list(range(cfg.fao_start_year, cfg.fao_end_year + 1))

    # Steps 1-3
    pp_raw = _read_fao_prices(cfg, years)
    pp_wide = cpf._reshape_prices(pp_raw)

    # Steps 4-7
    pp3 = cpf._reconstruct_slc_lcu(pp_wide, cfg)

    # Steps 9-12
    pp3, fx = _add_iso3_and_fx(pp3, cfg, years)

    # Steps 13-15
    pp_usd = cpf._build_usd_with_qc(pp3, fx, cfg)

    # Step 16
    outdir = cfg.output_dir
    pq_path = _save_price_outputs(pp_usd, outdir, cfg)

    logger.info("=== FAO Prices Pipeline COMPLETE ===")
    return pq_path


def fao_table_name(kind, cfg, extension):
    """The name of one FAO table, stamped with the span it was built over.

    The years are configuration, so the span is part of the name: a run over a different span
    writes a different file rather than overwriting another span's under the same name.
    """
    return 'fao_%s_%d_%d.%s' % (kind, cfg.fao_start_year, cfg.fao_end_year, extension)


def _read_and_filter_fao_production(
    cfg: cpf.FaoPriceSettings,
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
    item_codes = set(classif["item_code_fao"].unique()) - cfg.excluded_item_codes
    logger.info("Valid FAO crop item codes: %d", len(item_codes))

    # Year range from config
    years = set(range(cfg.fao_start_year, cfg.fao_end_year + 1))

    # The staged bulk, put in base data by the shared download task rather than pulled here.
    logger.info("Reading staged FAOSTAT production bulk from %s", cfg.fao_production_bulk_path)
    if not hb.path_exists(cfg.fao_production_bulk_path):
        raise NameError(
            'pollination has no FAOSTAT production bulk at %s. es_parameters carries its url and '
            'archive member, so the shared download task stages it.'
            % cfg.fao_production_bulk_path)
    with open(cfg.fao_production_bulk_path, 'rb') as f:
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
        & df["Element"].isin(cpf._PRODUCTION_ELEMENTS_KEEP)
        & df["Item Code"].isin(item_codes)
        & (df["Value"] > 0)
    ].copy()

    logger.info("Rows after filtering: %d", len(df))
    return df, classif


def _save_production_outputs(
    df: pd.DataFrame,
    outdir: str,
    cfg: cpf.FaoPriceSettings,
) -> str:
    """Save final production dataset in standard column order."""
    os.makedirs(outdir, exist_ok=True)

    col_order = [
        "area_code_m49", "iso3", "region_fao", "subregion_fao",
        "area_fao", "item_code_fao", "item_fao", "group_fao",
        "element", "year", "value",
    ]
    df = df[col_order]

    csv_out = os.path.join(outdir, fao_table_name('production', cfg, 'csv'))
    pq_out = os.path.join(outdir, fao_table_name('production', cfg, 'parquet'))

    save_csv(df, csv_out)
    save_parquet(df, pq_out)

    logger.info("Final dataset: %d rows", len(df))
    return pq_out


def _read_fao_prices(cfg: cpf.FaoPriceSettings, years: list[int]) -> pd.DataFrame:
    """Read the staged FAOSTAT producer-price bulk and return annual rows."""
    logger.info("=== 1) READING STAGED FAOSTAT PRODUCER PRICES ===")

    if not hb.path_exists(cfg.fao_prices_bulk_path):
        raise NameError(
            'pollination has no FAOSTAT producer-price bulk at %s. es_parameters carries its url '
            'and archive member, so the shared download task stages it.'
            % cfg.fao_prices_bulk_path)
    logger.info("Reading bulk file: %s", cfg.fao_prices_bulk_path)
    pp_raw = pd.read_csv(cfg.fao_prices_bulk_path, encoding="latin1", low_memory=False)

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

    cpf._log_df("pp_raw (annual PP, bulk)", pp_raw)
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
            price without saying so.
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
    cfg: cpf.FaoPriceSettings,
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
    wb_fx = world_bank_fx(cfg.fx_lcu_per_usd_path)
    wb_fx = wb_fx.dropna(subset=["iso3", "lcu_per_usd"]).copy()
    cpf._log_df("wb_fx", wb_fx)

    # 11) IMF FX (local file)
    logger.info("=== 11) LOADING IMF FX ===")
    # three levels up from the crosswalk file, then the currencies tree beside it
    imf_fx_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(cfg.crosswalk_m49_iso3_path))),
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
        imf_fx["country"] = cpf._recode_country(imf_fx["country"], cfg.imf_fao_names)

        # 11c) Map IMF to ISO3
        country_to_iso3 = pp3[["country", "iso3"]].dropna().drop_duplicates()
        imf_fx = imf_fx.merge(country_to_iso3, on="country", how="left")
        imf_fx = imf_fx.dropna(subset=["iso3"]).copy()

    cpf._log_df("imf_fx", imf_fx)

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
    for child, parent in cfg.fx_inherit_iso3.items():
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
    cpf._log_df("fx (after inheritance)", fx)

    return pp3, fx


def _save_price_outputs(
    pp_usd: pd.DataFrame,
    outdir: str,
    cfg: cpf.FaoPriceSettings,
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
        "price_usd_tonne", "usd_source", "usd_copy_of_lcu",
        "usd_tonne_obs", "usd_fx_implied", "fx_source",
        "slc_filled", "slc_source", "lcu_filled", "lcu_source",
        "price_index",
    ]

    pp_final = pp_usd[final_cols].copy()

    csv_out = os.path.join(outdir, fao_table_name('prices', cfg, 'csv'))
    pq_out = os.path.join(outdir, fao_table_name('prices', cfg, 'parquet'))

    save_csv(pp_final, csv_out)
    save_parquet(pp_final, pq_out)

    logger.info("Final price panel: %d rows", len(pp_final))
    return pq_out


def _load_production_and_prices(cfg: cpf.FaoPriceSettings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the production and price parquets from earlier pipeline steps."""
    prod_dir = cfg.output_dir
    prod_pq = os.path.join(prod_dir, fao_table_name('production', cfg, 'parquet'))
    logger.info("Loading production: %s", prod_pq)
    prod = pd.read_parquet(prod_pq)

    price_dir = cfg.output_dir
    price_pq = os.path.join(price_dir, fao_table_name('prices', cfg, 'parquet'))
    logger.info("Loading prices: %s", price_pq)
    prices = pd.read_parquet(price_pq)

    return prod, prices


def _compute_median_prices(
    prices: pd.DataFrame,
    price_years: list[int],
    outdir: str,
    cw: pd.DataFrame,
) -> pd.DataFrame:
    """Median price per hierarchical level over the price window: the country's own median,
    then the same crop's median across the FAO subregion, region and world (the account's
    adopted fallback, the author's crop_benefits rule). No cross-year step: a country-item with no
    price inside the window falls back geographically, never to another year."""
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
    pq_out = os.path.join(outdir, f"price_median_usd_tonne_{yr_start}_{yr_end}.parquet")
    save_parquet(median_prices, pq_out)

    logger.info("Median prices: %d rows", len(median_prices))
    return median_prices


def _merge_production_value(
    prod: pd.DataFrame,
    price_country: pd.DataFrame,
    price_subregion: pd.DataFrame,
    price_region: pd.DataFrame,
    price_world: pd.DataFrame,
) -> pd.DataFrame:
    """Production in tonnes times the same year's price in USD per tonne, falling back from the
    country's own price to the same crop's subregion, region and world medians (the account's
    adopted rule, the author's crop_benefits hierarchy). No cross-year step.

    Returns the frame rather than writing it: file writing inside what reads as arithmetic
    would make the fallback untestable without a disk.
    """
    logger.info("Merging production × annual prices (hierarchical)")

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

    # Apply the fallbacks in order, recording which level priced each row
    merged["price_source_agg_level"] = "country"
    miss_ctry = merged["price_usd_tonne"].isna()
    merged.loc[miss_ctry, "price_usd_tonne"] = merged.loc[miss_ctry, "subreg_price_usd_tonne"]
    merged.loc[miss_ctry & merged["subreg_price_usd_tonne"].notna(), "price_source_agg_level"] = "subregion"
    miss_subreg = merged["price_usd_tonne"].isna()
    merged.loc[miss_subreg, "price_usd_tonne"] = merged.loc[miss_subreg, "reg_price_usd_tonne"]
    merged.loc[miss_subreg & merged["reg_price_usd_tonne"].notna(), "price_source_agg_level"] = "region"
    miss_reg = merged["price_usd_tonne"].isna()
    merged.loc[miss_reg, "price_usd_tonne"] = merged.loc[miss_reg, "world_price_usd_tonne"]
    merged.loc[miss_reg & merged["world_price_usd_tonne"].notna(), "price_source_agg_level"] = "world"
    merged.loc[merged["price_usd_tonne"].isna(), "price_source_agg_level"] = "missing"

    merged["value_usd"] = merged["total_production_tonnes"] * merged["price_usd_tonne"]
    merged = merged.drop(columns=["subreg_price_usd_tonne", "reg_price_usd_tonne", "world_price_usd_tonne"])

    logger.info(
        "Price sources: %s",
        merged.groupby("price_source_agg_level").size().to_dict()
    )
    logger.info("Rows with value_usd: %d", merged["value_usd"].notna().sum())

    # Drop element column if present (it's always 'Production')
    if "element" in merged.columns:
        merged = merged.drop(columns=["element"])

    logger.info("Production value dataset: %d rows", len(merged))
    return merged


def run_fao_values(cfg: cpf.FaoPriceSettings) -> str:
    """
    Compute total production value by merging prices and production.

    Returns the path to the output parquet file.
    """
    logger.info("=== FAO Values Pipeline ===")

    prod, prices = _load_production_and_prices(cfg)

    # Side-effect: generate the hierarchical median prices the value raster reads
    cw = pd.read_csv(cfg.crosswalk_m49_iso3_path)
    cw["area_code_m49"] = cw["area_code_m49"].astype(str).str.zfill(3)

    _ = _compute_median_prices(
        prices,
        cfg.price_years,
        os.path.join(cfg.output_dir, 'median_prices'),
        cw,
    )

    # Main pipeline: annual prices with the geographic fallback
    price_country, price_subregion, price_region, price_world = cpf._compute_annual_prices(prices, cw)

    merged = _merge_production_value(
        prod, price_country, price_subregion, price_region, price_world)

    # The writing is the task layer's, which is what keeps the merge testable without a disk.
    os.makedirs(cfg.output_dir, exist_ok=True)
    csv_out = os.path.join(cfg.output_dir, fao_table_name('values', cfg, 'csv'))
    pq_out = os.path.join(cfg.output_dir, fao_table_name('values', cfg, 'parquet'))
    save_csv(merged, csv_out)
    save_parquet(merged, pq_out)

    logger.info("=== FAO Values Pipeline COMPLETE ===")
    return pq_out


def load_m49_iso3(cfg: cpf.FaoPriceSettings) -> Tuple[pd.DataFrame, Set[str]]:
    """
    Load the M49 -> ISO3 crosswalk and return *(df, valid_m49)*.

    The "area_code_m49" column is zero-filled to 3 chars to match the
    convention used throughout the FAO and raster pipelines.

    Parameters
    ----------
    cfg : cpf.FaoPriceSettings

    Returns
    -------
    df : pd.DataFrame
        Columns include "area_code_m49", "iso3", "area_fao",
        "region_fao", "subregion_fao".
    valid_m49 : set[str]
        The set of valid 3-digit M49 strings.
    """
    path = cfg.crosswalk_m49_iso3_path
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


def load_fao_classification(cfg: cpf.FaoPriceSettings) -> pd.DataFrame:
    """
    Load the FAO crop classification table.

    Parameters
    ----------
    cfg : cpf.FaoPriceSettings

    Returns
    -------
    pd.DataFrame
    """
    path = cfg.fao_classification_path
    logger.info("Loading FAO classification: %s", path)
    return pd.read_csv(path)


def fao_price_settings(p, output_dir):
    """The FAO price path's settings from the hydrated configuration, writing to output_dir."""
    return cpf.FaoPriceSettings(
        crosswalk_m49_iso3_path=str(p.get_path(p.pollination_crosswalk_m49_iso3_path)),
        fao_classification_path=str(p.get_path(p.pollination_fao_classification_path)),
        crosswalk_fao_cropgrids_path=str(p.get_path(p.pollination_crosswalk_fao_cropgrids_path)),
        fao_production_bulk_path=str(p.pollination_fao_production_path),
        fao_prices_bulk_path=str(p.pollination_fao_prices_path),
        fx_lcu_per_usd_path=str(p.pollination_fx_path),
        output_dir=str(output_dir),
        price_years=tuple(p.pollination_price_years),
        fao_start_year=int(p.pollination_fao_start_year),
        fao_end_year=int(p.pollination_fao_end_year),
        excluded_item_codes=load_excluded_item_codes(p),
        fx_inherit_iso3=load_fx_inherit_iso3(p),
        imf_fao_names=load_imf_fao_country_names(p),
        tol_near_equal_lcu_slc=float(p.pollination_price_tol_near_equal_lcu_slc),
        anchor_max_year_dist=int(p.pollination_price_anchor_max_year_dist),
        anchor_max_times_off=float(p.pollination_price_anchor_max_times_off),
        global_median_max_times_off=float(p.pollination_price_global_median_max_times_off),
        usd_copy_of_lcu_tol=float(p.pollination_price_usd_copy_of_lcu_tol),
        usd_copy_of_lcu_min_fx=float(p.pollination_price_usd_copy_of_lcu_min_fx),
        obs_global_median_max_times_off=float(p.pollination_price_obs_global_median_max_times_off),
        obs_global_median_min_countries=int(p.pollination_price_obs_global_median_min_countries),
        qc_min_overlap=int(p.pollination_price_qc_min_overlap),
        qc_bad_median_times_off=float(p.pollination_price_qc_bad_median_times_off),
        qc_bad_share_over_3x=float(p.pollination_price_qc_bad_share_over_3x))


def publish_inputs(p):
    """The task's first line: the pollination es_config and es_parameters rows, which hold the price
    settings, plus the shared country references and the results registry."""
    utilities.hydrate_es_config(p, 'pollination', log=hb.log)
    utilities.hydrate_es_parameters(p, 'pollination', log=hb.log)
    utilities.initialize_country_paths(p)
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def fao_median_prices(p):
    """Per-crop median producer prices in USD, built here from the staged FAOSTAT bulks.

    The pollination value raster is production times price times each crop's dependence on
    pollinators. This runs the path that makes the price half rather than taking it as a
    finished table: FAOSTAT production and producer prices are read from the bulks the shared
    download task stages, local currency is reconstructed where FAOSTAT reports only the
    discontinued series, USD is built against World Bank exchange rates, and a median is taken
    over the price years.

    Registered with skip_existing=1: it reads several hundred megabytes of FAOSTAT bulk and is
    deterministic once it has run, the same reason erosion's SDR step skips.
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
    settings = fao_price_settings(p, os.path.dirname(p.fao_median_prices_dir))
    run_fao_production(settings)
    run_fao_prices(settings)
    run_fao_values(settings)
    hb.log('FAO median prices written to %s' % p.fao_median_prices_dir)
    return True


def load_excluded_item_codes(p):
    """The FAO item codes the account leaves out: aggregates and derived equivalents."""
    return set(hb.df_read(p.pollination_excluded_item_codes_path)['item_code_fao'].astype(int))


def load_fx_inherit_iso3(p):
    """Countries with no exchange rate of their own, and the one whose rate they use."""
    df = hb.df_read(p.pollination_fx_inherit_path)
    return {str(r.src_label): str(r.dst_label) for r in df.itertuples()}


def load_imf_fao_country_names(p):
    """IMF country names to the FAO spelling, so the two tables join."""
    df = hb.df_read(p.pollination_imf_fao_names_path)
    return {str(r.imf_name): str(r.fao_name) for r in df.itertuples()}


def save_csv(df: pd.DataFrame, path: str | str) -> str:
    """Save DataFrame as CSV and log the result."""
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %d rows -> %s", len(df), path)
    return path


def median_prices_ref_path(price_years):
    """Where the median price table for one window lives, under base data.

    The window is part of the file name, so a run at a different base year reads a different file
    rather than silently reusing one built for another year.
    """
    years = sorted(int(y) for y in price_years)
    return os.path.join(FAO_MEDIAN_PRICES_REF_PATH,
                        'price_median_usd_tonne_%d_%d.parquet' % (years[0], years[-1]))

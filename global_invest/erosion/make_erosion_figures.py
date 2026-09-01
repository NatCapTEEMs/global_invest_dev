"""Erosion's publication figures: the maps, charts and histograms of a finished run.

A script rather than a task because nothing in the account reads what it draws. The valuation
writes `integrated_country_gep.csv` and `gep_calculation` turns that into the per-country table
every other service publishes; the thirty-four PNGs here are read by no task, no results page and
no gate. They were a flat script before the fold-in, and running the pipeline redrew them on every
pass whether or not anyone wanted them.

They are kept because they are the manuscript's figures, not the account's. Run this after a
valuation has produced `integrated_country_gep.csv`:

    python -m global_invest.erosion.make_erosion_figures

The one CSV it writes beside them, `integrated_country_gep_plus_overlap.csv`, is also read by
nothing; it is the figure data, and it stays with the figures.
"""
from __future__ import annotations

import glob
import os

import geopandas as gpd
import hazelbean as hb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rioxarray as rxr

from global_invest import utilities
from global_invest.erosion import erosion_tasks


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


def generate_all_maps_and_figures(p, paths):
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



if __name__ == '__main__':
    # erosion_paths reads the numbers from erosion_gep_output_dir, which publish_inputs sets to
    # wherever the valuation wrote them, and the figure directory from cur_dir. There is no
    # cur_dir outside a task, so name one before either call.
    p = hb.ProjectFlow(project_name='gep_erosion', run_mode='check')
    p.cur_dir = os.path.join(p.project_dir, 'output', 'erosion_figures')
    hb.create_directories([p.cur_dir])
    erosion_tasks.publish_inputs(p)
    paths = erosion_tasks.erosion_paths(p)
    utilities.assert_exists(paths.output.integrated_country_gep,
                            'Run the erosion valuation first: the figures draw its output.')
    generate_all_maps_and_figures(p, paths)
    hb.log('Figures written to ' + paths.output.figure_directory)

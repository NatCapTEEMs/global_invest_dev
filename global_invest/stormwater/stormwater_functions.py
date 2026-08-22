"""Stormwater retention: InVEST urban stormwater retention volume times a price per m3.

The retained volumes come from an InVEST Urban Stormwater Retention run over the inputs the
drive staged (biophysical coefficient table, ESA 2020 land cover, WorldClim precipitation,
SLGWRB soils). That run is made outside the task tree, and the step that turns its retention
raster into per-country volumes is stormwater_zonal.py in this folder;
base_data/global_invest/stormwater/run_recipe.md records the run's configuration and inputs.
The tasks module reads the per-country table those two steps produce, and the function below
prices it.

The run's replacement cost is 1 USD/m3, a placeholder rather than a valuation. The intended
price per cubic metre is the open ask, held in one named constant until the author answers.
"""
import pandas as pd

# The committed InVEST configuration prices retention at 1 USD/m3. That is a placeholder
# (the ask on the status sheet); every output built on it is provisional by construction.
STORMWATER_PRICE_PER_M3_PLACEHOLDER = 1.0


def stormwater_gep_by_country(retention_m3_df, price_per_m3):
    """One row per country: retained stormwater volume times the price per cubic metre.

    Args:
        retention_m3_df (pd.DataFrame): iso3_r250_label, retention_m3 (annual retained
            volume from the InVEST urban stormwater retention run).
        price_per_m3 (float): USD per cubic metre retained.

    Returns:
        pd.DataFrame: iso3_r250_label, retention_m3, stormwater_gep.
    """
    df = retention_m3_df.copy()
    df['stormwater_gep'] = df['retention_m3'] * float(price_per_m3)
    return df

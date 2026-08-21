"""Stormwater retention: InVEST urban stormwater retention volume times a price per m3.

The drive carries an InVEST 3.14.2 Urban Stormwater Retention configuration with its
inputs staged (biophysical coefficient table, ESA 2020 land cover, WorldClim
precipitation, SLGWRB soils) but no completed run and a replacement-cost price of 1 --
a placeholder, not a valuation. The library runs the retention model itself; the intended
price per cubic metre is the open ask, held in one named constant until the author
answers.
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

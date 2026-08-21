"""Non-timber forest products (NTFP): CWoN NWFP value per accessible forest hectare.

The source repo (geo_ntfp) multiplies the CWoN 2024 NWFP value per hectare per country by
the hectares of ACCESSIBLE forest: ESA CCI forest classes within a 10 km buffer of roads
and rivers. Two script variants exist side by side (a vector-buffer chain and a
distance-raster chain) and no output is committed, so the library runs BOTH and presents
both, the way fire_protection presents its variants.

The committed roads file is unusable (only sidecar files were committed, the .shp never
was), so the accessibility stage runs on a public global roads dataset; the original
roads file remains a replication ask.
"""
import pandas as pd

# From the source scripts: accessibility is a 10 km buffer around roads and rivers, and
# forest is the ESA CCI class range 50-90 inclusive.
NTFP_ACCESS_BUFFER_M = 10_000
ESA_FOREST_CLASS_MIN = 50
ESA_FOREST_CLASS_MAX = 90


def ntfp_gep_by_country(accessible_forest_ha_df, value_per_ha_df, year):
    """One row per country: accessible forest hectares times the NWFP value per hectare.

    Args:
        accessible_forest_ha_df (pd.DataFrame): iso3_r250_label, accessible_forest_ha.
        value_per_ha_df (pd.DataFrame): iso3_r250_label, year, nwfp_value_per_ha
            (the CWoN NWFP series, current USD per hectare).
        year (int): the GEP base year to value at.

    Returns:
        pd.DataFrame: iso3_r250_label, accessible_forest_ha, nwfp_value_per_ha, ntfp_gep.
    """
    values = value_per_ha_df[value_per_ha_df['year'] == int(year)]
    df = accessible_forest_ha_df.merge(
        values[['iso3_r250_label', 'nwfp_value_per_ha']], on='iso3_r250_label', how='left')
    df['ntfp_gep'] = df['accessible_forest_ha'] * df['nwfp_value_per_ha']
    return df

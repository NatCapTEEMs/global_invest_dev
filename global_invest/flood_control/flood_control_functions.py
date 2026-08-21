"""Flood-control (river flood mitigation) science: expected avoided damage per country.

The consortium drive's FINAL_GEP_FLOOD tree carries the full December 2025 pipeline (flood
hazard x JRC depth-damage x protection levels -> expected annual damage with and without
ecosystems) plus its committed global outputs and extensive method notes. The upstream
geospatial chain is taken as given; the committed per-country expected avoided damage
(2019 USD) is the anchor, and its sum reproduces the pipeline's own global summary exactly
($110.59bn, 148 countries non-zero of 236 assessed).
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

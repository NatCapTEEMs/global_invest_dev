"""Timber provision: the committed per-country output from the drive's Forestry pipeline.

The Forestry folder carries the raster pipeline (forest value, biomass, aligned log prices) and
its committed per-country output in the account's vocabulary. The upstream raster valuation is
taken as given; the committed table is the anchor.
"""
import pandas as pd


def timber_gep_by_country(timber_df, countries_df):
    """The committed timber table joined onto the r250 country list, one row per country."""
    df = countries_df.merge(
        timber_df[['iso3_r250_label', 'forestry_gep']].rename(
            columns={'forestry_gep': 'timber_provision_gep'}),
        on='iso3_r250_label', how='left')
    return df

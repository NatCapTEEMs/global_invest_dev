"""Extractive-energy provision (natural gas, petroleum, coal): committed per-fuel outputs.

The consortium drive's three Commercial subfolders each carry a committed per-country GEP CSV
in the account's own vocabulary (iso3_r250 keys). The upstream fuel valuations are taken as
given (their Data folders sit beside the outputs); the module joins the three components and
sums them into the service total -- the same commodity-sum convention extractive_materials
uses, so no combination question arises. Countries absent from a fuel's table stay NaN in
that component; the service total sums the components that exist (a country with only coal
is a coal country, not a missing one).
"""
import numpy as np
import pandas as pd

EXTRACTIVE_ENERGY_FUELS = ('gas', 'coal', 'oil')


def extractive_energy_gep_by_country(gas_df, coal_df, oil_df, countries_df):
    """One row per country: the three fuel components and their sum."""
    df = countries_df.copy()
    for fuel_df, col in ((gas_df, 'gep_gas'), (coal_df, 'gep_coal'), (oil_df, 'gep_oil')):
        component = fuel_df[['iso3_r250_label', col]].rename(
            columns={col: f'extractive_energy_{col.split("_")[1]}_gep'})
        df = df.merge(component, on='iso3_r250_label', how='left')
    component_cols = [f'extractive_energy_{f}_gep' for f in EXTRACTIVE_ENERGY_FUELS]
    df['extractive_energy_gep'] = df[component_cols].sum(axis=1, min_count=1)
    return df

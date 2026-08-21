"""Extractive-energy provision (natural gas, petroleum, coal): CWoN 2024 resource rents.

The valuation is the one the drive's per-fuel method appendices describe: each country's
resource rent in current USD from the World Bank's Changing Wealth of Nations 2024 package
(Output/Latest/<fuel>_rent_cd, the public reproducibility archive), read at the GEP base
year. The drive's three committed per-country CSVs replicate that read exactly -- except
eight rows that carry another country's full value (Christmas Island and Cocos = Australia,
Bouvet and Svalbard = Norway, Tokelau = New Zealand, Western Sahara = Morocco, Palestine =
Israel, and Kosovo = Serbia, where the committed name column itself reads "Serbia"): a
name-join artifact that counts the source countries twice and, for Kosovo, overrides the
CWoN table's own coal value. The module therefore computes from the CWoN tables, one
country one row, and keeps the committed CSVs as the replication anchor with those rows
as named exceptions.

Components sum by the same commodity convention extractive_materials uses. Countries absent
from a fuel's table stay NaN in that component; the service total sums the components that
exist (a country with only coal is a coal country, not a missing one).
"""
import numpy as np
import pandas as pd

EXTRACTIVE_ENERGY_FUELS = ('gas', 'coal', 'oil')

# The committed drive CSVs give each of these rows another country's full rent value
# (verified equal in every fuel where the row is non-empty). The CWoN source has no rows
# for the first six; for Kosovo it has its own coal value, which the duplicate overrode.
COMMITTED_DUPLICATE_VALUE_ROWS = {'BVT': 'NOR', 'SJM': 'NOR', 'CXR': 'AUS', 'CCK': 'AUS',
                                  'TKL': 'NZL', 'ESH': 'MAR', 'PSE': 'ISR', 'XKX': 'SRB'}


def fuel_rent_by_country_from_cwon(cwon_df, year):
    """One row per CWoN country: the fuel's rent in current USD at `year`.

    The CWoN tables are wide (countrycode, countryname, series, YR1970..YR2020) with
    World Bank codes, which match iso3_r250 labels directly (including XKX for Kosovo).
    """
    year_col = f'YR{int(year)}'
    return cwon_df[['countrycode', year_col]].rename(
        columns={'countrycode': 'iso3_r250_label', year_col: 'rent_cd'})


def extractive_energy_gep_from_cwon(gas_cwon, coal_cwon, oil_cwon, countries_df, year):
    """One row per country: the three fuel components from the CWoN rent tables, and their sum."""
    df = countries_df.copy()
    for cwon_df, fuel in ((gas_cwon, 'gas'), (coal_cwon, 'coal'), (oil_cwon, 'oil')):
        component = fuel_rent_by_country_from_cwon(cwon_df, year).rename(
            columns={'rent_cd': f'extractive_energy_{fuel}_gep'})
        df = df.merge(component, on='iso3_r250_label', how='left')
    component_cols = [f'extractive_energy_{f}_gep' for f in EXTRACTIVE_ENERGY_FUELS]
    df['extractive_energy_gep'] = df[component_cols].sum(axis=1, min_count=1)
    return df


def extractive_energy_gep_by_country(gas_df, coal_df, oil_df, countries_df):
    """One row per country from the committed drive CSVs (the replication anchor)."""
    df = countries_df.copy()
    for fuel_df, col in ((gas_df, 'gep_gas'), (coal_df, 'gep_coal'), (oil_df, 'gep_oil')):
        component = fuel_df[['iso3_r250_label', col]].rename(
            columns={col: f'extractive_energy_{col.split("_")[1]}_gep'})
        df = df.merge(component, on='iso3_r250_label', how='left')
    component_cols = [f'extractive_energy_{f}_gep' for f in EXTRACTIVE_ENERGY_FUELS]
    df['extractive_energy_gep'] = df[component_cols].sum(axis=1, min_count=1)
    return df

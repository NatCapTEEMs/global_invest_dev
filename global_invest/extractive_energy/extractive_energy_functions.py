"""Extractive-energy provision (natural gas, petroleum, coal): CWoN 2024 resource rents.

The valuation is the one the drive's per-fuel method appendices describe: each country's
resource rent in current USD from the World Bank's Changing Wealth of Nations 2024 package
(Output/Latest/, the public reproducibility archive), at the GEP base year. Gas and oil rent is
computed as CWoN computes it, production times unit rent from the package's two intermediates, and
checked against the published <fuel>_rent_cd table. Coal rent is the published coal_rent_cd table:
CWoN builds it from mine-level prices and costs by coal type that the package does not ship.
The drive's three committed per-country CSVs replicate the published rents, apart from rounding of
three oil values (AUS, IRQ, USA), South Sudan, which their join leaves empty (its `adm0_a3` is
SDS), and eight rows that carry another country's full value (Christmas Island and Cocos = Australia,
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

import math

EXTRACTIVE_ENERGY_FUELS = ('gas', 'coal', 'oil')

# The fuels whose rent is computed from production and unit rent, and the series label each CWoN
# table states for itself. A table carrying another label is refused rather than multiplied.
COMPUTED_RENT_FUELS = ('gas', 'oil')
CWON_PRODUCTION_SERIES = {'gas': 'Natural Gas, production (TJ)', 'oil': 'Oil, production (barrels)'}
CWON_UNIT_RENT_SERIES = {'gas': 'Natural Gas, unit rents (current USD / TJ)',
                         'oil': 'Oil, unit rents (current USD/barrel)'}

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


def _cwon_series_at_year(cwon_df, year, series, value_name):
    """One CWoN table's `YR<year>` column by country, after checking its label and its keys."""
    labels = list(cwon_df['series'].unique())
    if labels != [series]:
        raise ValueError(f'expected the CWoN series {series!r}, the table carries {labels}')
    repeated = sorted(cwon_df.loc[cwon_df['countrycode'].duplicated(), 'countrycode'].unique())
    if repeated:
        raise ValueError(f'{series}: country codes repeated {repeated}')
    year_col = f'YR{int(year)}'
    return cwon_df[['countrycode', year_col]].rename(columns={year_col: value_name})


def fuel_rent_from_production_and_unit_rent(production_df, unit_rent_df, year, production_series,
                                            unit_rent_series):
    """One row per CWoN country: production times unit rent at `year`, in current USD.

    The two tables are matched on `countrycode`, one to one. A country missing either input has no
    rent, and a zero in either is a zero rent: CWoN floors negative unit rents at zero upstream,
    and nothing is filled here.

    Raises:
        ValueError: if a table carries another series label or repeats a country code.
    """
    production = _cwon_series_at_year(production_df, year, production_series, 'production')
    unit_rent = _cwon_series_at_year(unit_rent_df, year, unit_rent_series, 'unit_rent')
    df = production.merge(unit_rent, on='countrycode', how='outer', validate='one_to_one')
    df['rent_cd'] = df['production'] * df['unit_rent']
    return df[['countrycode', 'rent_cd']].rename(columns={'countrycode': 'iso3_r250_label'})


def compare_with_published_rent(computed, published_cwon_df, year):
    """How a computed rent compares with CWoN's published `<fuel>_rent_cd` at `year`.

    Returns:
        dict: countries valued in both, whether the missing masks are equal, the largest relative
        difference, and both totals summed with math.fsum.
    """
    published = fuel_rent_by_country_from_cwon(published_cwon_df, year)
    j = computed.merge(published, on='iso3_r250_label', how='outer', suffixes=('', '_published'),
                       validate='one_to_one')
    both = j['rent_cd'].notna() & j['rent_cd_published'].notna()
    scale = j.loc[both, 'rent_cd_published'].abs().where(j.loc[both, 'rent_cd_published'] != 0, 1)
    relative = (j.loc[both, 'rent_cd'] - j.loc[both, 'rent_cd_published']).abs() / scale
    return {'countries_both': int(both.sum()),
            'masks_equal': bool((j['rent_cd'].isna() == j['rent_cd_published'].isna()).all()),
            'max_relative_difference': float(relative.max()) if both.any() else 0.0,
            'total_computed': math.fsum(j['rent_cd'].dropna()),
            'total_published': math.fsum(j['rent_cd_published'].dropna())}


def extractive_energy_gep_from_cwon(gas_production, gas_unit_rent, coal_cwon, oil_production,
                                    oil_unit_rent, countries_df, year):
    """One row per country: gas and oil as production times unit rent, coal as CWoN's published
    rent, and their sum.

    Raises:
        ValueError: if a component values a country code that is not an r250 label, or a table
            fails the checks in fuel_rent_from_production_and_unit_rent.
    """
    components = {'coal': fuel_rent_by_country_from_cwon(coal_cwon, year)}
    for fuel, production_df, unit_rent_df in (('gas', gas_production, gas_unit_rent),
                                              ('oil', oil_production, oil_unit_rent)):
        components[fuel] = fuel_rent_from_production_and_unit_rent(
            production_df, unit_rent_df, year, CWON_PRODUCTION_SERIES[fuel], CWON_UNIT_RENT_SERIES[fuel])
    df = countries_df.copy()
    for fuel in EXTRACTIVE_ENERGY_FUELS:
        component = components[fuel]
        outside = component.loc[component['rent_cd'].notna()
                                & ~component['iso3_r250_label'].isin(df['iso3_r250_label']), 'iso3_r250_label']
        if len(outside):
            raise ValueError(f'{fuel}: valued country codes with no r250 row {sorted(outside)}')
        df = df.merge(component.rename(columns={'rent_cd': f'extractive_energy_{fuel}_gep'}),
                      on='iso3_r250_label', how='left', validate='one_to_one')
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

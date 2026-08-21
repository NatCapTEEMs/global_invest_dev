"""Water-supply science. First component: hydropower direct use, CWoN resource-rent method.

The consortium drive's Hydropower folder carries the committed 2019 output
(gep_hydro_directuse_CWONresrent_20260720.csv) but not the script behind it; the method was
identified by reverse-engineering against that anchor and is exact across all 94 valued
countries (ratio constant to 1e-10): the 2019 GEP is CWoN 2024's capitalized hydropower
wealth divided by the annuity factor at the CWoN-standard 4 percent discount over a 100-year
horizon — the constant annual rent the capitalized value implies. The test suite asserts both
the annuity identity and the exact replication. An earlier direct-use script on the drive
(price x quantity from EIA/World Bank series, Nov 2024) is superseded by this method and not
ported.

The other water_supply components on the group's sheet (agriculture, household) are not yet
built anywhere we have seen; the module is laid out to receive them beside hydropower.
"""
import numpy as np
import pandas as pd

HYDROPOWER_DISCOUNT_RATE = 0.04   # the CWoN standard discount
HYDROPOWER_HORIZON_YEARS = 100    # the CWoN capitalization horizon
HYDROPOWER_GEP_YEAR = 2019
# Countries the reference output leaves EMPTY although the CWoN wealth table values them.
# No available material explains the drop (the 2026 script is not on the drive; the package
# lacks its raw workbook), so the exclusion is encoded to reproduce the reference and the
# reason is an open ask on the deck. Changing this list is a reviewed-commit decision.
HYDROPOWER_REFERENCE_EXCLUDED = ('AZE', 'BGR', 'DOM', 'GRC', 'GTM', 'HTI', 'KAZ', 'LBR',
                                 'MAR', 'MKD', 'NIC', 'POL', 'PRT', 'SLV', 'TJK', 'UKR', 'ZAF')


def annuity_factor(rate=HYDROPOWER_DISCOUNT_RATE, years=HYDROPOWER_HORIZON_YEARS):
    """Present value of one dollar per year for the horizon: sum of 1/(1+r)^t, t = 1..years."""
    return float(np.sum(1.0 / (1.0 + rate) ** np.arange(1, years + 1)))


def hydropower_rent_from_wealth(wealth_df, year=HYDROPOWER_GEP_YEAR):
    """CWoN capitalized hydropower wealth -> the implied constant annual rent per country.

    wealth_df: the CWoN hydro_wealth_cd table (countrycode + YR<year> columns, current USD --
    equal to real 2019 USD in the 2019 base year, and the only variant that covers Venezuela).
    Countries without wealth stay NaN -- no data is not zero rent; the reference's unexplained
    exclusions (HYDROPOWER_REFERENCE_EXCLUDED) are set NaN to reproduce it."""
    df = wealth_df[['countrycode', f'YR{year}']].rename(
        columns={'countrycode': 'iso3_r250_label',
                 f'YR{year}': 'hydropower_wealth_usd'})
    df['hydropower_gep'] = df['hydropower_wealth_usd'] / annuity_factor()
    df.loc[df['iso3_r250_label'].isin(HYDROPOWER_REFERENCE_EXCLUDED), 'hydropower_gep'] = float('nan')
    return df


def water_supply_gep_by_country(hydropower_df, countries_df):
    """Join the hydropower rent onto the r250 country list by iso3 label, one row per country."""
    df = countries_df.merge(hydropower_df[['iso3_r250_label', 'hydropower_gep']],
                            on='iso3_r250_label', how='left')
    return df


# =============================================================================
# Water-use components (agriculture / industrial / municipal), ported from the
# drive's two-script R chain: sector water-use efficiency (USD/m3, AQUASTAT
# SDG 6.4.1) x sector withdrawal volume = sector GEP, computed at the AQUASTAT
# survey years. The drive commits the withdrawal table and BOTH outputs (the
# agriculture component and the all-sector total); the cleaned efficiency table
# is on no drive we can reach -- the open staging ask. Until it lands, the
# committed outputs are carried as the components' values and this function
# holds the chain's arithmetic for the from-raw rebuild.
# =============================================================================
WATER_USE_YEARS = (2000, 2005, 2010, 2015)   # the AQUASTAT survey years the chain keeps


def water_use_sector_gep(efficiency_df, withdrawal_df):
    """Sector GEP per country-year: efficiency (USD/m3) x withdrawal volume, the source
    chain's arithmetic. efficiency_df: iso_code, year, wue_irrigation_usdpm3,
    wue_industrial_usdpm3, wue_municipal_usdpm3. withdrawal_df: iso_code, year,
    w_agriculture, w_industry, w_munucipal (the source's spelling, kept)."""
    df = efficiency_df.merge(withdrawal_df, on=['iso_code', 'year'], how='outer')
    df = df[df['year'].isin(WATER_USE_YEARS)]
    df['gep_water_agricultural'] = df['wue_irrigation_usdpm3'] * df['w_agriculture']
    df['gep_water_industrial'] = df['wue_industrial_usdpm3'] * df['w_industry']
    df['gep_water_municipal'] = df['wue_municipal_usdpm3'] * df['w_munucipal']
    return df


def water_use_components_by_country(agriculture_df, all_sector_df, countries_df):
    """The two committed outputs joined onto the r250 country list by label: the agriculture
    component (the sheet's agriculture subgroup) and the all-sector total."""
    df = countries_df.merge(
        agriculture_df[['iso3_r250_label', 'wateruse_ag_gep']].rename(
            columns={'wateruse_ag_gep': 'water_use_agriculture_gep'}),
        on='iso3_r250_label', how='left')
    df = df.merge(
        all_sector_df[['iso3_r250_label', 'wateruse_gep']].rename(
            columns={'wateruse_gep': 'water_use_all_sector_gep'}),
        on='iso3_r250_label', how='left')
    return df

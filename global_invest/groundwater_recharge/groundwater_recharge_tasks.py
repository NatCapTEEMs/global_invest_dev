"""Groundwater-recharge GEP tasks: the avoided-pumping-cost valuation on the committed recharge.

This layer owns every file read and write. The science it calls lives in
groundwater_recharge_functions, which never opens a file.

The country recharge depths are the seasonal water yield model's output, read from the staged
table and taken as given: the model itself runs elsewhere, per watershed. What is computed here
is the valuation on top, anchored against the committed `ground_water_gep.csv`.
"""
import os

import hazelbean as hb
import pandas as pd

from global_invest import utilities
from global_invest.groundwater_recharge import groundwater_recharge_functions as gr

# AQUASTAT names its areas differently from the account's country names where a state has a
# long form; the pairs the withdrawal join needs are spelled out so a silent drop cannot occur.
AQUASTAT_NAME_TO_ISO3 = {
    'Bolivia (Plurinational State of)': 'BOL', 'Brunei Darussalam': 'BRN',
    'China, mainland': 'CHN', 'Democratic People\'s Republic of Korea': 'PRK',
    'Iran (Islamic Republic of)': 'IRN', 'Lao People\'s Democratic Republic': 'LAO',
    'Netherlands (Kingdom of the)': 'NLD', 'Republic of Korea': 'KOR',
    'Republic of Moldova': 'MDA', 'Russian Federation': 'RUS', 'Syrian Arab Republic': 'SYR',
    'Türkiye': 'TUR', 'United Kingdom of Great Britain and Northern Ireland': 'GBR',
    'United Republic of Tanzania': 'TZA', 'United States of America': 'USA',
    'Venezuela (Bolivarian Republic of)': 'VEN', 'Viet Nam': 'VNM',
    'Czechia': 'CZE', 'North Macedonia': 'MKD', 'Palestine': 'PSE',
    'Côte d\'Ivoire': 'CIV', 'Cabo Verde': 'CPV', 'Congo': 'COG',
}

# The electricity price table's own spellings where they differ from the account's names.
PRICE_NAME_TO_ISO3 = {
    'Cape Verde': 'CPV', 'Caribbean Netherlands': 'BES', 'Cocos (Keeling) Islands': 'CCK',
    'Congo': 'COG', 'Congo (Democratic Republic of)': 'COD', 'Czech Republic': 'CZE',
    'Eswatini': 'SWZ', 'Hong Kong': 'HKG', 'Macau': 'MAC', 'Macedonia': 'MKD',
    'Mayotte': 'MYT', 'Palestine, State of': 'PSE', 'Russian Federation': 'RUS',
    'Saint Barthélemy (St. Barts)': 'BLM', 'Saint Helena': 'SHN',
    'Saint Vincent and the Grenadines': 'VCT', 'Saint-Martin (France)': 'MAF',
    'St. Pierre and Miquelon': 'SPM', 'The Netherlands': 'NLD', 'Vietnam': 'VNM',
    'Micronesia (Federated States of)': 'FSM', 'Virgin Islands (British)': 'VGB',
    'Virgin Islands (U.S.)': 'VIR', 'Brunei Darussalam': 'BRN',
    "Lao People's Democratic Republic": 'LAO', 'Åland Islands': 'ALA',
    'United States': 'USA', 'United Kingdom': 'GBR', 'South Korea': 'KOR',
    'Tanzania': 'TZA', 'Syria': 'SYR', 'Moldova': 'MDA', 'Iran': 'IRN',
    'Turkey': 'TUR', 'Bolivia': 'BOL', 'Timor-Leste': 'TLS',
}


def publish_inputs(p):
    """Every GEP task's first line: the groundwater_recharge es_config row and the parameter rows
    from es_parameters (defaults layer -- a caller-set value prevails), the shared country
    references and the results registry."""
    utilities.hydrate_es_config(p, 'groundwater_recharge', log=hb.log)
    utilities.hydrate_es_parameters(p, 'groundwater_recharge', log=hb.log)
    utilities.initialize_country_paths(p, simplified='30sec')
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def gep_calculation(p):
    """GEP valuation for groundwater recharge: avoided pumping cost per country.

    Mean recharge depth times groundwater withdrawal times the cost of a metre of lift at the
    country's electricity tariff, compared per country against the committed table.
    """
    publish_inputs(p)
    service_results, already_done = utilities.begin_gep_calculation(p, 'groundwater_recharge')
    if already_done:
        return

    recharge = hb.df_read(str(p.get_path(p.groundwater_recharge_l_values_input_path)))

    aquastat = hb.df_read(str(p.get_path(p.groundwater_recharge_withdrawal_input_path)))
    aquastat = aquastat[aquastat['Year'] == int(p.groundwater_recharge_withdrawal_year)].copy()
    names = utilities.collapse_countries_to_r250(p.df_countries)[['iso3_r250_label', 'iso3_r250_name']]
    name_to_iso3 = dict(zip(names['iso3_r250_name'], names['iso3_r250_label']))
    name_to_iso3.update(AQUASTAT_NAME_TO_ISO3)
    aquastat['iso3_r250_label'] = aquastat['Area'].map(name_to_iso3)
    unmapped = sorted(aquastat[aquastat['iso3_r250_label'].isna()]['Area'].unique())
    if unmapped:
        hb.log('  ⚠ %d AQUASTAT areas have no country mapping and their withdrawals are dropped: %s'
               % (len(unmapped), ', '.join(unmapped[:8]) + ('...' if len(unmapped) > 8 else '')))
    withdrawal = aquastat.rename(columns={'Value': 'gw_wdw_bil_m3'})[
        ['iso3_r250_label', 'gw_wdw_bil_m3']].dropna(subset=['iso3_r250_label'])

    prices = pd.read_excel(str(p.get_path(p.groundwater_recharge_electricity_price_input_path)))
    prices = prices.rename(columns={'Average price of 1KW/h (USD)': 'electricity_usd_per_kwh'})
    price_name_to_iso3 = dict(name_to_iso3)
    price_name_to_iso3.update(PRICE_NAME_TO_ISO3)
    prices['iso3_r250_label'] = prices['Country name'].map(price_name_to_iso3)
    unpriced = sorted(prices[prices['iso3_r250_label'].isna()]['Country name'].unique())
    if unpriced:
        hb.log('  ⚠ %d price-table names have no country mapping and their tariffs are dropped: %s'
               % (len(unpriced), ', '.join(unpriced[:8]) + ('...' if len(unpriced) > 8 else '')))
    prices = prices[['iso3_r250_label', 'electricity_usd_per_kwh']].dropna(subset=['iso3_r250_label'])

    df_gep = gr.groundwater_gep(recharge, withdrawal, prices,
                                float(p.groundwater_recharge_pump_efficiency))

    committed = hb.df_read(str(p.get_path(p.groundwater_recharge_reference_path)))
    df_gep = df_gep.merge(
        committed.rename(columns={'ground_water_gep': 'groundwater_recharge_gep_reference'})[
            ['iso3_r250_label', 'groundwater_recharge_gep_reference']],
        on='iso3_r250_label', how='left')

    countries = utilities.collapse_countries_to_r250(p.df_countries)
    df_gep = countries.merge(df_gep, on='iso3_r250_label', how='left')
    df_gep['year'] = int(p.gep_base_year)
    utilities.write_gep_by_country(
        p, df_gep[utilities.published_country_columns(df_gep, 'groundwater_recharge')],
        service_results['gep_by_country_base_year'])

    gdf = hb.df_merge(p.gdf_countries_simplified, df_gep, how='outer',
                      left_on='ee_r264_id', right_on='ee_r264_id')
    gdf.to_file(service_results['gep_by_country_base_year'].replace('.csv', '.gpkg'), driver='GPKG')

    ours = df_gep['groundwater_recharge_gep'].sum()
    ref = df_gep['groundwater_recharge_gep_reference'].sum()
    both = df_gep[df_gep['groundwater_recharge_gep'].notna()
                  & df_gep['groundwater_recharge_gep_reference'].notna()]
    agree = (abs(both['groundwater_recharge_gep'] - both['groundwater_recharge_gep_reference'])
             <= 0.01 + 1e-6 * both['groundwater_recharge_gep_reference'].abs()).sum()
    hb.log(f'Total groundwater_recharge GEP for base year {p.gep_base_year}: {ours:,.2f} '
           f'({int(df_gep["groundwater_recharge_gep"].notna().sum())} countries)')
    hb.log(f'  committed reference: {ref:,.2f}; countries agreeing to the cent: {agree} of {len(both)}')
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)

"""Pest-control GEP tasks: the organic-yield-benefit valuation on the staged tables.

This layer owns every file read and write. The science it calls lives in
pest_control_functions, which never opens a file.

The organic areas are FiBL's survey extract and are taken as given; the yields and producer
prices are FAOSTAT 2019 slices; the effect sizes and ratios are parameters with their sources
beside them in es_parameters. The committed `pest_control_gep.csv` is carried beside the result
as the comparison anchor -- it is a Monte Carlo mean from a pipeline that is not on the drive,
so the run log reports coverage and rank agreement rather than a to-the-cent match.
"""
import os

import hazelbean as hb
import numpy as np
import pandas as pd

from global_invest import utilities
from global_invest.pest_control import pest_control_functions as pc

# FiBL names crops its own way; each name the valuation uses is pinned to one FAOSTAT item so a
# silent drop cannot occur. A name mapped to None is a FiBL aggregate or wild-collection crop
# with no FAOSTAT production analogue; its area is reported and excluded, never guessed.


# FAOSTAT and FiBL name countries differently from the account where a state has a long form;
# the pairs the joins need are spelled out so a silent drop cannot occur.
SOURCE_NAME_TO_ISO3 = {
    'Bolivia (Plurinational State of)': 'BOL', 'Brunei Darussalam': 'BRN',
    'China, mainland': 'CHN', 'China': 'CHN',
    "Democratic People's Republic of Korea": 'PRK', 'Iran (Islamic Republic of)': 'IRN',
    "Lao People's Democratic Republic": 'LAO', 'Lao Peoples Democratic Republic': 'LAO',
    'Netherlands (Kingdom of the)': 'NLD', 'Republic of Korea': 'KOR',
    'Republic of Moldova': 'MDA', 'Russian Federation': 'RUS', 'Syrian Arab Republic': 'SYR',
    'Türkiye': 'TUR', 'United Kingdom of Great Britain and Northern Ireland': 'GBR',
    'United Republic of Tanzania': 'TZA', 'United States of America': 'USA',
    'Venezuela (Bolivarian Republic of)': 'VEN', 'Viet Nam': 'VNM', 'Czechia': 'CZE',
    'North Macedonia': 'MKD', 'Palestine': 'PSE', "Côte d'Ivoire": 'CIV', 'Côte dIvoire': 'CIV',
    'Cabo Verde': 'CPV', 'Cape Verde': 'CPV', 'Congo': 'COG',
    'Democratic Republic of the Congo': 'COD', 'United Kingdom': 'GBR', 'United States': 'USA',
    'South Korea': 'KOR', 'Moldova': 'MDA', 'Moldova, Republic of': 'MDA', 'Tanzania': 'TZA',
    'Vietnam': 'VNM', 'Bolivia': 'BOL', 'Iran': 'IRN', 'Syria': 'SYR', 'Laos': 'LAO',
    'Turkey': 'TUR', 'Russia': 'RUS', 'Czech Republic': 'CZE', 'Hong Kong, China': 'HKG',
    'Taiwan, Province of China': 'TWN', 'Taiwan': 'TWN', 'Kosovo': 'XKX',
    'Réunion': 'REU', 'Réunion (France)': 'REU', 'Ivory Coast': 'CIV', 'Swaziland': 'SWZ',
    'Eswatini': 'SWZ', 'Timor-Leste': 'TLS', 'East Timor': 'TLS',
    'Sao Tome and Principe': 'STP',
}


def publish_inputs(p):
    """Every GEP task's first line: the pest_control es_config row and the parameter rows from
    es_parameters (defaults layer -- a caller-set value prevails), the shared country
    references and the results registry."""
    utilities.hydrate_es_config(p, 'pest_control', log=hb.log)
    utilities.hydrate_es_parameters(p, 'pest_control', log=hb.log)
    utilities.initialize_country_paths(p, simplified='30sec')
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def read_fibl_organic_areas(fibl_path, year):
    """Read FiBL's two-row-header workbook into a long crop-country table for one year.

    Row 8 (0-indexed 7) carries the crop name over each pair of columns, row 9 the column kind;
    only the `Organic area [ha]` member of each pair is kept.
    """
    d = pd.read_excel(fibl_path, sheet_name='Data by country', header=None)
    crops = d.iloc[7, 2:].ffill()
    kinds = d.iloc[8, 2:]
    body = d.iloc[9:].reset_index(drop=True)
    body.columns = ['country', 'year'] + [f'{c}|{k}' for c, k in zip(crops, kinds)]
    ha_cols = [c for c in body.columns[2:] if 'Organic area [ha]' in c]
    long = body.melt(id_vars=['country', 'year'], value_vars=ha_cols,
                     var_name='crop', value_name='organic_ha')
    long['crop'] = long['crop'].str.split('|').str[0]
    long['organic_ha'] = pd.to_numeric(long['organic_ha'], errors='coerce')
    long['year'] = pd.to_numeric(long['year'], errors='coerce')
    return long[(long['year'] == year) & long['organic_ha'].notna() & (long['organic_ha'] > 0)]


def gep_calculation(p):
    """GEP valuation for pest control: the yield benefit on organic crops, per country.

    Effect size times organic production times organic producer price, summed over crops,
    compared per country against the committed table (coverage and rank, not to the cent).
    """
    publish_inputs(p)
    service_results, already_done = utilities.begin_gep_calculation(p, 'pest_control')
    if already_done:
        return

    if not p.run_this:
        return p
    fibl = read_fibl_organic_areas(
        str(p.get_path(p.pest_control_fibl_area_input_path)), int(p.gep_base_year))
    names = utilities.collapse_countries_to_r250(p.df_countries)[['iso3_r250_label', 'iso3_r250_name']]
    name_to_iso3 = dict(zip(names['iso3_r250_name'], names['iso3_r250_label']))
    name_to_iso3.update(SOURCE_NAME_TO_ISO3)

    cls = pd.read_csv(str(p.get_path(p.pest_control_fao_classification_input_path)),
                      encoding='utf-8-sig')
    qcl = pd.read_csv(str(p.get_path(p.pest_control_faostat_yield_input_path)))
    # FAOSTAT publishes 'China' (the aggregate with Taiwan, Hong Kong and Macao) BESIDE
    # 'China, mainland'; both names map to CHN, so keeping the aggregate doubles every
    # Chinese (country, item) row at the merge -- $1.96bn of duplicated value when found
    # on 2026-10-19. The plain name stays in the map because FiBL uses it for the mainland.
    qcl = qcl[qcl['Area'] != 'China']
    qcl['iso3_r250_label'] = qcl['Area'].map(name_to_iso3)
    yld = qcl[(qcl['Element'] == 'Yield') & (qcl['Unit'] == 'kg/ha')].dropna(
        subset=['iso3_r250_label'])[['iso3_r250_label', 'Item', 'Value']]
    yld = yld.rename(columns={'Value': 'yield_kg_ha'})
    prc = pd.read_csv(str(p.get_path(p.pest_control_faostat_price_input_path)))
    prc['iso3_r250_label'] = prc['Area'].map(name_to_iso3)
    prc = prc.dropna(subset=['iso3_r250_label'])[['iso3_r250_label', 'Item', 'Value']]
    prc = prc.rename(columns={'Value': 'price_usd_t'})

    lnrr = {'cereals': float(p.pest_control_effect_lnrr_cereals),
            'vegetables': float(p.pest_control_effect_lnrr_vegetables),
            'others': float(p.pest_control_effect_lnrr_others)}
    yield_ratio = {'cereals': float(p.pest_control_yield_ratio_cereals),
                   'fruit_nuts': float(p.pest_control_yield_ratio_fruit_nuts),
                   'vegetables': float(p.pest_control_yield_ratio_vegetables),
                   'pulses': float(p.pest_control_yield_ratio_pulses),
                   'oilseeds': float(p.pest_control_yield_ratio_oilseeds),
                   'stimulants': float(p.pest_control_yield_ratio_stimulants)}
    premium = float(p.pest_control_price_premium)
    d, info = pc.pest_control_by_crop_country(
        fibl, yld, prc, dict(zip(cls['FAO_item'], cls['FAO_group'])), name_to_iso3,
        lnrr, yield_ratio, premium)
    if info['n_unmatched_fibl']:
        hb.log('  %d FiBL aggregate/wild rows (%.1f%% of area) have no FAOSTAT analogue and are excluded'
               % (info['n_unmatched_fibl'], 100 * info['unmatched_area_share']))
    if info['unmapped_countries']:
        hb.log('  %d FiBL countries have no mapping and their areas are dropped: %s'
               % (len(info['unmapped_countries']), ', '.join(info['unmapped_countries'][:8])))
    hb.log('  %d yields and %d prices filled with the item\'s global median (the owner imputes with mixed models here)'
           % (info['n_yield_filled'], info['n_price_filled']))
    d.to_csv(os.path.join(p.cur_dir, 'pest_control_by_crop_country.csv'),
             index=False, encoding='utf-8-sig')

    df_gep = d.groupby('iso3_r250_label')['pest_control_value'].sum(min_count=1).reset_index()
    df_gep = df_gep.rename(columns={'pest_control_value': 'pest_control_gep'})

    committed = hb.df_read(str(p.get_path(p.pest_control_reference_path)))
    df_gep = df_gep.merge(
        committed.rename(columns={'pest_control': 'pest_control_gep_reference'})[
            ['iso3_r250_label', 'pest_control_gep_reference']],
        on='iso3_r250_label', how='outer')

    countries = utilities.collapse_countries_to_r250(p.df_countries)
    df_gep = countries.merge(df_gep, on='iso3_r250_label', how='left')
    df_gep['year'] = int(p.gep_base_year)
    utilities.write_gep_by_country(
        p, df_gep[utilities.published_country_columns(df_gep, 'pest_control')],
        service_results['gep_by_country_base_year'])

    gdf = hb.df_merge(p.gdf_countries_simplified, df_gep, how='outer',
                      left_on='ee_r264_id', right_on='ee_r264_id')
    gdf.to_file(service_results['gep_by_country_base_year'].replace('.csv', '.gpkg'), driver='GPKG')

    ours = df_gep['pest_control_gep'].sum()
    ref = df_gep['pest_control_gep_reference'].sum()
    both = df_gep[(df_gep['pest_control_gep'].fillna(0) > 0)
                  & (df_gep['pest_control_gep_reference'].fillna(0) > 0)]
    corr = np.corrcoef(np.log(both['pest_control_gep']),
                       np.log(both['pest_control_gep_reference']))[0, 1]
    n_missed = int(((df_gep['pest_control_gep_reference'].fillna(0) > 0)
                    & (df_gep['pest_control_gep'].fillna(0) <= 0)).sum())
    hb.log(f'Total pest_control GEP for base year {p.gep_base_year}: {ours:,.2f} '
           f'({int((df_gep["pest_control_gep"].fillna(0) > 0).sum())} countries) '
           f'at price premium {premium}; the value scales linearly in the premium '
           f'(premium 1.0 gives {ours / premium:,.2f})')
    hb.log(f'  committed reference (a Monte Carlo mean from the missing pipeline): {ref:,.2f}; '
           f'log-correlation over shared countries {corr:.3f}; '
           f'reference countries this run leaves unvalued: {n_missed}')
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    if not p.run_this:
        return p
    utilities.render_service_results(p)

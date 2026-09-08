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
FIBL_CROP_TO_FAOSTAT_ITEM = {
    'Almonds': 'Almonds, in shell', 'Andean grains': 'Quinoa', 'Apples': 'Apples',
    'Apricots': 'Apricots', 'Avocados': 'Avocados', 'Bananas': 'Bananas', 'Barley': 'Barley',
    'Beans': 'Beans, dry', 'Blackberries': 'Other berries and fruits of the genus vaccinium n.e.c.',
    'Blueberries': 'Blueberries', 'Buckwheat': 'Buckwheat', 'Cherries': 'Cherries',
    'Chestnuts': 'Chestnuts, in shell', 'Cocoa': 'Cocoa beans',
    'Coconuts for oil': 'Coconuts, in shell', 'Coconuts, no details': 'Coconuts, in shell',
    'Coffee associated with other crops': 'Coffee, green', 'Coffee, no details': 'Coffee, green',
    'Cotton': 'Seed cotton, unginned', 'Cranberries': 'Cranberries', 'Currants': 'Currants',
    'Dates': 'Dates', 'Easy peelers': 'Tangerines, mandarins, clementines',
    'Tangerine': 'Tangerines, mandarins, clementines', 'Figs': 'Figs',
    'Flax': 'Flax, raw or retted', 'Gooseberries': 'Gooseberries',
    'Grain maize and corn cob mix': 'Maize (corn)', 'Grapefruit/Pomelos': 'Pomelos and grapefruits',
    'Grapes, no details': 'Grapes', 'Grapes, other': 'Grapes', 'Grapes, raisins': 'Grapes',
    'Grapes, table': 'Grapes', 'Grapes, wine': 'Grapes', 'Hazelnuts': 'Hazelnuts, in shell',
    'Hemp': 'True hemp, raw or retted', 'Hops': 'Hop cones', 'Kiwis': 'Kiwi fruit',
    'Lemons and limes': 'Lemons and limes', 'Lentils': 'Lentils, dry',
    'Linseed (oil flax)': 'Linseed', 'Lupine': 'Lupins', 'Mustard': 'Mustard seed',
    'Nectarines': 'Peaches and nectarines', 'Oats': 'Oats', 'Oil palm': 'Oil palm fruit',
    'Olives, no details': 'Olives', 'Olives, oil': 'Olives', 'Olives, table': 'Olives',
    'Oranges': 'Oranges', 'Peaches': 'Peaches and nectarines',
    'Peaches and nectarines, no details': 'Peaches and nectarines', 'Pears': 'Pears',
    'Peas': 'Peas, dry', 'Persimmons': 'Persimmons', 'Pineapples': 'Pineapples',
    'Pistachios': 'Pistachios, in shell', 'Plums': 'Plums and sloes',
    'Pomegranate': 'Other fruits, n.e.c.', 'Potatoes': 'Potatoes',
    'Pulses': 'Other pulses n.e.c.', 'Quinces': 'Quinces',
    'Rape and turnip rape': 'Rape or colza seed', 'Raspberries': 'Raspberries', 'Rice': 'Rice',
    'Rye': 'Rye', 'Soybeans': 'Soya beans', 'Spelt': 'Wheat', 'Strawberries': 'Strawberries',
    'Sugar beet': 'Sugar beet', 'Sugarcane': 'Sugar cane', 'Sunflower seed': 'Sunflower seed',
    'Tea': 'Tea leaves', 'Tobacco': 'Unmanufactured tobacco', 'Triticale': 'Triticale',
    'Walnuts, with shell': 'Walnuts, in shell', 'Wheat': 'Wheat',
    'Berries, no details/n.e.c.': 'Other berries and fruits of the genus vaccinium n.e.c.',
    'Berries, other': 'Other berries and fruits of the genus vaccinium n.e.c.',
    'Other cereals n.e.c.': 'Cereals n.e.c.', 'Oilseeds, no details': 'Other oil seeds, n.e.c.',
    'Oilseeds, other, n.e.c': 'Other oil seeds, n.e.c.', 'Brassicas': 'Cabbages',
    'Black chokeberries': None, 'Buckthorn': None, 'Elder': None, 'Rosehips': None,
    'Fodder beet': None, 'Fresh vegetables and melons': None,
    'Fruit, temperate, no details': None, 'Fruit, temperate, other': None,
    'Fruit, tropical and subtropical, no details': None,
    'Fruit, tropical and subtropical, other': None, 'Industrial crops, no details': None,
    'Industrial crops, other': None, 'Nuts, no details': None, 'Nuts, other': None,
    'Pome fruit, no details': None, 'Pome fruit, other': None,
    'Protein crops, no details': None, 'Protein crops, other': None, 'Pumpkin seeds': None,
    'Root crops, no details': None, 'Root crops, other, n.e.c': None,
    'Stone fruit, no details': None, 'Textile crops, no details': None,
    'Vegetables and fruit': None, 'Vegetables, fruit': None,
    'Vegetables, leafy or stalked': None, 'Vegetables, other': None,
    'Vegetables, root tuber and bulb': None,
}

# Which effect size and which organic-to-conventional yield ratio each FAO group carries. The
# vegetable effect extends to oilseeds and pulses and the other-crops effect to stimulants,
# following the appendix's mapping; groups absent here carry a zero effect, deliberately.
GROUP_TO_EFFECT = {
    'Cereals': ('cereals', 'cereals'),
    'Vegetables': ('vegetables', 'vegetables'),
    'Oilseeds and oleaginous fruits': ('vegetables', 'oilseeds'),
    'Pulses (dried leguminous vegetables)': ('vegetables', 'pulses'),
    'Fruit and nuts': ('others', 'fruit_nuts'),
    'Stimulant, spice and aromatic crops': ('others', 'stimulants'),
}

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

    fibl = read_fibl_organic_areas(
        str(p.get_path(p.pest_control_fibl_area_input_path)), int(p.gep_base_year))
    fibl['item'] = fibl['crop'].map(FIBL_CROP_TO_FAOSTAT_ITEM)
    unmatched = fibl[fibl['item'].isna()]
    if len(unmatched):
        hb.log('  %d FiBL aggregate/wild rows (%.1f%% of area) have no FAOSTAT analogue and are excluded'
               % (len(unmatched), 100 * unmatched['organic_ha'].sum() / fibl['organic_ha'].sum()))
    fibl = fibl[fibl['item'].notna()]

    names = utilities.collapse_countries_to_r250(p.df_countries)[['iso3_r250_label', 'iso3_r250_name']]
    name_to_iso3 = dict(zip(names['iso3_r250_name'], names['iso3_r250_label']))
    name_to_iso3.update(SOURCE_NAME_TO_ISO3)
    fibl['iso3_r250_label'] = fibl['country'].map(name_to_iso3)
    unmapped = sorted(fibl[fibl['iso3_r250_label'].isna()]['country'].unique())
    if unmapped:
        hb.log('  ⚠ %d FiBL countries have no mapping and their areas are dropped: %s'
               % (len(unmapped), ', '.join(unmapped[:8])))
    fibl = fibl.dropna(subset=['iso3_r250_label'])

    cls = pd.read_csv(str(p.get_path(p.pest_control_fao_classification_input_path)),
                      encoding='utf-8-sig')
    fibl['group'] = fibl['item'].map(dict(zip(cls['FAO_item'], cls['FAO_group'])))

    qcl = pd.read_csv(str(p.get_path(p.pest_control_faostat_yield_input_path)))
    qcl['iso3_r250_label'] = qcl['Area'].map(name_to_iso3)
    yld = qcl[(qcl['Element'] == 'Yield') & (qcl['Unit'] == 'kg/ha')].dropna(
        subset=['iso3_r250_label'])[['iso3_r250_label', 'Item', 'Value']]
    yld = yld.rename(columns={'Value': 'yield_kg_ha'})
    prc = pd.read_csv(str(p.get_path(p.pest_control_faostat_price_input_path)))
    prc['iso3_r250_label'] = prc['Area'].map(name_to_iso3)
    prc = prc.dropna(subset=['iso3_r250_label'])[['iso3_r250_label', 'Item', 'Value']]
    prc = prc.rename(columns={'Value': 'price_usd_t'})

    d = fibl.merge(yld, left_on=['iso3_r250_label', 'item'],
                   right_on=['iso3_r250_label', 'Item'], how='left').drop(columns=['Item'])
    d = d.merge(prc, left_on=['iso3_r250_label', 'item'],
                right_on=['iso3_r250_label', 'Item'], how='left').drop(columns=['Item'])
    n_yield_filled = int(d['yield_kg_ha'].isna().sum())
    n_price_filled = int(d['price_usd_t'].isna().sum())
    d['yield_kg_ha'] = d['yield_kg_ha'].fillna(d['item'].map(yld.groupby('Item')['yield_kg_ha'].median()))
    d['price_usd_t'] = d['price_usd_t'].fillna(d['item'].map(prc.groupby('Item')['price_usd_t'].median()))
    hb.log('  %d yields and %d prices filled with the item\'s global median (the owner imputes with mixed models here)'
           % (n_yield_filled, n_price_filled))

    lnrr = {'cereals': float(p.pest_control_effect_lnrr_cereals),
            'vegetables': float(p.pest_control_effect_lnrr_vegetables),
            'others': float(p.pest_control_effect_lnrr_others)}
    yield_ratio = {'cereals': float(p.pest_control_yield_ratio_cereals),
                   'fruit_nuts': float(p.pest_control_yield_ratio_fruit_nuts),
                   'vegetables': float(p.pest_control_yield_ratio_vegetables),
                   'pulses': float(p.pest_control_yield_ratio_pulses),
                   'oilseeds': float(p.pest_control_yield_ratio_oilseeds),
                   'stimulants': float(p.pest_control_yield_ratio_stimulants)}
    d['effect_lnrr'] = d['group'].map(lambda g: lnrr[GROUP_TO_EFFECT[g][0]] if g in GROUP_TO_EFFECT else 0.0)
    d['yield_ratio'] = d['group'].map(lambda g: yield_ratio[GROUP_TO_EFFECT[g][1]] if g in GROUP_TO_EFFECT else 1.0)

    premium = float(p.pest_control_price_premium)
    d = pc.pest_control_value(d, premium)
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
    utilities.render_service_results(p)

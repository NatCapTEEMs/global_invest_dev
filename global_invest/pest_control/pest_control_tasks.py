"""Pest-control GEP tasks: the organic-yield-benefit valuation on the staged tables.

This layer owns every file read and write. The science it calls lives in
pest_control_functions, which never opens a file.

The organic areas are FiBL's survey extract, cleaned in the science layer. The yields and the
conventional producer prices come from the FAOSTAT tables the pollination chain builds, so every
crop service values a crop at the same base-year price. The effect sizes and yield ratios are
parameters with their sources beside them in es_parameters. The reference column is the service
owner's summary of the published simulations, whose prices and yield ratios are older than these,
so the run log reports both totals and where the two tables' masks differ.
"""
import os

import hazelbean as hb
import numpy as np
import pandas as pd

from global_invest import utilities
from global_invest.pest_control import pest_control_functions as pc

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

# FAOSTAT spells this item with an accent in its bulk files and the crop classification without.
FAOSTAT_ITEM_SPELLING = {'Maté leaves': 'Mate leaves'}


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


def read_fibl_organic_areas(fibl_path, year=None):
    """Read FiBL's two-row-header workbook into a long crop-country table.

    Row 8 (0-indexed 7) carries the crop name over each pair of columns, row 9 the column kind;
    only the `Organic area [ha]` member of each pair is kept. With a year, the rows of that year
    with area above zero. Without one, every record with a value, since cleaning reads neighbours.
    """
    d = pd.read_excel(fibl_path, sheet_name='Data by country', header=None)
    crops = d.iloc[7, 2:].ffill()
    kinds = d.iloc[8, 2:]
    body = d.iloc[9:].reset_index(drop=True)
    body.columns = ['country', 'year'] + [f'{c}|{k}' for c, k in zip(crops, kinds)]
    ha_cols = [c for c in body.columns[2:] if pc.ORGANIC_AREA_ELEMENT in c]
    long = body.melt(id_vars=['country', 'year'], value_vars=ha_cols,
                     var_name='crop', value_name='organic_ha')
    long['crop'] = long['crop'].str.split('|').str[0]
    long['organic_ha'] = long['organic_ha'].map(pc.organic_area_from_cell)
    long['year'] = pd.to_numeric(long['year'], errors='coerce')
    if year is None:
        return long[long['organic_ha'].notna()].reset_index(drop=True)
    return long[(long['year'] == year) & long['organic_ha'].notna() & (long['organic_ha'] > 0)]


def read_recent_organic_areas(path):
    """National statistics for years FiBL's extract lacks, in read_fibl_organic_areas' columns."""
    d = pd.read_csv(path, encoding='utf-8-sig')
    d = d[d['element'] == pc.ORGANIC_AREA_ELEMENT]
    return d.rename(columns={'item': 'crop', 'value': 'organic_ha'})[['country', 'year', 'crop', 'organic_ha']]


def read_ponisio_observations(path):
    """The columns of the Ponisio et al. 2015 file that the yield-ratio model reads.

    Every cell starts as text, so the year strings stay as the file holds them. Other columns of the
    file carry accented bytes, which Latin-1 reads without touching these ASCII columns.
    """
    d = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='latin-1',
                    usecols=['Study#', 'Crop species', 'year-org', 'lnRR', 'var lnRR'])
    d = d.rename(columns={'Study#': 'study', 'Crop species': 'crop', 'year-org': 'year', 'lnRR': 'lnrr', 'var lnRR': 'var_lnrr'})
    return d.astype({'lnrr': float, 'var_lnrr': float})[['study', 'year', 'crop', 'lnrr', 'var_lnrr']]


def read_shared_fao_table(values_path, production_path, year):
    """The shared FAOSTAT tables at one year: each crop's yield beside its resolved USD price.

    The values table carries the price and the level it resolved at (the country, or its
    subregion, region or the world), and the production table carries the yield in tonnes per
    hectare. Both are what the pollination chain builds and what crop provision values.
    """
    values = pd.read_csv(values_path, usecols=['area_fao', 'iso3', 'item_fao', 'year', 'price_usd_tonne',
                                               'price_source_agg_level'])
    values = values[values['year'] == year].drop(columns='year')
    production = pd.read_csv(production_path, usecols=['iso3', 'item_fao', 'element', 'year', 'value'])
    yields = production[(production['year'] == year) & (production['element'] == 'Yield')]
    d = values.merge(yields[['iso3', 'item_fao', 'value']], on=['iso3', 'item_fao'], how='left', validate='one_to_one')
    d = d.rename(columns={'item_fao': 'item', 'value': 'yield_t_ha', 'price_usd_tonne': 'price_usd_t',
                          'price_source_agg_level': 'price_level'})
    d['item'] = d['item'].replace(FAOSTAT_ITEM_SPELLING)
    return d


def read_faostat_crop_items_and_area(path, year):
    """FAOSTAT's individual crop items and their harvested areas in the year's slice.

    Item codes from 1700 are FAOSTAT's own aggregates. The 'China' aggregate sits beside 'China,
    mainland' and its parts, so it is dropped.
    """
    qcl = pd.read_csv(path)
    qcl = qcl[(qcl['Area'] != 'China') & (qcl['Item Code'] < 1700)].copy()
    qcl['Item'] = qcl['Item'].replace(FAOSTAT_ITEM_SPELLING)
    harvested = qcl[qcl['Element'] == 'Area harvested'][['Area', 'Item', 'Value']].rename(
        columns={'Area': 'area_fao', 'Item': 'item', 'Value': 'harvested_ha'})
    harvested['year'] = year
    return set(qcl['Item'].unique()), harvested


def parameter_pairs(p, prefix, se_prefix, keys):
    """Pairs of (mean, standard error) from the service's parameter rows, one per key."""
    return {key: (float(getattr(p, prefix + key)), float(getattr(p, se_prefix + key))) for key in keys}


def gep_calculation(p):
    """GEP valuation for pest control: the yield benefit on organic crops, per country.

    The central value per crop and country, then the Monte Carlo over the benefit, the yield ratio
    and the group-mean price ratio. The published value is the Monte Carlo mean, with its 95
    percent interval and the central value beside it. Conventional prices stay fixed in the draws,
    including prices resolved above the country, which the shared table gives no spread for.
    """
    publish_inputs(p)
    service_results, already_done = utilities.begin_gep_calculation(p, 'pest_control')
    if already_done:
        return

    if not p.run_this:
        return p
    base_year = int(p.gep_base_year)
    fao = read_shared_fao_table(str(p.get_path(p.pest_control_fao_values_table_path)),
                                str(p.get_path(p.pest_control_fao_production_table_path)), base_year)
    individual_items, harvested = read_faostat_crop_items_and_area(
        str(p.get_path(p.pest_control_faostat_area_input_path)), base_year)
    organic, info = pc.clean_organic_areas(
        read_fibl_organic_areas(str(p.get_path(p.pest_control_fibl_area_input_path))),
        read_recent_organic_areas(str(p.get_path(p.pest_control_recent_area_input_path))),
        individual_items, harvested, base_year)
    hb.log('  organic area %.0f ha in %d: %.0f ha in countries with no FAOSTAT crop series, %.0f ha in crops with no '
           'FAOSTAT equivalent, %.0f ha in FAOSTAT aggregates, %d isolated records corrected'
           % (info['total_ha'], base_year, info['no_fao_country_ha'], info['no_fao_crop_ha'], info['aggregate_ha'],
              len(info['spikes'])))
    if len(info['spikes']):
        info['spikes'].to_csv(os.path.join(p.cur_dir, 'pest_control_organic_area_spikes.csv'),
                              index=False, encoding='utf-8-sig')

    names = utilities.collapse_countries_to_r250(p.df_countries)[['iso3_r250_label', 'iso3_r250_name']]
    name_to_iso3 = dict(zip(names['iso3_r250_name'], names['iso3_r250_label']))
    name_to_iso3.update(SOURCE_NAME_TO_ISO3)
    name_to_iso3.update(dict(zip(fao['area_fao'], fao['iso3'])))

    classification = pd.read_csv(str(p.get_path(p.pest_control_fao_classification_input_path)),
                                 encoding='utf-8-sig')
    times = float(p.pest_control_organic_premium_plausibility_times)
    premiums, left_out_premiums = pc.plausible_premiums(
        pd.read_csv(str(p.get_path(p.pest_control_organic_premium_input_path)), encoding='utf-8-sig'), times)
    left_out_premiums.to_csv(os.path.join(p.cur_dir, 'pest_control_price_ratios_left_out.csv'), index=False,
                             encoding='utf-8-sig')
    hb.log('  %d matched price ratios kept, %d outside 1/%g to %g left out (%d at %d)'
           % (len(premiums), len(left_out_premiums), times, times,
              int((left_out_premiums['year'] == base_year).sum()), base_year))
    effect = parameter_pairs(p, 'pest_control_effect_lnrr_', 'pest_control_effect_lnrr_se_',
                             ['cereals', 'vegetables', 'others'])
    yield_ratio = parameter_pairs(p, 'pest_control_yield_ratio_lnrr_', 'pest_control_yield_ratio_lnrr_se_',
                                  sorted(pc.YIELD_RATIO_KEY_BY_GROUP.values()))
    include_cereals = bool(p.pest_control_include_cereals)
    d = pc.pest_control_by_crop_country(
        organic, fao, premiums, classification.set_index('item', verify_integrity=True)['FAO_group'].to_dict(),
        name_to_iso3, yield_ratio, effect, include_cereals, base_year)
    d.to_csv(os.path.join(p.cur_dir, 'pest_control_by_crop_country.csv'), index=False, encoding='utf-8-sig')
    unvalued = d[d['pest_control_value'].isna()]
    hb.log('  %d crop-country rows, %d without a value (%.0f ha): %s'
           % (len(d), len(unvalued), unvalued['organic_ha'].sum(),
              unvalued['note'].value_counts().to_dict()))

    premium_parameters = pc.premium_parameters_by_group(premiums)
    labels, totals = pc.monte_carlo_by_country(d, yield_ratio, premium_parameters, effect,
                                               int(p.pest_control_monte_carlo_draws),
                                               int(p.pest_control_monte_carlo_seed))
    df_gep = pd.DataFrame({'iso3_r250_label': labels,
                           'pest_control_gep': totals.mean(axis=0),
                           'pest_control_gep_lower': np.quantile(totals, 0.025, axis=0),
                           'pest_control_gep_upper': np.quantile(totals, 0.975, axis=0)})
    central = d.groupby('iso3_r250_label')['pest_control_value'].sum(min_count=1).rename('pest_control_gep_central')
    df_gep = df_gep.merge(central, left_on='iso3_r250_label', right_index=True, how='left')
    expected = pc.expected_value_by_country(d, yield_ratio, premium_parameters, effect)
    world = totals.sum(axis=1)
    summary = pd.DataFrame({'statistic': ['monte_carlo_mean', 'monte_carlo_lower', 'monte_carlo_upper', 'exact_expectation',
                                          'central_value', 'draws', 'seed'],
                            'value': [world.mean(), np.quantile(world, 0.025), np.quantile(world, 0.975), expected.sum(),
                                      df_gep['pest_control_gep_central'].sum(), totals.shape[0],
                                      int(p.pest_control_monte_carlo_seed)]})
    summary.to_csv(os.path.join(p.cur_dir, 'pest_control_world_uncertainty.csv'), index=False, encoding='utf-8-sig')

    committed = hb.df_read(str(p.get_path(p.pest_control_reference_path)))
    df_gep = df_gep.merge(
        committed.rename(columns={'pest_control': 'pest_control_gep_reference'})[
            ['iso3_r250_label', 'pest_control_gep_reference']],
        on='iso3_r250_label', how='outer')

    countries = utilities.collapse_countries_to_r250(p.df_countries)
    df_gep = countries.merge(df_gep, on='iso3_r250_label', how='left')
    left_out = info['left_out'].assign(iso3_r250_label=lambda f: f['name'].map(name_to_iso3))
    df_gep = pc.qualified_country_table(
        df_gep, pc.coverage_by_country(d), left_out.groupby('iso3_r250_label')['organic_ha'].sum(),
        ['pest_control_gep', 'pest_control_gep_lower', 'pest_control_gep_upper', 'pest_control_gep_central'])
    df_gep['year'] = base_year
    coverage = pc.coverage_summary(df_gep, info, 'pest_control_gep')
    coverage.to_csv(os.path.join(p.cur_dir, 'pest_control_coverage_summary.csv'), index=False, encoding='utf-8-sig')
    hb.log('  coverage: %s. %d crops with a benefit lack an input (%.0f ha), and %.1f%% of the area with a benefit is valued'
           % (df_gep['pest_control_coverage'].value_counts().to_dict(), df_gep['pest_control_crops_missing'].sum(),
              df_gep['pest_control_area_missing_ha'].sum(),
              100 * coverage.set_index('statistic').loc['area_share_with_a_benefit_valued', 'value']))
    utilities.write_gep_by_country(
        p, df_gep[utilities.published_country_columns(df_gep, 'pest_control')],
        service_results['gep_by_country_base_year'])

    gdf = hb.df_merge(p.gdf_countries_simplified, df_gep, how='outer',
                      left_on='ee_r264_id', right_on='ee_r264_id')
    gdf.to_file(service_results['gep_by_country_base_year'].replace('.csv', '.gpkg'), driver='GPKG')

    ours = df_gep['pest_control_gep'].sum()
    hb.log(f'Total pest_control GEP for base year {p.gep_base_year}: {ours:,.2f} '
           f'({int((df_gep["pest_control_gep"].fillna(0) > 0).sum())} countries), 95% interval of the world total '
           f'{np.quantile(world, 0.025):,.2f} to {np.quantile(world, 0.975):,.2f} over {totals.shape[0]} draws; '
           f'exact expectation {expected.sum():,.2f}; central value {df_gep["pest_control_gep_central"].sum():,.2f}')
    masks = pc.reference_masks(df_gep, 'pest_control_gep', 'pest_control_gep_reference')
    hb.log(f'  reference {df_gep["pest_control_gep_reference"].sum():,.2f} over {len(masks["both"])} shared countries. '
           f'A zero here and missing there: {masks["zero_here"]}. A value here and missing there: '
           f'{masks["value_here"]}. A value there and not here: {masks["value_there"]}')
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    if not p.run_this:
        return p
    utilities.render_service_results(p)

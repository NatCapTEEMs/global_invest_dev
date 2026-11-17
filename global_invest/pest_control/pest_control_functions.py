# -*- coding: utf-8 -*-
"""Pest-control science: biological pest control valued as the yield it adds to organic crops.

Nothing here opens a file. The task layer reads the organic areas, the shared FAOSTAT 2019 table,
the premiums and the crop classification, hands the frames in and writes back what it gets, so
every step can be pinned on a hand-built input in the test suite.

The valuation: predators and other natural enemies raise crop yields where synthetic pesticides
are absent, so the service is the share of the organic harvest owed to them, at the organic
producer price. Per crop and country, the value is the yield benefit (from a meta-analytic log
response ratio), times organic production (organic area times the conventional yield times an
organic-to-conventional yield ratio), times the organic producer price (the conventional price
times an organic-to-conventional price ratio). The uncertainty in the benefit, the yield ratio and
the group-mean price ratio is propagated by Monte Carlo draws.

A crop group without a valued benefit carries a zero by the write-up's rule where its area, yield
and price are present. A missing factor leaves a valued crop NA, never a silent zero, and the note names it. A
country's total covers the crops with data, so each country carries how much it covers, and a
country whose computed crops are all zero but which has valued crops without data has no total.
"""
import re

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import norm

ORGANIC_AREA_ELEMENT = 'Organic area [ha]'

# An isolated FiBL record is treated as a reporting error when it exceeds this multiple of the
# adjacent years and the crop's whole harvested area.
SPIKE_MULTIPLE = 10.0

# FiBL country names to the FAOSTAT spelling. None is a country with no FAOSTAT crop series.
FIBL_COUNTRY_TO_FAO = {
    'Andorra': None, 'Kosovo': None, 'Liechtenstein': None, 'Mayotte': None,
    'China': 'China, mainland', 'Cape Verde': 'Cabo Verde', 'Czech Republic': 'Czechia',
    'Côte dIvoire': "Côte d'Ivoire", 'French Guiana (France)': 'French Guiana',
    'Guadeloupe (France)': 'Guadeloupe', "Lao Peoples Democratic Republic": "Lao People's Democratic Republic",
    'Martinique (France)': 'Martinique', 'Moldova': 'Republic of Moldova',
    'Netherlands': 'Netherlands (Kingdom of the)', 'Réunion (France)': 'Réunion',
    'Serbia and Montenegro': None, 'Taiwan': 'China, Taiwan Province of', 'Turkey': 'Türkiye',
    'United Kingdom': 'United Kingdom of Great Britain and Northern Ireland',
}

# FiBL crop names to the FAOSTAT item, group or class. None is a crop with no FAOSTAT equivalent,
# so no yield or price can be found for it. A name left out already is the FAOSTAT name. FAOSTAT
# groups and classes survive this map and are removed afterwards as aggregates, so their area is
# reported rather than guessed onto one crop.
FIBL_CROP_TO_FAO = {
    'Almonds': 'Almonds, in shell', 'Andean grains': None, 'Beans': 'Beans, dry',
    'Berries, no details/n.e.c.': 'Berries and other fruits', 'Berries, other': 'Berries and other fruits',
    'Black chokeberries': None, 'Blackberries': None, 'Brassicas': 'Cabbages', 'Buckthorn': None,
    'Chestnuts': 'Chestnuts, in shell', 'Cocoa': 'Cocoa beans', 'Coconuts for oil': 'Coconuts, in shell',
    'Coconuts, no details': 'Coconuts, in shell', 'Coffee associated with other crops': 'Coffee, green',
    'Coffee, no details': 'Coffee, green', 'Cotton': 'Seed cotton, unginned', 'Dry pulses, mixed': 'Pulses, Total',
    'Easy peelers': 'Tangerines, mandarins, clementines', 'Egg fruit': None, 'Elder': None,
    'Flax': 'Flax, raw or retted', 'Fodder beet': None, 'Fresh vegetables and melons': 'Vegetables Primary',
    'Fruit, temperate, no details': 'Fruit Primary', 'Fruit, temperate, other': 'Fruit Primary',
    'Fruit, tropical and subtropical, no details': 'Fruit Primary',
    'Fruit, tropical and subtropical, other': 'Fruit Primary', 'Grain maize and corn cob mix': 'Maize (corn)',
    'Grapefruit/Pomelos': 'Pomelos and grapefruits', 'Grapes, no details': 'Grapes', 'Grapes, other': 'Grapes',
    'Grapes, raisins': 'Grapes', 'Grapes, table': 'Grapes', 'Grapes, wine': 'Grapes',
    'Hazelnuts': 'Hazelnuts, in shell', 'Hemp': 'Hempseed', 'Hops': 'Hop cones',
    'Industrial crops, no details': None, 'Industrial crops, other': None, 'Kiwis': 'Kiwi fruit',
    'Lentils': 'Lentils, dry', 'Linseed (oil flax)': 'Linseed', 'Litchi': None, 'Lupine': 'Lupins',
    'Mustard': 'Mustard seed', 'Nectarines': 'Peaches and nectarines',
    'Nuts, no details': 'Nuts (excluding wild edible nuts and groundnuts), in shell',
    'Nuts, other': 'Other nuts (excluding wild edible nuts and groundnuts), in shell, n.e.c.',
    'Oil palm': 'Oil palm fruit', 'Oilseeds, no details': 'Oilseeds and oleaginous fruits',
    'Oilseeds, other, n.e.c': 'Other oil seeds, n.e.c.', 'Oleaginous fruits': 'Oilseeds and oleaginous fruits',
    'Olives, no details': 'Olives', 'Olives, oil': 'Olives', 'Olives, other': 'Olives', 'Olives, table': 'Olives',
    'Opuntia': None, 'Other cereals n.e.c.': 'Cereals n.e.c.', 'Other fodder roots': None,
    'Peaches': 'Peaches and nectarines', 'Peaches and nectarines, no details': 'Peaches and nectarines',
    'Peas': 'Peas, green', 'Pistachios': 'Pistachios, in shell', 'Plums': 'Plums and sloes',
    'Pome fruit, no details': 'Pome fruits and stone fruits', 'Pome fruit, other': 'Other pome fruits',
    'Pomegranate': None, 'Protein crops, no details': None, 'Protein crops, other': None,
    'Pulses': 'Pulses, Total', 'Pumpkin seeds': 'Other oil seeds, n.e.c.', 'Rape and turnip rape': 'Rape or colza seed',
    'Root crops, no details': 'Edible roots and tubers with high starch or inulin content',
    'Root crops, other, n.e.c': 'Edible roots and tubers with high starch or inulin content', 'Rosehips': None,
    'Soybeans': 'Soya beans', 'Spelt': None, 'Stone fruit, no details': 'Pome fruits and stone fruits',
    'Stone fruit, other': 'Other stone fruits', 'Sugarcane': 'Sugar cane',
    'Tangerine': 'Tangerines, mandarins, clementines', 'Tea': 'Tea leaves',
    'Textile crops, no details': 'Jute, kenaf, and other textile bast fibres, raw or retted, except flax, true hemp and ramie',
    'Textile crops, other, n.e.c.': 'Jute, kenaf, and other textile bast fibres, raw or retted, except flax, true hemp and ramie',
    'Tobacco': 'Unmanufactured tobacco', 'Vegetables, fruit': 'Fruit-bearing vegetables',
    'Vegetables, leafy or stalked': 'Leafy or stem vegetables', 'Vegetables, other': 'Vegetables',
    'Vegetables, root tuber and bulb': 'Root, bulb or tuberous vegetables', 'Walnuts, with shell': 'Walnuts, in shell',
}

# The meta-analysis category each FAO group's benefit comes from. Cereals are valued only when the
# configuration asks, their effect not differing significantly from zero.
EFFECT_CATEGORY_BY_GROUP = {
    'Fruit and nuts': 'others',
    'Stimulant, spice and aromatic crops': 'others',
    'Vegetables': 'vegetables',
    'Pulses (dried leguminous vegetables)': 'vegetables',
    'Oilseeds and oleaginous fruits': 'vegetables',
    'Other oleaginous fruits': 'vegetables',
    'Cereals': 'cereals',
}

# The FAO groups the meta-analysis does not cover, which carry no benefit. A group in neither table
# is unclassified and leaves the value missing.
NO_BENEFIT_GROUPS = {
    'Edible roots and tubers with high starch or inulin content',
    'Sugar crops',
    'Forage products; fibre crops; plants used in perfumery, pharmacy, or for insecticidal, fungicidal or similar '
    'purposes; beet, forage plant and flower seeds; natural rubber; living plants, cut flowers and flower buds; '
    'unmanufactured tobacco; other raw vegetable materials',
}

# The configuration key of each FAO group's organic-to-conventional yield ratio.
YIELD_RATIO_KEY_BY_GROUP = {
    'Cereals': 'cereals',
    'Fruit and nuts': 'fruit_nuts',
    'Vegetables': 'vegetables',
    'Pulses (dried leguminous vegetables)': 'pulses',
    'Oilseeds and oleaginous fruits': 'oilseeds',
    'Stimulant, spice and aromatic crops': 'stimulants',
}

# Ponisio et al. 2015 crop names to the FAOSTAT item. 'barely' and 'muskmellon' are the file's spellings.
PONISIO_CROP_TO_FAO = {
    'alfalfa': 'Alfalfa', 'apple': 'Apples', 'apricot': 'Apricots', 'banana': 'Bananas', 'barely': 'Barley',
    'barley': 'Barley', 'bean': 'Beans, dry', 'beetroot': 'Carrots and turnips', 'buckwheat': 'Buckwheat',
    'cabbage': 'Cabbages', 'carrot': 'Carrots and turnips', 'cauliflower': 'Cauliflowers and broccoli', 'chard': 'Chard',
    'chili': 'Chillies and peppers, green (Capsicum spp. and Pimenta spp.)', 'coffee': 'Coffee, green',
    'cotton': 'Seed cotton, unginned', 'cucumber': 'Cucumbers and gherkins', 'elephant foot yam': 'Yams', 'endive': 'Endive',
    'fennel': 'Other vegetables, fresh n.e.c.', 'fig': 'Figs', 'flax': 'Flax, raw or retted', 'grapes': 'Grapes',
    'hazelnut': 'Hazelnuts, in shell', 'kiwi': 'Kiwi fruit', 'leek': 'Leeks and other alliaceous vegetables',
    'lettuce': 'Lettuce and chicory', 'maize': 'Maize (corn)', 'melon': 'Cantaloupes and other melons',
    'muskmellon': 'Cantaloupes and other melons', 'oat': 'Oats', 'onion': 'Onions and shallots, dry (excluding dehydrated)',
    'pea grain': 'Peas, dry', 'peach': 'Peaches and nectarines',
    'pepper': 'Chillies and peppers, green (Capsicum spp. and Pimenta spp.)', 'pidgeon pea': 'Pigeon peas, dry',
    'potato': 'Potatoes', 'pumpkin': 'Pumpkins, squash and gourds', 'raisins': 'Grapes', 'rice': 'Rice', 'rye': 'Rye',
    'safflower': 'Safflower seed', 'sorghum': 'Sorghum', 'soybean': 'Soya beans', 'spinach': 'Spinach',
    'spring wheat': 'Wheat', 'strawberry': 'Strawberries', 'sugar beet': 'Sugar beet', 'sunflower': 'Sunflower seed',
    'sweet corn': 'Green corn (maize)', 'sweetpotato': 'Sweet potatoes', 'tomato': 'Tomatoes', 'watermelon': 'Watermelons',
    'wheat': 'Wheat',
}

# Items the FAO classification does not hold, with the group the yield-ratio model gives them.
YIELD_RATIO_ITEM_GROUP_FILL = {
    'Alfalfa': 'Forage products; fibre crops; plants used in perfumery, pharmacy, or for insecticidal, fungicidal or similar '
               'purposes; beet, forage plant and flower seeds; natural rubber; living plants, cut flowers and flower buds; '
               'unmanufactured tobacco; other raw vegetable materials',
    'Chard': 'Vegetables',
    'Endive': 'Vegetables',
}

# The yield-ratio model's random intercepts, outermost first, each nested in the ones before it.
YIELD_RATIO_LEVELS = ['study', 'year', 'item', 'observation']

# A free variance component counts as converged when the REML gradient is below this.
REML_GRADIENT_TOLERANCE = 1e-9


def yield_benefit_from_lnrr(lnrr):
    """The share of OBSERVED production the service provides: 1 - exp(-lnRR).

    Boldorini et al.'s lnRR is ln(yield with predators / yield without), and the observed
    organic yields already have the predators present, so the avoided loss measured from the
    observed level is 1 - exp(-lnRR). exp(lnRR) - 1 would read the observed yield as the
    without-enemies yield and overstate the service by the factor exp(lnRR).

    Args:
        lnrr: the meta-analytic log response ratio, elementwise over whatever is passed.

    Returns:
        The share of observed production attributable to the service (0.12 means 12 percent
        of what is harvested is owed to the predators).
    """
    return 1.0 - np.exp(-np.asarray(lnrr))


def organic_area_from_cell(value):
    """The area a FiBL cell reports: its number, or the first of a cell holding two.

    Fourteen cells hold two numbers as text, such as '15899.42, 493.11', mostly for rye, with no
    label saying what the second is. The first is the value the service owner's reading keeps. A
    dash is no value.
    """
    if isinstance(value, str):
        match = re.search(r'\d+(?:\.\d+)?', value)
        return float(match.group()) if match else np.nan
    return float(value)


def _recode(values, mapping):
    """A series mapped through a recode table, with names the table leaves out kept as they are."""
    return values.map(lambda value: mapping.get(value, value))


def correct_organic_area_spikes(areas, harvested):
    """Replace isolated organic-area records that cannot be real.

    A record is corrected when it exceeds SPIKE_MULTIPLE times its adjacent years and also the
    crop's whole harvested area in the country, so real expansions stay. It takes the mean of the
    adjacent years, or the one adjacent year at the start or end of a series. A record inside a
    series with a gap on either side is not checked, and neither is a year without a harvested area.

    Args:
        areas (pd.DataFrame): `area_fao`, `item`, `year`, `organic_ha`, one row per key.
        harvested (pd.DataFrame): `area_fao`, `item`, `year`, `harvested_ha`.

    Returns:
        tuple: (areas with the corrections applied, one row per corrected record).
    """
    d = areas.sort_values(['area_fao', 'item', 'year']).reset_index(drop=True)
    by_series = d.groupby(['area_fao', 'item'])
    d['ha_prev'] = by_series['organic_ha'].shift(1).where(by_series['year'].shift(1) == d['year'] - 1)
    d['ha_next'] = by_series['organic_ha'].shift(-1).where(by_series['year'].shift(-1) == d['year'] + 1)
    first_year = d['year'] == by_series['year'].transform('min')
    last_year = d['year'] == by_series['year'].transform('max')
    both = d['ha_prev'].notna() & d['ha_next'].notna()
    d['replacement_ha'] = np.select(
        [both, first_year & d['ha_next'].notna(), last_year & d['ha_prev'].notna()],
        [(d['ha_prev'] + d['ha_next']) / 2.0, d['ha_next'], d['ha_prev']], default=np.nan)
    d = d.merge(harvested, on=['area_fao', 'item', 'year'], how='left', validate='one_to_one')
    is_spike = (d['replacement_ha'].notna()
                & (d['organic_ha'] > SPIKE_MULTIPLE * d[['ha_prev', 'ha_next']].max(axis=1))
                & d['harvested_ha'].notna() & (d['organic_ha'] > d['harvested_ha']))
    spikes = d.loc[is_spike, ['area_fao', 'item', 'year', 'organic_ha', 'ha_prev', 'ha_next', 'harvested_ha',
                              'replacement_ha']].reset_index(drop=True)
    d['organic_ha'] = d['organic_ha'].where(~is_spike, d['replacement_ha'])
    return d[areas.columns.tolist()], spikes


def clean_organic_areas(fibl, recent, individual_items, harvested, year):
    """FiBL's organic areas cleaned to one row per FAOSTAT crop and country in one year.

    The whole series is cleaned because the correction of an isolated record reads its neighbours.
    FiBL's crop and country names go to FAOSTAT's, a crop with no FAOSTAT equivalent and a FAOSTAT
    aggregate are removed and their area reported, several FiBL crops under one FAOSTAT item are
    summed, a negative area becomes zero, and spikes are corrected.

    Args:
        fibl (pd.DataFrame): `country`, `year`, `crop`, `organic_ha`, every year with a value.
        recent (pd.DataFrame): the same columns, national statistics for years FiBL lacks.
        individual_items (set): FAOSTAT's individual crop items.
        harvested (pd.DataFrame): `area_fao`, `item`, `year`, `harvested_ha`.
        year (int): the year returned.

    Returns:
        tuple: (`area_fao`, `item`, `organic_ha` with area above zero, info). `info` carries the
        year's `total_ha` and what left it: `no_fao_country_ha`, `no_fao_crop_ha`,
        `aggregate_ha`, the same per country in `left_out` (`name`, `reason`, `organic_ha`), and
        the `spikes` corrected in that year.
    """
    d = pd.concat([fibl, recent], ignore_index=True)
    d['area_fao'] = _recode(d['country'], FIBL_COUNTRY_TO_FAO)
    d['item'] = _recode(d['crop'], FIBL_CROP_TO_FAO)
    in_year = d['year'] == year
    no_series = d[in_year & d['area_fao'].isna()]
    no_counterpart = d[in_year & d['area_fao'].notna() & d['item'].isna()]
    info = {'total_ha': float(d.loc[in_year, 'organic_ha'].sum()),
            'no_fao_country_ha': float(no_series['organic_ha'].sum()),
            'no_fao_crop_ha': float(no_counterpart['organic_ha'].sum())}
    d = d.dropna(subset=['area_fao', 'item'])
    d = d.groupby(['area_fao', 'year', 'item'], as_index=False)['organic_ha'].sum()
    aggregate = ~d['item'].isin(individual_items)
    aggregates = d[aggregate & (d['year'] == year)]
    info['aggregate_ha'] = float(aggregates['organic_ha'].sum())
    info['left_out'] = pd.concat([
        no_series.assign(name=no_series['country'], reason='no FAOSTAT crop series'),
        no_counterpart.assign(name=no_counterpart['area_fao'], reason='no FAOSTAT counterpart'),
        aggregates.assign(name=aggregates['area_fao'], reason='FAOSTAT aggregate')],
        ignore_index=True)[['name', 'reason', 'organic_ha']]
    d = d[~aggregate].copy()
    d['organic_ha'] = d['organic_ha'].clip(lower=0.0)
    d, spikes = correct_organic_area_spikes(d, harvested)
    info['spikes'] = spikes[spikes['year'] == year].reset_index(drop=True)
    d = d[(d['year'] == year) & (d['organic_ha'] > 0)]
    return d[['area_fao', 'item', 'organic_ha']].reset_index(drop=True), info


def premium_parameters_by_group(premiums):
    """The mean organic-to-conventional price ratio of each FAO group and its log-scale spread.

    A crop, country and year without a matched ratio takes its group's mean. The log-scale mean
    and standard error are what the Monte Carlo draws from, the standard error by the delta method.

    Args:
        premiums (pd.DataFrame): `FAO_group`, `Org_premium`, every matched pair of every year.

    Returns:
        pd.DataFrame: indexed by FAO group, with `mean_ratio`, `log_mean` and `log_se`.
    """
    grouped = premiums.groupby('FAO_group')['Org_premium']
    out = pd.DataFrame({'mean_ratio': grouped.mean()})
    standard_error = grouped.std() / np.sqrt(grouped.count())
    out['log_mean'] = np.log(out['mean_ratio'])
    out['log_se'] = standard_error / out['mean_ratio']
    return out


def pest_control_by_crop_country(organic, fao, premiums, item_to_group, name_to_iso3, yield_ratio_lnrr, effect_lnrr,
                                 include_cereals, year):
    """The per-crop-country pest-control table at the central values of every parameter.

    Args:
        organic (pd.DataFrame): `area_fao`, `item`, `organic_ha` (clean_organic_areas output).
        fao (pd.DataFrame): `area_fao`, `item`, `yield_t_ha`, `price_usd_t`, `price_level`, the
            shared FAOSTAT table at the year.
        premiums (pd.DataFrame): `country`, `item`, `year`, `FAO_group`, `Org_premium`.
        item_to_group (dict): FAO item -> FAO group.
        name_to_iso3 (dict): FAOSTAT country names -> iso3_r250_label.
        yield_ratio_lnrr (dict): yield ratio key -> (log mean, log standard error).
        effect_lnrr (dict): effect category -> (log response ratio, standard error).
        include_cereals (bool): whether the cereal effect is valued.
        year (int): the year whose matched price ratios apply.

    Returns:
        pd.DataFrame: one row per crop-country. `pest_control_value` is the central value in USD,
        `benefit_status` says whether the crop's group is valued, carries no benefit or is
        unclassified, `note` names a missing factor, and the Monte Carlo reads the rest.

    Raises:
        pandas.errors.MergeError: when a crop and country repeat in the organic areas, the FAOSTAT
            table or the year's matched price ratios, which would double the value at the merge.
    """
    d = organic.merge(fao, on=['area_fao', 'item'], how='left', validate='one_to_one')
    d['iso3_r250_label'] = d['area_fao'].map(name_to_iso3)
    d['group'] = d['item'].map(item_to_group)

    d['effect_category'] = d['group'].map(EFFECT_CATEGORY_BY_GROUP)
    no_benefit = d['group'].isin(NO_BENEFIT_GROUPS)
    if not include_cereals:
        no_benefit = no_benefit | (d['effect_category'] == 'cereals')
        d['effect_category'] = d['effect_category'].where(d['effect_category'] != 'cereals')
    d['benefit_status'] = np.select([d['effect_category'].notna(), no_benefit], ['valued', 'no_benefit'],
                                    default='unclassified')
    d['effect_lnrr'] = d['effect_category'].map(lambda key: effect_lnrr[key][0] if key in effect_lnrr else np.nan)
    d['benefit_share'] = yield_benefit_from_lnrr(d['effect_lnrr'].to_numpy(dtype=float))
    ratio_key = d['group'].map(YIELD_RATIO_KEY_BY_GROUP)
    d['yield_ratio'] = ratio_key.map(lambda key: np.exp(yield_ratio_lnrr[key][0]) if key in yield_ratio_lnrr else np.nan)

    group_ratio = premium_parameters_by_group(premiums)['mean_ratio']
    observed = premiums[premiums['year'] == year].copy()
    observed['area_fao'] = _recode(observed['country'], FIBL_COUNTRY_TO_FAO)
    observed = observed.dropna(subset=['area_fao'])[['area_fao', 'item', 'Org_premium']]
    d = d.merge(observed, on=['area_fao', 'item'], how='left', validate='one_to_one')
    has_observed = d['Org_premium'].notna()
    d['price_ratio_source'] = np.where(has_observed, 'observed', 'group_mean')
    d['price_ratio'] = d['Org_premium'].where(has_observed, d['group'].map(group_ratio))
    d = d.drop(columns='Org_premium')

    d['yield_t_ha'] = d['yield_t_ha'].where(d['yield_t_ha'] > 0)
    d['price_usd_t'] = d['price_usd_t'].where(d['price_usd_t'] > 0)
    d['pest_control_value'] = (d['benefit_share'] * d['organic_ha'] * d['yield_t_ha'] * d['yield_ratio']
                               * d['price_usd_t'] * d['price_ratio'])
    has_base = d['yield_t_ha'].notna() & d['price_usd_t'].notna()
    d.loc[(d['benefit_status'] == 'no_benefit') & has_base, 'pest_control_value'] = 0.0

    absences = {'yield_t_ha': 'no yield', 'price_usd_t': 'no price'}
    valued_absences = {'yield_ratio': 'no yield ratio', 'price_ratio': 'no price ratio'}

    def note(row):
        names = [absences[c] for c in absences if pd.isna(row[c])]
        if row['benefit_status'] == 'valued':
            names += [valued_absences[c] for c in valued_absences if pd.isna(row[c])]
        if row['benefit_status'] == 'unclassified':
            names.append('no crop group')
        return '; '.join(names)

    d['note'] = d.apply(note, axis=1)
    return d


def monte_carlo_by_country(rows, yield_ratio_lnrr, premium_parameters, effect_lnrr, n_draws, seed):
    """Country totals across simulated values of the uncertain parameters.

    One draw per FAO group and simulation for the yield ratio and for the group-mean price ratio,
    and one per effect category and simulation for the benefit, used for every country in that
    group or category, so errors do not cancel when values are summed. Areas, yields, prices and
    matched price ratios are fixed. The legacy generator's stream is frozen by numpy, so a seed
    gives the same draws on any machine.

    Args:
        rows (pd.DataFrame): pest_control_by_crop_country output.
        yield_ratio_lnrr (dict): yield ratio key -> (log mean, log standard error).
        premium_parameters (pd.DataFrame): premium_parameters_by_group output.
        effect_lnrr (dict): effect category -> (log response ratio, standard error).
        n_draws (int): the number of simulations.
        seed (int): the generator's seed.

    Returns:
        tuple: (labels, totals). `labels` lists the countries with a value or a zero, and
        `totals` is one row per simulation and one column per label, in USD.
    """
    computed = rows[rows['pest_control_value'].notna()]
    labels = np.sort(computed['iso3_r250_label'].dropna().unique())
    position = pd.Series(np.arange(len(labels)), index=labels)
    valued = computed[computed['effect_category'].notna() & computed['iso3_r250_label'].notna()]
    groups = np.sort(valued['group'].unique())
    categories = sorted(valued['effect_category'].unique())
    group_position = pd.Series(np.arange(len(groups)), index=groups)

    generator = np.random.RandomState(seed)

    def draws(means, standard_errors):
        means, standard_errors = np.asarray(means, dtype=float), np.asarray(standard_errors, dtype=float)
        return means + standard_errors * generator.standard_normal((n_draws, len(means)))

    ratio_parameters = [yield_ratio_lnrr[YIELD_RATIO_KEY_BY_GROUP[g]] for g in groups]
    yield_ratio = np.exp(draws([m for m, _ in ratio_parameters], [s for _, s in ratio_parameters]))
    premium = np.exp(draws(premium_parameters.loc[groups, 'log_mean'], premium_parameters.loc[groups, 'log_se']))
    benefit = yield_benefit_from_lnrr(draws([effect_lnrr[c][0] for c in categories],
                                            [effect_lnrr[c][1] for c in categories]))
    benefit_by_group = benefit[:, np.array([categories.index(EFFECT_CATEGORY_BY_GROUP[g]) for g in groups], dtype=int)]

    base = valued['organic_ha'] * valued['yield_t_ha'] * valued['price_usd_t']
    is_observed = (valued['price_ratio_source'] == 'observed').to_numpy()
    country_at = valued['iso3_r250_label'].map(position).to_numpy(dtype=int)
    group_at = valued['group'].map(group_position).to_numpy(dtype=int)
    fixed = np.zeros((len(labels), len(groups)))
    drawn = np.zeros((len(labels), len(groups)))
    np.add.at(fixed, (country_at[is_observed], group_at[is_observed]),
              (base * valued['price_ratio']).to_numpy()[is_observed])
    np.add.at(drawn, (country_at[~is_observed], group_at[~is_observed]), base.to_numpy()[~is_observed])

    group_factor = yield_ratio * benefit_by_group
    return labels, group_factor @ fixed.T + (group_factor * premium) @ drawn.T


def expected_value_by_country(rows, yield_ratio_lnrr, premium_parameters, effect_lnrr):
    """The exact expectation of monte_carlo_by_country's totals, the check on its mean.

    The draws are independent across parameters, so the expectation of a row is the product of
    the lognormal means of its yield ratio and price ratio and the expected benefit, which is
    1 - exp(-lnRR + se^2 / 2). It differs from the central value because the draws are lognormal.

    Args:
        rows (pd.DataFrame): pest_control_by_crop_country output.
        yield_ratio_lnrr (dict): yield ratio key -> (log mean, log standard error).
        premium_parameters (pd.DataFrame): premium_parameters_by_group output.
        effect_lnrr (dict): effect category -> (log response ratio, standard error).

    Returns:
        pd.Series: the expected total in USD, indexed by iso3_r250_label.
    """
    computed = rows[rows['pest_control_value'].notna() & rows['iso3_r250_label'].notna()]
    valued = computed[computed['effect_category'].notna()].copy()
    mean_of = lambda key: np.exp(yield_ratio_lnrr[key][0] + yield_ratio_lnrr[key][1] ** 2 / 2.0)
    valued['yield_ratio_mean'] = valued['group'].map(lambda g: mean_of(YIELD_RATIO_KEY_BY_GROUP[g]))
    valued['benefit_mean'] = valued['effect_category'].map(
        lambda c: 1.0 - np.exp(-effect_lnrr[c][0] + effect_lnrr[c][1] ** 2 / 2.0))
    premium_mean = np.exp(premium_parameters['log_mean'] + premium_parameters['log_se'] ** 2 / 2.0)
    valued['price_ratio_mean'] = valued['price_ratio'].where(valued['price_ratio_source'] == 'observed',
                                                             valued['group'].map(premium_mean))
    valued['expected'] = (valued['organic_ha'] * valued['yield_t_ha'] * valued['price_usd_t']
                          * valued['yield_ratio_mean'] * valued['price_ratio_mean'] * valued['benefit_mean'])
    out = valued.groupby('iso3_r250_label')['expected'].sum()
    return out.reindex(np.sort(computed['iso3_r250_label'].unique()), fill_value=0.0)


def reference_masks(table, value_column, reference_column):
    """Where a country table and a reference differ in which countries they value.

    The service owner's summary keeps a country only when its mean is above zero, so a computed
    zero here reads as missing there.

    Args:
        table (pd.DataFrame): one row per country, with `iso3_r250_label` and both columns.
        value_column (str): this table's value.
        reference_column (str): the reference's value.

    Returns:
        dict: sorted labels under `both` (a value in each), `zero_here` (a computed zero here and
        missing there), `value_here` (a value here and missing there) and `value_there` (a value
        there and not here).
    """
    here, there = table[value_column], table[reference_column]
    has_value, has_reference = here > 0, there.notna()
    labels = lambda mask: sorted(table.loc[mask, 'iso3_r250_label'])
    return {'both': labels(has_value & has_reference),
            'zero_here': labels((here == 0) & ~has_reference),
            'value_here': labels(has_value & ~has_reference),
            'value_there': labels(~has_value & has_reference)}


def coverage_by_country(rows):
    """What each country's total covers and what it lacks, read from the crop rows.

    A crop with a benefit and no value is a missing contribution, and an unclassified crop may be
    one. A crop without a benefit contributes nothing with or without its inputs, so it never
    counts as missing. No missing contribution is estimated: the counts and hectares are all that
    is known about it.

    Args:
        rows (pd.DataFrame): pest_control_by_crop_country output.

    Returns:
        pd.DataFrame: indexed by iso3_r250_label. `crops` splits into `crops_valued`,
        `crops_no_benefit` and `crops_missing`, `area_valued_ha`, `area_no_benefit_ha` and
        `area_missing_ha` split the organic area the same way, and `coverage` is one of
        complete_value, partial_value (a total beside missing crops), complete_zero,
        zero_with_missing (computed zeros beside missing crops) and missing_inputs (nothing
        computed).
    """
    d = rows[rows['iso3_r250_label'].notna()]
    has_value = d['pest_control_value'].notna()
    no_benefit = d['benefit_status'] == 'no_benefit'
    kinds = {'valued': has_value & ~no_benefit, 'no_benefit': no_benefit, 'missing': ~has_value & ~no_benefit}
    parts = {'crops': pd.Series(1, index=d.index), 'above_zero': (d['pest_control_value'] > 0).astype(int),
             'computed': has_value.astype(int)}
    for kind, mask in kinds.items():
        parts['crops_' + kind] = mask.astype(int)
        parts['area_%s_ha' % kind] = d['organic_ha'].where(mask, 0.0)
    out = pd.DataFrame(parts).groupby(d['iso3_r250_label']).sum()
    has_missing = out['crops_missing'] > 0
    out['coverage'] = np.select(
        [(out['above_zero'] > 0) & has_missing, out['above_zero'] > 0, (out['computed'] > 0) & has_missing, out['computed'] > 0],
        ['partial_value', 'complete_value', 'zero_with_missing', 'complete_zero'], default='missing_inputs')
    return out.drop(columns=['above_zero', 'computed'])


def qualified_country_table(table, coverage, left_out_ha, value_columns):
    """The country table with its coverage beside each value and only supported values kept.

    A total of computed zeros beside missing crops is not known to be zero, so those values go
    missing and the crops stay counted. A partial total is kept and flagged. A country with no crop
    row reads left_out_area_only when FiBL reports area that cannot be valued, such as an aggregate,
    and no_organic_area otherwise.

    Args:
        table (pd.DataFrame): one row per country, with `iso3_r250_label` and the value columns.
        coverage (pd.DataFrame): coverage_by_country output.
        left_out_ha (pd.Series): FiBL area left out by cleaning, indexed by iso3_r250_label.
        value_columns (list): the columns that report a value.

    Returns:
        pd.DataFrame: the table with `pest_control_coverage`, the crop counts and hectares under a
        `pest_control_` prefix, and `pest_control_area_left_out_ha`.
    """
    out = table.merge(coverage.add_prefix('pest_control_'), left_on='iso3_r250_label', right_index=True, how='left')
    out['pest_control_area_left_out_ha'] = out['iso3_r250_label'].map(left_out_ha).fillna(0.0)
    counts = [c for c in out.columns if c.startswith(('pest_control_crops', 'pest_control_area_'))]
    out[counts] = out[counts].fillna(0)
    out['pest_control_coverage'] = out['pest_control_coverage'].where(
        out['pest_control_coverage'].notna(),
        np.where(out['pest_control_area_left_out_ha'] > 0, 'left_out_area_only', 'no_organic_area'))
    out.loc[out['pest_control_coverage'] == 'zero_with_missing', value_columns] = np.nan
    return out


def coverage_summary(table, info, value_column):
    """World statistics that qualify the total: countries and value by coverage, crops and hectares.

    Args:
        table (pd.DataFrame): qualified_country_table output.
        info (dict): clean_organic_areas' info.
        value_column (str): the published value.

    Returns:
        pd.DataFrame: `statistic`, `value`.
    """
    by_coverage = table.groupby('pest_control_coverage')
    stats = {}
    for status, group in by_coverage:
        stats['countries_' + status] = len(group)
        stats['value_usd_' + status] = group[value_column].sum()
    for kind in ['crops', 'crops_valued', 'crops_no_benefit', 'crops_missing', 'area_valued_ha', 'area_no_benefit_ha',
                 'area_missing_ha']:
        stats[kind] = table['pest_control_' + kind].sum()
    stats['area_share_with_a_benefit_valued'] = stats['area_valued_ha'] / (stats['area_valued_ha'] + stats['area_missing_ha'])
    stats['area_fibl_total_ha'] = info['total_ha']
    stats['area_no_fao_country_ha'] = info['no_fao_country_ha']
    stats['area_no_fao_crop_ha'] = info['no_fao_crop_ha']
    stats['area_aggregate_ha'] = info['aggregate_ha']
    stats['area_spike_correction_ha'] = float((info['spikes']['organic_ha'] - info['spikes']['replacement_ha']).sum())
    return pd.DataFrame({'statistic': list(stats), 'value': list(stats.values())})


def yield_ratio_effect_sizes(observations, item_to_group):
    """The rows of the yield-ratio meta-analysis, each with its FAO item and group.

    The log response ratio and its sampling variance are the file's own columns. A crop outside
    PONISIO_CROP_TO_FAO or an item with no group raises, since dropping it would change the model.

    Args:
        observations (pd.DataFrame): `study`, `year`, `crop`, `lnrr`, `var_lnrr`. Each distinct year
            string is a level of the nested model, so the strings are used as given.
        item_to_group (dict): FAO item -> FAO group.

    Returns:
        pd.DataFrame: `study`, `year`, `item`, `observation`, `group`, `lnrr`, `var_lnrr`.
    """
    unknown = sorted(set(observations['crop']) - set(PONISIO_CROP_TO_FAO))
    if unknown:
        raise ValueError('crops with no FAO item: %s' % unknown)
    d = observations[['study', 'year', 'crop', 'lnrr', 'var_lnrr']].copy()
    d['item'] = d.pop('crop').map(PONISIO_CROP_TO_FAO)
    d['group'] = d['item'].map(lambda item: item_to_group.get(item, YIELD_RATIO_ITEM_GROUP_FILL.get(item)))
    if d['group'].isna().any():
        raise ValueError('items with no FAO group: %s' % sorted(d.loc[d['group'].isna(), 'item'].unique()))
    d['observation'] = np.arange(1, len(d) + 1)
    return d[['study', 'year', 'item', 'observation', 'group', 'lnrr', 'var_lnrr']]


def _reml_blocks(outermost, partitions):
    """Per outermost cluster, its row positions and each partition's same-cluster indicator matrix."""
    blocks = []
    for label in pd.unique(outermost):
        rows = np.flatnonzero(outermost == label)
        grams = [(partition[rows][:, None] == partition[rows][None, :]).astype(float) for partition in partitions]
        blocks.append((rows, grams))
    return blocks


def _reml_terms(theta, y, v, blocks, second_order=False):
    """The REML fit's quantities at one set of variance components.

    The covariance is diag(v) + sum(theta_j G_j), with G_j the same-cluster indicator of partition j.
    It is block diagonal by outermost cluster, so each block is factorized on its own.

    Args:
        theta (np.ndarray): variance components, zero or above.
        y (np.ndarray): effect sizes.
        v (np.ndarray): known sampling variances.
        blocks (list): _reml_blocks output.
        second_order (bool): also return the Hessian.

    Returns:
        dict: `objective` (negative REML log likelihood, with the constants of the REML density),
        `gradient`, `mean`, `se`, and `hessian` when asked.
    """
    n_theta = len(theta)
    log_det, one_one, one_y = 0.0, 0.0, 0.0
    solved = []
    for rows, grams in blocks:
        covariance = np.diag(v[rows]) + sum((t * g for t, g in zip(theta, grams)), np.zeros((len(rows), len(rows))))
        factor = cho_factor(covariance, lower=True)
        log_det += 2.0 * np.log(np.diag(factor[0])).sum()
        inverse = cho_solve(factor, np.eye(len(rows)))
        to_one = inverse.sum(axis=1)
        one_one += to_one.sum()
        one_y += to_one @ y[rows]
        solved.append((rows, grams, inverse, to_one))
    mean = one_y / one_one
    rss = 0.0
    gradient = np.zeros(n_theta)
    trace, u_g_u, w_g_w, w_g_u = np.zeros(n_theta), np.zeros(n_theta), np.zeros(n_theta), np.zeros(n_theta)
    t1, t2, t3 = np.zeros((n_theta, n_theta)), np.zeros((n_theta, n_theta)), np.zeros((n_theta, n_theta))
    for rows, grams, inverse, u in solved:
        w = inverse @ (y[rows] - mean)
        rss += (y[rows] - mean) @ w
        g_u = [g @ u for g in grams]
        g_w = [g @ w for g in grams]
        for j, g in enumerate(grams):
            trace[j] += (inverse * g).sum()
            u_g_u[j] += u @ g_u[j]
            w_g_w[j] += w @ g_w[j]
            w_g_u[j] += w @ g_u[j]
        if second_order:
            inverse_g = [inverse @ g for g in grams]
            for i in range(n_theta):
                for j in range(i, n_theta):
                    t1[i, j] += (inverse_g[i] * inverse_g[j].T).sum()
                    t2[i, j] += g_u[i] @ inverse @ g_u[j]
                    t3[i, j] += g_w[i] @ inverse @ g_w[j]
    k = len(y)
    out = {'objective': 0.5 * ((k - 1) * np.log(2.0 * np.pi) - np.log(k) + log_det + np.log(one_one) + rss),
           'gradient': 0.5 * (trace - u_g_u / one_one - w_g_w), 'mean': mean, 'se': np.sqrt(1.0 / one_one)}
    if second_order:
        hessian = np.zeros((n_theta, n_theta))
        for i in range(n_theta):
            for j in range(i, n_theta):
                trace_pair = t1[i, j] - 2.0 * t2[i, j] / one_one + u_g_u[i] * u_g_u[j] / one_one ** 2
                quadratic = t3[i, j] - w_g_u[i] * w_g_u[j] / one_one
                hessian[i, j] = hessian[j, i] = -0.5 * trace_pair + quadratic
        out['hessian'] = hessian
    return out


def _reml_minimum_from(start, y, v, blocks):
    """Bounded quasi-Newton from one start, then active-set Newton steps to the gradient tolerance."""
    def value_and_gradient(theta):
        terms = _reml_terms(theta, y, v, blocks)
        return terms['objective'], terms['gradient']
    first = minimize(value_and_gradient, start, jac=True, method='L-BFGS-B', bounds=[(0.0, None)] * len(start),
                     options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-8})
    theta = np.maximum(first.x, 0.0)
    for _ in range(200):
        terms = _reml_terms(theta, y, v, blocks, second_order=True)
        gradient = terms['gradient']
        free = (theta > 0.0) | (gradient < 0.0)
        if not free.any() or np.abs(gradient[free]).max() < REML_GRADIENT_TOLERANCE:
            break
        step = -gradient.copy()
        curvature = terms['hessian'][np.ix_(free, free)]
        if np.linalg.eigvalsh(curvature).min() > 0.0:
            step = np.zeros(len(theta))
            step[free] = np.linalg.solve(curvature, -gradient[free])
        step[~free] = 0.0
        for halving in range(60):
            candidate = np.maximum(theta + step * 0.5 ** halving, 0.0)
            at_candidate = _reml_terms(candidate, y, v, blocks)
            # Near the minimum the objective stops resolving a step, while the gradient still can.
            flat = abs(at_candidate['objective'] - terms['objective']) < 1e-10
            kept = (candidate > 0.0) | (at_candidate['gradient'] < 0.0)
            smaller = kept.any() and np.abs(at_candidate['gradient'][kept]).max() < np.abs(gradient[free]).max()
            if at_candidate['objective'] < terms['objective'] or (flat and smaller):
                theta = candidate
                break
        else:
            break
    return theta


def fit_nested_reml(y, v, levels, level_names=None):
    """REML fit of one mean with nested random intercepts and known sampling variances.

    y_i = mean + sum over levels of u_level + e_i, with e_i ~ N(0, v_i) and the v_i known. The mean is
    the generalized least squares estimate, its standard error the inverse of the summed weights, and
    its interval the normal one, as in metafor's rma.mv with method REML and test z.

    Two cases are decided before fitting, because the likelihood cannot separate them:
    - A level with one cluster is the constant the mean already absorbs. The REML likelihood is flat in
      it while the standard error grows with it, so it is fixed at zero, as rma.mv does.
    - A level with the same clusters as an outer level has the same covariance, so only their sum is
      identified. The sum is reported on the outer level and the inner level says so.

    Components are searched from every corner of a grid in data-derived scales, plus zero, and the
    lowest objective is kept, so nothing comes from an earlier fit. A component may finish at exactly
    zero, which is a boundary solution when its gradient there is positive.

    Args:
        y (np.ndarray): effect sizes.
        v (np.ndarray): known sampling variances, above zero.
        levels (list): cluster labels per level, outermost first, each array nested in the one before.
        level_names (list): a name per level. Defaults to level_1, level_2 and so on.

    Returns:
        dict: `mean`, `se`, `ci_lower`, `ci_upper`, `reml_log_likelihood`, `levels` (a frame with
        `level`, `n_clusters`, `sigma2`, `status`, `gradient`), `max_abs_gradient_free`,
        `min_gradient_at_boundary`, `min_hessian_eigenvalue_free`, `n_starts`, `n_starts_at_best`,
        `worst_start_minus_best`, `largest_component_spread_at_best`.
    """
    y, v = np.asarray(y, dtype=float), np.asarray(v, dtype=float)
    if not (np.isfinite(y).all() and np.isfinite(v).all() and (v > 0).all()):
        raise ValueError('effect sizes must be finite and sampling variances finite and above zero')
    names = list(level_names) if level_names is not None else ['level_%d' % (j + 1) for j in range(len(levels))]
    codes = [pd.factorize(np.asarray(level))[0] for level in levels]
    report, partitions, partition_of_level = [], [], {}
    seen = {}
    for name, code in zip(names, codes):
        n_clusters = len(np.unique(code))
        if n_clusters == 1:
            report.append({'level': name, 'n_clusters': n_clusters, 'status': 'fixed_single_level'})
            continue
        key = code.tobytes()
        if key in seen:
            report.append({'level': name, 'n_clusters': n_clusters, 'status': 'shared_with_' + seen[key]})
            continue
        seen[key] = name
        partition_of_level[name] = len(partitions)
        partitions.append(code)
        report.append({'level': name, 'n_clusters': n_clusters, 'status': 'estimated'})
    blocks = _reml_blocks(codes[0], partitions)

    n_theta = len(partitions)
    scale = max(np.var(y, ddof=1) - v.mean(), 1e-3 * np.var(y, ddof=1))
    corners = np.array(np.meshgrid(*[[1e-3 * scale, scale]] * n_theta)).reshape(n_theta, -1).T if n_theta else np.zeros((1, 0))
    starts = np.vstack([np.zeros((1, n_theta)), corners])
    minima = [_reml_minimum_from(start, y, v, blocks) for start in starts]
    objectives = np.array([_reml_terms(theta, y, v, blocks)['objective'] for theta in minima])
    theta = minima[int(objectives.argmin())]
    at_best = objectives - objectives.min() < 1e-8

    terms = _reml_terms(theta, y, v, blocks, second_order=True)
    gradient = terms['gradient']
    free = theta > 0.0
    for row in report:
        j = partition_of_level.get(row['level'])
        row['sigma2'] = 0.0 if row['status'] == 'fixed_single_level' else (np.nan if j is None else float(theta[j]))
        row['gradient'] = np.nan if j is None else float(gradient[j])
        if j is not None and not free[j]:
            row['status'] = 'boundary'
    z = norm.ppf(0.975)
    return {'mean': float(terms['mean']), 'se': float(terms['se']), 'ci_lower': float(terms['mean'] - z * terms['se']),
            'ci_upper': float(terms['mean'] + z * terms['se']), 'reml_log_likelihood': float(-terms['objective']),
            'levels': pd.DataFrame(report)[['level', 'n_clusters', 'sigma2', 'status', 'gradient']],
            'max_abs_gradient_free': float(np.abs(gradient[free]).max()) if free.any() else 0.0,
            'min_gradient_at_boundary': float(gradient[~free].min()) if (~free).any() else np.nan,
            'min_hessian_eigenvalue_free': float(np.linalg.eigvalsh(terms['hessian'][np.ix_(free, free)]).min()) if free.any() else np.nan,
            'n_starts': len(starts), 'n_starts_at_best': int(at_best.sum()),
            'worst_start_minus_best': float(objectives.max() - objectives.min()),
            'largest_component_spread_at_best': float(np.ptp(np.array(minima)[at_best], axis=0).max()) if n_theta else 0.0}


def yield_ratios_by_group(effect_sizes):
    """Fits the nested model on each FAO group and on all crops together.

    Args:
        effect_sizes (pd.DataFrame): yield_ratio_effect_sizes output.

    Returns:
        tuple: the fit (pd.DataFrame, one row per group with the log mean, standard error and
        interval, and the ratio and its delta-method standard error), and the diagnostics
        (pd.DataFrame, one row per group and level, with each group's fit checks repeated).
    """
    fits, diagnostics = [], []
    for group in sorted(effect_sizes['group'].unique()) + ['All crops']:
        d = effect_sizes if group == 'All crops' else effect_sizes[effect_sizes['group'] == group]
        levels = [d.groupby(YIELD_RATIO_LEVELS[:j + 1], sort=False).ngroup().to_numpy() for j in range(len(YIELD_RATIO_LEVELS))]
        fit = fit_nested_reml(d['lnrr'].to_numpy(), d['var_lnrr'].to_numpy(), levels, YIELD_RATIO_LEVELS)
        fits.append({'FAO_group': group, 'n_effect_sizes': len(d), 'n_studies': d['study'].nunique(), 'lnRR_mean': fit['mean'],
                     'lnRR_se': fit['se'], 'lnRR_ci_lower': fit['ci_lower'], 'lnRR_ci_upper': fit['ci_upper'],
                     'Y_difference_org': np.exp(fit['mean']), 'Y_difference_ci_lower': np.exp(fit['ci_lower']),
                     'Y_difference_ci_upper': np.exp(fit['ci_upper']), 'Y_difference_se': np.exp(fit['mean']) * fit['se']})
        checks = {key: fit[key] for key in ['reml_log_likelihood', 'max_abs_gradient_free', 'min_gradient_at_boundary',
                                            'min_hessian_eigenvalue_free', 'n_starts', 'n_starts_at_best',
                                            'worst_start_minus_best', 'largest_component_spread_at_best']}
        diagnostics.append(fit['levels'].assign(FAO_group=group, **checks))
    return pd.DataFrame(fits), pd.concat(diagnostics, ignore_index=True)

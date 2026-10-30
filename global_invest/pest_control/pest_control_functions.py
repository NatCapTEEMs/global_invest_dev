# -*- coding: utf-8 -*-
"""Pest-control science: biological pest control valued as the yield it adds to organic crops.

Nothing here opens a file. The task layer reads the organic areas, yields, prices and the crop
classification, hands the joined frame in and writes back what it gets, so every step can be
pinned on a hand-built input in the test suite.

The valuation: predators and other natural enemies raise crop yields where synthetic pesticides
are absent, so the service is the extra production on organically managed area, at the organic
producer price. Per crop and country, the value is the proportional yield benefit (from a
meta-analytic log response ratio), times organic production (organic area times the
conventional yield scaled by an organic-to-conventional yield ratio), times the organic
producer price (the conventional price times an organic-to-conventional price premium). Crop
groups without a measured benefit carry a zero effect, deliberately. Their value is a true zero
only where the organic area, yield and price are all present; a missing yield or price leaves
the value NA, as for any other crop.
"""
import pandas as pd
import numpy as np

KG_PER_TONNE = 1000.0


def yield_benefit_from_lnrr(lnrr):
    """The share of OBSERVED production the service provides: 1 - exp(-lnRR).

    Boldorini et al.'s lnRR is ln(yield with predators / yield without), and the observed
    organic yields already have the predators present, so the avoided loss measured from the
    observed level is 1 - exp(-lnRR). exp(lnRR) - 1 would read the observed yield as the
    without-enemies yield and overstate the service by the factor exp(lnRR). The service
    owner's script (10_biocontrol_effects.R) applies the same share per Monte Carlo draw.

    Args:
        lnrr: the meta-analytic log response ratio, elementwise over whatever is passed.

    Returns:
        The share of observed production attributable to the service (0.12 means 12 percent
        of what is harvested is owed to the predators).
    """
    return 1.0 - np.exp(-np.asarray(lnrr))


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


def pest_control_by_crop_country(fibl, yields, prices, item_to_group, name_to_iso3,
                                 lnrr_by_effect, yield_ratio_by_group, price_premium):
    """The per-crop-country pest-control table: joins, fills and the valuation in one place.

    The frame-level pipeline behind `pest_control_by_crop_country.csv`, factored out of the
    task so another model (the pest engine's no-spread static limit) can import the logic
    rather than copy it. Opens nothing; the caller reads the staged inputs and shapes them.

    Args:
        fibl (pd.DataFrame): `country`, `crop`, `organic_ha` (read_fibl_organic_areas output).
        yields (pd.DataFrame): `iso3_r250_label`, `Item`, `yield_kg_ha` (FAOSTAT QCL, kg/ha).
        prices (pd.DataFrame): `iso3_r250_label`, `Item`, `price_usd_t` (FAOSTAT PP, USD/t).
        item_to_group (dict): FAO item -> FAO group, from the staged classification.
        name_to_iso3 (dict): source country names -> iso3_r250_label.
        lnrr_by_effect (dict): effect-size key -> lnRR (`cereals`, `vegetables`, `others`).
        yield_ratio_by_group (dict): ratio key -> organic-to-conventional yield ratio.
        price_premium (float): the organic-to-conventional producer price ratio.

    Returns:
        tuple: (table, info). The table is one row per crop-country with
        `pest_control_value` and the absence `note`; `info` carries the diagnostic counts the
        task logs (`n_unmatched_fibl`, `unmatched_area_share`, `unmapped_countries`,
        `n_yield_filled`, `n_price_filled`). A missing yield or price is filled with the
        item's global median before valuation, the recorded deviation from the owner's
        mixed-model imputation; pass unfilled frames through `pest_control_value` directly to
        keep NAs instead.
    """
    fibl = fibl.copy()
    fibl['item'] = fibl['crop'].map(FIBL_CROP_TO_FAOSTAT_ITEM)
    unmatched = fibl[fibl['item'].isna()]
    info = {'n_unmatched_fibl': int(len(unmatched)),
            'unmatched_area_share': (float(unmatched['organic_ha'].sum() / fibl['organic_ha'].sum())
                                     if len(fibl) else 0.0)}
    fibl = fibl[fibl['item'].notna()]

    fibl['iso3_r250_label'] = fibl['country'].map(name_to_iso3)
    info['unmapped_countries'] = sorted(fibl[fibl['iso3_r250_label'].isna()]['country'].unique())
    fibl = fibl.dropna(subset=['iso3_r250_label'])

    fibl['group'] = fibl['item'].map(item_to_group)

    for name, frame in (('yields', yields), ('prices', prices)):
        duplicated = frame.duplicated(['iso3_r250_label', 'Item'])
        if duplicated.any():
            raise ValueError(
                '%s holds %d duplicated (iso3_r250_label, Item) rows (e.g. %s); a duplicate '
                'key doubles every matching crop-country at the merge. FAOSTAT aggregate '
                'areas beside their parts are the known cause.'
                % (name, int(duplicated.sum()),
                   frame.loc[duplicated, ['iso3_r250_label', 'Item']].iloc[0].tolist()))
    d = fibl.merge(yields, left_on=['iso3_r250_label', 'item'],
                   right_on=['iso3_r250_label', 'Item'], how='left').drop(columns=['Item'])
    d = d.merge(prices, left_on=['iso3_r250_label', 'item'],
                right_on=['iso3_r250_label', 'Item'], how='left').drop(columns=['Item'])
    info['n_yield_filled'] = int(d['yield_kg_ha'].isna().sum())
    info['n_price_filled'] = int(d['price_usd_t'].isna().sum())
    d['yield_kg_ha'] = d['yield_kg_ha'].fillna(
        d['item'].map(yields.groupby('Item')['yield_kg_ha'].median()))
    d['price_usd_t'] = d['price_usd_t'].fillna(
        d['item'].map(prices.groupby('Item')['price_usd_t'].median()))

    d['effect_lnrr'] = d['group'].map(
        lambda g: lnrr_by_effect[GROUP_TO_EFFECT[g][0]] if g in GROUP_TO_EFFECT else 0.0)
    d['yield_ratio'] = d['group'].map(
        lambda g: yield_ratio_by_group[GROUP_TO_EFFECT[g][1]] if g in GROUP_TO_EFFECT else 1.0)

    return pest_control_value(d, price_premium), info


def pest_control_value(df, price_premium):
    """The service value per row: benefit times organic production times organic price.

    Args:
        df (pd.DataFrame): one row per crop-country, carrying `organic_ha`, `yield_kg_ha`,
            `price_usd_t`, `effect_lnrr` (0 for crop groups without a measured benefit) and
            `yield_ratio` (organic-to-conventional).
        price_premium (float): the organic-to-conventional producer price ratio.

    Returns:
        pd.DataFrame: the input with `pest_control_value` in USD added. A row missing a yield
        or a price carries NA there, never a silent zero, and the note column names which. This
        holds for a zero-effect crop group too: it is a true zero with an empty note only when
        its yield and price are present.
    """
    out = df.copy()
    out['pest_control_value'] = (
        yield_benefit_from_lnrr(out['effect_lnrr'])
        * out['organic_ha']
        * (out['yield_kg_ha'] / KG_PER_TONNE) * out['yield_ratio']
        * out['price_usd_t'] * price_premium)
    absences = {'yield_kg_ha': 'no yield', 'price_usd_t': 'no price'}
    out['note'] = out.apply(
        lambda row: '; '.join(absences[c] for c in absences if pd.isna(row[c])), axis=1)
    return out

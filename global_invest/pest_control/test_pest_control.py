"""Unit tests for pest_control's valuation.

Every function is pinned on a hand-built frame, so the benefit convention and the missing-input
rule are stated as executable facts, and the staged inputs anchor the joins.
"""
import numpy as np
import pandas as pd

from global_invest.pest_control import pest_control_functions as pc
from global_invest.pest_control import pest_control_tasks


def test_the_benefit_is_the_share_of_observed_production():
    # lnRR 0.1280735 (Boldorini Table S3, cereals): 12.0 percent of the observed harvest is
    # owed to the predators -- the owner-confirmed orientation, smaller than exp(lnRR) - 1.
    assert np.isclose(pc.yield_benefit_from_lnrr(0.1280735), 1.0 - np.exp(-0.1280735))
    assert pc.yield_benefit_from_lnrr(0.0) == 0.0
    assert pc.yield_benefit_from_lnrr(0.1280735) < np.exp(0.1280735) - 1.0


def test_the_valuation_is_benefit_times_organic_production_times_organic_price():
    df = pd.DataFrame({'organic_ha': [100.0], 'yield_kg_ha': [2000.0],
                       'price_usd_t': [500.0], 'effect_lnrr': [0.2], 'yield_ratio': [0.8]})
    out = pc.pest_control_value(df, price_premium=1.38)
    expected = (1.0 - np.exp(-0.2)) * 100.0 * (2000.0 / 1000.0) * 0.8 * 500.0 * 1.38
    assert np.isclose(out['pest_control_value'].iloc[0], expected)
    assert out['note'].iloc[0] == ''


def test_a_zero_effect_group_is_a_true_zero_and_a_missing_price_is_a_named_na():
    df = pd.DataFrame({'organic_ha': [50.0, 50.0], 'yield_kg_ha': [1000.0, 1000.0],
                       'price_usd_t': [400.0, np.nan], 'effect_lnrr': [0.0, 0.2],
                       'yield_ratio': [1.0, 0.8]})
    out = pc.pest_control_value(df, price_premium=1.0)
    assert out['pest_control_value'].iloc[0] == 0.0 and out['note'].iloc[0] == ''
    assert pd.isna(out['pest_control_value'].iloc[1])
    assert 'no price' in out['note'].iloc[1]


def test_the_crop_map_names_only_real_faostat_items():
    import os, tempfile, hazelbean as hb
    from global_invest import utilities
    base = utilities.service_data_dir(
        hb.ProjectFlow(project_dir=os.path.join(tempfile.mkdtemp(), 'anchors')),
        'pest_control')
    qcl = pd.read_csv(os.path.join(base, 'faostat_qcl_2019.csv'))
    items = set(qcl['Item'].unique())
    mapped = {v for v in pc.FIBL_CROP_TO_FAOSTAT_ITEM.values() if v is not None}
    assert mapped <= items, sorted(mapped - items)


def test_the_staged_classification_covers_every_mapped_item():
    import os, tempfile, hazelbean as hb
    from global_invest import utilities
    base = utilities.service_data_dir(
        hb.ProjectFlow(project_dir=os.path.join(tempfile.mkdtemp(), 'anchors')),
        'pest_control')
    cls = pd.read_csv(os.path.join(base, 'FAO_classification.csv'), encoding='utf-8-sig')
    groups = dict(zip(cls['FAO_item'], cls['FAO_group']))
    mapped = {v for v in pc.FIBL_CROP_TO_FAOSTAT_ITEM.values() if v is not None}
    missing = sorted(i for i in mapped if i not in groups)
    assert not missing, missing


# The pair of 2026-10-20: the owner-confirmed orientation (1 - exp(-lnRR)) on the
# China-deduplicated table. Before it, the FAOSTAT China aggregate dropped before the
# join (it doubled every Chinese crop-country row, $1.96bn). At premium 1.0 this
# sits within 3 percent of the owner's committed $6.25bn, where the inflated
# reading sat 20 percent above it.
PUBLISHED_TOTAL_PREMIUM = 6_467_213_436.55
PUBLISHED_TOTAL_NO_PREMIUM = 4_686_386_548.22


def test_the_pipeline_function_reproduces_the_published_total():
    """The factored frame-level pipeline, driven from the staged inputs exactly as the task
    drives it, returns the account's published figure: $10,338,294,391 at the appendix's
    premium of 1.38 (and the premium-1.0 reading beside it). This is the import seam the
    pest engine's no-spread static limit reuses, so the pin is what keeps the two models
    agreeing on the shared special case."""
    import os, tempfile, hazelbean as hb
    from global_invest import utilities
    p = hb.ProjectFlow(project_dir=os.path.join(tempfile.mkdtemp(), 'anchors'))
    utilities.initialize_country_paths(p, simplified='30sec')
    base = utilities.service_data_dir(p, 'pest_control')

    fibl = pest_control_tasks.read_fibl_organic_areas(os.path.join(base, 'Area_org.xlsx'), 2019)
    names = utilities.collapse_countries_to_r250(p.df_countries)[['iso3_r250_label', 'iso3_r250_name']]
    name_to_iso3 = dict(zip(names['iso3_r250_name'], names['iso3_r250_label']))
    name_to_iso3.update(pest_control_tasks.SOURCE_NAME_TO_ISO3)
    cls = pd.read_csv(os.path.join(base, 'FAO_classification.csv'), encoding='utf-8-sig')
    qcl = pd.read_csv(os.path.join(base, 'faostat_qcl_2019.csv'))
    qcl = qcl[qcl['Area'] != 'China']  # the aggregate beside 'China, mainland'; see the task
    qcl['iso3_r250_label'] = qcl['Area'].map(name_to_iso3)
    yld = qcl[(qcl['Element'] == 'Yield') & (qcl['Unit'] == 'kg/ha')].dropna(
        subset=['iso3_r250_label'])[['iso3_r250_label', 'Item', 'Value']].rename(columns={'Value': 'yield_kg_ha'})
    prc = pd.read_csv(os.path.join(base, 'faostat_pp_2019.csv'))
    prc['iso3_r250_label'] = prc['Area'].map(name_to_iso3)
    prc = prc.dropna(subset=['iso3_r250_label'])[['iso3_r250_label', 'Item', 'Value']].rename(columns={'Value': 'price_usd_t'})

    lnrr = {'cereals': 0.52, 'vegetables': 0.41, 'others': 0.26}
    yield_ratio = {'cereals': 0.82, 'fruit_nuts': 0.92, 'vegetables': 0.81,
                   'pulses': 0.88, 'oilseeds': 0.84, 'stimulants': 0.92}
    import csv
    params = {}
    with open(os.path.join(os.path.dirname(pest_control_tasks.__file__), '..',
                           'input_template', 'es_parameters.csv'), encoding='utf-8-sig') as fh:
        for row in csv.reader(fh):
            if row and row[0] == 'pest_control':
                params[row[1]] = row[2]
    lnrr = {k: float(params['pest_control_effect_lnrr_%s' % k]) for k in lnrr}
    yield_ratio = {k: float(params['pest_control_yield_ratio_%s' % k]) for k in yield_ratio}
    premium = float(params['pest_control_price_premium'])

    table, info = pc.pest_control_by_crop_country(
        fibl, yld, prc, dict(zip(cls['FAO_item'], cls['FAO_group'])), name_to_iso3,
        lnrr, yield_ratio, premium)
    total = float(table['pest_control_value'].sum())
    assert abs(total - PUBLISHED_TOTAL_PREMIUM) < 1.0, total
    no_premium, _ = pc.pest_control_by_crop_country(
        fibl, yld, prc, dict(zip(cls['FAO_item'], cls['FAO_group'])), name_to_iso3,
        lnrr, yield_ratio, 1.0)
    assert abs(float(no_premium['pest_control_value'].sum()) - PUBLISHED_TOTAL_NO_PREMIUM) < 2.0


def test_duplicate_faostat_keys_raise_instead_of_doubling():
    """The China-aggregate class of defect: a duplicated (iso3, Item) in yields or prices
    would double every matching crop-country at the merge, so the pipeline refuses it."""
    import pytest
    fibl = pd.DataFrame({'country': ['X'], 'crop': ['Apples'], 'organic_ha': [10.0]})
    yields = pd.DataFrame({'iso3_r250_label': ['XXX', 'XXX'], 'Item': ['Apples', 'Apples'],
                           'yield_kg_ha': [1.0, 2.0]})
    prices = pd.DataFrame({'iso3_r250_label': ['XXX'], 'Item': ['Apples'], 'price_usd_t': [5.0]})
    with pytest.raises(ValueError, match='duplicated'):
        pc.pest_control_by_crop_country(
            fibl, yields, prices, {'Apples': 'Fruit and nuts'}, {'X': 'XXX'},
            {'cereals': .5, 'vegetables': .4, 'others': .3},
            {'cereals': .8, 'fruit_nuts': .9, 'vegetables': .8, 'pulses': .9, 'oilseeds': .8,
             'stimulants': .9}, 1.0)

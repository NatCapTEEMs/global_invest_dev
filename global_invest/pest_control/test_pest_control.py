"""Unit tests for pest_control's valuation.

Every function is pinned on a hand-built frame, so the benefit convention and the missing-input
rule are stated as executable facts, and the staged inputs anchor the joins.
"""
import numpy as np
import pandas as pd

from global_invest.pest_control import pest_control_functions as pc
from global_invest.pest_control import pest_control_tasks


def test_the_benefit_is_the_exponentiated_log_response_ratio():
    # lnRR 0.1280735 (Boldorini Table S3, cereals) means yields 13.7 percent higher.
    assert np.isclose(pc.yield_benefit_from_lnrr(0.1280735), np.exp(0.1280735) - 1.0)
    assert pc.yield_benefit_from_lnrr(0.0) == 0.0


def test_the_valuation_is_benefit_times_organic_production_times_organic_price():
    df = pd.DataFrame({'organic_ha': [100.0], 'yield_kg_ha': [2000.0],
                       'price_usd_t': [500.0], 'effect_lnrr': [0.2], 'yield_ratio': [0.8]})
    out = pc.pest_control_value(df, price_premium=1.38)
    expected = (np.exp(0.2) - 1.0) * 100.0 * (2000.0 / 1000.0) * 0.8 * 500.0 * 1.38
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
    mapped = {v for v in pest_control_tasks.FIBL_CROP_TO_FAOSTAT_ITEM.values() if v is not None}
    assert mapped <= items, sorted(mapped - items)


def test_the_staged_classification_covers_every_mapped_item():
    import os, tempfile, hazelbean as hb
    from global_invest import utilities
    base = utilities.service_data_dir(
        hb.ProjectFlow(project_dir=os.path.join(tempfile.mkdtemp(), 'anchors')),
        'pest_control')
    cls = pd.read_csv(os.path.join(base, 'FAO_classification.csv'), encoding='utf-8-sig')
    groups = dict(zip(cls['FAO_item'], cls['FAO_group']))
    mapped = {v for v in pest_control_tasks.FIBL_CROP_TO_FAOSTAT_ITEM.values() if v is not None}
    missing = sorted(i for i in mapped if i not in groups)
    assert not missing, missing

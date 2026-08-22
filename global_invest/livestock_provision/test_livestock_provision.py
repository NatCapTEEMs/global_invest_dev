"""Unit tests for livestock_provision.

Every function the task calls is pinned on a hand-built table small enough that the expected
number is written out in the assertion: the FAOSTAT clean-up with its code-or-name item
selection, the decade rental-rate lookup, the as-of attribution, the country join, the two
groupings, and the GLEAM feed-share lambda (step two of the port).
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from global_invest.livestock_provision import livestock_provision_functions as lp

# The identifier columns collapse_countries_to_r250 keeps, with one split country: r264 carries
# AAA twice (the canonical row plus a sub-region Z01) and the collapse must keep only AAA.
COUNTRIES = pd.DataFrame({
    'ee_r264_id': [1, 2, 3],
    'iso3_r250_id': [10, 20, 10],
    'ee_r264_label': ['AAA', 'BBB', 'Z01'],
    'iso3_r250_label': ['AAA', 'BBB', 'AAA'],
    'ee_r264_name': ['Aaaland', 'Bbbland', 'Aaaland North'],
    'iso3_r250_name': ['Aaaland', 'Bbbland', 'Aaaland'],
    'continent': ['Africa', 'Asia', 'Africa'],
    'region_un': ['Africa', 'Asia', 'Africa'],
    'region_wb': ['SSA', 'EAP', 'SSA'],
    'income_grp': ['5. Low income', '3. Upper middle income', '5. Low income'],
    'subregion': ['Western Africa', 'Eastern Asia', 'Western Africa'],
})


def _raw_faostat_frame():
    """Four FAOSTAT rows: one selected by item code, one by item name, one aggregate area, and
    one whose element is not gross production value. The year columns are the file's full
    Y1961..Y2022 span with a flag column beside each, because the melt reads all of them."""
    years = range(lp.FAOSTAT_FIRST_YEAR, lp.FAOSTAT_LAST_YEAR + 1)
    frame = pd.DataFrame({
        'Area Code': [1, 1, 2, 223],
        'Area Code (M49)': ["'010", "'010", "'020", "'223"],
        'Area': ['Aaaland', 'Aaaland', 'World', 'Turkiye'],
        'Item Code': [1017, 882, 1017, 1017],
        'Item': ['Meat of goat', 'Raw milk of cattle', 'Meat of goat', 'Meat of goat'],
        'Element Code': [57, 57, 57, 152],
        'Element': ['Gross Production Value'] * 4,
        'Unit': ['1000 USD', '1000 USD', '1000 USD', '1000 USD'],
    })
    year_columns = {}
    for year in years:
        year_columns[f'Y{year}'] = [0.0] * 4
        year_columns[f'Y{year}F'] = ['A'] * 4
    year_columns['Y1961'] = [1.0, 2.0, 3.0, 4.0]
    year_columns['Y2022'] = [5.0, 6.0, 7.0, 8.0]
    return pd.concat([frame, pd.DataFrame(year_columns)], axis=1)


def test_clean_crop_values_selects_items_by_code_or_by_name():
    out = lp.clean_crop_values(_raw_faostat_frame(), items=[1017, 'Raw milk of cattle'])

    # Both Aaaland rows selected: one by code 1017, one by name. 'World' is an aggregate area
    # and area 223's row is element 152.
    assert set(out['country']) == {'Aaaland'}
    assert sorted(out['crop_code'].unique().tolist()) == [882, 1017]
    assert not [c for c in out.columns if c.endswith('F')]      # flag columns dropped
    n_years = lp.FAOSTAT_LAST_YEAR - lp.FAOSTAT_FIRST_YEAR + 1
    assert len(out) == 2 * n_years
    values = out.set_index(['crop_code', 'year'])['livestock_provision_gep']
    assert values.loc[(1017, 1961)] == 1.0
    assert values.loc[(882, 2022)] == 6.0
    assert values.loc[(1017, 1990)] == 0.0


def test_clean_crop_values_drops_an_item_matched_by_neither_code_nor_name():
    out = lp.clean_crop_values(_raw_faostat_frame(), items=[1017])
    assert out['crop_code'].unique().tolist() == [1017]


def test_clean_crop_values_renames_area_223_to_turkey():
    raw = _raw_faostat_frame()
    raw.loc[3, 'Element Code'] = lp.FAOSTAT_GROSS_PRODUCTION_VALUE_ELEMENT
    out = lp.clean_crop_values(raw, items=[1017])
    assert set(out.loc[out['area_code'] == 223, 'country']) == {'Turkey'}


def test_build_rental_rate_lookup_keys_each_decade_by_its_first_year():
    raw = pd.DataFrame({
        'Order': [1, 2],
        'FAO': [1.0, 2.0],
        'ISO3': ['AAA', 'BBB'],
        'Country/territory': ['Aaaland', 'Bbbland'],
        '1961-1970': [0.30, 0.40],
        '2011-2020': [0.35, 0.45],
    })
    out = lp.build_rental_rate_lookup(raw)
    assert sorted(out['year'].unique().tolist()) == [1961, 2011]
    assert out['FAO'].dtype.kind == 'i'
    # ISO3 is not a decade column, so it melts in and then falls out with the unparseable start.
    assert len(out) == 4
    rates = out.set_index(['FAO', 'year'])['rental_rate']
    assert rates.loc[(1, 1961)] == 0.30
    assert rates.loc[(2, 2011)] == 0.45


def test_merge_crop_with_coefs_applies_the_decade_in_force_and_leaves_uncovered_countries_missing():
    values = pd.DataFrame({
        'area_code': [1, 1, 1, 2],
        'year': [1960, 1961, 2015, 2015],
        'livestock_provision_gep': [100.0, 100.0, 200.0, 500.0],
    })
    coefs = pd.DataFrame({'FAO': [1, 1], 'year': [1961, 2011], 'rental_rate': [0.30, 0.35]})

    out = lp.merge_crop_with_coefs(values, coefs).set_index(['area_code', 'year'])['livestock_provision_gep']
    assert np.isnan(out.loc[(1, 1960)])          # before the first decade: no rate in force
    assert out.loc[(1, 1961)] == 30.0            # 100 x 0.30
    assert out.loc[(1, 2015)] == 70.0            # 200 x 0.35, the 2011-2020 decade
    assert pd.isna(out.loc[(2, 2015)])           # country absent from the CWoN table


def test_attach_countries_does_not_repeat_a_split_country_and_keeps_values_as_read():
    items = pd.DataFrame({
        'area_code_M49': [10, 10, 20],
        'area_code': [1, 1, 2],
        'country': ['Aaaland', 'Aaaland', 'Bbbland'],
        'crop_code': [1017, 882, 1017],
        'crop': ['Meat of goat', 'Raw milk of cattle', 'Meat of goat'],
        'year': [2019, 2019, 2019],
        'rental_rate': [0.3, 0.3, 0.4],
        'livestock_provision_gep': [30.0, 15.0, 40.0],
    })
    out = lp.attach_countries(items, COUNTRIES)

    # AAA is split across two r264 rows; the join must still produce three rows, not five.
    assert len(out) == 3
    # No unit conversion happens here, unlike crop_provision: the values pass through unchanged.
    assert out['livestock_provision_gep'].tolist() == [30.0, 15.0, 40.0]
    assert out.set_index('crop_code').loc[1017, 'iso3_r250_label'].tolist() == ['AAA', 'BBB']


def test_attach_countries_keeps_a_row_whose_code_the_correspondence_lacks():
    items = pd.DataFrame({
        'area_code_M49': [10, 999],
        'area_code': [1, 9],
        'country': ['Aaaland', 'Nowhere'],
        'crop_code': [1017, 1017],
        'crop': ['Meat of goat', 'Meat of goat'],
        'year': [2019, 2019],
        'rental_rate': [0.3, 0.3],
        'livestock_provision_gep': [30.0, 7.0],
    })
    out = lp.attach_countries(items, COUNTRIES)
    assert len(out) == 2
    unmatched = out[out['area_code_M49'] == 999].iloc[0]
    assert pd.isna(unmatched['iso3_r250_label'])
    assert unmatched['livestock_provision_gep'] == 7.0


def test_group_crops_then_group_countries_sum_to_the_same_total():
    item_rows = pd.DataFrame({
        'iso3_r250_id': [10, 10, 20, 10],
        'iso3_r250_label': ['AAA', 'AAA', 'BBB', 'AAA'],
        'year': [2019, 2019, 2019, 2018],
        'livestock_provision_gep': [30.0, 15.0, 40.0, 1.0],
    })
    by_country = lp.group_crops(item_rows)
    per_country = by_country.set_index(['iso3_r250_id', 'year'])['livestock_provision_gep']
    assert per_country.loc[(10, 2019)] == 45.0        # two items summed
    assert per_country.loc[(20, 2019)] == 40.0
    assert per_country.loc[(10, 2018)] == 1.0

    by_year = lp.group_countries(by_country).set_index('year')['livestock_provision_gep']
    assert by_year.loc[2019] == 85.0
    assert by_year.loc[2018] == 1.0


def test_normalize_m49_codes_unquotes_casts_and_maps_successors():
    df = pd.DataFrame({'area_code_M49': ["'156", "'159", "'891", "'076"], 'value': [1, 2, 3, 4]})
    out = lp.normalize_m49_codes(df)
    assert list(out['area_code_M49']) == [156, 156, 688, 76]
    assert list(df['area_code_M49']) == ["'156", "'159", "'891", "'076"]   # input untouched


def test_every_successor_maps_to_a_different_live_code():
    """A successor mapping that pointed at itself, or at another dissolved state, would leave
    production stranded."""
    for dissolved, successor in lp.M49_SUCCESSORS.items():
        assert dissolved != successor
        assert successor not in lp.M49_SUCCESSORS


def test_feed_lambda_is_ecosystem_share_of_total_intake():
    rows = []
    # Two species rows for AAA: eco feed 6 (4 grass + 2 residues), total 10.
    rows.append({'iso3_r250_id': 1, 'iso3_r250_label': 'AAA', 'By-products': 0.0, 'Crop residues': 2.0,
                 'Fodder crop': 0.0, 'Grass and leaves': 1.0, 'Grains': 3.0, 'Oil seed cakes': 1.0,
                 'Other edible': 0.0, 'Other non-edible': 0.0})
    rows.append({'iso3_r250_id': 1, 'iso3_r250_label': 'AAA', 'By-products': 0.0, 'Crop residues': 0.0,
                 'Fodder crop': 0.0, 'Grass and leaves': 3.0, 'Grains': 0.0, 'Oil seed cakes': 0.0,
                 'Other edible': 0.0, 'Other non-edible': 0.0})
    # BBB: all feed from grass, lambda 1.
    rows.append({'iso3_r250_id': 2, 'iso3_r250_label': 'BBB', 'By-products': 0.0, 'Crop residues': 0.0,
                 'Fodder crop': 0.0, 'Grass and leaves': 5.0, 'Grains': 0.0, 'Oil seed cakes': 0.0,
                 'Other edible': 0.0, 'Other non-edible': 0.0})
    out = lp.feed_lambda_by_country(pd.DataFrame(rows)).set_index('iso3_r250_label')['lambda']
    assert out['AAA'] == 0.6
    assert out['BBB'] == 1.0


def test_the_feed_category_lists_are_pinned():
    assert lp.GLEAM_ECOSYSTEM_FEED_COLS == ("By-products", "Crop residues", "Fodder crop", "Grass and leaves")
    assert len(lp.GLEAM_TOTAL_FEED_COLS) == 8


def test_es_config_row_hydrates_livestock_provision(tmp_path):
    from global_invest import utilities
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'livestock_provision', log=lambda *a: None)
    assert p.gep_base_year == 2019

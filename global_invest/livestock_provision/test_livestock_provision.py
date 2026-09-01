"""Unit tests for livestock_provision.

Every function the task calls is pinned on a hand-built table small enough that the expected
number is written out in the assertion: the FAOSTAT clean-up with its code-or-name item
selection, the decade rental-rate lookup, the as-of attribution, the country join, the two
groupings, and the GLEAM feed-share lambda (step two of the port). The task module's two readers
are pinned on files written into tmp_path.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from global_invest import utilities
from global_invest.livestock_provision import livestock_provision_functions as lp
from global_invest.livestock_provision import livestock_provision_tasks as lpt

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
    years = range(utilities.FAOSTAT_FIRST_YEAR, utilities.FAOSTAT_LAST_YEAR + 1)
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
    out = utilities.clean_faostat_values(_raw_faostat_frame(), items=[1017, 'Raw milk of cattle'],
                                         value_column='livestock_provision_gep',
                             aggregate_areas=['World'])

    # Both Aaaland rows selected: one by code 1017, one by name. 'World' is an aggregate area
    # and area 223's row is element 152.
    assert set(out['country']) == {'Aaaland'}
    assert sorted(out['crop_code'].unique().tolist()) == [882, 1017]
    assert not [c for c in out.columns if c.endswith('F')]      # flag columns dropped
    n_years = utilities.FAOSTAT_LAST_YEAR - utilities.FAOSTAT_FIRST_YEAR + 1
    assert len(out) == 2 * n_years
    values = out.set_index(['crop_code', 'year'])['livestock_provision_gep']
    assert values.loc[(1017, 1961)] == 1.0
    assert values.loc[(882, 2022)] == 6.0
    assert values.loc[(1017, 1990)] == 0.0


def test_clean_crop_values_drops_an_item_matched_by_neither_code_nor_name():
    out = utilities.clean_faostat_values(_raw_faostat_frame(), items=[1017], value_column='livestock_provision_gep', aggregate_areas=['World'])
    assert out['crop_code'].unique().tolist() == [1017]


def test_clean_crop_values_renames_area_223_to_turkey():
    raw = _raw_faostat_frame()
    raw.loc[3, 'Element Code'] = utilities.FAOSTAT_GROSS_PRODUCTION_VALUE_ELEMENT
    out = utilities.clean_faostat_values(raw, items=[1017], value_column='livestock_provision_gep', aggregate_areas=['World'])
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
    out = utilities.build_rental_rate_lookup(raw)
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

    out = utilities.apply_rental_rates(values, coefs, 'livestock_provision_gep').set_index(['area_code', 'year'])['livestock_provision_gep']
    assert np.isnan(out.loc[(1, 1960)])          # before the first decade: no rate in force
    assert out.loc[(1, 1961)] == 30.0            # 100 x 0.30
    assert out.loc[(1, 2015)] == 70.0            # 200 x 0.35, the 2011-2020 decade
    assert pd.isna(out.loc[(2, 2015)])           # country absent from the CWoN table


def test_attach_countries_does_not_repeat_a_split_country_and_converts_to_usd():
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
    # FAOSTAT's thousand USD becomes plain USD here, as in crop_provision.
    assert out['livestock_provision_gep'].tolist() == [30000.0, 15000.0, 40000.0]
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
    assert unmatched['livestock_provision_gep'] == 7000.0


def test_group_crops_then_group_countries_sum_to_the_same_total():
    item_rows = pd.DataFrame({
        'iso3_r250_id': [10, 10, 20, 10],
        'iso3_r250_label': ['AAA', 'AAA', 'BBB', 'AAA'],
        'year': [2019, 2019, 2019, 2018],
        'livestock_provision_gep': [30.0, 15.0, 40.0, 1.0],
    })
    by_country = utilities.sum_items_to_country_year(item_rows, 'livestock_provision_gep')
    per_country = by_country.set_index(['iso3_r250_id', 'year'])['livestock_provision_gep']
    assert per_country.loc[(10, 2019)] == 45.0        # two items summed
    assert per_country.loc[(20, 2019)] == 40.0
    assert per_country.loc[(10, 2018)] == 1.0

    by_year = utilities.sum_countries_to_year(by_country, 'livestock_provision_gep').set_index('year')['livestock_provision_gep']
    assert by_year.loc[2019] == 85.0
    assert by_year.loc[2018] == 1.0


def test_normalize_m49_codes_unquotes_casts_and_maps_successors():
    df = pd.DataFrame({'area_code_M49': ["'156", "'159", "'891", "'076"], 'value': [1, 2, 3, 4]})
    out = utilities.normalize_m49_codes(df)
    assert list(out['area_code_M49']) == [156, 156, 688, 76]
    assert list(df['area_code_M49']) == ["'156", "'159", "'891", "'076"]   # input untouched


def test_every_successor_maps_to_a_different_live_code():
    """A successor mapping that pointed at itself, or at another dissolved state, would leave
    production stranded."""
    for dissolved, successor in utilities.M49_SUCCESSORS.items():
        assert dissolved != successor
        assert successor not in utilities.M49_SUCCESSORS


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


def test_task_reader_cleans_the_faostat_bulk_file(tmp_path):
    """The bulk file is Latin-1, so an accented area name comes back mangled at any other
    encoding and the country would then miss its join."""
    path = str(tmp_path / 'Value_of_Production_E_All_Data.csv')
    raw = _raw_faostat_frame()
    raw.loc[0, 'Area'] = 'Côte'
    raw.to_csv(path, index=False, encoding='ISO-8859-1')

    out = lpt.read_crop_values(path, items=[1017], aggregate_areas=['World'])
    assert set(out['country']) == {'Côte'}
    assert out.set_index('year')['livestock_provision_gep'].loc[1961] == 1.0


def test_task_reader_reshapes_the_semicolon_delimited_coefficient_table(tmp_path):
    path = str(tmp_path / 'CWON2024_crop_coef.csv')
    pd.DataFrame({'Order': [1], 'FAO': [1.0], 'ISO3': ['AAA'], 'Country/territory': ['Aaaland'],
                  '1961-1970': [0.30], '2011-2020': [0.35]}).to_csv(path, sep=';', index=False)

    out = lpt.read_crop_coefs(path)
    assert out.set_index(['FAO', 'year'])['rental_rate'].loc[(1, 2011)] == 0.35


def test_es_config_row_hydrates_livestock_provision(tmp_path):
    from global_invest import utilities
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'livestock_provision', log=lambda *a: None)
    assert p.gep_base_year == 2019


def test_dashboard_categories_give_an_upper_bound_not_an_estimate():
    """The public dashboard serves six of the eight categories, and both absent ones belong to
    the denominator, so the share it yields is too high. The flag must say so."""
    full = pd.DataFrame([{'iso3_r250_id': 1, 'iso3_r250_label': 'AAA',
                          'By-products': 0.0, 'Crop residues': 2.0, 'Fodder crop': 0.0,
                          'Grass and leaves': 4.0, 'Grains': 3.0, 'Oil seed cakes': 1.0,
                          'Other edible': 5.0, 'Other non-edible': 5.0}])
    dashboard = full.drop(columns=['Other edible', 'Other non-edible'])

    complete = lp.feed_lambda_by_country(full).iloc[0]
    bounded = lp.feed_lambda_by_country(dashboard).iloc[0]

    assert complete['lambda'] == 0.3                     # 6 of 20
    assert bounded['lambda'] == 0.6                      # 6 of 10, the two others absent
    assert bounded['lambda'] > complete['lambda']        # biased upward, hence a bound
    assert bool(complete['lambda_is_upper_bound']) is False
    assert bool(bounded['lambda_is_upper_bound']) is True


def test_missing_an_ecosystem_category_raises_rather_than_understating():
    """An absent numerator category cannot be bounded in either direction."""
    df = pd.DataFrame([{'iso3_r250_id': 1, 'iso3_r250_label': 'AAA', 'By-products': 1.0,
                        'Crop residues': 1.0, 'Grains': 1.0}])       # no Fodder crop, no Grass
    with pytest.raises(ValueError, match='numerator'):
        lp.feed_lambda_by_country(df)


def test_the_two_species_layouts_together_cover_all_eight_categories():
    """The dashboard shows a different subset per species. Ruminants carry fodder crop and no
    "other" categories; chickens and pigs carry the two "other" categories and no fodder crop.
    Neither alone is the full set, and their union is, which is why a harvest across species
    yields an estimate rather than a bound."""
    ruminant = set(lp.GLEAM_RUMINANT_FEED_COLS)
    monogastric = set(lp.GLEAM_MONOGASTRIC_FEED_COLS)
    assert ruminant | monogastric == set(lp.GLEAM_TOTAL_FEED_COLS)
    assert ruminant != set(lp.GLEAM_TOTAL_FEED_COLS)
    assert monogastric != set(lp.GLEAM_TOTAL_FEED_COLS)
    assert {'Other edible', 'Other non-edible'} <= monogastric
    assert 'Fodder crop' in ruminant and 'Fodder crop' not in monogastric


def test_the_harvested_dashboard_table_gives_an_estimate_not_a_bound():
    """The file harvested from the dashboard carries all eight categories, so the share it
    yields is flagged as an estimate."""
    # Located the way a run locates it, from the base-data root rather than from one machine's
    # layout. A literal path passes here and silently skips everywhere else, which reads as a
    # green test on a machine where nothing was checked.
    import os
    import hazelbean as hb
    roots = [getattr(hb.config, 'BASE_DATA_DIR', None),
             os.path.join(os.path.expanduser('~'), 'Files', 'base_data')]
    reference = os.path.join('global_invest', 'livestock_provision', 'gleam3_dmi_dashboard.psv')
    path = next((os.path.join(r, reference) for r in roots
                 if r and os.path.exists(os.path.join(r, reference))), None)
    if path is None:
        import pytest
        pytest.skip('the harvested dashboard table is not in this machine\'s base data')
    raw = pd.read_csv(path, sep='|')
    assert set(lp.GLEAM_TOTAL_FEED_COLS) <= set(raw.columns)
    countries = pd.DataFrame({'iso3_r250_label': ['FRA'], 'iso3_r250_id': [250]})
    cleaned, _ = lp.clean_gleam_dashboard_intake(raw, countries)
    out = lp.feed_lambda_by_country(cleaned)
    assert bool(out['lambda_is_upper_bound'].iloc[0]) is False
    assert 0.0 <= out['lambda'].iloc[0] <= 1.0


def test_feed_share_attribution_runs_beside_the_rental_rate_not_instead_of_it():
    # The account attributes livestock value with the CWoN land rental rate, which belongs to the
    # crop method and stands in for the share of feed ecosystems provided. Both columns must come
    # out of one run, because choosing between them is the group's decision and they differ a lot.
    df_country_year = pd.DataFrame({
        'iso3_r250_id': [1, 2, 3],
        'year': [2019, 2019, 2019],
        'gross_production_value': [1000.0, 2000.0, 500.0],
        'livestock_provision_gep': [120.0, 240.0, 60.0],       # the rental-rate attribution
    })
    df_lambda = pd.DataFrame({
        'iso3_r250_id': [1, 2], 'lambda': [0.9, 0.5], 'lambda_is_upper_bound': [False, False]})

    out = lp.feed_share_gep(df_country_year, df_lambda).set_index('iso3_r250_id')

    assert out.loc[1, 'livestock_provision_gep_feed_share'] == pytest.approx(900.0)
    assert out.loc[2, 'livestock_provision_gep_feed_share'] == pytest.approx(1000.0)
    # The rental-rate column is untouched, so a reader can see both attributions side by side.
    assert out.loc[1, 'livestock_provision_gep'] == pytest.approx(120.0)
    # A country GLEAM does not model gets no feed-share value rather than a zero one, which would
    # say ecosystems contributed nothing to its livestock.
    assert pd.isna(out.loc[3, 'livestock_provision_gep_feed_share'])
    assert out.loc[3, 'livestock_provision_gep'] == pytest.approx(60.0)


def test_both_value_columns_leave_the_country_join_in_dollars():
    # FAOSTAT reports value in thousands of dollars and attach_countries converts. The conversion
    # reached the attributed column only, so the gross value stayed in thousands and the
    # feed-share attribution downstream came out a thousand times too small. Both must convert.
    df_crop_value = pd.DataFrame({
        'area_code_M49': [10], 'area_code': [1], 'country': ['Aaaland'], 'year': [2019],
        'livestock_provision_gep': [1_000.0],       # thousands of USD, after the rental rate
        'gross_production_value': [5_000.0],        # thousands of USD, before it
    })
    out = lp.attach_countries(df_crop_value, COUNTRIES)
    assert out['livestock_provision_gep'].iloc[0] == pytest.approx(1_000_000.0)
    assert out['gross_production_value'].iloc[0] == pytest.approx(5_000_000.0)

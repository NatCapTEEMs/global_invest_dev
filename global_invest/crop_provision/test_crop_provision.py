"""Unit tests for crop_provision's valuation steps.

Every function the task calls is pinned on a hand-built table small enough that the expected
number is written out in the assertion: the FAOSTAT clean-up, the decade rental-rate lookup,
the as-of attribution, the country join with its thousand-USD conversion, and the two groupings.
The task module's two readers are pinned on files written into tmp_path.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from global_invest import utilities
from global_invest.crop_provision import crop_provision_functions as cp
from global_invest.crop_provision import crop_provision_tasks as cpt

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
    """Four FAOSTAT rows: one that survives, one unrequested crop, one aggregate area, one whose
    element is not gross production value. The year columns are the file's full Y1961..Y2022 span
    with a flag column beside each, because the melt reads all of them."""
    years = range(cp.FAOSTAT_FIRST_YEAR, cp.FAOSTAT_LAST_YEAR + 1)
    frame = pd.DataFrame({
        'Area Code': [1, 1, 2, 223],
        'Area Code (M49)': ["'010", "'010", "'020", "'223"],
        'Area': ['Aaaland', 'Aaaland', 'World', 'Turkiye'],
        'Item Code': [100, 200, 100, 100],
        'Item': ['Wheat', 'Rye', 'Wheat', 'Wheat'],
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


def test_clean_crop_values_keeps_gross_production_value_and_melts_the_years():
    out = cp.clean_crop_values(_raw_faostat_frame(), items=['Wheat'], aggregate_areas=['World'])

    # 'Rye' is not requested, 'World' is an aggregate area, and area 223's row is element 152.
    assert set(out['crop']) == {'Wheat'}
    assert set(out['country']) == {'Aaaland'}
    assert not [c for c in out.columns if c.endswith('F')]      # flag columns dropped
    # One surviving source row, melted over the full year span.
    assert len(out) == cp.FAOSTAT_LAST_YEAR - cp.FAOSTAT_FIRST_YEAR + 1
    values = out.set_index('year')['crop_provision_gep']
    assert values.loc[1961] == 1.0
    assert values.loc[2022] == 5.0
    assert values.loc[1990] == 0.0
    assert out['year'].dtype.kind == 'i'
    assert out['area_code'].dtype.kind == 'i'


def test_clean_crop_values_renames_area_223_to_turkey():
    raw = _raw_faostat_frame()
    raw.loc[3, 'Element Code'] = cp.FAOSTAT_GROSS_PRODUCTION_VALUE_ELEMENT
    out = cp.clean_crop_values(raw, items=['Wheat'], aggregate_areas=['World'])
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
        'crop_provision_gep': [100.0, 100.0, 200.0, 500.0],
    })
    coefs = pd.DataFrame({'FAO': [1, 1], 'year': [1961, 2011], 'rental_rate': [0.30, 0.35]})

    out = cp.merge_crop_with_coefs(values, coefs).set_index(['area_code', 'year'])['crop_provision_gep']
    assert np.isnan(out.loc[(1, 1960)])          # before the first decade: no rate in force
    assert out.loc[(1, 1961)] == 30.0            # 100 x 0.30
    assert out.loc[(1, 2015)] == 70.0            # 200 x 0.35, the 2011-2020 decade
    assert pd.isna(out.loc[(2, 2015)])           # country absent from the CWoN table


def test_attach_countries_in_usd_converts_once_and_does_not_repeat_a_split_country():
    crops = pd.DataFrame({
        'area_code_M49': [10, 10, 20],
        'area_code': [1, 1, 2],
        'country': ['Aaaland', 'Aaaland', 'Bbbland'],
        'crop_code': [100, 200, 100],
        'crop': ['Wheat', 'Rye', 'Wheat'],
        'year': [2019, 2019, 2019],
        'rental_rate': [0.3, 0.3, 0.4],
        'crop_provision_gep': [30.0, 15.0, 40.0],
    })
    out = cp.attach_countries_in_usd(crops, COUNTRIES)

    # AAA is split across two r264 rows; the join must still produce three rows, not five.
    assert len(out) == 3
    assert out['crop_provision_gep'].tolist() == [30000.0, 15000.0, 40000.0]
    assert out.set_index('crop_code').loc[100, 'iso3_r250_label'].tolist() == ['AAA', 'BBB']


def test_attach_countries_in_usd_keeps_a_row_whose_code_the_correspondence_lacks():
    crops = pd.DataFrame({
        'area_code_M49': [10, 999],
        'area_code': [1, 9],
        'country': ['Aaaland', 'Nowhere'],
        'crop_code': [100, 100],
        'crop': ['Wheat', 'Wheat'],
        'year': [2019, 2019],
        'rental_rate': [0.3, 0.3],
        'crop_provision_gep': [30.0, 7.0],
    })
    out = cp.attach_countries_in_usd(crops, COUNTRIES)
    assert len(out) == 2
    unmatched = out[out['area_code_M49'] == 999].iloc[0]
    assert pd.isna(unmatched['iso3_r250_label'])
    assert unmatched['crop_provision_gep'] == 7000.0


def test_group_crops_then_group_countries_sum_to_the_same_total():
    crop_rows = pd.DataFrame({
        'iso3_r250_id': [10, 10, 20, 10],
        'iso3_r250_label': ['AAA', 'AAA', 'BBB', 'AAA'],
        'year': [2019, 2019, 2019, 2018],
        'crop_provision_gep': [30000.0, 15000.0, 40000.0, 1000.0],
    })
    by_country = cp.group_crops(crop_rows)
    per_country = by_country.set_index(['iso3_r250_id', 'year'])['crop_provision_gep']
    assert per_country.loc[(10, 2019)] == 45000.0        # two crops summed
    assert per_country.loc[(20, 2019)] == 40000.0
    assert per_country.loc[(10, 2018)] == 1000.0

    by_year = cp.group_countries(by_country).set_index('year')['crop_provision_gep']
    assert by_year.loc[2019] == 85000.0
    assert by_year.loc[2018] == 1000.0


def test_normalize_m49_codes_unquotes_casts_and_maps_successors():
    """FAOSTAT ships quoted codes and keeps dissolved states under their own code: both are
    resolved so every row joins to a current country."""
    df = pd.DataFrame({'area_code_M49': ["'156", "'159", "'891", "'076"], 'value': [1, 2, 3, 4]})
    out = utilities.normalize_m49_codes(df)
    assert list(out['area_code_M49']) == [156, 156, 688, 76]
    assert list(out['value']) == [1, 2, 3, 4]
    assert list(df['area_code_M49']) == ["'156", "'159", "'891", "'076"]   # input untouched


def test_every_successor_maps_to_a_different_live_code():
    """A successor mapping that pointed at itself, or at another dissolved state, would leave
    production stranded."""
    for dissolved, successor in cp.M49_SUCCESSORS.items():
        assert dissolved != successor
        assert successor not in cp.M49_SUCCESSORS


def test_the_faostat_unit_factor_is_the_thousand_usd_conversion():
    assert cp.FAOSTAT_THOUSAND_USD == 1000.0
    assert cp.FAOSTAT_VALUE_UNIT == '1000 USD'


def test_task_reader_cleans_the_faostat_bulk_file(tmp_path):
    """The bulk file is Latin-1, so an accented area name comes back mangled at any other
    encoding and the country would then miss its join."""
    path = str(tmp_path / 'Value_of_Production_E_All_Data.csv')
    raw = _raw_faostat_frame()
    raw.loc[0, 'Area'] = 'Côte'
    raw.to_csv(path, index=False, encoding='ISO-8859-1')

    out = cpt.read_crop_values(path, items=['Wheat'], aggregate_areas=['World'])
    assert set(out['country']) == {'Côte'}
    assert out.set_index('year')['crop_provision_gep'].loc[1961] == 1.0


def test_task_reader_reshapes_the_semicolon_delimited_coefficient_table(tmp_path):
    path = str(tmp_path / 'CWON2024_crop_coef.csv')
    pd.DataFrame({'Order': [1], 'FAO': [1.0], 'ISO3': ['AAA'], 'Country/territory': ['Aaaland'],
                  '1961-1970': [0.30], '2011-2020': [0.35]}).to_csv(path, sep=';', index=False)

    out = cpt.read_crop_coefs(path)
    assert out.set_index(['FAO', 'year'])['rental_rate'].loc[(1, 2011)] == 0.35


def test_es_config_row_hydrates_crop_provision(tmp_path):
    from global_invest import utilities
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'crop_provision', log=lambda *a: None)
    assert p.sheet_label == 'crop_provision'

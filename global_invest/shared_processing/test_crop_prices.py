"""Tests of the shared FAO price path: its filters, its fallback, its wiring and where it writes."""
import ast
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from global_invest.shared_processing import crop_prices_functions as cpf
from global_invest.shared_processing import crop_prices_tasks as cpt
from global_invest import utilities
from global_invest.pollination import pollination_functions as pf
from global_invest.pollination import pollination_initialize
from global_invest.pollination import pollination_tasks as pt


def test_annual_values_fall_back_country_subregion_region_world_in_that_order():
    """The adopted rule (the author's crop_benefits hierarchy): the country's own price for the
    year, else the same crop's subregion median, else region, else world — and never another
    YEAR's price. Value is production tonnes times USD per tonne."""
    prod = pd.DataFrame({
        'element': ['Production'] * 4,
        'year': [2019] * 4,
        'area_code_m49': ['004', '008', '012', '016'],
        'subregion_fao': ['S1', 'S1', 'S2', 'S3'],
        'region_fao': ['R1', 'R1', 'R1', 'R2'],
        'item_code_fao': [515] * 4,
        'value': [100.0, 50.0, 70.0, 10.0]})
    price_country = pd.DataFrame({
        'year': [2019, 2017], 'area_code_m49': ['004', '008'], 'item_code_fao': [515, 515],
        'price_usd_tonne': [10.0, 999.0]})
    price_subregion = pd.DataFrame({'year': [2019], 'subregion_fao': ['S1'], 'item_code_fao': [515],
                                    'subreg_price_usd_tonne': [20.0]})
    price_region = pd.DataFrame({'year': [2019], 'region_fao': ['R1'], 'item_code_fao': [515],
                                 'reg_price_usd_tonne': [30.0]})
    price_world = pd.DataFrame({'year': [2019], 'item_code_fao': [515],
                                'world_price_usd_tonne': [40.0]})
    merged = cpt._merge_production_value(prod, price_country, price_subregion, price_region, price_world)
    keyed = merged.set_index('area_code_m49')
    assert keyed.at['004', 'price_source_agg_level'] == 'country'
    assert keyed.at['004', 'value_usd'] == pytest.approx(100.0 * 10.0)
    # 008 has only a 2017 own price: that year is NOT used; its subregion median prices it
    assert keyed.at['008', 'price_source_agg_level'] == 'subregion'
    assert keyed.at['008', 'price_usd_tonne'] == pytest.approx(20.0)
    assert keyed.at['012', 'price_source_agg_level'] == 'region'
    assert keyed.at['016', 'price_source_agg_level'] == 'world'
    assert keyed.at['016', 'value_usd'] == pytest.approx(10.0 * 40.0)
    assert 'price_donor_year' not in merged.columns


def test_a_country_factor_within_tolerance_reads_as_near_equal_not_as_a_factor():
    """The near-equal test must be parenthesised as the author's is: `&` binds before `<=`, so an
    unparenthesised mask compares the whole chain with the tolerance. A country factor within 1%
    would then read as a full factor (and fall to the later currency check), and a row with no
    factor at all would be labelled near-equal."""
    nan = np.nan
    pp_wide = pd.DataFrame({
        'country': ['A', 'A', 'B', 'B', 'C', 'D'],
        'Item': ['x'] * 6,
        'year': [2000, 2001, 2000, 2001, 2001, 2001],
        'slc_tonne': [100.0, 200.0, 100.0, 200.0, 50.0, nan],
        'lcu_tonne': [100.5, nan, 150.0, nan, nan, nan],
        'price_index': [nan] * 6,
    })
    out = cpf._reconstruct_slc_lcu(pp_wide, SimpleNamespace(tol_near_equal_lcu_slc=0.01))
    row = out.set_index(['country', 'year'])
    assert row.loc[('A', 2001), 'lcu_source'] == 'slc_filled_country_factor_near_equal'
    assert row.loc[('A', 2001), 'lcu_filled'] == pytest.approx(200.0)
    assert row.loc[('B', 2001), 'lcu_source'] == 'slc_filled_country_factor'
    assert row.loc[('B', 2001), 'lcu_filled'] == pytest.approx(300.0)
    assert row.loc[('C', 2001), 'lcu_source'] == 'slc_filled_assume_equal'
    assert row.loc[('D', 2001), 'lcu_source'] == 'missing'


# The FAO price path's quality checks on one panel, with the service owner's thresholds.
_PRICE_QC = dict(anchor_max_year_dist=10, anchor_max_times_off=10.0, global_median_max_times_off=10.0,
                 usd_copy_of_lcu_tol=0.001, usd_copy_of_lcu_min_fx=1.1, obs_global_median_max_times_off=20.0,
                 obs_global_median_min_countries=10, fao_start_year=1993, qc_min_overlap=50,
                 qc_bad_median_times_off=3.0, qc_bad_share_over_3x=0.5)


def _qc(**changes):
    return SimpleNamespace(**{**_PRICE_QC, **changes})


def _price_rows(rows, item='x'):
    """pp3 and fx from (iso3, year, observed USD, reported LCU, LCU per USD) rows."""
    frame = pd.DataFrame(rows, columns=['iso3', 'year', 'usd_tonne_obs', 'lcu_tonne', 'lcu_per_usd'], dtype=object)
    for column in ['usd_tonne_obs', 'lcu_tonne', 'lcu_per_usd']:
        frame[column] = pd.to_numeric(frame[column])
    frame['year'] = frame['year'].astype(int)
    frame['Item'] = item
    frame['country'] = frame['iso3']
    frame['region_fao'] = 'R'
    frame['subregion_fao'] = 'S'
    frame['lcu_filled'] = frame['lcu_tonne']
    frame['lcu_source'] = np.where(frame['lcu_tonne'].notna(), 'observed', 'missing')
    fx = frame[['iso3', 'year', 'lcu_per_usd']].drop_duplicates()
    return frame.drop(columns='lcu_per_usd'), fx


def _row(out, iso3, year=2019):
    return out[(out['iso3'] == iso3) & (out['year'] == year)].iloc[0]


def test_usd_copy_of_local_currency_goes_missing_only_when_the_exchange_rate_is_not_about_one():
    pp3, fx = _price_rows([
        ('COP', 2019, 500.0, 500.0, 100.0),      # a copy: equal, the rate far from 1
        ('NEA', 2019, 500.4, 500.0, 100.0),      # equal within the 0.1 percent tolerance
        ('OFF', 2019, 510.0, 500.0, 100.0),      # not equal
        ('PEG', 2019, 500.0, 500.0, 1.05),       # equal, the rate within 1.1 of 1, so it can be real
        ('DIV', 2019, 500.0, 500.0, 1 / 100.0)])  # a rate far below 1 counts as far from 1
    out = cpf._build_usd_with_qc(pp3, fx, _qc())
    for iso3 in ['COP', 'NEA', 'DIV']:
        row = _row(out, iso3)
        assert row['usd_copy_of_lcu'] and row['usd_source'] == 'invalid_usd_copy_of_lcu', iso3
        assert np.isnan(row['usd_filled']) and np.isnan(row['usd_tonne_obs']), iso3
    for iso3, price in [('OFF', 510.0), ('PEG', 500.0)]:
        row = _row(out, iso3)
        assert not row['usd_copy_of_lcu'] and row['usd_source'] == 'observed' and row['usd_filled'] == price, iso3


def test_a_copy_is_no_anchor_for_a_later_fx_filled_price():
    """The filled price sits 100 times from the copy. With the filter off the copy is its anchor and
    drops it. With the filter on no anchor is left, so the FX price stands."""
    pp3, fx = _price_rows([('AAA', 2018, 1000.0, 1000.0, 100.0), ('AAA', 2019, np.nan, 1000.0, 100.0)])
    on = cpf._build_usd_with_qc(pp3, fx, _qc())
    assert _row(on, 'AAA')['usd_source'] == 'filled' and _row(on, 'AAA')['usd_filled'] == pytest.approx(10.0)
    off = cpf._build_usd_with_qc(pp3, fx, _qc(usd_copy_of_lcu_tol=0.0))
    assert _row(off, 'AAA')['usd_source'] == 'implausible_vs_closest_observed_10x'


def test_a_copy_leaves_the_global_median_the_filled_prices_are_checked_against():
    """Without the copy the item-year median is 10, 100 times below the filled price. With the copy in,
    the median is 2,505 and the filled price passes."""
    pp3, fx = _price_rows([('COP', 2019, 5000.0, 5000.0, 100.0), ('OBS', 2019, 10.0, np.nan, 1.0),
                           ('FIL', 2019, np.nan, 1000.0, 1.0)])
    on = cpf._build_usd_with_qc(pp3, fx, _qc())
    assert _row(on, 'FIL')['usd_source'] == 'implausible_vs_global_median_10x'
    assert _row(on, 'OBS')['global_median_usd'] == 10.0 and _row(on, 'OBS')['n_obs'] == 1
    off = cpf._build_usd_with_qc(pp3, fx, _qc(usd_copy_of_lcu_tol=0.0))
    assert _row(off, 'FIL')['usd_source'] == 'filled' and _row(off, 'OBS')['global_median_usd'] == 2505.0


def test_a_copy_leaves_the_fx_country_check():
    """A country whose only overlap is two copies looks 100 times off with the filter off, which
    invalidates all its filled prices. With the filter on it has no overlap to judge."""
    pp3, fx = _price_rows([('AAA', 2017, 1000.0, 1000.0, 100.0), ('AAA', 2018, 1000.0, 1000.0, 100.0),
                           ('AAA', 2019, np.nan, 1000.0, 100.0)])
    no_anchor_check = dict(anchor_max_times_off=1e9, qc_min_overlap=2)
    on = cpf._build_usd_with_qc(pp3, fx, _qc(**no_anchor_check))
    assert _row(on, 'AAA')['usd_source'] == 'filled'
    off = cpf._build_usd_with_qc(pp3, fx, _qc(usd_copy_of_lcu_tol=0.0, **no_anchor_check))
    assert _row(off, 'AAA')['usd_source'] == 'invalid_fx_country'


def _observed_panel(extreme, n_at_median=9):
    rows = [('M%02d' % i, 2019, 100.0, np.nan, 1.0) for i in range(n_at_median)]
    return rows + [('EXT', 2019, extreme, np.nan, 1.0)]


@pytest.mark.parametrize('extreme, dropped', [(2000.0, False), (2000.5, True), (5.0, False), (4.99, True)])
def test_observed_price_is_dropped_strictly_beyond_twenty_times_the_median_in_both_directions(extreme, dropped):
    out = cpf._build_usd_with_qc(*_price_rows(_observed_panel(extreme)), _qc())
    row = _row(out, 'EXT')
    assert row['usd_tonne_obs'] == extreme, 'the observed price stays for reference'
    if dropped:
        assert row['usd_source'] == 'implausible_observed_vs_global_median_20x' and np.isnan(row['usd_filled'])
    else:
        assert row['usd_source'] == 'observed' and row['usd_filled'] == extreme


def test_observed_price_check_needs_ten_reporting_countries_and_a_copy_does_not_count():
    # Nine reporters: the extreme stands.
    assert _row(cpf._build_usd_with_qc(*_price_rows(_observed_panel(9000.0, n_at_median=8)), _qc()),
                'EXT')['usd_source'] == 'observed'
    # A tenth row that is a copy leaves nine reporters with the filter on, and makes ten with it off.
    rows = _observed_panel(9000.0, n_at_median=8) + [('COP', 2019, 100.0, 100.0, 100.0)]
    assert _row(cpf._build_usd_with_qc(*_price_rows(rows), _qc()), 'EXT')['usd_source'] == 'observed'
    assert (_row(cpf._build_usd_with_qc(*_price_rows(rows), _qc(usd_copy_of_lcu_tol=0.0)), 'EXT')['usd_source']
            == 'implausible_observed_vs_global_median_20x')


def test_a_dropped_observed_price_stays_an_anchor_and_stays_in_the_median():
    """Unlike a copy, an implausible observed price keeps its place in usd_tonne_obs."""
    rows = _observed_panel(9000.0) + [('EXT', 2020, np.nan, 5.0, 1.0)]
    out = cpf._build_usd_with_qc(*_price_rows(rows), _qc())
    assert _row(out, 'EXT', 2020)['usd_source'] == 'implausible_vs_closest_observed_10x'
    assert _row(out, 'EXT')['n_obs'] == 10


def test_a_dropped_price_falls_back_to_the_geographic_medians_without_itself():
    pp_usd = cpf._build_usd_with_qc(*_price_rows(_observed_panel(9000.0)), _qc())
    prices = pp_usd.rename(columns={'usd_filled': 'price_usd_tonne', 'Item': 'item_fao'})
    prices['area_code_m49'] = prices['iso3']
    prices['item_code_fao'] = 1
    production = pd.DataFrame({'element': 'Production', 'year': 2019, 'area_code_m49': ['EXT', 'M00'], 'item_code_fao': 1,
                               'subregion_fao': 'S', 'region_fao': 'R', 'value': 2.0})
    out = cpt._merge_production_value(production, *cpf._compute_annual_prices(prices, None)).set_index('area_code_m49')
    assert out.loc['EXT', 'price_source_agg_level'] == 'subregion' and out.loc['EXT', 'price_usd_tonne'] == 100.0
    assert out.loc['EXT', 'value_usd'] == 200.0
    assert out.loc['M00', 'price_source_agg_level'] == 'country'


def test_price_panel_saves_the_copy_flag(tmp_path):
    pp_usd = cpf._build_usd_with_qc(*_price_rows([('COP', 2019, 500.0, 500.0, 100.0)]), _qc())
    pp_usd = pp_usd.assign(**{'Item Code': 1, 'Area Code (M49)': '001', 'price_index': np.nan, 'slc_filled': np.nan,
                              'slc_source': 'missing', 'fx_source': 'wb'})
    cpt._save_price_outputs(pp_usd, str(tmp_path), SimpleNamespace(fao_start_year=2019, fao_end_year=2019))
    saved = pd.read_csv(tmp_path / 'fao_prices_2019_2019.csv')
    assert saved['usd_copy_of_lcu'].tolist() == [True] and saved['usd_source'].tolist() == ['invalid_usd_copy_of_lcu']


def test_price_filter_thresholds_are_configuration():
    rows = pd.read_csv(os.path.join(os.path.dirname(cpf.__file__), '..', 'input_template', 'es_parameters.csv'),
                       names=['service', 'parameter', 'value'], usecols=[0, 1, 2], engine='python',
                       on_bad_lines='skip').set_index('parameter')['value']
    assert float(rows['pollination_price_usd_copy_of_lcu_tol']) == 0.001
    assert float(rows['pollination_price_usd_copy_of_lcu_min_fx']) == 1.1
    assert float(rows['pollination_price_obs_global_median_max_times_off']) == 20.0
    assert int(rows['pollination_price_obs_global_median_min_countries']) == 10


def test_every_price_threshold_matches_the_parameters_file_with_ten_and_twenty_times_apart():
    """A price filled from an exchange rate is checked at 10 times, an observed USD price at 20 times."""
    rows = pd.read_csv(os.path.join(os.path.dirname(cpf.__file__), '..', 'input_template', 'es_parameters.csv'),
                       names=['service', 'parameter', 'value'], usecols=[0, 1, 2], engine='python',
                       on_bad_lines='skip').set_index('parameter')['value']
    assert float(rows['pollination_price_anchor_max_times_off']) == 10.0
    assert float(rows['pollination_price_global_median_max_times_off']) == 10.0
    assert float(rows['pollination_price_obs_global_median_max_times_off']) == 20.0
    assert int(rows['pollination_price_anchor_max_year_dist']) == 10
    assert float(rows['pollination_price_tol_near_equal_lcu_slc']) == 0.01
    assert int(rows['pollination_price_qc_min_overlap']) == 50
    assert float(rows['pollination_price_qc_bad_median_times_off']) == 3.0
    assert float(rows['pollination_price_qc_bad_share_over_3x']) == 0.5


def _top_level_definitions(path):
    tree = ast.parse(open(path, encoding='utf-8').read())
    names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    return names | {target.id for node in tree.body if isinstance(node, ast.Assign)
                    for target in node.targets if isinstance(target, ast.Name)}


_TASK_NAMES = {'FAO_MEDIAN_PRICES_REF_PATH', 'fao_median_prices', 'save_parquet', '_add_geographic_and_classification_data', 'run_fao_production',
               'run_fao_prices', 'fao_table_name', '_read_and_filter_fao_production', '_save_production_outputs', '_read_fao_prices',
               'world_bank_fx', '_add_iso3_and_fx', '_save_price_outputs', '_load_production_and_prices', '_compute_median_prices',
               '_merge_production_value', 'run_fao_values', 'load_m49_iso3', 'load_fao_classification', 'fao_price_settings',
               'load_excluded_item_codes', 'load_fx_inherit_iso3', 'load_imf_fao_country_names', 'save_csv', 'median_prices_ref_path'}
_FUNCTION_NAMES = {'FaoPriceSettings', '_PRODUCTION_ELEMENTS_KEEP', '_PRICE_ELEMENTS_KEEP', '_convert_yield_units', '_recode_country',
                   '_log_df', '_reshape_prices', '_reconstruct_slc_lcu', '_build_usd_with_qc', '_compute_annual_prices'}


def test_the_price_path_has_one_definition_and_pollination_imports_that_one():
    assert _TASK_NAMES <= _top_level_definitions(cpt.__file__)
    assert _FUNCTION_NAMES <= _top_level_definitions(cpf.__file__)
    assert not (_TASK_NAMES | _FUNCTION_NAMES) & (_top_level_definitions(pt.__file__) | _top_level_definitions(pf.__file__))
    for name in ('FAO_MEDIAN_PRICES_REF_PATH', 'fao_median_prices', 'fao_price_settings', 'median_prices_ref_path',
                 'run_fao_production', 'run_fao_prices', 'run_fao_values'):
        assert getattr(pt, name) is getattr(cpt, name), name


def test_the_tables_stay_where_their_readers_look():
    """Crop provision, pollination and the base-data copies all find these tables by these relative paths."""
    assert cpt.FAO_MEDIAN_PRICES_REF_PATH == os.path.join('global_invest', 'pollination', 'fao', 'median_prices')
    assert cpt.median_prices_ref_path([2019]) == os.path.join(
        'global_invest', 'pollination', 'fao', 'median_prices', 'price_median_usd_tonne_2019_2019.parquet')
    assert cpt.median_prices_ref_path([2020, 2018]) == os.path.join(
        'global_invest', 'pollination', 'fao', 'median_prices', 'price_median_usd_tonne_2018_2020.parquet')
    full_span = SimpleNamespace(fao_start_year=1993, fao_end_year=2024)
    assert cpt.fao_table_name('values', full_span, 'csv') == 'fao_values_1993_2024.csv'
    rows = pd.read_csv(os.path.join(os.path.dirname(cpf.__file__), '..', 'input_template', 'es_parameters.csv'),
                       names=['service', 'parameter', 'value'], usecols=[0, 1, 2], engine='python',
                       on_bad_lines='skip').set_index('parameter')['value']
    assert rows['crop_provision_fao_values_table_path'] == 'global_invest/pollination/fao/fao_values_1993_2024.csv'


def test_fao_median_prices_runs_the_three_steps_into_base_data_once(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cpt, 'publish_inputs', lambda p: None)
    monkeypatch.setattr(cpt, 'fao_price_settings', lambda p, output_dir: calls.append(('settings', output_dir)) or 'the settings')
    for name in ('run_fao_production', 'run_fao_prices', 'run_fao_values'):
        monkeypatch.setattr(cpt, name, lambda settings, name=name: calls.append((name, settings)))
    p = SimpleNamespace(run_this=1, base_data_dir=str(tmp_path),
                        get_path=lambda ref, possible_dirs, raise_error_if_fail: os.path.join(possible_dirs[0], ref))
    target = tmp_path / 'global_invest' / 'pollination' / 'fao' / 'median_prices'
    assert cpt.fao_median_prices(p) is True
    assert p.fao_median_prices_dir == str(target)
    assert calls == [('settings', str(target.parent)), ('run_fao_production', 'the settings'),
                     ('run_fao_prices', 'the settings'), ('run_fao_values', 'the settings')]
    calls.clear()
    (target / 'price_median_usd_tonne_2019_2019.parquet').write_bytes(b'')
    assert cpt.fao_median_prices(p) is True
    assert calls == []


def test_the_pollination_tree_registers_the_shared_task():
    added = []

    def add_task(function, **options):
        added.append((function, options))
        return SimpleNamespace()
    pollination_initialize.build_gep_service_calculation_task_tree(SimpleNamespace(add_task=add_task))
    assert (cpt.fao_median_prices, {'skip_existing': 1}) in added


def test_the_shared_task_publishes_what_pollination_publishes(monkeypatch):
    """Both read the same rows, so the task hydrates the same whichever module a run reached first."""
    def calls_of(module):
        calls = []
        monkeypatch.setattr(utilities, 'hydrate_es_config', lambda p, service, log: calls.append(('config', service)))
        monkeypatch.setattr(utilities, 'hydrate_es_parameters', lambda p, service, log: calls.append(('parameters', service)))
        monkeypatch.setattr(utilities, 'initialize_country_paths', lambda p: calls.append(('countries',)))
        p = SimpleNamespace()
        assert module.publish_inputs(p) is p
        assert p.results == {}
        return calls
    assert calls_of(cpt) == calls_of(pt) == [('config', 'pollination'), ('parameters', 'pollination'), ('countries',)]


def test_price_settings_carry_each_parameter_to_its_own_field(tmp_path):
    """Every threshold gets its own value, so two swapped fields cannot pass."""
    (tmp_path / 'excluded.csv').write_text('item_code_fao\n20\n')
    (tmp_path / 'inherit.csv').write_text('src_label,dst_label\nXKX,SRB\n')
    (tmp_path / 'names.csv').write_text('imf_name,fao_name\nTurkiye,Turkey\n')
    p = SimpleNamespace(
        get_path=str,
        pollination_crosswalk_m49_iso3_path='crosswalk', pollination_fao_classification_path='classification',
        pollination_crosswalk_fao_cropgrids_path='cropgrids', pollination_fao_production_path='production',
        pollination_fao_prices_path='prices', pollination_fx_path='fx', pollination_price_years=[2019],
        pollination_fao_start_year=1993, pollination_fao_end_year=2024,
        pollination_excluded_item_codes_path=str(tmp_path / 'excluded.csv'), pollination_fx_inherit_path=str(tmp_path / 'inherit.csv'),
        pollination_imf_fao_names_path=str(tmp_path / 'names.csv'), pollination_price_tol_near_equal_lcu_slc=0.5,
        pollination_price_anchor_max_year_dist=7, pollination_price_anchor_max_times_off=1.5,
        pollination_price_global_median_max_times_off=2.5, pollination_price_usd_copy_of_lcu_tol=3.5,
        pollination_price_usd_copy_of_lcu_min_fx=4.5, pollination_price_obs_global_median_max_times_off=5.5,
        pollination_price_obs_global_median_min_countries=8, pollination_price_qc_min_overlap=9,
        pollination_price_qc_bad_median_times_off=6.5, pollination_price_qc_bad_share_over_3x=7.5)
    s = cpt.fao_price_settings(p, 'the output')
    assert (s.crosswalk_m49_iso3_path, s.fao_classification_path, s.crosswalk_fao_cropgrids_path) == ('crosswalk', 'classification', 'cropgrids')
    assert (s.fao_production_bulk_path, s.fao_prices_bulk_path, s.fx_lcu_per_usd_path, s.output_dir) == ('production', 'prices', 'fx', 'the output')
    assert (s.price_years, s.fao_start_year, s.fao_end_year) == ((2019,), 1993, 2024)
    assert (s.excluded_item_codes, s.fx_inherit_iso3, s.imf_fao_names) == ({20}, {'XKX': 'SRB'}, {'Turkiye': 'Turkey'})
    assert (s.tol_near_equal_lcu_slc, s.anchor_max_year_dist, s.anchor_max_times_off, s.global_median_max_times_off) == (0.5, 7, 1.5, 2.5)
    assert (s.usd_copy_of_lcu_tol, s.usd_copy_of_lcu_min_fx, s.obs_global_median_max_times_off) == (3.5, 4.5, 5.5)
    assert (s.obs_global_median_min_countries, s.qc_min_overlap, s.qc_bad_median_times_off, s.qc_bad_share_over_3x) == (8, 9, 6.5, 7.5)


def test_the_functions_module_opens_no_file():
    io_calls = {'open', 'read_csv', 'read_excel', 'read_parquet', 'read_stata', 'read_file', 'to_csv', 'to_parquet', 'to_file',
                'to_excel', 'df_read', 'df_write', 'open_rasterio', 'as_array', 'Open', 'rasterize', 'zonal_statistics_flex'}
    tree = ast.parse(open(cpf.__file__, encoding='utf-8').read())
    called = {node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', None)
              for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert not called & io_calls, sorted(called & io_calls)


def test_the_shared_modules_never_import_the_upstream_package():
    for path in (cpt.__file__, cpf.__file__):
        tree = ast.parse(open(path, encoding='utf-8').read())
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        assert not {name for name in imported if name and name.split('.')[0] == 'crop_benefits'}, path


def test_a_service_folder_can_import_the_tasks_without_any_service():
    """shared_processing imports no service, so a service importing it cannot form a cycle."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(cpt.__file__))))
    code = ("import sys; sys.path.insert(0, %r); import global_invest.shared_processing.crop_prices_tasks; "
            "print([m for m in sys.modules if m.startswith('global_invest.') and m != 'global_invest.utilities' and not m.startswith('global_invest.shared_processing')])" % root)
    done = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    assert done.stdout.strip().split('\n')[-1] == '[]', done.stdout


def test_retained_accounting_signs_the_price_code():
    """A change to the price path invalidates a retained-accounting workspace, as a change to the pollination code does."""
    tree = ast.parse(open(pt.__file__, encoding='utf-8').read())
    signed = [node.value for node in ast.walk(tree)
              if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', None) == 'signed']
    assert len(signed) == 1
    files = {node.value.id for node in ast.walk(signed[0])
             if isinstance(node, ast.Attribute) and node.attr == '__file__' and isinstance(node.value, ast.Name)}
    assert files == {'pf', 'crop_prices_tasks', 'crop_prices_functions'}
    assert pt.crop_prices_tasks is cpt and pt.crop_prices_functions is cpf


def _small_price_path(tmp_path):
    """Four countries and one crop on disk, in the shapes the staged bulks and tables have.

    AAA reports 200 USD a tonne. BBB reports 300 in local currency at 3 per USD, which fills 100.
    DDD reports 3000 in local currency at 1 per USD, 15 times the only observed price, so the 10 times
    check on filled prices drops it though the 20 times check on observed prices does not apply. CCC reports no price.
    """
    crosswalks = tmp_path / 'fao' / 'crosswalks'
    crosswalks.mkdir(parents=True)
    (crosswalks / 'crosswalk_m49_iso3.csv').write_text(
        'area_code_m49,iso3,area_fao,region_fao,subregion_fao\n1,AAA,Aland,R,S\n2,BBB,Bland,R,S\n3,CCC,Cland,R,S\n4,DDD,Dland,R,S\n')
    (tmp_path / 'classification.csv').write_text('item_code_fao,item_fao,group_fao\n10,Wheat,cereals\n20,All crops,aggregate\n')
    header = 'Area Code (M49),Area,Item Code,Item,Element,Year,Value\n'
    (tmp_path / 'production.csv').write_text(header + ''.join("'%s\n" % row for row in (
        '001,Aland,10,Wheat,Production,2019,100', '002,Bland,10,Wheat,Production,2019,50', '003,Cland,10,Wheat,Production,2019,10',
        '004,Dland,10,Wheat,Production,2019,2', '001,Aland,10,Wheat,Yield,2019,3000', '001,Aland,10,Wheat,Production,2018,999',
        '001,Aland,20,All crops,Production,2019,999', '002,Bland,10,Wheat,Yield,2019,0', '999,Nowhere,10,Wheat,Production,2019,77')))
    usd, lcu, slc, index = 'Producer Price (USD/tonne)', 'Producer Price (LCU/tonne)', 'Producer Price (SLC/tonne)', 'Producer Price Index (2014-2016 = 100)'
    (tmp_path / 'prices.csv').write_text('Area Code (M49),Area,Item Code,Item,Element,Months,Year,Value\n' + ''.join("'%s\n" % row for row in (
        '001,Aland,10,Wheat,%s,Annual value,2019,200' % usd, '001,Aland,10,Wheat,%s,January,2019,999' % usd,
        '001,Aland,10,Wheat,%s,Annual value,2019,200' % slc, '001,Aland,10,Wheat,%s,Annual value,2019,100' % index,
        '002,Bland,10,Wheat,%s,Annual value,2019,300' % lcu, '004,Dland,10,Wheat,%s,Annual value,2019,3000' % lcu)))
    (tmp_path / 'fx.csv').write_text('iso3,year,lcu_per_usd\nAAA,2019,1.0\nBBB,2019,3.0\nDDD,2019,1.0\n')
    return cpf.FaoPriceSettings(
        crosswalk_m49_iso3_path=str(crosswalks / 'crosswalk_m49_iso3.csv'), fao_classification_path=str(tmp_path / 'classification.csv'),
        crosswalk_fao_cropgrids_path='not read by these steps', fao_production_bulk_path=str(tmp_path / 'production.csv'),
        fao_prices_bulk_path=str(tmp_path / 'prices.csv'), fx_lcu_per_usd_path=str(tmp_path / 'fx.csv'), output_dir=str(tmp_path / 'out'),
        fao_start_year=2019, fao_end_year=2019, price_years=(2019,), excluded_item_codes={20}, fx_inherit_iso3={}, imf_fao_names={},
        tol_near_equal_lcu_slc=0.01, anchor_max_year_dist=10, anchor_max_times_off=10.0, global_median_max_times_off=10.0,
        usd_copy_of_lcu_tol=0.001, usd_copy_of_lcu_min_fx=1.1, obs_global_median_max_times_off=20.0, obs_global_median_min_countries=10,
        qc_min_overlap=50, qc_bad_median_times_off=3.0, qc_bad_share_over_3x=0.5)


def test_a_small_panel_runs_from_raw_tables_through_the_three_steps(tmp_path):
    """Every expected number below is worked by hand from the rows in _small_price_path."""
    settings = _small_price_path(tmp_path)
    cpt.run_fao_production(settings)
    cpt.run_fao_prices(settings)
    cpt.run_fao_values(settings)
    out = tmp_path / 'out'
    assert sorted(path.name for path in out.iterdir()) == [
        'fao_prices_2019_2019.csv', 'fao_prices_2019_2019.parquet', 'fao_production_2019_2019.csv', 'fao_production_2019_2019.parquet',
        'fao_values_2019_2019.csv', 'fao_values_2019_2019.parquet', 'median_prices']

    production = pd.read_csv(out / 'fao_production_2019_2019.csv', dtype={'area_code_m49': str})
    keyed = production.set_index(['iso3', 'element'])['value']
    assert keyed.to_dict() == {('AAA', 'Production'): 100.0, ('BBB', 'Production'): 50.0, ('CCC', 'Production'): 10.0,
                               ('DDD', 'Production'): 2.0, ('AAA', 'Yield'): 3.0}
    assert set(production['item_code_fao']) == {10} and set(production['year']) == {2019}

    prices = pd.read_csv(out / 'fao_prices_2019_2019.csv').set_index('iso3')
    assert sorted(prices.index) == ['AAA', 'BBB', 'DDD']
    assert prices.loc['AAA', 'usd_source'] == 'observed' and prices.loc['AAA', 'price_usd_tonne'] == 200.0
    assert prices.loc['BBB', 'usd_source'] == 'filled' and prices.loc['BBB', 'price_usd_tonne'] == 100.0
    assert prices.loc['DDD', 'usd_source'] == 'implausible_vs_global_median_10x' and np.isnan(prices.loc['DDD', 'price_usd_tonne'])

    values = pd.read_csv(out / 'fao_values_2019_2019.csv').set_index('iso3')
    assert values['price_source_agg_level'].to_dict() == {'AAA': 'country', 'BBB': 'country', 'CCC': 'subregion', 'DDD': 'subregion'}
    assert values['price_usd_tonne'].to_dict() == {'AAA': 200.0, 'BBB': 100.0, 'CCC': 150.0, 'DDD': 150.0}
    assert values['value_usd'].to_dict() == {'AAA': 20000.0, 'BBB': 5000.0, 'CCC': 1500.0, 'DDD': 300.0}

    medians = pd.read_parquet(out / 'median_prices' / 'price_median_usd_tonne_2019_2019.parquet')
    by_level = medians.groupby('agg_level')['price_median_usd_tonne'].apply(list).to_dict()
    assert by_level == {'country': [200.0, 100.0], 'subregion': [150.0], 'region': [150.0], 'world': [150.0]}

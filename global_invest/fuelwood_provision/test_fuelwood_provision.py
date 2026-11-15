"""Unit tests for the fuelwood-provision calculation: the units pin, the join rules and the
unallocatable handling, plus the staged-sheet anchor where the file is on the machine."""
import os

import pandas as pd
import pytest

from global_invest.fuelwood_provision import fuelwood_provision_functions as fw


def _countries():
    return pd.DataFrame({'iso3_r250_id': [76, 356, 528],
                         'iso3_r250_label': ['BRA', 'IND', 'NLD']})


def test_values_scale_from_the_sheets_stated_thousands():
    results = pd.DataFrame({'ISO-3': ['BRA', 'IND'], 'Value(2019 $)': [2.5, 10.0]})
    gep, unallocated = fw.fuelwood_gep_by_country(results, _countries())
    assert gep.set_index('iso3_r250_id')['fuelwood_provision_gep'].to_dict() == \
        {76: 2500.0, 356: 10000.0}
    assert unallocated.empty


def test_the_dissolved_antilles_is_reported_not_dropped_and_unknown_codes_raise():
    results = pd.DataFrame({'ISO-3': ['BRA', 'ANT'], 'Value(2019 $)': [1.0, 3.0]})
    gep, unallocated = fw.fuelwood_gep_by_country(results, _countries())
    assert len(gep) == 1
    assert unallocated['value_usd'].tolist() == [3000.0]
    assert 'dissolved' in unallocated['reason'].iloc[0]
    with pytest.raises(ValueError):
        fw.fuelwood_gep_by_country(
            pd.DataFrame({'ISO-3': ['XXX'], 'Value(2019 $)': [1.0]}), _countries())


def test_duplicates_and_negatives_refuse():
    with pytest.raises(ValueError):
        fw.fuelwood_gep_by_country(
            pd.DataFrame({'ISO-3': ['BRA', 'BRA'], 'Value(2019 $)': [1.0, 2.0]}), _countries())
    with pytest.raises(ValueError):
        fw.fuelwood_gep_by_country(
            pd.DataFrame({'ISO-3': ['BRA'], 'Value(2019 $)': [-1.0]}), _countries())


def test_the_staged_cas_sheet_totals_what_the_entry_says():
    path = os.path.expanduser('~/Files/base_data/global_invest/fuelwood/author_drive/'
                              'fuelwood_provision_gep.xlsx')
    if not os.path.exists(path):
        pytest.skip('the CAS fuelwood sheet is not on this machine')
    results = pd.read_excel(path)
    assert len(results) == 180
    total_usd = results['Value(2019 $)'].sum() * fw.FUELWOOD_RESULTS_UNIT_USD
    assert total_usd == pytest.approx(181_141_065_451, abs=1_000)

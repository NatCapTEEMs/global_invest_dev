"""Unit tests for the shared utilities.

These cover the parts every service depends on, where a defect is invisible in any one service's
tests because each one sees only its own configuration.
"""
import os

import pandas as pd
import pytest

from global_invest import utilities


def _write(path, records):
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def test_the_project_copy_takes_rows_columns_and_blanks_the_template_is_ahead_on(tmp_path):
    # Seeding copies a definitions CSV only when it is absent, so a copy made before a key was
    # added shadows the template forever. All three ways the template can be ahead are filled.
    template = _write(tmp_path / 't.csv',
                      [{'service': 'a', 'parameter': 'p1', 'value': 'TPL', 'note': 'n1'},
                       {'service': 'a', 'parameter': 'p2', 'value': 'TPL2', 'note': 'n2'}])
    local = _write(tmp_path / 'l.csv', [{'service': 'a', 'parameter': 'p1', 'value': ''}])

    utilities.add_rows_missing_from_template(str(local), str(template),
                                             ['service', 'parameter'], log=lambda *a: None)

    out = pd.read_csv(local).set_index('parameter')
    assert list(out.index) == ['p1', 'p2']          # the absent row was appended
    assert 'note' in out.columns                    # the absent column was added
    assert out.at['p1', 'value'] == 'TPL'           # the blank cell was filled
    assert out.at['p1', 'note'] == 'n1'


def test_a_value_the_machine_has_set_is_never_overwritten(tmp_path):
    # Machine-specific settings ship blank in the template and are filled in the project's copy.
    # Syncing the schema must not reach into them, or a run would silently use another machine's
    # ssh host or drive path.
    template = _write(tmp_path / 't.csv',
                      [{'service': 'flood', 'parameter': 'vm_ssh_host', 'value': ''},
                       {'service': 'flood', 'parameter': 'gempack_dir', 'value': ''}])
    local = _write(tmp_path / 'l.csv',
                   [{'service': 'flood', 'parameter': 'vm_ssh_host', 'value': 'user@192.168.64.2'},
                    {'service': 'flood', 'parameter': 'gempack_dir', 'value': 'C:\\GP'}])
    before = open(local).read()

    utilities.add_rows_missing_from_template(str(local), str(template),
                                             ['service', 'parameter'], log=lambda *a: None)

    assert open(local).read() == before             # nothing to do, so nothing written


def test_a_template_value_does_not_replace_a_different_local_one(tmp_path):
    """The template's value is a default, not an instruction: a copy that answers differently keeps
    its answer, because that is the whole reason the copy exists."""
    template = _write(tmp_path / 't.csv',
                      [{'service': 'a', 'parameter': 'p1', 'value': 'from_template'}])
    local = _write(tmp_path / 'l.csv',
                   [{'service': 'a', 'parameter': 'p1', 'value': 'set_on_this_machine'}])

    utilities.add_rows_missing_from_template(str(local), str(template),
                                             ['service', 'parameter'], log=lambda *a: None)

    assert pd.read_csv(local).at[0, 'value'] == 'set_on_this_machine'


def test_a_missing_file_on_either_side_is_left_alone(tmp_path):
    """A project without the copy yet is seeded elsewhere; this step has nothing to add to it."""
    template = _write(tmp_path / 't.csv', [{'service': 'a', 'parameter': 'p1', 'value': 'x'}])
    absent = tmp_path / 'nothing_here.csv'
    utilities.add_rows_missing_from_template(str(absent), str(template),
                                             ['service', 'parameter'], log=lambda *a: None)
    assert not os.path.exists(absent)

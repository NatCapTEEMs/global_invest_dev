"""Contract test over the SHIPPED es_config.csv: every service's row hydrates.

Chiara's point that most services had no test at all: this parametrizes over the real template,
so adding a row (or editing a cell) is automatically under test. For each service: non-empty
cells arrive on p (paths through get_path, integers typed), empty cells set nothing, and a
skeleton row (crosswalk only) hydrates nothing but the crosswalk columns. Science-level tests
per service remain a separate, open gap -- this pins only the configuration contract.
"""
import os
import types

import pandas as pd
import pytest

from global_invest import utilities

TEMPLATE_PATH = os.path.join(os.path.dirname(utilities.__file__), 'input_template', 'es_config.csv')
TEMPLATE = pd.read_csv(TEMPLATE_PATH)
GEP_COLUMNS = [c for c in TEMPLATE.columns if c != 'service']


def fake_p(tmp_path):
    p = types.SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    return p


@pytest.mark.parametrize('service', TEMPLATE['service'].tolist())
def test_every_shipped_row_hydrates_its_nonempty_cells_and_only_those(tmp_path, service):
    p = fake_p(tmp_path)
    utilities.hydrate_es_config(p, service, log=lambda *a: None)
    row = TEMPLATE[TEMPLATE['service'] == service].iloc[0]
    for column in GEP_COLUMNS:
        value = row[column]
        if pd.isna(value):
            assert not hasattr(p, column), f'{service}: empty cell {column} must set nothing'
        elif column.endswith('_path'):
            assert getattr(p, column) == '/resolved/' + str(value)
        elif column == 'gep_base_year':
            assert getattr(p, column) == int(value) and isinstance(getattr(p, column), int)
        else:
            assert getattr(p, column) == str(value)


def test_both_carbon_services_share_the_sheet_group_but_not_the_subgroup():
    carbon = TEMPLATE[TEMPLATE['sheet_label'] == 'global_climate_regulation']
    assert sorted(carbon['service']) == ['coastal_carbon', 'terrestrial_carbon']
    assert sorted(carbon['sheet_subgroup']) == ['coastal', 'terrestrial']


def test_pollination_base_year_and_raster_cells_agree():
    # The two cells name the year twice by design (the old f-string coupling, made explicit);
    # this is the gate that keeps them a PAIR when one is edited.
    row = TEMPLATE[TEMPLATE['service'] == 'pollination'].iloc[0]
    assert f"{int(row['gep_base_year'])}usd" in row['gep_quantity_input_path']

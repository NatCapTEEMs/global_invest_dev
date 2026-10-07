"""A pass that finds a finished shock table must return in seconds. The three heavy shock tasks
(pollination, erosion, timber) record what produced their table -- settings, content of their
es_parameters inputs, size and mtime of the scenario maps -- and reuse it when nothing changed.
On 21 Sep 2026 the NGFS rerun recomputed all three on every pass, about an hour lost each time."""
import inspect
import os
import time
import pytest
from global_invest import utilities
from global_invest.pollination import pollination_tasks
from global_invest.erosion import erosion_tasks
from global_invest.timber_provision import timber_provision_tasks


@pytest.fixture
def table_and_map(tmp_path):
    table = tmp_path / 'service_interpolated.csv'
    table.write_text('ENDW,ACTS,REG,scenario,year,shock_pct\nAEZ7,WHT,USA,net_zero,2050,-1.0\n')
    lulc = tmp_path / 'lulc_2050.tif'
    lulc.write_bytes(b'\x00' * 1024)
    return str(table), str(lulc)


def _signature(settings, lulc):
    return utilities._signature(settings, {}, light_inputs=[lulc])


def test_reused_when_nothing_changed(table_and_map, tmp_path):
    table, lulc = table_and_map
    sig_path = str(tmp_path / 'sig.json')
    sig = _signature({'alpha': 0.08}, lulc)
    utilities.write_outputs_signature(sig, sig_path)
    assert utilities.outputs_reuse_reason([table], sig, sig_path) is None


def test_missing_table_or_signature_recomputes(table_and_map, tmp_path):
    table, lulc = table_and_map
    sig = _signature({'alpha': 0.08}, lulc)
    assert 'no signature' in utilities.outputs_reuse_reason([table], sig, str(tmp_path / 'absent.json'))
    assert 'no ' in utilities.outputs_reuse_reason([str(tmp_path / 'gone.csv')], sig, str(tmp_path / 'absent.json'))


def test_changed_setting_or_touched_map_recomputes(table_and_map, tmp_path):
    table, lulc = table_and_map
    sig_path = str(tmp_path / 'sig.json')
    utilities.write_outputs_signature(_signature({'alpha': 0.08}, lulc), sig_path)
    assert 'alpha' in utilities.outputs_reuse_reason([table], _signature({'alpha': 0.10}, lulc), sig_path)
    later = time.time() + 5
    os.utime(lulc, (later, later))
    assert lulc in utilities.outputs_reuse_reason([table], _signature({'alpha': 0.08}, lulc), sig_path)


@pytest.mark.parametrize('task, heavy_step', [
    (pollination_tasks.pollination_shock, '_zonal_context('),
    (erosion_tasks.erosion_shock, 'resample_band_to_match('),
    (timber_provision_tasks.timber_provision_shock, '_write_eligible_value('),
])
def test_shock_task_checks_reuse_before_its_heavy_step_and_records_after_writing(task, heavy_step):
    src = inspect.getsource(task)
    check = src.index('utilities.reuse_reason(')
    heavy = src.index(heavy_step)
    write = src.index('.to_csv(')
    record = src.index('utilities.write_reuse_signature(')
    assert check < heavy, '%s: the reuse check must come before %s' % (task.__name__, heavy_step)
    assert 'return' in src[check:heavy], '%s: a reusable table must make the task return' % task.__name__
    assert write < record, '%s: the signature must be recorded after the table is written' % task.__name__
    assert 'light_inputs=' in src[check:check + 200], '%s: the scenario maps must be part of the signature' % task.__name__

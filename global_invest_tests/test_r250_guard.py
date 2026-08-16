"""Tripwire: the r250 canonical-row guard must stay present in each of the five accounting
services' valuation code. The guard (ee_r264_label == iso3_r250_label) is what prevents the
split-country double-count (China x6, India x6 -- see global_invest/utilities.py); the two carbon
services enforce the same rule by grouping on iso3_r250 instead and carry their own regression
tests. A source-level check, like test_es_seams: a merge that drops the line must fail a test,
not a reviewer's memory."""
import pathlib

SERVICES = ['crop_provision', 'livestock_provision', 'extractive_materials_provision',
            'renewable_energy_provision', 'coastal_protection']
GUARD = "ee_r264_label'] == "


def test_all_five_services_keep_the_canonical_row_guard():
    root = pathlib.Path(__file__).resolve().parents[1] / 'global_invest'
    missing = []
    for s in SERVICES:
        src = (root / s / f'{s}_tasks.py').read_text()
        if GUARD not in src:
            missing.append(s)
    assert not missing, f"r250 canonical guard missing from: {missing} -- split-country double-count risk"

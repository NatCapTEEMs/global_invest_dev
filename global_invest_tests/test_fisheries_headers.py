"""Fisheries header resolution: the path that carries every scenario once the legacy map goes.

Pins two properties: a header resolves purely through es_shock_climate_labels (the path with no
legacy map), and the HAR header-read list does not depend on header_map alone, because an empty
map then reads no headers and silently drops every scenario, whatever the RCP derivation resolved.
"""
from global_invest.fisheries.fisheries_functions import (
    FISH_HEADER_MAP, RCP_FI_MAP, fisheries_headers_to_read, resolve_fisheries_header)


def test_resolution_order_explicit_then_rcp_then_identity():
    climate = {'below_2c': 'rcp26', 'mystery': 'rcp19'}
    # explicit map wins over the RCP derivation
    assert resolve_fisheries_header('below_2c', {'below_2c': 'FI85'}, climate) == 'FI85'
    # no map entry: the scenario's RCP decides
    assert resolve_fisheries_header('below_2c', {}, climate) == 'FI26'
    # unknown RCP: identity (and the write-time assert stays the loud failure downstream)
    assert resolve_fisheries_header('mystery', {}, climate) == 'mystery'
    # no map, no climate entry: FI-native labels pass straight through
    assert resolve_fisheries_header('FI45', {}, {}) == 'FI45'


def test_climate_only_resolution_covers_all_ngfs_production_scenarios():
    # The post-deletion path: header_map empty, everything resolves through climate_label.
    ngfs_climate = {'below_2c': 'rcp26', 'net_zero': 'rcp26', 'low_demand': 'rcp26',
                    'ndcs': 'rcp45', 'delayed_transition': 'rcp45',
                    'baseline_ignore_dependencies': 'rcp45',
                    'current_policies': 'rcp70', 'fragmented_world': 'rcp70', 'stress_test': 'rcp70',
                    # The es20 variant shares its climate with the policy it varies: it cuts the
                    # service shock, not the RCP. It is listed so its header is checked like every
                    # other scenario's.
                    'current_policies-es20': 'rcp70'}
    resolved = {s: resolve_fisheries_header(s, {}, ngfs_climate) for s in ngfs_climate}
    assert set(resolved.values()) == {'FI26', 'FI45', 'FI85'}
    # and where the legacy dict has an opinion, the derivation agrees with it
    missing = sorted(set(FISH_HEADER_MAP) - set(ngfs_climate))
    assert not missing, 'the climate fixture does not cover %s, so they go unchecked' % missing
    for scen, hdr in FISH_HEADER_MAP.items():
        assert resolved[scen] == hdr


def test_headers_to_read_survives_an_empty_map():
    assert fisheries_headers_to_read({}) == ('FI26', 'FI45', 'FI85')
    # a consumer map pointing at the same universe adds nothing new
    assert fisheries_headers_to_read(FISH_HEADER_MAP) == ('FI26', 'FI45', 'FI85')
    # a consumer override to a header outside the RCP universe still gets read
    assert 'FI99' in fisheries_headers_to_read({'exotic': 'FI99'})
    assert set(RCP_FI_MAP.values()) <= set(fisheries_headers_to_read({'exotic': 'FI99'}))

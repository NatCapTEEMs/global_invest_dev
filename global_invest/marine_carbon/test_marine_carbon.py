"""Unit tests for the marine_carbon port (four ocean carbon pools shared out by area).

Synthetic polygons pin the selection, the dissolve and the measurement in the equal-area CRS,
including the source's sum-without-union total. Synthetic pools pin the arithmetic: each pool's
value is conserved across the countries and the high seas, the high-seas row is one row kept out
of the country table, and a country with no EEZ publishes missing rather than zero. The pinned
reference (the author's code at 0dac21f run on the shared marine layer, staged in the service's
base-data folder) pins the port to the source: fed the reference's own areas and the shipped
pools, it must return the reference's allocation.
"""
import os
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from global_invest import utilities
from global_invest.marine_carbon import marine_carbon_functions as mcf

AREA_CRS = 'ESRI:54009'
EEZ = 'Marine EEZ'
OCEAN = 'Marine EEZ|Nearest Open Ocean'
POOLS = [{'pool': 'a', 'stock_pg_c': 2.0, 'rental_price_usd_per_tc_per_year': 10.0},
         {'pool': 'b', 'stock_pg_c': 1.0, 'rental_price_usd_per_tc_per_year': 30.0}]


def _layer(rows):
    """A marine-correspondence-shaped layer from (name, id, label, polygon) rows in EPSG:4326."""
    return gpd.GeoDataFrame(
        {'eemarine_r566_name': [r[0] for r in rows], 'iso3_r250_id': [r[1] for r in rows],
         'iso3_r250_label': [r[2] for r in rows], 'iso3_r250_name': [r[2] for r in rows]},
        geometry=[r[3] for r in rows], crs='EPSG:4326')


def _km2(*polygons):
    return float(gpd.GeoSeries(list(polygons), crs='EPSG:4326').to_crs(AREA_CRS).area.sum() / 1e6)


LAYER = _layer([
    ('Aaa Marine EEZ', 1.0, 'AAA', box(0, 0, 1, 1)),
    ('Aaa Marine EEZ', 1.0, 'AAA', box(1, 0, 2, 1)),      # a second polygon, dissolved into AAA
    ('Bbb Marine EEZ', 2.0, 'BBB', box(0, 1, 1, 2)),
    ('Aaa Nearest Open Ocean', 1.0, 'AAA', box(3, 0, 5, 2)),
    ('Aaa', 1.0, 'AAA', box(10, 10, 11, 11)),             # land, in neither selection
])


def test_eez_areas_dissolve_each_country_and_measure_in_the_equal_area_crs():
    areas = mcf.eez_areas_by_country(LAYER, EEZ, AREA_CRS).set_index('iso3_r250_label')
    assert list(areas.index) == ['AAA', 'BBB']
    assert np.isclose(areas.loc['AAA', 'eez_area_km2'], _km2(box(0, 0, 1, 1), box(1, 0, 2, 1)))
    assert np.isclose(areas.loc['BBB', 'eez_area_km2'], _km2(box(0, 1, 1, 2)))
    assert np.isclose(areas['proportion_of_global_eez'].sum(), 1.0)
    assert areas['iso3_r250_id'].tolist() == [1, 2]


def test_an_eez_polygon_without_a_country_id_stays_its_own_row():
    layer = pd.concat([LAYER, _layer([('Joint Marine EEZ', np.nan, None, box(0, 3, 1, 4))])],
                      ignore_index=True)
    areas = mcf.eez_areas_by_country(layer, EEZ, AREA_CRS)
    assert len(areas) == 3
    assert areas['iso3_r250_id'].isna().sum() == 1


def test_total_ocean_sums_eez_and_open_ocean_without_union():
    """The source's construction, pinned so that changing it is a decision: an area two ocean
    polygons share is counted once per polygon."""
    total = mcf.total_ocean_area_km2(LAYER, OCEAN, AREA_CRS)
    assert np.isclose(total, _km2(box(0, 0, 2, 1), box(0, 1, 1, 2), box(3, 0, 5, 2)))
    overlapping = pd.concat([LAYER, _layer([('Bbb Nearest Open Ocean', 2.0, 'BBB', box(4, 0, 6, 2))])],
                            ignore_index=True)
    doubled = mcf.total_ocean_area_km2(overlapping, OCEAN, AREA_CRS)
    assert np.isclose(doubled - total, _km2(box(4, 0, 6, 2)))


def _areas():
    return pd.DataFrame({'iso3_r250_id': [1, 2], 'iso3_r250_label': ['AAA', 'BBB'],
                         'iso3_r250_name': ['Aaa', 'Bbb'], 'eez_area_km2': [30.0, 10.0],
                         'proportion_of_global_eez': [0.75, 0.25]})


def test_each_pool_is_valued_and_conserved_across_countries_and_high_seas():
    allocation = mcf.allocate_marine_carbon_value(_areas(), 100.0, POOLS)
    assert np.isclose(allocation['share_of_total_ocean_area'].sum(), 1.0)
    # stock (Pg C) x 1e9 tonnes per Pg x price per tonne C
    assert np.isclose(allocation['allocated_value_a_usd'].sum(), 2.0 * 1e9 * 10.0)
    assert np.isclose(allocation['allocated_value_b_usd'].sum(), 1.0 * 1e9 * 30.0)
    pool_sum = allocation[['allocated_value_a_usd', 'allocated_value_b_usd']].sum(axis=1)
    assert np.allclose(allocation['allocated_value_usd'], pool_sum, rtol=0)
    aaa = allocation.set_index('iso3_r250_label').loc['AAA']
    assert np.isclose(aaa['allocated_value_usd'], 0.30 * (20e9 + 30e9))


def test_the_high_seas_are_one_row_holding_the_ocean_outside_every_eez():
    allocation = mcf.allocate_marine_carbon_value(_areas(), 100.0, POOLS)
    high_seas = allocation[allocation['iso3_r250_label'] == mcf.HIGH_SEAS_LABEL]
    assert len(high_seas) == 1
    assert high_seas['iso3_r250_id'].iloc[0] == mcf.HIGH_SEAS_ID
    assert np.isclose(high_seas['area_km2'].iloc[0], 60.0)
    with pytest.raises(ValueError, match='exceed the total ocean area'):
        mcf.allocate_marine_carbon_value(_areas(), 39.0, POOLS)


def test_the_country_rows_never_carry_the_high_seas():
    allocation = mcf.allocate_marine_carbon_value(_areas(), 100.0, POOLS)
    national, high_seas = mcf.national_and_high_seas(allocation)
    assert mcf.HIGH_SEAS_LABEL not in set(national['iso3_r250_label'])
    assert mcf.HIGH_SEAS_ID not in set(national['iso3_r250_id'])
    assert np.isclose(national['marine_carbon_gep'].sum() + high_seas['allocated_value_usd'].sum(),
                      sum(mcf.pool_value_usd(pool) for pool in POOLS))
    relabelled = allocation.copy()
    relabelled.loc[relabelled['iso3_r250_label'] == 'BBB', 'iso3_r250_id'] = mcf.HIGH_SEAS_ID
    with pytest.raises(ValueError, match='exactly one high-seas row'):
        mcf.national_and_high_seas(relabelled)


def test_a_country_with_no_eez_publishes_missing_not_zero(tmp_path):
    """On the account's country list, a country the allocation never mentions is NA."""
    countries = pd.DataFrame({'ee_r264_id': [1, 2, 3], 'iso3_r250_id': [1, 2, 3],
                              'ee_r264_label': ['AAA', 'BBB', 'CCC'],
                              'iso3_r250_label': ['AAA', 'BBB', 'CCC'],
                              'iso3_r250_name': ['Aaa', 'Bbb', 'Ccc']})
    p = SimpleNamespace(df_countries=countries)
    national, high_seas = mcf.national_and_high_seas(
        mcf.allocate_marine_carbon_value(_areas(), 100.0, POOLS))
    path = utilities.write_gep_by_country(p, national, os.path.join(tmp_path, 'gep.csv'),
                                          log=lambda *a: None)
    published = pd.read_csv(path).set_index('iso3_r250_label')
    assert list(published.index) == ['AAA', 'BBB', 'CCC']
    assert np.isnan(published.loc['CCC', 'marine_carbon_gep'])
    assert np.isclose(published['marine_carbon_gep'].sum() + high_seas['allocated_value_usd'].iloc[0],
                      sum(mcf.pool_value_usd(pool) for pool in POOLS))


def test_the_port_returns_the_pinned_reference_allocation_from_the_references_areas(tmp_path):
    """The author's code at 0dac21f, run on the shared marine layer, is staged as the reference.
    From the reference's own areas and the shipped pools the port returns the same table, the
    same rows in the same order, every value equal."""
    import hazelbean as hb
    # A bare ProjectFlow resolves the reference rows of es_parameters against base data, the
    # way a run finds its inputs.
    p = hb.ProjectFlow(project_dir=os.path.join(tmp_path, 'anchors'))
    utilities.hydrate_es_parameters(p, 'marine_carbon', log=lambda *a: None)
    exact = dict(float_precision='round_trip')
    eez_areas = pd.read_csv(p.marine_carbon_reference_eez_areas_path, **exact)
    total_km2 = float(pd.read_csv(p.marine_carbon_reference_total_ocean_area_path, **exact)
                      ['total_ocean_area_km2'].iloc[0])
    reference = pd.read_csv(p.marine_carbon_reference_value_path, **exact)
    pools = p.marine_carbon_pools

    port = mcf.allocate_marine_carbon_value(eez_areas, total_km2, pools)

    assert list(port.columns) == list(reference.columns)
    assert port['iso3_r250_label'].tolist() == reference['iso3_r250_label'].tolist()
    assert port['iso3_r250_id'].tolist() == reference['iso3_r250_id'].tolist()
    numeric = reference.select_dtypes('number').columns
    assert (port[numeric].to_numpy() == reference[numeric].to_numpy()).all()
    national, high_seas = mcf.national_and_high_seas(port)
    assert len(national) == 187
    assert np.isclose(national['marine_carbon_gep'].sum() + high_seas['allocated_value_usd'].iloc[0],
                      sum(mcf.pool_value_usd(pool) for pool in pools), rtol=1e-12)


def test_es_config_and_parameters_rows_hydrate_marine_carbon(tmp_path):
    p = SimpleNamespace(input_dir=str(tmp_path / 'input'), get_path=lambda *a, **k: '/resolved/' + '/'.join(a))
    utilities.hydrate_es_config(p, 'marine_carbon', log=lambda *a: None)
    assert p.gep_base_year == 2019
    assert p.gep_regions_input_path.endswith('cartographic/ee/eemarine_r566_correspondence.gpkg')
    assert not hasattr(p, 'sheet_label')     # no sheet column mapped yet (reported separately, outside the total)
    utilities.hydrate_es_parameters(p, 'marine_carbon', log=lambda *a: None)
    assert [pool['pool'] for pool in p.marine_carbon_pools] == [
        'DIC', 'DOC_and_biota', 'sediment_carbon', 'methane_hydrates']
    assert p.marine_carbon_area_crs == AREA_CRS
    assert p.marine_carbon_eez_name_pattern == EEZ
    assert p.marine_carbon_ocean_name_pattern == OCEAN

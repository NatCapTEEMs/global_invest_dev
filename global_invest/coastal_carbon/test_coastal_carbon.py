"""Regression tests for the coastal_carbon gep_calculation (develop_yanxu rework, merged).

Synthetic data only (the full chain needs the coastal source data). Pins the two aggregation
contracts of the final iso3_r250 stage: only marine `_EEZ` rows are valued, and the r264
correspondence is collapsed to its canonical row per country (`ee_r264_label == iso3_r250_label`)
before the join, so a split country is counted once (see global_invest/utilities.py).
"""
from types import SimpleNamespace

import pandas as pd

from global_invest.coastal_carbon import coastal_carbon_tasks as cct


def test_gep_calculation_eez_only_and_one_row_per_country(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box

    # r566 combined-area frame: CHN and NLD EEZ rows, plus a CHN land row that must be EXCLUDED
    # (only *_EEZ rows are valued) and would double CHN if it leaked through.
    pd.DataFrame({
        'eemarine_r566_id': [1, 2, 3],
        'eemarine_r566_label': ['CHN_EEZ', 'NLD_EEZ', 'CHN_LAND'],
        'iso3_r250_label': ['CHN', 'NLD', 'CHN'],
        'mangrove_total_c_stock_mg': [50.0, 5.0, 50.0],
        'salt_marsh_total_c_stock_mg': [30.0, 3.0, 30.0],
        'seagrass_total_c_stock_mg': [20.0, 2.0, 20.0],
        'total_coastal_carbon_area_ha': [10.0, 1.0, 10.0],
    }).to_csv(tmp_path / 'combined_areas.csv', index=False)

    pd.DataFrame({'year': [2019], 'p': [2.0]}).to_excel(tmp_path / 'carbon_prices.xlsx', index=False)

    # r264 correspondence: CHN split into two r264 rows (canonical row = ee_r264_label == 'CHN');
    # without the canonical filter the join would duplicate CHN and the sum would double it.
    pd.DataFrame({
        'ee_r264_label': ['CHN', 'CHN_x', 'NLD'],
        'iso3_r250_label': ['CHN', 'CHN', 'NLD'],
        'iso3_r250_name': ['China', 'China', 'Netherlands'],
        'continent': ['Asia', 'Asia', 'Europe'],
    }).to_csv(tmp_path / 'ee_r264_correspondence.csv', index=False)

    gpd.GeoDataFrame({'eemarine_r566_id': [1, 2, 3],
                      'geometry': [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)]},
                     crs='EPSG:4326').to_file(str(tmp_path / 'marine.gpkg'), driver='GPKG')

    p = SimpleNamespace(cur_dir=str(tmp_path), results={},
                        combined_area_path=str(tmp_path / 'combined_areas.csv'),
                        carbon_prices_path=str(tmp_path / 'carbon_prices.xlsx'), carbon_price='p',
                        gdf_countries_marine_vector_path=str(tmp_path / 'marine.gpkg'),
                        df_countries_csv_path=str(tmp_path / 'ee_r264_correspondence.csv'))

    total = cct.gep_calculation(p)

    # CHN EEZ = (50+30+20)*2 = 200, NLD EEZ = (5+3+2)*2 = 20. Land row excluded, split country
    # counted once -> 220 (not 400 via the duplicate r264 join, not 420 via the land leak).
    assert total == 220.0
    out = pd.read_csv(tmp_path / 'gep_by_country_base_year.csv').set_index('iso3_r250_label')
    assert out.loc['CHN', 'value'] == 200.0
    assert out.loc['NLD', 'value'] == 20.0
    assert (out['value'].index.value_counts() == 1).all()      # one row per iso3 country

    # Stage-1 r566 detail written for csv + map gpkg (per-region values, never summed downstream).
    r566 = pd.read_csv(tmp_path / 'gep_by_country_base_year_r566.csv')
    assert r566.loc[r566['eemarine_r566_label'] == 'CHN_EEZ', 'coastal_carbon_storage_value'].item() == 200.0

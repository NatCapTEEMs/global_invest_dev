"""Regression test for the coastal_carbon national-GEP double-count fix.

The only part of this module the current branch changes is gep_calculation (r250-only sum + the
results contract), so that is what is tested -- on synthetic data, since no coastal project data is
local. The rest of the module cannot build (see the docstring atop coastal_carbon_initialize.py);
when the develop_yanxu rework merges, update this test to its valuation interface.
"""
from types import SimpleNamespace

import pandas as pd

from global_invest.coastal_carbon import coastal_carbon_tasks as cct

ATTRS = ['iso3_r250_label', 'iso3_r250_name', 'continent', 'region_un',
         'region_wb', 'income_grp', 'subregion']


def test_gep_calculation_sums_one_row_per_country_not_per_r264_region(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box

    # A split country (2 r264 sub-regions, one iso3) and a normal country. Quantity totals:
    # CHN 60 + 40 = 100, NLD 10. Price 2.0 -> correct national GEP = 220.
    # The pre-fix merge-then-sum counted CHN once per sub-region: (100*2)*2 + 20 = 420.
    rows = [
        {'ee_r264_id': 1, 'iso3_r250_id': 156, 'total': 60.0},
        {'ee_r264_id': 2, 'iso3_r250_id': 156, 'total': 40.0},
        {'ee_r264_id': 3, 'iso3_r250_id': 528, 'total': 10.0},
    ]
    df = pd.DataFrame(rows)
    df['year'] = 2019
    names = {156: 'CHN', 528: 'NLD'}
    for a in ATTRS:
        df[a] = df['iso3_r250_id'].map(names)
    q_csv = tmp_path / 'carbon_by_region.csv'
    df.to_csv(q_csv, index=False)

    prices = tmp_path / 'carbon_prices.xlsx'
    pd.DataFrame({'year': [2019], 'p': [2.0]}).to_excel(prices, index=False)

    regions = tmp_path / 'regions.gpkg'
    gpd.GeoDataFrame({'ee_r264_id': [1, 2, 3],
                      'geometry': [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)]},
                     crs='EPSG:4326').to_file(str(regions), driver='GPKG')

    p = SimpleNamespace(cur_dir=str(tmp_path), results={},
                        carbon_by_region_base_year_path=str(q_csv),
                        carbon_prices_path=str(prices), carbon_price='p',
                        gdf_countries_simplified=str(regions))

    total = cct.gep_calculation(p)

    assert total == 220.0                                   # not 420: split country counted once
    out = pd.read_csv(tmp_path / 'gep_by_country_base_year.csv')
    assert len(out) == 2                                    # one row per iso3 country
    by = out.set_index('iso3_r250_label')
    assert by.loc['CHN', 'coastal_carbon_quantity'] == 100.0
    assert by.loc['CHN', 'coastal_carbon_gep'] == 200.0
    assert by.loc['NLD', 'coastal_carbon_gep'] == 20.0

    # Map contract: the r264 gpkg carries each sub-region with its COUNTRY's value, never summed.
    gdf = gpd.read_file(tmp_path / 'gep_by_country_base_year.gpkg')
    assert len(gdf) == 3
    chn_rows = gdf[gdf['ee_r264_id'].isin([1, 2])]
    assert (chn_rows['coastal_carbon_gep'] == 200.0).all()

"""Derive the bound on each table value the prediction lets move. Takes no arguments.

For a login node, read only outside prediction/. The prediction gives the share step on each moved cell and the
production there. This adds what the bound also needs: the author's value of each row, how many cells each country's
sum runs over, and a lower limit on how many cells its mean runs over, the cells with a nonzero author share. Writes
prediction/magnitude_bounds.csv, one row per key and column, and refuses if it exists. See launch/eps5_prediction.py
for the arithmetic behind each bound.

Run: python -I -B run_eps5_prediction_bounds.py
"""
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features

HERE = os.path.dirname(os.path.abspath(__file__))
PREDICTION = os.path.join(HERE, 'prediction')
OUT = os.path.join(PREDICTION, 'magnitude_bounds.csv')
sys.path.insert(0, os.path.join(HERE, 'launch'))
import eps5_gates as eg
import eps5_prediction as epr


def country_cells(band, transform, crs, boundary_path, iso3):
    """How many analysis cells the country holds under the task's centre rule, and how many carry a nonzero share."""
    gdf = gpd.read_file(boundary_path)
    gdf = gdf[gdf['iso3_r250_label'] == iso3].to_crs(crs)
    mask = rasterio.features.rasterize([(g, 1) for g in gdf.geometry], out_shape=band.shape, transform=transform, fill=0, dtype='uint8', all_touched=False).astype(bool)
    return int(mask.sum()), int(np.count_nonzero(np.nan_to_num(band[mask])))


def main():
    if os.path.exists(OUT):
        raise eg.Mismatch('%s already exists: move it aside first' % OUT)
    q = eg.read_parameters(os.path.join(HERE, 'msi_parameters.csv'))
    role = {r['role']: os.path.join(q['author_archive_dir'], r['relative_path']) for r in eg.read_rows(os.path.join(HERE, 'author_pins.csv'))}
    expected = pd.read_csv(os.path.join(PREDICTION, epr.EXPECTED), encoding='utf-8-sig', float_precision='round_trip')
    expected = expected[expected['run'] == 'eps5']
    cells = pd.read_csv(os.path.join(PREDICTION, epr.CELLS), encoding='utf-8-sig', float_precision='round_trip', keep_default_na=False, na_values=[''])
    moved = cells[cells['eps5_onfarm_predicted_library'] != cells['eps5_onfarm_stored_author']]
    steps = (moved['eps5_onfarm_predicted_library'] - moved['eps5_onfarm_stored_author']).abs()
    if set(steps) != set(expected['largest_abs_delta_share']):
        raise eg.Mismatch('the moved cells do not share the one step the table bound needs: %s' % sorted(set(steps)))
    author = pd.read_csv(role['eps5_long'], float_precision='round_trip', keep_default_na=False, na_values=['']).set_index(['component', 'ISO3', 'crop'])
    diagnostics = pd.read_csv(os.path.join(os.path.dirname(role['eps5_long']), 'country_ps_diagnostics.csv'), float_precision='round_trip', keep_default_na=False, na_values=['']).set_index('ISO3')
    with rasterio.open(role['eps5_ps_combined']) as d:
        band, transform, crs = d.read(1), d.transform, d.crs
    boundary = os.path.join(q['base_data_dir'], 'cartographic', 'ee', 'ee_r250.gpkg')
    rows = []
    for iso3, group in expected.groupby('ISO3'):
        n, valued = country_cells(band, transform, crs, boundary, iso3)
        for r in group.itertuples():
            a = author.loc[(r.component, iso3, r.crop)]
            protected = epr.protected_bound(r.delta_protected_production, r.largest_abs_delta_share, a['protected_production_tons'], n)
            common = {'component': r.component, 'ISO3': iso3, 'crop': r.crop, 'country_cells': n, 'valued_cells_at_least': valued}
            rows.append(dict(common, column='protected_production_tons', predicted_delta=r.delta_protected_production, bound=protected, author_value=a['protected_production_tons']))
            rows.append(dict(common, column='share_protected_production', predicted_delta=r.delta_protected_production / a['total_production_tons'],
                             bound=epr.share_bound(protected, a['total_production_tons'], a['share_protected_production']), author_value=a['share_protected_production']))
        total_step = float(steps[moved['iso3_library_boundary'] == iso3].sum())
        mean = float(diagnostics.loc[iso3, epr.DIAGNOSTIC_VALUE_COLUMN])
        rows.append({'component': '', 'ISO3': iso3, 'crop': '', 'column': epr.DIAGNOSTIC_VALUE_COLUMN, 'country_cells': n, 'valued_cells_at_least': valued, 'predicted_delta': total_step / valued,
                     'bound': epr.mean_bound(total_step, valued, n, mean), 'author_value': mean})
    frame = pd.DataFrame(rows, columns=['component', 'ISO3', 'crop', 'column', 'predicted_delta', 'bound', 'author_value', 'country_cells', 'valued_cells_at_least'])
    frame.to_csv(OUT, index=False, encoding='utf-8-sig', float_format='%.17g')
    print('%d bounds over %d table rows, countries %s, %d cells in the country, %d valued' % (len(frame), len(expected), ' '.join(sorted(expected['ISO3'].unique())), n, valued))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except eg.Mismatch as e:
        print('STOP: %s' % e, file=sys.stderr)
        sys.exit(2)

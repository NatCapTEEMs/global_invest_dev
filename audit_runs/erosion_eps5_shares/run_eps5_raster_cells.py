"""Locate every cell where a staged SDR raster differs from the author's, and ask what it can reach. Takes no arguments.

For a login node. Read only outside this folder. A share is nonzero only on cropland where the
resampled soil loss exceeds the country's threshold, so a differing cell can change the EPS-5 shares
only if its footprint holds SPAM area and soil loss above the lowest threshold there. The footprint is
the cell and its eight neighbours, wider than the average resampling reaches. Writes
preflight/sdr_cells_that_differ.csv and preflight/sdr_cells_summary.txt. Refuses if either exists.

Run: python -I -B run_eps5_raster_cells.py
"""
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform, transform_bounds
from rasterio.windows import Window
from shapely.geometry import box

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'preflight')
CELLS = os.path.join(OUT, 'sdr_cells_that_differ.csv')
SUMMARY = os.path.join(OUT, 'sdr_cells_summary.txt')
sys.path.insert(0, os.path.join(HERE, 'launch'))
import eps5_compare as ec
import eps5_gates as eg

STAGED = {'usle': 'usle_fresh.tif', 'avoided': 'avoided_erosion_fresh.tif', 'upstream': 'upstream_prevention_share_fresh.tif'}
SDR_REF_DIR = os.path.join('global_invest', 'erosion', 'author_repo_2026_10', 'full_chain_618f67e')
ISO3_COLUMNS = ['iso3_r250_label', 'iso3', 'ISO3', 'iso_a3', 'adm0_a3', 'ADM0_A3']
BLOCK_ROWS = 512


def differing_cells(left_path, right_path):
    """Every (row, col) whose valid mask or stored value differs, read in strips."""
    found = []
    with rasterio.open(left_path) as a, rasterio.open(right_path) as b:
        for row in range(0, a.height, BLOCK_ROWS):
            window = Window(0, row, a.width, min(BLOCK_ROWS, a.height - row))
            x, y = a.read(1, window=window).astype('float64'), b.read(1, window=window).astype('float64')
            vx, vy = ec.valid_cells(x, a.nodata), ec.valid_cells(y, b.nodata)
            rows, cols = np.nonzero((vx != vy) | (vx & vy & (x != y)))
            found += [(int(r) + row, int(c)) for r, c in zip(rows, cols)]
    return found


def footprint_window(dataset, row, col):
    """The cell and its eight neighbours, clamped to the raster."""
    r0, c0 = max(row - 1, 0), max(col - 1, 0)
    return Window(c0, r0, min(col + 2, dataset.width) - c0, min(row + 2, dataset.height) - r0)


def window_in(dataset, bounds):
    """The cells of a raster that a bounding box in its own CRS touches, or None outside it.

    From the four corners, so a south-up raster or a box across the antimeridian widens the window rather than failing.
    """
    corners = [~dataset.transform * (x, y) for x in (bounds[0], bounds[2]) for y in (bounds[1], bounds[3])]
    cols, rows = [c for c, _ in corners], [r for _, r in corners]
    c0, r0 = max(int(np.floor(min(cols))), 0), max(int(np.floor(min(rows))), 0)
    c1, r1 = min(int(np.ceil(max(cols))), dataset.width), min(int(np.ceil(max(rows))), dataset.height)
    return Window(c0, r0, c1 - c0, r1 - r0) if c1 > c0 and r1 > r0 else None


def read_boundary(path, crs):
    gdf = gpd.read_file(path)
    column = next(c for c in ISO3_COLUMNS if c in gdf.columns)
    gdf = gdf[[column, 'geometry']].rename(columns={column: 'ISO3'}).to_crs(crs)
    gdf['geometry'] = gdf.geometry.make_valid()
    return gdf


def main():
    for path in (CELLS, SUMMARY):
        if os.path.exists(path):
            raise eg.Mismatch('%s already exists: move it aside first' % path)
    os.makedirs(OUT, exist_ok=True)
    q = eg.read_parameters(os.path.join(HERE, 'msi_parameters.csv'))
    pins = eg.read_rows(os.path.join(HERE, 'author_pins.csv'))
    role = {r['role']: os.path.join(q['author_archive_dir'], r['relative_path']) for r in pins}
    base = q['base_data_dir']
    left = {name: os.path.join(base, SDR_REF_DIR, file_name) for name, file_name in STAGED.items()}
    right = {name: role[name] for name in STAGED}

    by_cell = {}
    counts = {}
    for name in STAGED:
        cells = differing_cells(left[name], right[name])
        counts[name] = len(cells)
        for cell in cells:
            by_cell.setdefault(cell, []).append(name)

    policy = {name: pd.read_csv(role[name + '_threshold_policy'], encoding='utf-8-sig').set_index('ISO3')['threshold_t_ha_yr'] for name in ('eps5', 'ses11')}
    default_high = {name: float(policy[name].max()) for name in policy}

    opened = {name: (rasterio.open(left[name]), rasterio.open(right[name])) for name in STAGED}
    grid = opened['usle'][0]
    author_boundary = read_boundary(role['boundary'], grid.crs)
    library_boundary = read_boundary(os.path.join(base, 'cartographic', 'ee', 'ee_r250.gpkg'), grid.crs)
    ps_paths = {'eps5_%s' % part: role['eps5_ps_%s' % part] for part in ('onfarm', 'upstream')}
    ps_paths['eps5_combined'] = role['eps5_ps_combined']
    ses11_dir = os.path.join(q['author_archive_dir'], 'results', 'ses11_headline')
    ps_paths.update({'ses11_onfarm': os.path.join(ses11_dir, 'ps_onfarm_cropland_severe.tif'), 'ses11_upstream': os.path.join(ses11_dir, 'ps_upstream_cropland_severe.tif'),
                     'ses11_combined': os.path.join(ses11_dir, 'ps_combined_union_cropland_severe.tif')})
    ps_rasters = {name: rasterio.open(path) for name, path in ps_paths.items()}
    area_stack = rasterio.open(role['area_stack'])

    rows = []
    for (row, col), differs_in in sorted(by_cell.items()):
        record = {'row': row, 'col': col, 'differs_in': ' '.join(differs_in)}
        x, y = grid.xy(row, col)
        lon, lat = transform(grid.crs, 'EPSG:4326', [x], [y])
        record.update(x=x, y=y, lon=lon[0], lat=lat[0])
        window = footprint_window(grid, row, col)
        bounds = grid.window_bounds(window)
        largest_usle = 0.0
        for name, (a, b) in opened.items():
            for side, dataset in (('library', a), ('author', b)):
                values = dataset.read(1, window=Window(col, row, 1, 1)).astype('float64')
                valid = ec.valid_cells(values, dataset.nodata)
                record['%s_%s' % (name, side)] = float(values[0, 0]) if valid[0, 0] else np.nan
                if name == 'usle':
                    around = dataset.read(1, window=window).astype('float64')
                    around = around[ec.valid_cells(around, dataset.nodata)]
                    largest_usle = max(largest_usle, float(around.max()) if around.size else 0.0)
        record['usle_largest_in_footprint'] = largest_usle

        centre = gpd.GeoSeries(gpd.points_from_xy([x], [y]), crs=grid.crs)
        for side, boundary in (('author', author_boundary), ('library', library_boundary)):
            record['iso3_%s_boundary' % side] = ' '.join(sorted(boundary.loc[boundary.geometry.contains(centre.iloc[0]), 'ISO3'].astype(str)))
        touching = sorted(set(author_boundary.loc[author_boundary.geometry.intersects(box(*bounds)), 'ISO3'].astype(str)) | set(library_boundary.loc[library_boundary.geometry.intersects(box(*bounds)), 'ISO3'].astype(str)))
        record['iso3_in_footprint'] = ' '.join(touching)
        for name in policy:
            known = [float(policy[name][iso3]) for iso3 in touching if iso3 in policy[name].index]
            record['lowest_threshold_%s' % name] = min(known + [default_high[name]])

        stack_window = window_in(area_stack, transform_bounds(grid.crs, area_stack.crs, *bounds, densify_pts=21))
        spam = 0.0
        if stack_window is not None:
            cube = area_stack.read(window=stack_window).astype('float64')
            spam = float(np.nansum(np.clip(cube, 0.0, None)[np.isfinite(cube) & (cube != area_stack.nodata if area_stack.nodata is not None else True)]))
        record['spam_area_ha_in_footprint'] = spam

        for name, dataset in ps_rasters.items():
            ps_window = window_in(dataset, transform_bounds(grid.crs, dataset.crs, *bounds, densify_pts=21))
            values = dataset.read(1, window=ps_window) if ps_window is not None else np.zeros((0, 0))
            record['author_%s_nonzero_cells' % name] = int(np.count_nonzero(np.nan_to_num(values)))
        record['target_cells_in_footprint'] = int(values.size)

        record['has_cropland'] = spam > 0
        record['above_eps5_threshold'] = largest_usle > record['lowest_threshold_eps5']
        record['above_ses11_threshold'] = largest_usle > record['lowest_threshold_ses11']
        record['can_reach_eps5_shares'] = record['has_cropland'] and record['above_eps5_threshold']
        record['can_reach_ses11_shares'] = record['has_cropland'] and record['above_ses11_threshold']
        rows.append(record)

    frame = pd.DataFrame(rows)
    frame.to_csv(CELLS, index=False, encoding='utf-8-sig', float_format='%.17g')
    mask_rows = frame['usle_library'].notna() != frame['usle_author'].notna()
    lines = ['cells that differ by raster: %s' % ', '.join('%s %d' % item for item in counts.items()),
             'distinct cells: %d, of which usle valid only on one side: %d' % (len(frame), int(mask_rows.sum())),
             'cells with SPAM area in the footprint: %d' % int(frame['has_cropland'].sum()),
             'cells with soil loss above the lowest EPS-5 threshold in the footprint: %d' % int(frame['above_eps5_threshold'].sum()),
             'cells that can reach the EPS-5 shares: %d' % int(frame['can_reach_eps5_shares'].sum()),
             'cells that can reach the SES-11 shares: %d' % int(frame['can_reach_ses11_shares'].sum()),
             'cells whose footprint holds a nonzero author EPS-5 share: %d' % int((frame[['author_eps5_onfarm_nonzero_cells', 'author_eps5_upstream_nonzero_cells', 'author_eps5_combined_nonzero_cells']].sum(axis=1) > 0).sum()),
             'cells whose footprint holds a nonzero author SES-11 share: %d' % int((frame[['author_ses11_onfarm_nonzero_cells', 'author_ses11_upstream_nonzero_cells', 'author_ses11_combined_nonzero_cells']].sum(axis=1) > 0).sum()),
             'countries at cell centres, author boundary: %s' % ' '.join(sorted(set(' '.join(frame['iso3_author_boundary']).split()))),
             'countries at cell centres, library boundary: %s' % ' '.join(sorted(set(' '.join(frame['iso3_library_boundary']).split())))]
    with open(SUMMARY, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except eg.Mismatch as e:
        print('STOP: %s' % e, file=sys.stderr)
        sys.exit(2)

"""Carry the staged and the author's SDR rasters through the task's own resampling, and predict every share that differs. Takes no arguments.

For an MSI login node, one core, read only outside this folder. A login session is killed near 6 GB, so
each grid is resampled in its own process and held on disk under preflight/tmp/reach, which is removed at
the end. Finds the analysis-grid cells where the two sets differ and predicts the three shares there at
SES-11 and EPS-5. Outside those cells the inputs are equal, so the shares are too. Two falsifiers come
first: the author's rasters must reproduce the author's stored shares at those cells, and the staged
rasters must reproduce the SES-11 shares, which are byte identical between the library and the author.
It then asks which cells take another country between the boundary files and whether a threshold or a
share there can move. Writes preflight/sdr_reach_*.csv, preflight/boundary_reach.csv and
preflight/sdr_reach_summary.txt. Refuses if the summary exists.

Run: python -I -B run_eps5_raster_reach.py
"""
import gc
import multiprocessing
import os
import resource
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
import xarray as xr
from affine import Affine
from rasterio.crs import CRS

HERE = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT = os.path.join(HERE, 'code', 'global_invest_dev')
OUT = os.path.join(HERE, 'preflight')
WORK = os.path.join(OUT, 'tmp', 'reach')
SUMMARY = os.path.join(OUT, 'sdr_reach_summary.txt')
sys.path.insert(0, os.path.join(HERE, 'launch'))
sys.path.insert(0, SNAPSHOT)
import eps5_gates as eg
from global_invest.erosion import erosion_functions as ef
from global_invest.erosion import erosion_tasks as et

SDR_REF_DIR = os.path.join('global_invest', 'erosion', 'author_repo_2026_10', 'full_chain_618f67e')
PARTS = {'onfarm': 'ps_onfarm_cropland_severe.tif', 'upstream': 'ps_upstream_cropland_severe.tif', 'combined': 'ps_combined_union_cropland_severe.tif'}
RUN_DIRS = {'eps5': os.path.join('results', 'eps5_sensitivity'), 'ses11': os.path.join('results', 'ses11_headline')}
LAYERS = ('usle', 'avoided', 'upstream')
CLUSTER_REACH = 20
STRIP_ROWS = 512


def resample_layer(layer, paths, epsg, out_path):
    """The lines of the task from one native raster to the array it thresholds, with the soil loss grid as its template."""
    crs = CRS.from_epsg(epsg)
    usle = ef.reproject_to_analysis_grid(et.open_raster_1band(paths['usle']), crs, ef.RESAMPLE_USLE_AVOID)
    if layer == 'usle':
        values = ef._clean_nonneg(usle).values
    elif layer == 'avoided':
        avo = ef.reproject_to_analysis_grid(et.open_raster_1band(paths['avoided']), crs, ef.RESAMPLE_USLE_AVOID)
        values = ef._clean_nonneg(avo.rio.reproject_match(usle, resampling=ef.RESAMPLE_USLE_AVOID)).values
    else:
        ups = ef.reproject_to_analysis_grid(et.open_raster_1band(paths['upstream']), crs, ef.RESAMPLE_UPS)
        values = ef._clip01_arr(ups.rio.reproject_match(usle, resampling=ef.RESAMPLE_UPS).values)
    np.save(out_path, values)
    return {'x': usle['x'].values, 'y': usle['y'].values, 'transform': tuple(usle.rio.transform())[:6], 'crs': usle.rio.crs.to_wkt()}


def rasterize_boundary(path, grid, out_path):
    """The task's country raster for one boundary file, held on disk."""
    crs = CRS.from_wkt(grid['crs'])
    like = SimpleNamespace(rio=SimpleNamespace(crs=crs, height=grid['y'].size, width=grid['x'].size, transform=lambda: Affine(*grid['transform'])))
    p = SimpleNamespace(erosion_rasterize_all_touched=False, erosion_boundary_source_epsg=None)
    paths = SimpleNamespace(input=SimpleNamespace(country_boundary=path))
    raster, lut = et.rasterize_iso3(p, et.load_countries_iso3_alpha(p, paths, crs), like)
    np.save(out_path, raster)
    return lut


def in_own_process(function, *args):
    """Memory goes back to the system when the process ends, which a login node needs."""
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
        return pool.submit(function, *args).result()


def differs(a, b):
    return ~((a == b) | (np.isnan(a) & np.isnan(b)))


def threshold_lookup(lut, policy_path):
    """threshold_by_id of the task: the policy by country id, the default high elsewhere."""
    policy = pd.read_csv(policy_path, encoding='utf-8-sig').set_index('ISO3')['threshold_t_ha_yr']
    by_id = np.full(int(lut['iso_id'].max()) + 1, float(policy.max()), dtype='float32')
    for iso3, iso_id in zip(lut['ISO3'], lut['iso_id']):
        if iso3 in policy.index:
            by_id[int(iso_id)] = policy[iso3]
    return by_id


def shares(usle, avoided, upstream, threshold, cropland):
    severe = usle > threshold
    onfarm = ef.restrict_to_valued_pixels(ef.onfarm_prevention_share(avoided, usle), cropland, severe).astype('float32')
    up = ef.restrict_to_valued_pixels(upstream, cropland, severe).astype('float32')
    return {'onfarm': onfarm, 'upstream': up, 'combined': ef.combined_prevention_share(onfarm, up).astype('float32')}


def stacks_at(rows, cols, grid, area_path, yield_path, bandmap):
    """Harvested area and yield at the cells, per crop, warped band by band onto a window of the grid around each cluster."""
    boxes = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        for b in boxes:
            if b[0] - CLUSTER_REACH <= r <= b[1] + CLUSTER_REACH and b[2] - CLUSTER_REACH <= c <= b[3] + CLUSTER_REACH:
                b[0], b[1], b[2], b[3] = min(b[0], r), max(b[1], r), min(b[2], c), max(b[3], c)
                break
        else:
            boxes.append([r, r, c, c])
    crs = CRS.from_wkt(grid['crs'])
    windows = []
    for r_min, r_max, c_min, c_max in boxes:
        here = (rows >= r_min) & (rows <= r_max) & (cols >= c_min) & (cols <= c_max)
        r0, c0 = max(r_min - 2, 0), max(c_min - 2, 0)
        y, x = grid['y'][r0:r_max + 3], grid['x'][c0:c_max + 3]
        template = xr.DataArray(np.zeros((y.size, x.size), dtype='float32'), coords={'y': y, 'x': x}, dims=('y', 'x')).rio.write_crs(crs)
        windows.append((here, r0, c0, template))
    area = np.zeros((len(bandmap), rows.size), dtype='float64')
    yields = np.zeros_like(area)
    for k, band in enumerate(bandmap['band']):
        a_pos = rxr.open_rasterio(area_path, masked=True).sel(band=int(band)).squeeze().clip(min=0.0).fillna(0.0)
        y_native = rxr.open_rasterio(yield_path, masked=True).sel(band=int(band)).squeeze()
        for here, r0, c0, template in windows:
            ha = a_pos.rio.reproject_match(template, resampling=ef.RESAMPLE_AREA).fillna(0.0).values
            y = y_native.rio.reproject_match(template, resampling=ef.RESAMPLE_YIELD).fillna(0.0).values
            area[k, here] = ha[rows[here] - r0, cols[here] - c0]
            yields[k, here] = y[rows[here] - r0, cols[here] - c0]
    return area, yields, len(boxes)


def main():
    if os.path.exists(SUMMARY):
        raise eg.Mismatch('%s already exists: move it aside first' % SUMMARY)
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    os.makedirs(WORK)
    started = time.time()
    q = eg.read_parameters(os.path.join(HERE, 'msi_parameters.csv'))
    settings = {r['setting']: r['value'] for r in eg.read_rows(os.path.join(HERE, 'eps5_settings.csv'))}
    pins = eg.read_rows(os.path.join(HERE, 'author_pins.csv'))
    base, archive = q['base_data_dir'], q['author_archive_dir']
    role = {r['role']: os.path.join(archive, r['relative_path']) for r in pins}
    epsg = int(settings['analysis_epsg'])
    lowest = float(settings['threshold_low'])
    staged = dict(zip(LAYERS, [os.path.join(base, SDR_REF_DIR, n) for n in ('usle_fresh.tif', 'avoided_erosion_fresh.tif', 'upstream_prevention_share_fresh.tif')]))
    author = {layer: role[layer] for layer in LAYERS}
    lines = []

    grid = None
    for side, paths in (('library', staged), ('author', author)):
        for layer in LAYERS:
            info = in_own_process(resample_layer, layer, paths, epsg, os.path.join(WORK, '%s_%s.npy' % (layer, side)))
            if grid is None:
                grid = info
            elif tuple(info['transform']) != tuple(grid['transform']) or info['x'].size != grid['x'].size or info['y'].size != grid['y'].size:
                raise eg.Mismatch('%s %s resampled to another grid' % (side, layer))
    with rasterio.open(role['eps5_ps_onfarm']) as d:
        if (d.height, d.width) != (grid['y'].size, grid['x'].size) or d.crs != CRS.from_wkt(grid['crs']) or not d.transform.almost_equals(Affine(*grid['transform']), precision=1e-6):
            raise eg.Mismatch('the resampled grid is not the grid of the author shares')
    lines.append('analysis grid %d x %d, EPSG %d, equal to the grid of the author shares, on both sets of rasters' % (grid['x'].size, grid['y'].size, epsg))

    grids = {layer: (np.load(os.path.join(WORK, '%s_library.npy' % layer), mmap_mode='r'), np.load(os.path.join(WORK, '%s_author.npy' % layer), mmap_mode='r')) for layer in LAYERS}
    counts = dict.fromkeys(LAYERS, 0)
    one_sided = dict.fromkeys(LAYERS, 0)
    any_count, largest_one_sided, one_sided_above = 0, 0.0, 0
    found_rows, found_cols = [], []
    for row in range(0, grid['y'].size, STRIP_ROWS):
        strip = slice(row, row + STRIP_ROWS)
        hit = None
        for layer in LAYERS:
            left, right = np.asarray(grids[layer][0][strip]), np.asarray(grids[layer][1][strip])
            d = differs(left, right)
            counts[layer] += int(d.sum())
            lonely = np.isnan(left) != np.isnan(right)
            one_sided[layer] += int(lonely.sum())
            hit = d if hit is None else hit | d
            if layer == 'usle':
                usle_top = np.maximum(np.nan_to_num(left), np.nan_to_num(right))
                if lonely.any():
                    largest_one_sided = max(largest_one_sided, float(usle_top[lonely].max()))
                    one_sided_above += int((usle_top[lonely] > lowest).sum())
        any_count += int(hit.sum())
        r, c = np.nonzero(hit & (usle_top > lowest))
        found_rows.append(r + row)
        found_cols.append(c)
    rows, cols = np.concatenate(found_rows), np.concatenate(found_cols)
    lines.append('analysis cells that differ: %s, any %d' % (', '.join('%s %d' % item for item in counts.items()), any_count))
    lines.append('analysis cells valid on one side only: %s. Largest soil loss among the usle ones %.6g, above %g: %d' % (', '.join('%s %d' % item for item in one_sided.items()), largest_one_sided, lowest, one_sided_above))
    lines.append('of the differing cells, soil loss above %g, the lowest threshold of the policy: %d. Every other differing cell has a share of zero on both sides' % (lowest, rows.size))
    at_cells = {'%s_%s' % (layer, side): np.asarray(grids[layer][k][rows, cols]) for layer in LAYERS for k, side in enumerate(('library', 'author'))}

    lut_l = in_own_process(rasterize_boundary, os.path.join(base, 'cartographic', 'ee', 'ee_r250.gpkg'), grid, os.path.join(WORK, 'iso_library.npy'))
    lut_a = in_own_process(rasterize_boundary, role['boundary'], grid, os.path.join(WORK, 'iso_author.npy'))
    iso_l, iso_a = np.load(os.path.join(WORK, 'iso_library.npy'), mmap_mode='r'), np.load(os.path.join(WORK, 'iso_author.npy'), mmap_mode='r')
    name_l, name_a = dict(zip(lut_l['iso_id'].astype(int), lut_l['ISO3'])), dict(zip(lut_a['iso_id'].astype(int), lut_a['ISO3']))
    policies = {'eps5': (os.path.join(OUT, 'library_threshold_policy_predicted_at_eps5.csv'), role['eps5_threshold_policy']),
                'ses11': (os.path.join(q['library_ses11_dir'], 'threshold_policy.csv'), role['ses11_threshold_policy'])}
    by_id = {run: (threshold_lookup(lut_l, left), threshold_lookup(lut_a, right)) for run, (left, right) in policies.items()}

    stored = {}
    for run in RUN_DIRS:
        for part, file_name in PARTS.items():
            with rasterio.open(os.path.join(archive, RUN_DIRS[run], file_name)) as d:
                stored[run, part] = d.read(1)[rows, cols]
            gc.collect()

    frame = pd.DataFrame(eg.read_rows(role['bandmap']))
    frame.columns = [c.strip().lower() for c in frame.columns]
    area, yields, clusters = stacks_at(rows, cols, grid, role['area_stack'], role['yield_stack'], frame)
    cropland = (area > 0).any(axis=0)
    lines.append('those cells fall in %d clusters, on cropland in %d' % (clusters, int(cropland.sum())))

    iso_at_l, iso_at_a = np.asarray(iso_l[rows, cols]), np.asarray(iso_a[rows, cols])
    table = pd.DataFrame({'row': rows, 'col': cols, 'iso3_library_boundary': [name_l.get(int(i), '') for i in iso_at_l], 'iso3_author_boundary': [name_a.get(int(i), '') for i in iso_at_a], **at_cells, 'cropland': cropland})
    held, tons = [], []
    for run in RUN_DIRS:
        threshold_l, threshold_a = by_id[run][0][iso_at_l], by_id[run][1][iso_at_a]
        table['%s_threshold_library_boundary' % run] = threshold_l
        library = shares(at_cells['usle_library'], at_cells['avoided_library'], at_cells['upstream_library'], threshold_l, cropland)
        authors = shares(at_cells['usle_author'], at_cells['avoided_author'], at_cells['upstream_author'], threshold_a, cropland)
        for part in PARTS:
            table['%s_%s_predicted_library' % (run, part)] = library[part]
            table['%s_%s_stored_author' % (run, part)] = stored[run, part]
            missed = int((authors[part] != stored[run, part]).sum())
            moved = library[part] != stored[run, part]
            gap = np.abs(library[part].astype('float64') - stored[run, part].astype('float64'))
            lines.append('%s %s: author rasters miss the stored author share at %d cells, staged rasters differ from it at %d cells, largest %.3g' % (run, part, missed, int(moved.sum()), float(gap.max()) if gap.size else 0.0))
            held.append(missed == 0 and (run != 'ses11' or not moved.any()))
            delta = (library[part].astype('float64') - stored[run, part].astype('float64'))[moved]
            production = (area * yields)[:, moved]
            for k, crop in enumerate(frame['crop']):
                for iso3, d, prod in zip(table.loc[moved, 'iso3_library_boundary'], delta, production[k]):
                    if prod > 0:
                        tons.append({'run': run, 'component': part, 'ISO3': iso3, 'crop': crop, 'delta_share': d, 'delta_protected_production': d * prod})
    lines.insert(0, 'falsifiers: %s' % ('HOLD, the prediction stands' if all(held) else 'FAIL, the prediction below does not stand'))
    table.to_csv(os.path.join(OUT, 'sdr_reach_cells.csv'), index=False, encoding='utf-8-sig', float_format='%.17g')
    expected = pd.DataFrame(tons, columns=['run', 'component', 'ISO3', 'crop', 'delta_share', 'delta_protected_production'])
    grouped = expected.groupby(['run', 'component', 'ISO3', 'crop'], as_index=False).agg(cells=('delta_share', 'size'), delta_protected_production=('delta_protected_production', 'sum'), largest_abs_delta_share=('delta_share', lambda s: float(np.abs(s).max())))
    grouped.to_csv(os.path.join(OUT, 'sdr_reach_expected_differences.csv'), index=False, encoding='utf-8-sig', float_format='%.17g')
    for run in RUN_DIRS:
        part_rows = grouped[grouped['run'] == run]
        lines.append('predicted differences in protected production, %s: %d country-crop-component rows, countries %s, largest %.3g in the units of the stacks' % (run, len(part_rows), ' '.join(sorted(part_rows['ISO3'].unique())) or 'none', float(part_rows['delta_protected_production'].abs().max()) if len(part_rows) else 0.0))

    names = sorted(set(name_l.values()) | set(name_a.values()))
    code = {name: k + 1 for k, name in enumerate(names)}
    common_l = np.array([0] + [code[name_l[i]] for i in sorted(name_l)], dtype='int16')
    common_a = np.array([0] + [code[name_a[i]] for i in sorted(name_a)], dtype='int16')
    found_rows, found_cols = [], []
    for row in range(0, grid['y'].size, STRIP_ROWS):
        strip = slice(row, row + STRIP_ROWS)
        r, c = np.nonzero(common_l[np.asarray(iso_l[strip])] != common_a[np.asarray(iso_a[strip])])
        found_rows.append(r + row)
        found_cols.append(c)
    rr, cc = np.concatenate(found_rows), np.concatenate(found_cols)
    lines.append('cells that take another country between the boundary files: %d of %d' % (rr.size, grid['x'].size * grid['y'].size))
    label = np.array(['none'] + names, dtype=object)
    iso_rr_l, iso_rr_a = np.asarray(iso_l[rr, cc]), np.asarray(iso_a[rr, cc])
    boundary = pd.DataFrame({'iso3_library': label[common_l[iso_rr_l]], 'iso3_author': label[common_a[iso_rr_a]], 'threshold_eps5_library': by_id['eps5'][0][iso_rr_l], 'threshold_eps5_author': by_id['eps5'][1][iso_rr_a],
                             'threshold_ses11_library': by_id['ses11'][0][iso_rr_l], 'threshold_ses11_author': by_id['ses11'][1][iso_rr_a], 'usle': np.asarray(grids['usle'][1][rr, cc])})
    with rasterio.open(role['eps5_ps_combined']) as d:
        boundary['author_eps5_share_nonzero'] = np.nan_to_num(d.read(1)[rr, cc]) != 0
    boundary['above_lower_eps5_threshold'] = np.nan_to_num(boundary['usle']) > boundary[['threshold_eps5_library', 'threshold_eps5_author']].min(axis=1)
    boundary['thresholds_differ_eps5'] = boundary['threshold_eps5_library'] != boundary['threshold_eps5_author']
    keys = ['iso3_library', 'iso3_author', 'threshold_eps5_library', 'threshold_eps5_author', 'threshold_ses11_library', 'threshold_ses11_author']
    pairs = boundary.groupby(keys, as_index=False).agg(cells=('usle', 'size'), above_lower_eps5_threshold=('above_lower_eps5_threshold', 'sum'), author_eps5_share_nonzero=('author_eps5_share_nonzero', 'sum'))
    pairs.to_csv(os.path.join(OUT, 'boundary_reach.csv'), index=False, encoding='utf-8-sig')
    lines.append('of those: threshold differs at EPS-5 %d, soil loss above the lower EPS-5 threshold %d, nonzero author EPS-5 share %d, country pairs %d' % (int(boundary['thresholds_differ_eps5'].sum()), int(boundary['above_lower_eps5_threshold'].sum()), int(boundary['author_eps5_share_nonzero'].sum()), len(pairs)))
    lines.append('elapsed %.0f s, peak resident memory of the main process %.1f GB' % (time.time() - started, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))
    with open(SUMMARY, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    # Open memory maps leave .nfs files on scratch until the process ends, so a remainder goes then.
    shutil.rmtree(WORK, ignore_errors=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except eg.Mismatch as e:
        print('STOP: %s' % e, file=sys.stderr)
        sys.exit(2)

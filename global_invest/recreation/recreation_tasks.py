"""Recreation/tourism GEP tasks: site quality -> banded travel-cost flows -> three channels.

The method is the author's zonal travel-cost rewrite; tasks publish their own inputs
(publish_inputs), read data references from es_parameters.csv, and aggregate DIRECTLY on r250
(the country-id raster is iso3_r250_id, so the aggregation surface and the one-row-per-country
collapse coincide).

The flow engine windows by COUNTRY where the source windows by parameter group. The maths is
pixel-local within the fixed 50 km cutoff and the country ids partition every group's pixels,
so the two give the same flows (convolution is linear over the partition); a test pins that on
a shared toy grid. What the change buys: a worldwide archetype group's window is the whole
globe, a country's window is the country plus a 50 km buffer. Convolutions run in memory
(scipy fftconvolve) on those windows instead of through file-based tiles. The source also
computes a global distance-to-site transform its engine never reads; that step is not ported.

Data staging map (drive Recreation/data/ tree -> base_data/global_invest/recreation/):
0_inputs/* -> the module root; 0_processed_raster_inputs/<sub>/* -> <sub>/*. File names are
kept EXACTLY as shipped (including the fuel-cost CSV's spelling) so staging is a copy, not a
rename layer. The airfare matrix is the author's own staged scrape output; the scraper stays
in his repository, so this module makes no network request.
"""
import os

import numpy as np
import pandas as pd
import pygeoprocessing
from osgeo import gdal, osr
from scipy import ndimage, signal

import hazelbean as hb
from global_invest import utilities
from global_invest.recreation import recreation_functions as rf


# --- Raster-chain steps (site quality) ---
def calculate_environment_index(lulc_share_paths, pa_share_path, output_path):
    """LULC class-share rasters + PA share -> environment class raster (0-3)."""
    def op(crop, forest, grass, othernat, urban, water, pa):
        return rf.environment_class_array(crop, forest, grass, othernat, urban, water, pa)
    base_list = [(lulc_share_paths[c], 1) for c in rf.RECREATION_LULC_CLASSES] + [(pa_share_path, 1)]
    pygeoprocessing.raster_calculator(base_list, op, output_path, gdal.GDT_Int32, rf.INT_NDV,
                                      calc_raster_stats=True)


def calculate_accessibility_index(urban_share_path, road_distance_path, output_path):
    """Urban share + distance-to-road rasters -> accessibility class raster (1-5)."""
    pygeoprocessing.raster_calculator(
        [(urban_share_path, 1), (road_distance_path, 1)], rf.accessibility_class_array,
        output_path, gdal.GDT_Int32, rf.INT_NDV)


def rank_recreation_sites(accessibility_index_path, environment_index_path, output_path):
    pygeoprocessing.raster_calculator(
        [(accessibility_index_path, 1), (environment_index_path, 1)], rf.site_rank_array,
        output_path, gdal.GDT_Int32, rf.INT_NDV)


def extract_high_quality_sites(ranked_sites_path, output_path):
    def op(site_class):
        return np.where(site_class == rf.RECREATION_HQ_SITE_CLASS, 1, 0).astype(np.int32)
    pygeoprocessing.raster_calculator([(ranked_sites_path, 1)], op, output_path,
                                      gdal.GDT_Int32, rf.INT_NDV)


def dissolve_high_quality_sites(hq_sites_path, output_path, site_sizes_path):
    """Contiguous high-quality pixels labelled into discrete sites (8-connected), so a large
    park is one destination rather than many independent site pixels. Writes the site-id
    raster (0 = not a site, 1..N = site id) and a per-site pixel-count table."""
    hq_nodata = pygeoprocessing.get_raster_info(hq_sites_path)['nodata'][0]
    ds = gdal.Open(hq_sites_path)
    hq_array = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    is_hq = (hq_array == 1) if hq_nodata is None else ((hq_array != hq_nodata) & (hq_array == 1))
    labeled, n_sites = ndimage.label(is_hq, structure=np.ones((3, 3)))
    hb.log('recreation: %d discrete high-quality sites (%d HQ pixels, %.1f px/site mean)'
           % (n_sites, int(is_hq.sum()), is_hq.sum() / max(n_sites, 1)))

    pygeoprocessing.new_raster_from_base(hq_sites_path, output_path, gdal.GDT_Int32, [0])
    out_ds = gdal.Open(output_path, gdal.GA_Update)
    out_ds.GetRasterBand(1).WriteArray(labeled.astype(np.int32))
    out_ds.GetRasterBand(1).FlushCache()
    out_ds = None

    site_ids, pixel_counts = np.unique(labeled[labeled > 0], return_counts=True)
    pd.DataFrame({'site_id': site_ids, 'pixel_count': pixel_counts}).to_csv(
        site_sizes_path, index=False)


# --- Overnights ---
def rasterize_presence(vector_path, ref_raster_path, output_path):
    """Vector features -> additive presence counts on the reference grid (hotel points)."""
    ref_info = pygeoprocessing.geoprocessing.get_raster_info(ref_raster_path)
    pygeoprocessing.geoprocessing.create_raster_from_bounding_box(
        target_bounding_box=ref_info['bounding_box'], target_raster_path=output_path,
        target_pixel_size=ref_info['pixel_size'], target_pixel_type=gdal.GDT_Int16,
        target_srs_wkt=ref_info['projection_wkt'], target_nodata=rf.INT_NDV, fill_value=0)
    pygeoprocessing.geoprocessing.rasterize(
        vector_path=vector_path, target_raster_path=output_path, burn_values=[1],
        option_list=['ALL_TOUCHED=TRUE', 'MERGE_ALG=ADD'])


def count_hotels_by_country(hotel_raster_path, country_id_path):
    """National hotel-pixel counts in one streaming pass, so the allocation's denominator is
    the country's whole hotel stock rather than whatever one raster block holds."""
    totals = {}
    hotel_ds = gdal.Open(hotel_raster_path); hotel_band = hotel_ds.GetRasterBand(1)
    country_ds = gdal.Open(country_id_path); country_band = country_ds.GetRasterBand(1)
    for block_info in pygeoprocessing.iterblocks((hotel_raster_path, 1), offset_only=True):
        xoff, yoff = block_info['xoff'], block_info['yoff']
        wx, wy = block_info['win_xsize'], block_info['win_ysize']
        hotels = hotel_band.ReadAsArray(xoff, yoff, wx, wy)
        countries = country_band.ReadAsArray(xoff, yoff, wx, wy)
        mask = hotels > 0
        for country_id in np.unique(countries[mask]):
            if country_id > 0:
                totals[int(country_id)] = totals.get(int(country_id), 0.0) + float(
                    hotels[mask & (countries == country_id)].sum())
    hotel_band = None; hotel_ds = None
    country_band = None; country_ds = None
    return totals


def allocate_overnights_raster(country_overnights_map, hotel_raster_path, country_id_path,
                               output_path):
    country_hotel_totals = count_hotels_by_country(hotel_raster_path, country_id_path)
    pygeoprocessing.raster_calculator(
        [(hotel_raster_path, 1), (country_id_path, 1)],
        lambda hotels, countries: rf.allocate_overnights_array(
            hotels, countries, country_overnights_map, country_hotel_totals),
        output_path, gdal.GDT_Float32, rf.FLOAT_NDV)


def read_unwto_sheets(path):
    """The UNWTO all-data workbook's two accommodation sheets, each read from its own header row.

    Args:
        path (str): the UNWTO all-data workbook.

    Returns:
        dict: tourism type -> that type's accommodation sheet, ready for clean_unwto_data.
    """
    sheets = {}
    for tourism_type, sheet_name in rf.UNWTO_ACCOMMODATION_SHEETS.items():
        banner_sheet = pd.read_excel(path, sheet_name=sheet_name)
        sheets[tourism_type] = pd.read_excel(
            path, sheet_name=sheet_name, skiprows=rf.unwto_header_row_index(banner_sheet))
    return sheets


def read_unwto_arrivals_sheet(path):
    """The Inbound Tourism-Regions sheet read from its own header row (same banner rule)."""
    banner_sheet = pd.read_excel(path, sheet_name='Inbound Tourism-Regions')
    return pd.read_excel(path, sheet_name='Inbound Tourism-Regions',
                         skiprows=rf.unwto_header_row_index(banner_sheet))


# --- The flow engine ---
def convert_pixel_size_to_km(raster_path, pixel_size_native):
    """The raster's pixel size in kilometres: geographic rasters take the mean of the
    latitude and the centre-latitude longitude degree lengths, projected ones their linear
    unit, exactly as the source converts."""
    info = pygeoprocessing.get_raster_info(raster_path)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(info['projection_wkt'])
    if srs.IsGeographic():
        bbox = info['bounding_box']
        center_lat = (bbox[1] + bbox[3]) / 2.0
        km_per_degree_lat = 111.32
        km_per_degree_lon = 111.32 * np.cos(np.radians(center_lat))
        return pixel_size_native * (km_per_degree_lat + km_per_degree_lon) / 2.0
    linear_unit_name = srs.GetLinearUnitsName()
    if linear_unit_name and 'kilomet' in linear_unit_name.lower():
        return pixel_size_native
    return pixel_size_native / 1000.0


def assert_rasters_aligned(reference_path, other_paths_by_name, tolerance=1e-6):
    """Raise if any raster leaves the reference's pixel grid: window offsets are reused
    across rasters, and a grid mismatch returns geographically wrong data instead of an
    error."""
    ref_info = pygeoprocessing.get_raster_info(reference_path)
    problems = []
    for name, path in other_paths_by_name.items():
        info = pygeoprocessing.get_raster_info(path)
        if info['raster_size'] != ref_info['raster_size']:
            problems.append(f"{name}: raster_size {info['raster_size']} != {ref_info['raster_size']}")
            continue
        if any(abs(a - b) > tolerance for a, b in zip(info['pixel_size'], ref_info['pixel_size'])):
            problems.append(f"{name}: pixel_size {info['pixel_size']} != {ref_info['pixel_size']}")
        if any(abs(a - b) > tolerance for a, b in zip(info['bounding_box'], ref_info['bounding_box'])):
            problems.append(f"{name}: bounding_box differs from the reference")
    if problems:
        raise ValueError('raster grid mismatch; align to the population grid first:\n'
                         + '\n'.join(problems))


def scan_id_bboxes(id_raster_path, nodata):
    """One streaming pass over an id raster: id -> [row_min, row_max, col_min, col_max]."""
    bboxes = {}
    ds = gdal.Open(id_raster_path)
    band = ds.GetRasterBand(1)
    for block_info in pygeoprocessing.iterblocks((id_raster_path, 1), offset_only=True):
        xoff, yoff = block_info['xoff'], block_info['yoff']
        block = band.ReadAsArray(xoff, yoff, block_info['win_xsize'], block_info['win_ysize'])
        present = np.unique(block[(block != nodata) & (block > 0)])
        for raw_id in present:
            raw_id = int(raw_id)
            rows, cols = np.where(block == raw_id)
            r0, r1 = rows.min() + yoff, rows.max() + yoff
            c0, c1 = cols.min() + xoff, cols.max() + xoff
            if raw_id not in bboxes:
                bboxes[raw_id] = [r0, r1, c0, c1]
            else:
                box = bboxes[raw_id]
                box[0], box[1] = min(box[0], r0), max(box[1], r1)
                box[2], box[3] = min(box[2], c0), max(box[3], c1)
    ds = None
    return bboxes


def accumulate_window(band, xoff, yoff, window_contribution):
    """Read-modify-write add: buffered country windows overlap, so a later country's
    contribution to a shared window adds to, never overwrites, an earlier one's."""
    existing = band.ReadAsArray(xoff, yoff, window_contribution.shape[1],
                                window_contribution.shape[0])
    band.WriteArray(existing + window_contribution, xoff, yoff)


def calculate_recreation_flows(inputs, outputs, country_to_group_map, group_params,
                               max_distance_km, max_bands):
    """The banded flow model over per-country windows; maths as the source per pixel.

    Per population pixel with its country's (a, b): budget = pop * a; each band's share is
    the closed-form CDF difference; a band's fixed budget splits evenly over the sites its
    ring reaches (any country's sites -- the window is buffered by the cutoff); no reachable
    site makes the share unmet; value prices the round trip to the band midpoint at the
    origin country's per-km fuel cost; demand beyond the cutoff is unconditionally unmet.

    inputs: 'hq_sites', 'hq_sites_dissolved', 'population', 'country_id', 'country_costs'.
    outputs: 'visits', 'value', 'unmet' rasters; 'site_table', 'group_table',
             'missing_cost_iso3' CSVs.
    """
    assert_rasters_aligned(inputs['population'], {
        'hq_sites': inputs['hq_sites'], 'country_id': inputs['country_id']})

    raster_info = pygeoprocessing.get_raster_info(inputs['population'])
    px_x, px_y = abs(raster_info['pixel_size'][0]), abs(raster_info['pixel_size'][1])
    if abs(px_x - px_y) > 1e-9 * max(px_x, px_y):
        raise ValueError('population raster has non-square pixels; the ring geometry assumes '
                         'one pixel_size_km for both axes')
    pixel_size_km = convert_pixel_size_to_km(inputs['population'], px_x)
    pop_nodata = raster_info['nodata'][0]
    n_cols, n_rows = raster_info['raster_size']

    cost_df = pd.read_csv(inputs['country_costs'])
    cost_map = dict(zip(cost_df['iso3_r250_id'], cost_df[rf.RECREATION_FUEL_COST_COL]))

    for key in ('visits', 'value', 'unmet'):
        pygeoprocessing.new_raster_from_base(
            inputs['population'], outputs[key], gdal.GDT_Float32, [rf.FLOAT_NDV])
        ds = gdal.Open(outputs[key], gdal.GA_Update)
        ds.GetRasterBand(1).Fill(0)
        ds = None

    country_nodata = pygeoprocessing.get_raster_info(inputs['country_id'])['nodata'][0]
    bboxes = scan_id_bboxes(inputs['country_id'], country_nodata)

    # The cutoff and band edges are the same for every group (b only shapes the split), so
    # the ring kernels are built once.
    cutoff_km, edges = rf.compute_group_bands(pixel_size_km, max_distance_km, max_bands)
    n_bands = len(edges) - 1
    kernels = [rf.ring_kernel_array(edges[k], edges[k + 1], pixel_size_km)
               for k in range(n_bands)]
    buffer_px = int(np.ceil(cutoff_km / pixel_size_km))

    no_params, no_cost = [], []
    group_qa = {}

    hq_nodata = pygeoprocessing.get_raster_info(inputs['hq_sites'])['nodata'][0]
    pop_ds = gdal.Open(inputs['population']); pop_band = pop_ds.GetRasterBand(1)
    country_ds = gdal.Open(inputs['country_id']); country_band = country_ds.GetRasterBand(1)
    hq_ds = gdal.Open(inputs['hq_sites']); hq_band = hq_ds.GetRasterBand(1)
    out_handles = {}
    for key in ('visits', 'value', 'unmet'):
        ds = gdal.Open(outputs[key], gdal.GA_Update)
        out_handles[key] = (ds, ds.GetRasterBand(1))

    for country_id in sorted(bboxes):
        group_id = country_to_group_map.get(country_id)
        params = group_params.get(group_id) if group_id is not None else None
        if params is None:
            no_params.append(country_id)
            continue
        a, b = params['a'], params['b']
        cost_per_km = cost_map.get(country_id)
        if cost_per_km is None or not np.isfinite(cost_per_km):
            no_cost.append(country_id)
            cost_per_km = 0.0

        r0, r1, c0, c1 = bboxes[country_id]
        r0b, r1b = max(r0 - buffer_px, 0), min(r1 + buffer_px + 1, n_rows)
        c0b, c1b = max(c0 - buffer_px, 0), min(c1 + buffer_px + 1, n_cols)
        xoff, yoff, wx, wy = c0b, r0b, c1b - c0b, r1b - r0b

        pop_sub = pop_band.ReadAsArray(xoff, yoff, wx, wy).astype(np.float64)
        country_sub = country_band.ReadAsArray(xoff, yoff, wx, wy)
        hq_sub = hq_band.ReadAsArray(xoff, yoff, wx, wy)

        valid = (pop_sub != pop_nodata) & ~np.isnan(pop_sub) & (pop_sub > 0) \
            & (country_sub == country_id)
        budget = np.where(valid, pop_sub * a, 0.0)
        site_ind = (hq_sub == 1).astype(np.float32) if hq_nodata is None \
            else ((hq_sub != hq_nodata) & (hq_sub == 1)).astype(np.float32)

        visits_accum = np.zeros_like(pop_sub)
        value_accum = np.zeros_like(pop_sub)
        unmet_total = np.zeros_like(pop_sub)
        realized_sum = 0.0

        for k in range(n_bands):
            r_lo, r_hi = edges[k], edges[k + 1]
            r_mid = 0.5 * (r_lo + r_hi)
            band_share = rf.cdf_exp(r_hi, b) - rf.cdf_exp(r_lo, b)
            q_band = budget * band_share

            denom = signal.fftconvolve(site_ind.astype(np.float64), kernels[k].astype(np.float64),
                                       mode='same')
            denom_safe = np.where(denom > 1e-6, denom, np.nan)

            q_visits = np.where(np.isfinite(denom_safe), q_band / denom_safe, 0.0)
            q_value = q_visits * cost_per_km * 2.0 * r_mid

            no_site_in_ring = ~np.isfinite(denom_safe) & valid
            unmet_total[no_site_in_ring] += q_band[no_site_in_ring]
            realized_sum += float(q_band[valid & ~no_site_in_ring].sum())

            visits_accum += signal.fftconvolve(q_visits, kernels[k].astype(np.float64),
                                               mode='same') * site_ind
            value_accum += signal.fftconvolve(q_value, kernels[k].astype(np.float64),
                                              mode='same') * site_ind

        beyond_share = 1.0 - rf.cdf_exp(cutoff_km, b)
        unmet_total += np.where(valid, budget * beyond_share, 0.0)

        qa = group_qa.setdefault(group_id, {
            'param_group': group_id, 'participation_param_a': a, 'distance_param_b': b,
            'cutoff_km': cutoff_km, 'n_bands': n_bands,
            'budget_total': 0.0, 'realized_total': 0.0, 'unmet_total': 0.0})
        qa['budget_total'] += float(budget.sum())
        qa['realized_total'] += realized_sum
        qa['unmet_total'] += float(unmet_total.sum())

        accumulate_window(out_handles['visits'][1], xoff, yoff, visits_accum.astype(np.float32))
        accumulate_window(out_handles['value'][1], xoff, yoff, value_accum.astype(np.float32))
        accumulate_window(out_handles['unmet'][1], xoff, yoff, unmet_total.astype(np.float32))

    for key in ('visits', 'value', 'unmet'):
        out_handles[key][1].FlushCache()
    pop_band = None; pop_ds = None
    country_band = None; country_ds = None
    hq_band = None; hq_ds = None
    for key in list(out_handles):
        out_handles[key] = None

    hb.log('recreation flows: %d countries computed, %d with no demand parameters, %d priced '
           'at zero for missing fuel cost (value set missing downstream)'
           % (len(bboxes) - len(no_params), len(no_params), len(no_cost)))
    pd.DataFrame({'iso3_r250_id': sorted(no_cost)}).to_csv(outputs['missing_cost_iso3'],
                                                           index=False)
    pd.DataFrame(sorted(group_qa.values(), key=lambda row: row['param_group'])).to_csv(
        outputs['group_table'], index=False)
    write_site_table(inputs['hq_sites'], inputs['country_id'], outputs['visits'],
                     outputs['value'], outputs['site_table'],
                     site_id_path=inputs.get('hq_sites_dissolved'))


def write_site_table(hq_sites_path, country_id_path, visits_path, value_path, output_path,
                     site_id_path=None):
    """One row per dissolved site: summed visits and value, the pixel count, and the site's
    dominant country. Streams over blocks; site pixels are a tiny share of the raster."""
    hq_nodata = pygeoprocessing.get_raster_info(hq_sites_path)['nodata'][0]
    hq_ds = gdal.Open(hq_sites_path); hq_band = hq_ds.GetRasterBand(1)
    country_ds = gdal.Open(country_id_path); country_band = country_ds.GetRasterBand(1)
    visits_ds = gdal.Open(visits_path); visits_band = visits_ds.GetRasterBand(1)
    value_ds = gdal.Open(value_path); value_band = value_ds.GetRasterBand(1)
    site_id_ds = gdal.Open(site_id_path) if site_id_path and os.path.exists(site_id_path) else None
    site_id_band = site_id_ds.GetRasterBand(1) if site_id_ds is not None else None

    chunks = []
    for block_info in pygeoprocessing.iterblocks((hq_sites_path, 1), offset_only=True):
        xoff, yoff = block_info['xoff'], block_info['yoff']
        wx, wy = block_info['win_xsize'], block_info['win_ysize']
        hq_block = hq_band.ReadAsArray(xoff, yoff, wx, wy)
        is_hq = (hq_block == 1) if hq_nodata is None \
            else ((hq_block != hq_nodata) & (hq_block == 1))
        if not np.any(is_hq):
            continue
        country_block = country_band.ReadAsArray(xoff, yoff, wx, wy)
        visits_block = visits_band.ReadAsArray(xoff, yoff, wx, wy)
        value_block = value_band.ReadAsArray(xoff, yoff, wx, wy)
        site_id_block = site_id_band.ReadAsArray(xoff, yoff, wx, wy) \
            if site_id_band is not None else None
        mask = is_hq if site_id_block is None else (is_hq & (site_id_block > 0))
        rows, cols = np.where(mask)
        chunk = {'iso3_r250_id': country_block[rows, cols],
                 'visits': visits_block[rows, cols], 'value': value_block[rows, cols]}
        if site_id_block is not None:
            chunk['site_id'] = site_id_block[rows, cols]
        chunks.append(pd.DataFrame(chunk))

    hq_band = None; hq_ds = None
    country_band = None; country_ds = None
    visits_band = None; visits_ds = None
    value_band = None; value_ds = None
    has_site_id = site_id_band is not None
    site_id_band = None; site_id_ds = None

    per_pixel = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
        columns=['site_id', 'iso3_r250_id', 'visits', 'value'])
    if has_site_id and len(per_pixel):
        per_pixel = per_pixel.groupby('site_id').agg(
            visits=('visits', 'sum'), value=('value', 'sum'),
            pixel_count=('visits', 'size'),
            iso3_r250_id=('iso3_r250_id', lambda s: s.value_counts().idxmax())).reset_index()
    per_pixel.to_csv(output_path, index=False)


# --- Tasks ---
def publish_inputs(p):
    """Every GEP task's first line: the recreation es_config row (defaults layer -- a
    caller-set value wins), the recreation data references from es_parameters (the staged
    drive data), the shared country references and the results registry. gep_base_year
    (2019) is the UNWTO target year, the fuel-cost price year and the airfare matrix's
    year."""
    utilities.hydrate_es_config(p, 'recreation', log=hb.log)
    utilities.hydrate_es_parameters(p, 'recreation', log=hb.log)
    utilities.initialize_country_paths(p)
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def environment_index(p):
    """Environment class raster (0-3) from the six SEALS7 class-share rasters + PA share."""
    publish_inputs(p)
    p.environment_index_path = os.path.join(p.cur_dir, 'env_index_1km.tif')
    if not p.run_this:
        return
    if not hb.path_exists(p.environment_index_path):
        lulc_share_paths = {
            lulc_class: p.get_path(p.recreation_lulc_share_path_template.format(lulc_class=lulc_class))
            for lulc_class in rf.RECREATION_LULC_CLASSES}
        calculate_environment_index(lulc_share_paths, p.recreation_pa_share_path,
                                    p.environment_index_path)
    return True


def accessibility_index(p):
    """Accessibility class raster (1-5) from urban share + distance-to-road."""
    publish_inputs(p)
    p.accessibility_index_path = os.path.join(p.cur_dir, 'accessibility_index_1km.tif')
    if not p.run_this:
        return
    if not hb.path_exists(p.accessibility_index_path):
        urban_share_path = p.get_path(
            p.recreation_lulc_share_path_template.format(lulc_class='urban'))
        calculate_accessibility_index(urban_share_path, p.recreation_road_distance_path,
                                      p.accessibility_index_path)
    return True


def recreation_sites(p):
    """Site rank raster (1-9), the high-quality mask (rank 9), and the dissolved site ids."""
    publish_inputs(p)
    p.recreation_sites_ranked_path = os.path.join(p.cur_dir, 'rec_sites_ranked_1km.tif')
    p.recreation_hq_sites_path = os.path.join(p.cur_dir, 'high_quality_sites_1km.tif')
    p.recreation_hq_sites_dissolved_path = os.path.join(p.cur_dir, 'hq_sites_dissolved_1km.tif')
    p.recreation_site_sizes_path = os.path.join(p.cur_dir, 'hq_site_pixel_counts.csv')
    if not p.run_this:
        return
    if not hb.path_exists(p.recreation_sites_ranked_path):
        rank_recreation_sites(p.accessibility_index_path, p.environment_index_path,
                              p.recreation_sites_ranked_path)
    if not hb.path_exists(p.recreation_hq_sites_path):
        extract_high_quality_sites(p.recreation_sites_ranked_path, p.recreation_hq_sites_path)
    if not hb.path_exists(p.recreation_hq_sites_dissolved_path):
        dissolve_high_quality_sites(p.recreation_hq_sites_path,
                                    p.recreation_hq_sites_dissolved_path,
                                    p.recreation_site_sizes_path)
    return True


def overnight_allocation(p):
    """UNWTO national overnights allocated to hotel pixels, with the nearest reported year
    standing in where the base year is unreported; also rasterizes the r250 country ids (the
    id raster every downstream task keys on). The substitution and missing-country reports
    are what keep a data gap distinguishable from zero tourism."""
    publish_inputs(p)
    p.recreation_country_id_path = os.path.join(p.cur_dir, 'iso3_r250_ids_1km.tif')
    p.recreation_hotels_raster_path = os.path.join(p.cur_dir, 'hotels_1km.tif')
    p.recreation_unwto_panel_path = os.path.join(p.cur_dir, 'unwto_panel.csv')
    p.recreation_overnights_path = os.path.join(p.cur_dir, 'overnights_1km.tif')
    p.recreation_overnights_substitution_path = os.path.join(
        p.cur_dir, 'overnights_year_substitution.csv')
    p.recreation_missing_overnights_path = os.path.join(p.cur_dir, 'missing_overnights_iso3.csv')
    if not p.run_this:
        return
    # The road-length raster is used ONLY as the 1 km reference grid for rasterization.
    if not hb.path_exists(p.recreation_hotels_raster_path):
        rasterize_presence(p.recreation_hotels_path, p.recreation_road_length_path,
                           p.recreation_hotels_raster_path)
    if not hb.path_exists(p.recreation_country_id_path):
        utilities.rasterize_id_column(p.gep_regions_input_path, p.recreation_road_length_path,
                                      p.gep_regions_id_col, p.recreation_country_id_path)
    if not hb.path_exists(p.recreation_overnights_path):
        if hb.path_exists(p.recreation_unwto_panel_path):
            overnight_df = hb.df_read(p.recreation_unwto_panel_path)
        else:
            overnight_df = rf.clean_unwto_data(read_unwto_sheets(p.recreation_unwto_path))
            overnight_df.to_csv(p.recreation_unwto_panel_path, index=False)
        country_overnights_map, substitution_rows = rf.build_country_overnights_map(
            overnight_df, int(p.gep_base_year))
        exact = sum(1 for row in substitution_rows if row['year_distance'] == 0)
        hb.log('recreation: overnight totals mapped for %d countries (%d at %d exactly, %d '
               'from a substitute year; %.4g total overnights)'
               % (len(country_overnights_map), exact, int(p.gep_base_year),
                  len(country_overnights_map) - exact, sum(country_overnights_map.values())))
        pd.DataFrame(substitution_rows).sort_values('year_distance', ascending=False).to_csv(
            p.recreation_overnights_substitution_path, index=False)
        allocate_overnights_raster(country_overnights_map, p.recreation_hotels_raster_path,
                                   p.recreation_country_id_path, p.recreation_overnights_path)
        countries = utilities.collapse_countries_to_r250(p.df_countries)
        missing = sorted(set(countries['iso3_r250_id'].astype(int))
                         - set(int(k) for k in country_overnights_map))
        pd.DataFrame({'iso3_r250_id': missing}).to_csv(
            p.recreation_missing_overnights_path, index=False)
    return True


def _flow_io(p, task_dir):
    outputs = {
        'visits': os.path.join(task_dir, 'site_visits_1km.tif'),
        'value': os.path.join(task_dir, 'site_value_1km.tif'),
        'unmet': os.path.join(task_dir, 'unmet_demand_1km.tif'),
        'site_table': os.path.join(task_dir, 'site_visits_value.csv'),
        'group_table': os.path.join(task_dir, 'param_group_qa.csv'),
        'missing_cost_iso3': os.path.join(task_dir, 'missing_cost_countries.csv'),
    }
    params_df = pd.read_csv(str(p.get_path(p.recreation_demand_params_path)))
    country_to_group_map, group_params = rf.validate_recreation_params(params_df)
    return outputs, country_to_group_map, group_params


def daily_recreation(p):
    """Resident population -> site flows (visits, value, unmet demand)."""
    publish_inputs(p)
    p.daily_recreation_visits_path = os.path.join(p.cur_dir, 'site_visits_1km.tif')
    p.daily_recreation_value_path = os.path.join(p.cur_dir, 'site_value_1km.tif')
    p.daily_recreation_site_table_path = os.path.join(p.cur_dir, 'site_visits_value.csv')
    p.daily_recreation_missing_cost_path = os.path.join(p.cur_dir, 'missing_cost_countries.csv')
    if not p.run_this:
        return
    outputs, country_to_group_map, group_params = _flow_io(p, p.cur_dir)
    if not hb.path_all_exist(list(outputs.values())):
        calculate_recreation_flows(
            {'hq_sites': p.recreation_hq_sites_path,
                'hq_sites_dissolved': p.recreation_hq_sites_dissolved_path,
                'population': str(p.get_path(p.recreation_population_path)),
                'country_id': p.recreation_country_id_path,
                'country_costs': str(p.get_path(p.recreation_fuel_cost_path))},
            outputs, country_to_group_map, group_params,
            float(p.recreation_max_distance_km), int(p.recreation_max_bands))
    return True


def tourist_recreation(p):
    """Allocated overnights -> site flows: the same engine with overnights as population."""
    publish_inputs(p)
    p.tourist_recreation_visits_path = os.path.join(p.cur_dir, 'site_visits_1km.tif')
    p.tourist_recreation_value_path = os.path.join(p.cur_dir, 'site_value_1km.tif')
    p.tourist_recreation_site_table_path = os.path.join(p.cur_dir, 'site_visits_value.csv')
    p.tourist_recreation_missing_cost_path = os.path.join(p.cur_dir, 'missing_cost_countries.csv')
    if not p.run_this:
        return
    outputs, country_to_group_map, group_params = _flow_io(p, p.cur_dir)
    if not hb.path_all_exist(list(outputs.values())):
        calculate_recreation_flows(
            {'hq_sites': p.recreation_hq_sites_path,
                'hq_sites_dissolved': p.recreation_hq_sites_dissolved_path,
                'population': p.recreation_overnights_path,
                'country_id': p.recreation_country_id_path,
                'country_costs': str(p.get_path(p.recreation_fuel_cost_path))},
            outputs, country_to_group_map, group_params,
            float(p.recreation_max_distance_km), int(p.recreation_max_bands))
    return True


def accommodation_value(p):
    """Overnight stays priced at the hotel-price surface: overnights x price per person per
    night, with a price gap kept missing rather than free, and the per-country coverage share
    published beside the value."""
    publish_inputs(p)
    p.accommodation_price_1km_path = os.path.join(p.cur_dir, 'hotel_price_per_occupancy_mean_1km.tif')
    p.accommodation_value_raster_path = os.path.join(p.cur_dir, 'accommodation_value_1km.tif')
    p.accommodation_table_path = os.path.join(p.cur_dir, 'accommodation_value_by_country.csv')
    if not p.run_this:
        return
    if hb.path_all_exist([p.accommodation_value_raster_path, p.accommodation_table_path]):
        return True

    overnights_info = pygeoprocessing.get_raster_info(p.recreation_overnights_path)
    overnights_nodata = overnights_info['nodata'][0]
    if not hb.path_exists(p.accommodation_price_1km_path):
        pygeoprocessing.warp_raster(
            str(p.get_path(p.recreation_hotel_price_path)), overnights_info['pixel_size'],
            p.accommodation_price_1km_path, 'bilinear',
            target_bb=overnights_info['bounding_box'],
            target_projection_wkt=overnights_info['projection_wkt'])

    def value_op(overnights, price):
        has_overnights = (overnights != overnights_nodata) & (overnights > 0) \
            if overnights_nodata is not None else (overnights > 0)
        price_missing = np.isnan(price)
        result = np.zeros_like(overnights, dtype=np.float32)
        result[has_overnights & price_missing] = np.nan
        priced = has_overnights & ~price_missing
        result[priced] = overnights[priced] * price[priced]
        return result

    pygeoprocessing.raster_calculator(
        [(p.recreation_overnights_path, 1), (p.accommodation_price_1km_path, 1)], value_op,
        p.accommodation_value_raster_path, gdal.GDT_Float32, rf.FLOAT_NDV)

    country_nodata = pygeoprocessing.get_raster_info(p.recreation_country_id_path)['nodata'][0]
    overnights_ds = gdal.Open(p.recreation_overnights_path)
    overnights_band = overnights_ds.GetRasterBand(1)
    country_ds = gdal.Open(p.recreation_country_id_path)
    country_band = country_ds.GetRasterBand(1)
    value_ds = gdal.Open(p.accommodation_value_raster_path)
    value_band = value_ds.GetRasterBand(1)

    totals = {}
    for block_info in pygeoprocessing.iterblocks((p.recreation_overnights_path, 1),
                                                 offset_only=True):
        xoff, yoff = block_info['xoff'], block_info['yoff']
        wx, wy = block_info['win_xsize'], block_info['win_ysize']
        overnights_block = overnights_band.ReadAsArray(xoff, yoff, wx, wy)
        country_block = country_band.ReadAsArray(xoff, yoff, wx, wy)
        value_block = value_band.ReadAsArray(xoff, yoff, wx, wy)
        has_overnights = (overnights_block != overnights_nodata) & (overnights_block > 0) \
            if overnights_nodata is not None else (overnights_block > 0)
        valid_country = (country_block != country_nodata) & (country_block > 0) \
            if country_nodata is not None else (country_block > 0)
        mask = has_overnights & valid_country
        if not np.any(mask):
            continue
        priced_mask = mask & ~np.isnan(value_block)
        for country_id in np.unique(country_block[mask]):
            country_id = int(country_id)
            country_mask = mask & (country_block == country_id)
            entry = totals.setdefault(country_id, {
                'total_overnights': 0.0, 'priced_overnights': 0.0, 'accommodation_value': 0.0})
            entry['total_overnights'] += float(overnights_block[country_mask].sum())
            entry['priced_overnights'] += float(
                overnights_block[priced_mask & (country_block == country_id)].sum())
            entry['accommodation_value'] += float(np.nansum(value_block[country_mask]))
    overnights_band = None; overnights_ds = None
    country_band = None; country_ds = None
    value_band = None; value_ds = None

    rows = []
    for country_id, entry in totals.items():
        coverage = entry['priced_overnights'] / entry['total_overnights'] \
            if entry['total_overnights'] > 0 else np.nan
        rows.append({'iso3_r250_id': country_id, 'total_overnights': entry['total_overnights'],
                     'priced_overnights': entry['priced_overnights'],
                     'accommodation_value': entry['accommodation_value'],
                     'price_coverage_share': coverage})
    country_df = pd.DataFrame(rows).sort_values('price_coverage_share')
    country_df.to_csv(p.accommodation_table_path, index=False)
    low = country_df[country_df['price_coverage_share'] < 0.5]
    if not low.empty:
        hb.log('recreation: %d countries hold hotel-price coverage under half their overnight '
               'stays, so their accommodation value is a partial sum' % len(low))
    return True


def air_travel_value(p):
    """International arrivals by origin region, priced at the staged region-to-region 2019
    airfare matrix. The matrix is the author's own scrape output deflated by the jet-fuel
    PPI; this task reads it as a staged input and makes no network request."""
    publish_inputs(p)
    p.air_travel_table_path = os.path.join(p.cur_dir, 'air_travel_value_by_country.csv')
    if not p.run_this:
        return
    if hb.path_exists(p.air_travel_table_path):
        return True
    arrivals = rf.clean_unwto_arrivals_by_region(
        read_unwto_arrivals_sheet(p.recreation_unwto_path))
    crosswalk = pd.read_csv(str(p.get_path(p.recreation_region_crosswalk_path)))[
        ['iso3_r250_id', 'unwto_region']]
    airfare = pd.read_csv(str(p.get_path(p.recreation_airfare_matrix_path)))[
        ['origin_region', 'destination_region', 'predicted_fare_2019_usd']]
    country_df = rf.air_travel_value_by_country(arrivals, crosswalk, airfare,
                                                int(p.gep_base_year))
    country_df.to_csv(p.air_travel_table_path, index=False)
    hb.log('recreation: arrivals-by-region data usable for %d countries' % len(country_df))
    return True


def gep_calculation(p):
    """GEP valuation for recreation: the four channels on one row per country, and their sum.

    The convention per channel follows the source's aggregation: a country with no fuel-cost
    data gets a missing travel value rather than a free-travel zero; a country with no UNWTO
    overnights in any year gets missing tourist and accommodation channels, never zeros; a
    country without arrivals-by-region data gets a missing air channel. recreation_gep is the
    sum of the channels the country HAS (a partial sum where coverage is partial, with the
    channel columns and coverage shares published beside it); a country with no channel at
    all stays missing.

    The tourist ground-travel column is published and NOT summed into recreation_gep: with
    the overnights read at the workbook's stated thousands, that channel multiplies
    person-nights by the ANNUAL participation rate, and whether the author intends a
    per-night rate instead moves it by orders of magnitude. The column carries his
    construction as delivered; the headline waits for his answer.
    """
    publish_inputs(p)
    service_results, already_done = utilities.begin_gep_calculation(p, 'recreation')
    if already_done:
        return

    daily = pd.read_csv(p.daily_recreation_site_table_path).groupby('iso3_r250_id').agg(
        daily_visits=('visits', 'sum'), daily_value=('value', 'sum')).reset_index()
    tourist = pd.read_csv(p.tourist_recreation_site_table_path).groupby('iso3_r250_id').agg(
        tourist_visits=('visits', 'sum'), tourist_value=('value', 'sum')).reset_index()
    df = daily.merge(tourist, on='iso3_r250_id', how='outer').fillna(0.0)
    # Band convolution round-trips accumulate float32 noise that can leave a true-zero flow
    # as a tiny negative; clamp rather than publish a sign that reads as negative demand.
    for col in ('daily_visits', 'daily_value', 'tourist_visits', 'tourist_value'):
        df[col] = df[col].clip(lower=0.0)

    missing_cost_daily = pd.read_csv(p.daily_recreation_missing_cost_path)['iso3_r250_id']
    missing_cost_tourist = pd.read_csv(p.tourist_recreation_missing_cost_path)['iso3_r250_id']
    df.loc[df['iso3_r250_id'].isin(missing_cost_daily), 'daily_value'] = np.nan
    df.loc[df['iso3_r250_id'].isin(missing_cost_tourist), 'tourist_value'] = np.nan
    missing_overnights = pd.read_csv(p.recreation_missing_overnights_path)['iso3_r250_id']
    df.loc[df['iso3_r250_id'].isin(missing_overnights),
           ['tourist_visits', 'tourist_value']] = np.nan

    accommodation = pd.read_csv(p.accommodation_table_path)[
        ['iso3_r250_id', 'accommodation_value', 'price_coverage_share']]
    df = df.merge(accommodation, on='iso3_r250_id', how='left')
    df.loc[df['iso3_r250_id'].isin(missing_overnights),
           ['accommodation_value', 'price_coverage_share']] = np.nan

    air_travel = pd.read_csv(p.air_travel_table_path)[
        ['iso3_r250_id', 'total_arrivals', 'air_travel_value', 'arrivals_coverage_share']]
    df = df.merge(air_travel, on='iso3_r250_id', how='left')
    no_arrivals = df['total_arrivals'].isna() | (df['total_arrivals'] == 0)
    df.loc[no_arrivals, ['air_travel_value', 'arrivals_coverage_share']] = np.nan
    df = df.drop(columns='total_arrivals')

    channel_cols = ['daily_value', 'accommodation_value', 'air_travel_value']
    df['recreation_gep'] = df[channel_cols].sum(axis=1, min_count=1)
    df['year'] = int(p.gep_base_year)

    attr_cols = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
                 'continent', 'region_un', 'region_wb', 'income_grp', 'subregion']
    attrs = utilities.collapse_countries_to_r250(p.df_countries)[attr_cols]
    keep_cols = attr_cols + ['year', 'daily_visits', 'daily_value', 'tourist_visits',
                             'tourist_value', 'accommodation_value', 'price_coverage_share',
                             'air_travel_value', 'arrivals_coverage_share', 'recreation_gep']
    df_gep = attrs.merge(df, how='left', on='iso3_r250_id')[keep_cols]
    hb.df_write(df_gep, service_results['gep_by_country_base_year'])

    map_df = (p.df_countries[['ee_r264_id', 'iso3_r250_id']]
              .merge(df[['iso3_r250_id', 'recreation_gep']], how='left', on='iso3_r250_id'))
    gdf = hb.df_merge(p.gdf_countries_simplified, map_df,
                      how='outer', left_on='ee_r264_id', right_on='ee_r264_id')
    gdf.to_file(service_results['gep_by_country_base_year'].replace('.csv', '.gpkg'),
                driver='GPKG')

    total = df_gep['recreation_gep'].sum()
    hb.log('Total recreation GEP for base year %s: %s over %d countries (summed channels: '
           'daily ground travel %s, accommodation %s, air travel %s; tourist ground travel '
           '%s published beside the sum, held out pending the per-night participation '
           'question)'
           % (p.gep_base_year, f'{total:,.2f}',
              int(df_gep['recreation_gep'].notna().sum()),
              f"{df_gep['daily_value'].sum():,.0f}",
              f"{df_gep['accommodation_value'].sum():,.0f}",
              f"{df_gep['air_travel_value'].sum():,.0f}",
              f"{df_gep['tourist_value'].sum():,.0f}"))
    return total


def gep_load_results(p):
    """Load GEP results from a PRIOR calculation run so the report renders without
    recomputing. Fails loudly if absent (run run_recreation.py first)."""
    publish_inputs(p)
    result_path = os.path.join(p.intermediate_dir, 'gep_calculation', 'gep_by_country_base_year.csv')
    if not hb.path_exists(result_path):
        raise FileNotFoundError(
            f'recreation GEP results not found at {result_path}. '
            f'Run the calculation first (run_recreation.py), then re-run results.')
    p.results.setdefault('recreation', {})
    p.results['recreation']['gep_by_country_base_year'] = result_path


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)

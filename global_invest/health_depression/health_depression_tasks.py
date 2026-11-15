"""Mental-health GEP tasks: the prevented-depression valuation on the staged tables.

This layer owns every file read and write. The science it calls lives in
health_depression_functions, which never opens a file.

The raster work is one number per country: the prevented-cases factor times population summed
over urban pixels, computed blockwise on WorldPop's grid at one risk ratio for every country,
converted from the odds ratio at the authors' reference-group risk p0. The rasters are the ones
the hypertension strand stages (same NDVI, same urban mask, same population); only the disease
tables differ. The author's high-resolution results file is carried beside the result as the
comparison point -- its spatial scope is an open question, so the log reports the gap rather
than judging it.
"""
import os

import hazelbean as hb
import numpy as np
import pandas as pd
from osgeo import gdal

from global_invest import utilities
from global_invest.health_depression import health_depression_functions as hd
from global_invest.health_hypertension.health_hypertension_functions import risk_ratio_from_odds_ratio

# The cost table's own spellings where they differ from the account's names.
COST_NAME_TO_ISO3 = {
    'Czech Republic': 'CZE', 'United States': 'USA', 'United Kingdom': 'GBR',
}


def publish_inputs(p):
    """Every GEP task's first line: the health_depression es_config row and the parameter rows
    from es_parameters (defaults layer -- a caller-set value prevails), the shared country
    references and the results registry."""
    utilities.hydrate_es_config(p, 'health_depression', log=hb.log)
    utilities.hydrate_es_parameters(p, 'health_depression', log=hb.log)
    utilities.initialize_country_paths(p, simplified='30sec')
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def factor_reuse_reason(p):
    """Why the saved urban factor sums cannot be used, or None when these inputs made them.

    The sums carry the risk ratio inside an exponential, so a table built at one odds ratio or
    reference risk cannot be rescaled to another: any change to the conversion, the urban
    threshold, a raster or the code means the global pass again.

    Returns:
        tuple: (the reason or None, the signature to record once the sums are rebuilt).
    """
    settings = {name: getattr(p, name) for name in (
        'health_depression_odds_ratio', 'health_depression_prevalence_nonexposed',
        'health_depression_urban_smod_min')}
    inputs = {str(p.get_path(path)): utilities.file_fingerprint(str(p.get_path(path))) for path in (
        p.health_depression_population_input_path, p.health_depression_ndvi_input_path,
        p.health_depression_urban_input_path, p.gep_regions_input_path)}
    for code in (__file__, hd.__file__):
        inputs[code] = utilities.file_fingerprint(code, content=False)
    signature = {'settings': settings, 'inputs': inputs}
    return utilities.outputs_reuse_reason(
        [p.health_depression_factor_path], signature, p.health_depression_factor_signature_path), signature


def aligned_raster_reuse_reason(path, sources, settings):
    """Why an aligned raster cannot be reused, or None when these sources made it on this grid.

    Existence is not enough: an old aligned raster behind a changed source or population grid
    would feed stale pixels into sums signed with the new sources' fingerprints.

    Returns:
        tuple: (the reason or None, the signature to record once the raster is rebuilt).
    """
    inputs = {source: utilities.file_fingerprint(source) for source in sources}
    inputs[__file__] = utilities.file_fingerprint(__file__, content=False)
    signature = {'settings': settings, 'inputs': inputs}
    return utilities.outputs_reuse_reason([path], signature, path + '.signature.json'), signature


def rebuild_aligned_raster(path, signature, build):
    """Build an aligned raster beside the old one, displace the old one rather than delete it,
    and sign the new one."""
    staged_path = hb.suri(path, 'staged')
    build(staged_path)
    if hb.path_exists(path):
        hb.displace_file(staged_path, path, raise_on_fails=True)
    else:
        os.rename(staged_path, path)
    utilities.write_outputs_signature(signature, path + '.signature.json')


def warp_onto_population_grid(src_path, pop_path, out_path):
    """Bilinear warp of a raster onto the population raster's extent and resolution."""
    ref = gdal.Open(pop_path)
    gt = ref.GetGeoTransform()
    bounds = (gt[0], gt[3] + ref.RasterYSize * gt[5], gt[0] + ref.RasterXSize * gt[1], gt[3])
    gdal.Warp(out_path, src_path,
              dstSRS='EPSG:4326', outputBounds=bounds, xRes=gt[1], yRes=abs(gt[5]),
              resampleAlg='bilinear', outputType=gdal.GDT_Float32, dstNodata=-9999.0,
              creationOptions=['COMPRESS=DEFLATE', 'TILED=YES'])


def urban_prevented_factor(p):
    """Per-country sum of the prevented-cases factor times population over urban pixels.

    The factor is exp(-10 ln(RR) NDVI) - 1 at the one risk ratio the odds ratio converts to at
    p0, so -10 ln(RR) is a single number across the blockwise pass.
    """
    publish_inputs(p)
    p.health_depression_country_id_path = os.path.join(p.cur_dir, 'iso3_r250_ids_1km.tif')
    p.health_depression_ndvi_on_pop_grid_path = os.path.join(p.cur_dir, 'ndvi_annual_on_pop_grid.tif')
    p.health_depression_factor_path = os.path.join(p.cur_dir, 'urban_prevented_factor_by_country.csv')
    p.health_depression_factor_signature_path = os.path.join(p.cur_dir, 'urban_prevented_factor_signature.json')
    if not p.run_this:
        return
    reason, signature = factor_reuse_reason(p)
    if reason is None:
        hb.log('  the urban factor sums are reused: their signature is unchanged')
        return True
    hb.log('  the urban factor sums are recomputed, %s' % reason)

    pop_path = str(p.get_path(p.health_depression_population_input_path))
    regions_path = str(p.get_path(p.gep_regions_input_path))
    ndvi_source_path = str(p.get_path(p.health_depression_ndvi_input_path))
    ids_reason, ids_signature = aligned_raster_reuse_reason(
        p.health_depression_country_id_path, [regions_path, pop_path], {'id_col': p.gep_regions_id_col})
    if ids_reason is not None:
        hb.log('  the country-id raster is rebuilt, %s' % ids_reason)
        rebuild_aligned_raster(p.health_depression_country_id_path, ids_signature, lambda out: (
            utilities.rasterize_id_column(regions_path, pop_path, p.gep_regions_id_col, out)))
    ndvi_reason, ndvi_signature = aligned_raster_reuse_reason(
        p.health_depression_ndvi_on_pop_grid_path, [ndvi_source_path, pop_path], {'resample': 'bilinear'})
    if ndvi_reason is not None:
        hb.log('  the NDVI on the population grid is rebuilt, %s' % ndvi_reason)
        rebuild_aligned_raster(p.health_depression_ndvi_on_pop_grid_path, ndvi_signature, lambda out: (
            warp_onto_population_grid(ndvi_source_path, pop_path, out)))
    ndvi_path = p.health_depression_ndvi_on_pop_grid_path

    rr = risk_ratio_from_odds_ratio(float(p.health_depression_odds_ratio),
                                    float(p.health_depression_prevalence_nonexposed))
    k = -10.0 * np.log(rr)
    hb.log('  risk ratio %.6f per 0.1 NDVI: odds ratio %s at reference-group risk p0 = %s'
           % (rr, p.health_depression_odds_ratio, p.health_depression_prevalence_nonexposed))

    smod_min = int(p.health_depression_urban_smod_min)
    rasters = [gdal.Open(path) for path in (
        pop_path, ndvi_path,
        str(p.get_path(p.health_depression_urban_input_path)), p.health_depression_country_id_path)]
    n_cols, n_rows = rasters[0].RasterXSize, rasters[0].RasterYSize
    sums = {}
    strip = 512
    for row0 in range(0, n_rows, strip):
        height = min(strip, n_rows - row0)
        pop, ndvi, smod, ids = (r.GetRasterBand(1).ReadAsArray(0, row0, n_cols, height)
                                for r in rasters)
        valid = (pop > 0) & (ndvi > -1.0) & (ndvi <= 1.0) & (smod >= smod_min) & (ids > 0)
        if not valid.any():
            continue
        ndvi_pos = np.where(ndvi[valid] > 0, ndvi[valid], 0.0)
        factor = (np.exp(k * ndvi_pos) - 1.0) * pop[valid]
        for country_id in np.unique(ids[valid]):
            mask = ids[valid] == country_id
            sums[int(country_id)] = sums.get(int(country_id), 0.0) + float(factor[mask].sum())
    df = pd.DataFrame(sorted(sums.items()), columns=['iso3_r250_id', 'urban_prevented_factor_pop_sum'])
    df.to_csv(p.health_depression_factor_path, index=False, encoding='utf-8-sig')
    utilities.write_outputs_signature(signature, p.health_depression_factor_signature_path)
    hb.log('  prevented-cases factor summed for %d countries' % len(df))
    return True


def gep_calculation(p):
    """GEP valuation for mental health: prevented depression cases at societal cost.

    Prevalence times the greenness-driven prevented-cases factor over urban residents, priced
    per country where a cost study exists, reported beside the author's results file.
    """
    publish_inputs(p)
    service_results, already_done = utilities.begin_gep_calculation(p, 'health_depression')
    outputs = [service_results['gep_by_country_base_year']]
    light_inputs = [__file__, hd.__file__]
    if already_done and utilities.reuse_reason(p, 'health_depression', outputs,
                                               light_inputs=light_inputs) is None:
        return

    if not p.run_this:
        return p
    reason, _ = factor_reuse_reason(p)
    if reason is not None:
        raise ValueError('The urban factor sums at %s cannot be used: %s. They carry the risk ratio '
                         'inside an exponential, so run urban_prevented_factor again rather than '
                         'rescale them.' % (p.health_depression_factor_path, reason))
    countries = utilities.collapse_countries_to_r250(p.df_countries)
    factor = pd.read_csv(p.health_depression_factor_path)
    df = countries.merge(factor, on='iso3_r250_id', how='left')
    prevalence = pd.read_csv(str(p.get_path(p.health_depression_prevalence_input_path)))
    df = df.merge(prevalence.rename(columns={'iso3': 'iso3_r250_label'}), on='iso3_r250_label', how='left')
    df['prevalence'] = df['prevalence_pct'] / 100.0

    costs = pd.read_csv(str(p.get_path(p.health_depression_cost_input_path)))
    name_to_iso3 = dict(zip(countries['iso3_r250_name'], countries['iso3_r250_label']))
    name_to_iso3.update(COST_NAME_TO_ISO3)
    costs['iso3_r250_label'] = costs['country'].map(name_to_iso3)
    unmapped = sorted(costs[costs['iso3_r250_label'].isna()]['country'].unique())
    if unmapped:
        hb.log('  %d cost-table rows are aggregates or unmapped and are dropped: %s'
               % (len(unmapped), ', '.join(unmapped)))
    costs = costs.dropna(subset=['iso3_r250_label']).rename(
        columns={'Price_USD_PPP': 'cost_per_case_usd'})[['iso3_r250_label', 'cost_per_case_usd']]
    # the Christensen costs are international (PPP) dollars of their own year; the shared
    # price-level and CPI series take them to the account's market USD at the base year
    pli = pd.read_csv(str(p.get_path(p.health_depression_price_level_input_path)))
    costs = utilities.international_to_usd(
        costs, pli, int(p.health_depression_cost_year), 'cost_per_case_usd')
    cpi = pd.read_csv(str(p.get_path(p.health_depression_us_cpi_input_path)))
    deflation = utilities.usd_deflation_factor(
        cpi, int(p.health_depression_cost_year), int(p.gep_base_year))
    costs['cost_per_case_usd'] = costs['cost_per_case_usd'] * deflation
    hb.log('  costs converted from %s international dollars to %s USD: country price levels '
           'then the shared CPI factor %.6f'
           % (p.health_depression_cost_year, p.gep_base_year, deflation))
    df = df.merge(costs[['iso3_r250_label', 'cost_per_case_usd']],
        on='iso3_r250_label', how='left')

    df = hd.health_depression_gep(df)

    reference = pd.read_csv(str(p.get_path(p.health_depression_reference_path)))
    df = df.merge(reference.rename(columns={
        'urban mental health service': 'health_depression_gep_reference'})[
        ['iso3_r250_label', 'health_depression_gep_reference']],
        on='iso3_r250_label', how='left')

    df['year'] = int(p.gep_base_year)
    utilities.write_gep_by_country(
        p, df[utilities.published_country_columns(df, 'health_depression')],
        service_results['gep_by_country_base_year'])
    gdf = hb.df_merge(p.gdf_countries_simplified, df, how='outer',
                      left_on='ee_r264_id', right_on='ee_r264_id')
    gdf.to_file(service_results['gep_by_country_base_year'].replace('.csv', '.gpkg'), driver='GPKG')
    utilities.write_reuse_signature(p, 'health_depression', outputs, light_inputs=light_inputs)

    ours = df['health_depression_gep'].sum()
    ref = df['health_depression_gep_reference'].sum()
    hb.log(f'Total health_depression GEP for base year {p.gep_base_year}: {ours:,.2f} '
           f'({int((df["health_depression_gep"].fillna(0) > 0).sum())} countries with a cost study; '
           f'{df["prevented_cases"].sum():,.0f} prevented cases over all countries)')
    hb.log(f'  the author\'s high-resolution file: {ref:,.2f}; the ratio measures the run-scope '
           f'question recorded in the status entry')
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    if not p.run_this:
        return p
    utilities.render_service_results(p)

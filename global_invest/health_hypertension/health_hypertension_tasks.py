"""Physical-health GEP tasks: the avoided-hypertension valuation on the staged tables.

This layer owns every file read and write. The science it calls lives in
health_hypertension_functions, which never opens a file.

The raster work reduces to one number per country: the sum of annual NDVI times population over
urban pixels, computed blockwise on WorldPop's own 1 km grid with the urban mask and country
ids warped or rasterized onto it. Everything after that is country-level arithmetic. The
appendix's own results table is carried beside the result as the comparison anchor.
"""
import os

import hazelbean as hb
import numpy as np
import pandas as pd
from osgeo import gdal

from global_invest import utilities
from global_invest.health_hypertension import health_hypertension_functions as hf


def publish_inputs(p):
    """Every GEP task's first line: the health_hypertension es_config row and the parameter
    rows from es_parameters (defaults layer -- a caller-set value prevails), the shared country
    references and the results registry."""
    utilities.hydrate_es_config(p, 'health_hypertension', log=hb.log)
    utilities.hydrate_es_parameters(p, 'health_hypertension', log=hb.log)
    utilities.initialize_country_paths(p, simplified='30sec')
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def urban_greenness(p):
    """Per-country sum of NDVI times population over urban pixels, on WorldPop's grid.

    Warps the annual NDVI onto the population grid, rasterizes the country ids onto it, and
    accumulates blockwise so the three global rasters never sit in memory together.
    """
    publish_inputs(p)
    p.health_country_id_path = os.path.join(p.cur_dir, 'iso3_r250_ids_1km.tif')
    p.health_ndvi_on_pop_grid_path = os.path.join(p.cur_dir, 'ndvi_annual_on_pop_grid.tif')
    p.health_urban_ndvi_pop_path = os.path.join(p.cur_dir, 'urban_ndvi_pop_by_country.csv')
    if not p.run_this:
        return

    pop_path = str(p.get_path(p.health_hypertension_population_input_path))
    if not hb.path_exists(p.health_country_id_path):
        utilities.rasterize_id_column(str(p.gep_regions_input_path), pop_path,
                                      p.gep_regions_id_col, p.health_country_id_path)
    if not hb.path_exists(p.health_ndvi_on_pop_grid_path):
        ref = gdal.Open(pop_path)
        gt = ref.GetGeoTransform()
        bounds = (gt[0], gt[3] + ref.RasterYSize * gt[5], gt[0] + ref.RasterXSize * gt[1], gt[3])
        gdal.Warp(p.health_ndvi_on_pop_grid_path,
                  str(p.get_path(p.health_hypertension_ndvi_input_path)),
                  dstSRS='EPSG:4326', outputBounds=bounds, xRes=gt[1], yRes=abs(gt[5]),
                  resampleAlg='bilinear', outputType=gdal.GDT_Float32, dstNodata=-9999.0,
                  creationOptions=['COMPRESS=DEFLATE', 'TILED=YES'])

    if not hb.path_exists(p.health_urban_ndvi_pop_path):
        smod_min = int(p.health_hypertension_urban_smod_min)
        rasters = [gdal.Open(path) for path in (
            pop_path, p.health_ndvi_on_pop_grid_path,
            str(p.get_path(p.health_hypertension_urban_input_path)), p.health_country_id_path)]
        n_cols, n_rows = rasters[0].RasterXSize, rasters[0].RasterYSize
        sums = {}
        strip = 512
        for row0 in range(0, n_rows, strip):
            height = min(strip, n_rows - row0)
            pop, ndvi, smod, ids = (r.GetRasterBand(1).ReadAsArray(0, row0, n_cols, height)
                                    for r in rasters)
            valid = ((pop > 0) & (ndvi > -1.0) & (ndvi <= 1.0)
                     & (smod >= smod_min) & (ids > 0))
            if not valid.any():
                continue
            weighted = np.where(ndvi[valid] > 0, ndvi[valid], 0.0) * pop[valid]
            for country_id in np.unique(ids[valid]):
                mask = ids[valid] == country_id
                sums[int(country_id)] = sums.get(int(country_id), 0.0) + float(weighted[mask].sum())
        df = pd.DataFrame(sorted(sums.items()), columns=['iso3_r250_id', 'urban_ndvi_pop_sum'])
        df.to_csv(p.health_urban_ndvi_pop_path, index=False, encoding='utf-8-sig')
        hb.log('  urban NDVI x population summed for %d countries' % len(df))
    return True


def gep_calculation(p):
    """GEP valuation for physical health: avoided hypertension cases at treatment cost.

    Prevalence times greenness-driven risk reduction times the urban population aged 30-79,
    priced per country, compared against the appendix's own results table.
    """
    publish_inputs(p)
    service_results, already_done = utilities.begin_gep_calculation(p, 'health_hypertension')
    if already_done:
        return

    countries = utilities.collapse_countries_to_r250(p.df_countries)
    greenness = pd.read_csv(p.health_urban_ndvi_pop_path)
    df = countries.merge(greenness, on='iso3_r250_id', how='left')

    prevalence = pd.read_csv(str(p.get_path(p.health_hypertension_prevalence_input_path)))
    df = df.merge(prevalence.rename(columns={'iso3': 'iso3_r250_label'}), on='iso3_r250_label', how='left')
    df['prevalence'] = df['prevalence_pct'] / 100.0
    shares = pd.read_csv(str(p.get_path(p.health_hypertension_age_share_input_path)))
    df = df.merge(shares.rename(columns={'iso3': 'iso3_r250_label'}), on='iso3_r250_label', how='left')
    costs = pd.read_csv(str(p.get_path(p.health_hypertension_cost_input_path)))
    df = df.merge(costs.rename(columns={'cost_per_case_usd2019': 'cost_per_case_usd'})[
        ['iso3_r250_label', 'cost_per_case_usd']], on='iso3_r250_label', how='left')

    df = hf.health_hypertension_gep(df, float(p.health_hypertension_odds_ratio))

    reference = pd.read_csv(str(p.get_path(p.health_hypertension_reference_path)))
    df = df.merge(reference.rename(columns={
        'health_hypertension_gep': 'health_hypertension_gep_reference'}),
        on='iso3_r250_label', how='left')

    df['year'] = int(p.gep_base_year)
    utilities.write_gep_by_country(
        p, df[utilities.published_country_columns(df, 'health_hypertension')],
        service_results['gep_by_country_base_year'])
    gdf = hb.df_merge(p.gdf_countries_simplified, df, how='outer',
                      left_on='ee_r264_id', right_on='ee_r264_id')
    gdf.to_file(service_results['gep_by_country_base_year'].replace('.csv', '.gpkg'), driver='GPKG')

    ours = df['health_hypertension_gep'].sum()
    both = df[(df['health_hypertension_gep'].fillna(0) > 0)
              & (df['health_hypertension_gep_reference'].fillna(0) > 0)]
    corr = np.corrcoef(np.log(both['health_hypertension_gep']),
                       np.log(both['health_hypertension_gep_reference']))[0, 1] if len(both) > 2 else np.nan
    hb.log(f'Total health_hypertension GEP for base year {p.gep_base_year}: {ours:,.2f} '
           f'({int((df["health_hypertension_gep"].fillna(0) > 0).sum())} countries with a cost study)')
    hb.log(f'  appendix reference: {df["health_hypertension_gep_reference"].sum():,.2f} over '
           f'{int((df["health_hypertension_gep_reference"].fillna(0) > 0).sum())}; '
           f'log-correlation over shared countries {corr:.3f}')
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)

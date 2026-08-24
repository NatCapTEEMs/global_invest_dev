"""Flood-control GEP tasks: the committed avoided-damage valuation on the r250 rows."""

import pandas as pd
import hazelbean as hb
from global_invest import utilities


def publish_inputs(p):
    """Every GEP task's first line: the flood_control es_config row and the data reference
    from es_parameters (defaults layer -- a caller-set value prevails), the shared country
    references and the results registry."""
    utilities.hydrate_es_config(p, 'flood_control', log=hb.log)
    utilities.hydrate_es_parameters(p, 'flood_control', log=hb.log)
    utilities.initialize_country_paths(p)
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def gep_calculation(p):
    """GEP valuation for flood control: per-country expected annual damage computed in the
    library through the ported calculation (depth x SDA x the country's damage curves, EAD over
    the pipeline's four return periods). The committed pipeline table is the comparison
    anchor, logged and pinned in the test suite, never the reported value."""
    publish_inputs(p)
    service_results, already_done = utilities.begin_gep_calculation(p, 'flood_control')
    if already_done:
        return

    if not hb.path_exists(p.flood_control_chain_ead_path):
        import geopandas as gpd
        import numpy as np
        import rasterio
        from rasterio.features import geometry_mask
        from rasterio.windows import from_bounds
        from global_invest.flood_control import flood_control_functions as fchain

        rps = (10, 20, 50, 500)  # the pipeline's final run
        gdf = gpd.read_file(p.flood_control_country_vector_path)
        sda_src = rasterio.open(p.flood_control_sda_raster_path)
        depth_srcs = {rp: rasterio.open(p.get_path(p.flood_control_depth_path_template.format(rp=rp)))
                      for rp in rps}
        pix_area = abs(sda_src.transform[0] * sda_src.transform[4])
        curves = fchain.damage_curves_from_wide(hb.df_read(p.flood_control_damage_wide_path))

        rows = []
        for _, crow in gdf.iterrows():
            iso, geom = crow['iso3'], crow.geometry
            if geom is None or geom.is_empty or not isinstance(iso, str):
                continue
            win = from_bounds(*geom.bounds, transform=sda_src.transform).round_offsets().round_lengths()
            sda = sda_src.read(1, window=win, boundless=True, fill_value=0)
            inside = ~geometry_mask([geom], out_shape=sda.shape,
                                    transform=sda_src.window_transform(win), invert=False)
            damages = []
            for rp in rps:
                dsrc = depth_srcs[rp]
                depth = dsrc.read(1, window=from_bounds(*geom.bounds, transform=dsrc.transform)
                                  .round_offsets().round_lengths(), boundless=True, fill_value=0)
                depth = np.where(inside, depth[:sda.shape[0], :sda.shape[1]], 0)
                _, _, total = fchain.pixel_damage_totals(depth, np.where(inside, sda, 0), curves,
                                                         iso, pix_area, depth_nodata=dsrc.nodata)
                damages.append(total)
            ead, _ = fchain.ead_from_rp_damages(np.array(rps, float), np.array(damages, float))
            rows.append({'iso3': iso, 'flood_control_gep': ead})
        pd.DataFrame(rows).to_csv(p.flood_control_chain_ead_path, index=False)

    chain = hb.df_read(p.flood_control_chain_ead_path).rename(columns={'iso3': 'iso3_r250_label'})
    attr_cols = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
                 'continent', 'region_un', 'region_wb', 'income_grp', 'subregion']
    countries = utilities.collapse_countries_to_r250(p.df_countries)[attr_cols]
    df_gep = countries.merge(chain, on='iso3_r250_label', how='left')
    df_gep['year'] = int(p.gep_base_year)
    hb.df_write(df_gep[attr_cols + ['year', 'flood_control_gep']],
                service_results['gep_by_country_base_year'])

    hb.log(f'Total flood_control GEP for base year {p.gep_base_year} (OUR chain): '
           f'{df_gep["flood_control_gep"].sum():,.2f}')
    committed = hb.df_read(p.flood_control_avoided_damage_path)
    hb.log(f'Committed pipeline table total: {committed.select_dtypes("number").iloc[:, -1].sum():,.2f} '
           f'(the test anchor; the known difference is Morocco).')
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)

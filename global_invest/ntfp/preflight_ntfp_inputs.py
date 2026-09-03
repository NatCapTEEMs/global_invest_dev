"""Resolve every input the NTFP run reads, and exit non-zero if one is not on this machine.

Run before the job proper, so a 4.3 GB roads shapefile that was never staged costs seconds to
discover rather than the queue wait plus however far the warps get before reaching it.

    python global_invest/ntfp/preflight_ntfp_inputs.py

⚠ This has to be a FILE inside the repository rather than a heredoc in the sbatch. ProjectFlow
infers where project directories live from the calling script's git repo, and a script piped in
on stdin is not inside one, so the constructor raises before any input is checked.
"""
import sys

import hazelbean as hb

from global_invest import utilities

INPUTS = ('gep_lulc_input_path', 'ntfp_ndvi_mean_path', 'ntfp_roads_vector_path',
          'ntfp_rivers_path', 'ntfp_value_per_ha_path')


def main():
    p = hb.ProjectFlow(project_name='gep_ntfp', run_mode='check')
    # The same four calls the tasks make in publish_inputs, in the same order, so what this
    # resolves is what the run will resolve. ⚠ Writing them out here is what caught
    # initialize_pyramid_paths missing from the task's own publish_inputs: ha_per_cell was read
    # by the warps and published by nobody, and the run would have died on the first one.
    utilities.hydrate_es_config(p, 'ntfp', log=hb.log)
    utilities.hydrate_es_parameters(p, 'ntfp', log=hb.log)
    utilities.initialize_country_paths(p)
    utilities.initialize_pyramid_paths(p)

    missing = []
    for name in INPUTS:
        path = getattr(p, name, None)
        found = hb.path_exists(path)
        print('[preflight] %-24s %-8s %s' % (name, 'OK' if found else 'MISSING', path))
        if not found:
            missing.append(name)

    # The pyramid the whole run is now on. Its absence would surface much later, as a warp with
    # no template, so it is named here with the rest.
    print('[preflight] %-24s %-8s %s'
          % ('ha_per_cell_10sec_path', 'OK' if hb.path_exists(p.ha_per_cell_10sec_path)
             else 'MISSING', p.ha_per_cell_10sec_path))
    if not hb.path_exists(p.ha_per_cell_10sec_path):
        missing.append('ha_per_cell_10sec_path')

    # ⚠⚠ EXISTENCE IS NOT VALIDITY. On MSI (job 17890607) every path resolved and the run died 25
    # minutes later because `ee_r264_correspondence.csv` was an older vintage carrying neither
    # `iso3_r250_id` nor `iso3_r250_label` -- the columns the country stage keys on. A file being
    # present says nothing about it being the one the code expects, so the columns are checked here.
    required = ['iso3_r250_id', 'iso3_r250_label']
    absent = [c for c in required if c not in p.df_countries.columns]
    print('[preflight] %-24s %-8s %s'
          % ('df_countries columns', 'OK' if not absent else 'STALE',
             'has %s' % ', '.join(required) if not absent else 'MISSING %s' % ', '.join(absent)))
    if absent:
        missing.append('df_countries is missing %s, so its vintage is not the one this code reads'
                       % ', '.join(absent))

    # ⚠⚠ The ENVIRONMENT is an input too, and it is checked by behaviour rather than version: a
    # broken PROJ installation makes gdal.Warp and RasterizeLayer write empty rasters WITH exit 0
    # (measured: an NDVI warp all-nodata across the Amazon, a country burn with zero cells).
    # A one-degree burn and warp must produce data, or nothing bigger is worth starting.
    from osgeo import gdal, ogr
    target = gdal.GetDriverByName('MEM').Create('', 360, 180, 1, gdal.GDT_Int32)
    target.SetGeoTransform((-180, 1, 0, 90, 0, -1))
    target.SetProjection(gdal.Open(p.ha_per_cell_10sec_path).GetProjection())
    vector = ogr.Open(p.gdf_countries_vector_path)
    gdal.RasterizeLayer(target, [1], vector.GetLayer(0),
                        options=['ALL_TOUCHED=FALSE', 'ATTRIBUTE=iso3_r250_id'])
    burned = int((target.ReadAsArray() > 0).sum())
    print('[preflight] %-24s %-8s %d cells with a country id'
          % ('one-degree burn', 'OK' if burned else 'EMPTY', burned))
    if not burned:
        missing.append('a one-degree country burn produced zero cells: the GDAL/PROJ '
                       'environment writes empty rasters with exit 0')

    if missing:
        sys.exit('[preflight] %d input(s) do not resolve on this machine: %s'
                 % (len(missing), ', '.join(missing)))
    print('[preflight] every input resolves')


if __name__ == '__main__':
    main()

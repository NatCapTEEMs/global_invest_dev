"""NTFP GEP tasks: accessible forest hectares per country, then the CWoN NWFP value per hectare.

Everything is on the account's own grid, WGS84 at 10 arcsec, the grid `ha_per_cell_10sec.tif`
defines. Accessibility is the reach grown 10 km from the road and river lines, and the forest
mask is screened by a five-year mean NDVI. The roads and rivers are the source module's own
layers, and the screens, thresholds and class ranges are its choices, pinned by tests.

⚠ The source module worked on a second grid, Mollweide at 300 m, because equal-area made a cell
a flat 9.0 hectares. The area that bought is already exact on the pyramid, read from ha_per_cell,
and the projection cost the map: `hb.make_path_pog` refuses a Mollweide raster, so nothing this
service produced could be published or compared with another service cell for cell.
"""
import os
from osgeo import gdal, ogr
import numpy as np

import hazelbean as hb
from global_invest import utilities

from global_invest.ntfp import ntfp_functions as nf

# The country id raster the zonal totals are keyed on, and the highest id it carries. The r250
# ids are ISO numeric codes, so they run past 250; bincount needs the ceiling, not the count.
COUNTRY_ID_MAX = 900



# The analysis grid is the account's own: WGS84 at 10 arcsec, the grid `ha_per_cell_10sec.tif`
# defines and every other service works on. A cell is about 309 m tall everywhere and
# 309*cos(latitude) m wide, so its area is not a constant and is read from `ha_per_cell` rather
# than assumed -- which is the house rule, and is exact where a fixed hectares-per-cell is not.
#
# ⚠⚠ This replaced a second grid, Mollweide at 300 m, carried over from the source module.
# Equal-area made the hectares a constant 9.0, which is why it was chosen, and it cost the map:
# `hb.make_path_pog` refuses a Mollweide raster outright, because the account's pyramid is
# defined in degrees. Nothing this service produced could be published, and its cells could not
# be compared with any other service's. The area a projection was buying is already exact on the
# pyramid, so there was nothing left on the other side of the trade.
GTIFF_CREATION_OPTIONS = ('TILED=YES', 'COMPRESS=DEFLATE', 'BIGTIFF=YES')
# How far to grow the reach in one pass. A band takes the cell width of its own middle latitude,
# so what has to stay small is the LATITUDE the band spans, not the number of rows: half a degree
# changes cos(latitude) by under a percent below 60 degrees, and the row count that comes to
# depends on the grid. Bounding the rows as well keeps the working array a sane size.
REACHABLE_BAND_MAX_DEGREES = 0.5
REACHABLE_BAND_ROWS = 256
# Above this latitude the reach in cells runs away as cos(latitude) goes to zero, and there is
# no forest there to reach. The growth stops rather than widening without bound.
REACHABLE_MAX_LATITUDE = 84.0


BURN_STRIPE_ROWS = 4096


def burn_lines_in_stripes(vector_paths, template_path, out_path, log=None):
    """Burn line layers onto the template grid a horizontal stripe at a time.

    One RasterizeLayer call over the whole world is what segfaulted: the target is 8.4 billion
    cells and the roads layer carries 21,438,033 features, and GDAL was asked to hold both at
    once. A stripe is bounded by construction, and the layer's spatial filter is set to the
    stripe's own extent so each pass touches only the features that fall inside it.

    ⚠ The raster is created SPARSE_OK. Away from the road network most stripes write no data at
    all, and an unwritten block in a sparse GeoTIFF occupies no disk -- which is what keeps an
    8.4 GB nominal raster to a fraction of that.
    """
    log = log or hb.log
    template = gdal.Open(template_path)
    width, height = template.RasterXSize, template.RasterYSize
    transform = template.GetGeoTransform()
    target = gdal.GetDriverByName('GTiff').Create(
        out_path, width, height, 1, gdal.GDT_Byte,
        options=list(GTIFF_CREATION_OPTIONS) + ['SPARSE_OK=TRUE'])
    target.SetGeoTransform(transform)
    target.SetProjection(template.GetProjection())
    target = None                     # closed, so each stripe reopens it for update

    # ⚠⚠ A spatial filter without an index is a full sequential scan, so striping a layer of
    # 21,438,033 features would read every one of them once per stripe. Shapefiles carry their
    # index in a sidecar `.qix`, which the roads layer does not ship with; building it once here
    # turns each stripe's filter into a lookup. The file is written beside the shapefile, so a
    # second run finds it already there.
    sources = []
    for path in vector_paths:
        source = ogr.Open(path)
        if str(path).lower().endswith('.shp') and not hb.path_exists(str(path)[:-4] + '.qix'):
            log('  building a spatial index for %s' % os.path.basename(str(path)))
            layer_name = os.path.splitext(os.path.basename(str(path)))[0]
            source.ExecuteSQL('CREATE SPATIAL INDEX ON "%s"' % layer_name)
        sources.append(source)
    for row in range(0, height, BURN_STRIPE_ROWS):
        rows = min(BURN_STRIPE_ROWS, height - row)
        top = transform[3] + transform[5] * row
        bottom = transform[3] + transform[5] * (row + rows)
        stripe = gdal.Open(out_path, gdal.GA_Update)
        for source in sources:
            layer = source.GetLayer(0)
            layer.SetSpatialFilterRect(transform[0], bottom,
                                       transform[0] + transform[1] * width, top)
            gdal.RasterizeLayer(stripe, [1], layer, burn_values=[1],
                                options=['ALL_TOUCHED=TRUE'])
            layer.SetSpatialFilter(None)
        stripe = None
        if row % (BURN_STRIPE_ROWS * 4) == 0:
            log('  burned to row %d of %d' % (row, height))
    return out_path


def reachable_mask_on_pyramid(vector_paths, template_path, out_path, distance_m,
                              log=None):
    """The reachable area, grown from the road and river lines on the account's own grid.

    The source module buffered the geometries in Mollweide and burned the dissolved polygon.
    That buffer is only as round as the projection is conformal, and Mollweide is not: it holds
    area, not distance, so a nominal 10 km stretched further the further it sat from the central
    meridian. Growing the reach on the pyramid instead makes the distance explicit -- the cell
    is 309 m tall everywhere and 309*cos(latitude) m wide -- and removes the second grid.

    The growth runs a band of rows at a time, because the exact transform over the whole world
    at 10 arcsec is 8.4 billion cells. Each band takes the cell width of its own middle latitude,
    and is read with a halo deep enough to see every line that could reach into it. The world
    wraps at the antimeridian, so the halo wraps too: a road in Chukotka reaches Alaska.

    ⚠ `hb.distance_transform_edt` is pygeoprocessing's, which takes one sampling distance for a
    whole raster and writes a file. The cell width here changes with latitude, so the sampling
    has to change with it, which that signature cannot express. This calls scipy's array form,
    the same transform underneath.

    Args:
        vector_paths (list): the road and river line layers, in the template's own CRS.
        template_path (str): a raster on the account's grid, for shape, transform and projection.
        out_path (str): where the 0/1 reachable mask is written.
        distance_m (float): how far from a road or river counts as reachable.
        log (callable): where progress goes.

    Returns:
        str: out_path.
    """
    import math
    from scipy import ndimage

    log = log or hb.log
    lines_path = hb.suri(out_path, 'lines')
    if not hb.path_exists(lines_path):
        # all_touched, because a road is narrower than a cell everywhere. Burning on the centre
        # rule would drop most of the network before anything was grown from it.
        #
        # ⚠⚠ Burned in LATITUDE STRIPES, not in one call. The first version pre-created the whole
        # 129,600 x 64,800 byte raster and handed it to RasterizeLayer with the roads layer's
        # 21,438,033 line features; on MSI (job 17879365) that segfaulted after four minutes,
        # having written the three warps first. Rasterizing a stripe at a time bounds what GDAL
        # holds at once, and a spatial filter means each stripe only sees the features that fall
        # in it. SPARSE_OK, because a line raster is almost entirely empty and unwritten blocks
        # then cost nothing on disk.
        log('  burning the road and river lines onto the grid, in stripes')
        burn_lines_in_stripes(vector_paths, template_path, lines_path, log=log)

    template = gdal.Open(template_path)
    width, height = template.RasterXSize, template.RasterYSize
    origin_y, cell_degrees = template.GetGeoTransform()[3], -template.GetGeoTransform()[5]
    metres_per_degree = 111_320.0
    cell_height_m = cell_degrees * metres_per_degree
    halo_rows = int(math.ceil(distance_m / cell_height_m))

    lines = gdal.Open(lines_path)
    target = gdal.GetDriverByName('GTiff').Create(
        out_path, width, height, 1, gdal.GDT_Byte, options=list(GTIFF_CREATION_OPTIONS))
    target.SetGeoTransform(template.GetGeoTransform())
    target.SetProjection(template.GetProjection())
    band = target.GetRasterBand(1)

    band_rows = max(1, min(REACHABLE_BAND_ROWS,
                           int(REACHABLE_BAND_MAX_DEGREES / cell_degrees)))
    for row in range(0, height, band_rows):
        rows = min(band_rows, height - row)
        top = max(0, row - halo_rows)
        bottom = min(height, row + rows + halo_rows)
        latitude = origin_y - (row + rows / 2.0) * cell_degrees
        cell_width_m = cell_height_m * max(math.cos(math.radians(
            min(abs(latitude), REACHABLE_MAX_LATITUDE))), 1e-6)
        halo_cols = min(int(math.ceil(distance_m / cell_width_m)), width // 2)

        block = lines.GetRasterBand(1).ReadAsArray(0, top, width, bottom - top) > 0
        # The world wraps, so the halo is taken from the far edge rather than padded with
        # emptiness, which would make every cell near the antimeridian unreachable.
        wrapped = np.concatenate(
            [block[:, width - halo_cols:], block, block[:, :halo_cols]], axis=1)
        distance = ndimage.distance_transform_edt(
            ~wrapped, sampling=(cell_height_m, cell_width_m))
        reachable = (distance <= distance_m)[:, halo_cols:halo_cols + width]
        band.WriteArray(reachable[row - top:row - top + rows].astype('uint8'), 0, row)
        if row % (band_rows * 32) == 0:
            log('  grown to row %d of %d' % (row, height))
    band.FlushCache()
    target = None
    return out_path


def rasterize_polygon_to_grid(vector_path, reference_raster_path, out_path,
                              attribute=None, output_type=gdal.GDT_Byte, all_touched=False,
                              append=False):
    """Burn a vector layer onto the reference grid, as a 0/1 mask or as one attribute's value.

    Args:
        vector_path (str): the geometries, in the grid's own CRS.
        reference_raster_path (str): a raster on the analysis grid, for shape and geotransform.
        out_path (str): where the burned raster is written.
        attribute (str): the field whose value is burned. None burns 1 everywhere a geometry covers.
        output_type: the GDAL type, wide enough to hold the attribute.
        all_touched (bool): whether a cell any part of the geometry touches is burned, or only one
            whose centre it covers. Passed explicitly at every call rather than left to
            the GDAL default, because it is the rule that decides what a boundary cell counts as.
        append (bool): burn into the existing raster instead of creating one, so several layers
            land in a single mask.
    """
    if append:
        target = gdal.Open(out_path, gdal.GA_Update)
    else:
        reference = gdal.Open(reference_raster_path)
        target = gdal.GetDriverByName('GTiff').Create(
            out_path, reference.RasterXSize, reference.RasterYSize, 1, output_type,
            options=list(GTIFF_CREATION_OPTIONS))
        target.SetGeoTransform(reference.GetGeoTransform())
        target.SetProjection(reference.GetProjection())
    source = ogr.Open(vector_path)
    options = ['ALL_TOUCHED=%s' % ('TRUE' if all_touched else 'FALSE')]
    if attribute is None:
        gdal.RasterizeLayer(target, [1], source.GetLayer(0), burn_values=[1], options=options)
    else:
        gdal.RasterizeLayer(target, [1], source.GetLayer(0),
                            options=options + ['ATTRIBUTE=%s' % attribute])
    target = None
    return out_path


def warp_to_analysis_grid(src_path, out_path, template_path, resample_algorithm,
                          output_type=gdal.GDT_Float32, src_nodata=None, dst_nodata=None):
    """One raster on the account's grid, taking its projection, extent and cell size from
    `ha_per_cell_10sec.tif` rather than from the input.

    Reading them off the template is what keeps every raster the same shape so the arrays can be
    combined cell by cell. Deriving them per input is what produced a country raster of all zeros
    and an NDVI raster 56 rows short of the land cover, neither of which announced itself.
    """
    template = gdal.Open(template_path)
    transform = template.GetGeoTransform()
    bounds = (transform[0], transform[3] + transform[5] * template.RasterYSize,
              transform[0] + transform[1] * template.RasterXSize, transform[3])
    gdal.Warp(out_path, src_path,
              dstSRS=template.GetProjection(),
              xRes=transform[1], yRes=-transform[5],
              outputBounds=bounds,
              resampleAlg=resample_algorithm, outputType=output_type,
              srcNodata=src_nodata, dstNodata=dst_nodata,
              multithread=True, creationOptions=list(GTIFF_CREATION_OPTIONS))
    return out_path


ROWS_PER_BLOCK = 512


def accessible_forest_hectares_by_country(lulc_path, access_path, country_id_path,
                                          ha_per_cell_path, n_countries, ndvi_path=None):
    """Accessible forest hectares summed per country id, read in blocks.

    A cell counts when the land cover calls it forest, the reachable mask covers it, and, where
    an NDVI raster is given, it carries enough live vegetation to yield a product.

    ⚠ A cell's area is READ, from `ha_per_cell_10sec.tif`, not assumed. On the account's grid it
    runs from about 9.5 hectares at the equator to nearly nothing at the poles, where the old
    equal-area grid made it a flat 9.0 everywhere.

    The rasters are read a few hundred rows at a time. A single band on this grid is 8.4 billion
    cells, so the five this needs cannot be held at once, and the totals accumulate across blocks
    instead.

    Args:
        lulc_path (str): land cover on the analysis grid.
        access_path (str): the reachable mask on the same grid.
        country_id_path (str): country ids on the same grid.
        ha_per_cell_path (str): the account's hectares-per-cell grid.
        n_countries (int): the highest country id, so the accumulator is long enough.
        ndvi_path (str): the NDVI on the same grid, or None to skip that screen.

    Returns:
        np.ndarray: hectares per country id, length n_countries + 1.
    """
    sources = [gdal.Open(lulc_path), gdal.Open(access_path), gdal.Open(country_id_path),
               gdal.Open(ha_per_cell_path)]
    if ndvi_path is not None:
        sources.append(gdal.Open(ndvi_path))
    shapes = {(s.RasterXSize, s.RasterYSize) for s in sources}
    if len(shapes) != 1:
        raise ValueError('the inputs are not on one grid: %s' % sorted(shapes))

    width, height = sources[0].RasterXSize, sources[0].RasterYSize
    totals = np.zeros(n_countries + 1, dtype='float64')
    for row in range(0, height, ROWS_PER_BLOCK):
        rows = min(ROWS_PER_BLOCK, height - row)
        blocks = [s.GetRasterBand(1).ReadAsArray(0, row, width, rows) for s in sources]
        forest = nf.forest_mask(blocks[0])
        if ndvi_path is not None:
            forest = nf.vegetated_forest_mask(forest, blocks[4])
        per_cell = nf.accessible_forest_hectares(forest, blocks[1] > 0, blocks[3])
        totals += nf.hectares_by_zone(per_cell, blocks[2], n_countries)
    return totals


def publish_inputs(p):
    """Every GEP task's first line: the ntfp es_config row and the data references from
    es_parameters (defaults layer -- a caller-set value prevails), the shared country
    references and the results registry."""
    utilities.hydrate_es_config(p, 'ntfp', log=hb.log)
    utilities.hydrate_es_parameters(p, 'ntfp', log=hb.log)
    utilities.initialize_country_paths(p)
    # The grid every raster here now lands on. coastal_carbon and terrestrial_carbon publish it
    # the same way; a service that resolves the pyramid for itself is how two of them drifted
    # onto different grids without anyone noticing.
    utilities.initialize_pyramid_paths(p)
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def accessible_forest(p):
    """Forest hectares reachable from a road or river, per country, on the account's grid.

    The land cover and the NDVI are put on the grid `ha_per_cell_10sec.tif` defines, the reach is
    grown 10 km from the road and river lines, and forest is counted where it is inside that reach
    and green enough to yield a product. Every screen and threshold is the source module's; the
    grid is not, and the module docstring says why.

    Everything is read in blocks. The grid is 129,600 by 64,800, so a single band is 8.4 billion
    cells and the five this needs would not fit in memory at once.
    """
    publish_inputs(p)
    p.ntfp_accessible_forest_path = os.path.join(p.cur_dir, 'accessible_forest_ha_by_country.csv')
    if not p.run_this:
        return
    if hb.path_exists(p.ntfp_accessible_forest_path):
        hb.log('Accessible forest already computed. Skipping.')
        return

    import numpy as np
    from osgeo import gdal

    # Every raster here lands on the account's own grid, whose projection, extent and cell size
    # are read off ha_per_cell rather than declared. Nothing is reprojected into a second CRS,
    # so no vector needs reprojecting either: the boundaries, roads and rivers are already in
    # the grid's CRS and burn onto it directly.
    template_path = p.ha_per_cell_10sec_path

    # Nearest neighbour rather than dominant class: 10 arcsec is about 309 m at the equator,
    # near enough the land cover's own 300 m that there is no majority to take, and classifying
    # before or after a nearest-neighbour resample gives the same cells either way.
    lulc_path = os.path.join(p.cur_dir, 'lulc_10sec.tif')
    if not hb.path_exists(lulc_path):
        hb.log('ntfp: putting the land cover on the account grid')
        warp_to_analysis_grid(p.gep_lulc_input_path, lulc_path, template_path, 'near',
                              output_type=gdal.GDT_Int16)

    # Countries come from the boundary polygons burned onto the grid, which is the zonal step the
    # source module takes, done once for every country at once. A cell goes to whichever polygon
    # covers its centre. Reprojecting a ready-made id raster instead would assign border and
    # coastal cells by nearest neighbour, which is a different rule.
    countries_path = os.path.join(p.cur_dir, 'country_id_10sec.tif')
    if not hb.path_exists(countries_path):
        hb.log('ntfp: burning the country ids onto the account grid')
        # Centre rule, so a cell belongs to exactly one country. Burning every country a cell
        # touches would give a border cell to whichever country happens to be drawn last.
        # coastal_carbon builds its country id raster on the same rule.
        rasterize_polygon_to_grid(p.gdf_countries_vector_path, template_path, countries_path,
                                  attribute='iso3_r250_id', output_type=gdal.GDT_Int32,
                                  all_touched=False)

    # Bilinear because NDVI is continuous, and the integer type is kept so the threshold is
    # compared in the raster's own units rather than on scaled floats.
    ndvi_path = None
    if hb.path_exists(getattr(p, 'ntfp_ndvi_mean_path', None)):
        ndvi_path = os.path.join(p.cur_dir, 'ndvi_10sec.tif')
        if not hb.path_exists(ndvi_path):
            hb.log('ntfp: putting the five-year mean NDVI on the account grid')
            warp_to_analysis_grid(p.ntfp_ndvi_mean_path, ndvi_path, template_path, 'bilinear',
                                  output_type=gdal.GDT_Int16,
                                  src_nodata=nf.NDVI_NODATA, dst_nodata=nf.NDVI_NODATA)
    else:
        hb.log('ntfp: no NDVI raster staged, so the forest mask is the land-cover class alone.')

    access_path = os.path.join(p.cur_dir, 'reachable_10sec.tif')
    if not hb.path_exists(access_path):
        hb.log('ntfp: growing the reach %d km from the roads and rivers'
               % (nf.NTFP_ACCESS_BUFFER_M / 1000))
        reachable_mask_on_pyramid(
            [p.ntfp_roads_vector_path, p.ntfp_rivers_path],
            template_path, access_path, nf.NTFP_ACCESS_BUFFER_M, log=hb.log)

    hectares = accessible_forest_hectares_by_country(
        lulc_path, access_path, countries_path, template_path, COUNTRY_ID_MAX,
        ndvi_path=ndvi_path)

    countries = p.df_countries[['iso3_r250_id', 'iso3_r250_label']].drop_duplicates('iso3_r250_id')
    countries = countries[countries['iso3_r250_id'].notna()]
    countries['accessible_forest_ha'] = [
        float(hectares[int(i)]) if int(i) <= COUNTRY_ID_MAX else float('nan')
        for i in countries['iso3_r250_id']]
    hb.df_write(countries, p.ntfp_accessible_forest_path)
    hb.log(f'ntfp: {countries["accessible_forest_ha"].sum():,.0f} accessible forest hectares '
           f'over {int((countries["accessible_forest_ha"] > 0).sum())} countries')
    return True

def gep_calculation(p):
    """GEP valuation for NTFP: accessible forest hectares times the CWoN NWFP value per hectare."""
    publish_inputs(p)
    service_results, already_done = utilities.begin_gep_calculation(p, 'ntfp')
    if already_done:
        return

    accessible = hb.df_read(p.ntfp_accessible_forest_path)
    value_per_ha = nf.nwfp_rate_long(hb.df_read(p.ntfp_value_per_ha_path))
    df_gep = nf.ntfp_gep_by_country(accessible, value_per_ha, int(p.gep_base_year))

    countries = utilities.country_attributes(p)
    df_gep = countries.merge(df_gep.drop(columns=['iso3_r250_id'], errors='ignore'),
                             on='iso3_r250_label', how='left')
    df_gep['year'] = int(p.gep_base_year)
    hb.df_write(df_gep, service_results['gep_by_country_base_year'])
    hb.log('Total ntfp GEP for base year %d: %s over %d countries'
           % (int(p.gep_base_year), format(df_gep['ntfp_gep'].sum(), ',.2f'),
              int(df_gep['ntfp_gep'].notna().sum())))
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)

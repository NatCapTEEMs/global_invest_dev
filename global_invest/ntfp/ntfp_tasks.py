"""NTFP GEP tasks: accessible forest hectares per country, then the CWoN NWFP value per hectare.

The stage reproduces the source module rather than approximating it. Everything is on one
Mollweide grid at 300 m, the land cover's own resolution; accessibility is the dissolved 10 km
buffer of the road and river geometries; and the forest mask is screened by a five-year mean
NDVI. The roads and rivers are the source module's own layers.
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



# The analysis grid, taken from the source module so the two produce the same thing. Mollweide
# is equal-area, so a cell is a fixed number of hectares anywhere on it, and 300 m is the ESA
# land cover's own resolution: at 309 m per pixel at the equator, a coarser grid throws the
# land-cover detail away before the accessibility and NDVI screens ever see it.
MOLLWEIDE_WKT = (
    'PROJCS["World_Mollweide",'
    'GEOGCS["GCS_WGS_1984",'
    'DATUM["WGS_1984",'
    'SPHEROID["WGS_84",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],'
    'UNIT["Degree",0.017453292519943295]],'
    'PROJECTION["Mollweide"],'
    'PARAMETER["False_Easting",0],'
    'PARAMETER["False_Northing",0],'
    'PARAMETER["Central_Meridian",0],'
    'UNIT["Meter",1]]'
)
ACCESSIBILITY_CELL_SIZE_M = 300.0
HECTARES_PER_CELL = (ACCESSIBILITY_CELL_SIZE_M / 100.0) ** 2
# A fixed world extent rather than one derived from each input. Transforming 90 degrees north
# into Mollweide is undefined, so the poles are trimmed, and pinning the extent is also what
# keeps every input on one grid.
MOLLWEIDE_BBOX = (-17_900_000.0, -8_900_000.0, 17_900_000.0, 8_900_000.0)
GTIFF_CREATION_OPTIONS = ('TILED=YES', 'COMPRESS=DEFLATE', 'BIGTIFF=YES')


def reproject_vector(src_path, out_path, target_wkt=MOLLWEIDE_WKT):
    """One vector on the analysis grid's projection, so a metre buffer means metres."""
    import geopandas as gpd

    gpd.read_file(src_path).to_crs(target_wkt).to_file(out_path, driver='GPKG')
    return out_path


def buffer_and_union_access(source_vector_paths, out_path, buffer_distance_m):
    """The reachable area, as one dissolved polygon.

    Accessibility is built from the road and river geometries rather than by dilating a raster.
    Buffering the lines and dissolving them is exact at any distance and costs nothing per cell,
    where a raster dilation over seven billion cells would dominate the run and would round the
    distance to whole cells.

    Args:
        source_vector_paths (list): the road and river vectors, already in the target projection.
        out_path (str): where the single dissolved polygon is written.
        buffer_distance_m (float): how far from a road or river counts as reachable.

    Returns:
        str: out_path.
    """
    import geopandas as gpd
    from shapely.ops import unary_union

    geometries, crs = [], None
    for path in source_vector_paths:
        gdf = gpd.read_file(path)
        crs = crs or gdf.crs
        geometries.extend(gdf.buffer(buffer_distance_m).tolist())
    gpd.GeoDataFrame(geometry=[unary_union(geometries)], crs=crs).to_file(out_path, driver='GPKG')
    return out_path


def rasterize_polygon_to_grid(vector_path, reference_raster_path, out_path,
                              attribute=None, output_type=gdal.GDT_Byte, all_touched=False):
    """Burn a polygon layer onto the reference grid, as a 0/1 mask or as one attribute's value.

    Args:
        vector_path (str): the polygons, already in the grid's projection.
        reference_raster_path (str): a raster on the analysis grid, for shape and geotransform.
        out_path (str): where the burned raster is written.
        attribute (str): the field whose value is burned. None burns 1 everywhere a polygon covers.
        output_type: the GDAL type, wide enough to hold the attribute.
        all_touched (bool): whether a cell any part of the polygon touches is burned, or only one
            whose centre the polygon covers. Passed explicitly at every call rather than left to
            the GDAL default, because it is the rule that decides what a boundary cell counts as.
    """
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


def warp_to_analysis_grid(src_path, out_path, resample_algorithm, output_type=gdal.GDT_Float32,
                          src_nodata=None, dst_nodata=None):
    """One raster on the analysis grid: Mollweide, 300 m, the fixed world extent.

    The extent and cell size are given rather than derived from the input, so every raster comes
    out the same shape and the arrays can be combined cell by cell. Deriving them per input is
    what produced a country raster of all zeros and an NDVI raster 56 rows short of the land
    cover, neither of which announced itself.
    """
    gdal.Warp(out_path, src_path,
              dstSRS=MOLLWEIDE_WKT,
              xRes=ACCESSIBILITY_CELL_SIZE_M, yRes=ACCESSIBILITY_CELL_SIZE_M,
              outputBounds=MOLLWEIDE_BBOX,
              resampleAlg=resample_algorithm, outputType=output_type,
              srcNodata=src_nodata, dstNodata=dst_nodata,
              multithread=True, creationOptions=list(GTIFF_CREATION_OPTIONS))
    return out_path


ROWS_PER_BLOCK = 512


def accessible_forest_hectares_by_country(lulc_path, access_path, country_id_path,
                                          n_countries, ndvi_path=None):
    """Accessible forest hectares summed per country id, read in blocks.

    A cell counts when the land cover calls it forest, the reachable polygon covers it, and,
    where an NDVI raster is given, it carries enough live vegetation to yield a product. A cell
    is a fixed 9 hectares because the grid is equal-area.

    The rasters are read a few hundred rows at a time. On the 300 m grid a single band is 14 GB,
    so the four this needs cannot be held at once, and the totals accumulate across blocks
    instead.

    Args:
        lulc_path (str): land cover on the analysis grid.
        access_path (str): the reachable mask on the same grid.
        country_id_path (str): country ids on the same grid.
        n_countries (int): the highest country id, so the accumulator is long enough.
        ndvi_path (str): the NDVI on the same grid, or None to skip that screen.

    Returns:
        np.ndarray: hectares per country id, length n_countries + 1.
    """
    sources = [gdal.Open(lulc_path), gdal.Open(access_path), gdal.Open(country_id_path)]
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
            forest = nf.vegetated_forest_mask(forest, blocks[3])
        per_cell = nf.accessible_forest_hectares(
            forest, blocks[1] > 0,
            np.full(forest.shape, HECTARES_PER_CELL, dtype=np.float32))
        totals += nf.hectares_by_zone(per_cell, blocks[2], n_countries)
    return totals


def publish_inputs(p):
    """Every GEP task's first line: the ntfp es_config row and the data references from
    es_parameters (defaults layer -- a caller-set value prevails), the shared country
    references and the results registry."""
    utilities.hydrate_es_config(p, 'ntfp', log=hb.log)
    utilities.hydrate_es_parameters(p, 'ntfp', log=hb.log)
    utilities.initialize_country_paths(p)
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def accessible_forest(p):
    """Forest hectares reachable from a road or river, per country, on the 300 m analysis grid.

    The stage follows the source module step for step: the land cover and the NDVI are put on
    one Mollweide grid at the land cover's own resolution, the road and river geometries are
    buffered and dissolved into a single reachable polygon, and forest is counted where it is
    inside that polygon and green enough to yield a product.

    Everything is read in blocks. The grid is 119,333 by 59,333, so a single band is 14 GB and
    the four this needs would not fit in memory at once.
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

    # The land cover, on the analysis grid. Nearest neighbour rather than dominant class: at
    # 300 m the grid is the source's own resolution, so there is no majority to take, and
    # classifying before or after a nearest-neighbour resample gives the same cells either way.
    lulc_path = os.path.join(p.cur_dir, 'lulc_mollweide_300m.tif')
    if not hb.path_exists(lulc_path):
        hb.log('ntfp: putting the land cover on the 300 m Mollweide grid')
        warp_to_analysis_grid(p.gep_lulc_input_path, lulc_path, 'near',
                              output_type=gdal.GDT_Int16)

    # Countries come from the boundary polygons, reprojected and then burned onto the analysis
    # grid, which is the order the source module uses: it reprojects the same vector and takes
    # zonal statistics over it. Burning the id is that zonal step done once for every country at
    # once, and it assigns a cell on the same rule, by which polygon covers the cell centre.
    # Reprojecting a ready-made id raster instead would assign border and coastal cells by
    # nearest neighbour from a 10 arcsec grid, which is a different rule.
    countries_path = os.path.join(p.cur_dir, 'countries_mollweide_300m.tif')
    if not hb.path_exists(countries_path):
        countries_vector = os.path.join(p.cur_dir, 'countries_mollweide.gpkg')
        if not hb.path_exists(countries_vector):
            hb.log('ntfp: reprojecting the country boundaries to Mollweide')
            reproject_vector(p.ntfp_countries_vector_path, countries_vector)
        hb.log('ntfp: burning the country ids onto the same grid')
        # Centre rule, so a cell belongs to exactly one country. Burning every country a cell
        # touches would give a border cell to whichever country happens to be drawn last.
        # coastal_carbon builds its country id raster on the same rule.
        rasterize_polygon_to_grid(countries_vector, lulc_path, countries_path,
                                  attribute='iso3_r250_id', output_type=gdal.GDT_Int32,
                                  all_touched=False)

    # Bilinear because NDVI is continuous, and the integer type is kept so the threshold is
    # compared in the raster's own units rather than on scaled floats.
    ndvi_path = None
    if hb.path_exists(getattr(p, 'ntfp_ndvi_mean_path', None)):
        ndvi_path = os.path.join(p.cur_dir, 'ndvi_mollweide_300m.tif')
        if not hb.path_exists(ndvi_path):
            hb.log('ntfp: putting the five-year mean NDVI on the same grid')
            warp_to_analysis_grid(p.ntfp_ndvi_mean_path, ndvi_path, 'bilinear',
                                  output_type=gdal.GDT_Int16,
                                  src_nodata=nf.NDVI_NODATA, dst_nodata=nf.NDVI_NODATA)
    else:
        hb.log('ntfp: no NDVI raster staged, so the forest mask is the land-cover class alone.')

    # Accessibility, built from the geometries rather than by dilating a raster.
    union_path = os.path.join(p.cur_dir, 'reachable_from_road_or_river.gpkg')
    if not hb.path_exists(union_path):
        reprojected = []
        for name, src in (('roads', p.ntfp_roads_vector_path), ('rivers', p.ntfp_rivers_path)):
            out = os.path.join(p.cur_dir, f'{name}_mollweide.gpkg')
            if not hb.path_exists(out):
                hb.log(f'ntfp: reprojecting the {name} to Mollweide')
                reproject_vector(src, out)
            reprojected.append(out)
        hb.log(f'ntfp: buffering roads and rivers by {nf.NTFP_ACCESS_BUFFER_M / 1000:.0f} km '
               f'and dissolving them into one polygon')
        buffer_and_union_access(reprojected, union_path, nf.NTFP_ACCESS_BUFFER_M)

    access_path = os.path.join(p.cur_dir, 'reachable_mollweide_300m.tif')
    if not hb.path_exists(access_path):
        hb.log('ntfp: burning the reachable polygon onto the grid')
        # Centre rule again, matching the source module. An extent mask elsewhere in the library
        # burns every cell it touches, because a habitat sliver narrower than a cell would
        # otherwise vanish; a dissolved 10 km buffer is nowhere near that, so the choice only
        # moves single cells along its perimeter.
        rasterize_polygon_to_grid(union_path, lulc_path, access_path, all_touched=False)

    hectares = accessible_forest_hectares_by_country(
        lulc_path, access_path, countries_path, COUNTRY_ID_MAX, ndvi_path=ndvi_path)

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

"""
Coastal carbon pipeline helpers.

All ecosystem-density and per-EEZ stock primitives used by the four-file
pipeline (run / initialization / tasks / functions) live here. There are no
runtime dependencies on the per-ecosystem reference modules in `archieve/`.
"""

import contextlib
import json
import mmap
import os
import shutil
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod
import rasterio
import rasterio.features
import rasterio.warp
import rasterio.windows
import shapely
from rasterio.enums import Resampling
from shapely.geometry import box
from tqdm import tqdm


SEAGRASS_RASTER_ALGORITHM_VERSION = 'raster-eez-v2'
EXACT_COVERAGE_ALGORITHM_VERSION = 'exact-coverage-bounds-covers-v1'
SALT_MARSH_AREA_ALGORITHM_VERSION = 'cell-zone-area-covers-checkpoint-v1'
SEAGRASS_ID_COLUMN = 'eemarine_r566_id'
SEAGRASS_LABEL_COLUMN = 'eemarine_r566_label'
SEAGRASS_GEOD = Geod(ellps='WGS84')


def normalize_integer_id_column(dataframe, id_column):
    """Return a copy with finite, integer-valued IDs stored as int64."""
    if id_column not in dataframe.columns:
        raise KeyError(f'Missing ID column: {id_column!r}')

    numeric_ids = pd.to_numeric(dataframe[id_column], errors='raise')
    if numeric_ids.isna().any():
        raise ValueError(f'{id_column} must be non-null')
    values = numeric_ids.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f'{id_column} must be finite')
    if not np.equal(values, np.floor(values)).all():
        raise ValueError(f'{id_column} must contain integer-valued IDs')

    normalized = dataframe.copy()
    normalized[id_column] = numeric_ids.astype(np.int64)
    return normalized


# ============================================================================
# Per-pixel ecosystem carbon density helpers
# ============================================================================

# Mangrove (Hamilton & Friess 2018 + IPCC 2014 zone BGB + Sanderman 2018 SOC)
MANGROVE_BGB_AGB_RATIO_TROPICAL_WET = 0.49
MANGROVE_BGB_AGB_RATIO_TROPICAL_DRY = 0.29
MANGROVE_BGB_AGB_RATIO_SUBTROPICAL = 0.96
MANGROVE_TROPICAL_LAT_THRESHOLD = 23.5
MANGROVE_TROPICAL_WET_PRECIP_THRESHOLD_MM = 2000.0
MANGROVE_AGB_TO_C = 0.48   # Hamilton & Friess 2018
MANGROVE_BGB_TO_C = 0.39   # Howard et al. 2014
MANGROVE_SOIL_SOURCE_DEPTH_CM = 100
MANGROVE_SOIL_C_DEPTH_CM = 30
MANGROVE_CARBON_STOCK_ALGORITHM_VERSION = 'mangrove_30cm_v1'


def calculate_mangrove_density_array(latitude_arr, precipitation_arr=None,
                                     soc_arr=None):
    """
    Vectorised mangrove total carbon density (Mg C/ha).

    AGB: Hamilton & Friess (2018) EQ5 latitude regression
        AGB (t/ha) = max(0, -6.4305 * |lat| + 271.747)
    BGB: AGB x IPCC 2014 zone-specific ratio (0.49 tropical wet, 0.29 tropical
         dry, 0.96 subtropical). If precipitation_arr is None, all tropics use
         the wet ratio (0.49). Tropical/subtropical split at |lat| = 23.5 deg.
    SOC: from the Sanderman 2018 0--100 cm raster when provided; otherwise a
         0--100 cm latitude step function. Both are scaled to the 30-cm
         reference depth using a linear depth-profile assumption.
    """
    lat_abs = np.abs(latitude_arr).astype(np.float64)
    agb_t_per_ha = np.maximum(0.0, -6.4305 * lat_abs + 271.747)

    is_tropical = lat_abs < MANGROVE_TROPICAL_LAT_THRESHOLD
    if precipitation_arr is not None:
        is_wet = precipitation_arr > MANGROVE_TROPICAL_WET_PRECIP_THRESHOLD_MM
        bgb_ratio = np.where(
            is_tropical & is_wet, MANGROVE_BGB_AGB_RATIO_TROPICAL_WET,
            np.where(is_tropical & ~is_wet, MANGROVE_BGB_AGB_RATIO_TROPICAL_DRY,
                     MANGROVE_BGB_AGB_RATIO_SUBTROPICAL),
        )
    else:
        bgb_ratio = np.where(
            is_tropical, MANGROVE_BGB_AGB_RATIO_TROPICAL_WET,
            MANGROVE_BGB_AGB_RATIO_SUBTROPICAL,
        )

    bgb_t_per_ha = agb_t_per_ha * bgb_ratio
    agb_c = agb_t_per_ha * MANGROVE_AGB_TO_C
    bgb_c = bgb_t_per_ha * MANGROVE_BGB_TO_C

    soil_depth_scale = (
        MANGROVE_SOIL_C_DEPTH_CM / MANGROVE_SOIL_SOURCE_DEPTH_CM
    )
    if soc_arr is not None:
        # Scale Sanderman 1 m data to 30 cm assuming linear depth profile.
        soc = soc_arr.astype(np.float64) * soil_depth_scale
    else:
        # Fallback values are 1 m estimates, scaled to the 30-cm reference depth.
        soc = np.full_like(lat_abs, 250.0, dtype=np.float64)
        soc = np.where(lat_abs < 25, 300.0, soc)
        soc = np.where(lat_abs < 20, 350.0, soc)
        soc = np.where(lat_abs < 15, 375.0, soc)
        soc = np.where(lat_abs < 10, 400.0, soc)
        soc *= soil_depth_scale

    return {
        'agb_c_mg_per_ha':   agb_c,
        'bgb_c_mg_per_ha':   bgb_c,
        'soil_c_mg_per_ha':  soc,
        'total_c_mg_per_ha': agb_c + bgb_c + soc,
    }


# Salt marsh (Chmura 2003 AGB + 2.5 BGB ratio + Maxwell 2024 MarSOC)
SALT_MARSH_AGB_MEDIAN_T_HA = 5.0
SALT_MARSH_BGB_AGB_RATIO = 2.5
SALT_MARSH_AGB_TO_C = 0.45
SALT_MARSH_BGB_TO_C = 0.41
SALT_MARSH_TROPICAL_BOOST_LAT = 25.0
SALT_MARSH_TROPICAL_BOOST_FACTOR = 1.2
SALT_MARSH_SOIL_SOURCE_DEPTH_CM = 100
SALT_MARSH_SOIL_C_DEPTH_CM = 30
SALT_MARSH_CARBON_STOCK_ALGORITHM_VERSION = 'salt_marsh_30cm_v1'


def calculate_salt_marsh_density_array(latitude_arr, soc_arr=None):
    """
    Vectorised salt marsh total carbon density (Mg C/ha).

    AGB: Chmura et al. 2003 median (5 t/ha) with 1.2x tropical boost (|lat| < 25).
    BGB: AGB x 2.5 (extensive root systems).
    SOC: from the Maxwell 2024 0--100 cm MarSOC raster when provided;
         otherwise a 0--100 cm latitude step function. Both are scaled to
         the 30-cm reference depth using a linear depth-profile assumption.
    """
    lat_abs = np.abs(latitude_arr).astype(np.float64)

    agb_t_per_ha = np.where(
        lat_abs < SALT_MARSH_TROPICAL_BOOST_LAT,
        SALT_MARSH_AGB_MEDIAN_T_HA * SALT_MARSH_TROPICAL_BOOST_FACTOR,
        SALT_MARSH_AGB_MEDIAN_T_HA,
    ).astype(np.float64)
    bgb_t_per_ha = agb_t_per_ha * SALT_MARSH_BGB_AGB_RATIO
    agb_c = agb_t_per_ha * SALT_MARSH_AGB_TO_C
    bgb_c = bgb_t_per_ha * SALT_MARSH_BGB_TO_C

    soil_depth_scale = (
        SALT_MARSH_SOIL_C_DEPTH_CM / SALT_MARSH_SOIL_SOURCE_DEPTH_CM
    )
    if soc_arr is not None:
        # Scale MarSOC 1 m data to 30 cm assuming linear depth profile.
        soc = soc_arr.astype(np.float64) * soil_depth_scale
    else:
        # Fallback values are 1 m estimates, scaled to the 30-cm reference depth.
        soc = np.full_like(lat_abs, 350.0, dtype=np.float64)
        soc = np.where(lat_abs < 45, 250.0, soc)
        soc = np.where(lat_abs < 35, 220.0, soc)
        soc = np.where(lat_abs < 20, 180.0, soc)
        soc *= soil_depth_scale

    return {
        'agb_c_mg_per_ha':   agb_c,
        'bgb_c_mg_per_ha':   bgb_c,
        'soil_c_mg_per_ha':  soc,
        'total_c_mg_per_ha': agb_c + bgb_c + soc,
    }


# GlobalSeagrass2019_2020 has binary presence values only.
SEAGRASS_BIOMASS_C_MG_HA = 1.55  # Gomis et al. 2025 global mean
SEAGRASS_BGB_FRACTION = 0.70
SEAGRASS_AGB_FRACTION = 0.30
SEAGRASS_SOIL_C_DEPTH_CM = 30
SEAGRASS_SOIL_C_MG_HA_30CM = 24.2  # Krause et al. 2025 global median
SEAGRASS_SOIL_C_MG_HA_30CM_IQR = (12.4, 44.9)  # Krause et al. 2025
SEAGRASS_CARBON_STOCK_ALGORITHM_VERSION = 'krause2025_30cm_v1'


def calculate_global_seagrass_pool_densities():
    """Return GlobalSeagrass pool densities in Mg C/ha.

    Soil carbon uses the Krause et al. (2025) global median for the top
    30 cm. The binary extent raster has no genus or functional-group fields,
    so this remains a Tier 1 global density rather than a taxon-specific map.
    """
    agb = SEAGRASS_BIOMASS_C_MG_HA * SEAGRASS_AGB_FRACTION
    bgb = SEAGRASS_BIOMASS_C_MG_HA * SEAGRASS_BGB_FRACTION
    soil = SEAGRASS_SOIL_C_MG_HA_30CM
    return {
        'agb_c_mg_per_ha': agb,
        'bgb_c_mg_per_ha': bgb,
        'soil_c_mg_per_ha': soil,
        'total_c_mg_per_ha': agb + bgb + soil,
    }


# ============================================================================
# Raster utilities
# ============================================================================

def rasterize_polygons_to_template(vector_path, template_raster_path, out_path,
                                   field=None, default_value=1, dtype='uint8',
                                   nodata=0, all_touched=False,
                                   filter_column=None, filter_suffix=None):
    """
    Burn polygons from `vector_path` onto the grid of `template_raster_path`.

    Defaults produce a center-based binary mask. Pass `field` to burn a numeric
    attribute (e.g., EEZ id). Set `filter_column` and `filter_suffix` to
    rasterize only matching source rows. Vectors are reprojected into the
    template CRS as needed.
    """
    gdf = gpd.read_file(vector_path)
    if filter_suffix is not None:
        if filter_column is None or filter_column not in gdf.columns:
            raise ValueError(
                f'Filter column missing from {vector_path}: {filter_column!r}'
            )
        gdf = gdf.loc[
            gdf[filter_column].astype(str).str.endswith(filter_suffix)
        ].copy()
        if gdf.empty:
            raise ValueError(
                f'No rows match {filter_column} suffix {filter_suffix!r} '
                f'in {vector_path}'
            )
    with rasterio.open(template_raster_path) as src:
        template_crs = src.crs
        if gdf.crs != template_crs:
            gdf = gdf.to_crs(template_crs)
        meta = src.meta.copy()
        meta.update({'count': 1, 'dtype': dtype, 'nodata': nodata, 'compress': 'lzw'})
        if field is None:
            shapes = ((geom, default_value) for geom in gdf.geometry if geom is not None)
        else:
            shapes = (
                (geom, val)
                for geom, val in zip(gdf.geometry, gdf[field])
                if geom is not None and pd.notna(val)
            )
        arr = rasterio.features.rasterize(
            shapes=shapes,
            out_shape=(src.height, src.width),
            fill=nodata,
            transform=src.transform,
            dtype=dtype,
            all_touched=all_touched,
        )
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(arr, 1)
    return out_path


def rasterize_exact_coverage_to_template(vector_path, template_raster_path,
                                         out_path, chunk_rows=512,
                                         chunk_cols=512):
    """Write fractional polygon coverage on a template grid.

    Each output value is the planar intersection fraction of the source
    polygon union and one template cell. The template must use an unrotated,
    north-up affine transform. Coverage is float32 in [0, 1].
    """
    if chunk_rows <= 0 or chunk_cols <= 0:
        raise ValueError("chunk_rows and chunk_cols must be positive")
    if chunk_rows % 16 or chunk_cols % 16:
        raise ValueError("chunk_rows and chunk_cols must be multiples of 16")

    gdf = gpd.read_file(vector_path)
    if gdf.empty:
        raise ValueError(f"No geometries found: {vector_path}")
    if gdf.crs is None:
        raise ValueError(f"Vector CRS missing: {vector_path}")

    marker_path = f'{out_path}.complete.json'
    temp_path = f'{out_path}.tmp'
    if os.path.exists(marker_path):
        os.unlink(marker_path)

    with rasterio.open(template_raster_path) as template:
        if template.crs is None:
            raise ValueError(f"Template CRS missing: {template_raster_path}")
        if template.transform.b != 0 or template.transform.d != 0:
            raise ValueError("Exact coverage requires an unrotated template grid")
        if gdf.crs != template.crs:
            gdf = gdf.to_crs(template.crs)

        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
        invalid = ~gdf.geometry.is_valid
        if invalid.any():
            gdf.loc[invalid, 'geometry'] = gdf.loc[invalid, 'geometry'].map(
                shapely.make_valid
            )
        gdf = gdf.explode(index_parts=False)
        gdf = gdf[
            gdf.geom_type.isin(['Polygon', 'MultiPolygon'])
            & gdf.geometry.notna()
            & ~gdf.geometry.is_empty
        ].reset_index(drop=True)
        if gdf.empty:
            raise ValueError(f"No polygon geometries found: {vector_path}")

        geoms = gdf.geometry.to_numpy()
        spatial_index = gdf.sindex
        profile = template.profile.copy()
        profile.update({
            'driver': 'GTiff',
            'count': 1,
            'dtype': 'float32',
            'nodata': 0.0,
            'compress': 'deflate',
            'predictor': 3,
            'tiled': True,
            'blockxsize': chunk_cols,
            'blockysize': chunk_rows,
            'BIGTIFF': 'IF_SAFER',
            'SPARSE_OK': True,
        })

        with rasterio.open(temp_path, 'w', **profile) as dst:
            for row_off in tqdm(
                range(0, template.height, chunk_rows),
                desc=f"Exact coverage {os.path.basename(vector_path)}",
            ):
                rows = min(chunk_rows, template.height - row_off)
                for col_off in range(0, template.width, chunk_cols):
                    cols = min(chunk_cols, template.width - col_off)
                    window = rasterio.windows.Window(col_off, row_off, cols, rows)
                    bounds = rasterio.windows.bounds(window, template.transform)
                    candidate_ids = list(spatial_index.intersection(bounds))
                    if not candidate_ids:
                        continue

                    tile = box(*bounds)
                    local = shapely.intersection(
                        shapely.union_all(geoms[candidate_ids]), tile
                    )
                    if local.is_empty or shapely.area(local) == 0:
                        continue

                    win_transform = template.window_transform(window)

                    pixel_width = abs(win_transform.a)
                    pixel_height = abs(win_transform.e)
                    coverage = np.zeros((rows, cols), dtype='float32')
                    cell_area = pixel_width * pixel_height

                    # A tile can contain thousands of disjoint GMW patches.
                    # Intersecting one large MultiPolygon with every candidate
                    # cell causes GEOS to repeatedly scan all patches and can
                    # effectively hang on dense coastal tiles. Split the exact
                    # union into disjoint polygon parts, then process each part
                    # only over its local cell range. Additive accumulation is
                    # required because multiple disjoint parts can occupy one
                    # target cell.
                    for part in shapely.get_parts(local):
                        minx, miny, maxx, maxy = shapely.bounds(part)
                        col_min_coord = (minx - win_transform.c) / pixel_width
                        col_max_coord = (maxx - win_transform.c) / pixel_width
                        row_min_coord = (win_transform.f - maxy) / pixel_height
                        row_max_coord = (win_transform.f - miny) / pixel_height

                        row_start = max(0, int(np.floor(row_min_coord)) - 1)
                        row_stop = min(rows, int(np.ceil(row_max_coord)) + 2)
                        col_start = max(0, int(np.floor(col_min_coord)) - 1)
                        col_stop = min(cols, int(np.ceil(col_max_coord)) + 2)
                        if row_start >= row_stop or col_start >= col_stop:
                            continue

                        candidate_rows, candidate_cols = np.meshgrid(
                            np.arange(row_start, row_stop),
                            np.arange(col_start, col_stop),
                            indexing='ij',
                        )
                        x0 = win_transform.c + candidate_cols * win_transform.a
                        x1 = win_transform.c + (candidate_cols + 1) * win_transform.a
                        y0 = win_transform.f + candidate_rows * win_transform.e
                        y1 = win_transform.f + (candidate_rows + 1) * win_transform.e
                        cells = shapely.box(
                            np.minimum(x0, x1), np.minimum(y0, y1),
                            np.maximum(x0, x1), np.maximum(y0, y1),
                        ).ravel()
                        flat_rows = candidate_rows.ravel()
                        flat_cols = candidate_cols.ravel()

                        # `covers` is an exact full-cell test. It replaces the
                        # negative buffer shortcut without classifying a
                        # center-hit partial cell as full.
                        shapely.prepare(part)
                        full = np.asarray(shapely.covers(part, cells), dtype=bool)
                        values = np.zeros(len(cells), dtype='float32')
                        values[full] = 1.0

                        edge = ~full
                        if edge.any():
                            # Bounding boxes can still include many cells that
                            # the part does not touch, especially for narrow
                            # coastal fragments. Prepared intersection removes
                            # those cells before invoking expensive GEOS
                            # polygon clipping.
                            edge_intersects = edge & np.asarray(
                                shapely.intersects(part, cells), dtype=bool
                            )
                        else:
                            edge_intersects = edge
                        if edge_intersects.any():
                            clipped = shapely.area(
                                shapely.intersection(cells[edge_intersects], part)
                            )
                            values[edge_intersects] = np.clip(
                                clipped / cell_area, 0.0, 1.0
                            ).astype('float32')

                        # Parts come from exact union, so their interiors do
                        # not overlap. Sum contributions when several parts
                        # fall in the same target cell.
                        np.add.at(coverage, (flat_rows, flat_cols), values)

                    coverage = np.clip(coverage, 0.0, 1.0)

                    if coverage.any():
                        dst.write(coverage, 1, window=window)
    # Publish only after every output block has been written. A killed run
    # leaves the temporary raster without a completion marker, so reruns
    # cannot silently accept a partial coverage raster.
    os.replace(temp_path, out_path)
    with open(marker_path, 'w', encoding='utf-8') as marker:
        json.dump({
            'algorithm_version': EXACT_COVERAGE_ALGORITHM_VERSION,
            'source': _file_fingerprint(vector_path),
            'template': _file_fingerprint(template_raster_path),
        }, marker, indent=2, sort_keys=True)
    return out_path


def _file_fingerprint(path):
    """Return stable file identity metadata for derived-output validation."""
    stat = os.stat(path)
    return {
        'path': os.path.abspath(path),
        'size': stat.st_size,
        'mtime_ns': stat.st_mtime_ns,
    }


def is_exact_coverage_raster_complete(out_path, vector_path,
                                      template_raster_path):
    """Return whether exact-coverage output matches current inputs/version."""
    marker_path = f'{out_path}.complete.json'
    if not os.path.exists(out_path) or not os.path.exists(marker_path):
        return False
    try:
        with open(marker_path, encoding='utf-8') as marker:
            metadata = json.load(marker)
        if metadata.get('algorithm_version') != EXACT_COVERAGE_ALGORITHM_VERSION:
            return False
        if metadata.get('source') != _file_fingerprint(vector_path):
            return False
        if metadata.get('template') != _file_fingerprint(template_raster_path):
            return False
        with rasterio.open(out_path) as raster:
            return (
                raster.count == 1
                and raster.dtypes[0] == 'float32'
                and raster.nodata == 0.0
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def intersect_polygon_layer_with_zones(feature_gdf, zone_gdf,
                                       zone_id_column,
                                       description='intersecting polygons'):
    """Return exact feature-zone intersections with only zone IDs retained.

    Spatial-index candidates are intersected with each zone using vectorized
    Shapely operations. This avoids the large attribute-overlay table while
    preserving exact geometry for downstream equal-area measurements.
    """
    if feature_gdf.crs is None or zone_gdf.crs is None:
        raise ValueError('Feature and zone layers must both define CRS')
    if feature_gdf.crs != zone_gdf.crs:
        feature_gdf = feature_gdf.to_crs(zone_gdf.crs)

    feature_gdf = feature_gdf[
        feature_gdf.geometry.notna() & ~feature_gdf.geometry.is_empty
    ].copy()
    invalid = ~feature_gdf.geometry.is_valid
    if invalid.any():
        feature_gdf.loc[invalid, 'geometry'] = feature_gdf.loc[
            invalid, 'geometry'
        ].map(shapely.make_valid)
    feature_gdf = feature_gdf.explode(index_parts=False)
    feature_gdf = feature_gdf[
        feature_gdf.geom_type.isin(['Polygon', 'MultiPolygon'])
        & feature_gdf.geometry.notna()
        & ~feature_gdf.geometry.is_empty
    ].reset_index(drop=True)
    if feature_gdf.empty:
        return gpd.GeoDataFrame(
            columns=[zone_id_column, 'geometry'], crs=zone_gdf.crs
        )

    zone_gdf = zone_gdf[
        zone_gdf.geometry.notna() & ~zone_gdf.geometry.is_empty
    ].copy()
    invalid = ~zone_gdf.geometry.is_valid
    if invalid.any():
        zone_gdf.loc[invalid, 'geometry'] = zone_gdf.loc[
            invalid, 'geometry'
        ].map(shapely.make_valid)

    feature_geometries = feature_gdf.geometry.to_numpy()
    feature_sindex = feature_gdf.sindex
    intersections = []
    for _, zone in tqdm(
        zone_gdf.iterrows(), total=len(zone_gdf), desc=description
    ):
        candidate_ids = feature_sindex.query(
            zone.geometry, predicate='intersects'
        )
        if candidate_ids.size == 0:
            continue
        clipped = shapely.intersection(
            feature_geometries[candidate_ids], zone.geometry
        )
        valid = np.fromiter(
            (
                geometry is not None
                and not geometry.is_empty
                and shapely.area(geometry) > 0
                for geometry in clipped
            ),
            dtype=bool,
            count=len(clipped),
        )
        if valid.any():
            intersections.append(
                gpd.GeoDataFrame(
                    {zone_id_column: np.repeat(
                        zone[zone_id_column], int(valid.sum())
                    )},
                    geometry=clipped[valid],
                    crs=zone_gdf.crs,
                )
            )

    if not intersections:
        return gpd.GeoDataFrame(
            columns=[zone_id_column, 'geometry'], crs=zone_gdf.crs
        )
    return gpd.GeoDataFrame(
        pd.concat(intersections, ignore_index=True), crs=zone_gdf.crs
    )


def _quote_sql_identifier(identifier):
    """Quote a trusted GeoPackage table or column identifier for SQLite."""
    return '"' + str(identifier).replace('"', '""') + '"'


def _get_feature_tile_fid_ranges(feature_path, layer_name, tile_column='tile_name',
                                 fid_column='fid'):
    """Return contiguous FID ranges grouped by a source tile column."""
    # SQLite's URI parser does not reliably decode percent-escaped spaces from
    # pathlib.Path.as_uri() on macOS. Keep raw absolute path after file: prefix.
    database_uri = f'file:{Path(feature_path).resolve()}?mode=ro'
    layer_sql = _quote_sql_identifier(layer_name)
    tile_sql = _quote_sql_identifier(tile_column)
    fid_sql = _quote_sql_identifier(fid_column)
    query = (
        f'SELECT {tile_sql}, COUNT(*), MIN({fid_sql}), MAX({fid_sql}) '
        f'FROM {layer_sql} '
        f'WHERE {tile_sql} IS NOT NULL '
        f'GROUP BY {tile_sql} ORDER BY MIN({fid_sql})'
    )
    with sqlite3.connect(database_uri, uri=True) as connection:
        rows = connection.execute(query).fetchall()
    ranges = [
        {
            'tile_name': str(tile_name),
            'feature_count': int(feature_count),
            'fid_min': int(fid_min),
            'fid_max': int(fid_max),
        }
        for tile_name, feature_count, fid_min, fid_max in rows
    ]
    non_contiguous = [
        item for item in ranges
        if item['fid_max'] - item['fid_min'] + 1 != item['feature_count']
    ]
    if non_contiguous:
        raise ValueError(
            'Tile FIDs are not contiguous; refusing unsafe range partition: '
            f"{non_contiguous[0]['tile_name']}"
        )
    return ranges


def _safe_checkpoint_name(tile_name):
    """Create a stable, filesystem-safe checkpoint stem from a tile name."""
    stem = Path(str(tile_name)).stem
    safe = ''.join(
        character if character.isalnum() or character in ('-', '_') else '_'
        for character in stem
    )
    return safe or 'tile'


def _prepare_cell_tile(feature_gdf, zone_crs):
    """Clean cell polygons using same validity rules as exact overlay helper."""
    if feature_gdf.crs is None:
        raise ValueError('Feature layer must define CRS')
    if feature_gdf.crs != zone_crs:
        feature_gdf = feature_gdf.to_crs(zone_crs)
    feature_gdf = feature_gdf[
        feature_gdf.geometry.notna() & ~feature_gdf.geometry.is_empty
    ].copy()
    invalid = ~feature_gdf.geometry.is_valid
    if invalid.any():
        feature_gdf.loc[invalid, 'geometry'] = feature_gdf.loc[
            invalid, 'geometry'
        ].map(shapely.make_valid)
    feature_gdf = feature_gdf.explode(index_parts=False)
    feature_gdf = feature_gdf[
        feature_gdf.geom_type.isin(['Polygon', 'MultiPolygon'])
        & feature_gdf.geometry.notna()
        & ~feature_gdf.geometry.is_empty
    ].reset_index(drop=True)
    return feature_gdf


def _calculate_cell_tile_zone_areas(feature_path, layer_name, fid_min, fid_max,
                                    zone_path, zone_id_column, area_crs,
                                    zone_label_column=None,
                                    zone_label_suffix=None):
    """Calculate exact cell-zone areas for one source tile.

    Cells completely covered by a zone use their precomputed equal-area cell
    area. Only boundary cells enter GEOS intersection. Each zone is processed
    independently so overlapping/shared EEZ assignments remain unchanged.
    """
    feature_gdf = gpd.read_file(
        feature_path,
        layer=layer_name,
        where=f'fid >= {int(fid_min)} AND fid <= {int(fid_max)}',
        columns=[],
        use_arrow=True,
    )
    zone_columns = [zone_id_column]
    if zone_label_column and zone_label_column not in zone_columns:
        zone_columns.append(zone_label_column)
    zone_gdf = gpd.read_file(zone_path, columns=zone_columns, use_arrow=True)
    if zone_label_suffix is not None:
        if zone_label_column is None or zone_label_column not in zone_gdf.columns:
            raise ValueError(
                f'Zone layer missing filter column {zone_label_column!r}'
            )
        zone_gdf = zone_gdf.loc[
            zone_gdf[zone_label_column].astype(str).str.endswith(zone_label_suffix)
        ].copy()
    if feature_gdf.empty:
        raise ValueError(
            f'No features read for FID range {fid_min}:{fid_max} '
            f'in {feature_path}'
        )
    if zone_gdf.crs is None:
        raise ValueError('Zone layer must define CRS')
    zone_gdf = zone_gdf[
        zone_gdf.geometry.notna() & ~zone_gdf.geometry.is_empty
    ][[zone_id_column, 'geometry']].copy()
    invalid = ~zone_gdf.geometry.is_valid
    if invalid.any():
        zone_gdf.loc[invalid, 'geometry'] = zone_gdf.loc[
            invalid, 'geometry'
        ].map(shapely.make_valid)

    feature_gdf = _prepare_cell_tile(feature_gdf, zone_gdf.crs)
    if feature_gdf.empty:
        raise ValueError(
            f'No polygon features remain for FID range {fid_min}:{fid_max}'
        )

    feature_geometries = feature_gdf.geometry.to_numpy()
    feature_areas_m2 = gpd.GeoSeries(
        feature_geometries, crs=zone_gdf.crs
    ).to_crs(epsg=area_crs).area.to_numpy(dtype=np.float64)
    feature_sindex = feature_gdf.sindex

    rows = []
    total_candidates = 0
    total_full_cells = 0
    total_boundary_cells = 0
    total_positive_boundary_intersections = 0
    for _, zone in zone_gdf.iterrows():
        candidate_ids = np.asarray(
            feature_sindex.query(zone.geometry, predicate='intersects'),
            dtype=np.int64,
        )
        total_candidates += int(candidate_ids.size)
        area_m2 = 0.0
        if candidate_ids.size:
            candidate_geometries = feature_geometries[candidate_ids]
            full = np.asarray(
                shapely.covers(zone.geometry, candidate_geometries),
                dtype=bool,
            )
            full_ids = candidate_ids[full]
            total_full_cells += int(full_ids.size)
            area_m2 += float(feature_areas_m2[full_ids].sum())

            boundary_ids = candidate_ids[~full]
            total_boundary_cells += int(boundary_ids.size)
            if boundary_ids.size:
                clipped = shapely.intersection(
                    feature_geometries[boundary_ids], zone.geometry
                )
                valid = np.fromiter(
                    (
                        geometry is not None
                        and not geometry.is_empty
                        and shapely.area(geometry) > 0
                        for geometry in clipped
                    ),
                    dtype=bool,
                    count=len(clipped),
                )
                if valid.any():
                    clipped_area_m2 = gpd.GeoSeries(
                        clipped[valid], crs=zone_gdf.crs
                    ).to_crs(epsg=area_crs).area
                    area_m2 += float(clipped_area_m2.sum())
                    total_positive_boundary_intersections += int(valid.sum())

        rows.append({
            zone_id_column: int(zone[zone_id_column]),
            'area_ha': area_m2 / 10000.0,
        })

    result = pd.DataFrame(rows, columns=[zone_id_column, 'area_ha'])
    diagnostics = {
        'algorithm_version': SALT_MARSH_AREA_ALGORITHM_VERSION,
        'fid_min': int(fid_min),
        'fid_max': int(fid_max),
        'source_feature_count': int(len(feature_gdf)),
        'zone_count': int(len(zone_gdf)),
        'candidate_count': total_candidates,
        'full_cell_count': total_full_cells,
        'boundary_cell_count': total_boundary_cells,
        'positive_boundary_intersection_count': total_positive_boundary_intersections,
    }
    return result, diagnostics


def _run_cell_tile_area_job(job):
    """Process one checkpointable tile in a spawned worker process."""
    (
        feature_path, layer_name, tile_range, checkpoint_path,
        zone_path, zone_id_column, area_crs,
        zone_label_column, zone_label_suffix,
    ) = job
    result, diagnostics = _calculate_cell_tile_zone_areas(
        feature_path=feature_path,
        layer_name=layer_name,
        fid_min=tile_range['fid_min'],
        fid_max=tile_range['fid_max'],
        zone_path=zone_path,
        zone_id_column=zone_id_column,
        area_crs=area_crs,
        zone_label_column=zone_label_column,
        zone_label_suffix=zone_label_suffix,
    )
    return tile_range['tile_name'], checkpoint_path, result, diagnostics


def _read_valid_cell_area_checkpoint(path, zone_id_column, expected_zone_ids):
    """Read one completed tile checkpoint, or return None if invalid."""
    try:
        dataframe = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    required = {zone_id_column, 'area_ha'}
    if not required.issubset(dataframe.columns):
        return None
    if len(dataframe) != len(expected_zone_ids):
        return None
    try:
        observed_ids = set(dataframe[zone_id_column].astype(int))
    except (TypeError, ValueError):
        return None
    if observed_ids != set(expected_zone_ids):
        return None
    dataframe[zone_id_column] = dataframe[zone_id_column].astype(int)
    dataframe['area_ha'] = pd.to_numeric(dataframe['area_ha'], errors='raise')
    return dataframe[[zone_id_column, 'area_ha']]


def calculate_cell_zone_areas_checkpointed(
        feature_path, zone_path, zone_id_column, checkpoint_dir,
        layer_name='lulc_target_pixels', tile_column='tile_name',
        fid_column='fid', area_crs=6933, max_workers=None,
        zone_label_column=None, zone_label_suffix=None):
    """Calculate exact cell-zone areas with tile checkpoints.

    The source layer is expected to contain raster-cell polygons grouped by a
    tile column. Work is partitioned by contiguous FID ranges, allowing each
    tile to be resumed independently after interruption. The returned table
    contains only EEZ-level areas; no million-feature intersection layer
    is materialized.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tile_ranges = _get_feature_tile_fid_ranges(
        feature_path, layer_name, tile_column=tile_column,
        fid_column=fid_column,
    )
    if not tile_ranges:
        raise ValueError(f'No tile ranges found in {feature_path}')

    zone_columns = [zone_id_column]
    if zone_label_column and zone_label_column not in zone_columns:
        zone_columns.append(zone_label_column)
    zone_ids_gdf = gpd.read_file(
        zone_path, columns=zone_columns, use_arrow=True
    )
    if zone_label_suffix is not None:
        if zone_label_column is None or zone_label_column not in zone_ids_gdf.columns:
            raise ValueError(
                f'Zone layer missing filter column {zone_label_column!r}'
            )
        zone_ids_gdf = zone_ids_gdf.loc[
            zone_ids_gdf[zone_label_column].astype(str).str.endswith(
                zone_label_suffix
            )
        ].copy()
    expected_zone_ids = sorted(
        zone_ids_gdf[zone_id_column].dropna().astype(int).unique().tolist()
    )
    if not expected_zone_ids:
        raise ValueError(f'No zone IDs found in {zone_path}')

    manifest = {
        'algorithm_version': SALT_MARSH_AREA_ALGORITHM_VERSION,
        'feature_source': _file_fingerprint(feature_path),
        'zone_source': _file_fingerprint(zone_path),
        'zone_id_column': zone_id_column,
        'zone_label_column': zone_label_column,
        'zone_label_suffix': zone_label_suffix,
        'layer_name': layer_name,
        'tile_column': tile_column,
        'fid_column': fid_column,
        'area_crs': int(area_crs),
        'tile_ranges': tile_ranges,
    }
    manifest_path = checkpoint_dir / 'manifest.json'
    reuse_checkpoints = False
    if manifest_path.exists():
        try:
            with manifest_path.open(encoding='utf-8') as manifest_file:
                reuse_checkpoints = json.load(manifest_file) == manifest
        except (OSError, json.JSONDecodeError):
            reuse_checkpoints = False
    _write_json_atomic(manifest, manifest_path)

    completed = {}
    pending = []
    for tile_range in tile_ranges:
        checkpoint_path = checkpoint_dir / (
            f"tile_{_safe_checkpoint_name(tile_range['tile_name'])}.csv"
        )
        if reuse_checkpoints and checkpoint_path.exists():
            dataframe = _read_valid_cell_area_checkpoint(
                checkpoint_path, zone_id_column, expected_zone_ids
            )
            if dataframe is not None:
                completed[tile_range['tile_name']] = dataframe
                continue
        pending.append((tile_range, checkpoint_path))

    print(
        'Salt marsh area checkpoints: '
        f'{len(completed)} reused, {len(pending)} pending',
        flush=True,
    )

    if max_workers is None:
        configured_workers = os.environ.get(
            'COASTAL_CARBON_SALT_MARSH_WORKERS'
        )
        max_workers = int(configured_workers) if configured_workers else min(
            4, os.cpu_count() or 1
        )
    max_workers = max(1, int(max_workers))
    jobs = [
        (
            feature_path, layer_name, tile_range, str(checkpoint_path),
            zone_path, zone_id_column, area_crs,
            zone_label_column, zone_label_suffix,
        )
        for tile_range, checkpoint_path in pending
    ]

    def record_completed_tile(tile_result):
        tile_name, checkpoint_path, dataframe, diagnostics = tile_result
        _write_csv_atomic(dataframe, checkpoint_path)
        _write_json_atomic(
            diagnostics,
            Path(checkpoint_path).with_suffix('.json'),
        )
        completed[tile_name] = dataframe
        print(
            'Salt marsh area tile '
            f'{len(completed)}/{len(tile_ranges)} finished: {tile_name}; '
            f"{diagnostics['source_feature_count']} cells, "
            f"{diagnostics['boundary_cell_count']} boundary cells",
            flush=True,
        )

    if max_workers == 1 or len(pending) <= 1:
        for job in jobs:
            record_completed_tile(_run_cell_tile_area_job(job))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {
                executor.submit(_run_cell_tile_area_job, job): job
                for job in jobs
            }
            for future in as_completed(future_to_item):
                record_completed_tile(future.result())

    if set(completed) != {item['tile_name'] for item in tile_ranges}:
        raise RuntimeError('Tile checkpoint set is incomplete')

    combined = pd.concat(
        [completed[tile_range['tile_name']] for tile_range in tile_ranges],
        ignore_index=True,
    )
    return (
        combined.groupby(zone_id_column, as_index=False)['area_ha'].sum(),
        {
            'algorithm_version': SALT_MARSH_AREA_ALGORITHM_VERSION,
            'tile_count': len(tile_ranges),
            'reused_tile_count': len(tile_ranges) - len(pending),
            'computed_tile_count': len(pending),
            'worker_count': max_workers,
            'checkpoint_dir': str(checkpoint_dir),
        },
    )


def _write_json_atomic(data, path):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    with open(temporary_path, 'w', encoding='utf-8') as file_obj:
        json.dump(data, file_obj, indent=2, sort_keys=True)
    os.replace(temporary_path, path)


def _write_csv_atomic(dataframe, path):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    dataframe.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def write_task_completion_marker(task_dir, task_name, outputs, metadata=None):
    """Publish explicit task-finished marker after all outputs are complete."""
    marker_path = Path(task_dir) / 'task_finished.json'
    payload = {
        'task_name': task_name,
        'finished': True,
        'finished_unix': time.time(),
        'outputs': {
            str(name): os.path.abspath(str(path))
            for name, path in outputs.items()
        },
    }
    if metadata:
        payload['metadata'] = metadata
    _write_json_atomic(payload, marker_path)
    return marker_path


def _candidate_nonempty_seagrass_blocks(tif_path):
    """Find likely nonempty blocks in sparse LZW GlobalSeagrass COGs."""
    from osgeo import gdal

    gdal.UseExceptions()
    dataset = gdal.Open(str(tif_path), gdal.GA_ReadOnly)
    if dataset is None:
        raise RuntimeError(f'Could not open {tif_path}')
    band = dataset.GetRasterBand(1)
    block_width, block_height = band.GetBlockSize()
    blocks_across = (dataset.RasterXSize + block_width - 1) // block_width
    blocks_down = (dataset.RasterYSize + block_height - 1) // block_height

    records = []
    for block_row in range(blocks_down):
        for block_col in range(blocks_across):
            size_string = band.GetMetadataItem(
                f'BLOCK_SIZE_{block_col}_{block_row}', 'TIFF'
            )
            offset_string = band.GetMetadataItem(
                f'BLOCK_OFFSET_{block_col}_{block_row}', 'TIFF'
            )
            if size_string is None or offset_string is None:
                raise RuntimeError(f'Missing TIFF block metadata in {tif_path}.')
            records.append(
                (block_col, block_row, int(offset_string), int(size_string))
            )

    size_counts = Counter(size for _, _, _, size in records)
    empty_size = size_counts.most_common(1)[0][0]
    _, _, reference_offset, reference_size = next(
        record for record in records if record[3] == empty_size
    )
    with open(tif_path, 'rb') as file_obj:
        mapped_file = mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_READ)
        reference_payload = mapped_file[
            reference_offset:reference_offset + reference_size
        ]
        empty_blocks = []
        candidate_blocks = []
        for block_col, block_row, offset, size in records:
            payload = mapped_file[offset:offset + size]
            if size == empty_size and payload == reference_payload:
                empty_blocks.append((block_col, block_row))
            else:
                candidate_blocks.append((block_col, block_row))
        mapped_file.close()

    empty_fraction = len(empty_blocks) / len(records)
    if empty_fraction < 0.50:
        raise RuntimeError(
            f'{Path(tif_path).name}: dominant compressed-zero payload fraction '
            f'is only {empty_fraction:.4f}; sparse-block shortcut is unsafe.'
        )
    return {
        'candidate_blocks': candidate_blocks,
        'empty_blocks': empty_blocks,
        'block_width': block_width,
        'block_height': block_height,
        'empty_size': empty_size,
        'empty_fraction': empty_fraction,
        'total_blocks': len(records),
    }


def _seagrass_pixel_area_ha_by_row(transform, row_offset, row_count):
    """Return WGS84 ellipsoidal area for one pixel in each raster row."""
    longitude_left = 0.0
    longitude_right = abs(transform.a)
    areas = np.empty(row_count, dtype=np.float64)
    for local_row in range(row_count):
        global_row = row_offset + local_row
        latitude_top = transform.f + global_row * transform.e
        latitude_bottom = latitude_top + transform.e
        area_m2, _ = SEAGRASS_GEOD.polygon_area_perimeter(
            [longitude_left, longitude_right, longitude_right, longitude_left],
            [latitude_top, latitude_top, latitude_bottom, latitude_bottom],
        )
        areas[local_row] = abs(area_m2) / 10_000.0
    return areas


def _seagrass_block_window(src, block_col, block_row, block_width,
                           block_height):
    col_offset = block_col * block_width
    row_offset = block_row * block_height
    width = min(block_width, src.width - col_offset)
    height = min(block_height, src.height - row_offset)
    return rasterio.windows.Window(col_offset, row_offset, width, height)


def _validate_empty_seagrass_blocks(src, block_metadata, sample_count=8):
    """Decode modal-size blocks and confirm they contain no positive pixels."""
    empty_blocks = block_metadata['empty_blocks']
    if not empty_blocks:
        raise RuntimeError(f'{Path(src.name).name}: no modal-size blocks found.')
    sample_indices = np.linspace(
        0, len(empty_blocks) - 1,
        min(sample_count, len(empty_blocks)), dtype=int
    )
    for sample_index in sample_indices:
        block_col, block_row = empty_blocks[sample_index]
        window = _seagrass_block_window(
            src, block_col, block_row,
            block_metadata['block_width'], block_metadata['block_height']
        )
        if np.any(src.read(1, window=window) > 0):
            raise RuntimeError(
                f'{Path(src.name).name}: modal compressed-size block contains '
                'positive pixels; sparse-block shortcut is unsafe.'
            )


def _process_seagrass_tile(tif_path, eez):
    start_time = time.monotonic()
    block_metadata = _candidate_nonempty_seagrass_blocks(tif_path)
    area_by_eez = defaultdict(float)
    positive_pixels = 0
    raw_presence_area_ha = 0.0
    unique_matched_area_ha = 0.0
    overlap_counted_area_ha = 0.0
    decoded_candidate_blocks = 0

    with rasterio.open(tif_path) as src:
        if src.crs is None or src.crs.to_epsg() != 4326:
            raise ValueError(f'{tif_path.name}: expected EPSG:4326, found {src.crs}.')
        if src.count != 1 or src.dtypes[0] != 'uint8' or src.nodata != 0:
            raise ValueError(
                f'{tif_path.name}: expected one uint8 band with nodata 0; '
                f'found count={src.count}, dtype={src.dtypes[0]}, nodata={src.nodata}.'
            )
        if src.block_shapes[0] != (
            block_metadata['block_height'], block_metadata['block_width']
        ):
            raise ValueError(f'{tif_path.name}: GDAL/Rasterio block-shape mismatch.')

        _validate_empty_seagrass_blocks(src, block_metadata)
        tile_box = box(*src.bounds)
        tile_positions = eez.sindex.query(tile_box, predicate='intersects')
        tile_eez = eez.iloc[tile_positions].copy()
        tile_eez['geometry'] = tile_eez.geometry.intersection(tile_box)
        tile_eez = tile_eez[~tile_eez.geometry.is_empty]
        row_area_cache = {}

        for block_col, block_row in block_metadata['candidate_blocks']:
            window = _seagrass_block_window(
                src, block_col, block_row,
                block_metadata['block_width'], block_metadata['block_height']
            )
            presence = src.read(1, window=window) > 0
            if not presence.any():
                continue
            decoded_candidate_blocks += 1
            positive_pixels += int(presence.sum())

            row_offset = int(window.row_off)
            if row_offset not in row_area_cache:
                row_area_cache[row_offset] = _seagrass_pixel_area_ha_by_row(
                    src.transform, row_offset, int(window.height)
                )
            pixel_area_ha = np.broadcast_to(
                row_area_cache[row_offset][:, None], presence.shape
            )
            raw_presence_area_ha += float(pixel_area_ha[presence].sum())

            block_box = box(*rasterio.windows.bounds(window, src.transform))
            block_positions = tile_eez.sindex.query(
                block_box, predicate='intersects'
            )
            block_eez = tile_eez.iloc[block_positions]
            if block_eez.empty:
                continue

            block_transform = rasterio.windows.transform(window, src.transform)
            matched_any = np.zeros(presence.shape, dtype=bool)
            block_assigned_area_ha = 0.0

            for _, zone in block_eez.iterrows():
                clipped_geometry = zone.geometry.intersection(block_box)
                if clipped_geometry.is_empty:
                    continue
                zone_mask = rasterio.features.rasterize(
                    [(clipped_geometry, 1)],
                    out_shape=presence.shape,
                    transform=block_transform,
                    fill=0,
                    dtype='uint8',
                    all_touched=False,
                ).astype(bool)
                selected = presence & zone_mask
                if not selected.any():
                    continue
                selected_area_ha = float(pixel_area_ha[selected].sum())
                area_by_eez[int(zone[SEAGRASS_ID_COLUMN])] += selected_area_ha
                block_assigned_area_ha += selected_area_ha
                matched_any |= selected

            block_unique_area_ha = float(
                pixel_area_ha[presence & matched_any].sum()
            )
            unique_matched_area_ha += block_unique_area_ha
            overlap_counted_area_ha += (
                block_assigned_area_ha - block_unique_area_ha
            )

    tile_result = pd.DataFrame([
        {SEAGRASS_ID_COLUMN: zone_id, 'area_ha': area_ha}
        for zone_id, area_ha in sorted(area_by_eez.items())
    ])
    if tile_result.empty:
        tile_result = pd.DataFrame(columns=[SEAGRASS_ID_COLUMN, 'area_ha'])
    source_stat = Path(tif_path).stat()
    diagnostic = {
        'algorithm_version': SEAGRASS_RASTER_ALGORITHM_VERSION,
        'tile': Path(tif_path).name,
        'source_size_bytes': source_stat.st_size,
        'source_mtime_ns': source_stat.st_mtime_ns,
        'total_blocks': block_metadata['total_blocks'],
        'candidate_blocks': len(block_metadata['candidate_blocks']),
        'decoded_candidate_blocks': decoded_candidate_blocks,
        'empty_block_fraction': block_metadata['empty_fraction'],
        'empty_block_compressed_bytes': block_metadata['empty_size'],
        'positive_pixels': positive_pixels,
        'raw_presence_area_ha': raw_presence_area_ha,
        'unique_matched_area_ha': unique_matched_area_ha,
        'unmatched_area_ha': raw_presence_area_ha - unique_matched_area_ha,
        'overlap_counted_area_ha': overlap_counted_area_ha,
        'summed_eez_area_ha': float(tile_result['area_ha'].sum()),
        'matched_eez_count': len(tile_result),
        'elapsed_seconds': time.monotonic() - start_time,
    }
    return tile_result, diagnostic


def _aggregate_seagrass_tile_results(tile_results):
    frames = []
    for frame in tile_results:
        if not frame.empty:
            frames.append(frame[[SEAGRASS_ID_COLUMN, 'area_ha']])
    if not frames:
        return pd.DataFrame(columns=[SEAGRASS_ID_COLUMN, 'area_ha'])
    return (
        pd.concat(frames, ignore_index=True)
        .groupby(SEAGRASS_ID_COLUMN, as_index=False)['area_ha']
        .sum()
        .sort_values('area_ha', ascending=False)
        .reset_index(drop=True)
    )


def _write_seagrass_final_outputs(eez, area_by_eez, diagnostics, output_dir):
    output_dir = Path(output_dir)
    eez = normalize_integer_id_column(eez, SEAGRASS_ID_COLUMN)
    area_by_eez = normalize_integer_id_column(
        area_by_eez, SEAGRASS_ID_COLUMN
    )
    eez_result = eez.merge(area_by_eez, on=SEAGRASS_ID_COLUMN, how='left')
    eez_result['area_ha'] = eez_result['area_ha'].fillna(0.0)
    eez_result = eez_result.sort_values('area_ha', ascending=False)
    eez_result = normalize_integer_id_column(eez_result, SEAGRASS_ID_COLUMN)

    outputs = {
        'eez_csv': output_dir / 'seagrass_area_by_eez2019.csv',
        'eez_gpkg': output_dir / 'seagrass_area_by_eez2019.gpkg',
        'diagnostics_csv': output_dir / 'tile_diagnostics.csv',
    }
    _write_csv_atomic(eez_result.drop(columns='geometry'), outputs['eez_csv'])
    _write_csv_atomic(pd.DataFrame(diagnostics), outputs['diagnostics_csv'])
    if outputs['eez_gpkg'].exists():
        outputs['eez_gpkg'].unlink()
    eez_result.to_file(outputs['eez_gpkg'], driver='GPKG')
    return outputs


def _remove_legacy_seagrass_outputs(output_dir):
    """Remove persistent per-tile and duplicate pre-EEZ outputs."""
    output_dir = Path(output_dir)
    for directory_name in ('tile_results', 'tile_diagnostics'):
        directory = output_dir / directory_name
        if directory.exists():
            shutil.rmtree(directory)
    for filename in (
        'seagrass_area_by_countries2019.csv',
        'seagrass_area_by_countries2019.gpkg',
    ):
        legacy_path = output_dir / filename
        if legacy_path.exists():
            legacy_path.unlink()


def calculate_global_seagrass_area_by_eez(seagrass_dir, eez_path, output_dir,
                                          max_tiles=None, overwrite=False):
    """Aggregate GlobalSeagrass2019_2020 pixels to marine EEZs.

    Source tiles are binary uint8 EPSG:4326 rasters. Pixel area uses WGS84
    ellipsoidal area by row; EEZ assignment uses pixel-center inclusion.
    Tile results and diagnostics stay in memory until aggregate outputs are
    written, so no persistent per-tile files are created.
    """
    seagrass_dir = Path(seagrass_dir)
    eez_path = Path(eez_path)
    output_dir = Path(output_dir)
    tif_paths = sorted(seagrass_dir.glob('*.tif'))
    if not tif_paths:
        raise FileNotFoundError(f'No TIFF files found in {seagrass_dir}.')
    if max_tiles is not None:
        if max_tiles <= 0:
            raise ValueError('max_tiles must be positive')
        tif_paths = tif_paths[:max_tiles]
    if not eez_path.exists():
        raise FileNotFoundError(eez_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    eez = gpd.read_file(eez_path)
    if eez.crs is None:
        raise ValueError(f'EEZ CRS missing: {eez_path}')
    if eez.crs.to_epsg() != 4326:
        eez = eez.to_crs(epsg=4326)
    eez = eez[
        eez[SEAGRASS_LABEL_COLUMN].astype(str).str.endswith('_EEZ')
    ].copy()
    eez = normalize_integer_id_column(eez, SEAGRASS_ID_COLUMN)
    if len(eez) != 188:
        raise ValueError(f'Expected 188 _EEZ geometries, found {len(eez)}.')
    if eez[SEAGRASS_ID_COLUMN].duplicated().any():
        raise ValueError('EEZ IDs must be non-null and unique.')
    if eez.geometry.isna().any() or eez.geometry.is_empty.any():
        raise ValueError('EEZ geometries must be non-null and nonempty.')
    if not eez.geometry.is_valid.all():
        raise ValueError('EEZ geometry source contains invalid geometries.')
    eez_columns = [SEAGRASS_ID_COLUMN, SEAGRASS_LABEL_COLUMN]
    if 'eemarine_r566_name' in eez.columns:
        eez_columns.append('eemarine_r566_name')
    eez_columns.append('geometry')
    eez = eez[eez_columns].copy()

    diagnostics = []
    tile_results = []
    run_start = time.monotonic()
    for tile_number, tif_path in enumerate(tif_paths, start=1):
        tile_result, diagnostic = _process_seagrass_tile(tif_path, eez)
        tile_results.append(tile_result)
        diagnostics.append(diagnostic)
        print(
            f'[{tile_number:03d}/{len(tif_paths):03d}] computed '
            f'{tif_path.name}: pixels={diagnostic["positive_pixels"]:,}, '
            f'raw={diagnostic["raw_presence_area_ha"]:,.3f} ha, '
            f'EEZ={diagnostic["summed_eez_area_ha"]:,.3f} ha, '
            f'{diagnostic["elapsed_seconds"]:.2f}s',
            flush=True,
        )

    area_by_eez = _aggregate_seagrass_tile_results(tile_results)
    output_paths = _write_seagrass_final_outputs(
        eez, area_by_eez, diagnostics, output_dir
    )
    raw_area_ha = sum(item['raw_presence_area_ha'] for item in diagnostics)
    unique_matched_area_ha = sum(
        item['unique_matched_area_ha'] for item in diagnostics
    )
    summary = {
        'algorithm_version': SEAGRASS_RASTER_ALGORITHM_VERSION,
        'source_dataset': 'GlobalSeagrass2019_2020',
        'source_time_label': '2019_2020',
        'source_directory': str(seagrass_dir),
        'eez_geometry_path': str(eez_path),
        'eez_filter': 'eemarine_r566_label ends with _EEZ',
        'pixel_area_method': 'WGS84 ellipsoidal area by raster row',
        'zone_assignment_method': 'pixel center (all_touched=False)',
        'checkpoint_mode': 'in_memory; no persistent tile files',
        'compatibility_output_year': 2019,
        'source_tile_count': len(tif_paths),
        'computed_tile_count_this_invocation': len(tif_paths),
        'cached_tile_count_this_invocation': 0,
        'eez_count': len(eez),
        'positive_pixel_count': int(sum(item['positive_pixels'] for item in diagnostics)),
        'raw_presence_area_ha': raw_area_ha,
        'unique_matched_area_ha': unique_matched_area_ha,
        'unmatched_area_ha': raw_area_ha - unique_matched_area_ha,
        'summed_eez_area_ha': float(area_by_eez['area_ha'].sum()),
        'overlap_counted_area_ha': sum(
            item['overlap_counted_area_ha'] for item in diagnostics
        ),
        'tile_processing_seconds_total': sum(
            item['elapsed_seconds'] for item in diagnostics
        ),
        'invocation_elapsed_seconds': time.monotonic() - run_start,
        'outputs': {key: str(path) for key, path in output_paths.items()},
    }
    _write_json_atomic(summary, output_dir / 'run_summary.json')
    _remove_legacy_seagrass_outputs(output_dir)
    print(json.dumps(summary, indent=2), flush=True)
    return output_paths


def align_raster_to_template(src_raster_path, template_raster_path, out_path,
                             resampling='average', dtype=None):
    """
    Reproject and resample a source raster to match the grid of a template raster.

    Used to bring external products like Sanderman 2018 SOC or Maxwell 2024
    MarSOC onto the ha_per_cell template so they can be read in lockstep with
    the windowed streaming pass. A completion marker records source,
    template, resampling, and output-dtype metadata so stale aligned rasters
    are rebuilt when inputs change.
    """
    marker_path = f'{out_path}.complete.json'
    expected_metadata = {
        'algorithm_version': 'raster-alignment-v1',
        'source': _file_fingerprint(src_raster_path),
        'template': _file_fingerprint(template_raster_path),
        'resampling': resampling,
        'dtype': dtype,
    }
    if os.path.exists(out_path) and os.path.exists(marker_path):
        try:
            with open(marker_path, encoding='utf-8') as marker:
                if json.load(marker) == expected_metadata:
                    return out_path
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    resampling_method = getattr(Resampling, resampling)
    temporary_path = f'{out_path}.tmp'
    with rasterio.open(template_raster_path) as tmpl, \
         rasterio.open(src_raster_path) as src:
        out_dtype = dtype or src.dtypes[0]
        out_meta = tmpl.meta.copy()
        out_meta.update({
            'count': 1,
            'dtype': out_dtype,
            'nodata': src.nodata if src.nodata is not None else 0,
            'compress': 'lzw',
        })
        with rasterio.open(temporary_path, 'w', **out_meta) as dst:
            rasterio.warp.reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=tmpl.transform,
                dst_crs=tmpl.crs,
                resampling=resampling_method,
            )
    os.replace(temporary_path, out_path)
    _write_json_atomic(expected_metadata, marker_path)
    return out_path


# ============================================================================
# Per-EEZ streaming carbon stock pass
# ============================================================================

def compute_carbon_stock_by_eez(ecosystem_mask_path, eez_id_raster_path,
                                ha_per_cell_path, density_func, eez_ids,
                                chunk_rows=2048, extra_raster_paths=None):
    """
    Stream a global per-pixel carbon stock calculation aggregated per EEZ.

    For each row-block, computes density(latitude, **extras) x ha_per_cell x
    coverage fraction and accumulates per-EEZ totals via np.bincount.

    Parameters
    ----------
    ecosystem_mask_path : str
        Binary or fractional coverage raster aligned to ha_per_cell. Values must
        lie in [0, 1].
    eez_id_raster_path : str
        Integer EEZ-id raster on the same grid (0 = nodata / outside).
    ha_per_cell_path : str
        Hectares-per-cell raster on the same grid; must have a geographic CRS.
    density_func : callable
        density_func(lat_array, **extras) -> dict with 'agb_c_mg_per_ha',
        'bgb_c_mg_per_ha', 'soil_c_mg_per_ha', 'total_c_mg_per_ha'.
    eez_ids : iterable[int]
        Set of valid EEZ IDs (used to size accumulators and order output rows).
    chunk_rows : int
        Row-block size to read per iteration.
    extra_raster_paths : dict[str, str], optional
        Mapping kwarg name -> raster path. Each raster must be aligned to the
        ha_per_cell grid (use align_raster_to_template). The block is read per
        chunk and passed as a kwarg to density_func.

    Returns
    -------
    pandas.DataFrame
        Columns: eemarine_r566_id, agb_c_total_mg, bgb_c_total_mg,
        soil_c_total_mg, total_c_total_mg.
    """
    eez_ids = np.asarray(list(eez_ids), dtype=np.int64)
    if eez_ids.size == 0:
        raise ValueError("eez_ids must contain at least one ID")
    max_id = int(eez_ids.max()) + 1
    extra_raster_paths = extra_raster_paths or {}

    with contextlib.ExitStack() as stack:
        src_ha = stack.enter_context(rasterio.open(ha_per_cell_path))
        src_mask = stack.enter_context(rasterio.open(ecosystem_mask_path))
        src_cid = stack.enter_context(rasterio.open(eez_id_raster_path))
        extra_srcs = {
            name: stack.enter_context(rasterio.open(path))
            for name, path in extra_raster_paths.items()
        }

        if not src_ha.crs.is_geographic:
            raise ValueError(
                f"ha_per_cell raster must be in a geographic CRS (degrees); got {src_ha.crs}"
            )
        rasters = [src_mask, src_cid, *extra_srcs.values()]
        for raster in rasters:
            if (
                raster.crs != src_ha.crs
                or raster.width != src_ha.width
                or raster.height != src_ha.height
                or raster.transform != src_ha.transform
            ):
                raise ValueError(
                    f"Raster is not aligned to {ha_per_cell_path}: {raster.name}"
                )

        height, width = src_ha.height, src_ha.width
        transform = src_ha.transform

        agb_sum = np.zeros(max_id, dtype=np.float64)
        bgb_sum = np.zeros(max_id, dtype=np.float64)
        soil_sum = np.zeros(max_id, dtype=np.float64)
        total_sum = np.zeros(max_id, dtype=np.float64)

        for row_off in tqdm(
            range(0, height, chunk_rows),
            desc=f"Stock {os.path.basename(ecosystem_mask_path)}",
        ):
            rows = min(chunk_rows, height - row_off)
            window = rasterio.windows.Window(0, row_off, width, rows)
            mask_arr = src_mask.read(1, window=window)
            if not np.any(mask_arr > 0):
                continue
            if np.any((mask_arr < 0) | (mask_arr > 1)):
                raise ValueError(
                    f"Coverage values outside [0, 1] in {ecosystem_mask_path}"
                )
            ha_arr = src_ha.read(1, window=window).astype(np.float64)
            cid_arr = src_cid.read(1, window=window)

            row_indices = np.arange(row_off, row_off + rows)
            lats_col = np.array(
                [rasterio.transform.xy(transform, r, 0, offset='center')[1]
                 for r in row_indices],
                dtype=np.float64,
            )
            lat_arr = np.broadcast_to(lats_col[:, None], (rows, width))

            valid = (mask_arr > 0) & (cid_arr > 0) & (ha_arr > 0)
            if not valid.any():
                continue

            extras = {}
            for name, src in extra_srcs.items():
                arr = src.read(1, window=window).astype(np.float64)
                if src.nodata is not None:
                    arr = np.where(arr == src.nodata, np.nan, arr)
                extras[name] = arr

            d = density_func(lat_arr, **extras)
            ha_v = ha_arr[valid]
            coverage_v = mask_arr[valid].astype(np.float64)
            cid_v = cid_arr[valid].astype(np.int64)

            agb_pix = ha_v * coverage_v * d['agb_c_mg_per_ha'][valid]
            bgb_pix = ha_v * coverage_v * d['bgb_c_mg_per_ha'][valid]
            soil_pix = (
                ha_v * coverage_v
                * np.nan_to_num(d['soil_c_mg_per_ha'][valid], nan=0.0)
            )
            total_pix = agb_pix + bgb_pix + soil_pix

            agb_sum += np.bincount(cid_v, weights=agb_pix, minlength=max_id)
            bgb_sum += np.bincount(cid_v, weights=bgb_pix, minlength=max_id)
            soil_sum += np.bincount(cid_v, weights=soil_pix, minlength=max_id)
            total_sum += np.bincount(cid_v, weights=total_pix, minlength=max_id)

    rows_out = []
    for cid in eez_ids:
        cid_i = int(cid)
        rows_out.append({
            'eemarine_r566_id': cid_i,
            'agb_c_total_mg': agb_sum[cid_i],
            'bgb_c_total_mg': bgb_sum[cid_i],
            'soil_c_total_mg': soil_sum[cid_i],
            'total_c_total_mg': total_sum[cid_i],
        })
    return pd.DataFrame(rows_out)


# ============================================================================
# Per-ecosystem stock orchestrators
# ============================================================================

def compute_mangrove_carbon_stock_with_sanderman(
    project_dir,
    mangrove_mask_path,
    eez_id_raster_path,
    ha_per_cell_path,
    eez_ids,
    sanderman_soc_path=None,
    precipitation_path=None,
):
    """
    Single-call orchestrator for per-pixel mangrove carbon stock by EEZ.

    Aligns Sanderman 2018 SOC and (optionally) a precipitation raster onto the
    ha_per_cell grid (cached on disk under project_dir), then streams a global
    carbon stock pass.
    """
    extra_raster_paths = {}

    if sanderman_soc_path and os.path.exists(sanderman_soc_path):
        soc_aligned = os.path.join(project_dir, "sanderman_soc_aligned_10sec.tif")
        align_raster_to_template(sanderman_soc_path, ha_per_cell_path, soc_aligned,
                                 resampling='average', dtype='float32')
        extra_raster_paths['soc_arr'] = soc_aligned
        print(f"  Mangrove SOC source: Sanderman 2018 raster ({sanderman_soc_path})")
    else:
        print("  Mangrove SOC source: latitude step function fallback")

    if precipitation_path and os.path.exists(precipitation_path):
        precip_aligned = os.path.join(project_dir, "precipitation_aligned_10sec.tif")
        align_raster_to_template(precipitation_path, ha_per_cell_path, precip_aligned,
                                 resampling='average', dtype='float32')
        extra_raster_paths['precipitation_arr'] = precip_aligned
        print(f"  Mangrove BGB ratio: IPCC zones using precipitation ({precipitation_path})")
    else:
        print("  Mangrove BGB ratio: latitude-only IPCC zones (tropics treated as wet)")

    return compute_carbon_stock_by_eez(
        ecosystem_mask_path=mangrove_mask_path,
        eez_id_raster_path=eez_id_raster_path,
        ha_per_cell_path=ha_per_cell_path,
        density_func=calculate_mangrove_density_array,
        eez_ids=eez_ids,
        extra_raster_paths=extra_raster_paths,
    )


def compute_salt_marsh_carbon_stock_with_maxwell(
    project_dir,
    salt_marsh_mask_path,
    eez_id_raster_path,
    ha_per_cell_path,
    eez_ids,
    maxwell_soc_path=None,
):
    """
    Single-call orchestrator for per-pixel salt marsh carbon stock by EEZ.

    Aligns the Maxwell et al. 2024 MarSOC raster onto the ha_per_cell grid
    (cached on disk under project_dir), then streams a global carbon stock pass.
    """
    extra_raster_paths = {}

    if maxwell_soc_path and os.path.exists(maxwell_soc_path):
        soc_aligned = os.path.join(project_dir, "maxwell_marsoc_aligned_10sec.tif")
        align_raster_to_template(maxwell_soc_path, ha_per_cell_path, soc_aligned,
                                 resampling='average', dtype='float32')
        extra_raster_paths['soc_arr'] = soc_aligned
        print(f"  Salt marsh SOC source: Maxwell 2024 MarSOC raster ({maxwell_soc_path})")
    else:
        print("  Salt marsh SOC source: latitude step function fallback")

    return compute_carbon_stock_by_eez(
        ecosystem_mask_path=salt_marsh_mask_path,
        eez_id_raster_path=eez_id_raster_path,
        ha_per_cell_path=ha_per_cell_path,
        density_func=calculate_salt_marsh_density_array,
        eez_ids=eez_ids,
        extra_raster_paths=extra_raster_paths,
    )

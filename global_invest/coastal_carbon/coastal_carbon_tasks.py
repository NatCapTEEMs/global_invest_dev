import json
import os
import subprocess
import sys

import geopandas as gpd
import hazelbean as hb
import numpy as np
import pandas as pd

from global_invest.coastal_carbon import coastal_carbon_functions


COASTAL_CARBON_GEP_SCENARIO_COLUMNS = [
    'coastal_carbon_gep_r1pct',
    'coastal_carbon_gep_r1dot5pct',
    'coastal_carbon_gep_r2pct',
    'coastal_carbon_gep_r2dot5pct',
    'coastal_carbon_gep_r3pct',
]

# 2019 int$/Mg C/year values reported in the coastal-carbon discount-rate
# table. Baseline r2% and r3% are read from carbon_prices.xlsx below.
COASTAL_CARBON_FIXED_RENTAL_SCC_2019 = {
    'coastal_carbon_gep_r1pct': 20.11,
    'coastal_carbon_gep_r1dot5pct': 16.50,
    'coastal_carbon_gep_r2dot5pct': 10.49,
}


def _task_outputs_exist(*paths):
    """Return True if every given path exists on disk.

    Used as a skip-if-already-done guard at the top of each task so reruns
    only re-do work for missing outputs. Pass primary output paths only,
    not intermediates.
    """
    return all(bool(p) and os.path.exists(p) for p in paths)


def _outputs_are_current(output_paths, input_paths):
    """Return True when outputs exist and are newer than all inputs."""
    if not _task_outputs_exist(*output_paths):
        return False
    if not _task_outputs_exist(*input_paths):
        return False
    oldest_output = min(os.path.getmtime(path) for path in output_paths)
    newest_input = max(os.path.getmtime(path) for path in input_paths)
    return oldest_output >= newest_input


def _task_marker_is_current(path, task_name, metadata):
    """Return True when completion marker matches task and metadata."""
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding='utf-8') as marker_file:
            payload = json.load(marker_file)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        payload.get('task_name') == task_name
        and payload.get('finished') is True
        and payload.get('metadata', {}) == metadata
    )


def _optional_file_fingerprint(path):
    """Return source identity metadata, including an explicit missing state."""
    if not path:
        return {'path': None, 'exists': False}
    if not os.path.exists(path):
        return {'path': os.path.abspath(path), 'exists': False}
    return coastal_carbon_functions._file_fingerprint(path)


def _eez_zones(gdf):
    """Return marine-zone rows representing national EEZs only."""
    label_column = 'eemarine_r566_label'
    if label_column not in gdf.columns:
        raise ValueError(f'Marine zone layer missing {label_column!r}')
    eez = gdf.loc[
        gdf[label_column].astype(str).str.endswith('_EEZ')
    ].copy()
    if eez.empty:
        raise ValueError('Marine zone layer contains no rows ending in _EEZ')
    eez = coastal_carbon_functions.normalize_integer_id_column(
        eez, 'eemarine_r566_id'
    )
    ids = eez['eemarine_r566_id']
    if ids.duplicated().any():
        raise ValueError('EEZ IDs must be non-null and unique')
    eez_columns = [
        'eemarine_r566_id',
        'eemarine_r566_label',
        'eemarine_r566_name',
        'geometry',
    ]
    return eez[[column for column in eez_columns if column in eez.columns]].copy()


def _eez_ids(p):
    gdf = _eez_zones(gpd.read_file(p.gdf_eez_vector_path))
    return set(gdf['eemarine_r566_id'].dropna().astype(int))


def _csv_has_expected_ids(path, expected_ids):
    if not expected_ids or not os.path.exists(path):
        return False
    try:
        ids = pd.read_csv(path, usecols=['eemarine_r566_id'])[
            'eemarine_r566_id'
        ]
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return False
    if not pd.api.types.is_integer_dtype(ids.dtype):
        return False
    ids = ids.astype(np.int64, copy=False)
    return (
        len(ids) == len(expected_ids)
        and not ids.duplicated().any()
        and set(ids) == expected_ids
    )


def _eez_csv_is_current(path, p):
    if not os.path.exists(path):
        return False
    try:
        columns = pd.read_csv(path, nrows=0).columns
        if not set(COASTAL_CARBON_GEP_SCENARIO_COLUMNS).issubset(columns):
            return False
        if 'eemarine_r566_label' in columns:
            labels = pd.read_csv(
                path, usecols=['eemarine_r566_id', 'eemarine_r566_label']
            )['eemarine_r566_label'].astype(str)
            if labels.empty or not labels.str.endswith('_EEZ').all():
                return False
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return False
    return _csv_has_expected_ids(path, _eez_ids(p))


def _csv_rows_are_eez(path):
    if not os.path.exists(path):
        return False
    try:
        dataframe = pd.read_csv(
            path, usecols=['eemarine_r566_id', 'eemarine_r566_label']
        )
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return False
    labels = dataframe['eemarine_r566_label'].astype(str)
    ids = dataframe['eemarine_r566_id']
    return (
        not dataframe.empty
        and labels.str.endswith('_EEZ').all()
        and pd.api.types.is_integer_dtype(ids.dtype)
        and not ids.duplicated().any()
    )


def task_calculate_mangrove_area_within_eez(p):
    """
    Calculate mangrove area within each marine EEZ.
    Uses Global Mangrove Watch (GMW) vector data.
    """
    p.mangrove_area_by_eez_base_year_path = os.path.join(
        p.cur_dir, "mangrove_area_by_eez2019.gpkg"
    )
    csv_path = p.mangrove_area_by_eez_base_year_path.replace('.gpkg', '.csv')
    detail_path = os.path.join(p.cur_dir, "mangroves_within_eez2019.gpkg")
    if (
        _task_outputs_exist(
            p.mangrove_area_by_eez_base_year_path, csv_path, detail_path
        )
        and _eez_csv_is_current(csv_path, p)
    ):
        hb.log(f"task_calculate_mangrove_area_within_eez: skipped (outputs exist)")
        return

    eez = _eez_zones(
        gpd.read_file(p.gdf_eez_vector_path)
    )
    gdf_mangroves = gpd.read_file(
        p.mangrove_vector_path, columns=[], use_arrow=True
    )

    print(f"Loaded {len(gdf_mangroves)} mangrove polygons")
    print(f"Loaded {len(eez)} EEZ zones")

    gdf_mangroves_within_eez = (
        coastal_carbon_functions.intersect_polygon_layer_with_zones(
            feature_gdf=gdf_mangroves,
            zone_gdf=eez,
            zone_id_column='eemarine_r566_id',
            description='Intersecting mangroves with EEZs',
        )
    )
    if gdf_mangroves_within_eez.empty:
        raise ValueError("No intersections found between mangroves and EEZs")
    gdf_mangroves_within_eez = (
        coastal_carbon_functions.normalize_integer_id_column(
            gdf_mangroves_within_eez, 'eemarine_r566_id'
        )
    )

    # Reproject to equal-area projection for area calculation
    gdf_mangroves_within_eez = gdf_mangroves_within_eez.to_crs(epsg=6933)

    # Calculate area
    gdf_mangroves_within_eez["area_m2"] = gdf_mangroves_within_eez.geometry.area
    gdf_mangroves_within_eez["area_ha"] = gdf_mangroves_within_eez["area_m2"] / 10000

    # Save detailed intersection
    gdf_mangroves_within_eez.to_file(
        detail_path,
        driver="GPKG",
        use_arrow=True,
    )

    # Aggregate by EEZ
    mangrove_area_by_eez_base_year = (
        gdf_mangroves_within_eez.groupby("eemarine_r566_id")["area_ha"]
        .sum()
        .reset_index()
    )

    # Merge only EEZ attributes; no terrestrial or country-level fields.
    mangrove_area_by_eez_base_year = mangrove_area_by_eez_base_year.merge(
        eez, how="right", on="eemarine_r566_id"
    )
    mangrove_area_by_eez_base_year['area_ha'] = (
        mangrove_area_by_eez_base_year['area_ha'].fillna(0)
    )
    mangrove_area_by_eez_base_year = (
        coastal_carbon_functions.normalize_integer_id_column(
            mangrove_area_by_eez_base_year, 'eemarine_r566_id'
        )
    )

    # Create GeoDataFrame with EEZ geometries
    mangrove_area_by_eez_base_year = gpd.GeoDataFrame(
        mangrove_area_by_eez_base_year,
        geometry='geometry',
        crs=eez.crs,
    )

    mangrove_area_by_eez_base_year.to_file(
        p.mangrove_area_by_eez_base_year_path, driver="GPKG"
    )
    mangrove_area_by_eez_base_year.drop(columns=['geometry']).to_csv(csv_path, index=False)
    print(f"✅ Saved mangrove area by EEZ: {p.mangrove_area_by_eez_base_year_path}")


def task_calculate_salt_marsh_area_within_eez(p):
    """
    Calculate salt marsh area within each marine EEZ.
    Uses exact vector intersections and equal-area geometry measurements.
    """
    p.salt_marsh_area_by_eez_base_year_path = os.path.join(
        p.cur_dir, "salt_marsh_area_by_eez2019.gpkg"
    )
    csv_path = p.salt_marsh_area_by_eez_base_year_path.replace('.gpkg', '.csv')
    task_finished_path = os.path.join(p.cur_dir, 'task_finished.json')
    if (
        _task_outputs_exist(
            p.salt_marsh_area_by_eez_base_year_path,
            csv_path,
            task_finished_path,
        )
        and _eez_csv_is_current(csv_path, p)
    ):
        hb.log("task_calculate_salt_marsh_area_within_eez: skipped (outputs exist)")
        return

    # Read EEZ geometries for final EEZ-level GeoPackage output.
    eez = _eez_zones(
        gpd.read_file(p.gdf_eez_vector_path)
    )
    print(f"Loaded {len(eez)} EEZ zones")

    p.salt_marsh_area_checkpoint_dir = os.path.join(
        p.cur_dir, 'salt_marsh_area_checkpoints'
    )
    salt_marsh_area_by_eez, run_metadata = (
        coastal_carbon_functions.calculate_cell_zone_areas_checkpointed(
        feature_path=p.salt_marsh_vector_path,
        zone_path=p.gdf_eez_vector_path,
        zone_id_column='eemarine_r566_id',
        checkpoint_dir=p.salt_marsh_area_checkpoint_dir,
        layer_name='lulc_target_pixels',
        tile_column='tile_name',
        fid_column='fid',
        area_crs=6933,
        max_workers=getattr(p, 'salt_marsh_area_workers', None),
        zone_label_column='eemarine_r566_label',
        zone_label_suffix='_EEZ',
        )
    )
    hb.log(
        'Salt marsh area: '
        f"{run_metadata['tile_count']} tiles, "
        f"{run_metadata['computed_tile_count']} computed, "
        f"{run_metadata['reused_tile_count']} reused, "
        f"{run_metadata['worker_count']} workers"
    )

    if salt_marsh_area_by_eez.empty:
        raise ValueError("No intersections found between salt marsh and EEZs")
    salt_marsh_area_by_eez = coastal_carbon_functions.normalize_integer_id_column(
        salt_marsh_area_by_eez, 'eemarine_r566_id'
    )

    # Merge EEZ geometries only after tile-level area aggregation. The
    # former million-feature salt_marsh_with_area.gpkg is diagnostic-only and
    # is intentionally no longer materialized by this production task.
    salt_marsh_area_by_eez = salt_marsh_area_by_eez.merge(
        eez, how="right", on="eemarine_r566_id"
    )
    salt_marsh_area_by_eez['area_ha'] = (
        salt_marsh_area_by_eez['area_ha'].fillna(0)
    )
    salt_marsh_area_by_eez = coastal_carbon_functions.normalize_integer_id_column(
        salt_marsh_area_by_eez, 'eemarine_r566_id'
    )
    salt_marsh_area_by_eez = gpd.GeoDataFrame(
        salt_marsh_area_by_eez,
        geometry='geometry',
        crs=eez.crs,
    )

    salt_marsh_area_by_eez.to_file(
        p.salt_marsh_area_by_eez_base_year_path,
        driver="GPKG",
        use_arrow=True,
    )
    salt_marsh_area_by_eez.drop(columns=['geometry']).to_csv(
        csv_path, index=False
    )
    coastal_carbon_functions.write_task_completion_marker(
        p.cur_dir,
        'task_calculate_salt_marsh_area_within_eez',
        {
            'area_gpkg': p.salt_marsh_area_by_eez_base_year_path,
            'area_csv': csv_path,
        },
        metadata=run_metadata,
    )
    print(
        f"✅ Saved salt marsh area by EEZ: "
        f"{p.salt_marsh_area_by_eez_base_year_path}"
    )


def _build_eez_id_raster_if_needed(p):
    """Rasterize EEZ polygons to a uint16 ID grid aligned with ha_per_cell.

    Cross-task reuse: if a previous task already set p.eez_id_raster_path
    and the file exists, just reuse it (no rebuild and no path overwrite).
    Otherwise the raster is built into the calling task's p.cur_dir, so the
    first stock task to run becomes the owner of the file and downstream
    callers find it via the shared p.eez_id_raster_path attribute.
    """
    p.eez_id_raster_path = os.path.join(
        p.cur_dir, "eez_id_raster_10sec.tif"
    )
    if os.path.exists(p.eez_id_raster_path):
        return
    coastal_carbon_functions.rasterize_polygons_to_template(
        vector_path=p.gdf_eez_vector_path,
        template_raster_path=p.ha_per_cell_10sec_path,
        out_path=p.eez_id_raster_path,
        field='eemarine_r566_id',
        dtype='uint16',
        nodata=0,
        all_touched=False,
        filter_column='eemarine_r566_label',
        filter_suffix='_EEZ',
    )


def task_calculate_mangrove_carbon_stock(p):
    """
    Compute per-EEZ mangrove carbon stocks via per-pixel density x ha_per_cell.

    Delegates the heavy lifting to
    coastal_carbon_functions.compute_mangrove_carbon_stock_with_sanderman:
      - AGB: Hamilton & Friess (2018) latitude regression
      - BGB: AGB x IPCC 2014 zone-specific ratio (uses p.precipitation_path if set)
      - SOC: Sanderman 0--100 cm raster at p.mangrove_soc_path, scaled to
             30 cm (fallback otherwise)
    """
    p.mangrove_carbon_stock_by_eez_path = os.path.join(
        p.cur_dir, "mangrove_carbon_stock_by_eez2019.csv"
    )
    stock_marker_path = os.path.join(p.cur_dir, 'task_finished.json')
    mangrove_soc_path = getattr(p, 'mangrove_soc_path', None)
    precipitation_path = getattr(p, 'precipitation_path', None)
    mangrove_stock_metadata = {
        'algorithm_version': (
            coastal_carbon_functions.MANGROVE_CARBON_STOCK_ALGORITHM_VERSION
        ),
        'soil_depth_cm': coastal_carbon_functions.MANGROVE_SOIL_C_DEPTH_CM,
        'soil_source_depth_cm': (
            coastal_carbon_functions.MANGROVE_SOIL_SOURCE_DEPTH_CM
        ),
        'soil_source': _optional_file_fingerprint(mangrove_soc_path),
        'precipitation_source': _optional_file_fingerprint(precipitation_path),
    }
    mangrove_area_csv_path = getattr(
        p, 'mangrove_area_by_eez_base_year_path', ''
    ).replace('.gpkg', '.csv')
    if (
        _outputs_are_current(
            [p.mangrove_carbon_stock_by_eez_path, stock_marker_path],
            [mangrove_area_csv_path],
        )
        and _eez_csv_is_current(p.mangrove_carbon_stock_by_eez_path, p)
        and _task_marker_is_current(
            stock_marker_path,
            'task_calculate_mangrove_carbon_stock',
            mangrove_stock_metadata,
        )
    ):
        hb.log("task_calculate_mangrove_carbon_stock: skipped (output exists)")
        return

    _build_eez_id_raster_if_needed(p)

    mangrove_coverage_path = os.path.join(
        p.cur_dir, 'mangrove_coverage_10sec.tif'
    )
    if not coastal_carbon_functions.is_exact_coverage_raster_complete(
            mangrove_coverage_path,
            p.mangrove_vector_path,
            p.ha_per_cell_10sec_path):
        coastal_carbon_functions.rasterize_exact_coverage_to_template(
            vector_path=p.mangrove_vector_path,
            template_raster_path=p.ha_per_cell_10sec_path,
            out_path=mangrove_coverage_path,
        )

    eez = _eez_zones(
        gpd.read_file(p.gdf_eez_vector_path)
    )
    eez_ids = eez['eemarine_r566_id'].dropna().astype(int).unique()

    df_stock = coastal_carbon_functions.compute_mangrove_carbon_stock_with_sanderman(
        project_dir=p.cur_dir,
        mangrove_mask_path=mangrove_coverage_path,
        eez_id_raster_path=p.eez_id_raster_path,
        ha_per_cell_path=p.ha_per_cell_10sec_path,
        eez_ids=eez_ids,
        sanderman_soc_path=mangrove_soc_path,
        precipitation_path=precipitation_path,
    )
    df_stock = df_stock.rename(columns={
        'agb_c_total_mg': 'mangrove_agb_c_total_mg',
        'bgb_c_total_mg': 'mangrove_bgb_c_total_mg',
        'soil_c_total_mg': 'mangrove_soil_c_total_mg',
        'total_c_total_mg': 'mangrove_total_c_stock_mg',
    })
    df_stock = coastal_carbon_functions.normalize_integer_id_column(
        df_stock, 'eemarine_r566_id'
    )
    df_stock.to_csv(p.mangrove_carbon_stock_by_eez_path, index=False)
    coastal_carbon_functions.write_task_completion_marker(
        p.cur_dir,
        'task_calculate_mangrove_carbon_stock',
        {'stock_csv': p.mangrove_carbon_stock_by_eez_path},
        metadata=mangrove_stock_metadata,
    )
    print(f"✅ Saved mangrove carbon stock by EEZ: {p.mangrove_carbon_stock_by_eez_path}")


def _calculate_storage_value(p, stock_csv_path, pool_columns,
                              ecosystem_label, output_csv_filename,
                              expected_zone_ids=None):
    """
    Per-EEZ storage value CSV from per-pool physical stocks.

    Splits the GEP calculation into two stages explicitly:
      - Physical stage: per-pool carbon stock columns (Mg C) read from
        stock_csv_path. Produced upstream by the carbon-stock task.
      - GEP stage: each stock column multiplied by the base-year rental SCC
        to produce a matching value column ($).

    The helper writes one output CSV with all stock columns plus all value
    columns, plus 'year' and the rental SCC column itself.

    Parameters
    ----------
    p : ProjectFlow
    stock_csv_path : str
        Path to the per-EEZ stock CSV (output of the stock task).
    pool_columns : dict[str, str]
        Mapping {stock_column_name: output_value_column_name} in the order
        you want them written. The LAST entry is treated as the 'total' for
        logging purposes.
    ecosystem_label : str
    output_csv_filename : str
        Written under the calling task's p.cur_dir.
    """
    out_path = os.path.join(p.cur_dir, output_csv_filename)
    if (
        _outputs_are_current([out_path], [stock_csv_path])
        and (
            expected_zone_ids is None
            or _csv_has_expected_ids(out_path, expected_zone_ids)
        )
    ):
        hb.log(f"{ecosystem_label} storage value: skipped (output exists at {out_path})")
        return out_path

    if not stock_csv_path or not os.path.exists(stock_csv_path):
        raise FileNotFoundError(
            f"{ecosystem_label} stock CSV not found: {stock_csv_path!r}"
        )
    if (
        expected_zone_ids is not None
        and not _csv_has_expected_ids(stock_csv_path, expected_zone_ids)
    ):
        raise ValueError(
            f"{ecosystem_label} stock CSV is not a complete EEZ-only table: "
            f"{stock_csv_path}"
        )

    df_stock = coastal_carbon_functions.normalize_integer_id_column(
        pd.read_csv(stock_csv_path), 'eemarine_r566_id'
    )

    df_carbon_p = pd.read_excel(p.carbon_prices_path)
    df_carbon_p = df_carbon_p[['year', p.carbon_price]]
    base_year = 2019
    rental_scc = float(
        df_carbon_p.loc[df_carbon_p['year'] == base_year, p.carbon_price].iloc[0]
    )

    df_stock['year'] = base_year
    df_stock[p.carbon_price] = rental_scc

    # Physical -> GEP per pool
    for stock_col, value_col in pool_columns.items():
        df_stock[value_col] = df_stock[stock_col] * rental_scc

    df_stock.to_csv(out_path, index=False)

    # Log per-pool plus total
    pool_items = list(pool_columns.items())
    for stock_col, value_col in pool_items[:-1]:
        pool_total_stock = df_stock[stock_col].sum()
        pool_total_value = df_stock[value_col].sum()
        hb.log(
            f"  {ecosystem_label} {stock_col}: {pool_total_stock:,.0f} Mg C, "
            f"value ${pool_total_value:,.2f}"
        )
    total_stock_col, total_value_col = pool_items[-1]
    total_stock = df_stock[total_stock_col].sum()
    total_value = df_stock[total_value_col].sum()
    hb.log(
        f"{ecosystem_label} TOTAL stock (2019): {total_stock:,.0f} Mg C "
        f"({total_stock / 1e9:.3f} Pg C)"
    )
    hb.log(f"{ecosystem_label} TOTAL storage value (2019): ${total_value:,.2f}")
    print(f"✅ Saved {ecosystem_label.lower()} storage value: {out_path}")
    return out_path


def task_calculate_mangrove_storage_value(p):
    """
    Per-EEZ mangrove storage value (USD), per pool.

    Stage split:
      - Physical part (upstream task_calculate_mangrove_carbon_stock):
            mangrove_agb_c_total_mg, mangrove_bgb_c_total_mg,
            mangrove_soil_c_total_mg, mangrove_total_c_stock_mg.
      - GEP part (this task): each pool stock x rental SCC ->
            mangrove_agb_storage_value, mangrove_bgb_storage_value,
            mangrove_soil_storage_value, mangrove_storage_value.
    """
    p.mangrove_storage_value_path = _calculate_storage_value(
        p,
        stock_csv_path=p.mangrove_carbon_stock_by_eez_path,
        pool_columns={
            'mangrove_agb_c_total_mg':   'mangrove_agb_storage_value',
            'mangrove_bgb_c_total_mg':   'mangrove_bgb_storage_value',
            'mangrove_soil_c_total_mg':  'mangrove_soil_storage_value',
            'mangrove_total_c_stock_mg': 'mangrove_storage_value',
        },
        ecosystem_label='Mangrove',
        output_csv_filename='mangrove_storage_value_by_eez2019.csv',
        expected_zone_ids=_eez_ids(p),
    )


def task_calculate_salt_marsh_storage_value(p):
    """
    Per-EEZ salt marsh storage value (USD), per pool.

    Same physical/GEP split as mangrove. Output columns:
        salt_marsh_agb_storage_value, salt_marsh_bgb_storage_value,
        salt_marsh_soil_storage_value, salt_marsh_storage_value.
    """
    p.salt_marsh_storage_value_path = _calculate_storage_value(
        p,
        stock_csv_path=p.salt_marsh_carbon_stock_by_eez_path,
        pool_columns={
            'salt_marsh_agb_c_total_mg':   'salt_marsh_agb_storage_value',
            'salt_marsh_bgb_c_total_mg':   'salt_marsh_bgb_storage_value',
            'salt_marsh_soil_c_total_mg':  'salt_marsh_soil_storage_value',
            'salt_marsh_total_c_stock_mg': 'salt_marsh_storage_value',
        },
        ecosystem_label='Salt marsh',
        output_csv_filename='salt_marsh_storage_value_by_eez2019.csv',
        expected_zone_ids=_eez_ids(p),
    )


def task_calculate_seagrass_area_within_eez(p):
    """Calculate GlobalSeagrass2019_2020 area within each marine EEZ."""
    # Keep raster-derived outputs under seagrass area task directory.
    p.seagrass_area_dir = os.path.join(
        p.intermediate_dir,
        'task_calculate_seagrass_area_within_eez',
        'seagrass_raster_eez_intersection',
    )
    p.seagrass_area_by_eez_base_year_path = os.path.join(
        p.seagrass_area_dir, 'seagrass_area_by_eez2019.gpkg'
    )
    seagrass_area_csv = p.seagrass_area_by_eez_base_year_path.replace(
        '.gpkg', '.csv'
    )
    seagrass_diagnostics_csv = os.path.join(
        p.seagrass_area_dir, 'tile_diagnostics.csv'
    )
    run_summary_path = os.path.join(p.seagrass_area_dir, 'run_summary.json')
    if (
        _task_outputs_exist(
            p.seagrass_area_by_eez_base_year_path,
            seagrass_area_csv,
            seagrass_diagnostics_csv,
            run_summary_path,
        )
        and _eez_csv_is_current(seagrass_area_csv, p)
    ):
        p.seagrass_area_by_eez_path = p.seagrass_area_by_eez_base_year_path
        hb.log(
            'task_calculate_seagrass_area_within_eez: '
            'skipped (EEZ outputs exist)'
        )
        return
    if not getattr(p, 'seagrass_raster_dir', None):
        raise ValueError('p.seagrass_raster_dir must point to GlobalSeagrass tiles')

    outputs = coastal_carbon_functions.calculate_global_seagrass_area_by_eez(
        seagrass_dir=p.seagrass_raster_dir,
        eez_path=p.gdf_eez_vector_path,
        output_dir=p.seagrass_area_dir,
    )
    p.seagrass_area_by_eez_path = str(outputs['eez_gpkg'])
    p.seagrass_area_by_eez_base_year_path = str(outputs['eez_gpkg'])
    print(
        'Saved GlobalSeagrass area by EEZ: '
        f'{p.seagrass_area_by_eez_base_year_path}'
    )


def task_calculate_seagrass_carbon_stock(p):
    """Calculate per-EEZ seagrass carbon stock from raster-derived area."""
    p.seagrass_carbon_stock_by_eez_path = os.path.join(
        p.cur_dir, "seagrass_carbon_stock_by_eez2019.csv"
    )
    stock_marker_path = os.path.join(p.cur_dir, 'task_finished.json')
    stock_metadata = {
        'algorithm_version': (
            coastal_carbon_functions.SEAGRASS_CARBON_STOCK_ALGORITHM_VERSION
        ),
        'soil_depth_cm': coastal_carbon_functions.SEAGRASS_SOIL_C_DEPTH_CM,
        'soil_c_mg_per_ha': coastal_carbon_functions.SEAGRASS_SOIL_C_MG_HA_30CM,
    }
    area_csv_path = p.seagrass_area_by_eez_base_year_path.replace(
        '.gpkg', '.csv'
    )
    if (
        _outputs_are_current(
            [p.seagrass_carbon_stock_by_eez_path, stock_marker_path],
            [area_csv_path],
        )
        and _eez_csv_is_current(p.seagrass_carbon_stock_by_eez_path, p)
        and _task_marker_is_current(
            stock_marker_path,
            'task_calculate_seagrass_carbon_stock',
            stock_metadata,
        )
    ):
        hb.log("task_calculate_seagrass_carbon_stock: skipped (output exists)")
        return

    if not os.path.exists(area_csv_path):
        raise FileNotFoundError(f'Seagrass area CSV not found: {area_csv_path!r}')

    df_area = pd.read_csv(area_csv_path)
    required_columns = {'eemarine_r566_id', 'area_ha'}
    missing_columns = required_columns.difference(df_area.columns)
    if missing_columns:
        raise ValueError(
            f'Seagrass area CSV missing columns: {sorted(missing_columns)}'
        )
    df_area = df_area[['eemarine_r566_id', 'area_ha']].copy()
    df_area = coastal_carbon_functions.normalize_integer_id_column(
        df_area, 'eemarine_r566_id'
    )
    df_area['area_ha'] = pd.to_numeric(df_area['area_ha'], errors='raise')
    if not np.isfinite(df_area['area_ha']).all() or (df_area['area_ha'] < 0).any():
        raise ValueError('Seagrass area must be finite and nonnegative')

    df_stock = df_area.groupby(
        'eemarine_r566_id', as_index=False
    )['area_ha'].sum()
    densities = coastal_carbon_functions.calculate_global_seagrass_pool_densities()
    for density_name, stock_name in {
        'agb_c_mg_per_ha': 'seagrass_agb_c_total_mg',
        'bgb_c_mg_per_ha': 'seagrass_bgb_c_total_mg',
        'soil_c_mg_per_ha': 'seagrass_soil_c_total_mg',
        'total_c_mg_per_ha': 'seagrass_total_c_stock_mg',
    }.items():
        df_stock[stock_name] = df_stock['area_ha'] * densities[density_name]
    df_stock = df_stock.drop(columns='area_ha')
    df_stock.to_csv(p.seagrass_carbon_stock_by_eez_path, index=False)
    hb.log(
        f"Seagrass stock: {df_area['area_ha'].sum():,.3f} ha, "
        f"{df_stock['seagrass_total_c_stock_mg'].sum():,.3f} Mg C."
    )
    coastal_carbon_functions.write_task_completion_marker(
        p.cur_dir,
        'task_calculate_seagrass_carbon_stock',
        {'stock_csv': p.seagrass_carbon_stock_by_eez_path},
        metadata=stock_metadata,
    )


def task_calculate_seagrass_storage_value(p):
    """
    Per-EEZ seagrass storage value (USD), per pool.

    Same physical/GEP split as mangrove and salt marsh. Output columns:
        seagrass_agb_storage_value, seagrass_bgb_storage_value,
        seagrass_soil_storage_value, seagrass_storage_value.
    """
    p.seagrass_storage_value_path = _calculate_storage_value(
        p,
        stock_csv_path=p.seagrass_carbon_stock_by_eez_path,
        pool_columns={
            'seagrass_agb_c_total_mg':   'seagrass_agb_storage_value',
            'seagrass_bgb_c_total_mg':   'seagrass_bgb_storage_value',
            'seagrass_soil_c_total_mg':  'seagrass_soil_storage_value',
            'seagrass_total_c_stock_mg': 'seagrass_storage_value',
        },
        ecosystem_label='Seagrass',
        output_csv_filename='seagrass_storage_value_by_eez2019.csv',
        expected_zone_ids=_eez_ids(p),
    )


def task_calculate_salt_marsh_carbon_stock(p):
    """
    Compute per-EEZ salt marsh carbon stocks via per-pixel density x ha_per_cell.

    Delegates to coastal_carbon_functions.compute_salt_marsh_carbon_stock_with_maxwell:
      - AGB: Chmura et al. 2003 median with tropical productivity boost
      - BGB: AGB x 2.5 (extensive root systems)
      - SOC: Maxwell et al. 2024 0--100 cm MarSOC raster at
             p.salt_marsh_soc_path, scaled to 30 cm (if set); fallback to
             latitude step function if path is missing or file absent.
    """
    p.salt_marsh_carbon_stock_by_eez_path = os.path.join(
        p.cur_dir, "salt_marsh_carbon_stock_by_eez2019.csv"
    )
    stock_marker_path = os.path.join(p.cur_dir, 'task_finished.json')
    maxwell_soc_path = getattr(p, 'salt_marsh_soc_path', None)
    salt_marsh_stock_metadata = {
        'algorithm_version': (
            coastal_carbon_functions.SALT_MARSH_CARBON_STOCK_ALGORITHM_VERSION
        ),
        'soil_depth_cm': coastal_carbon_functions.SALT_MARSH_SOIL_C_DEPTH_CM,
        'soil_source_depth_cm': (
            coastal_carbon_functions.SALT_MARSH_SOIL_SOURCE_DEPTH_CM
        ),
        'soil_source': _optional_file_fingerprint(maxwell_soc_path),
    }
    salt_marsh_area_csv_path = getattr(
        p, 'salt_marsh_area_by_eez_base_year_path', ''
    ).replace('.gpkg', '.csv')
    if (
        _outputs_are_current(
            [p.salt_marsh_carbon_stock_by_eez_path, stock_marker_path],
            [salt_marsh_area_csv_path],
        )
        and _eez_csv_is_current(p.salt_marsh_carbon_stock_by_eez_path, p)
        and _task_marker_is_current(
            stock_marker_path,
            'task_calculate_salt_marsh_carbon_stock',
            salt_marsh_stock_metadata,
        )
    ):
        hb.log("task_calculate_salt_marsh_carbon_stock: skipped (output exists)")
        return

    _build_eez_id_raster_if_needed(p)

    salt_marsh_coverage_path = os.path.join(
        p.cur_dir, 'salt_marsh_coverage_10sec.tif'
    )
    if not coastal_carbon_functions.is_exact_coverage_raster_complete(
            salt_marsh_coverage_path,
            p.salt_marsh_vector_path,
            p.ha_per_cell_10sec_path):
        coastal_carbon_functions.rasterize_exact_coverage_to_template(
            vector_path=p.salt_marsh_vector_path,
            template_raster_path=p.ha_per_cell_10sec_path,
            out_path=salt_marsh_coverage_path,
        )

    eez = _eez_zones(
        gpd.read_file(p.gdf_eez_vector_path)
    )
    eez_ids = eez['eemarine_r566_id'].dropna().astype(int).unique()

    df_stock = coastal_carbon_functions.compute_salt_marsh_carbon_stock_with_maxwell(
        project_dir=p.cur_dir,
        salt_marsh_mask_path=salt_marsh_coverage_path,
        eez_id_raster_path=p.eez_id_raster_path,
        ha_per_cell_path=p.ha_per_cell_10sec_path,
        eez_ids=eez_ids,
        maxwell_soc_path=maxwell_soc_path,
    )
    df_stock = df_stock.rename(columns={
        'agb_c_total_mg': 'salt_marsh_agb_c_total_mg',
        'bgb_c_total_mg': 'salt_marsh_bgb_c_total_mg',
        'soil_c_total_mg': 'salt_marsh_soil_c_total_mg',
        'total_c_total_mg': 'salt_marsh_total_c_stock_mg',
    })
    df_stock = coastal_carbon_functions.normalize_integer_id_column(
        df_stock, 'eemarine_r566_id'
    )
    df_stock.to_csv(p.salt_marsh_carbon_stock_by_eez_path, index=False)
    coastal_carbon_functions.write_task_completion_marker(
        p.cur_dir,
        'task_calculate_salt_marsh_carbon_stock',
        {'stock_csv': p.salt_marsh_carbon_stock_by_eez_path},
        metadata=salt_marsh_stock_metadata,
    )
    print(f"✅ Saved salt marsh carbon stock by EEZ: {p.salt_marsh_carbon_stock_by_eez_path}")


def task_combine_ecosystem_areas(p):
    """
    Combine mangrove, salt marsh, and seagrass areas plus carbon stocks into a
    single dataset. All three ecosystem branches are required upstream.
    """
    p.combined_eez_path = os.path.join(
        p.cur_dir, "combined_ecosystem_areas_by_eez2019.csv"
    )
    area_and_stock_inputs = [
        p.mangrove_area_by_eez_base_year_path.replace('.gpkg', '.csv'),
        p.salt_marsh_area_by_eez_base_year_path.replace('.gpkg', '.csv'),
        p.seagrass_area_by_eez_base_year_path.replace('.gpkg', '.csv'),
        p.mangrove_carbon_stock_by_eez_path,
        p.salt_marsh_carbon_stock_by_eez_path,
        p.seagrass_carbon_stock_by_eez_path,
    ]
    if (
        _outputs_are_current([p.combined_eez_path], area_and_stock_inputs)
        and _eez_csv_is_current(p.combined_eez_path, p)
    ):
        hb.log("task_combine_ecosystem_areas: skipped (output exists)")
        return

    # Use EEZ IDs as the base table. This prevents any non-EEZ row from
    # entering the combined output, even if an upstream file is malformed.
    eez = _eez_zones(gpd.read_file(p.gdf_eez_vector_path))
    df_combined = eez[['eemarine_r566_id']].copy()

    # Read mangrove areas
    df_mangrove = pd.read_csv(
        p.mangrove_area_by_eez_base_year_path.replace('.gpkg', '.csv')
    )
    df_mangrove = df_mangrove[['eemarine_r566_id', 'area_ha']].rename(
        columns={'area_ha': 'mangrove_area_ha'}
    )
    df_mangrove = coastal_carbon_functions.normalize_integer_id_column(
        df_mangrove, 'eemarine_r566_id'
    )

    # Read salt marsh areas
    df_salt_marsh = pd.read_csv(
        p.salt_marsh_area_by_eez_base_year_path.replace('.gpkg', '.csv')
    )
    df_salt_marsh = df_salt_marsh[['eemarine_r566_id', 'area_ha']].rename(
        columns={'area_ha': 'salt_marsh_area_ha'}
    )
    df_salt_marsh = coastal_carbon_functions.normalize_integer_id_column(
        df_salt_marsh, 'eemarine_r566_id'
    )

    # Read GlobalSeagrass raster-derived areas.
    seagrass_area_csv = p.seagrass_area_by_eez_base_year_path.replace('.gpkg', '.csv')
    df_seagrass = pd.read_csv(seagrass_area_csv)
    df_seagrass = df_seagrass[['eemarine_r566_id', 'area_ha']].rename(
        columns={'area_ha': 'seagrass_area_ha'}
    )
    df_seagrass = coastal_carbon_functions.normalize_integer_id_column(
        df_seagrass, 'eemarine_r566_id'
    )

    # Merge areas onto complete EEZ ID frame.
    df_combined = df_combined.merge(
        df_mangrove, on='eemarine_r566_id', how='left'
    )
    df_combined = df_combined.merge(
        df_salt_marsh, on='eemarine_r566_id', how='left'
    )
    df_combined = df_combined.merge(
        df_seagrass, on='eemarine_r566_id', how='left'
    )

    # Fill NaN with 0
    df_combined['mangrove_area_ha'] = df_combined['mangrove_area_ha'].fillna(0)
    df_combined['salt_marsh_area_ha'] = df_combined['salt_marsh_area_ha'].fillna(0)
    df_combined['seagrass_area_ha'] = df_combined['seagrass_area_ha'].fillna(0)

    # Calculate total coastal carbon area
    df_combined['total_coastal_carbon_area_ha'] = (
        df_combined['mangrove_area_ha']
        + df_combined['salt_marsh_area_ha']
        + df_combined['seagrass_area_ha']
    )

    # Read carbon stocks (per-pixel computed by stock tasks).
    # Each upstream stock task sets its own p.<ecosystem>_carbon_stock_by_eez_path
    # pointing into its own task folder.
    mangrove_stock_path = p.mangrove_carbon_stock_by_eez_path
    salt_marsh_stock_path = p.salt_marsh_carbon_stock_by_eez_path
    seagrass_stock_path = p.seagrass_carbon_stock_by_eez_path
    mangrove_stock_cols = [
        'eemarine_r566_id',
        'mangrove_agb_c_total_mg', 'mangrove_bgb_c_total_mg',
        'mangrove_soil_c_total_mg', 'mangrove_total_c_stock_mg',
    ]
    salt_marsh_stock_cols = [
        'eemarine_r566_id',
        'salt_marsh_agb_c_total_mg', 'salt_marsh_bgb_c_total_mg',
        'salt_marsh_soil_c_total_mg', 'salt_marsh_total_c_stock_mg',
    ]
    seagrass_stock_cols = [
        'eemarine_r566_id',
        'seagrass_agb_c_total_mg', 'seagrass_bgb_c_total_mg',
        'seagrass_soil_c_total_mg', 'seagrass_total_c_stock_mg',
    ]
    df_mangrove_stock = coastal_carbon_functions.normalize_integer_id_column(
        pd.read_csv(mangrove_stock_path)[mangrove_stock_cols],
        'eemarine_r566_id',
    )
    df_salt_marsh_stock = coastal_carbon_functions.normalize_integer_id_column(
        pd.read_csv(salt_marsh_stock_path)[salt_marsh_stock_cols],
        'eemarine_r566_id',
    )

    df_combined = df_combined.merge(df_mangrove_stock, on='eemarine_r566_id', how='left')
    df_combined = df_combined.merge(df_salt_marsh_stock, on='eemarine_r566_id', how='left')

    df_seagrass_stock = coastal_carbon_functions.normalize_integer_id_column(
        pd.read_csv(seagrass_stock_path)[seagrass_stock_cols],
        'eemarine_r566_id',
    )
    df_combined = df_combined.merge(
        df_seagrass_stock, on='eemarine_r566_id', how='left'
    )

    stock_cols_all = (
        mangrove_stock_cols[1:] + salt_marsh_stock_cols[1:] + seagrass_stock_cols[1:]
    )
    df_combined[stock_cols_all] = df_combined[stock_cols_all].fillna(0)

    df_combined['total_carbon_stock_mg'] = (
        df_combined['mangrove_total_c_stock_mg']
        + df_combined['salt_marsh_total_c_stock_mg']
        + df_combined['seagrass_total_c_stock_mg']
    )

    # Retain EEZ attributes only; no terrestrial or country-level join.
    df_eez = eez.drop(columns=['geometry'])

    # Merge with EEZ attributes
    df_combined = df_combined.merge(
        df_eez, on='eemarine_r566_id', how='left'
    )
    df_combined = coastal_carbon_functions.normalize_integer_id_column(
        df_combined, 'eemarine_r566_id'
    )

    df_combined.to_csv(p.combined_eez_path, index=False)
    print(f"✅ Saved combined ecosystem areas by EEZ: {p.combined_eez_path}")


def _load_iso250_universe(p):
    """Load one metadata row for every canonical ISO-250 country or territory."""
    correspondence_path = getattr(p, 'df_countries_csv_path', None)
    if not correspondence_path:
        correspondence_path = p.get_path(
            'cartographic', 'ee', 'ee_r264_correspondence.csv'
        )
    if not os.path.exists(correspondence_path):
        raise FileNotFoundError(
            f'ISO-250 correspondence not found: {correspondence_path}'
        )

    correspondence = pd.read_csv(correspondence_path)
    correspondence['iso_250_id'] = pd.to_numeric(
        correspondence['iso3_r250_id'], errors='raise'
    ).astype('int64')
    correspondence['iso_250'] = correspondence['iso3_r250_label'].astype(str)
    correspondence['iso_250_name'] = correspondence['iso3_r250_name'].astype(str)
    metadata_columns = [
        'iso_250_id', 'iso_250', 'iso_250_name', 'continent', 'region_un',
        'region_wb', 'adm0_a3', 'income_grp', 'name_long', 'type',
    ]
    iso250_universe = (
        correspondence[metadata_columns]
        .drop_duplicates('iso_250_id')
        .sort_values('iso_250_id')
        .reset_index(drop=True)
    )
    if len(iso250_universe) != 250:
        raise ValueError(
            'Expected 250 ISO-250 countries or territories, found '
            f'{len(iso250_universe)}'
        )
    return iso250_universe


def _add_coastal_carbon_gep_scenarios(p, dataframe):
    """Add annual GEP values for five Ramsey discount-rate scenarios."""
    dataframe = dataframe.copy()
    missing_columns = [
        column for column in COASTAL_CARBON_GEP_SCENARIO_COLUMNS
        if column not in dataframe.columns
    ]
    if not missing_columns:
        return dataframe
    if 'total_carbon_stock_mg' not in dataframe.columns:
        raise KeyError(
            'Cannot calculate coastal-carbon GEP scenarios without '
            'total_carbon_stock_mg'
        )
    if not getattr(p, 'carbon_prices_path', None):
        raise AttributeError(
            'p.carbon_prices_path is required for coastal-carbon GEP scenarios'
        )

    price_table = pd.read_excel(p.carbon_prices_path)
    price_table['year'] = pd.to_numeric(price_table['year'], errors='raise')
    base_year = int(pd.to_numeric(
        dataframe.get('year', pd.Series([2019])), errors='coerce'
    ).dropna().iloc[0])
    price_row = price_table.loc[price_table['year'] == base_year]
    if price_row.empty:
        raise ValueError(
            f'Carbon-price table has no row for coastal-carbon year {base_year}'
        )
    baseline_price = float(price_row[p.carbon_price].iloc[0])
    r3_price = float(price_row['rental scc r3%'].iloc[0])
    stock = pd.to_numeric(dataframe['total_carbon_stock_mg'], errors='raise')

    scenario_prices = {
        **COASTAL_CARBON_FIXED_RENTAL_SCC_2019,
        'coastal_carbon_gep_r2pct': baseline_price,
        'coastal_carbon_gep_r3pct': r3_price,
    }
    for column in missing_columns:
        if column == 'coastal_carbon_gep_r2pct' and 'value' in dataframe:
            dataframe[column] = pd.to_numeric(
                dataframe['value'], errors='raise'
            )
        else:
            dataframe[column] = stock * scenario_prices[column]
    return dataframe


def build_iso250_results_from_eez(p, df_eez):
    """Aggregate EEZ results to all canonical ISO-250 country rows."""
    df_eez = _add_coastal_carbon_gep_scenarios(p, df_eez)
    correspondence_path = os.path.join(
        p.base_data_dir,
        'coastal_carbon',
        'eemarine_r566_correspondence.csv',
    )
    if not os.path.exists(correspondence_path):
        correspondence_path = p.df_eez_csv_path
    correspondence = pd.read_csv(correspondence_path)
    correspondence = correspondence.loc[
        correspondence['eemarine_r566_label'].astype(str).str.endswith('_EEZ')
    ].copy()
    correspondence = coastal_carbon_functions.normalize_integer_id_column(
        correspondence, 'eemarine_r566_id'
    )
    correspondence['iso_250_id'] = pd.to_numeric(
        correspondence['iso3_r250_id'], errors='raise'
    ).astype('int64')
    correspondence['iso_250'] = correspondence['iso3_r250_label'].astype(str)
    correspondence = correspondence.drop_duplicates('eemarine_r566_id')

    metadata = correspondence[
        ['eemarine_r566_id', 'iso_250_id', 'iso_250', 'iso3_r250_name',
         'continent', 'region_un', 'region_wb', 'adm0_a3', 'income_grp',
         'name_long', 'type']
    ].copy()
    metadata = metadata.rename(columns={'iso3_r250_name': 'iso_250_name'})
    joined = df_eez.merge(
        metadata,
        on='eemarine_r566_id',
        how='left',
        validate='one_to_one',
    )
    if joined['iso_250_id'].isna().any():
        raise ValueError('ISO-250 correspondence missing for EEZ result row')

    metric_columns = [
        column for column in joined.columns
        if column.endswith(('_ha', '_mg', '_value'))
        or column in {
            'coastal_carbon_storage_value', 'value',
            *COASTAL_CARBON_GEP_SCENARIO_COLUMNS,
        }
    ]
    iso250_universe = _load_iso250_universe(p)
    metadata_columns = iso250_universe.columns.tolist()
    aggregated_metrics = joined.groupby(
        'iso_250_id', as_index=False, sort=True, dropna=False
    )[metric_columns].sum()
    aggregated = iso250_universe.merge(
        aggregated_metrics, on='iso_250_id', how='left', validate='one_to_one'
    )

    # Countries or territories without an EEZ result retain metadata and get
    # numeric zero for every coastal-carbon metric.
    for column in metric_columns:
        aggregated[column] = pd.to_numeric(
            aggregated[column], errors='raise'
        ).fillna(0.0)

    for column in ['year', p.carbon_price]:
        if column in joined.columns:
            values = joined.groupby(
                'iso_250_id', as_index=False, sort=True, dropna=False
            )[column].first()
            aggregated = aggregated.merge(
                values, on='iso_250_id', how='left', validate='one_to_one'
            )
            aggregated[column] = pd.to_numeric(
                aggregated[column], errors='coerce'
            )
            observed_values = aggregated[column].dropna()
            default_value = (
                observed_values.iloc[0]
                if not observed_values.empty
                else (2019 if column == 'year' else np.nan)
            )
            aggregated[column] = aggregated[column].fillna(default_value)
            if column == 'year':
                aggregated[column] = aggregated[column].astype('int64')
    ordered_columns = (
        metadata_columns
        + [column for column in aggregated.columns if column not in metadata_columns]
    )
    aggregated = aggregated[ordered_columns]
    aggregated = coastal_carbon_functions.normalize_integer_id_column(
        aggregated, 'iso_250_id'
    )

    iso_csv = os.path.join(p.cur_dir, 'gep_by_iso2502019.csv')
    iso_gpkg = iso_csv.replace('.csv', '.gpkg')
    aggregated.to_csv(iso_csv, index=False)

    eez_geometry = _eez_zones(gpd.read_file(p.gdf_eez_vector_path))
    eez_geometry = eez_geometry.merge(
        metadata,
        on='eemarine_r566_id',
        how='left',
        validate='one_to_one',
    )
    # Dissolve EEZ geometries into one geometry per ISO-250 ID.
    iso_geometry = eez_geometry.dissolve(
        by='iso_250_id', as_index=False, aggfunc='first'
    )[['iso_250_id', 'geometry']].merge(
        aggregated,
        on='iso_250_id',
        how='left',
        validate='one_to_one',
    )
    iso_geometry.to_file(iso_gpkg, driver='GPKG')
    return iso_csv, iso_gpkg


def gep_calculation(p):
    """
    Calculate coastal-carbon GEP for EEZs only.

    Storage-only value equals each ecosystem's carbon stock (Mg C) multiplied
    by rental SCC ($/Mg C). EEZ outputs remain canonical; an ISO-250 aggregate
    is also written using the coastal-carbon correspondence table.
    """
    service_results = {}
    p.results['coastal_carbon'] = service_results

    eez_csv = os.path.join(p.cur_dir, 'gep_by_eez2019.csv')
    eez_gpkg = eez_csv.replace('.csv', '.gpkg')
    iso_csv = os.path.join(p.cur_dir, 'gep_by_iso2502019.csv')
    iso_gpkg = iso_csv.replace('.csv', '.gpkg')
    p.results['coastal_carbon']['gep_by_eez_base_year'] = eez_csv
    p.results['coastal_carbon']['gep_by_country_base_year'] = iso_csv

    if (
        _outputs_are_current(
            [eez_csv, eez_gpkg, iso_csv, iso_gpkg],
            [p.combined_eez_path],
        )
        and _eez_csv_is_current(eez_csv, p)
    ):
        hb.log('gep_calculation: skipped (EEZ outputs exist)')
        return pd.read_csv(eez_csv)['value'].sum()

    if (
        _outputs_are_current([eez_csv, eez_gpkg], [p.combined_eez_path])
        and _eez_csv_is_current(eez_csv, p)
    ):
        hb.log('gep_calculation: EEZ outputs current; building ISO-250 version')
        df_existing = pd.read_csv(eez_csv)
        build_iso250_results_from_eez(p, df_existing)
        return df_existing['value'].sum()

    hb.log(
        'gep_calculation: computing EEZ storage values '
        '(mangrove + salt marsh + seagrass)...'
    )
    df_areas = coastal_carbon_functions.normalize_integer_id_column(
        pd.read_csv(p.combined_eez_path), 'eemarine_r566_id'
    )
    df_areas['year'] = 2019

    df_carbon_p = pd.read_excel(p.carbon_prices_path)
    df_carbon_p = df_carbon_p[['year', p.carbon_price]]
    df_gep = df_areas.merge(df_carbon_p, on='year', how='left')

    df_gep['mangrove_storage_value'] = (
        df_gep['mangrove_total_c_stock_mg'] * df_gep[p.carbon_price]
    )
    df_gep['salt_marsh_storage_value'] = (
        df_gep['salt_marsh_total_c_stock_mg'] * df_gep[p.carbon_price]
    )
    df_gep['seagrass_storage_value'] = (
        df_gep['seagrass_total_c_stock_mg'] * df_gep[p.carbon_price]
    )
    df_gep['coastal_carbon_storage_value'] = (
        df_gep['mangrove_storage_value']
        + df_gep['salt_marsh_storage_value']
        + df_gep['seagrass_storage_value']
    )
    df_gep['value'] = df_gep['coastal_carbon_storage_value']
    df_gep = _add_coastal_carbon_gep_scenarios(p, df_gep)

    eez = _eez_zones(gpd.read_file(p.gdf_eez_vector_path))
    eez_ids = set(eez['eemarine_r566_id'])
    df_gep = df_gep.loc[
        df_gep['eemarine_r566_id'].isin(eez_ids)
    ].copy()
    df_gep = eez.drop(columns=['geometry']).merge(
        df_gep.drop(
            columns=['eemarine_r566_label', 'eemarine_r566_name'],
            errors='ignore',
        ),
        on='eemarine_r566_id',
        how='left',
        validate='one_to_one',
    )
    numeric_columns = [
        'mangrove_area_ha', 'salt_marsh_area_ha', 'seagrass_area_ha',
        'total_coastal_carbon_area_ha',
        'mangrove_agb_c_total_mg', 'mangrove_bgb_c_total_mg',
        'mangrove_soil_c_total_mg', 'mangrove_total_c_stock_mg',
        'salt_marsh_agb_c_total_mg', 'salt_marsh_bgb_c_total_mg',
        'salt_marsh_soil_c_total_mg', 'salt_marsh_total_c_stock_mg',
        'seagrass_agb_c_total_mg', 'seagrass_bgb_c_total_mg',
        'seagrass_soil_c_total_mg', 'seagrass_total_c_stock_mg',
        'mangrove_storage_value', 'salt_marsh_storage_value',
        'seagrass_storage_value', 'coastal_carbon_storage_value', 'value',
        *COASTAL_CARBON_GEP_SCENARIO_COLUMNS,
    ]
    for column in numeric_columns:
        if column in df_gep.columns:
            df_gep[column] = df_gep[column].fillna(0)
    df_gep = coastal_carbon_functions.normalize_integer_id_column(
        df_gep, 'eemarine_r566_id'
    )

    hb.df_write(df_gep, eez_csv)
    gdf_gep = eez.merge(
        df_gep.drop(
            columns=[
                'geometry', 'eemarine_r566_label', 'eemarine_r566_name'
            ],
            errors='ignore',
        ),
        on='eemarine_r566_id',
        how='left',
        validate='one_to_one',
    )
    gdf_gep.to_file(eez_gpkg, driver='GPKG')
    build_iso250_results_from_eez(p, df_gep)

    hb.log(
        f"EEZ total coastal-carbon storage value (2019): "
        f"${df_gep['value'].sum():,.2f}"
    )
    return df_gep['value'].sum()


def gep_result(p):
    """Display the results of the GEP calculation."""

    # Set the quarto path
    os.environ['QUARTO_PYTHON'] = sys.executable

    # Get the list of current services run
    services_run = list(p.results.keys())

    # Imply from the service name the file_path for the results_qmd
    module_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for service_label in services_run:
        print(service_label)
        results_qmd_path = os.path.join(
            module_root, service_label, f'{service_label}_results.qmd'
        )
        results_qmd_project_path = os.path.join(p.cur_dir, f'{service_label}_results.qmd')
        if not os.path.exists(results_qmd_path):
            raise FileNotFoundError(f"Results QMD template not found: {results_qmd_path}")
        hb.create_directories(results_qmd_project_path)
        hb.path_copy(results_qmd_path, results_qmd_project_path)

        # Also copy any bibliography/CSL/template assets sitting next to the QMD
        # source so Quarto's citeproc filter can resolve them in cur_dir.
        qmd_src_dir = os.path.dirname(results_qmd_path)
        copied_sidecar_paths = []
        for sidecar_name in os.listdir(qmd_src_dir):
            if sidecar_name.endswith(('.bib', '.csl', '.bst', '.yml', '.yaml')):
                if sidecar_name in ('_quarto.yml', '_quarto.yaml'):
                    continue
                src = os.path.join(qmd_src_dir, sidecar_name)
                dst = os.path.join(p.cur_dir, sidecar_name)
                if os.path.isfile(src):
                    hb.path_copy(src, dst)
                    copied_sidecar_paths.append(dst)

        quarto_command = f"quarto render {results_qmd_project_path}"
        hb.log(f"Running quarto command: {quarto_command}")

        # Set environment for more verbose output
        env = os.environ.copy()
        env['QUARTO_LOG_LEVEL'] = 'DEBUG'
        repo_root = os.path.dirname(module_root)
        env['GLOBAL_INVEST_REPO_ROOT'] = repo_root
        env['COASTAL_CARBON_PROJECT_DIR'] = p.project_dir
        existing_pythonpath = env.get('PYTHONPATH')
        env['PYTHONPATH'] = (
            repo_root if not existing_pythonpath
            else repo_root + os.pathsep + existing_pythonpath
        )

        cmd = ['quarto', 'render', results_qmd_project_path, '--verbose']

        print(f"Working directory: {os.getcwd()}")
        print(f"File exists: {os.path.exists(results_qmd_project_path)}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Read line by line as they come
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.rstrip())
                sys.stdout.flush()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

        # Remove temporary files
        hb.path_remove(results_qmd_project_path)
        for sidecar in copied_sidecar_paths:
            hb.path_remove(sidecar)

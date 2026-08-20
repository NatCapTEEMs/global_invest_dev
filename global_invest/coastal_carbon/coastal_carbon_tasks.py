import os
import subprocess
import sys

import geopandas as gpd
import hazelbean as hb
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats
from tqdm import tqdm

from global_invest import utilities
from global_invest.coastal_carbon import coastal_carbon_functions



def publish_inputs(p):
    """Every config-consuming task's first line. TWO SURFACES BY DESIGN: the marine r566
    correspondence (the gep_regions_input_path cell) is the per-EEZ aggregation surface with its
    gep_regions_id_col; the r264 correspondence (initialize_country_paths) is the iso3_r250-
    collapse crosswalk gep_calculation reads. Science inputs (habitat extents, SOC rasters,
    precipitation) hydrate from es_parameters -- the optional ones resolve permissively and the
    stock tasks keep their documented fallbacks. ha_per_cell is pyramid infrastructure, in code."""
    utilities.hydrate_es_config(p, 'coastal_carbon', log=hb.log)
    utilities.hydrate_es_parameters(p, 'coastal_carbon', log=hb.log)
    utilities.initialize_country_paths(p, simplified='30sec')
    p.ha_per_cell_10sec_path = p.get_path('pyramids', 'ha_per_cell_10sec.tif')
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def _task_outputs_exist(*paths):
    """Return True if every given path is non-empty and exists on disk.

    INTERNAL sub-step cache only (heavy intermediates inside a task, the r566 stage in
    gep_calculation). Task-level skipping is ProjectFlow-native: tasks are registered with
    skip_existing=1, so a task whose directory exists gets p.run_this=0, publishes its output
    paths and returns -- delete the task's directory to force a rerun.
    """
    return all(bool(p) and os.path.exists(p) for p in paths)


def mangrove_area_within_countries(p):
    """
    Calculate mangrove area within each country's marine EEZ.
    Uses Global Mangrove Watch (GMW) vector data.
    """
    publish_inputs(p)
    p.mangrove_area_by_countries_base_year_path = os.path.join(
        p.cur_dir, "mangrove_area_by_countries2019.gpkg"
    )
    csv_path = p.mangrove_area_by_countries_base_year_path.replace('.gpkg', '.csv')
    if not p.run_this:
        return

    gdf_countries_marine_vector = gpd.read_file(p.gep_regions_input_path)
    gdf_mangroves = gpd.read_file(p.mangrove_vector_path)

    print(f"Loaded {len(gdf_mangroves)} mangrove polygons")
    print(f"Loaded {len(gdf_countries_marine_vector)} country marine zones")

    # Ensure same CRS
    gdf_mangroves = gdf_mangroves.to_crs(gdf_countries_marine_vector.crs)

    # Build spatial index once so each country only checks nearby mangroves
    mangrove_sindex = gdf_mangroves.sindex

    # Intersect per country (progress bar visible vs single bulk gpd.overlay)
    intersect_list = []
    for idx, country in tqdm(
        gdf_countries_marine_vector.iterrows(),
        total=len(gdf_countries_marine_vector),
        desc="Intersecting mangroves with countries",
    ):
        candidate_idx = list(mangrove_sindex.intersection(country.geometry.bounds))
        if not candidate_idx:
            continue
        candidates = gdf_mangroves.iloc[candidate_idx]
        country_gdf = gpd.GeoDataFrame([country], crs=gdf_countries_marine_vector.crs)
        clipped = gpd.overlay(candidates, country_gdf, how="intersection")
        if clipped.empty:
            continue
        intersect_list.append(clipped)

    if not intersect_list:
        raise ValueError("No intersections found between mangroves and countries")

    gdf_mangroves_within_countries = gpd.GeoDataFrame(
        pd.concat(intersect_list, ignore_index=True),
        crs=gdf_countries_marine_vector.crs,
    )

    # Reproject to equal-area projection for area calculation
    gdf_mangroves_within_countries = gdf_mangroves_within_countries.to_crs(epsg=6933)

    # Calculate area
    gdf_mangroves_within_countries["area_m2"] = gdf_mangroves_within_countries.geometry.area
    gdf_mangroves_within_countries["area_ha"] = gdf_mangroves_within_countries["area_m2"] / 10000

    # Save detailed intersection
    gdf_mangroves_within_countries_base_year_path = os.path.join(
        p.cur_dir, "mangroves_within_countries2019.gpkg"
    )
    gdf_mangroves_within_countries.to_file(
        gdf_mangroves_within_countries_base_year_path, driver="GPKG"
    )

    # Aggregate by country
    mangrove_area_by_countries_base_year = (
        gdf_mangroves_within_countries.groupby(p.gep_regions_id_col)["area_ha"]
        .sum()
        .reset_index()
    )

    # Merge with country info
    mangrove_area_by_countries_base_year = mangrove_area_by_countries_base_year.merge(
        gdf_countries_marine_vector, how="left", on=p.gep_regions_id_col
    )

    # Create GeoDataFrame with country geometries
    mangrove_area_by_countries_base_year = gpd.GeoDataFrame(
        mangrove_area_by_countries_base_year,
        geometry="geometry",   # the merge on eemarine_r566_id already brought each country's geometry; never attach by row position
        crs=gdf_countries_marine_vector.crs
    )

    mangrove_area_by_countries_base_year.to_file(
        p.mangrove_area_by_countries_base_year_path, driver="GPKG"
    )
    mangrove_area_by_countries_base_year.drop(columns=['geometry']).to_csv(csv_path, index=False)
    print(f"✅ Saved mangrove area by countries: {p.mangrove_area_by_countries_base_year_path}")


def salt_marsh_area_within_countries(p):
    """
    Calculate salt marsh area within each country's marine EEZ.
    Uses zonal statistics on salt marsh raster data.
    """
    publish_inputs(p)
    p.salt_marsh_area_by_countries_base_year_path = os.path.join(
        p.cur_dir, "salt_marsh_area_by_countries2019.gpkg"
    )
    csv_path = p.salt_marsh_area_by_countries_base_year_path.replace('.gpkg', '.csv')
    if not p.run_this:
        return

    # Read salt marsh vector data
    gdf_salt_marsh = gpd.read_file(p.salt_marsh_vector_path)
    gdf_countries_marine_vector = gpd.read_file(p.gep_regions_input_path)

    print(f"Loaded {len(gdf_salt_marsh)} salt marsh polygons")
    print(f"Loaded {len(gdf_countries_marine_vector)} country marine zones")

    # Reproject salt marsh to match ha_per_cell CRS
    with rasterio.open(p.ha_per_cell_10sec_path) as src:
        ha_per_cell_crs = src.crs
        gdf_salt_marsh = gdf_salt_marsh.to_crs(ha_per_cell_crs)

    # Calculate zonal stats for each salt marsh polygon
    stats_list = []
    for i, geom in enumerate(tqdm(gdf_salt_marsh.geometry, desc="Computing zonal stats for salt marsh")):
        stats = zonal_stats(
            vectors=[geom],
            raster=p.ha_per_cell_10sec_path,
            stats=['sum'],
            geojson_out=True,
            nodata=-9999.0,
            all_touched=True
        )
        stats_list.extend(stats)

    # Convert to GeoDataFrame
    gdf_salt_marsh_zonal = gpd.GeoDataFrame.from_features(stats_list)
    gdf_salt_marsh_zonal.set_crs(ha_per_cell_crs, inplace=True)

    # Rename sum column to area_ha
    gdf_salt_marsh_zonal.rename(columns={'sum': 'area_ha'}, inplace=True)

    # Reproject to match countries vector CRS
    gdf_salt_marsh_zonal = gdf_salt_marsh_zonal.to_crs(gdf_countries_marine_vector.crs)

    # Save intermediate result
    salt_marsh_zonal_path = os.path.join(p.cur_dir, "salt_marsh_with_area.gpkg")
    gdf_salt_marsh_zonal.to_file(salt_marsh_zonal_path, driver="GPKG")

    # Intersect with country marine boundaries
    intersect_list = []
    for idx, country in tqdm(
        gdf_countries_marine_vector.iterrows(),
        total=len(gdf_countries_marine_vector),
        desc="Intersecting salt marsh with countries"
    ):
        # Create single-row GeoDataFrame for this country
        country_gdf = gpd.GeoDataFrame([country], crs=gdf_countries_marine_vector.crs)

        # Perform intersection
        clipped = gpd.overlay(gdf_salt_marsh_zonal, country_gdf, how="intersection")

        if clipped.empty:
            continue

        # Add country ID
        if p.gep_regions_id_col in gdf_countries_marine_vector.columns:
            clipped[p.gep_regions_id_col] = country[p.gep_regions_id_col]

        intersect_list.append(clipped)

    # Combine all intersections
    if len(intersect_list) > 0:
        intersected = pd.concat(intersect_list, ignore_index=True)
        intersected = gpd.GeoDataFrame(intersected, crs=gdf_countries_marine_vector.crs)
    else:
        raise ValueError("No intersections found between salt marsh and countries")

    # Calculate area for intersected polygons
    intersected = intersected.to_crs(epsg=6933)  # Equal area projection
    intersected["area_m2"] = intersected.geometry.area
    intersected["area_ha"] = intersected["area_m2"] / 10000

    # Aggregate by country
    salt_marsh_area_by_countries = (
        intersected.groupby(p.gep_regions_id_col)["area_ha"]
        .sum()
        .reset_index()
    )

    # Merge with country info
    salt_marsh_area_by_countries = salt_marsh_area_by_countries.merge(
        gdf_countries_marine_vector, how="right", on=p.gep_regions_id_col
    )

    # Fill NaN areas with 0 (countries with no salt marsh)
    salt_marsh_area_by_countries['area_ha'] = salt_marsh_area_by_countries['area_ha'].fillna(0)

    # Create GeoDataFrame
    salt_marsh_area_by_countries = gpd.GeoDataFrame(
        salt_marsh_area_by_countries,
        geometry="geometry",   # the merge on eemarine_r566_id already brought each country's geometry; never attach by row position
        crs=gdf_countries_marine_vector.crs
    )

    salt_marsh_area_by_countries.to_file(
        p.salt_marsh_area_by_countries_base_year_path, driver="GPKG"
    )
    salt_marsh_area_by_countries.drop(columns=['geometry']).to_csv(csv_path, index=False)
    print(f"✅ Saved salt marsh area by countries: {p.salt_marsh_area_by_countries_base_year_path}")


def _build_country_id_raster_if_needed(p):
    """Rasterize the country EEZ vector to a uint16 id grid aligned with ha_per_cell.

    Cross-task reuse: if a previous task already set p.country_id_raster_path
    and the file exists, just reuse it (no rebuild and no path overwrite).
    Otherwise the raster is built into the calling task's p.cur_dir, so the
    first stock task to run becomes the owner of the file and downstream
    callers find it via the shared p.country_id_raster_path attribute.
    """
    existing = getattr(p, 'country_id_raster_path', None)
    if existing and os.path.exists(existing):
        return
    p.country_id_raster_path = os.path.join(p.cur_dir, "country_id_raster_10sec.tif")
    if not os.path.exists(p.country_id_raster_path):
        coastal_carbon_functions.rasterize_polygons_to_template(
            vector_path=p.gep_regions_input_path,
            template_raster_path=p.ha_per_cell_10sec_path,
            out_path=p.country_id_raster_path,
            field=p.gep_regions_id_col,
            dtype='uint16',
            nodata=0,
            all_touched=False,
        )


def mangrove_carbon_stock(p):
    """
    Compute per-country mangrove carbon stocks via per-pixel density x ha_per_cell.

    Delegates the heavy lifting to
    coastal_carbon_functions.compute_mangrove_carbon_stock_with_sanderman:
      - AGB: Hamilton & Friess (2018) latitude regression
      - BGB: AGB x IPCC 2014 zone-specific ratio (uses p.precipitation_path if set)
      - SOC: Sanderman 2018 raster at p.mangrove_soc_path (fallback otherwise)
    """
    publish_inputs(p)
    p.mangrove_carbon_stock_path = os.path.join(
        p.cur_dir, "mangrove_carbon_stock_by_countries2019.csv"
    )
    if not p.run_this:
        return

    _build_country_id_raster_if_needed(p)

    mangrove_mask_path = os.path.join(p.cur_dir, "mangrove_mask_10sec.tif")
    if not os.path.exists(mangrove_mask_path):
        coastal_carbon_functions.rasterize_polygons_to_template(
            vector_path=p.mangrove_vector_path,
            template_raster_path=p.ha_per_cell_10sec_path,
            out_path=mangrove_mask_path,
            default_value=1,
            dtype='uint8',
            nodata=0,
            all_touched=True,
        )

    gdf_countries = gpd.read_file(p.gep_regions_input_path)
    country_ids = gdf_countries[p.gep_regions_id_col].dropna().astype(int).unique()

    df_stock = coastal_carbon_functions.compute_mangrove_carbon_stock_with_sanderman(
        project_dir=p.cur_dir,
        mangrove_mask_path=mangrove_mask_path,
        country_id_raster_path=p.country_id_raster_path,
        ha_per_cell_path=p.ha_per_cell_10sec_path,
        country_ids=country_ids,
        sanderman_soc_path=getattr(p, 'mangrove_soc_path', None),
        precipitation_path=getattr(p, 'precipitation_path', None),
    )
    df_stock = df_stock.rename(columns={
        'agb_c_total_mg': 'mangrove_agb_c_total_mg',
        'bgb_c_total_mg': 'mangrove_bgb_c_total_mg',
        'soil_c_total_mg': 'mangrove_soil_c_total_mg',
        'total_c_total_mg': 'mangrove_total_c_stock_mg',
    })
    df_stock.to_csv(p.mangrove_carbon_stock_path, index=False)
    print(f"✅ Saved mangrove carbon stock by countries: {p.mangrove_carbon_stock_path}")


def _calculate_storage_value(p, stock_csv_path, pool_columns,
                              ecosystem_label, output_csv_filename):
    """
    Per-country storage value CSV from per-pool physical stocks.

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
        Path to the per-country stock CSV (output of the stock task).
    pool_columns : dict[str, str]
        Mapping {stock_column_name: output_value_column_name} in the order
        you want them written. The LAST entry is treated as the 'total' for
        logging purposes.
    ecosystem_label : str
    output_csv_filename : str
        Written under the calling task's p.cur_dir.
    """
    out_path = os.path.join(p.cur_dir, output_csv_filename)
    if _task_outputs_exist(out_path):
        hb.log(f"{ecosystem_label} storage value: skipped (output exists at {out_path})")
        return out_path

    if not stock_csv_path or not os.path.exists(stock_csv_path):
        raise FileNotFoundError(
            f"{ecosystem_label} storage value: stock CSV missing ({stock_csv_path!r}) -- the stock "
            f"task upstream in the same tree must produce it; a silent skip here would drop the "
            f"ecosystem's per-pool value output without a trace.")

    df_stock = pd.read_csv(stock_csv_path)

    df_carbon_p = pd.read_excel(p.gep_price_input_path)
    df_carbon_p = df_carbon_p[['year', p.gep_price_convention]]
    gep_base_year = p.gep_base_year
    rental_scc = float(
        df_carbon_p.loc[df_carbon_p['year'] == gep_base_year, p.gep_price_convention].iloc[0]
    )

    df_stock['year'] = gep_base_year
    df_stock[p.gep_price_convention] = rental_scc

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


def mangrove_storage_value(p):
    """
    Per-country mangrove storage value (USD), per pool.

    Stage split:
      - Physical part (upstream mangrove_carbon_stock):
            mangrove_agb_c_total_mg, mangrove_bgb_c_total_mg,
            mangrove_soil_c_total_mg, mangrove_total_c_stock_mg.
      - GEP part (this task): each pool stock x rental SCC ->
            mangrove_agb_storage_value, mangrove_bgb_storage_value,
            mangrove_soil_storage_value, mangrove_storage_value.
    """
    publish_inputs(p)
    p.mangrove_storage_value_path = os.path.join(p.cur_dir, "mangrove_storage_value_by_countries2019.csv")
    if not p.run_this:
        return
    p.mangrove_storage_value_path = _calculate_storage_value(
        p,
        stock_csv_path=p.mangrove_carbon_stock_path,
        pool_columns={
            'mangrove_agb_c_total_mg':   'mangrove_agb_storage_value',
            'mangrove_bgb_c_total_mg':   'mangrove_bgb_storage_value',
            'mangrove_soil_c_total_mg':  'mangrove_soil_storage_value',
            'mangrove_total_c_stock_mg': 'mangrove_storage_value',
        },
        ecosystem_label='Mangrove',
        output_csv_filename='mangrove_storage_value_by_countries2019.csv',
    )


def salt_marsh_storage_value(p):
    """
    Per-country salt marsh storage value (USD), per pool.

    Same physical/GEP split as mangrove. Output columns:
        salt_marsh_agb_storage_value, salt_marsh_bgb_storage_value,
        salt_marsh_soil_storage_value, salt_marsh_storage_value.
    """
    publish_inputs(p)
    p.salt_marsh_storage_value_path = os.path.join(p.cur_dir, "salt_marsh_storage_value_by_countries2019.csv")
    if not p.run_this:
        return
    p.salt_marsh_storage_value_path = _calculate_storage_value(
        p,
        stock_csv_path=p.salt_marsh_carbon_stock_path,
        pool_columns={
            'salt_marsh_agb_c_total_mg':   'salt_marsh_agb_storage_value',
            'salt_marsh_bgb_c_total_mg':   'salt_marsh_bgb_storage_value',
            'salt_marsh_soil_c_total_mg':  'salt_marsh_soil_storage_value',
            'salt_marsh_total_c_stock_mg': 'salt_marsh_storage_value',
        },
        ecosystem_label='Salt marsh',
        output_csv_filename='salt_marsh_storage_value_by_countries2019.csv',
    )


# ----------------------------------------------------------------------------
# Seagrass tasks: implemented (area -> genus-aware stock -> storage value).
#
# Extent = WCMC013-014 SeagrassPtPy v7.1 (p.seagrass_vector_path, set in
# initialize_paths); the per-country intersection keeps the GENUS attribute so
# the stock task can apply Gomis 2025 per-genus pool densities. A missing
# extent RAISES: excluding seagrass is done by building the tree with
# include_seagrass=False, never by letting a data gap silently understate the
# coastal GEP total.
# ----------------------------------------------------------------------------

def seagrass_area_within_countries(p):
    """
    Calculate seagrass area within each country's marine EEZ.

    Source: WCMC013-014 SeagrassPtPy v7.1 polygon dataset (UNEP-WCMC).
    Per-country intersection runs polygon-by-polygon with a spatial index
    pre-filter (matching the mangrove area task pattern). The intermediate
    polygon-level GPKG retains the GENUS attribute so the downstream
    genus-aware stock task can apply Gomis 2025 per-genus densities.

    Outputs
    -------
    seagrass_within_countries2019.gpkg
        Per-polygon intersection with GENUS, eemarine_r566_id, area_ha.
    seagrass_area_by_countries2019.{gpkg,csv}
        Country-level area total.
    """
    publish_inputs(p)
    p.seagrass_within_countries_path = os.path.join(
        p.cur_dir, "seagrass_within_countries2019.gpkg"
    )
    p.seagrass_area_by_countries_base_year_path = os.path.join(
        p.cur_dir, "seagrass_area_by_countries2019.gpkg"
    )
    csv_path = p.seagrass_area_by_countries_base_year_path.replace('.gpkg', '.csv')
    if not p.run_this:
        return

    if not getattr(p, 'seagrass_vector_path', None) or not os.path.exists(p.seagrass_vector_path):
        raise FileNotFoundError(
            "seagrass extent not found (p.seagrass_vector_path=%r). A built seagrass tree with no "
            "data would silently understate the coastal GEP total; to exclude seagrass, build the "
            "tree with include_seagrass=False instead." % (getattr(p, 'seagrass_vector_path', None),))

    gdf_countries_marine_vector = gpd.read_file(p.gep_regions_input_path)
    gdf_seagrass = gpd.read_file(p.seagrass_vector_path, columns=['GENUS', 'FAMILY'])

    print(f"Loaded {len(gdf_seagrass)} seagrass polygons")
    print(f"Loaded {len(gdf_countries_marine_vector)} country marine zones")

    # Reproject seagrass to country CRS
    gdf_seagrass = gdf_seagrass.to_crs(gdf_countries_marine_vector.crs)
    seagrass_sindex = gdf_seagrass.sindex

    # Per-country intersect with progress bar
    intersect_list = []
    for idx, country in tqdm(
        gdf_countries_marine_vector.iterrows(),
        total=len(gdf_countries_marine_vector),
        desc="Intersecting seagrass with countries",
    ):
        candidate_idx = list(seagrass_sindex.intersection(country.geometry.bounds))
        if not candidate_idx:
            continue
        candidates = gdf_seagrass.iloc[candidate_idx]
        country_gdf = gpd.GeoDataFrame([country], crs=gdf_countries_marine_vector.crs)
        clipped = gpd.overlay(candidates, country_gdf, how="intersection")
        if clipped.empty:
            continue
        intersect_list.append(clipped)

    if not intersect_list:
        raise ValueError("No intersections found between seagrass and countries")

    gdf_seagrass_within_countries = gpd.GeoDataFrame(
        pd.concat(intersect_list, ignore_index=True),
        crs=gdf_countries_marine_vector.crs,
    )

    # Equal-area projection for area calculation
    gdf_seagrass_within_countries = gdf_seagrass_within_countries.to_crs(epsg=6933)
    gdf_seagrass_within_countries['area_m2'] = gdf_seagrass_within_countries.geometry.area
    gdf_seagrass_within_countries['area_ha'] = gdf_seagrass_within_countries['area_m2'] / 10000

    # Save polygon-level intersection (carries GENUS for downstream stock task)
    gdf_seagrass_within_countries.to_file(p.seagrass_within_countries_path, driver="GPKG")

    # Aggregate area by country
    area_by_country = (
        gdf_seagrass_within_countries.groupby(p.gep_regions_id_col)["area_ha"]
        .sum()
        .reset_index()
    )
    area_by_country = area_by_country.merge(
        gdf_countries_marine_vector, how="right", on=p.gep_regions_id_col
    )
    area_by_country['area_ha'] = area_by_country['area_ha'].fillna(0)
    area_by_country = gpd.GeoDataFrame(
        area_by_country,
        geometry="geometry",   # the merge on eemarine_r566_id already brought each country's geometry; never attach by row position
        crs=gdf_countries_marine_vector.crs,
    )

    area_by_country.to_file(p.seagrass_area_by_countries_base_year_path, driver="GPKG")
    area_by_country.drop(columns=['geometry']).to_csv(csv_path, index=False)
    print(f"✅ Saved seagrass area by countries: {p.seagrass_area_by_countries_base_year_path}")


def seagrass_carbon_stock(p):
    """
    Per-country seagrass carbon stocks (Mg C) using Gomis et al. 2025 genus-specific
    biomass densities and a Fourqurean et al. 2012 soil constant scaled to 1 m depth.

    Reads the polygon-level GPKG produced by seagrass_area_within_countries
    (carries GENUS + eemarine_r566_id + area_ha), looks up per-genus carbon density via
    seagrass_carbon.calculate_pool_densities_array, multiplies density x area per
    polygon, then aggregates to country level. Non-marine genera (Trapa, Myriophyllum,
    Valisneria, Najas, etc.) are zeroed out before aggregation.
    """
    publish_inputs(p)
    p.seagrass_carbon_stock_path = os.path.join(
        p.cur_dir, "seagrass_carbon_stock_by_countries2019.csv"
    )
    if not p.run_this:
        return

    if not getattr(p, 'seagrass_within_countries_path', None) or \
            not os.path.exists(p.seagrass_within_countries_path):
        raise FileNotFoundError(
            "seagrass polygon-level GPKG missing -- seagrass_area_within_countries "
            "must run first (same tree). Skipping here would silently zero seagrass in the total.")

    gdf = gpd.read_file(p.seagrass_within_countries_path)
    print(f"Read {len(gdf)} seagrass-within-country polygons")

    # Per-polygon densities (Mg C/ha) by genus (Gomis et al. 2025 lookup
    # lives in coastal_carbon_functions.calculate_seagrass_pool_densities_array)
    densities = coastal_carbon_functions.calculate_seagrass_pool_densities_array(
        gdf['GENUS'].values
    )

    # Per-polygon carbon stocks (Mg C) = density x area
    area_ha = gdf['area_ha'].values.astype(np.float64)
    gdf['seagrass_agb_c_mg']   = densities['agb_c_mg_per_ha']  * area_ha
    gdf['seagrass_bgb_c_mg']   = densities['bgb_c_mg_per_ha']  * area_ha
    gdf['seagrass_soil_c_mg']  = densities['soil_c_mg_per_ha'] * area_ha
    gdf['seagrass_total_c_mg'] = densities['total_c_mg_per_ha'] * area_ha

    # Aggregate to country
    df_stock = (
        gdf.groupby(p.gep_regions_id_col)[
            ['seagrass_agb_c_mg', 'seagrass_bgb_c_mg',
             'seagrass_soil_c_mg', 'seagrass_total_c_mg']
        ].sum().reset_index()
    )
    df_stock = df_stock.rename(columns={
        'seagrass_agb_c_mg':   'seagrass_agb_c_total_mg',
        'seagrass_bgb_c_mg':   'seagrass_bgb_c_total_mg',
        'seagrass_soil_c_mg':  'seagrass_soil_c_total_mg',
        'seagrass_total_c_mg': 'seagrass_total_c_stock_mg',
    })

    df_stock.to_csv(p.seagrass_carbon_stock_path, index=False)

    total_stock = df_stock['seagrass_total_c_stock_mg'].sum()
    print(
        f"✅ Saved seagrass carbon stock by countries: {p.seagrass_carbon_stock_path}  "
        f"(global total {total_stock:,.0f} Mg C, {total_stock/1e9:.4f} Pg C)"
    )


def seagrass_storage_value(p):
    """
    Per-country seagrass storage value (USD), per pool.

    Same physical/GEP split as mangrove and salt marsh. Output columns:
        seagrass_agb_storage_value, seagrass_bgb_storage_value,
        seagrass_soil_storage_value, seagrass_storage_value.
    """
    publish_inputs(p)
    p.seagrass_storage_value_path = os.path.join(p.cur_dir, "seagrass_storage_value_by_countries2019.csv")
    if not p.run_this:
        return
    p.seagrass_storage_value_path = _calculate_storage_value(
        p,
        stock_csv_path=getattr(p, 'seagrass_carbon_stock_path', None),
        pool_columns={
            'seagrass_agb_c_total_mg':   'seagrass_agb_storage_value',
            'seagrass_bgb_c_total_mg':   'seagrass_bgb_storage_value',
            'seagrass_soil_c_total_mg':  'seagrass_soil_storage_value',
            'seagrass_total_c_stock_mg': 'seagrass_storage_value',
        },
        ecosystem_label='Seagrass',
        output_csv_filename='seagrass_storage_value_by_countries2019.csv',
    )


def salt_marsh_carbon_stock(p):
    """
    Compute per-country salt marsh carbon stocks via per-pixel density x ha_per_cell.

    Delegates to coastal_carbon_functions.compute_salt_marsh_carbon_stock_with_maxwell:
      - AGB: Chmura et al. 2003 median with tropical productivity boost
      - BGB: AGB x 2.5 (extensive root systems)
      - SOC: Maxwell et al. 2024 MarSOC raster at p.salt_marsh_soc_path (if set);
             fallback to latitude step function if path is missing or file absent.
    """
    publish_inputs(p)
    p.salt_marsh_carbon_stock_path = os.path.join(
        p.cur_dir, "salt_marsh_carbon_stock_by_countries2019.csv"
    )
    if not p.run_this:
        return

    _build_country_id_raster_if_needed(p)

    salt_marsh_mask_path = os.path.join(p.cur_dir, "salt_marsh_mask_10sec.tif")
    if not os.path.exists(salt_marsh_mask_path):
        coastal_carbon_functions.rasterize_polygons_to_template(
            vector_path=p.salt_marsh_vector_path,
            template_raster_path=p.ha_per_cell_10sec_path,
            out_path=salt_marsh_mask_path,
            default_value=1,
            dtype='uint8',
            nodata=0,
            all_touched=True,
        )

    gdf_countries = gpd.read_file(p.gep_regions_input_path)
    country_ids = gdf_countries[p.gep_regions_id_col].dropna().astype(int).unique()

    df_stock = coastal_carbon_functions.compute_salt_marsh_carbon_stock_with_maxwell(
        project_dir=p.cur_dir,
        salt_marsh_mask_path=salt_marsh_mask_path,
        country_id_raster_path=p.country_id_raster_path,
        ha_per_cell_path=p.ha_per_cell_10sec_path,
        country_ids=country_ids,
        maxwell_soc_path=getattr(p, 'salt_marsh_soc_path', None),
    )
    df_stock = df_stock.rename(columns={
        'agb_c_total_mg': 'salt_marsh_agb_c_total_mg',
        'bgb_c_total_mg': 'salt_marsh_bgb_c_total_mg',
        'soil_c_total_mg': 'salt_marsh_soil_c_total_mg',
        'total_c_total_mg': 'salt_marsh_total_c_stock_mg',
    })
    df_stock.to_csv(p.salt_marsh_carbon_stock_path, index=False)
    print(f"✅ Saved salt marsh carbon stock by countries: {p.salt_marsh_carbon_stock_path}")


def combined_ecosystem_areas(p):
    """
    Combine mangrove, salt marsh, and seagrass areas plus carbon stocks into a
    single dataset. Seagrass is optional: skipped silently if its area / stock
    CSVs do not exist (e.g. include_seagrass=False on the run script).
    """
    publish_inputs(p)
    p.combined_area_path = os.path.join(p.cur_dir, "combined_ecosystem_areas.csv")
    if not p.run_this:
        return

    # Read mangrove areas
    df_mangrove = pd.read_csv(
        p.mangrove_area_by_countries_base_year_path.replace('.gpkg', '.csv')
    )
    df_mangrove = df_mangrove[[p.gep_regions_id_col, 'area_ha']].rename(
        columns={'area_ha': 'mangrove_area_ha'}
    )

    # Read salt marsh areas
    df_salt_marsh = pd.read_csv(
        p.salt_marsh_area_by_countries_base_year_path.replace('.gpkg', '.csv')
    )
    df_salt_marsh = df_salt_marsh[[p.gep_regions_id_col, 'area_ha']].rename(
        columns={'area_ha': 'salt_marsh_area_ha'}
    )

    # Read seagrass areas (optional, depends on include_seagrass)
    seagrass_area_gpkg = getattr(p, 'seagrass_area_by_countries_base_year_path', None)
    seagrass_area_csv = (
        seagrass_area_gpkg.replace('.gpkg', '.csv') if seagrass_area_gpkg else None
    )
    has_seagrass_area = bool(seagrass_area_csv) and os.path.exists(seagrass_area_csv)
    if has_seagrass_area:
        df_seagrass = pd.read_csv(seagrass_area_csv)
        df_seagrass = df_seagrass[[p.gep_regions_id_col, 'area_ha']].rename(
            columns={'area_ha': 'seagrass_area_ha'}
        )
    else:
        hb.log(
            "combined_ecosystem_areas: seagrass area CSV not found "
            f"({seagrass_area_csv!r}); seagrass area set to 0."
        )

    # Merge areas
    df_combined = df_mangrove.merge(
        df_salt_marsh, on=p.gep_regions_id_col, how='outer'
    )
    if has_seagrass_area:
        df_combined = df_combined.merge(
            df_seagrass, on=p.gep_regions_id_col, how='outer'
        )
    else:
        df_combined['seagrass_area_ha'] = 0

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
    # Each upstream stock task sets its own p.<ecosystem>_carbon_stock_path
    # pointing into its own task folder.
    mangrove_stock_path = p.mangrove_carbon_stock_path
    salt_marsh_stock_path = p.salt_marsh_carbon_stock_path
    seagrass_stock_path = getattr(p, 'seagrass_carbon_stock_path', None)
    mangrove_stock_cols = [
        p.gep_regions_id_col,
        'mangrove_agb_c_total_mg', 'mangrove_bgb_c_total_mg',
        'mangrove_soil_c_total_mg', 'mangrove_total_c_stock_mg',
    ]
    salt_marsh_stock_cols = [
        p.gep_regions_id_col,
        'salt_marsh_agb_c_total_mg', 'salt_marsh_bgb_c_total_mg',
        'salt_marsh_soil_c_total_mg', 'salt_marsh_total_c_stock_mg',
    ]
    seagrass_stock_cols = [
        p.gep_regions_id_col,
        'seagrass_agb_c_total_mg', 'seagrass_bgb_c_total_mg',
        'seagrass_soil_c_total_mg', 'seagrass_total_c_stock_mg',
    ]
    df_mangrove_stock = pd.read_csv(mangrove_stock_path)[mangrove_stock_cols]
    df_salt_marsh_stock = pd.read_csv(salt_marsh_stock_path)[salt_marsh_stock_cols]

    df_combined = df_combined.merge(df_mangrove_stock, on=p.gep_regions_id_col, how='left')
    df_combined = df_combined.merge(df_salt_marsh_stock, on=p.gep_regions_id_col, how='left')

    has_seagrass_stock = bool(seagrass_stock_path) and os.path.exists(seagrass_stock_path)
    if has_seagrass_stock:
        df_seagrass_stock = pd.read_csv(seagrass_stock_path)[seagrass_stock_cols]
        df_combined = df_combined.merge(
            df_seagrass_stock, on=p.gep_regions_id_col, how='left'
        )
    else:
        for col in seagrass_stock_cols[1:]:
            df_combined[col] = 0
        if has_seagrass_area:
            hb.log(
                "combined_ecosystem_areas: seagrass area present but seagrass "
                f"stock CSV missing ({seagrass_stock_path!r}); seagrass stock columns set to 0."
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

    # Read country info
    gdf_countries = gpd.read_file(p.gep_regions_input_path)
    df_countries = gdf_countries.drop(columns=['geometry'])

    # Merge with country info
    df_combined = df_combined.merge(
        df_countries, on=p.gep_regions_id_col, how='left'
    )

    df_combined.to_csv(p.combined_area_path, index=False)
    print(f"✅ Saved combined ecosystem areas: {p.combined_area_path}")


def gep_calculation(p):
    """
    GEP calculation for coastal blue carbon (mangrove + salt marsh + seagrass).
    Storage-only: per-ecosystem carbon stock (Mg C) x rental SCC ($/Mg C).

    Two outputs in p.cur_dir:
      - gep_by_country_base_year_r566.{csv,gpkg}: full per-eemarine-region
        detail (per-ecosystem area, stock pools, storage values).
      - gep_by_country_base_year.csv: final iso3_r250 aggregation. Built by
        keeping only marine '_EEZ' rows from the r566 frame, taking the
        coastal_carbon_storage_value as 'value', joining onto the canonical
        r264 correspondence (one row per iso3_r250), filling missing with 0,
        and summing per iso3_r250_label.
    """
    publish_inputs(p)
    service_results = {}
    p.results['coastal_carbon'] = service_results

    p.results['coastal_carbon']['gep_by_country_base_year_r566'] = os.path.join(
        p.cur_dir, "gep_by_country_base_year_r566.csv"
    )
    p.results['coastal_carbon']['gep_by_country_base_year'] = os.path.join(
        p.cur_dir, "gep_by_country_base_year.csv"
    )
    # Only register results this task actually writes. Per-year results (gep_by_country_year,
    # gep_by_year) belong to a multi-year run and are registered there, not in this base-year
    # valuation (same contract fix as terrestrial_carbon).

    final_csv = p.results['coastal_carbon']['gep_by_country_base_year']
    if hb.path_all_exist(list(service_results.values())):
        hb.log("gep_calculation: skipped (all registered results exist)")
        return

    r566_csv = p.results['coastal_carbon']['gep_by_country_base_year_r566']
    r566_gpkg = r566_csv.replace('.csv', '.gpkg')

    # Stage 1: per-eemarine-region (r566) frame.
    if _task_outputs_exist(r566_csv, r566_gpkg):
        df_gep = pd.read_csv(r566_csv)
        hb.log("gep_calculation: r566 cached, reusing.")
    else:
        hb.log(
            "gep_calculation: computing r566 storage values "
            "(mangrove + salt marsh + seagrass)..."
        )

        df_areas = pd.read_csv(p.combined_area_path)
        df_areas['year'] = 2019

        df_carbon_p = pd.read_excel(p.gep_price_input_path)
        df_carbon_p = df_carbon_p[['year', p.gep_price_convention]]
        df_gep = df_areas.merge(df_carbon_p, on='year', how='left')

        df_gep['mangrove_storage_value'] = (
            df_gep['mangrove_total_c_stock_mg'] * df_gep[p.gep_price_convention]
        )
        df_gep['salt_marsh_storage_value'] = (
            df_gep['salt_marsh_total_c_stock_mg'] * df_gep[p.gep_price_convention]
        )
        df_gep['seagrass_storage_value'] = (
            df_gep['seagrass_total_c_stock_mg'] * df_gep[p.gep_price_convention]
        )
        df_gep['coastal_carbon_storage_value'] = (
            df_gep['mangrove_storage_value']
            + df_gep['salt_marsh_storage_value']
            + df_gep['seagrass_storage_value']
        )

        df_gep = df_gep[df_gep['total_coastal_carbon_area_ha'] > 0]

        hb.df_write(df_gep, r566_csv)
        gdf_countries = gpd.read_file(p.gep_regions_input_path)
        gdf_gep = gdf_countries.merge(df_gep, on=p.gep_regions_id_col, how='right')
        gdf_gep.to_file(r566_gpkg, driver='GPKG')

        hb.log(
            f"r566 Mangrove   storage value (2019): "
            f"${df_gep['mangrove_storage_value'].sum():,.2f}"
        )
        hb.log(
            f"r566 Salt marsh storage value (2019): "
            f"${df_gep['salt_marsh_storage_value'].sum():,.2f}"
        )
        hb.log(
            f"r566 Seagrass   storage value (2019): "
            f"${df_gep['seagrass_storage_value'].sum():,.2f}"
        )
        hb.log(
            f"r566 Total      storage value (2019): "
            f"${df_gep['coastal_carbon_storage_value'].sum():,.2f}"
        )

    # Stage 2: iso3_r250 final.
    # EEZ rows only; take Total storage value as 'value'.
    df_eez_value = (
        df_gep[df_gep['eemarine_r566_label'].astype(str).str.endswith('_EEZ')]
        [['iso3_r250_label', 'coastal_carbon_storage_value']]
        .rename(columns={'coastal_carbon_storage_value': 'value'})
    )

    # r264 correspondence -> canonical row per iso3 (ee_r264_label == iso3_r250_label)
    # Filter prevents downstream sum from inflating values across multi-r264 iso3s.
    df_r264 = pd.read_csv(p.df_countries_csv_path)
    df_r264_canonical = df_r264[
        df_r264['ee_r264_label'] == df_r264['iso3_r250_label']
    ]

    df_merged = df_r264_canonical.merge(
        df_eez_value, on='iso3_r250_label', how='left'
    )
    df_merged['value'] = df_merged['value'].fillna(0)

    metadata_cols = [
        c for c in df_merged.columns if c not in ('iso3_r250_label', 'value')
    ]
    agg = {'value': 'sum'}
    agg.update({c: 'first' for c in metadata_cols})
    df_r250_final = (
        df_merged.groupby('iso3_r250_label', as_index=False).agg(agg)
    )

    df_r250_final.to_csv(final_csv, index=False)
    hb.log(
        f"Final iso3_r250 GEP saved: {final_csv}  "
        f"({len(df_r250_final)} iso3 countries, "
        f"total ${df_r250_final['value'].sum():,.2f})"
    )

    return df_r250_final['value'].sum()


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities (this module's variant --
    sidecar copying, PYTHONPATH, crash-loudly -- became the shared one)."""
    publish_inputs(p)
    utilities.render_service_results(p)

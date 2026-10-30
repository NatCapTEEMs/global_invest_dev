"""Marine carbon science: four ocean carbon pools valued at a rental price and shared out by area.

Ported from github.com/Observarun/marine-carbon-gep at commit 0dac21fd, keeping its geometry
and arithmetic. Each pool's global value is its stock in Pg C times 1e9 tonnes per Pg
times its rental price. A country receives the share of that value equal to its dissolved EEZ
area over the total ocean area. The ocean outside every EEZ is one high-seas row, the source's
ABNJ with id 999, which is never a country and never enters the country table.

Properties of the source construction, measured on eemarine_r566_correspondence.gpkg (md5
b7844333e878a97e36dd9a06e0bdd499) and recorded in the method note:
- The total ocean area sums the Marine EEZ and Nearest Open Ocean polygons with no union, so
  area where two polygons overlap is counted twice. The pairwise overlaps sum to about 1.28
  million km2 of the 360.67 million: 748,889 among open-ocean polygons, 527,594 between EEZ and
  open-ocean polygons, and 423 between different countries' EEZs.
- Where each double count lands differs, because the EEZ dissolve is within a country only.
  Overlap between two countries' EEZs stays in both countries' areas, so it is counted twice
  among the country rows. Overlap between an EEZ and an open-ocean polygon is counted once in the
  country and once in the high-seas residual. Overlap among open-ocean polygons is counted twice
  in the high-seas residual.
- The high-seas residual equals the summed Nearest Open Ocean polygons exactly, because no two
  EEZ polygons of the same country overlap, so the dissolve removes nothing.
"""
import pandas as pd

TONNES_PER_PG = 1e9
M2_PER_KM2 = 1e6

# The source's identifiers for the ocean beyond every EEZ. 999 sits in the ISO 3166-1
# user-assigned range so the id column stays integer.
HIGH_SEAS_ID = 999
HIGH_SEAS_LABEL = 'ABNJ'
HIGH_SEAS_NAME = 'High Seas (Areas Beyond National Jurisdiction)'

COUNTRY_COLUMNS = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name']


def ocean_polygons(gdf, name_pattern, area_crs):
    """The marine-correspondence rows whose name matches, reprojected and repaired.

    Args:
        gdf (gpd.GeoDataFrame): the eemarine correspondence layer.
        name_pattern (str): a regular expression on `eemarine_r566_name`.
        area_crs (str): the equal-area CRS the areas are measured in.

    Returns:
        gpd.GeoDataFrame: the matching rows in `area_crs`, invalid geometries made valid.
    """
    selected = gdf[gdf['eemarine_r566_name'].str.contains(name_pattern, na=False)].copy()
    selected = selected.to_crs(area_crs)
    selected['geometry'] = selected.geometry.make_valid()
    return selected


def eez_areas_by_country(gdf, eez_name_pattern, area_crs):
    """Each country's EEZ area, its polygons dissolved first so overlaps within it count once.

    A polygon with no country id stays its own row (dropna=False) rather than disappearing.

    Returns:
        pd.DataFrame: id, label, name, `eez_area_km2` and `proportion_of_global_eez`, largest
        first.
    """
    eez = ocean_polygons(gdf, eez_name_pattern, area_crs)[COUNTRY_COLUMNS + ['geometry']]
    dissolved = eez.dissolve(by='iso3_r250_id', as_index=False, dropna=False,
                             aggfunc={'iso3_r250_label': 'first', 'iso3_r250_name': 'first'})
    dissolved['iso3_r250_id'] = dissolved['iso3_r250_id'].astype('Int64')
    dissolved['eez_area_km2'] = dissolved.geometry.area / M2_PER_KM2
    dissolved['proportion_of_global_eez'] = (dissolved['eez_area_km2']
                                             / dissolved['eez_area_km2'].sum())
    return (dissolved[COUNTRY_COLUMNS + ['eez_area_km2', 'proportion_of_global_eez']]
            .sort_values('eez_area_km2', ascending=False).reset_index(drop=True))


def total_ocean_area_km2(gdf, ocean_name_pattern, area_crs):
    """The summed area of every ocean polygon, EEZ and open ocean, with no union."""
    return float(ocean_polygons(gdf, ocean_name_pattern, area_crs).geometry.area.sum()
                 / M2_PER_KM2)


def pool_value_usd(pool):
    """One pool's global annual value: stock in Pg C x tonnes per Pg x rental price per tonne C."""
    return pool['stock_pg_c'] * TONNES_PER_PG * pool['rental_price_usd_per_tc_per_year']


def allocate_marine_carbon_value(eez_areas, total_ocean_area_km2, pools):
    """Share every pool's value out by area: each country by its EEZ, the rest as high seas.

    Args:
        eez_areas (pd.DataFrame): `eez_areas_by_country` output.
        total_ocean_area_km2 (float): `total_ocean_area_km2` output.
        pools (list): one dict per pool with `pool`, `stock_pg_c` and
            `rental_price_usd_per_tc_per_year`.

    Returns:
        pd.DataFrame: one row per country plus the high-seas row, with `area_km2`,
        `share_of_total_ocean_area`, one `allocated_value_<pool>_usd` column per pool and their
        sum `allocated_value_usd`, largest first.

    Raises:
        ValueError: when the EEZs together exceed the ocean they sit in.
    """
    df = eez_areas[COUNTRY_COLUMNS + ['eez_area_km2']].rename(columns={'eez_area_km2': 'area_km2'})
    high_seas_area_km2 = total_ocean_area_km2 - df['area_km2'].sum()
    if high_seas_area_km2 < 0:
        raise ValueError('EEZ areas (%.0f km2) exceed the total ocean area (%.0f km2); check the '
                         'polygon selection.' % (df['area_km2'].sum(), total_ocean_area_km2))
    high_seas = pd.DataFrame([{'iso3_r250_id': HIGH_SEAS_ID, 'iso3_r250_label': HIGH_SEAS_LABEL,
                               'iso3_r250_name': HIGH_SEAS_NAME, 'area_km2': high_seas_area_km2}])
    df = pd.concat([df, high_seas], ignore_index=True)
    df['share_of_total_ocean_area'] = df['area_km2'] / total_ocean_area_km2
    value_columns = []
    for pool in pools:
        column = 'allocated_value_%s_usd' % pool['pool']
        df[column] = pool_value_usd(pool) * df['share_of_total_ocean_area']
        value_columns.append(column)
    df['allocated_value_usd'] = df[value_columns].sum(axis=1)
    return df.sort_values('allocated_value_usd', ascending=False).reset_index(drop=True)


def national_and_high_seas(allocation):
    """The allocation split into the country rows and the one high-seas row.

    The country rows carry the total as `marine_carbon_gep`, the account's column name, and keep
    every pool column. The high-seas row is returned separately so it can be written on its own:
    the country table is put on the account's country list, where a row labelled ABNJ has no
    place and would be dropped.

    Raises:
        ValueError: unless exactly one row carries the high-seas id and label together.
    """
    is_high_seas = ((allocation['iso3_r250_id'] == HIGH_SEAS_ID)
                    | (allocation['iso3_r250_label'] == HIGH_SEAS_LABEL))
    high_seas = allocation[is_high_seas]
    if len(high_seas) != 1 or high_seas['iso3_r250_label'].iloc[0] != HIGH_SEAS_LABEL \
            or high_seas['iso3_r250_id'].iloc[0] != HIGH_SEAS_ID:
        raise ValueError('expected exactly one high-seas row with id %d and label %s, found %s'
                         % (HIGH_SEAS_ID, HIGH_SEAS_LABEL,
                            high_seas[['iso3_r250_id', 'iso3_r250_label']].values.tolist()))
    national = allocation[~is_high_seas].rename(columns={'allocated_value_usd': 'marine_carbon_gep'})
    return national.reset_index(drop=True), high_seas.reset_index(drop=True)

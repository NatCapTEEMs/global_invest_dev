"""Terrestrial-carbon science: the density lookup, the two raster reducers, and the frame
arithmetic the shock task and the GEP valuation share.

Nothing here reads or writes a table. The task layer opens the CSVs and the workbook, hands the
frames in, and writes back what it gets, so every step below can be pinned on a hand-built input
in the test suite. The two raster reducers still stream their own pixels through hazelbean,
because the rasters they cross are global at 300 m and will not fit in memory.

The valuation is storage-only: mapped carbon density x hectares per cell = stock (Mg C), stock x
the base-year rental social cost of carbon = GEP ($).
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
import hazelbean as hb

# The raw-Spawn density build (uint -> float32 scaling, aboveground+belowground add) is a one-off
# base-data job, not part of the per-run tree; it lives in howto/rebuild_spawn_total_carbon_density.md
# and the run consumes its finished product (spawn_total_biomass_carbon_2010.tif) from base_data.

# hazelbean's output_data_type code for Float32. The LULC grid the density rasters are written
# against is uint8, which would round every density to a whole Mg C/ha.
GDAL_FLOAT32 = 6

# The shocks are reported in percent, so the ratio is scaled by this before it is written.
PERCENT = 100.0

# The country attributes every GEP per-country CSV carries, in the order the CSV writes them.
CARBON_ATTR_COLS = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
                    'continent', 'region_un', 'region_wb', 'income_grp', 'subregion']

# The lookup is keyed on whole numbers stored as float in the table and as unsigned integers in
# the rasters, so both sides are cast to this before they are matched by value.
LOOKUP_KEY_DTYPE = 'int64'


# =============================================================================
# Raster reducers. Both stream their pixels because the rasters are global at
# 300 m; both return what they measured instead of writing it.
# =============================================================================

def stack_layers_summary(group_layer1_path, group_layer2_path, value_layer_path,
                         group1_name="group1", group2_name="group2", value_name="value"):
    """A value raster summarized over the pairs of two category rasters.

    Args:
        group_layer1_path (str): first grouping raster.
        group_layer2_path (str): second grouping raster, on the same grid.
        value_layer_path (str): the raster being summarized, on the same grid.
        group1_name (str): column the first grouping raster is written under.
        group2_name (str): column the second grouping raster is written under.
        value_name (str): stem of the four summary columns.

    Returns:
        pd.DataFrame: one row per (group1, group2) pair, carrying the mean, min, max and count
        of the value raster over the cells where all three rasters are valid.
    """
    from osgeo import gdal
    # hb.iterblocks streams the value raster block by block; the two category rasters share its grid, so
    # read them at the same window (raw gdal -- hb has no aligned multi-raster block reader). A cell is
    # kept only where all three are valid. Groupby per block, then
    # combine: only ~thousands of (g1, g2) pairs, so the accumulator stays tiny and nothing 33 GB is
    # written (unlike composite-key + zonal).
    ds1 = gdal.Open(group_layer1_path); g1b = ds1.GetRasterBand(1)   # hold the datasets, else the band handle dies
    ds2 = gdal.Open(group_layer2_path); g2b = ds2.GetRasterBand(1)
    ndv1, ndv2 = g1b.GetNoDataValue(), g2b.GetNoDataValue()
    dsv = gdal.Open(value_layer_path); ndv_val = dsv.GetRasterBand(1).GetNoDataValue()

    parts = []
    for offset, value_block in tqdm(hb.iterblocks((value_layer_path, 1)), desc="Summarizing blocks"):
        w = (offset['xoff'], offset['yoff'], offset['win_xsize'], offset['win_ysize'])
        v = value_block.astype('float32').ravel()
        g1 = g1b.ReadAsArray(*w).ravel()
        g2 = g2b.ReadAsArray(*w).ravel()

        keep = ~np.isnan(v)
        if ndv_val is not None: keep &= v != ndv_val
        if ndv1 is not None: keep &= g1 != ndv1
        if ndv2 is not None: keep &= g2 != ndv2
        if not keep.any():   # block is all-nodata -- expected at raster edges
            continue

        block = pd.DataFrame({group1_name: g1[keep], group2_name: g2[keep], value_name: v[keep]})
        parts.append(block.groupby([group1_name, group2_name], as_index=False)[value_name]
                     .agg(_sum='sum', _min='min', _max='max', _count='count'))

    summary = pd.concat(parts).groupby([group1_name, group2_name], as_index=False).agg(
        _sum=('_sum', 'sum'), _min=('_min', 'min'), _max=('_max', 'max'), _count=('_count', 'sum'))
    summary[f'{value_name}_mean'] = summary['_sum'] / summary['_count']
    summary = summary.rename(columns={'_min': f'{value_name}_min', '_max': f'{value_name}_max',
                                      '_count': f'{value_name}_count'})
    return summary[[group1_name, group2_name, f'{value_name}_mean', f'{value_name}_min',
                    f'{value_name}_max', f'{value_name}_count']]


def carbon_density_lookup(lookup_df):
    """The (carbon_zone_id, lulc_id) -> Mg C/ha map the density rasters are built from.

    Args:
        lookup_df (pd.DataFrame): a LONG/TIDY table with one row per (carbon_zone_id, lulc_id) and
            a single carbon_density_mean column. NOT a wide table indexed by carbon_zone_id with
            one column per LULC type: this reads a `lulc_id` COLUMN, so a wide table matches
            nothing and yields an all-NoData density raster.

    Returns:
        pd.Series: float32 densities on a (carbon_zone_id, lulc_id) index. A pair absent from the
        table reindexes to NaN downstream, so a missing combination becomes NoData, never 0.
    """
    return lookup_df.astype({'carbon_zone_id': LOOKUP_KEY_DTYPE, 'lulc_id': LOOKUP_KEY_DTYPE}).set_index(
        ['carbon_zone_id', 'lulc_id'])['carbon_density_mean'].astype('float32')


def generate_carbon_density_raster(lulc_path, cz_path, density_lookup, out_path):
    """Write the per-cell carbon density a land-cover map implies under the lookup.

    Args:
        lulc_path (str): land-use land-cover raster.
        cz_path (str): carbon-zone raster, on the same grid.
        density_lookup (pd.Series): the lookup from carbon_density_lookup.
        out_path (str): where the density raster is written.

    Raises:
        ValueError: when the lookup matched no valid cell at all, which is what an ESA-classed map
            fed to the SEALS7-keyed lookup looks like.
    """
    # Tally what the lookup actually matched, so a wholesale mismatch raises instead of flowing on as an
    # all-NoData raster. The concrete trap: an ESA-classed map (ids 10..220) against the SEALS7-keyed
    # lookup (ids 1..7) matches nothing, the density raster is all NaN, the zonal means drop every zone,
    # and downstream emits an empty shock CSV -- GTAP then runs a silent zero. Per-pair misses stay NaN
    # (a legitimately absent combination is not an error); only zero matches over valid cells raises.
    lulc_ndv = hb.get_ndv_from_path(lulc_path)
    seen_lulc_ids = set()
    counts = {'valid': 0, 'matched': 0}

    def carbon_density(lulc_block, cz_block):
        idx = pd.MultiIndex.from_arrays([cz_block.astype(LOOKUP_KEY_DTYPE).ravel(),
                                         lulc_block.astype(LOOKUP_KEY_DTYPE).ravel()])
        out = density_lookup.reindex(idx).to_numpy('float32').reshape(lulc_block.shape)
        valid = lulc_block != lulc_ndv if lulc_ndv is not None else np.ones(lulc_block.shape, dtype=bool)
        seen_lulc_ids.update(np.unique(lulc_block[valid]).tolist())
        counts['valid'] += int(valid.sum())
        counts['matched'] += int(np.isfinite(out).sum())
        return out

    hb.raster_calculator_flex([lulc_path, cz_path], carbon_density, out_path,
                              datatype=GDAL_FLOAT32, ndv=np.nan)
    if counts['valid'] > 0 and counts['matched'] == 0:
        raise ValueError(
            f"carbon density lookup matched ZERO of {counts['valid']} valid cells: the LULC raster "
            f"{lulc_path} carries classes {sorted(seen_lulc_ids)[:20]} but the lookup is keyed on "
            f"lulc_id {sorted(set(density_lookup.index.get_level_values('lulc_id')))} "
            f"(likely an ESA-classed map fed to the SEALS7-keyed lookup, or a wrong carbon-zones raster). "
            f"Refusing to emit an all-NoData density raster.")
    print(f"Saved: {out_path}")

# Promoted to global_invest.utilities on its second caller (pollination GEP); re-exported here
# so existing imports keep working.
from global_invest.utilities import summarize_raster_by_region  # noqa: F401


# =============================================================================
# Shock arithmetic. The task measures a mean carbon density per zone per map
# year; these functions are everything that happens to those means afterwards.
# =============================================================================

def shock_percent(scenario_values, baseline_values, denominator_values=None):
    """The scenario's departure from its baseline, in percent.

    The two measures the chain reports share this numerator and differ only in what they
    divide by: the contemporaneous measure divides by that year's baseline, the fixed-base
    measure by the base year's, which is what `denominator_values` supplies.

    Args:
        scenario_values (pd.Series): per-zone scenario means for one year.
        baseline_values (pd.Series): per-zone baseline means for the same year.
        denominator_values (pd.Series): what to divide by, defaulting to baseline_values.

    Returns:
        pd.Series: percent departure, with a zero denominator giving NaN rather than an
        infinite shock.
    """
    denominator = baseline_values if denominator_values is None else denominator_values
    return (scenario_values - baseline_values) / denominator.replace(0, np.nan) * PERCENT


def interpolate_annual_shock(years, anchor_years, anchor_values, base_year):
    """Anchor-year shocks spread over every year, starting from no shock at the base year.

    The chain computes a shock only at the years the scenario maps exist for; the economic
    model reads one value per year, so the anchors are joined by straight lines with the
    base year pinned at zero.
    """
    return np.interp(years, [base_year] + list(anchor_years), [0.0] + list(anchor_values))


def zone_labels_from_boundary(regions_df, id_column, endw_column, reg_column, endw_format):
    """Zone id -> the (ENDW, REG) pair every shock row is keyed on.

    The zonal summary drops empty zones and keys what is left on the boundary's own stable id
    column, so the shock rows are matched on that id and never on gpkg row position.

    Args:
        regions_df (pd.DataFrame): the boundary attributes, one row per zone.
        id_column (str): the stable zone id column, falling back to `id` then row position.
        endw_column (str): the column the ENDW label is built from.
        reg_column (str): the column carrying the REG label.
        endw_format (str): a percent-format turning a numeric AEZ id into its label, or None
            when the boundary already carries the label.

    Returns:
        dict: zone id -> (ENDW, REG).
    """
    def endw_label(value):
        return endw_format % int(value) if endw_format is not None else value

    return {(int(row[id_column]) if id_column in row.index else row.get('id', position)):
            (endw_label(row[endw_column]), row[reg_column])
            for position, row in regions_df.iterrows()}


def dynamic_shock_rows(scenario_by_year, baseline_by_year, baseline_at_base_year, zone_labels,
                       base_year, sector, scenario):
    """Anchor-year zone means expanded to one row per zone and year, on both denominators.

    shock_pct is the contemporaneous measure and is what the economic model reads: afeall is a
    productivity deviation from the baseline path, so it is normalised by the year's own baseline.
    shock_pct_fixedbase divides the same numerator by the base year's baseline instead, which is
    the share of base-year value that pollination also reports.

    Args:
        scenario_by_year (dict): anchor year -> pd.Series of per-zone mean carbon density.
        baseline_by_year (dict): anchor year -> the same for the nature-off baseline.
        baseline_at_base_year (pd.Series): per-zone baseline density at the base year, or None
            when the base-year map is not available, which leaves shock_pct_fixedbase NaN.
        zone_labels (dict): zone id -> (ENDW, REG). A zone the boundary does not label is dropped.
        base_year (int): the year the interpolation starts from, at no shock.
        sector (str): the GTAP activity the shock applies to.
        scenario (str): the scenario label written into every row.

    Returns:
        list: dicts, one per zone and year from base_year through the last anchor year.
    """
    anchor_years = sorted(scenario_by_year)
    all_years = list(range(base_year, anchor_years[-1] + 1))
    contemporaneous = pd.DataFrame({
        y: shock_percent(scenario_by_year[y], baseline_by_year[y]) for y in anchor_years}).dropna()
    fixedbase = (pd.DataFrame({
        y: shock_percent(scenario_by_year[y], baseline_by_year[y], baseline_at_base_year)
        for y in anchor_years}).dropna()
        if baseline_at_base_year is not None else None)

    rows = []
    for zone_id, contemp_anchors in contemporaneous.iterrows():
        if zone_id not in zone_labels:
            continue
        endw, reg = zone_labels[zone_id]
        annual_contemp = interpolate_annual_shock(all_years, anchor_years, contemp_anchors.values, base_year)
        if fixedbase is not None and zone_id in fixedbase.index:
            annual_fixed = interpolate_annual_shock(all_years, anchor_years,
                                                    fixedbase.loc[zone_id].values, base_year)
        else:
            annual_fixed = [np.nan] * len(all_years)
        for year, contemp_value, fixed_value in zip(all_years, annual_contemp, annual_fixed):
            # Explicit, same-named columns in both ES files (carbon + pollination) for the #14 diagnostic.
            rows.append({'ENDW': endw, 'ACTS': sector, 'REG': reg, 'scenario': scenario,
                         'year': year, 'shock_pct': contemp_value,
                         'shock_pct_fixedbase': fixed_value, 'shock_pct_contemp': contemp_value})
    return rows


def static_shock_rows(baseline_values, scenario_values, scenario, sector, base_year, end_year):
    """The frozen table's scenario-minus-baseline difference, ramped linearly from the base year.

    Only zones present in both series are shocked: a zone the scenario or the baseline does not
    cover has no difference to ramp, and inventing one would put a fabricated number into the
    economic model.

    Args:
        baseline_values (pd.Series): per-zone nature-off baseline value, indexed by (ENDW, REG).
        scenario_values (pd.Series): per-zone scenario value over the same index.
        scenario (str): the scenario label written into every row.
        sector (str): the GTAP activity the shock applies to.
        base_year (int): the year the ramp starts from, at zero.
        end_year (int): the year the ramp reaches the full difference.

    Returns:
        list: dicts, one per zone and year from base_year through end_year.
    """
    common = baseline_values.index.intersection(scenario_values.index)
    shock = (scenario_values.loc[common] - baseline_values.loc[common]).dropna()
    n_years = end_year - base_year

    rows = []
    for year in range(base_year, end_year + 1):
        fraction = (year - base_year) / n_years
        for (endw, reg), value in shock.items():
            rows.append({'ENDW': endw, 'ACTS': sector, 'REG': reg,
                         'scenario': scenario, 'year': year, 'shock_pct': value * fraction})
    return rows


# =============================================================================
# GEP valuation arithmetic. The zonal summary comes in on the r264 aggregation
# surface, and the only work is pricing it and getting to one row per country.
# =============================================================================

def collapse_regions_to_countries(df_regions, df_price, price_column):
    """Per-region carbon stock summed to one row per country, priced at the base-year carbon price.

    Summing the r264-expanded table as it stands would count a split country once per sub-region
    (China spans 6 r264 rows, India 6, France, Turkey, the UK and Pakistan 2), so the sum is taken
    on the r250 country id and the country attributes are attached afterwards from one
    representative sub-region each. This table is the source of truth for the national total and
    for every aggregation.

    Args:
        df_regions (pd.DataFrame): the zonal summary, one row per r264 region, with
            iso3_r250_id, year, total and the country attribute columns.
        df_price (pd.DataFrame): the carbon price by year, carrying `year` and `price_column`.
        price_column (str): the price convention in force, e.g. 'rental scc r2%'.

    Returns:
        pd.DataFrame: the attribute columns, year, quantity, price and terrestrial_carbon_gep,
        one row per country and year.
    """
    df_quantity = (df_regions.groupby(['iso3_r250_id', 'year'], as_index=False)['total'].sum()
                   .rename(columns={'total': 'terrestrial_carbon_quantity'}))
    df_gep = df_quantity.merge(df_price, how='left', on='year')
    df_gep['terrestrial_carbon_gep'] = df_gep['terrestrial_carbon_quantity'] * df_gep[price_column]

    attributes = df_regions[CARBON_ATTR_COLS].drop_duplicates('iso3_r250_id')
    return df_gep.merge(attributes, how='left', on='iso3_r250_id')[
        CARBON_ATTR_COLS + ['year', 'terrestrial_carbon_quantity', price_column, 'terrestrial_carbon_gep']]


def expand_country_values_to_regions(df_regions, df_gep_by_country):
    """Each r264 region carrying its COUNTRY's GEP, for the map only.

    The sub-region rows repeat the national value rather than splitting it, so this table is
    never summed. It exists because the choropleth draws r264 polygons.

    Args:
        df_regions (pd.DataFrame): the zonal summary, one row per r264 region.
        df_gep_by_country (pd.DataFrame): the collapsed per-country table.

    Returns:
        pd.DataFrame: df_regions with terrestrial_carbon_gep attached.
    """
    return df_regions.merge(df_gep_by_country[['iso3_r250_id', 'terrestrial_carbon_gep']],
                            how='left', on='iso3_r250_id')

"""Terrestrial-carbon science: the density lookup, the density raster and the GEP valuation
arithmetic.

The shock arithmetic the carbon seam runs on its zone means (shock_percent, dynamic_shock_rows,
summed_shock_rows, static_shock_rows) knows nothing about carbon and is shared with the other ES
seams, so it lives in global_invest.utilities.

Nothing here reads or writes a table. The task layer opens the CSVs and the workbook, hands the
frames in, and writes back what it gets, so every step below can be pinned on a hand-built input
in the test suite. The density raster still streams its own pixels through hazelbean, because the
rasters it crosses are global at 300 m and will not fit in memory.

The valuation is storage-only: mapped carbon density x hectares per cell = stock (Mg C), stock x
the base-year rental social cost of carbon = GEP ($).
"""
import numpy as np
import pandas as pd
import hazelbean as hb

from global_invest import utilities

# The raw-Spawn density build (uint -> float32 scaling, aboveground+belowground add) is a one-off
# base-data job, not part of the per-run tree; it is the gep_preprocess task, and the run consumes
# its finished product (spawn_total_biomass_carbon_2010.tif) from base_data.

# hazelbean's output_data_type code for Float32. The LULC grid the density rasters are written
# against is uint8, which would round every density to a whole Mg C/ha.
GDAL_FLOAT32 = 6

# The country attributes every GEP per-country CSV carries, in the order the CSV writes them.
CARBON_ATTR_COLS = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
                    'continent', 'region_un', 'region_wb', 'income_grp', 'subregion']

# The lookup is keyed on whole numbers stored as float in the table and as unsigned integers in
# the rasters, so both sides are cast to this before they are matched by value.
LOOKUP_KEY_DTYPE = 'int64'


# =============================================================================
# The density lookup and the density raster, which streams its pixels because
# the rasters are global at 300 m.
# =============================================================================


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
    hb.log(f"Saved: {out_path}")


# =============================================================================
# GEP valuation arithmetic. The zonal summary comes in on the r264 aggregation
# surface, and the only work is pricing it and getting to one row per country.
# =============================================================================

def collapse_regions_to_countries(df_regions, df_price, price_column):
    """Per-region carbon stock summed to one row per country, priced at the base-year carbon price.

    Args:
        df_regions (pd.DataFrame): the per-region carbon stock table.
        df_price (pd.DataFrame): the carbon price by year, carrying `year` and `price_column`.
        price_column (str): the price convention in force, e.g. 'rental scc r2%'.

    Returns:
        pd.DataFrame: the attribute columns, year, quantity, price and terrestrial_carbon_gep,
        one row per country and year.
    """
    df = utilities.collapse_regions_to_countries(
        df_regions, CARBON_ATTR_COLS, 'terrestrial_carbon_quantity')
    df = df.merge(df_price, how='left', on='year')
    df['terrestrial_carbon_gep'] = df['terrestrial_carbon_quantity'] * df[price_column]
    return df[CARBON_ATTR_COLS + ['year', 'terrestrial_carbon_quantity', price_column,
                                  'terrestrial_carbon_gep']]




# =============================================================================
# Scenario valuation at the social cost of carbon. The same arithmetic as the GEP
# valuation above -- stock x the base-year price -- applied to every scenario map
# and reported as the change from the base year.
# =============================================================================

def scc_valuation_rows(stock_by_scenario_year, base_stock, price, base_year, retained=None,
                       from_year=None, cut_source=None, cut_label=None, baseline_by_year=None):
    """Carbon stock per country and year, valued at one price, as its change from the base year.

    Args:
        stock_by_scenario_year (dict): scenario -> {anchor year -> pd.Series of Mg C per country,
            indexed by iso3_r250_id}.
        base_stock (pd.Series): Mg C per country in the base year, the same for every scenario.
        price (float): $ per Mg C, the GEP price convention at the GEP base year.
        base_year (int): the year every change is measured from.
        retained, from_year, cut_source, cut_label: the reduced-provision configuration. When all
            four are given, `cut_label` is derived from `cut_source` by retaining `retained` of its
            stock from `from_year` on, and any modelled rows under `cut_label` are replaced.
        baseline_by_year (dict): anchor year -> pd.Series of Mg C per country for the nature-off
            baseline at the same year. When given, the change from that contemporaneous baseline
            is reported beside the change from the base year.

    Returns:
        pd.DataFrame: iso3_r250_id, scenario, year (base_year..last anchor, annual, PCHIP between
        anchors), stock_mgc, delta_stock_mgc, delta_value_usd, and with a baseline
        delta_stock_vs_baseline_mgc, delta_value_vs_baseline_usd.
    """
    cut = all(x is not None for x in (retained, from_year, cut_source, cut_label))
    frames = []
    for scenario, by_year in stock_by_scenario_year.items():
        if cut and scenario == cut_label:
            continue
        anchors = sorted(by_year)
        years = list(range(base_year, anchors[-1] + 1))
        table = pd.DataFrame({base_year: base_stock, **{y: by_year[y] for y in anchors}})
        table = table.dropna()
        # PCHIP on the STOCK in Mg C -- the physical quantity -- never the multiplicative
        # productivity transform: a carbon account is additive and does not compound. Anchors are
        # preserved exactly, so cumulative change and implied cumulative emissions match straight
        # lines between anchors, while the annual flow does not jump at the reporting years. Against
        # straight lines on the bounded outputs: anchor identity 2.4e-07 Mg C, cumulative emissions
        # identical, mean flow jump at 2030/2040 2.9e6 -> 6.3e5 Mg C/yr, NPV @1.4% -3.82%.
        _present = [c for c in table.columns if table[c].notna().any()]
        annual = pd.DataFrame(
            utilities.anchored_series(years, _present, table[_present].to_numpy().T).T,
            index=table.index, columns=years)
        long = annual.stack().rename('stock_mgc').reset_index()
        long.columns = ['iso3_r250_id', 'year', 'stock_mgc']
        long['scenario'] = scenario
        frames.append(long)
        if cut and scenario == cut_source:
            stressed = long.copy()
            stressed['scenario'] = cut_label
            stressed.loc[stressed['year'] >= from_year, 'stock_mgc'] *= retained
            frames.append(stressed)
    out = pd.concat(frames, ignore_index=True)
    base = base_stock.rename('base_stock_mgc').reset_index()
    base.columns = ['iso3_r250_id', 'base_stock_mgc']
    out = out.merge(base, on='iso3_r250_id', how='left')
    out['delta_stock_mgc'] = out['stock_mgc'] - out['base_stock_mgc']
    out['delta_value_usd'] = out['delta_stock_mgc'] * price
    columns = ['iso3_r250_id', 'scenario', 'year', 'stock_mgc', 'delta_stock_mgc', 'delta_value_usd']
    if baseline_by_year is not None:
        anchors = sorted(baseline_by_year)
        years = list(range(base_year, anchors[-1] + 1))
        table = pd.DataFrame({base_year: base_stock, **{y: baseline_by_year[y] for y in anchors}}).dropna()
        # The baseline stock is interpolated the same way: PCHIP on the stock, anchors exact.
        _present = [c for c in table.columns if table[c].notna().any()]
        annual = pd.DataFrame(
            utilities.anchored_series(years, _present, table[_present].to_numpy().T).T,
            index=table.index, columns=years)
        long = annual.stack().rename('baseline_stock_mgc').reset_index()
        long.columns = ['iso3_r250_id', 'year', 'baseline_stock_mgc']
        out = out.merge(long, on=['iso3_r250_id', 'year'], how='left')
        out['delta_stock_vs_baseline_mgc'] = out['stock_mgc'] - out['baseline_stock_mgc']
        out['delta_value_vs_baseline_usd'] = out['delta_stock_vs_baseline_mgc'] * price
        columns += ['delta_stock_vs_baseline_mgc', 'delta_value_vs_baseline_usd']
    return out[columns]


def plot_scc_valuation(df, attributes, price, price_column, base_year, out_path, aside=None):
    """Three panels: the world's change in carbon value by scenario from the base year, the same
    against the nature-off baseline at the same year, and the last year's change by continent.

    Args:
        df (pd.DataFrame): scc_valuation_rows output.
        attributes (pd.DataFrame): iso3_r250_id -> continent, one row per country.
        price (float): the price used, for the title.
        price_column (str): its convention name, for the title.
        base_year (int): the year changes are measured from.
        out_path (str): where the PNG goes.
        aside (str): a scenario to keep off the panels and report in the caption instead -- the
            reduced-provision scenario removes a fifth of the stock at a stroke and flattens every
            other line to the axis when drawn with them.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    shown = df if aside is None else df[df['scenario'] != aside]
    last_year = int(df['year'].max())
    world = shown.groupby(['scenario', 'year'])['delta_value_usd'].sum().unstack('scenario') / 1e9
    has_baseline = 'delta_value_vs_baseline_usd' in shown.columns
    last = shown[shown['year'] == last_year].merge(attributes, on='iso3_r250_id', how='left')
    by_continent = last.groupby(['continent', 'scenario'])['delta_value_usd'].sum().unstack('scenario') / 1e9

    n = 3 if has_baseline else 2
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.2))
    ax = axes[0]
    world.plot(ax=ax, linewidth=1.8)
    ax.axhline(0, color='0.4', linewidth=0.8)
    ax.set_title('World: change from %d' % base_year)
    ax.set_ylabel('billion USD per year')
    ax.set_xlabel('')
    ax.legend(title='scenario', fontsize=8, title_fontsize=8, frameon=False)
    if has_baseline:
        vs = shown.groupby(['scenario', 'year'])['delta_value_vs_baseline_usd'].sum().unstack('scenario') / 1e9
        ax = axes[1]
        vs.plot(ax=ax, linewidth=1.8, legend=False)
        ax.axhline(0, color='0.4', linewidth=0.8)
        ax.set_title('World: change from the nature-off baseline, same year')
        ax.set_ylabel('billion USD per year')
        ax.set_xlabel('')
    ax = axes[-1]
    by_continent.plot(kind='bar', ax=ax, width=0.8, legend=False)
    ax.axhline(0, color='0.4', linewidth=0.8)
    ax.set_title('%d: change from %d by continent' % (last_year, base_year))
    ax.set_ylabel('billion USD per year')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    for a in axes[:-1]:
        a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: '%d' % v))
    caption = 'Carbon stock valued at %s: %.2f USD per Mg C per year, held at its %s value' % (
        price_column, price, 'base-year')
    if aside is not None and aside in set(df['scenario']):
        away = df[(df['scenario'] == aside) & (df['year'] == last_year)]['delta_value_usd'].sum() / 1e9
        caption += '. %s not drawn: %.0f bn per year at %d' % (aside, away, last_year)
    fig.suptitle(caption, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

"""Timber provision: the committed per-country output from the drive's Forestry pipeline.

The Forestry folder carries the raster pipeline (forest value, biomass, aligned log prices) and
its committed per-country output in the account's vocabulary. The upstream raster valuation is
taken as given; the committed table is the anchor.

Timber provision valuation calculation, reimplemented from the drive's Forestry rasters.

The Forestry pipeline values managed forestry at 10 arcsec: annual biomass yield (m3/ha, MC2
forest types x Tian et al. timber regions, NPP-adjusted) times the national log price
(USD/m3, Siikamaki and Santiago-Avila 2014), converted from gross to net return with the
GTAP land factor share (Damania et al. 2023), minus per-pixel transport cost (Weiss et al.
travel time x trucking cost). Pixels are kept only where the Lesiv et al. (2022) forest
management map marks current timber activity (categories 2-5 on forested ESA classes);
negative net returns and nodata floor to zero.

Verified against the staged rasters (base_data/global_invest/timber_provision/input/):

  - current_forest_value.tif equals forestry_land_share_return_tcost_before_2022_07_23.tif
    bit-exactly on every one of its 265,210,379 positive pixels (0 mismatches globally);
    the remaining 802,119,863 positive net-return pixels are dropped by the management
    mask, which is external to the staged layers (it is NOT derivable from them).
  - The decomposition net_return = land_factor_share * biomass * price - transport_cost
    reproduces the net-return raster with a per-region fitted share (R2 0.78-0.9999 over
    12 continental windows, median relative error 0.5-4%): the share applies at GTAP
    region x AEZ level with separate plantation/natural estimates, so it is an input
    here, not a constant.
  - The committed per-country table (timber_provision_gep.csv, $88.74bn over 166
    countries) is the plain pixel sum of current_forest_value.tif inside each r250
    country polygon (ratio 1.0000 on all countries tested): the raster is per-pixel USD.
"""
import numpy as np


def timber_gep_by_country(timber_df, countries_df):
    """The committed timber table joined onto the r250 country list, one row per country."""
    df = countries_df.merge(
        timber_df[['iso3_r250_label', 'forestry_gep']].rename(
            columns={'forestry_gep': 'timber_provision_gep'}),
        on='iso3_r250_label', how='left')
    return df


NET_RETURN_NDV = -99999.0


def net_forest_return(annual_biomass, log_price, transport_cost, land_factor_share):
    """Per-pixel net forestry return in USD: the appendix decomposition.

    Args:
        annual_biomass (np.ndarray): Annual harvestable biomass, m3 per pixel-hectare.
        log_price (np.ndarray): National producer log price, USD/m3.
        transport_cost (np.ndarray): Per-pixel transport cost, USD.
        land_factor_share (np.ndarray or float): GTAP land factor share converting gross
            to net returns (regional; broadcastable against the rasters).

    Returns:
        np.ndarray: land_factor_share * annual_biomass * log_price - transport_cost.
    """
    return land_factor_share * annual_biomass * log_price - transport_cost


def forest_value_from_net_return(net_return, managed_mask, ndv=NET_RETURN_NDV):
    """The current-forest-value raster: net return kept only where forestry is current.

    Args:
        net_return (np.ndarray): Net forestry return (may hold negatives and `ndv`).
        managed_mask (np.ndarray): Boolean, True where the Lesiv et al. (2022) management
            map marks current timber activity on forested ESA classes.
        ndv (float): The net-return raster's nodata value.

    Returns:
        np.ndarray: float32, net_return where managed and positive, else 0 (the value
        raster's own nodata convention).
    """
    kept = managed_mask & (net_return > 0) & (net_return != ndv)
    return np.where(kept, net_return, 0.0).astype(np.float32)


def timber_gep_from_zone_sums(zone_sums, countries_df):
    """One row per country: the zonal pixel sums keyed to iso3_r250_id.

    Args:
        zone_sums (np.ndarray): Per-zone sums from `sum_by_zone` (accumulated), indexed
            by iso3_r250_id.
        countries_df (pd.DataFrame): One row per country, with an iso3_r250_id column.

    Returns:
        pd.DataFrame: countries_df plus a timber_provision_gep column.
    """
    df = countries_df.copy()
    df["timber_provision_gep"] = zone_sums[df["iso3_r250_id"].to_numpy(dtype=int)]
    return df


# The CWoN rental alternative. The issues document recommends valuing this service from FAOSTAT and
# CWoN rather than from the staged value raster, and CWoN publishes the answer rather than the
# ingredients: `Forest, rents (current US$)`, the same reproducibility package the gas, oil, coal
# and hydropower rents come from. Both figures are published side by side because the choice
# between them is live: the raster is spatial and cannot be rebuilt here, the rent is a country
# table with no spatial detail and is the source every other rent-based service already uses.
CWON_FOREST_RENT_SERIES = 'Forest, rents (current US$)'


def cwon_forest_rent_by_country(rent_df, countries_df, year):
    """CWoN's published forest rent for one year, on the account's country rows.

    Args:
        rent_df (pd.DataFrame): forest_timber_rent_cd, wide by year (countrycode + YR<year>).
        countries_df (pd.DataFrame): the r250 country rows, carrying iso3_r250_label.
        year (int): the base year.

    Returns:
        pd.DataFrame: countries_df with timber_provision_gep_cwon_rent added. A country CWoN does
        not value stays NaN rather than zero, because no rent published is not a rent of nothing.

    Raises:
        ValueError: if the file does not carry the expected series, since a silently different
            series would publish some other quantity under this name.
    """
    import pandas as pd
    if 'series' in rent_df.columns:
        series = set(str(v) for v in rent_df['series'].dropna().unique())
        if CWON_FOREST_RENT_SERIES not in series:
            raise ValueError('%s does not carry %r; it carries %r'
                             % ('the CWoN rent table', CWON_FOREST_RENT_SERIES, sorted(series)[:3]))
        rent_df = rent_df[rent_df['series'] == CWON_FOREST_RENT_SERIES]
    column = 'YR%d' % int(year)
    rent = rent_df[['countrycode', column]].copy()
    rent['timber_provision_gep_cwon_rent'] = pd.to_numeric(rent[column], errors='coerce')
    return countries_df.merge(
        rent[['countrycode', 'timber_provision_gep_cwon_rent']],
        left_on='iso3_r250_label', right_on='countrycode', how='left').drop(columns=['countrycode'])


def roundwood_gross_value_by_country(fao_df, countries_df, year):
    """FAOSTAT roundwood production priced at each country's own export unit value.

    Not a third estimate of the service: a BOUND. A land factor share, however derived, is a
    fraction of the gross value of the wood it comes from, so a country whose timber GEP exceeds
    the gross value of all the roundwood it produced is saying something impossible. Publishing
    the bound beside the two valuations lets that be seen per country rather than argued globally.

    The price is the country's own industrial-roundwood export unit value where FAOSTAT reports
    one, and the world export unit value otherwise. A country's own price matters: the world value
    is about $111/m3 and tropical hardwood exporters are several times that, so a single global
    price manufactures exceedances that are only a price error.

    Args:
        fao_df (pd.DataFrame): the staged FAOSTAT forestry slice, normalized long.
        countries_df (pd.DataFrame): the r250 country rows, carrying iso3_r250_id (the M49 code).
        year (int): the base year.

    Returns:
        pd.DataFrame: countries_df with roundwood_m3 and timber_roundwood_gross_value added.
    """
    import numpy as np
    import pandas as pd
    d = fao_df[fao_df['Year'] == int(year)].copy()
    d['iso3_r250_id'] = (d['Area Code (M49)'].astype(str)
                         .str.replace("'", '', regex=False).astype(int))
    world = d[(d['Area'] == 'World') & (d['Item'] == 'Industrial roundwood')]
    world_price = (float(world[world['Element'] == 'Export value']['Value'].iloc[0]) * 1000.0
                   / float(world[world['Element'] == 'Export quantity']['Value'].iloc[0]))

    countries = d[d['Area'] != 'World']
    industrial = countries[countries['Item'] == 'Industrial roundwood']
    quantity = industrial[industrial['Element'] == 'Export quantity'].groupby('iso3_r250_id')['Value'].sum()
    value = industrial[industrial['Element'] == 'Export value'].groupby('iso3_r250_id')['Value'].sum() * 1000.0
    own = (value / quantity).replace([np.inf, -np.inf], np.nan).dropna()
    # A unit value outside this band is a reporting artifact rather than a price, and using it
    # would move the bound further than the thing it is meant to catch.
    own = own[(own > 5) & (own < 3000)]

    produced = (countries[(countries['Element'] == 'Production') & (countries['Item'] == 'Roundwood')]
                .groupby('iso3_r250_id')['Value'].sum().rename('roundwood_m3'))
    out = pd.DataFrame(produced).join(own.rename('price'))
    out['price'] = out['price'].fillna(world_price)
    out['timber_roundwood_gross_value'] = out['roundwood_m3'] * out['price']
    return countries_df.merge(out[['roundwood_m3', 'timber_roundwood_gross_value']].reset_index(),
                              on='iso3_r250_id', how='left')

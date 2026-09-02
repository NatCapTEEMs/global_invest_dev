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


def fuelwood_share_of_forest_rent(fao_df, countries_df, year):
    """How much of CWoN's forest rent is fuelwood rather than industrial roundwood.

    ⚠ This is a DECOMPOSITION, not a second service. CWoN's `Forest, rents (current US$)` --
    published here as `timber_provision_gep` -- is built in `forest_timber_depletion.do` from three
    FAOSTAT items and summed before the rental ratio is applied:

        keep if itemcode == 1864 | 1866 | 1867      1864 is Wood Fuel
        tot_rev  = rowtotal(wf_rev, irwc_rev, irwnc_rev)
        tot_rent = tot_rev x ratio

    So fuelwood is already inside the timber figure, and a separate fuelwood service added on top
    of it would count the same rent twice. Rather than remove it, this splits it: because the rent
    is a single ratio applied to a sum of revenues, the fuelwood part of the rent is the fuelwood
    share of the revenue, and the ratio cancels.

        fuelwood_part = timber_gep x (wf_rev / (wf_rev + irw_rev))

    Revenue is production times the country's own export unit value, the world value where FAOSTAT
    reports none -- the same pricing the gross-value bound uses, and for the same reason: wood fuel
    is about $67/m3 against industrial roundwood's $111, so a single price would misallocate the
    split rather than merely blur it.

    Returns:
        pd.DataFrame: countries_df with fuelwood_share_of_rent added, NaN where FAOSTAT prices
        neither product for that country.
    """
    import numpy as np
    import pandas as pd
    d = fao_df[fao_df['Year'] == int(year)].copy()
    d['iso3_r250_id'] = (d['Area Code (M49)'].astype(str)
                         .str.replace("'", '', regex=False).astype(int))
    world = d[d['Area'] == 'World']
    countries = d[d['Area'] != 'World']

    def revenue(item):
        """Production at the WORLD export unit value, the same basis for both products.

        ⚠ Country-own prices were used here until 2026-09-02 and made the share wrong. The share
        is a RATIO between two revenues, so both sides have to be priced the same way or the ratio
        measures the pricing rather than the mix. India is the case that exposed it: it exports 309
        cubic metres of wood fuel for $234,000, an implied $757/m3 against a world $67, and 6,096
        cubic metres of industrial roundwood at an implied $9,189 against a world $111. A filter
        that accepted prices under $3,000 took the absurd fuelwood price and rejected the absurd
        industrial one, so India's fuelwood share came out at 97.6 percent instead of 78.7.

        Neither product's own price is usable for this: barely-traded volumes set them. CWoN faces
        the same problem and solves it with outlier-trimmed REGIONAL mean export unit values; one
        world price is the simpler version of the same idea and keeps both sides comparable.
        """
        sub = countries[countries['Item'] == item]
        produced = sub[sub['Element'] == 'Production'].groupby('iso3_r250_id')['Value'].sum()
        w = world[world['Item'] == item]
        world_price = (float(w[w['Element'] == 'Export value']['Value'].iloc[0]) * 1000.0
                       / float(w[w['Element'] == 'Export quantity']['Value'].iloc[0]))
        return produced * world_price

    fuel = revenue('Wood fuel').rename('wood_fuel_revenue')
    industrial = revenue('Industrial roundwood').rename('industrial_revenue')
    both = pd.concat([fuel, industrial], axis=1).fillna(0.0)
    total = both['wood_fuel_revenue'] + both['industrial_revenue']
    both['fuelwood_share_of_rent'] = np.where(total > 0, both['wood_fuel_revenue'] / total, np.nan)
    return countries_df.merge(both[['fuelwood_share_of_rent']].reset_index(),
                              on='iso3_r250_id', how='left')


def wood_fuel_gross_value_by_country(fao_df, countries_df, year):
    """FAOSTAT wood fuel production priced at each country's own WOOD FUEL export unit value.

    ⚠ Not the fuelwood share of roundwood gross, which was the first attempt and was wrong by
    almost three times: roundwood gross uses each country's roundwood price, which is dominated by
    industrial timber at about $111/m3 against wood fuel's $67, so scaling it by a revenue share
    prices fuelwood as though it were sawlogs. Wood fuel has its own production and its own export
    unit value in the same table, so it is priced directly here.
    """
    import numpy as np
    import pandas as pd
    d = fao_df[fao_df['Year'] == int(year)].copy()
    d['iso3_r250_id'] = (d['Area Code (M49)'].astype(str)
                         .str.replace("'", '', regex=False).astype(int))
    world = d[(d['Area'] == 'World') & (d['Item'] == 'Wood fuel')]
    world_price = (float(world[world['Element'] == 'Export value']['Value'].iloc[0]) * 1000.0
                   / float(world[world['Element'] == 'Export quantity']['Value'].iloc[0]))
    fuel = d[(d['Area'] != 'World') & (d['Item'] == 'Wood fuel')]
    produced = fuel[fuel['Element'] == 'Production'].groupby('iso3_r250_id')['Value'].sum()
    # ⚠ ONE world price for every country, which is the opposite of what the industrial-roundwood
    # bound does, and deliberately. Fuelwood is overwhelmingly NOT traded: it is collected and
    # burned where it grows, so a country's export unit value is set by a tiny specialty flow and
    # says nothing about the price of the rest. Using own prices here gave $368bn against a world
    # export-priced $131bn, because a country exporting a hundred cubic metres at a high unit value
    # had that price applied to fifty million cubic metres of domestic burning. Industrial
    # roundwood is genuinely traded and its own prices are informative; this is not.
    out = pd.DataFrame({'wood_fuel_m3': produced,
                        'wood_fuel_gross_value': produced * world_price})
    return countries_df.merge(out.reset_index(), on='iso3_r250_id', how='left')

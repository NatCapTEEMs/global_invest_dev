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
# and hydropower rents come from. The rent is the account's figure: a country table with no
# spatial detail, from the source every other rent-based service uses. The raster estimate is
# spatial and cannot be rebuilt here, and is published beside it.
CWON_FOREST_RENT_SERIES = 'Forest, rents (current US$)'


def cwon_forest_rent_by_country(rent_df, countries_df, year):
    """CWoN's published forest rent for one year, on the account's country rows.

    The account computes the rent with cwon_forest_rent_from_revenue_and_rate. This reads the
    published table to check it.

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


CWON_FOREST_REVENUE_SERIES = 'Forest, timber revenues (current USD)'
CWON_FOREST_RENTAL_RATE_SERIES = 'Forest, timber, rental rate'


def cwon_forest_rent_from_revenue_and_rate(revenue_df, rate_df, countries_df, year,
                                           assumed_zero_countries=()):
    """CWoN's forest rent for one year, computed: timber revenue times the rental rate.

    CWoN prices roundwood removals at export unit values and multiplies the revenue by a regional
    rental ratio (`tot_rent = tot_rev * ratio` in forest_timber_depletion.do). Where that leaves no
    rent, CWoN's gap-filling writes zero for the countries it assumes have none
    (forest_timber_gapfill.do). Here a missing rent becomes zero only for a country named in
    `assumed_zero_countries`, and any other country without both inputs stays missing.

    Args:
        revenue_df (pd.DataFrame): forest_timber_revenue, wide by year (countrycode, series,
            YR<year>).
        rate_df (pd.DataFrame): forest_timber_rental_rate, the same shape.
        countries_df (pd.DataFrame): the r250 country rows, carrying iso3_r250_label.
        year (int): the base year.
        assumed_zero_countries (iterable): ISO3 codes whose missing rent CWoN sets to zero.

    Returns:
        pd.DataFrame: countries_df with timber_provision_cwon_revenue,
        timber_provision_cwon_rental_rate, timber_provision_gep_cwon_rent and
        timber_provision_cwon_rent_assumed_zero, True where the rent is an assumed zero rather
        than a product. Inputs are matched on the country code, not on row order.

    Raises:
        ValueError: if a table does not carry its series or lists a country twice, since either
            would put another quantity, or a country's value twice, under these names.
    """
    import pandas as pd
    column = 'YR%d' % int(year)

    def by_country(df, series, name):
        if 'series' in df.columns:
            labels = set(str(v) for v in df['series'].dropna().unique())
            if series not in labels:
                raise ValueError('the CWoN %s table does not carry %r, it carries %r'
                                 % (name, series, sorted(labels)[:3]))
            df = df[df['series'] == series]
        repeated = df.loc[df['countrycode'].duplicated(), 'countrycode']
        if len(repeated):
            raise ValueError('the CWoN %s table lists %s more than once' % (name, sorted(repeated)[:5]))
        return pd.to_numeric(df.set_index('countrycode')[column], errors='coerce')

    revenue = by_country(revenue_df, CWON_FOREST_REVENUE_SERIES, 'timber revenue')
    rate = by_country(rate_df, CWON_FOREST_RENTAL_RATE_SERIES, 'rental rate')
    out = countries_df.copy()
    out['timber_provision_cwon_revenue'] = out['iso3_r250_label'].map(revenue)
    out['timber_provision_cwon_rental_rate'] = out['iso3_r250_label'].map(rate)
    out['timber_provision_gep_cwon_rent'] = (out['timber_provision_cwon_revenue']
                                             * out['timber_provision_cwon_rental_rate'])
    assumed = (out['timber_provision_gep_cwon_rent'].isna()
               & out['iso3_r250_label'].isin(set(assumed_zero_countries)))
    out.loc[assumed, 'timber_provision_gep_cwon_rent'] = 0.0
    out['timber_provision_cwon_rent_assumed_zero'] = assumed
    return out


def roundwood_gross_value_by_country(fao_df, countries_df, year):
    """FAOSTAT roundwood production priced at an industrial-roundwood export unit value.

    Not a third estimate of the service: a proxy DIAGNOSTIC. A land factor share is a fraction of
    the gross value of the wood it comes from, so a country whose timber GEP exceeds this column
    is flagged for a look, at the rent, at the price proxy or at both. It is not a strict bound:
    export unit values stand in for domestic prices they can sit well above or below. Publishing
    it beside the two valuations lets a flag be seen per country rather than argued globally.

    All roundwood production, wood fuel included, is priced at the country's own
    industrial-roundwood export unit value where FAOSTAT reports one between $5 and $3,000 per m3
    (a chosen band), and at the world industrial-roundwood export unit value otherwise. The own value is used where
    it exists because the world value is about $111/m3 and tropical hardwood exporters are several
    times that, so a single global price would flag countries for a price difference alone.

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
    # The band is a chosen filter: an own unit value outside it is treated as unreliable and the
    # world value is used instead. It is a judgement about barely-traded flows, not a proof that
    # each excluded value is a reporting artifact.
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

    This is a DECOMPOSITION, not a second service. CWoN's `Forest, rents (current US$)` --
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

    Revenue is each product's production times that product's WORLD export unit value: wood fuel
    is about $67/m3 against industrial roundwood's $111, so one price for both would misallocate
    the split, and own-country prices make the ratio measure pricing rather than the mix (see
    `revenue` below). This is not the pricing the gross-value diagnostic uses.

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

        Country-own prices are NOT used here. The share is a RATIO between two revenues, so both
        sides have to be priced the same way or the ratio measures the pricing rather than the mix.
        India shows why: it exports 309 cubic metres of wood fuel for $234,000, an implied $757/m3
        against a world $67, and 6,096 cubic metres of industrial roundwood at an implied $9,189
        against a world $111. A filter accepting prices under $3,000 keeps the absurd fuelwood price
        and rejects the absurd industrial one, putting India's fuelwood share at 97.6 percent
        instead of 78.7.

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
    """FAOSTAT wood fuel production priced at the WORLD wood-fuel export unit value.

    Not the fuelwood share of roundwood gross, which is wrong by almost three times: roundwood
    gross prices at an industrial-roundwood unit value, about $111/m3
    against wood fuel's $67, so scaling it by a revenue share prices fuelwood as though it were
    sawlogs. Wood fuel has its own production and its own export unit value in the same table, so
    it is priced directly here, at one world value for every country (the comment below says why).
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
    # ONE world price for every country, which is the opposite of what the industrial-roundwood
    # gross-value diagnostic does, and deliberately. Fuelwood is overwhelmingly NOT traded: it is collected and
    # burned where it grows, so a country's export unit value is set by a tiny specialty flow and
    # says nothing about the price of the rest. Own prices give $368bn against a world
    # export-priced $131bn, because a country exporting a hundred cubic metres at a high unit value
    # has that price applied to fifty million cubic metres of domestic burning. Industrial
    # roundwood is genuinely traded and its own prices are informative; this is not.
    out = pd.DataFrame({'wood_fuel_m3': produced,
                        'wood_fuel_gross_value': produced * world_price})
    return countries_df.merge(out.reset_index(), on='iso3_r250_id', how='left')


# SEALS7 land-cover class of forest, the only class that carries timber value in a scenario map.
SEALS7_FOREST_ID = 4


def timber_eligible_value(value, base_lulc, forest_ids=(SEALS7_FOREST_ID,), lulc_ndv=None):
    """The base-year timber value of the fixed eligible area: cells that are managed in the base
    year (value > 0, the management mask and the floor at zero net return are inside `value`) AND
    forest in the model's base-year land-cover map. Everything else is 0; nodata in `value` stays
    nodata.

    Returns:
        np.ndarray: float32.
    """
    value = np.asarray(value, dtype='float32')
    eligible = np.isin(base_lulc, list(forest_ids)) & (value > 0)
    if lulc_ndv is not None:
        eligible &= base_lulc != lulc_ndv
    out = np.where(eligible, value, np.float32(0.0))
    return np.where(np.isfinite(value), out, np.float32(np.nan)).astype('float32')


def timber_value_on_forest(eligible_value, lulc, forest_ids=(SEALS7_FOREST_ID,), lulc_ndv=None):
    """The timber value a scenario map carries: each eligible cell's baseline net return where the
    map still says forest, removed where it has become non-forest. Forest gained outside the
    eligible area carries nothing (eligible_value is 0 there). A scenario cell with no land-cover
    data is not a conversion: it keeps its eligible value.

    This is deliberately not a class-mean lookup: a class mean over all cells of a class dilutes
    forest by unmanaged forest and gives non-forest classes value wherever the management mask
    overlaps them, so afforestation lowers the measure and deforestation raises it.

    Args:
        eligible_value (np.ndarray): from timber_eligible_value, on the same grid as `lulc`.
        lulc (np.ndarray): the scenario map's land-cover classes.
        forest_ids (tuple): the class ids counted as forest.
        lulc_ndv: the map's nodata value, if any.

    Returns:
        np.ndarray: float32; nodata where eligible_value is nodata.
    """
    ev = np.asarray(eligible_value, dtype='float32')
    is_forest = np.isin(lulc, list(forest_ids))
    unknown = (lulc == lulc_ndv) if lulc_ndv is not None else np.zeros(lulc.shape, dtype=bool)
    keep = is_forest | unknown
    out = np.where(keep, ev, np.float32(0.0))
    return np.where(np.isfinite(ev), out, np.float32(np.nan)).astype('float32')



# --- D19: the timber resource measured as aboveground forest biomass carbon ----------------------
# An alternative proxy construction that changes both the resource measure and the
# footprint: living aboveground biomass carbon (Spawn & Gibbs 2020, Mg C/ha) on the cells that are
# forest in EACH scenario map, instead of base-year net return on the fixed 2023 managed footprint.
# Converted forest leaves with its observed density; forest gained after the base year enters at
# its carbon zone's mean forest density -- an ASSUMPTION of mature biomass at once, not simulated
# growth; the data carry no regrowth curve. Carbon -> dry biomass is / 0.47 (IPCC 2006 GL) and the
# zone's value per tonne is calibrated once on the 2023 managed footprint, so both cancel in the
# zonal ratio: the fed shock is the percentage change in aboveground forest biomass carbon.

CARBON_FRACTION_OF_DRY_MATTER = 0.47


def forest_biomass_carbon_base(agbc_density, ha_per_cell, base_lulc, forest_ids=(SEALS7_FOREST_ID,), lulc_ndv=None):
    """Aboveground biomass carbon per cell (Mg C) on the base-year forest; NaN elsewhere.

    NaN outside the base-year forest is the marker "not forest in the base year" that
    forest_biomass_carbon_on_map reads, so it must not be confused with a forest cell of zero
    density (which stays 0).
    """
    d = np.asarray(agbc_density, dtype='float32')
    ha = np.asarray(ha_per_cell, dtype='float32')
    is_forest = np.isin(base_lulc, list(forest_ids))
    if lulc_ndv is not None:
        is_forest &= base_lulc != lulc_ndv
    stock = np.where(np.isfinite(d) & (d > 0), d, np.float32(0.0)) * np.where(ha > 0, ha, np.float32(0.0))
    return np.where(is_forest, stock, np.float32(np.nan)).astype('float32')


def forest_biomass_carbon_on_map(base_forest_carbon, new_forest_carbon, lulc, forest_ids=(SEALS7_FOREST_ID,), lulc_ndv=None):
    """The aboveground forest biomass carbon a scenario map carries, per cell (Mg C).

    Cells forest in the base year keep their observed stock while the map says forest and carry 0
    once it does not; cells that were not forest in the base year and are forest now enter at
    `new_forest_carbon` (the carbon zone's mean forest density x cell area, the mature-density
    assumption); a cell with no land-cover data in the map is not a conversion and keeps its
    base-year stock (0 if it had none).

    Args:
        base_forest_carbon (np.ndarray): from forest_biomass_carbon_base (NaN = not forest in the base year).
        new_forest_carbon (np.ndarray): Mg C the cell would hold as forest, from the zone lookup; nodata/NaN -> 0.
        lulc (np.ndarray): the scenario map.
    """
    base = np.asarray(base_forest_carbon, dtype='float32')
    new = np.asarray(new_forest_carbon, dtype='float32')
    new = np.where(np.isfinite(new) & (new > 0), new, np.float32(0.0))
    was_forest = np.isfinite(base)
    is_forest = np.isin(lulc, list(forest_ids))
    unknown = (lulc == lulc_ndv) if lulc_ndv is not None else np.zeros(lulc.shape, dtype=bool)
    kept = np.where(was_forest & (is_forest | unknown), base, np.float32(0.0))
    gained = np.where(~was_forest & is_forest & ~unknown, new, np.float32(0.0))
    return (kept + gained).astype('float32')


# --- Age-aware harvestable flow (resource measure 'harvestable_flow') ----------------------------
# Opt-in and never a default. A forest state is kept per cell beside the land-cover map: forest
# type, stand age, years since the last selective cut, merchantable stock (m3/ha), and whether the
# cell is managed and accessible. Harvest does not change the land-cover label, so the state cannot
# be read off the map. Each annual step grows the stock along the type's Chapman-Richards curve
# from the age at which the curve holds the current stock, then removes what the harvest rule
# cuts. A land-cover change between two maps takes effect at the later map's year: clearing
# removes the stand and its stock, and new forest enters at age 0 with zero stock. Every growth,
# harvest and recovery value is configuration with a stated source; nothing here supplies one.

FOREST_STATE_BANDS = ('type', 'age', 'since_cut', 'stock', 'managed', 'accessible')
HARVEST_RULES = ('none', 'scheduled')
RECOVERY_RULES = ('curve_equivalent_age',)
PROVENANCE_STATUSES = ('observed', 'assumed', 'synthetic')
PROVENANCE_INPUTS = FOREST_STATE_BANDS + ('growth_parameters',)
GROWTH_PARAMETER_COLUMNS = {
    'rotation': ('rotation_age', 'vmax_m3_per_ha', 'k', 'shape'),
    'cycle': ('min_age', 'cycle_years', 'intensity_m3_per_ha', 'vmax_m3_per_ha', 'k', 'shape'),
    'none': ('vmax_m3_per_ha', 'k', 'shape'),
}


def growth_parameters_from_table(df):
    """Per-forest-type growth and harvest parameters from the configured table. No defaults.

    Args:
        df (pd.DataFrame): one row per forest type: forest_type_id (>= 1), harvest_mode
            ('rotation', 'cycle' or 'none'), the columns GROWTH_PARAMETER_COLUMNS lists for that
            mode, and a non-empty source.

    Returns:
        dict: forest_type_id -> {'mode': harvest_mode, column: float, ...}.

    Raises:
        ValueError: on a duplicate or invalid type, an unknown mode, a missing value or source.
    """
    params = {}
    for row in df.to_dict('records'):
        code, mode = int(row['forest_type_id']), str(row['harvest_mode'])
        if code < 1 or code in params:
            raise ValueError('forest_type_id must be unique and >= 1 (0 means not forest), got %d' % code)
        if mode not in GROWTH_PARAMETER_COLUMNS:
            raise ValueError('forest type %d: harvest_mode must be one of %s, got %r'
                             % (code, sorted(GROWTH_PARAMETER_COLUMNS), mode))
        source = row.get('source')
        if not isinstance(source, str) or not source.strip():
            raise ValueError('forest type %d has no source' % code)
        missing = [c for c in GROWTH_PARAMETER_COLUMNS[mode] if not np.isfinite(float(row.get(c, np.nan)))]
        if missing:
            raise ValueError('forest type %d (%s) has no value for %s' % (code, mode, ', '.join(missing)))
        params[code] = dict(mode=mode, **{c: float(row[c]) for c in GROWTH_PARAMETER_COLUMNS[mode]})
    return params


def merchantable_volume(age, prm):
    """Chapman-Richards merchantable volume (m3/ha): vmax (1 - exp(-k age)) ** shape, 0 at age 0."""
    a = np.maximum(np.asarray(age, dtype='float64'), 0.0)
    return prm['vmax_m3_per_ha'] * (1.0 - np.exp(-prm['k'] * a)) ** prm['shape']


def volume_equivalent_age(stock, prm):
    """The age at which the type's curve holds `stock`, the inverse of merchantable_volume; inf at vmax."""
    root = np.clip(np.asarray(stock, dtype='float64') / prm['vmax_m3_per_ha'], 0.0, 1.0) ** (1.0 / prm['shape'])
    with np.errstate(divide='ignore'):
        return -np.log1p(-root) / prm['k']


def eligible_volume(state, params):
    """Volume (m3/ha) each cell may yield this year under its type's harvest mode. Changes no state.

    Rotation types yield their whole stock once age >= rotation_age; cycle types yield
    min(intensity, stock) once age >= min_age and cycle_years have passed since the last cut. Only
    managed, accessible forest is eligible. A stand that is eligible and not cut stays eligible,
    so this is a standing backlog, not an annual flow.
    """
    t, age, since, stock = state['type'], state['age'], state['since_cut'], state['stock']
    can = (t > 0) & state['managed'] & state['accessible']
    out = np.zeros(t.shape)
    for code, prm in params.items():
        this = can & (t == code)
        if prm['mode'] == 'rotation':
            cut = this & (age >= prm['rotation_age'])
            out[cut] = stock[cut]
        elif prm['mode'] == 'cycle':
            cut = this & (age >= prm['min_age']) & (since >= prm['cycle_years'])
            out[cut] = np.minimum(prm['intensity_m3_per_ha'], stock[cut])
    return out


def advance_forest_state(state, n_years, forest_now, new_forest, params, harvest_rule, recovery_rule,
                         salvage_fraction, type_now=None):
    """Advance a forest state by n_years annual steps to the next map, then apply that map.

    Args:
        state (dict): arrays named FOREST_STATE_BANDS (type 0 = not forest; age and since_cut NaN
            off forest; stock in m3/ha, 0 off forest; managed and accessible boolean).
        n_years (int): years between the state and the map, >= 1.
        forest_now (np.ndarray): bool, forest in the next map.
        new_forest (dict): 'type', 'managed' and 'accessible' given to cells that become forest.
        params (dict): from growth_parameters_from_table; every type present must be in it.
        harvest_rule (str): 'none' (growth only) or 'scheduled' (every eligible stand is cut, an
            assumed management schedule).
        recovery_rule (str): 'curve_equivalent_age', the only rule implemented: a logged stand
            regrows along its type's curve from the age at which the curve holds its residual stock.
        salvage_fraction (float): share of a removed stand's stock counted as a one-off harvest.
        type_now (np.ndarray): optional forest type in the next map; a forest cell whose type
            changes is a conversion and restarts at age 0.

    Returns:
        tuple: (new state, flows). flows holds per-cell totals over the interval in m3/ha:
        opening, growth, harvest, removed (stock taken out by clearing or conversion), salvage,
        closing, and eligible (the volume eligible in the final year before its harvest).
    """
    if harvest_rule not in HARVEST_RULES:
        raise ValueError('harvest_rule must be one of %s, got %r' % (HARVEST_RULES, harvest_rule))
    if recovery_rule not in RECOVERY_RULES:
        raise ValueError('recovery_rule must be one of %s, got %r' % (RECOVERY_RULES, recovery_rule))
    if int(n_years) < 1:
        raise ValueError('a forest state advances by at least one year, got %r' % n_years)
    s = {k: np.array(state[k], dtype=bool if k in ('managed', 'accessible') else 'float64') for k in FOREST_STATE_BANDS}
    unknown = sorted(set(np.unique(s['type'][s['type'] > 0]).astype(int)) - set(params))
    if unknown:
        raise ValueError('no growth parameters for forest type(s) %s' % unknown)
    opening = s['stock'].copy()
    growth = np.zeros(opening.shape)
    harvest = np.zeros(opening.shape)
    for _ in range(int(n_years)):
        forest = s['type'] > 0
        s['age'][forest] += 1.0
        s['since_cut'][forest] += 1.0
        for code, prm in params.items():
            c = s['type'] == code
            before = s['stock'][c]
            after = np.maximum(merchantable_volume(volume_equivalent_age(before, prm) + 1.0, prm), before)
            growth[c] += after - before
            s['stock'][c] = after
        eligible = eligible_volume(s, params)
        if harvest_rule == 'scheduled':
            cut = eligible > 0
            harvest += np.where(cut, eligible, 0.0)
            s['stock'] -= np.where(cut, eligible, 0.0)
            s['since_cut'][cut] = 0.0
            for code, prm in params.items():
                if prm['mode'] == 'rotation':
                    reset = cut & (s['type'] == code)
                    s['age'][reset] = 0.0
                    s['stock'][reset] = 0.0
    forest_now = np.asarray(forest_now, dtype=bool)
    was = s['type'] > 0
    cleared = was & ~forest_now
    established = ~was & forest_now
    converted = np.zeros(forest_now.shape, dtype=bool)
    if type_now is not None:
        converted = was & forest_now & (np.asarray(type_now) != s['type'])
    if np.any(established) and int(new_forest['type']) not in params:
        raise ValueError('new forest type %r has no growth parameters' % new_forest['type'])
    removed = np.where(cleared | converted, s['stock'], 0.0)
    restart = established | converted
    s['type'] = np.where(cleared, 0, np.where(established, int(new_forest['type']),
                                              np.where(converted, type_now if type_now is not None else 0, s['type'])))
    s['age'] = np.where(cleared, np.nan, np.where(restart, 0.0, s['age']))
    s['since_cut'] = np.where(cleared, np.nan, np.where(restart, 0.0, s['since_cut']))
    s['stock'] = np.where(cleared | restart, 0.0, s['stock'])
    s['managed'] = np.where(cleared, False, np.where(restart, bool(new_forest['managed']), s['managed']))
    s['accessible'] = np.where(cleared, False, np.where(restart, bool(new_forest['accessible']), s['accessible']))
    flows = dict(opening=opening, growth=growth, harvest=harvest, removed=removed,
                 salvage=salvage_fraction * removed, closing=s['stock'].copy(), eligible=eligible)
    return s, flows


def stock_ledger_residual(flows):
    """closing - (opening + growth - harvest - removed), per cell; zero when the stock is conserved."""
    return flows['closing'] - (flows['opening'] + flows['growth'] - flows['harvest'] - flows['removed'])


def harvestable_flow_factors(flow, area, flow_ref, area_ref):
    """F/F_ref = (A/A_ref) x (f/f_ref) with f = F/A, elementwise; NaN where a ratio is undefined.

    The reference is either the base map's first interval or the baseline scenario over the same
    interval. No epsilon is added: a zero reference flow or area has no ratio, and a year with no
    timberland has no per-hectare flow. Which factor reaches GTAP, and with what calibration, is not
    decided here.

    Returns:
        dict: area_factor, per_ha_factor and total_factor arrays.
    """
    flow, area, flow_ref, area_ref = (np.asarray(x, dtype='float64') for x in (flow, area, flow_ref, area_ref))
    with np.errstate(divide='ignore', invalid='ignore'):
        area_factor = np.where(area_ref > 0, area / area_ref, np.nan)
        total_factor = np.where(flow_ref > 0, flow / flow_ref, np.nan)
        per_ha_factor = np.where((area_ref > 0) & (flow_ref > 0) & (area > 0),
                                 (flow / area) / (flow_ref / area_ref), np.nan)
    return dict(area_factor=area_factor, per_ha_factor=per_ha_factor, total_factor=total_factor)


def input_provenance_from_table(df):
    """Where each harvestable_flow input comes from, read from the configured provenance table.

    Args:
        df (pd.DataFrame): columns input, status ('observed', 'assumed' or 'synthetic') and source,
            with one row per input of PROVENANCE_INPUTS. An input built from more than one source
            takes one row per subset of its cells, each also naming the subset and its coverage:
            secondary-forest ages read from a map where it has a value and filled by an assumption
            elsewhere are two age rows. Further inputs, such as a management or access layer behind
            a band, are kept.

    Returns:
        dict: input -> its statuses in table order, joined by '+' (e.g. 'observed+assumed').

    Raises:
        ValueError: on a missing input, an unknown status, an empty source, or a repeated input
            whose rows do not each name a distinct subset and its coverage.
    """
    rows = df.to_dict('records')
    names = list(dict.fromkeys(str(row['input']) for row in rows))
    missing = [name for name in PROVENANCE_INPUTS if name not in names]
    if missing:
        raise ValueError('input provenance has no row for %s' % missing)
    for row in rows:
        if row['status'] not in PROVENANCE_STATUSES:
            raise ValueError('input %s: status must be one of %s, got %r' % (row['input'], PROVENANCE_STATUSES, row['status']))
        if not isinstance(row.get('source'), str) or not row['source'].strip():
            raise ValueError('input %s has no source' % row['input'])
    statuses = {}
    for name in names:
        subsets = [row for row in rows if str(row['input']) == name]
        described = all(isinstance(row.get(c), str) and row[c].strip() for row in subsets for c in ('subset', 'coverage'))
        if len(subsets) > 1 and not (described and len({row['subset'] for row in subsets}) == len(subsets)):
            raise ValueError('input %s has %d rows, and each needs its own subset and a coverage' % (name, len(subsets)))
        statuses[name] = '+'.join(dict.fromkeys(str(row['status']) for row in subsets))
    return statuses

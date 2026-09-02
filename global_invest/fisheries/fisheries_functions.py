"""Fisheries ES science: read the pre-computed marine-fisheries shock headers.

Marine-fisheries productivity is mapped by RCP (DBEM vs BOATS provenance is a paper question, see task
#16) and is never derived from the terrestrial SEALS maps, so unlike carbon and pollination it is READ
from a pre-computed HAR (cwon_shocks.har, headers FI26/FI45/FI85, one per RCP) rather than recomputed.

The headers carry a FULL ANNUAL series (Y2017..Y2050, 50 regions x 34 years). For the current file the
series is a 2017->2018 step that then holds flat, so it is already constant over the 2023-2050 solve
window; the read returns the whole annual series regardless, so the task reads each year directly and a
genuinely dynamic future source (DBEM/Fish-MIP, #45) needs no read change.
"""


def read_fisheries_headers(cwon_path, headers):
    """Per-region ANNUAL fisheries shock (%) per RCP header -> {header: {reg: {year_int: value}}}."""
    from gtappy.harpy.har_file import HarFileObj
    h = HarFileObj(filename=cwon_path)
    out = {}
    for hdr in headers:
        arr = h[hdr].array
        regs = [s.strip() for s in h[hdr].setElements[0]]
        years = [int(s.strip().lstrip('Yy')) for s in h[hdr].setElements[1]]
        out[hdr] = {reg: {yr: arr[i, j] for j, yr in enumerate(years)}
                    for i, reg in enumerate(regs)}
    return out


# NGFS scenario -> fisheries RCP header. RCP2.6=FI26 (below_2c/net_zero/low_demand),
# RCP4.5=FI45 (ndcs/delayed_transition), RCP7.0=FI85 (current_policies/fragmented_world/stress_test).
# RCP -> FI header. The headers ARE RCP-named (FI26=RCP2.6, FI45=RCP4.5, FI85=RCP8.5; FI85 also
# serves RCP7.0 as the closest available -- provenance #16). When the scenarios CSV carries a
# climate_label column (hydrate_es_scenarios publishes p.es_shock_climate_labels), the header is
# derived from the scenario's RCP and scenario NAMES need no translation at all.
RCP_FI_MAP = {'rcp26': 'FI26', 'rcp45': 'FI45', 'rcp60': 'FI85', 'rcp70': 'FI85', 'rcp85': 'FI85'}

FISH_HEADER_MAP = {
    'below_2c': 'FI26', 'net_zero': 'FI26', 'low_demand': 'FI26',
    'ndcs': 'FI45', 'delayed_transition': 'FI45',
    'current_policies': 'FI85', 'fragmented_world': 'FI85', 'stress_test': 'FI85',
}
FISH_CAP = 2.0          # +-2% backstop. Every legitimate FI value across FI26/FI45/FI85 is <=1.6%, so real
                        # signal passes untouched. Kept as a catch-all; known-bad values are now IMPUTED by
                        # FISH_VALUE_OVERRIDES below rather than merely clipped.

# (header, region) -> imputed value, for entries that are demonstrably corrupt at source.
#
# 'nor' FI26 = +13.504 while its FI45 = +0.565 and FI85 = +0.558. A 24x larger gain under the WEAKEST
# warming inverts the physics -- DBEM's high-latitude gains grow with warming, they do not peak at RCP2.6
# -- so the value is an error, not signal. Clipping it to the +-2 cap does not fix the problem: with only
# 50 regions the global mean is dominated by this one cell (RCP2.6 mean +0.3037, of which 'nor' alone
# contributes +0.270; capped it still supplies ~half the remaining mean), so the SIGN of the below_2c
# fisheries shock rested on a number we know is wrong.
#
# Imputed from Norway's OWN other-RCP values by OLS across the other 49 regions:
#   FI26 ~ FI45 : r=+0.813, slope +0.7411, intercept +0.0583 -> +0.4767   <- used (best correlated)
#   FI26 ~ FI85 : r=+0.482                                   -> +0.2916
#   median FI26/FI45 ratio (n=39, |FI45|>0.05) = +0.8902     -> +0.5026   (independent corroboration)
# Result: nor = +0.477 (2.6), +0.565 (4.5), +0.558 (8.5) -- gains rise then flatten with warming.
# ⚠ This is an IMPUTATION, not a correction at source. Flag it to Erwin with #16.
FISH_VALUE_OVERRIDES = {('FI26', 'nor'): 0.4767}


def resolve_fisheries_header(scen, header_map, climate_labels):
    """FI header for a scenario: explicit map (consumer) -> the scenario's RCP (scenarios CSV)
    -> identity (FI-native labels pass straight through)."""
    return header_map.get(scen) or RCP_FI_MAP.get(climate_labels.get(scen), scen)


def fisheries_headers_to_read(header_map):
    """The HAR headers to read: the union of the consumer map's targets and every RCP-derivable
    header. Must NOT depend on header_map alone -- with the legacy default deleted, an empty map
    would read no headers and every scenario would be dropped regardless of what the RCP
    derivation resolves (found by the ngfs session: 9/9 scenarios -> 0/9 in simulation)."""
    return tuple(sorted(set(header_map.values()) | set(RCP_FI_MAP.values())))


def static_shock_rows(fi_data, scenarios, header_map, climate_labels, overrides, sectors,
                      base_year, end_year, time_varying, constant_year,
                      ramp_to_end, ramp_end_year, log=print):
    """The per-region FSH shock rows, one per (sector, region, scenario, year), uncapped.

    Each scenario resolves to an FI header (a scenario whose header the data lacks contributes no
    rows), each (header, region) takes its imputation override if one exists, and the value is
    either ramped linearly from 0 at base_year to its full size at ramp_end_year or read from the
    year's own series entry. The caller applies the +-cap AFTER asserting the table sound, so a
    contaminated source value fails the assertion instead of being clamped into health.

    Args:
        fi_data (dict): {header: {region: {year: value}}} from read_fisheries_headers.
        scenarios (list): the scenario labels to shock.
        header_map (dict): scenario -> FI header; resolve_fisheries_header's first layer.
        climate_labels (dict): scenario -> rcp label; its second layer.
        overrides (dict): (header, region) -> imputed value for source-corrupt entries.
        sectors (tuple): the ACTS labels every row is repeated over.
        base_year (int): the ramp's zero point and the first year written.
        end_year (int): the last year written.
        time_varying (bool): with the ramp off, read each year's own series entry rather than
            holding constant_year's value.
        constant_year (int): the series entry used when a year is missing or time_varying is off.
        ramp_end_year (int): the horizon the full FI value belongs to; sets the ramp's slope.

    Returns:
        list: row dicts ready for the shock table.
    """
    n_years = max(ramp_end_year - base_year, 1)
    rows = []
    for scen in scenarios:
        hdr = resolve_fisheries_header(scen, header_map, climate_labels)
        if hdr not in fi_data:
            continue
        for reg, series in fi_data[hdr].items():
            const_val = series.get(constant_year)
            full_val = const_val if not time_varying else series.get(end_year, const_val)
            if (hdr, reg) in overrides:
                full_val = overrides[(hdr, reg)]
                log('  fisheries shock: OVERRIDE %s/%s -> %+.4f (corrupt at source, imputed)'
                    % (hdr, reg, full_val))
            for year in range(base_year, end_year + 1):
                if ramp_to_end:
                    # clipped at 1.0 so a run extending past ramp_end_year holds the full value
                    # rather than extrapolating the ramp beyond the horizon it was defined on
                    val = full_val * min((year - base_year) / n_years, 1.0)
                else:
                    val = series.get(year, const_val) if time_varying else const_val
                for sector in sectors:
                    rows.append({'ACTS': sector, 'REG': reg, 'scenario': scen,
                                 'year': year, 'shock_pct': val, 'fisheries_header': hdr})
    return rows


# =============================================================================
# GEP valuation (commercial capture fisheries, CWoN method). Ported from the
# source repo's 2026 script (gep_commcapturefisheries_cwonmethod_20260720.R):
# CWoN 2024 (FR_WLD_2024_195) fisheries economic rent, deflated to 2019 USD,
# per-country OLS trend over 2009-2018 predicting 2019, floored at zero.
# =============================================================================
FISHERIES_CPI_YEARS = (2009, 2019)          # CPI + rent window read from CWoN
FISHERIES_TREND_YEARS = (2009, 2018)        # the OLS window
FISHERIES_TREND_MIDPOINT = 2013.5           # mean of the OLS window years
FISHERIES_GEP_YEAR = 2019
# Country exclusions from the source script (CWoN data quality drops).
FISHERIES_RENT_EXCLUDED = ('CYP', 'MAF', 'PSE')
FISHERIES_RENT_EXCLUDED_COUNTRY_YEARS = (('CUW', 2009), ('CUW', 2010))


def clean_cwon_cpi(cpi_df):
    """CWoN cpi2019 table -> (wb_code, year, cpi2019) for the window, with a missing
    country-year CPI imputed by that YEAR's cross-country mean (the source's imputation)."""
    import pandas as pd
    df = cpi_df[(cpi_df['year'] >= FISHERIES_CPI_YEARS[0])
                & (cpi_df['year'] <= FISHERIES_CPI_YEARS[1])].copy()
    df['cpi2019'] = df['cpi2019'].fillna(df.groupby('year')['cpi2019'].transform('mean'))
    df = df.rename(columns={'countrycode': 'wb_code'})
    return df[['wb_code', 'year', 'cpi2019']]


def compute_econ_rent(rent_df):
    """Economic rent from its parts: landed value less the cost of catching it, less subsidy.

    CWoN's table carries a finished `FAOEconRent` column, and reading it would be shorter. It
    also carries every part that column is made of, and the parts rebuild it: on the 4,732-row
    table, rent equals profit minus subsidy on all 4,200 rows where both are present, and profit
    equals landed value minus total cost on all 4,648. Substituting variable plus fixed cost for
    their total-cost column moves the result by at most $67 on figures of order $10^9, which is
    float32 storage precision rather than a disagreement, and leaves the 2019 country totals at
    $29.0051bn either way. Computing it is therefore the same number and a step we own, so a
    reviewer asking where the rent comes from gets an equation rather than a column name.

    Subsidy is treated as absent-means-none, because a country-year with no recorded subsidy is
    one that received none, not one whose rent is unknown. Landed value and the costs are not
    filled: a missing catch or cost genuinely leaves the rent undefined.

    Args:
        rent_df (pd.DataFrame): CWoN EconRent_Analysis_AllYears, with FAO_landval2018,
            FAOVarCost, FAOFixCost and SubsidyUSD2018.

    Returns:
        pd.DataFrame: year, wb_code, econ_rent.
    """
    df = rent_df.copy()
    total_cost = df['FAOVarCost'] + df['FAOFixCost']
    df['econ_rent'] = df['FAO_landval2018'] - total_cost - df['SubsidyUSD2018'].fillna(0.0)
    return df


def clean_cwon_econ_rent(rent_df):
    """CWoN economic-rent table filtered to the window with the source's exclusions."""
    df = compute_econ_rent(rent_df).rename(columns={'Year': 'year'})
    df = df[(df['year'] >= FISHERIES_CPI_YEARS[0]) & (df['year'] <= FISHERIES_CPI_YEARS[1])]
    df = df[~df['wb_code'].isin(FISHERIES_RENT_EXCLUDED)]
    for wb_code, year in FISHERIES_RENT_EXCLUDED_COUNTRY_YEARS:
        df = df[~((df['wb_code'] == wb_code) & (df['year'] == year))]
    return df[['year', 'wb_code', 'econ_rent']].copy()


def deflate_rent_to_2019usd(rent_df, cpi_df):
    """Real 2019-USD economic rent: nominal rent / cpi2019 x 100."""
    df = rent_df.merge(cpi_df, on=['wb_code', 'year'], how='left')
    df['resrent_2019usd'] = df['econ_rent'] / df['cpi2019'] * 100
    return df


def fisheries_rent_trends(deflated_df):
    """Per-country 2019 rent estimate: mean real rent over the OLS window plus the OLS
    time-slope times the distance from the window midpoint to 2019, floored at zero.
    A country with fewer than two years of data gets no slope and no estimate (NaN),
    exactly as in the source."""
    import numpy as np
    import pandas as pd
    window = deflated_df[(deflated_df['year'] >= FISHERIES_TREND_YEARS[0])
                         & (deflated_df['year'] <= FISHERIES_TREND_YEARS[1])]
    rows = []
    for wb_code, group in window.groupby('wb_code'):
        valid = group[np.isfinite(group['resrent_2019usd'])]
        n_years = valid['year'].nunique()
        mean_rent = valid['resrent_2019usd'].mean() if len(valid) else np.nan
        beta = (np.polyfit(valid['year'].astype(float), valid['resrent_2019usd'], 1)[0]
                if n_years >= 2 else np.nan)
        rows.append({'wb_code': wb_code, 'n_years': n_years,
                     'mean_resrent_2009_2018': mean_rent, 'beta_hat': beta})
    trends = pd.DataFrame(rows)
    trends['resrent_2019_hat'] = (trends['mean_resrent_2009_2018']
                                  + trends['beta_hat'] * (FISHERIES_GEP_YEAR - FISHERIES_TREND_MIDPOINT))
    trends['positive_resrent_2019_hat'] = trends['resrent_2019_hat'].clip(lower=0)
    return trends


def commfish_gep_by_country(trends_df, countries_df):
    """Join the rent estimates onto the r250 country list by iso3 label (the CWoN wb_code IS
    the iso3_r250_label; the source joins the same way). Countries without an estimate stay
    NaN rather than zero -- no data is not zero rent."""
    trends = trends_df.rename(columns={'wb_code': 'iso3_r250_label'})
    df = countries_df.merge(trends[['iso3_r250_label', 'positive_resrent_2019_hat']],
                            on='iso3_r250_label', how='left')
    return df.rename(columns={'positive_resrent_2019_hat': 'commfish_provision'})


# =============================================================================
# Subsistence fisheries GEP (Lynch et al. 2024, USGS data release). Ported from
# the gep-subsistence-fisheries repo; the committed output CSV is the anchor.
# =============================================================================
# The Lynch et al. release carries the whole valuation, not only its result: a harvested
# quantity at the price's unit, a price per kilogram in USD, and their product per species.
# So the country value is computed here rather than adopted from the published TCUV column.
# Two reasons beyond principle: reading TCUV kept the FIRST value per admin, which is
# order-dependent where an admin carries more than one, and a published total cannot show us
# a discrepancy. Recomputing reproduces the release exactly (9,952,528,093.60 either way).
SUBSISTENCE_QUANTITY_COLUMN = 'total_biomass_harv_kg_unitofprice'
SUBSISTENCE_PRICE_COLUMN = 'final_kg_price_USD'
SUBSISTENCE_PUBLISHED_VALUE_COLUMN = 'TCUV'


def subsistence_value_by_admin(lynch_df):
    """Consumptive-use value per admin, summed over species from quantity times price.

    Args:
        lynch_df (pd.DataFrame): the Lynch et al. release, one row per admin and species.

    Returns:
        pd.DataFrame: admin, subsistence_fisheries_gep (ours), and
        subsistence_fisheries_gep_published (the release's own TCUV) beside it as the anchor.
    """
    import pandas as pd
    df = lynch_df.copy()
    for column in (SUBSISTENCE_QUANTITY_COLUMN, SUBSISTENCE_PRICE_COLUMN,
                   SUBSISTENCE_PUBLISHED_VALUE_COLUMN):
        df[column] = pd.to_numeric(df[column], errors='coerce')
    df['species_value'] = df[SUBSISTENCE_QUANTITY_COLUMN] * df[SUBSISTENCE_PRICE_COLUMN]
    # min_count=1 so an admin whose species all lack a quantity or a price stays empty rather
    # than becoming a zero. Summing to 0.0 there would assert no subsistence value where the
    # truth is that none was measured.
    grouped = df.groupby('admin', as_index=False).agg(
        subsistence_fisheries_gep=('species_value', lambda v: v.sum(min_count=1)),
        subsistence_fisheries_gep_published=(SUBSISTENCE_PUBLISHED_VALUE_COLUMN, 'first'))
    return grouped


def subsistence_fisheries_by_country(lynch_df, countries_df):
    """The computed per-admin value name-joined onto the canonical r250 country rows.

    The join is on brk_name, and a country absent from the release stays NaN: no data is not
    zero value.
    """
    data = subsistence_value_by_admin(lynch_df)
    df = countries_df.merge(data, how='left', left_on='brk_name', right_on='admin')
    return df.drop(columns=['admin'])


# =============================================================================
# Aquaculture. The third fisheries subgroup, beside commercial capture and
# subsistence.
#
# The source values capture and aquaculture as one figure:
#   GEP_fish := (total_aqua_value_usd1000 + total_cap_value_usd1000) * NatRes_Share
# (gep_fisheries_02_calc_gep_fish.R). Aquaculture is the first term times the
# same share, so nothing new is invented here -- one term of a sum is separated.
#
# ⚠ Aquaculture is farmed. What the share values is the natural-resource input to
# farmed production, not a wild stock, and whether that belongs in the account
# beside capture is a scope question for the paper rather than a computation.
# =============================================================================
AQUACULTURE_VALUE_MEASURE = 'V_USD_1000'   # FAO's own code: value in thousand USD
# FAO's Major_Group for seaweeds and other aquatic plants. Excluded by default, and the reason is
# the share rather than the reference: the multiplier is GTAP's natural-resource share of the
# FISHING sector, and GTAP does not put seaweed farming in fishing. Applying a fishing-sector
# share to seaweed value prices one sector's output with another's factor structure.
AQUATIC_PLANTS_MAJOR_GROUP = 'PLANTAE AQUATICAE'
GTAP_FISHING_SECTOR = 'fsh'
GTAP_NATURAL_RESOURCE_ENDOWMENT = 'NatRes'


def natural_resource_share_of_fishing(evfp_array, endowments, activities, regions,
                                      sector=GTAP_FISHING_SECTOR,
                                      endowment=GTAP_NATURAL_RESOURCE_ENDOWMENT):
    """The natural-resource share of fishing value added, one value per GTAP region.

    GTAP's EVFP is primary factor purchases at purchasers' prices, dimensioned endowment by
    activity by region, so the share is the natural-resource row over the column's total. This
    replaces the source's `fsh_endowment_gtap.xlsx`, which is the same quantity read from a
    workbook nobody sent us; computing it from the base data we already hold means the service
    does not rest on a file only one machine has.

    Args:
        evfp_array: the EVFP array, (endowment, activity, region).
        endowments, activities, regions (list[str]): its set element names, in order.

    Returns:
        pandas.DataFrame: gtap_region_label and natural_resource_share.

    Raises:
        ValueError: if a region's fishing sector has no factor payments at all, because a zero
            denominator would otherwise publish a silent zero share and value the whole sector
            at nothing.
    """
    import numpy as np
    import pandas as pd
    e = endowments.index(endowment)
    a = activities.index(sector)
    natural_resource = np.asarray(evfp_array)[e, a, :]
    total = np.asarray(evfp_array)[:, a, :].sum(axis=0)
    if (total <= 0).any():
        empty = [regions[i] for i, t in enumerate(total) if t <= 0]
        raise ValueError('GTAP regions with no %s factor payments, so the share is undefined '
                         'rather than zero: %s' % (sector, ', '.join(empty)))
    return pd.DataFrame({'gtap_region_label': regions,
                         'natural_resource_share': natural_resource / total})


def aquaculture_value_by_country(value_df, year, species_groups_df=None,
                                 exclude_aquatic_plants=True):
    """FAO FishStatJ aquaculture value for one year, summed over species, area and environment.

    The export is long: one row per country-species-area-environment-year, `VALUE` in thousand
    USD under the MEASURE code V_USD_1000. Returns whole dollars on the account's country key,
    since FAO's COUNTRY.UN_CODE is the M49 code that `iso3_r250_id` carries.
    """
    df = value_df[value_df['PERIOD'] == int(year)]
    if 'MEASURE' in df.columns:
        df = df[df['MEASURE'] == AQUACULTURE_VALUE_MEASURE]
    if exclude_aquatic_plants:
        if species_groups_df is None:
            raise ValueError('excluding aquatic plants needs the species-group table, so that '
                             'which species were dropped is readable rather than assumed')
        plants = set(species_groups_df.loc[
            species_groups_df['Major_Group'] == AQUATIC_PLANTS_MAJOR_GROUP, '3A_Code'])
        df = df[~df['SPECIES.ALPHA_3_CODE'].isin(plants)]
    out = df.groupby('COUNTRY.UN_CODE', as_index=False)['VALUE'].sum()
    out['aquaculture_value_usd'] = out['VALUE'] * 1000.0
    return out.rename(columns={'COUNTRY.UN_CODE': 'iso3_r250_id'})[
        ['iso3_r250_id', 'aquaculture_value_usd']]


def aquaculture_gep_by_country(value_df, share_df, countries_df, year,
                               species_groups_df=None, exclude_aquatic_plants=True):
    """Aquaculture GEP: FAO value times the natural-resource share of the country's GTAP region.

    A country FAO does not value stays NaN rather than zero: no data is not no aquaculture.
    """
    values = aquaculture_value_by_country(value_df, year, species_groups_df,
                                          exclude_aquatic_plants)
    df = countries_df.merge(values, on='iso3_r250_id', how='left')
    df = df.merge(share_df, left_on='gtap_region_label', right_on='gtap_region_label', how='left')
    df['aquaculture_gep'] = df['aquaculture_value_usd'] * df['natural_resource_share']
    return df


def natural_resource_share_of_fishing_gross_output(evfp_array, maks_array, endowments,
                                                   activities, regions,
                                                   sector=GTAP_FISHING_SECTOR,
                                                   endowment=GTAP_NATURAL_RESOURCE_ENDOWMENT):
    """The same natural-resource payments as a share of GROSS OUTPUT, not of value added.

    ⚠ Why both exist. `natural_resource_share_of_fishing` divides the natural-resource payment by
    the sector's total FACTOR payments -- its value added. FAO's aquaculture figure is a REVENUE,
    which is gross output. Multiplying a revenue by a share of value added overstates it by the
    ratio between the two, and for fishing that ratio is 0.585 on average and as low as 0.265.

    The check that this is right rather than a preference is forestry, where two independent
    sources meet: GTAP's land share of forestry value added is 0.589, and 0.589 x 0.644 = 0.380 on
    gross output, against CWoN's separately-derived forest rental ratio of 0.376. One percent apart.

    ⚠ The conversion is per region and cannot be a single factor: value added over gross output
    runs 0.265 to 0.928 across the 50 GTAP regions, so a world average would move small fishing
    economies by more than the correction itself.

    Returns:
        pandas.DataFrame: gtap_region_label and natural_resource_share_of_gross_output.
    """
    import numpy as np
    import pandas as pd
    e = endowments.index(endowment)
    a = activities.index(sector)
    evfp = np.asarray(evfp_array)
    value_added = evfp[:, a, :].sum(axis=0)
    gross_output = np.asarray(maks_array)[:, a, :].sum(axis=0)
    if (gross_output <= 0).any():
        empty = [regions[i] for i, g in enumerate(gross_output) if g <= 0]
        raise ValueError('GTAP regions with no %s gross output, so the share is undefined rather '
                         'than zero: %s' % (sector, ', '.join(empty)))
    return pd.DataFrame({'gtap_region_label': regions,
                         'natural_resource_share_of_gross_output': evfp[e, a, :] / gross_output})

"""Fisheries ES-shock task (static, mapped by RCP).

Mirrors the carbon/pollination tasks on the add_<es>_tasks seam, but the source is a pre-computed HAR
(marine, never from SEALS maps): cwon_shocks.har FI26/FI45/FI85 by RCP, constant across years, FSH only.
Writes the per-region FSH shock CSV the same way carbon/pollination write theirs, so
build_combined_afeall reads it identically. Ported verbatim from the old prepare_es_shocks fisheries
block onto the seam (Chiara's 'static ES go through the seam too' restructuring).
"""
from global_invest import utilities
import os

import hazelbean as hb
import pandas as pd

from global_invest.fisheries import fisheries_functions as ff

# NGFS scenario -> fisheries RCP header. RCP2.6=FI26 (below_2c/net_zero/low_demand),
# RCP4.5=FI45 (ndcs/delayed_transition), RCP7.0=FI85 (current_policies/fragmented_world/stress_test).
# RCP -> FI header. The headers ARE RCP-named (FI26=RCP2.6, FI45=RCP4.5, FI85=RCP8.5; FI85 also
# serves RCP7.0 as the closest available -- provenance #16). When the scenarios CSV carries a
# climate_label column (hydrate_es_scenarios publishes p.es_shock_climate_labels), the header is
# derived from the scenario's RCP and scenario NAMES need no translation at all.
RCP_FI_MAP = {'rcp26': 'FI26', 'rcp45': 'FI45', 'rcp60': 'FI85', 'rcp70': 'FI85', 'rcp85': 'FI85'}

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


def fisheries_shock(p):
    """Static marine-fisheries shock -> FSH, constant across years, mapped by RCP header.

    Caller sets on p before calling: es_shock_scenarios, es_shock_base_year,
    es_shock_end_year, fisheries_shock_output_path. cwon_shocks.har defaults via
    base_data_dir / aggregation_label (override with p.cwon_shocks_path); scenario->header via
    p.fisheries_header_map (default FISH_HEADER_MAP).
    """
    # Default into the es_shocks parent dir. Runtime, not build time: p.es_shock_dir is
    # published by that task, which ProjectFlow runs before this one.
    if not getattr(p, 'fisheries_shock_output_path', None):
        p.fisheries_shock_output_path = os.path.join(getattr(p, 'es_shock_dir', None) or p.project_dir, 'fisheries_interpolated.csv')
    if not p.run_this:
        return

    utilities.hydrate_es_parameters(p, 'fisheries', log=hb.log)   # shipped defaults; caller wins
    es_shock_base_year = int(p.es_shock_base_year)
    es_shock_end_year = int(p.es_shock_end_year)
    es_shock_scenarios = list(p.es_shock_scenarios)
    fisheries_header_map = getattr(p, 'fisheries_header_map', FISH_HEADER_MAP)
    es_shock_climate_labels = getattr(p, 'es_shock_climate_labels', None) or {}

    # The HAR reference is the cwon_shocks_path_template row (the es_lulc_path_template
    # pattern: a template cell, formatted at use -- here with the aggregation label).
    cwon_shocks_path = getattr(p, 'cwon_shocks_path', None) or p.get_path(
        p.cwon_shocks_path_template.format(aggregation_label=p.aggregation_label),
        raise_error_if_fail=False)
    if not os.path.exists(cwon_shocks_path):
        print('  fisheries shock: cwon_shocks.har not found (%s) -- skipping' % cwon_shocks_path)
        return

    fi_data = ff.read_fisheries_headers(cwon_shocks_path, headers=fisheries_headers_to_read(fisheries_header_map))

    # Read each year's own value from the FI annual series -- the honest default, no artificial freeze.
    # For the current cwon_shocks.har every year 2023..2050 already equals the 2050 value (the series is a
    # 2017->2018 step, then flat), so per-year == constant NUMERICALLY here; the distinction only bites once
    # a genuinely dynamic source (DBEM/Fish-MIP, #45) carries a real trajectory -- then this reads it for
    # free. Set p.fisheries_time_varying=False (+ fisheries_constant_year) to force a freeze if ever needed.
    time_varying = bool(getattr(p, 'fisheries_time_varying', True))
    constant_year = int(getattr(p, 'fisheries_constant_year', es_shock_end_year))

    # RAMP the FI value linearly from 0 at es_shock_base_year to its full size at es_shock_end_year, instead of taking the
    # HAR's series as-is. The HAR is a STEP -- 0 in its first slice (2017) then a constant to 2050 -- which
    # asserts the FULL RCP impact from 2018 and holds it for 32 years. That cannot be right on any reading
    # of the source: RCP2.6 and RCP4.5 have barely diverged by 2018, and no warming has accumulated by our
    # base year either, so the marine impact should start at ~0 and grow. The step looks like one number
    # filling a series rather than a temporal profile, and the header carries NO horizon metadata (it just
    # says "Fish RCP2.6 shocks") -- provenance is #16. Ramping also makes fisheries consistent with the
    # other three services, which all start at 0 in the base year.
    # ⚠ This IMPOSES a profile the data lacks: state it in methods. #16 fixes only the endpoint year (and
    # so the slope); es_shock_end_year is the current assumption.
    ramp_to_end = bool(getattr(p, 'fisheries_ramp_to_end_year', True))
    # The ramp DENOMINATOR is the horizon the FI value belongs to -- NOT the length of this run.
    # Using es_shock_end_year (= max of the scenario CSV's `years`) silently rescales the shock whenever the run
    # is short: on a 2024-2025 test run it would deliver the whole 2050 impact by 2025, a ~13x
    # overstatement, while erosion/carbon/pollination interpolate against calendar anchors and self-limit
    # correctly. Anchor on the ES anchor years (max of es_shock_years, i.e. the last SEALS year) so the
    # slope is a property of the science, not of how many years we happened to solve.
    # ⚠ Which horizon the FI number actually belongs to is undocumented upstream (#16) -- the last anchor
    # year is our assumption, and it is what sets the slope. Override with p.fisheries_ramp_end_year.
    ramp_end_year = int(getattr(p, 'fisheries_ramp_end_year', 0)) or max(
        [int(y) for y in getattr(p, 'es_shock_years', []) or []] or [es_shock_end_year])
    n_years = max(ramp_end_year - es_shock_base_year, 1)

    rows = []
    for scen in es_shock_scenarios:
        hdr = resolve_fisheries_header(scen, fisheries_header_map, es_shock_climate_labels)
        if hdr not in fi_data:
            continue
        overrides = getattr(p, 'fisheries_value_overrides', FISH_VALUE_OVERRIDES)
        for reg, series in fi_data[hdr].items():
            const_val = series.get(constant_year)
            full_val = const_val if not time_varying else series.get(es_shock_end_year, const_val)
            if (hdr, reg) in overrides:
                full_val = overrides[(hdr, reg)]
                print('  fisheries shock: OVERRIDE %s/%s -> %+.4f (corrupt at source, imputed)'
                      % (hdr, reg, full_val))
            for year in range(es_shock_base_year, es_shock_end_year + 1):
                if ramp_to_end:
                    # clipped at 1.0 so a run extending past ramp_end_year holds the full value
                    # rather than extrapolating the ramp beyond the horizon it was defined on
                    val = full_val * min((year - es_shock_base_year) / n_years, 1.0)
                else:
                    val = series.get(year, const_val) if time_varying else const_val
                for sector in p.fisheries_shock_acts:
                    rows.append({'ACTS': sector, 'REG': reg, 'scenario': scen,
                                 'year': year, 'shock_pct': val, 'fisheries_header': hdr})

    out = pd.DataFrame(rows)
    # Assert BEFORE the cap: the clip silently absorbs whatever the CWoN table delivers, so a
    # contaminated source value would otherwise be clamped to +-2 and look healthy -- the same
    # silent-failure shape the assertion exists to catch. After the clip the magnitude check
    # could never fire.
    utilities.assert_shock_table_sound(out, es_shock_scenarios, 'fisheries')
    if len(out):
        out['shock_pct'] = out['shock_pct'].clip(-FISH_CAP, FISH_CAP)
    out.to_csv(p.fisheries_shock_output_path, index=False)
    print('  fisheries shock: %d rows, %d scenarios (%s, capped +-%.0f%%) -> %s'
          % (len(out), out['scenario'].nunique() if len(out) else 0,
             ('per-year' if time_varying else 'constant @%d' % constant_year),
             FISH_CAP, p.fisheries_shock_output_path))
    return True


# =============================================================================
# GEP valuation tasks (commercial capture fisheries, CWoN method). The shock
# above and the valuation below are separate consumers of separate CWoN
# products: the shock reads the FI headers of cwon_shocks.har, the valuation
# the economic-rent tables of the CWoN 2024 reproducibility package.
# PROVISIONAL as the account's fisheries GEP until the source choice is
# blessed (the deck's open question); ported from the author's 2026 script.
# =============================================================================

def publish_inputs(p):
    """Every GEP task's first line: the fisheries es_config row and the CWoN data references
    from es_parameters (defaults layer -- a caller-set value wins), the shared country
    references and the results registry."""
    utilities.hydrate_es_config(p, 'fisheries', log=hb.log)
    utilities.hydrate_es_parameters(p, 'fisheries', log=hb.log)
    utilities.initialize_country_paths(p)
    if not hasattr(p, 'results'):
        p.results = {}
    return p


def fisheries_rent_trends(p):
    """CWoN CPI + economic rent -> per-country real-rent trend and 2019 estimate."""
    publish_inputs(p)
    p.fisheries_rent_trends_path = os.path.join(p.cur_dir, 'fisheries_rent_trends.csv')
    if not p.run_this:
        return
    if not hb.path_exists(p.fisheries_rent_trends_path):
        cpi = ff.clean_cwon_cpi(pd.read_stata(p.fisheries_cwon_cpi_path))
        rent = ff.clean_cwon_econ_rent(pd.read_stata(p.fisheries_cwon_econ_rent_path))
        deflated = ff.deflate_rent_to_2019usd(rent, cpi)
        ff.fisheries_rent_trends(deflated).to_csv(p.fisheries_rent_trends_path, index=False)
    return True


def gep_calculation(p):
    """GEP valuation for fisheries: the 2019 rent estimate joined onto the r250 country list
    (by iso3 label -- CWoN's wb_code is the same vocabulary), one row per country."""
    publish_inputs(p)
    service_results = {}
    p.results['fisheries'] = service_results
    service_results['gep_by_country_base_year'] = os.path.join(p.cur_dir, 'gep_by_country_base_year.csv')

    if hb.path_all_exist(list(service_results.values())):
        hb.log('All results already exist. Skipping GEP calculation for fisheries.')
        return
    hb.log('Starting GEP calculation for fisheries.')

    trends = pd.read_csv(p.fisheries_rent_trends_path)
    attr_cols = ['iso3_r250_id', 'iso3_r250_label', 'iso3_r250_name',
                 'continent', 'region_un', 'region_wb', 'income_grp', 'subregion']
    countries = p.df_countries[attr_cols].drop_duplicates('iso3_r250_id')
    df_gep = ff.commfish_gep_by_country(trends, countries)
    df_gep['year'] = int(p.gep_base_year)
    df_gep = df_gep.rename(columns={'commfish_provision': 'fisheries_gep'})
    hb.df_write(df_gep[attr_cols + ['year', 'fisheries_gep']],
                service_results['gep_by_country_base_year'])

    hb.log(f'Total fisheries GEP (commercial capture, CWoN method, PROVISIONAL) for base year '
           f'{p.gep_base_year}: {df_gep["fisheries_gep"].sum():,.2f}')
    return True


def gep_result(p):
    """Render the results report(s). Shared implementation in utilities."""
    publish_inputs(p)
    utilities.render_service_results(p)

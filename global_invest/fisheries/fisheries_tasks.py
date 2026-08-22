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
    fisheries_header_map = getattr(p, 'fisheries_header_map', ff.FISH_HEADER_MAP)

    # The HAR reference is the cwon_shocks_path_template row (the es_lulc_path_template
    # pattern: a template cell, formatted at use -- here with the aggregation label).
    cwon_shocks_path = getattr(p, 'cwon_shocks_path', None) or p.get_path(
        p.cwon_shocks_path_template.format(aggregation_label=p.aggregation_label),
        raise_error_if_fail=False)
    if not os.path.exists(cwon_shocks_path):
        print('  fisheries shock: cwon_shocks.har not found (%s) -- skipping' % cwon_shocks_path)
        return

    fi_data = ff.read_fisheries_headers(
        cwon_shocks_path, headers=ff.fisheries_headers_to_read(fisheries_header_map))

    # Read each year's own value from the FI annual series -- the honest default, no artificial freeze.
    # For the current cwon_shocks.har every year 2023..2050 already equals the 2050 value (the series is a
    # 2017->2018 step, then flat), so per-year == constant NUMERICALLY here; the distinction only bites once
    # a genuinely dynamic source (DBEM/Fish-MIP, #45) carries a real trajectory -- then this reads it for
    # free. Set p.fisheries_time_varying=False (+ fisheries_constant_year) to force a freeze if ever needed.
    time_varying = bool(getattr(p, 'fisheries_time_varying', True))

    # RAMP the FI value from 0 at the base year rather than taking the HAR's series as-is: the HAR
    # is a STEP asserting the full RCP impact from 2018, which no reading of the source supports (no
    # warming has accumulated by our base year), and the other three services all start at 0.
    # ⚠ This IMPOSES a profile the data lacks: state it in methods; provenance is #16.
    # The ramp anchors on the last SEALS anchor year, NOT the run length -- anchoring on the run's
    # own end year would deliver the whole 2050 impact by 2025 on a short test run (~13x). Which
    # horizon the FI number actually belongs to is undocumented upstream (#16); override with
    # p.fisheries_ramp_end_year. Mechanics are on ff.static_shock_rows.
    rows = ff.static_shock_rows(
        fi_data, es_shock_scenarios, fisheries_header_map,
        climate_labels=getattr(p, 'es_shock_climate_labels', None) or {},
        overrides=getattr(p, 'fisheries_value_overrides', ff.FISH_VALUE_OVERRIDES),
        sectors=p.fisheries_shock_acts,
        base_year=es_shock_base_year, end_year=es_shock_end_year,
        time_varying=time_varying,
        constant_year=int(getattr(p, 'fisheries_constant_year', es_shock_end_year)),
        ramp_to_end=bool(getattr(p, 'fisheries_ramp_to_end_year', True)),
        ramp_end_year=int(getattr(p, 'fisheries_ramp_end_year', 0)) or max(
            [int(y) for y in getattr(p, 'es_shock_years', []) or []] or [es_shock_end_year]))

    out = pd.DataFrame(rows)
    # Assert BEFORE the cap: the clip silently absorbs whatever the CWoN table delivers, so a
    # contaminated source value would otherwise be clamped to +-2 and look healthy -- the same
    # silent-failure shape the assertion exists to catch. After the clip the magnitude check
    # could never fire.
    utilities.assert_shock_table_sound(out, es_shock_scenarios, 'fisheries')
    if len(out):
        out['shock_pct'] = out['shock_pct'].clip(-ff.FISH_CAP, ff.FISH_CAP)
    out.to_csv(p.fisheries_shock_output_path, index=False)
    print('  fisheries shock: %d rows, %d scenarios (%s, capped +-%.0f%%) -> %s'
          % (len(out), out['scenario'].nunique() if len(out) else 0,
             ('per-year' if time_varying else 'constant @%d'
              % int(getattr(p, 'fisheries_constant_year', es_shock_end_year))),
             ff.FISH_CAP, p.fisheries_shock_output_path))
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


def fisheries_subsistence_gep(p):
    """Subsistence-fisheries GEP: the Lynch et al. (2024) per-country consumptive-use value
    on the canonical r250 rows (the split-country guard: one row per country, never summed
    across sub-regions). A separate component from the commercial CWoN estimate -- whether
    and how the two combine into the account's fisheries value is the open scope question."""
    publish_inputs(p)
    p.fisheries_subsistence_gep_path = os.path.join(p.cur_dir, 'subsistence_gep_by_country.csv')
    if not p.run_this:
        return
    if not hb.path_exists(p.fisheries_subsistence_gep_path):
        lynch = pd.read_excel(p.fisheries_subsistence_lynch_path, engine='openpyxl')
        # Canonical r250 rows with the Natural Earth name key the release joins on.
        countries = p.df_countries[p.df_countries['ee_r264_label'] == p.df_countries['iso3_r250_label']]
        countries = countries[['brk_name', 'ee_r264_id', 'iso3_r250_id', 'iso3_r250_label',
                               'ee_r264_description']].drop_duplicates('iso3_r250_id')
        out = ff.subsistence_fisheries_by_country(lynch, countries)
        out.to_csv(p.fisheries_subsistence_gep_path, index=False)
        p.results.setdefault('fisheries', {})
        hb.log('fisheries subsistence GEP (Lynch et al. 2024): %d countries with values, '
               'total %.4g USD' % (out['subsistence_fisheries_gep'].notna().sum(),
                                   out['subsistence_fisheries_gep'].sum()))
    return True

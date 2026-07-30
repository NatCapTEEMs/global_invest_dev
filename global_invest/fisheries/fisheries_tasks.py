"""Fisheries ES-shock task (static, mapped by RCP).

Mirrors the carbon/pollination tasks on the add_<es>_tasks seam, but the source is a pre-computed HAR
(marine, never from SEALS maps): cwon_shocks.har FI26/FI45/FI85 by RCP, constant across years, FSH only.
Writes the per-region FSH shock CSV the same way carbon/pollination write theirs, so
build_combined_afeall reads it identically. Ported verbatim from the old prepare_es_shocks fisheries
block onto the seam (Chiara's 'static ES go through the seam too' restructuring).
"""
import os
import pandas as pd

from global_invest.fisheries import fisheries_functions as ff

# NGFS scenario -> fisheries RCP header. RCP2.6=FI26 (below_2c/net_zero/low_demand),
# RCP4.5=FI45 (ndcs/delayed_transition), RCP7.0=FI85 (current_policies/fragmented_world/stress_test).
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
FISH_SECTORS = ('FSH',)


def task_compute_fisheries_shock(p):
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

    base_year = int(p.es_shock_base_year)
    end_year = int(p.es_shock_end_year)
    scenarios = list(p.es_shock_scenarios)
    header_map = getattr(p, 'fisheries_header_map', FISH_HEADER_MAP)

    cwon_path = getattr(p, 'cwon_shocks_path', None) or os.path.join(
        p.base_data_dir, 'gtappy', 'cge_releases', 'gtapv7-aez-rd', 'data',
        p.aggregation_label, 'cwon_shocks.har')
    if not os.path.exists(cwon_path):
        print('  fisheries shock: cwon_shocks.har not found (%s) -- skipping' % cwon_path)
        return

    fi_data = ff.read_fisheries_headers(cwon_path, headers=tuple(sorted(set(header_map.values()))))

    # Read each year's own value from the FI annual series -- the honest default, no artificial freeze.
    # For the current cwon_shocks.har every year 2023..2050 already equals the 2050 value (the series is a
    # 2017->2018 step, then flat), so per-year == constant NUMERICALLY here; the distinction only bites once
    # a genuinely dynamic source (DBEM/Fish-MIP, #45) carries a real trajectory -- then this reads it for
    # free. Set p.fisheries_time_varying=False (+ fisheries_constant_year) to force a freeze if ever needed.
    time_varying = bool(getattr(p, 'fisheries_time_varying', True))
    constant_year = int(getattr(p, 'fisheries_constant_year', end_year))

    # RAMP the FI value linearly from 0 at base_year to its full size at end_year, instead of taking the
    # HAR's series as-is. The HAR is a STEP -- 0 in its first slice (2017) then a constant to 2050 -- which
    # asserts the FULL RCP impact from 2018 and holds it for 32 years. That cannot be right on any reading
    # of the source: RCP2.6 and RCP4.5 have barely diverged by 2018, and no warming has accumulated by our
    # base year either, so the marine impact should start at ~0 and grow. The step looks like one number
    # filling a series rather than a temporal profile, and the header carries NO horizon metadata (it just
    # says "Fish RCP2.6 shocks") -- provenance is #16. Ramping also makes fisheries consistent with the
    # other three services, which all start at 0 in the base year.
    # ⚠ This IMPOSES a profile the data lacks: state it in methods. #16 fixes only the endpoint year (and
    # so the slope); end_year is the current assumption.
    ramp_to_end = bool(getattr(p, 'fisheries_ramp_to_end_year', True))
    # The ramp DENOMINATOR is the horizon the FI value belongs to -- NOT the length of this run.
    # Using end_year (= max of the scenario CSV's `years`) silently rescales the shock whenever the run
    # is short: on a 2024-2025 test run it would deliver the whole 2050 impact by 2025, a ~13x
    # overstatement, while erosion/carbon/pollination interpolate against calendar anchors and self-limit
    # correctly. Anchor on the ES anchor years (max of es_shock_years, i.e. the last SEALS year) so the
    # slope is a property of the science, not of how many years we happened to solve.
    # ⚠ Which horizon the FI number actually belongs to is undocumented upstream (#16) -- the last anchor
    # year is our assumption, and it is what sets the slope. Override with p.fisheries_ramp_end_year.
    ramp_end_year = int(getattr(p, 'fisheries_ramp_end_year', 0)) or max(
        [int(y) for y in getattr(p, 'es_shock_years', []) or []] or [end_year])
    n_years = max(ramp_end_year - base_year, 1)

    rows = []
    for scen in scenarios:
        hdr = header_map.get(scen)
        if hdr is None or hdr not in fi_data:
            continue
        overrides = getattr(p, 'fisheries_value_overrides', FISH_VALUE_OVERRIDES)
        for reg, series in fi_data[hdr].items():
            const_val = series.get(constant_year)
            full_val = const_val if not time_varying else series.get(end_year, const_val)
            if (hdr, reg) in overrides:
                full_val = overrides[(hdr, reg)]
                print('  fisheries shock: OVERRIDE %s/%s -> %+.4f (corrupt at source, imputed)'
                      % (hdr, reg, full_val))
            for year in range(base_year, end_year + 1):
                if ramp_to_end:
                    # clipped at 1.0 so a run extending past ramp_end_year holds the full value
                    # rather than extrapolating the ramp beyond the horizon it was defined on
                    val = full_val * min((year - base_year) / n_years, 1.0)
                else:
                    val = series.get(year, const_val) if time_varying else const_val
                for sector in FISH_SECTORS:
                    rows.append({'ACTS': sector, 'REG': reg, 'scenario': scen,
                                 'year': year, 'shock_pct': val, 'fisheries_header': hdr})

    out = pd.DataFrame(rows)
    if len(out):
        out['shock_pct'] = out['shock_pct'].clip(-FISH_CAP, FISH_CAP)
    out.to_csv(p.fisheries_shock_output_path, index=False)
    print('  fisheries shock: %d rows, %d scenarios (%s, capped +-%.0f%%) -> %s'
          % (len(out), out['scenario'].nunique() if len(out) else 0,
             ('per-year' if time_varying else 'constant @%d' % constant_year),
             FISH_CAP, p.fisheries_shock_output_path))
    return True

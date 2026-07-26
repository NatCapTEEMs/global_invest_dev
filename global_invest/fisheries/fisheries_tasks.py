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
FISH_CAP = 5.0          # +-5% cap (consistent with the afeall cap)
FISH_SECTORS = ('FSH',)


def task_compute_fisheries_shock(p):
    """Static marine-fisheries shock -> FSH, constant across years, mapped by RCP header.

    Caller sets on p before calling: fisheries_shock_scenarios, fisheries_shock_base_year,
    fisheries_shock_end_year, fisheries_shock_output_path. cwon_shocks.har defaults via
    base_data_dir / aggregation_label (override with p.cwon_shocks_path); scenario->header via
    p.fisheries_header_map (default FISH_HEADER_MAP).
    """
    if not p.run_this:
        return

    base_year = int(p.fisheries_shock_base_year)
    end_year = int(p.fisheries_shock_end_year)
    scenarios = list(p.fisheries_shock_scenarios)
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

    rows = []
    for scen in scenarios:
        hdr = header_map.get(scen)
        if hdr is None or hdr not in fi_data:
            continue
        for reg, series in fi_data[hdr].items():
            const_val = series.get(constant_year)
            for year in range(base_year, end_year + 1):
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

"""Erosion-control ES-shock task (STATIC, per-scenario). Mirrors carbon/pollination/fisheries on the
add_<es>_tasks seam.

Ported verbatim from the old prepare_es_shocks erosion block: read erosion_prevention_dependency.csv,
subtract the baseline_ignore_damages reference, linearly ramp 0 -> the scenario value over the horizon,
apply to the 8 erosion-affected crop sectors, write erosion_prevention_interpolated.csv. UNCAPPED here --
the cap is applied later on the COMBINED value in build_combined_afeall_cc_es (matches the old block).
The paper wants this DYNAMIC (InVEST SDR on each SEALS map -- Nfamara's global_erosion_gep), the heavy
upgrade tracked in #26; this module is the static seam so the dynamic swap is contained later.
"""
import os
import pandas as pd

from global_invest.erosion import erosion_functions as ef

# 8 crop sectors whose productivity depends on erosion control (sediment retention).
EROSION_SECTORS = ('PDR', 'WHT', 'GRO', 'V_F', 'OSD', 'C_B', 'PFB', 'OCR')
# our scenario -> raw_dependencies scenario name(s), with fallbacks (stress_test reuses current_policies).
EROSION_SCENARIO_MAP = {
    'below_2c': ['below_2c'], 'current_policies': ['current_policies'],
    'delayed_transition': ['delayed_transition'], 'fragmented_world': ['fragmented_world'],
    'low_demand': ['low_demand'], 'ndcs': ['ndcs'],
    'net_zero': ['net_zero', 'net_zero_2050'], 'stress_test': ['current_policies'],
}


def task_compute_erosion_shock(p):
    """Static per-scenario erosion shock -> 8 crop sectors, linear ramp 0->end_year.

    Caller sets on p before calling: erosion_shock_scenarios, erosion_shock_base_year,
    erosion_shock_end_year, erosion_shock_output_path. Dependency csv defaults to
    input_dir/raw_dependencies/erosion_prevention_dependency.csv (override p.erosion_dependency_path);
    scenario->raw name via p.erosion_scenario_map (default EROSION_SCENARIO_MAP).
    """
    if not p.run_this:
        return

    base_year = int(p.erosion_shock_base_year)
    end_year = int(p.erosion_shock_end_year)
    n_years = end_year - base_year
    scenario_map = getattr(p, 'erosion_scenario_map', EROSION_SCENARIO_MAP)
    scenarios = list(p.erosion_shock_scenarios)

    ero_path = getattr(p, 'erosion_dependency_path', None) or os.path.join(
        p.input_dir, 'raw_dependencies', 'erosion_prevention_dependency.csv')
    if not os.path.exists(ero_path):
        print('  erosion shock: dependency csv not found (%s) -- skipping' % ero_path)
        return

    df, base_vals = ef.read_erosion_dependency(ero_path)

    rows = []
    for our_scn in scenarios:
        candidates = scenario_map.get(our_scn)
        raw_scn = ef.find_scenario(df, candidates) if candidates else None
        if not raw_scn:
            continue
        scn_vals = df[df['scenario'] == raw_scn].set_index(
            ['aez18_id', 'gtapv7_r50_label'])['value'].astype(float).fillna(0)
        common = scn_vals.index.intersection(base_vals.index)
        shock = scn_vals.loc[common] - base_vals.loc[common]
        for year in range(base_year, end_year + 1):
            frac = (year - base_year) / n_years
            for (aez_id, reg), val in shock.items():
                endw = 'AEZ%d' % int(aez_id)
                for sector in EROSION_SECTORS:
                    rows.append({'ENDW': endw, 'ACTS': sector, 'REG': reg,
                                 'scenario': our_scn, 'year': year, 'shock_pct': val * frac})

    out = pd.DataFrame(rows)
    out.to_csv(p.erosion_shock_output_path, index=False)
    nz = out[(out['year'] == end_year) & (out['shock_pct'] != 0)] if len(out) else out
    print('  erosion shock: %d rows, %d scenarios, %d nonzero @%d (static, uncapped) -> %s'
          % (len(out), out['scenario'].nunique() if len(out) else 0, len(nz), end_year,
             p.erosion_shock_output_path))
    return True

"""Dynamic pollination ES-shock task (V_F, OSD).

Runs William Sidemo-Holm's crop_benefits pollination chain on our SEALS 300 m maps at EACH SEALS
anchor year (seals_years), then piecewise-linearly interpolates the shock to annual values. Writes
pollination_interpolated.csv -- the file build_combined_afeall_cc_es reads -- into the
prepare_es_shocks folder, standing in for the static pollination path when pollination is in
dynamic_es.
"""
import glob
import numpy as np
import pandas as pd

from global_invest.pollination import pollination_functions as pf


def task_compute_pollination_shock(p):
    """Per-scenario 300 m LULC at each SEALS anchor year -> V_F/OSD shock, piecewise-interp to annual.

    Caller sets on p: pollination_shock_years (SEALS anchor years, from seals_years),
    pollination_shock_base_year, pollination_shock_scenarios, pollination_lulc_path_template
    ({scenario}/{year}) or scenario_lulc_paths, pollination_baseline_lulc_path,
    pollination_shock_output_path. Optional: pollination_shock_base_scenario, pollination_shock_acts,
    region_boundary_path.
    """
    if not p.run_this:
        return

    base_scn     = getattr(p, 'pollination_shock_base_scenario', 'baseline_ignore_dependencies')
    base_year    = int(p.pollination_shock_base_year)
    acts         = getattr(p, 'pollination_shock_acts', ('V_F', 'OSD'))
    scenarios    = list(p.pollination_shock_scenarios)
    anchor_years = sorted(y for y in map(int, p.pollination_shock_years) if y > base_year)
    end_year     = anchor_years[-1]

    if not getattr(p, 'region_boundary_path', None):
        p.region_boundary_path = p.get_path('gtap_invest/region_boundaries/ee_r50_aez18_correspondence.gpkg')

    cfg = pf.configure_crop_benefits(p, base_year)            # crop_benefits Config -> our base_data + task dir

    if not getattr(p, 'scenario_lulc_paths', None):
        tmpl = p.pollination_lulc_path_template
        p.scenario_lulc_paths = {s: {y: glob.glob(tmpl.format(scenario=s, year=y))[0]
                                     for y in anchor_years if glob.glob(tmpl.format(scenario=s, year=y))}
                                 for s in [base_scn] + scenarios}

    # denominator (unpaired 2023 value) is year-independent -> compute once
    denom = pf.baseline_denominator(cfg, p.pollination_baseline_lulc_path, base_year)

    # value[scenario][year] = per-region % change of that scenario's year-map vs the 2023 baseline (stable ag)
    value = {}
    for year in anchor_years:
        for scen in [base_scn] + scenarios:
            value.setdefault(scen, {})[year] = pf.scenario_region_pct_change(
                cfg, scenario=f'{scen}_{year}', lulc_path=p.scenario_lulc_paths[scen][year],
                baseline_lulc_path=p.pollination_baseline_lulc_path,
                denominator_path=denom, correspondence_gpkg=p.region_boundary_path,
                target_year=base_year)

    # ES shock numerator = scenario minus baseline_ignore_dependencies value at each anchor (S_y - B_y); interp annually.
    # TWO denominators emitted under explicit names (carbon uses the SAME names) for the #14 diagnostic:
    #   shock_pct_fixedbase = (S_y - B_y) / B0    -- B0 = base-year (2023) value, denominator FIXED across years
    #   shock_pct_contemp   = (S_y - B_y) / B_y   -- B_y = baseline year-y value = (poll_scenario - poll_base)/poll_base
    #   shock_pct           = GTAP-primary = shock_pct_contemp (÷B_y), matching carbon. afeall is a productivity
    #                         deviation from the baseline path (scenarios take productivity as given from BAU), so
    #                         normalize by the year-y baseline, not the fixed 2023 value.
    # contemp is an EXACT rescale of fixedbase (no extra rasters): (sum B_y*area)/(sum B0*area) = 1 + value[base_scn][y]/100,
    # so contemp = fixedbase / that factor. Guards only exact-zero; near-zero baselines handled after seeing #14.
    all_years, rows = list(range(base_year, end_year + 1)), []
    for scen in scenarios:
        anchor_shock = pd.DataFrame({y: value[scen][y] - value[base_scn][y] for y in anchor_years}).dropna()
        # reindex the growth factor onto anchor_shock's own zones, so contemp has exactly the same rows
        base_factor = pd.DataFrame({y: 1.0 + value[base_scn][y] / 100.0
                                    for y in anchor_years}).reindex(anchor_shock.index).replace(0, np.nan)
        anchor_contemp = anchor_shock / base_factor
        for (endw, reg), s in anchor_shock.iterrows():
            annual   = np.interp(all_years, [base_year] + anchor_years, [0.0] + list(s.values))
            annual_c = np.interp(all_years, [base_year] + anchor_years,
                                 [0.0] + list(anchor_contemp.loc[(endw, reg)].values))
            for year, v, vc in zip(all_years, annual, annual_c):
                for sector in acts:
                    rows.append({'ENDW': endw, 'ACTS': sector, 'REG': reg, 'scenario': scen,
                                 'year': year, 'shock_pct': vc,
                                 'shock_pct_fixedbase': v, 'shock_pct_contemp': vc})

    out = pd.DataFrame(rows)
    out.to_csv(p.pollination_shock_output_path, index=False)
    print('  pollination shock: %d rows, %d scenarios (shock_pct=shock_pct_contemp=/baseline-year value, shock_pct_fixedbase=/2023 value) -> %s'
          % (len(out), out['scenario'].nunique() if rows else 0, p.pollination_shock_output_path))
    return True

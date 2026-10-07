"""Opt-in retained-cropland dollar accounting for the ordinary pollination task."""
from pathlib import Path
import numpy as np
import pandas as pd


def build_retained_table(p, cfg, base_map, denominator_path, anchor_years,
                         scenarios, base_year):
    from global_invest.pollination import pollination_tasks as tasks
    from global_invest.pollination import pollination_functions as pf
    from global_invest.pollination.retained_raster import retained_fraction_raster
    from global_invest.pollination.retained_value import annual_retained_rows
    baseline, area, zones, labels = tasks._zonal_context(p, denominator_path,p.region_boundary_path)
    unpaired = pf.zonal_weighted_sum(baseline,area,zones,labels)
    frames=[]
    for scenario in scenarios:
        bases, futures = {}, {}
        for year in anchor_years:
            tag=f'{scenario}_{year}'
            tasks.scenario_diff_raster(cfg,tag,p.scenario_lulc_paths[scenario][year],
                                      base_map,base_year,p.pollination_shock_baseline_label)
            fraction_path=Path(cfg.output_dir)/f'retained_fraction_{tag}_v2.tif'
            retained_fraction_raster(base_map,p.scenario_lulc_paths[scenario][year],
                                     denominator_path,fraction_path)
            fraction,_=tasks._read_masked(str(fraction_path))
            b,_=tasks._read_masked(tasks.paired_baseline_value_path(cfg,tag,p.pollination_shock_baseline_label))
            f,_=tasks._read_masked(tasks.paired_scenario_value_path(cfg,tag))
            active=fraction>0
            if np.any(active & (np.isfinite(b)!=np.isfinite(f))):
                raise ValueError(f'{tag}: inconsistent paired valuation coverage')
            # No-crop-value cells may be absent in both rasters. A baseline value
            # cannot silently vanish from both paired rasters on retained cropland.
            if np.any(active & np.isfinite(baseline) & (baseline>0) & ~np.isfinite(b)):
                raise ValueError(f'{tag}: baseline value missing on retained support')
            rb=np.where(active,b*fraction,0.)
            rf=np.where(active,f*fraction,0.)
            bases[year]=pf.zonal_weighted_sum(rb,area,zones,labels)
            futures[year]=pf.zonal_weighted_sum(rf,area,zones,labels)
        frames.append(annual_retained_rows(bases,futures,unpaired,scenario,base_year,p.pollination_shock_acts))
    return pd.concat(frames,ignore_index=True)

# Regenerating the pollination value raster

The GEP pollination value comes from the source author's raster,
`poll_value_global_<year>usd.tif`, not from a construction of ours. His published files cover only
some years, so when the GEP base year is not among them the raster has to be built by running his
pipeline at that year. This is how the 2019 file in `base_data/crop_benefits/` was produced on
2026-08-28.

Everything below runs **his** code from his repository. Nothing here is our science.

## Prerequisites

His repo cloned beside the others, and importable:

```
/Users/ccs/Files/gep_repos/crop_benefits
```

His `config/default.yaml` already carries `target_year: 2019` and `deflation_base_year: 2019`, which
are the values the GEP account wants. Confirm before running — they are on lines 118 and 128.

His `config/local.yaml` is gitignored and machine-specific. Ours points at our base_data and adds one
path override, because his default expects the ESA maps at the base_data root while ours are nested:

``` yaml
base_dirs:
  inputs: /Users/ccs/Files/base_data
  outputs: /Users/ccs/Files/gep_repos/crop_benefits_outputs
paths:
  lulc_esa: lulc/esa/lulc_esa_{target_year}.tif
```

That override matches the pattern in the `natcap.yaml` he ships, so it is his convention, not an
invention of ours.

## Step 0 — median prices for the target year's window

His `--step prices` downloads the FAO bulk tables and fetches World Bank FX. We already hold the
output of that step as `base_data/fao/fao_prices_1993_2024.parquet`, with the FX reconstruction done,
so the download is unnecessary. What is missing is only the median over the target year's window,
which for 2019 is **2017-2021** (his window is centred on the target year; ours elsewhere in the
library is 2018-2022, which is a different thing and not interchangeable).

Build it by calling his own function, so the country to subregion to region to world fallback
hierarchy is his:

``` python
import sys; sys.path.insert(0, 'src')
import pandas as pd
from pathlib import Path
from crop_benefits.fao.values import _compute_median_prices

prices = pd.read_parquet('/Users/ccs/Files/base_data/fao/fao_prices_1993_2024.parquet')
cw = pd.read_csv('/Users/ccs/Files/base_data/fao/crosswalks/crosswalk_m49_iso3.csv',
                 encoding='utf-8-sig')
outdir = Path('/Users/ccs/Files/gep_repos/crop_benefits_outputs/fao/total/fao_prices/'
              'median_prices_2017_2021')
outdir.mkdir(parents=True, exist_ok=True)
_compute_median_prices(prices, [2017, 2018, 2019, 2020, 2021], outdir, cw)
```

It writes `price_median_usd_tonne_2017_2021.csv`, 11,549 rows: 8,250 country, 2,172 subregion, 915
region, 212 world.

⚠ This is the one substitution in the whole runbook. Running `--step prices` instead would download
fresh data and is the more faithful route; it was avoided only to keep the run offline.

## Steps 1-4 — his raster chain

Run from the repo root, in order. Each step depends on the one before it.

``` bash
python scripts/pipelines/run_fao_pipeline.py    --step yield_change
python scripts/pipelines/run_raster_pipeline.py --step yield
python scripts/pipelines/run_raster_pipeline.py --step production
python scripts/pipelines/run_raster_pipeline.py --step poll_value
```

What each does, and what to check:

| step | writes | check |
|---|---|---|
| `yield_change` | `fao/total/fao_production/yield_change_2000_2019/yield_change_ratios.csv` | 11,168 rows; log should say late period `[2017, 2018, 2019, 2020, 2021]` |
| `yield` | `rasters_2019/total/yield_2019/` | 158 files; this is the Monfreda step and the slow one, roughly 20 minutes |
| `production` | `rasters_2019/total/production_2019/` | 158 files |
| `poll_value` | `rasters_2019/pollination/value_2019/poll_value_global_2019usd.tif` | area-weighted total $383.68bn |

## Step 5 — stage it where the GEP pipeline looks

``` bash
cp /Users/ccs/Files/gep_repos/crop_benefits_outputs/rasters_2019/pollination/value_2019/poll_value_global_2019usd.tif \
   /Users/ccs/Files/base_data/crop_benefits/
```

`pollination_functions.find_source_value_raster` then resolves the exact base year and applies a
deflator of 1.0000. The GEP task multiplies by cell area, because his file is a density in USD/km2,
and writes USD in the cell.

## Optional — the sufficiency-weighted figure

Not part of the GEP headline, which carries no sufficiency term, but this is how his $190bn is
reproduced:

``` bash
python scripts/pipelines/run_pollination_sufficiency.py --step sufficiency_300m
python scripts/pipelines/run_pollination_sufficiency.py --step sufficiency_5km
python scripts/pipelines/run_pollination_sufficiency.py --step valuation_5km
```

The last logs `value_pollination_sufficiency_2019_5km.tif = 189.732 B USD`.

## What the numbers should come out as

| quantity | value | his published figure |
|---|---|---|
| unweighted, area-weighted | $383.68bn | $385bn |
| sufficiency-weighted | $189.732bn | $190bn |
| ratio | 0.4945 | 0.4935 |

If the unweighted total lands near $388bn instead, the GEP path is reading our own rebuilt raster
(`pollination_value_raster_rebuilt`) rather than his — that function is kept as a cross-check, not as
the GEP path.

If the ratio lands near 0.42, the value raster is his **2020** vintage rather than 2019. The two are
close in total but differ spatially enough to move the sufficiency-weighted ratio by seven points,
which is what made this difference so hard to find.

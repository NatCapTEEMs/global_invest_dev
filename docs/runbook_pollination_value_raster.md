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

## Step 0 — FAO prices

Run his price step, which downloads the FAO bulk tables and fetches World Bank FX, then his values
step, which computes the median over the target year's window (2017-2021 for 2019 — his window is
centred on the target year, and is NOT the 2018-2022 window used elsewhere in this library):

``` bash
python scripts/pipelines/run_fao_pipeline.py --step prices
python scripts/pipelines/run_fao_pipeline.py --step values
```

`prices` writes a 404,966-row panel; `values` writes
`median_prices_2017_2021/price_median_usd_tonne_2017_2021.csv` with the country to subregion to
region to world fallback hierarchy.

⚠ Requires network access. A previous version of this runbook computed the medians from a cached
copy of the price panel to stay offline; that produced a value raster 0.1 percent different, so
prefer the download.

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
| `poll_value` | `rasters_2019/pollination/value_2019/poll_value_global_2019usd.tif` | area-weighted total $384.56bn |

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

The last logs `value_pollination_sufficiency_2019_5km.tif = 190.119 B USD`.

⚠⚠ That figure is NOT area-weighted correctly. His `_sum_raster_tiled` computes cell area once per
4096-row tile from the tile midpoint; the raster is 3600 rows, so it is a single tile at the equator
and every cell on Earth is given 30.980 km2. Weighting each row by its own latitude gives
**$161.24bn**, 18 percent lower. His unweighted summary does area-weight correctly, so the two
totals in his own pipeline are not computed the same way.

## What the numbers should come out as

| quantity | value | his published figure |
|---|---|---|
| unweighted, area-weighted | $384.56bn | $385bn |
| sufficiency-weighted, area-weighted correctly | $161.24bn | $190bn (see below) |
| ratio | 0.4193 | 0.4935 (see below) |

If the unweighted total lands near $388bn instead, the GEP path is reading our own rebuilt raster
(`pollination_value_raster_rebuilt`) rather than his — that function is kept as a cross-check, not as
the GEP path.

A ratio near **0.42 is correct** — it is what both pipelines give once cell area is handled properly.
The 0.4935 that appears in his published figures is the equator-constant summing described above, not
a different model. An earlier version of this runbook said the opposite, attributing the gap to his
2020 versus 2019 value raster; that was wrong, and it was wrong because his 2020 raster had been
compared against his 2019 sufficiency.

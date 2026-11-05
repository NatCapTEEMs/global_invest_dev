# EPS-5 raster-to-share comparison: launch package

This package runs the library's own `prevention_shares` at the EPS-5 threshold from the rasters and compares the output with the author's EPS-5 shares. It reads the original stacks, and the gates refuse any path into the repaired stacks. It writes only inside its own folder.

## What differs from the library's SES-11 run

The high threshold and two labels, set before hydration in `launch/eps5_probe.py`. The library code is `git archive` of the commit named in `parameters_template.csv`. `eps5_settings.csv` holds each pinned setting with its evidence, and `author_pins.csv` holds the author files by SHA-256.

## The failed identity gate

The staged SDR rasters are on the author's grid but are not identical to the author's rasters. The gate has no tolerance, so `run_eps5_preflight.py` reports FAIL and keeps reporting it. `prediction/sdr_rasters_library_vs_author.csv` holds the counts and `prediction/eps5_preflight_first_run.txt` the first report.

`run_eps5_raster_reach.py` carries both sets of rasters through the task's resampling and predicts every share that differs, with two falsifiers: the author's rasters must reproduce the author's stored shares, and the staged rasters must reproduce the SES-11 shares. Both hold on MSI, and a laptop fails them at the rounding scale, so the check belongs on the job's platform. The prediction for EPS-5:

- `ps_onfarm_cropland_severe.tif` differs from the author's at exactly four cells, rows 1443 and 1444, columns 9258 and 9259, each by 5.96e-08.
- The upstream and combined rasters do not differ.
- At most the 21 on-farm rows in `prediction/sdr_reach_expected_differences.csv` move in the tables, and each value must stay within its bound in `prediction/magnitude_bounds.csv`.

`run_eps5_compare.py` writes `comparison/prediction_check.csv`, which reports location and magnitude apart for each output. Raster cells must equal their predicted value to the last bit. The prediction does not fix table values to the last bit, because the windowed resampling behind it is not bit exact with the job. So they are held to a bound derived from the predicted share step and the rounding of the task's float32 products and float64 sums, in `launch/eps5_prediction.py`. A value within that bound is reported as within it, not as equal. A broken structure, repeated key, changed mask, difference outside the predicted cells, keys and columns, or a value over its bound falsifies the prediction.

## Files

| Path | Where it runs | What it does |
|---|---|---|
| `run_eps5_code_snapshot.py` | local | exports the library commit into `code/global_invest_dev/` |
| `run_eps5_local_checks.py` | local | controls on held files, writes `preflight/local_checks.txt` |
| `launch/test_eps5_compare.py` | anywhere with pytest | 45 tests of the comparison, gates and prediction checks |
| `run_eps5_manifest.py` | MSI login | pins library inputs, author files and this package |
| `run_eps5_preflight.py` | MSI login | decodes the rasters and checks everything the laptop cannot |
| `run_eps5_raster_cells.py` | MSI login | lists the native cells that differ. Footprints across the antimeridian span the world, so its cropland counts overstate the last column |
| `run_eps5_raster_reach.py` | MSI login | writes the prediction, one grid in memory at a time |
| `run_eps5_prediction_bounds.py` | MSI login | writes `prediction/magnitude_bounds.csv` from the prediction and the author's values |
| `run_eps5_rule37_resolution.py` | local, after the commit | writes `manifest/rule37_resolution.txt` |
| `run_eps5_shares.sbatch`, `run_eps5_shares.py` | MSI job | `prevention_shares` in `project/gep_erosion_eps5_library` |
| `run_eps5_heartbeat_check.py` | MSI login, during the job | says whether CPU time is advancing |
| `run_eps5_compare.py` | MSI login, after the job | writes `comparison/`, with no tolerance |

Every run file takes no arguments: `python -I -B <file>`. `parameters_template.csv` lists the keys of `msi_parameters.csv` and `local_parameters.csv`, the only machine-specific values.

## Launch order

1. Copy `parameters_template.csv` into the two parameter files and fill the blanks. Run `run_eps5_code_snapshot.py` and `run_eps5_local_checks.py`.
2. Copy the folder to the cluster. Run `run_eps5_manifest.py`, then `run_eps5_preflight.py`.
3. With this code in a pushed commit, set `code_commit`, run `run_eps5_rule37_resolution.py` and copy `manifest/rule37_resolution.txt` to the cluster.
4. Record the decision to run against the prediction in `manifest/raster_gate_decision.txt`, naming the SHA-256 of `prediction/prediction_sha256.txt`. The sbatch refuses without it.
5. Move `preflight/eps5_preflight.txt` aside, rerun the preflight, and `sbatch run_eps5_shares.sbatch`.
6. After the job, run `run_eps5_compare.py`.

## Resources

`agsmall`, 2 CPUs, 48 GB, 24 h. The library's SES-11 run of the same task took 9 h 42 min with one core busy and 25.8 GB at peak, and the author's EPS-5 run took 1.7 percent longer than the author's SES-11 run. Output is under 0.2 GB. A login session on the cluster is killed near 6 GB, which is why the reach check holds grids on disk.

## Reading the result

Verdicts are `BYTE_IDENTICAL`, `VALUES_IDENTICAL` or `DIFFERENT`. On the original stacks the `*_production_tons` columns hold thousands of tonnes, because the supplied build divides a t/ha yield by 1000, so they do not compare with a run on the repaired stacks. A held prediction shows the library reproduces the author's shares from these rasters, apart from the predicted cells. It does not validate the method and carries no conclusion to the repaired stacks.

## Reproducibility state

Tests and the local controls run from a fresh clone off the cluster. The job has not been run off the cluster, the input manifests carry path, size and SHA-256 but not source or licence, and the reference for the job is the author's table named in `author_pins.csv`, which is not in this repository.

#!/bin/bash -l
#SBATCH --job-name=gep_ntfp
#SBATCH --partition=agsmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --output=gep_ntfp_%j.out
#SBATCH --error=gep_ntfp_%j.err
#
# Recompute the NTFP accessible-forest stage on the account's grid:
#
#     sbatch global_invest/ntfp/run_ntfp_msi.sbatch
#
# Machine-specific locations are es_parameters rows rather than values in here, so this file is
# the same on every machine and the CSV is the only thing a machine edits. It reads one of them,
# the conda environment, because the shell needs it before python exists to read anything.
#
# ⚠ WHY THIS RUNS AT ALL: the service moved off Mollweide at 300 m onto the account's grid,
# WGS84 at 10 arcsec, on 2026-09-02. Every raster the previous run wrote is on the old grid, so
# nothing is reusable and the country hectares -- the only thing the valuation reads -- have to
# come out of a fresh raster pass. The $14.07bn on the status row is the last Mollweide run's.
#
# Memory is a MEASUREMENT of the parts, not a guess at the whole. The reach growth was timed at
# 1.5 s and 0.3 GB for one band of 180 rows across the full 129,600-column width, and there are
# 360 bands, so that stage is about 9 minutes and never holds more than a gigabyte. The block
# zonal pass reads 512 rows of five rasters at a time, which is a few hundred megabytes. Nothing
# here reads a whole band: a single band on this grid is 8.4 billion cells. 64g is therefore
# generous rather than tight, and asking for more only costs queue time.
#
# ⚠⚠ DISK, not memory, is what this job actually needs. The intermediates it writes are about
# 84 GB uncompressed -- land cover and NDVI at 16.8 GB each as Int16, country ids at 33.6 GB as
# Int32, the burned lines and the reach at 8.4 GB each as Byte -- which DEFLATE takes to roughly
# 20 GB. ⚠ On MSI that means /scratch.global/$USER, not home: the flood cold starts write to
# ~/Files/global_invest/projects and that is fine for their size, but 84 GB of NTFP intermediates
# is not what a home directory is for. Set the project dir explicitly if home is where
# ProjectFlow would otherwise infer it.
#
# Time is a guess above the measured 9 minutes: the warps of a 3.1 GB NDVI and a 4.3 GB roads
# shapefile have not been timed on MSI, and they are the rest of the job. Set --time from the
# first log that finishes.
set -euo pipefail

CODE_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
PARAMETERS="$CODE_ROOT/global_invest/input_template/es_parameters.csv"

read_row () {
    python3 - "$PARAMETERS" "$1" <<'PY'
import csv, sys
path, key = sys.argv[1], sys.argv[2]
with open(path, encoding='utf-8-sig') as f:
    for row in csv.reader(f):
        if len(row) > 2 and row[1] == key:
            print(row[2]); break
    else:
        sys.exit('es_parameters has no row %s' % key)
PY
}

# ⚠ The environment can come from NTFP_CONDA_ENV instead of the CSV, and that is the better way
# on a machine that also pulls. es_parameters.csv is TRACKED, so filling a machine value into it
# leaves a local modification that every `git pull` has to be stashed around -- which is what
# flood's checkout on MSI carries today. Export the variable in the submitting shell and the
# tracked file stays clean:
#
#     NTFP_CONDA_ENV=/users/4/coles081/envs/earth_econ sbatch global_invest/ntfp/run_ntfp_msi.sbatch
#
# The CSV row stays as the fallback so the file is still the one place a machine CAN be recorded.
CONDA_ENV="${NTFP_CONDA_ENV:-$(read_row ntfp_msi_conda_env)}"
[ -n "$CONDA_ENV" ] || { echo "no environment: set NTFP_CONDA_ENV or fill the ntfp_msi_conda_env row"; exit 1; }
PYTHON="$CONDA_ENV/bin/python"
[ -x "$PYTHON" ] || { echo "no interpreter at $PYTHON"; exit 1; }

echo "[msi] code   : $CODE_ROOT"
echo "[msi] python : $PYTHON"
echo "[msi] started: $(date)"

# The environment interpreter directly, rather than `module load conda; conda activate`:
# conda activate needs a shell whose rc has been sourced, and a batch job does not have one, so
# the activate silently leaves the system python in place and the job dies on the first import.
#
# ⚠⚠ The environment's bin goes on PATH as well, which the interpreter alone does not do.
# `hb.make_path_pog` shells out to gdalinfo, and without it on PATH hazelbean's fallback builds
# `<env>/bin/bin/gdalinfo`, a doubled path, and the failure surfaces as
# `'int' object has no attribute 'info'` -- which says nothing about the cause.
#
# PYTHONPATH, because `python path/to/script.py` puts the SCRIPT's directory on sys.path and not
# the repository root, so `import global_invest` would find whichever copy is installed in the
# environment rather than this clone. This is what makes the run independent: it is this
# checkout that executes, whatever else is installed.
cd "$CODE_ROOT"
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
command -v gdalinfo >/dev/null || { echo "gdalinfo is not on PATH: make_path_pog cannot run"; exit 1; }
"$PYTHON" -c "import global_invest, os; print('[msi] package:', os.path.dirname(global_invest.__file__))"

# Preflight: every input this run reads, resolved through get_path before any work starts.
"$PYTHON" global_invest/ntfp/preflight_ntfp_inputs.py

# GDAL's block cache. ⚠ Kept MODEST on purpose: the line raster is 8.4 billion cells, and a cache
# large enough to hold a big share of it competes with the rasterizer for the same memory rather
# than helping. The warps are the only step that benefits from a big cache and they are not the
# expensive part here.
export GDAL_CACHEMAX=2048
# ⚠ GDAL_NUM_THREADS is deliberately NOT set. On a node whose hardware has 128 cores but whose
# cgroup allows 8, ALL_CPUS oversubscribes by 16x, and the full-resolution burn and warp came
# out EMPTY with exit 0 while the same calls succeed without it.

"$PYTHON" global_invest/ntfp/run_ntfp.py

echo "[msi] finished: $(date)"

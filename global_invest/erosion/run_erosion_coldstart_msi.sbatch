#!/bin/bash -l
#SBATCH --job-name=gep_erosion_cold
#SBATCH --partition=agsmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=36:00:00
#SBATCH --output=gep_erosion_cold_%j.out
#SBATCH --error=gep_erosion_cold_%j.err
#
# Erosion from an empty project directory, with invest_sdr skipped:
#
#     NTFP_CONDA_ENV=... sbatch global_invest/erosion/run_erosion_coldstart_msi.sbatch
#
# ⚠⚠ WHAT THIS SETTLES. Condition 10 for erosion. A cold start on 2026-09-02 that let `invest_sdr`
# run produced $12,099,377,565 against the published $18,240,930,373 -- a third of the total gone,
# clean exit, no error in the log. The cause was task precedence, not the valuation: `invest_sdr`
# reassigns `erosion_usle_path` to its own output whenever it runs, so the run stopped reading the
# author's staged `usle_2019_revised_feb_13.tif` and read its own SDR output, which reproduces his
# DECEMBER raster instead. Skipping the task leaves the staged rasters in place. If this run
# returns $18,240,930,373, condition 10 closes for erosion; if it does not, the gap is in the
# valuation and not the input, and that is worth knowing too.
#
# TIME. `prevention_shares` took 818 minutes on a laptop and 1057 on the run before it, so 36h is
# that plus room. It is the only expensive task here -- InVEST SDR, which dominates a full run, is
# the one being skipped.
#
# MEMORY. 64g. The prevention-share pass reads the 6217x4030 rasters and the SPAM stacks band by
# band rather than whole, and the completed laptop runs did it inside far less.
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

# EROSION_CONDA_ENV keeps the machine value out of the tracked CSV; the row is the fallback.
CONDA_ENV="${EROSION_CONDA_ENV:-${NTFP_CONDA_ENV:-$(read_row flood_msi_conda_env)}}"
[ -n "$CONDA_ENV" ] || { echo "no environment: set EROSION_CONDA_ENV"; exit 1; }
PYTHON="$CONDA_ENV/bin/python"
[ -x "$PYTHON" ] || { echo "no interpreter at $PYTHON"; exit 1; }

echo "[msi] code   : $CODE_ROOT"
echo "[msi] python : $PYTHON"
echo "[msi] started: $(date)"

cd "$CODE_ROOT"
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -c "import global_invest, os; print('[msi] package:', os.path.dirname(global_invest.__file__))"

# Preflight. It also ASSERTS that the SDR rasters resolve to base_data rather than to a project
# directory, which is the one thing that would silently turn this into the December answer again.
"$PYTHON" global_invest/erosion/preflight_erosion_inputs.py

export GDAL_CACHEMAX=8192
export GDAL_NUM_THREADS=ALL_CPUS

"$PYTHON" global_invest/erosion/run_erosion_coldstart.py

echo "[msi] finished: $(date)"

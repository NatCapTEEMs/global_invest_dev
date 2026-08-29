#!/bin/bash -l
#SBATCH --job-name=gep_flood
#SBATCH --partition=agsmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=400g
#SBATCH --time=24:00:00
#SBATCH --output=gep_flood_%j.out
#SBATCH --error=gep_flood_%j.err
#
# Submit the flood GEP run on MSI:
#
#     sbatch global_invest/flood/run_flood_msi.sbatch
#
# The machine-specific locations are es_parameters rows rather than values in here, so this file
# is the same on every machine and the CSV is the only thing a machine edits. It reads two of
# them, the conda environment and the code directory, because the shell needs those before python
# exists to read anything.
#
# Memory is measured rather than guessed: 64g was OOM-killed inside Section A within a minute
# (job 17488753). A global 300 m grid is about 7e9 cells, so a whole-array read is tens of GB and
# several at once exceeds 64. agsmall nodes carry 514g. If this is comfortable, trim it -- a
# smaller ask queues sooner.
#
# Time is still a guess: the valuation chain alone took 1h13m (job 17378863), and Section B and
# the 4A path have never run. Set --time from the first log that finishes.
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

CONDA_ENV="$(read_row flood_msi_conda_env)"
PYTHON="$CONDA_ENV/bin/python"
[ -x "$PYTHON" ] || { echo "no interpreter at $PYTHON"; exit 1; }

echo "[msi] code   : $CODE_ROOT"
echo "[msi] python : $PYTHON"
echo "[msi] started: $(date)"

# The environment interpreter directly, rather than `module load conda; conda activate`:
# conda activate needs a shell whose rc has been sourced, and a batch job does not have one,
# so the activate silently leaves the system python in place and the job dies on the first
# import.
# PYTHONPATH, because `python path/to/script.py` puts the SCRIPT's directory on sys.path and
# not the repository root -- so `import global_invest` finds whichever copy is installed in
# the environment rather than this clone. On MSI the installed one is a personal checkout
# with no flood directory, so the job died on the import. This is also what makes the run
# independent: it is this checkout that executes, whatever else is installed.
cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -c "import global_invest, os; print('[msi] package:', os.path.dirname(global_invest.__file__))"
"$PYTHON" global_invest/flood/run_flood.py

echo "[msi] finished: $(date)"

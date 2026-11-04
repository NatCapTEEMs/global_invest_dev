#!/bin/bash
#SBATCH --job-name=spam_stack_repair
#SBATCH --account=jajohns
#SBATCH --partition=agsmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Submit from the run folder that holds this file: sbatch build_spam_stacks.sbatch
# Checks that the code, parameters and input manifest are the pinned ones, then runs
# run_spam_stack_repair.py, which takes no arguments. Writes nothing outside this folder. Staging
# the stacks into base_data is a separate, later step. The environment is the one job 5347130
# used.
set -euo pipefail
module load gcc/11.3.0
run=$SLURM_SUBMIT_DIR
env=$HOME/envs/earth_econ
export PATH="$env/bin:$PATH" LD_LIBRARY_PATH="$env/lib:${LD_LIBRARY_PATH:-}"
export GDAL_DATA="$env/share/gdal" PROJ_LIB="$env/share/proj"
export GDAL_NUM_THREADS=1 OMP_NUM_THREADS=1
cd "$run"
# Rule 37: nothing that runs may exist only as uncommitted edits or an MSI copy. This file names the
# commit holding this builder, and exists only once that commit does.
test -s manifest/rule37_resolution.txt
test ! -e output && test ! -e output.partial && test ! -e check
sha256sum --check --strict manifest/pinned_sha256.txt
"$env/bin/python" -I -B -u run_spam_stack_repair.py
sha256sum output/*.tif output/*.csv output/*.txt check/*.csv > check/result_sha256.txt

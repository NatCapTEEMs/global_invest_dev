#!/bin/bash
#SBATCH --job-name=erosion_spam_repair_shares
#SBATCH --account=jajohns
#SBATCH --partition=agsmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Submit from the shares folder that holds this file, after the stack repair beside it has passed:
#   sbatch run_erosion_spam_repair_shares.sbatch
# Sized from job 5347130 (the same task, grid and SDR rasters): 9 h 42 min elapsed, 9 h 39 min
# of CPU, so one core busy, and 25.8 GB peak memory. 48 GB is 1.9 times that peak and 24 h 2.5 times
# that time. heartbeat.tsv records the run's elapsed and CPU seconds and resident memory every 10
# minutes, because the task logs only a few lines in ten hours: check it with
# run_shares_heartbeat_check.py. Writes nothing outside this folder.
set -euo pipefail
module load gcc/11.3.0
run=$SLURM_SUBMIT_DIR
env=$HOME/envs/earth_econ
export PATH="$env/bin:$PATH" LD_LIBRARY_PATH="$env/lib:${LD_LIBRARY_PATH:-}"
export GDAL_DATA="$env/share/gdal" PROJ_LIB="$env/share/proj"
export GDAL_NUM_THREADS=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HB_TEMP_DIR="$run/tmp" TMPDIR="$run/tmp" MPLBACKEND=Agg MPLCONFIGDIR="$run/tmp/matplotlib"
cd "$run"
# Rule 37: nothing that runs may exist only as uncommitted edits or an MSI copy. The library is a pushed
# commit. This file names the commit holding the stack builder and this launch code, and exists only
# once that commit does.
test -s ../manifest/rule37_resolution.txt
test ! -e project && test ! -e result && test ! -e heartbeat.tsv
sha256sum --check --strict manifest/pinned_sha256.txt
mkdir -p tmp/matplotlib
"$env/bin/python" -I -B -u run_erosion_spam_repair_shares.py &
pid=$!
(
    printf 'utc\telapsed_s\tcpu_s\trss_kb\n' > heartbeat.tsv
    while kill -0 "$pid" 2>/dev/null; do
        printf '%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(ps -o etimes=,times=,rss= -p "$pid" | awk '{print $1"\t"$2"\t"$3}')" >> heartbeat.tsv
        sleep 600
    done
) &
wait "$pid"

#!/bin/bash
#SBATCH --job-name=erosion_eps5_library_shares
#SBATCH --account=jajohns
#SBATCH --partition=agsmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Submit from the folder that holds this file: sbatch run_eps5_shares.sbatch
# Sized from job 5347130, the library's SES-11 run of the same task on the same inputs: 9 h 42 min
# elapsed, one core busy, 25.8 GB peak memory. The author's EPS-5 run took 1.7 percent longer than the
# author's SES-11 run. 48 GB is 1.9 times that peak and 24 h is about 2.4 times the expected time.
# heartbeat.tsv records elapsed and CPU seconds and resident memory every 10 minutes, because the task
# logs little: check it with run_eps5_heartbeat_check.py. Writes nothing outside this folder.
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
# commit. This file names the commit holding this launch code, and exists only once that commit does.
test -s manifest/rule37_resolution.txt
# The staged rasters fail the identity gate against the author's, and that stays on the record. This file
# exists only once a person has decided to run against the prediction it names by SHA-256.
test -s manifest/raster_gate_decision.txt
grep -q "$(sha256sum prediction/prediction_sha256.txt | cut -c1-64)" manifest/raster_gate_decision.txt
test ! -e project && test ! -e result && test ! -e heartbeat.tsv
sha256sum --check --strict manifest/pinned_sha256.txt
mkdir -p tmp/matplotlib
"$env/bin/python" -I -B -u run_eps5_shares.py &
pid=$!
(
    printf 'utc\telapsed_s\tcpu_s\trss_kb\n' > heartbeat.tsv
    while kill -0 "$pid" 2>/dev/null; do
        printf '%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(ps -o etimes=,times=,rss= -p "$pid" | awk '{print $1"\t"$2"\t"$3}')" >> heartbeat.tsv
        sleep 600
    done
) &
wait "$pid"

#!/bin/bash -l        
#SBATCH --time=0:30:00
#SBATCH --account=jajohns
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60g
#SBATCH --tmp=60g
#SBATCH --job-name=ndr1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=salmamun@d.umn.edu
#SBATCH -p msismall

SECONDS=0

conda activate nci

python /users/5/salmamun/GEP/gep/ndr/model_runs.py "/users/5/salmamun/GEP/gep/ndr/gep_sm_msi.yaml"
wait

duration=$SECONDS
echo "==================================================================="
echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed."
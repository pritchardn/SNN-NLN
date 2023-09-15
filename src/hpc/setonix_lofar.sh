#!/bin/bash
#SBATCH --job-name=SNN-NLN
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=snn_%A_%a.out
#SBATCH --error=snn_%A_%a.err
#SBATCH --array=0-0
#SBATCH --partition=work
#SBATCH --account=pawsey0411

module load python/3.10.10

cd /software/projects/pawsey0411/npritchard/setonix/2023.08/python/SNN-NLN/src
source /software/projects/pawsey0411/npritchard/setonix/2023.08/python/snn-nln/bin/activate


export NUM_TRIALS=1
export TASK_TYPE="STANDARD"
export OUTPUT_DIR="/scratch/pawsey0411/npritchard/outputs"
export DATA_DIR="/scratch/pawsey0411/npritchard/data"
export INPUT_DIR="/scratch/pawsey0411/npritchard/data"
export DATASET="LOFAR"

srun -N 1 -n 1 -c 32 python3 main_hpc.py

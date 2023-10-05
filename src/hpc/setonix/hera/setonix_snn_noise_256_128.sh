#!/bin/bash
#SBATCH --job-name=SNN-NLN
#SBATCH --nodes=1
#SBATCH --mem=230GB
#SBATCH --time=1:00:00
#SBATCH --output=snn_%A_%a.out
#SBATCH --error=snn_%A_%a.err
#SBATCH --array=0-49
#SBATCH --partition=work
#SBATCH --account=pawsey0411


module load python/3.10.10

cd /software/projects/pawsey0411/npritchard/setonix/2023.08/python/SNN-NLN/src
source /software/projects/pawsey0411/npritchard/setonix/2023.08/python/snn-nln/bin/activate

export NUM_TRIALS=1
export TASK_TYPE="SNN"
export MODEL_TYPE="DAE-NOISE"
export OUTPUT_DIR="/scratch/pawsey0411/npritchard/outputs/hera/noise"
export DATA_DIR="/scratch/pawsey0411/npritchard/data"
export INPUT_DIR="/scratch/pawsey0411/npritchard/data"
export DATASET="HERA"
export TIME_LENGTH=256
export AVERAGE_N=128

srun -N 1 -n 1 -c 32 python3 main_hpc.py
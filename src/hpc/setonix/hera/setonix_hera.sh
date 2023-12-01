#!/bin/bash
#SBATCH --job-name=SNN-NLN
#SBATCH --nodes=1
#SBATCH --time=0:30:00
#SBATCH --output=snn_%A_%a.out
#SBATCH --error=snn_%A_%a.err
#SBATCH --array=0-0
#SBATCH --partition=gpu-dev
#SBATCH --gpus-per-node=1
#SBATCH --account=pawsey0411-gpu


module load python/3.10.10
module load rocm

cd /software/projects/pawsey0411/npritchard/setonix/2023.08/python/SNN-NLN/src
source /software/projects/pawsey0411/npritchard/setonix/2023.08/python/snn-nln/bin/activate

export OMP_NUM_THREADS=1
export NUM_TRIALS=1
export TASK_TYPE="STANDARD"
export OUTPUT_DIR="/scratch/pawsey0411/npritchard/outputs/hera"
export DATA_DIR="/scratch/pawsey0411/npritchard/data"
export INPUT_DIR="/scratch/pawsey0411/npritchard/data"
export DATASET="HERA"

srun -N 1 -n 1 -c 8 --gpus-per-node=1 --gpus-per-task=1 python3 main_hpc.py

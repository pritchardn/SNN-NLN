#!/bin/bash
#SBATCH --job-name=SNN-NLN
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=snn_%A_%a.out
#SBATCH --error=snn_%A_%a.err
#SBATCH --array=0-0
#SBATCH --gres=gpu
#SBATCH --accounts=pawsey0411-gpu

module load python/3.8.12

cd /home/npritchard/SNN-NLN
source /home/npritchard/SNN-NLN/snn-nln/bin/activate

export NUM_TRIALS=1
export TASK_TYPE="STANDARD"
export OUTPUT_DIR="/scratch/pawsey0411/npritchard/outputs"
export DATA_DIR="/scratch/pawsey0411/npritchard/data"
export INPUT_DIR="/scratch/pawsey0411/npritchard/data"
export DATASET="LOFAR"

srun -N 1 -n 1 -c 8 --gpus-per-node=1 --gpus-per-task=1 python3 main_hpc.py

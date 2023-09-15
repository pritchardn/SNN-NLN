#!/bin/bash
#SBATCH --job-name=SNN-NLN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256GB
#SBATCH --time=0:15:00
#SBATCH --output=snn_%A_%a.out
#SBATCH --error=snn_%A_%a.err
#SBATCH --array=0-49
#SBATCH --gres=gpu

module load python/3.8.12

cd /home/npritchard/SNN-NLN
source /home/npritchard/SNN-NLN/snn-nln/bin/activate

export NUM_TRIALS=10
export TASK_TYPE="NOISE"
export OUTPUT_DIR="/scratch/npritchard/outputs"
export DATA_DIR="/scratch/npritchard/data"
export INPUT_DIR="/scratch/npritchard/data"
export DATASET="HERA"

python3 main_hpc.py

#!/bin/bash
#SBATCH --job-name=SNN-NLN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128GB
#SBATCH --time=168:00:00
#SBATCH --output=snn_%A_%a.out
#SBATCH --error=snn_%A_%a.err
#SBATCH --array=0-0%1
#SBATCH --gres=gpu:1

cd /home/npritchard/SNN-NLN/src
source /home/npritchard/snn-nln-a100/bin/activate

export NUM_TRIALS=32
export TASK_TYPE="OPTUNA"
export OUTPUT_DIR="/home/npritchard/outputs/optuna/tabascal"
export DATA_DIR="/home/npritchard/data"
export INPUT_DIR="/home/npritchard/data"
export DATASET="LOFAR"
export MODEL_TYPE="SDDAE"

python3 main_hpc.py



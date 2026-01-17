#!/bin/bash

#SBATCH --job-name=finetune_llava_valeo
#SBATCH --account=EU-25-10
#SBATCH --time=40:00:00 # Time limit for running
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dmytro.khursenko@valeo.com
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Load necessary modules (if applicable)	
source /mnt/proj1/eu-25-10/envs/roadcap-gen/bin/activate
cd /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/

# Define variables
MODEL="llava"
# DATASET="vqa_valeo_karolina"
DATASET="drivelm"
TRAINING="full_finetune"

# ----- CHANGE EXPERIMENT NAME FOR EACH RUN -----
# experiment_name="karolina_srun_llava_fullfinetune_MMP_Valeo_3.1"
experiment_name="karolina_srun_llava_fullfinetune_MMP_DriveLM-data_usage-0.15"

# -----------------------------------------------

# Run the command
srun python scripts/02_finetuning/train.py \
    model=$MODEL     \
    dataset=$DATASET     \
    training=$TRAINING     \
    experiment_name=$experiment_name     \



# sbatch scripts/02_finetuning/karolina_train.sh
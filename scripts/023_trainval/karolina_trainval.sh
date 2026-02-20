#!/bin/bash

#SBATCH --job-name=trainval_llava_drivelm
#SBATCH --account=open-36-7
#SBATCH --time=30:00:00 # Time limit for running
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dmytro.khursenko@valeo.com
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Load necessary modules (if applicable)	
source /mnt/proj1/eu-25-10/envs/roadcap-gen/bin/activate
cd /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/
ml Java/1.8.0_221

# Define variables
GPUS=8
MODEL="llava"
DATASET="drivelm"
# DATASET="vqa_valeo_karolina"

# DATASET="roadvqa"

TRAINING="full_finetune"

# ----- CHANGE EXPERIMENT NAME FOR EACH RUN -----
# experiment_name="karolina_srun_llava_fullfinetune_MMP_Valeo_3.1"
# experiment_name="karolina_srun_llava_fullfinetune_MMP_DriveLM-data_usage-0.15"


# experiment_name="latest_karolina_torchrun_train_llava_fullfinetune_MMP_DriveLM-data_usage-1.0_gpus8_bs4"
# experiment_name="latest_karolina_torchrun_train_llava_fullfinetune_MMP_Valeo-data_usage-1.0_gpu4_bs4"


# experiment_name="karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-data_usage-0.03_gpu8_bs4"
# experiment_name="karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-data_usage-0.3_gpu8_bs4"
# experiment_name="karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-data_usage-0.2_gpu8_bs4"
# experiment_name="karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-data_usage-0.05_gpu8_bs4"


experiment_name="karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-data_usage-1.0_gpu8_bs4"
 
# -----------------------------------------------

# Run the command

# # one gpu run
# srun python scripts/02_finetuning/train.py \
#     model=$MODEL     \
#     dataset=$DATASET     \
#     training=$TRAINING     \
#     experiment_name=$experiment_name     \

# multi gpu run
PYTHONPATH=. torchrun --nproc_per_node=$GPUS scripts/023_trainval_exp/trainval.py \
    model=$MODEL     \
    dataset=$DATASET     \
    training=$TRAINING     \
    experiment_name=$experiment_name     \


# sbatch scripts/012_trainval_exp/karolina_trainval.sh
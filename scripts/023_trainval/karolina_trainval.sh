#!/bin/bash

#SBATCH --job-name=trainval_llava_drivelm
#SBATCH --account=OPEN-36-38
#SBATCH --time=40:00:00 # Time limit for running
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
# TRAINING="full_finetune_cps_offline_best"
# TRAINING="full_finetune_nju_online_best"


# ----- CHANGE EXPERIMENT NAME FOR EACH RUN -----
# experiment_name="karolina_srun_llava_fullfinetune_MMP_Valeo_3.1"
# experiment_name="karolina_srun_llava_fullfinetune_MMP_DriveLM-data_usage-0.15"


# experiment_name="latest_karolina_torchrun_train_llava_fullfinetune_MMP_DriveLM-data_usage-1.0_gpus8_bs4"
# experiment_name="latest_karolina_torchrun_train_llava_fullfinetune_MMP_Valeo-data_usage-1.0_gpu4_bs4"


# experiment_name="tmp_karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-data_usage-0.03_gpu2_bs4"
# experiment_name="karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-data_usage-0.3_gpu8_bs4"
# experiment_name="karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-data_usage-0.2_gpu8_bs4"
# experiment_name="karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-data_usage-0.05_gpu8_bs4"


# experiment_name="karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-train1.0-all-data_data_usage-1.0_gpu8_bs4_260226"

# 113026 - TODO: check outputs!!!
# experiment_name="karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-v1_2_combined-all-data_val_data_usage-1.0_gpu8_bs4_110326"
# experiment_name="karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-v1_2_combined-all-data_val_data_usage-1.0_gpu8_bs4_CPSargs_110326"
# experiment_name="karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-v1_2_combined-all-data_val_data_usage-1.0_gpu8_bs4_NJUargs_110326"


# 180326
# experiment_name="karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-v2_0_trainaug-all-data_val_data_usage-1.0_gpu8_bs4_180326"
# experiment_name="karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-v2_0_aug-all-data_val_data_usage-1.0_gpu8_bs4_180326"
# experiment_name="karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-v1_2_train-all-data_val_data_usage-1.0_gpu8_bs4_180326"

experiment_name="karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-v2_0_valaug-all-data_val_data_usage-1.0_gpu8_bs4_180326"


 
# -----------------------------------------------

# Run the command

# # one gpu run
# srun python scripts/02_finetuning/train.py \
#     model=$MODEL     \
#     dataset=$DATASET     \
#     training=$TRAINING     \
#     experiment_name=$experiment_name     \

# export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG=INFO

# multi gpu run
PYTHONPATH=. torchrun --nproc_per_node=$GPUS scripts/023_trainval/trainval.py \
    model=$MODEL     \
    dataset=$DATASET     \
    training=$TRAINING     \
    experiment_name=$experiment_name     \
    +jobid=$SLURM_JOB_ID \


# sbatch scripts/023_trainval/karolina_trainval.sh
#!/bin/bash

#SBATCH --job-name=trainval_llava_lora_drivelm
#SBATCH --account=OPEN-36-38
#SBATCH --time=47:00:00 # Time limit for running
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

# TRAINING="full_finetune"
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

# experiment_name="karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-v2_0_valaug-all-data_val_data_usage-1.0_gpu8_bs4_180326"

# next to eval
# 250326
# experiment_name="karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-v1_2_val-all-data_val_data_usage-0.1_gpu8_bs4_250326"
# experiment_name="karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-v2_0_fromtrain300_85k-all_data_gpu8_bs4_250326"
# experiment_name="karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-v2_0_train85kaug-all-data_gpu8_bs4_250326"

# next to eval
# 250326
TRAINING="lora_llm" 
# experiment_name="train_llava_lora_llm_r16_DriveLM-v1_2_val-all-data_val_data_usage-0.1_gpu8_bs4"
# experiment_name="train_llava_lora_llm_r16_MMPft_DriveLM-v1_2_val-all-data_val_data_usage-0.1_gpu8_bs4"
# experiment_name="train_llava_lora_vit_llm_r16_MMPft_DriveLM-v1_2_val-all-data_val_data_usage-0.1_gpu8_bs4"

# experiment_name="train_llava_lora_vit_llm_r16_MMPft_DriveLM-v2_0_train85kaug_qas_Qwen3-14B_nuscenes_think_no-tracks_dynamic_q_gpu8_bs4"
# experiment_name="train_llava_lora_llm_r16_DriveLM-v2_0_train85kaug_qas_Qwen3-14B_nuscenes_think_no-tracks_dynamic_q_gpu8_bs4"

# 070426
# experiment_name="train_llava_lora_llm_r16_MMPft_DriveLM-v2_0_aug33k_qas_Qwen3-14B_nuscenes_think_no-tracks_dynamic_q_gpu8_bs4"
# experiment_name="train_llava_lora_llm_r16_MMPft_DriveLM-v2_0_aug287k_qas_Qwen3-14B_nuscenes_think_rel-tracks_dynamic_q_gpu8_bs4"
experiment_name="train_llava_lora_llm_r16_MMPft_DriveLM-v2_0_aug88k_qas_Qwen3-14B_nuscenes_think_global-tracks_dynamic_q_gpu8_bs4"




 
# -----------------------------------------------
# Run the command
# -----------------------------------------------
OUTPUT_DIR="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/llava-v1.6-mistral-7b/${experiment_name}"
mkdir -p "$OUTPUT_DIR"

# Define the log file path inside that checkpoint directory
LOG_FILE="${OUTPUT_DIR}/training_debug_${SLURM_JOB_ID}.log"

echo "========================================================"
echo "🚀 Starting multi-GPU training..."
echo "📂 Checkpoints and full debug logs will be saved to:"
echo "   $OUTPUT_DIR"
echo "========================================================"

# multi gpu run
PYTHONPATH=. torchrun --nproc_per_node=$GPUS scripts/023_trainval/trainval.py \
    model=$MODEL \
    dataset=$DATASET \
    training=$TRAINING \
    mode=simple \
    experiment_name=$experiment_name \
    +jobid=$SLURM_JOB_ID 2> >(tee -a "$LOG_FILE" >&2) | tee -a "$LOG_FILE"

echo "🎉 Training script finished. Logs saved to $LOG_FILE"

# sbatch scripts/023_trainval/karolina_trainval.sh
#!/bin/bash

#SBATCH --job-name=trainval_llava_lora_drivelm_depth
#SBATCH --account=open-36-38
#SBATCH --time=47:00:00
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /mnt/proj1/eu-25-10/envs/roadcap-gen/bin/activate
cd /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/
ml Java/1.8.0_221

# export WANDB_MODE=disabled
# -----------------------------------------------
# Config
# -----------------------------------------------
GPUS=8
MODEL="llava"
DATASET="drivelm"
TRAINING="lora_llm_with_depth"

# 070426
# experiment_name="trainval_llava_lora_llm_r16_DriveLM-v2_0_train85kaug_depth_distill_gpu8_bs4_fixed"
# experiment_name="train_llava_lora_llm_r16_MMPft_depth_distill_DriveLM-v2_0_aug33k_qas_Qwen3-14B_nuscenes_think_no-tracks_dynamic_q_gpu8_bs4"
experiment_name="train_llava_lora_llm_r16_MMPft_depth_distill_DriveLM-v2_0_aug287k_qas_Qwen3-14B_nuscenes_think_rel-tracks_dynamic_q_gpu8_bs4_fixed"
# experiment_name="train_llava_lora_llm_r16_MMPft_depth_distill_DriveLM-v2_0_aug88k_qas_Qwen3-14B_nuscenes_think_global-tracks_dynamic_q_gpu8_bs4"

# -----------------------------------------------
# Run
# -----------------------------------------------
OUTPUT_DIR="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/llava-v1.6-mistral-7b/${experiment_name}"
mkdir -p "$OUTPUT_DIR"

LOG_FILE="${OUTPUT_DIR}/training_debug_${SLURM_JOB_ID}.log"

echo "========================================================"
echo "Starting train+val with depth distillation loss..."
echo "Checkpoints will be saved to: $OUTPUT_DIR"
echo "========================================================"

# Kernel 4.18 fix: disable NCCL P2P and IB to prevent the first collective from hanging.
# P2P (NVLink/PCIe direct) and IB both require kernel ≥ 5.5 semaphore fixes.
# Disabling them forces NCCL to use shared-memory (intra-node) and socket (inter-node),
# which are slower but work correctly on old kernels.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Crash loudly if a rank hangs for more than 10 minutes instead of waiting forever.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
# Reduce memory fragmentation: allows the allocator to extend segments rather than
# allocating new ones, which avoids the "58 MiB free but can't allocate 228 MiB" OOM.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Single-GPU: use plain python to avoid NCCL/DDP init which deadlocks on kernel 4.18
# torchrun initialises NCCL even for nproc=1; the first collective then hangs on old kernels.
# PYTHONPATH=. python /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/scripts/023_trainval/trainval.py \

PYTHONPATH=. torchrun --nproc_per_node=$GPUS /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/scripts/023_trainval/trainval.py \
    model=$MODEL \
    dataset=$DATASET \
    training=$TRAINING \
    mode=extended \
    experiment_name=$experiment_name \
    +jobid=$SLURM_JOB_ID 2> >(tee -a "$LOG_FILE" >&2) | tee -a "$LOG_FILE"

echo "Done. Logs saved to $LOG_FILE"

# cd /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen
# sbatch scripts/023_trainval/karolina_trainval_with_depth.sh
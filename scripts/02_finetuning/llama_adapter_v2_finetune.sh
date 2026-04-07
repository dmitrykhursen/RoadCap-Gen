#!/bin/bash
#SBATCH --job-name=train_llama_adapter_v2
#SBATCH --account=OPEN-36-38
#SBATCH --time=47:00:00
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dmytro.khursenko@valeo.com
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# --- 1. ENVIRONMENT SETUP ---
source /mnt/proj1/eu-25-10/envs/roadcap-gen/bin/activate

# Navigate to where main_finetune.py and finetune.sh are located
cd /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/external/DriveLM/challenge/llama_adapter_v2_multimodal7b

# --- 2. ARGUMENTS ---
LLAMA_PATH="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/external/DriveLM/challenge/llama_adapter_v2_multimodal7b/ckpts/llama_model_weights"

# ARG 2: Path to the pre-trained LLaMA-Adapter checkpoint you downloaded
PRETRAINED_PATH="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/external/DriveLM/challenge/llama_adapter_v2_multimodal7b/ckpts/1bcbffc43484332672092e0024a8699a6eb5f558161aebf98a7c6b1db67224d1_LORA-BIAS-7B.pth"


# ARG 3: Path to the training data configuration file (usually a YAML or JSON provided by the challenge)
CONFIG="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/external/DriveLM/challenge/llama_adapter_v2_multimodal7b/finetune_data_config.yaml"

# ARG 4: Where to save the fine-tuned weights and logs
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
# OUTPUT_DIR="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/llama_adapter_v2_finetune_${TIMESTAMP}_v2_0_aug_qas_Qwen3-14B_nuscenes_think_no-tracks_dynamic_q_converted"
# OUTPUT_DIR="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/llama_adapter_v2_finetune_${TIMESTAMP}_v2_0_aug_qas_Qwen3-14B_nuscenes_think_no-tracks_dynamic_q_converted"

#OUTPUT_DIR="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/llama_adapter_v2_finetune_${TIMESTAMP}_v1_2_val0.1"
OUTPUT_DIR="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/llama_adapter_v2_finetune_${TIMESTAMP}_v2_0_train85kaug_qas_Qwen3-14B_nuscenes_think_no-tracks_dynamic_q_converted"


mkdir -p "$OUTPUT_DIR"
echo "🚀 Starting LLaMA-Adapter v2 Fine-Tuning..."
echo "📂 Saving checkpoints and logs to: $OUTPUT_DIR"

# --- 3. EXECUTION ---
# Call the bash script with the 4 required arguments
bash exps/finetune.sh \
    "$LLAMA_PATH" \
    "$PRETRAINED_PATH" \
    "$CONFIG" \
    "$OUTPUT_DIR"


echo "🎉 Fine-tuning job completed!"

# cd /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/
# sbatch scripts/02_finetuning/llama_adapter_v2_finetune.sh
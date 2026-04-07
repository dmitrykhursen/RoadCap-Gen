#!/bin/bash
#SBATCH --job-name=infer_val_llava_drivelm
#SBATCH --account=OPEN-36-38
#SBATCH --time=10:30:00
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dmytro.khursenko@valeo.com
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

GPUS=8
MODEL="llava"
DATASET="drivelm_v1_2_val0.1"
INFERENCE="drivelm_infer"
LOAD_MODE="finetune"
DATASET_PATH="/mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/v1_2_val0.1_converted_llama_with_tags.json"
# EVAL_DATASET_PATH="/mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/v1_2_val0.1.json"
EVAL_DATASET_PATH="/mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/v1_2_val0.1_converted_llama_with_tags.json"

WANDB_PROJECT="RoadCap-Gen" 

# FIXING NOW
# MODEL_BASE_DIR="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/llava-v1.6-mistral-7b/karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-v1_2_combined-all-data_val_data_usage-1.0_gpu8_bs4_110326"
# WANDB_RUN_ID="od1f9295"
# JOB_TIMESTAMP=2026-03-18_19-09-52

# not to fix, finished training
# MODEL_BASE_DIR="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/llava-v1.6-mistral-7b/karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-v1_2_combined-all-data_val_data_usage-1.0_gpu8_bs4_CPSargs_110326"
# WANDB_RUN_ID="f9q01hlm"

# not to fix, finished training
# MODEL_BASE_DIR="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/llava-v1.6-mistral-7b/karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-v1_2_combined-all-data_val_data_usage-1.0_gpu8_bs4_NJUargs_110326"
# WANDB_RUN_ID="3l8wnnq6"

# FIXING NOW
# MODEL_BASE_DIR="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/llava-v1.6-mistral-7b/karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-v2_0_aug-all-data_val_data_usage-1.0_gpu8_bs4_180326"
# WANDB_RUN_ID="bz5s4xpt"
# JOB_TIMESTAMP=2026-03-18_17-59-37

# FIXING NOW
# MODEL_BASE_DIR="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/llava-v1.6-mistral-7b/karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-v2_0_valaug-all-data_val_data_usage-1.0_gpu8_bs4_180326"
# WANDB_RUN_ID="v5toiuv9"
# JOB_TIMESTAMP=2026-03-18_18-00-20

# FIXING NOW
# MODEL_BASE_DIR="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/llava-v1.6-mistral-7b/karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-v1_2_train-all-data_val_data_usage-1.0_gpu8_bs4_180326"
# WANDB_RUN_ID="koe24h7y"
# JOB_TIMESTAMP=2026-03-20_04-21-13


# FIXING NOW
# MODEL_BASE_DIR="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/llava-v1.6-mistral-7b/karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-v2_0_trainaug-all-data_val_data_usage-1.0_gpu8_bs4_180326"
# WANDB_RUN_ID="a7gevtvh"
# JOB_TIMESTAMP=2026-03-20_04-23-12


# new report results
# MODEL_BASE_DIR="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/llava-v1.6-mistral-7b/karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-v1_2_val-all-data_val_data_usage-0.1_gpu8_bs4_250326"
# WANDB_RUN_ID="n3dfs4uc"
# JOB_TIMESTAMP=2026-03-25_22-11-46

# new report results
# MODEL_BASE_DIR="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/llava-v1.6-mistral-7b/karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-v1_2_val-all-data_val_data_usage-0.1_gpu8_bs4_250326"
# WANDB_RUN_ID="n3dfs4uc"
# JOB_TIMESTAMP=2026-03-25_22-11-46

# # new report results
# MODEL_BASE_DIR="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/llava-v1.6-mistral-7b/train_llava_lora_llm_r16_DriveLM-v1_2_val-all-data_val_data_usage-0.1_gpu8_bs4"
# WANDB_RUN_ID="fahroovz"

# new report results
# MODEL_BASE_DIR="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/llava-v1.6-mistral-7b/train_llava_lora_llm_r16_MMPft_DriveLM-v1_2_val-all-data_val_data_usage-0.1_gpu8_bs4"
# WANDB_RUN_ID="z4tqp9hr"

# # new report results
# MODEL_BASE_DIR="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/llava-v1.6-mistral-7b/train_llava_lora_vit_llm_r16_MMPft_DriveLM-v1_2_val-all-data_val_data_usage-0.1_gpu8_bs4"
# WANDB_RUN_ID="d4nw8j1s"

# new report results (but also still training!!!)
# MODEL_BASE_DIR="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/llava-v1.6-mistral-7b/train_llava_lora_llm_r16_DriveLM-v2_0_train85kaug_qas_Qwen3-14B_nuscenes_think_no-tracks_dynamic_q_gpu8_bs4"
# WANDB_RUN_ID="2jb8ri0e"

# # new report results (but also still training!!!)
MODEL_BASE_DIR="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/llava-v1.6-mistral-7b/train_llava_lora_vit_llm_r16_MMPft_DriveLM-v2_0_train85kaug_qas_Qwen3-14B_nuscenes_think_no-tracks_dynamic_q_gpu8_bs4"
WANDB_RUN_ID="g2pwnrn7"

MODEL_DIR_NAME=$(basename "$MODEL_BASE_DIR")
WANDB_RUN_NAME="${MODEL_DIR_NAME}_${WANDB_RUN_ID}"
JOB_TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Define the master experiment name and folder path
MASTER_EXP_NAME="infer_${MODEL}_${DATASET}_${LOAD_MODE}_llava-hf_llava-v1.6-mistral-7b-hf_gpu${GPUS}_${WANDB_RUN_NAME}_${JOB_TIMESTAMP}"
MASTER_INFER_DIR="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/inference/${MASTER_EXP_NAME}"


# MASTER_INFER_DIR="infer_llava_drivelm_v1_2_val0.1_finetune_llava-hf_llava-v1.6-mistral-7b-hf_gpu8_karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-v1_2_val-all-data_val_data_usage-0.1_gpu8_bs4_250326_n3dfs4uc_2026-03-25_22-11-46"


# Create the master folder
mkdir -p "$MASTER_INFER_DIR"
echo "📂 Grouping all checkpoints for this run under: $MASTER_INFER_DIR"
# ==========================================

CHECKPOINTS=$(ls -d ${MODEL_BASE_DIR}/checkpoint-* 2>/dev/null | sort -V)

if [ -z "$CHECKPOINTS" ]; then
    echo "❌ No checkpoints found in $MODEL_BASE_DIR"
    exit 1
fi

for MODEL_CKPT in $CHECKPOINTS; do
    STEP=$(basename "$MODEL_CKPT" | sed 's/checkpoint-//g')
    CKPT_SHORT="ckpt${STEP}"
    
    # Define the specific subfolder for this checkpoint INSIDE the master folder
    CKPT_INFER_DIR="${MASTER_INFER_DIR}/${CKPT_SHORT}"
    mkdir -p "$CKPT_INFER_DIR"
    
    LOG_FILE="${CKPT_INFER_DIR}/process_step_${STEP}.log"

    echo "========================================================" | tee -a "$LOG_FILE"
    echo "🚀 Processing Checkpoint: $CKPT_SHORT (Global Step: $STEP)" | tee -a "$LOG_FILE"
    echo "📂 Output Directory: $CKPT_INFER_DIR" | tee -a "$LOG_FILE"
    echo "========================================================" | tee -a "$LOG_FILE"
    
    # --- 1. INFERENCE ---
    source /mnt/proj1/eu-25-10/envs/roadcap-gen/bin/activate
    cd /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/
    
    # Check if prediction file already exists
    PRED_FILE=$(ls -t ${CKPT_INFER_DIR}/drivelm_pred_for_HF_*.json 2>/dev/null | head -n 1)

    if [ -z "$PRED_FILE" ]; then
        echo "⚙️  Running Generation..." | tee -a "$LOG_FILE"
        export MASTER_PORT=$(expr 10000 + $RANDOM % 1000)
        
        # NOTE: We are passing the combined path into experiment_name
        PYTHONPATH="." torchrun --nproc_per_node=$GPUS --master_port=$MASTER_PORT scripts/04_inference/ddp_inference.py \
            model=$MODEL \
            dataset=$DATASET \
            inference=$INFERENCE \
            experiment_name="${MASTER_EXP_NAME}/${CKPT_SHORT}" \
            inference.load_mode=$LOAD_MODE \
            inference.checkpoint_model=$MODEL_CKPT \
            dataset.data_path=$DATASET_PATH \
            inference.tokenizer_path=$MODEL_CKPT 2> >(tee -a "$LOG_FILE" >&2) | tee -a "$LOG_FILE"
            
        # Find the newly generated file inside our nested folder
        # PRED_FILE=$(ls -t output/inference/${MASTER_EXP_NAME}/${CKPT_SHORT}*/drivelm_pred_for_HF_*.json 2>/dev/null | head -n 1)
        PRED_FILE="${CKPT_INFER_DIR}/drivelm_pred_for_HF_test_qas.json"
    else
        echo "✅ Inference JSON already exists, skipping generation..." | tee -a "$LOG_FILE"
    fi

    # Ensure we actually found a file before evaluating
    if [ -z "$PRED_FILE" ]; then
        echo "❌ Error: Inference failed to generate a prediction file for $CKPT_SHORT." | tee -a "$LOG_FILE"
        continue
    fi

    # --- 2. EVALUATION ---
    deactivate
    source /mnt/proj1/eu-25-10/envs/drivelm/bin/activate
    ml Java
    
    echo "📊 Evaluating $CKPT_SHORT and syncing to WandB ID: $WANDB_RUN_ID..." | tee -a "$LOG_FILE"
    
    PYTHONPATH="." python external/DriveLM/challenge/evaluation.py \
        --root_path1="$PRED_FILE" \
        --root_path2="$EVAL_DATASET_PATH" \
        --step=$STEP \
        --model_name="${MODEL} Step ${STEP}" \
        --latex_out="$CKPT_INFER_DIR/latex_results.txt" \
        --wandb_project="$WANDB_PROJECT" \
        --wandb_run_name="$WANDB_RUN_ID" 2> >(tee -a "$LOG_FILE" >&2) | tee -a "$LOG_FILE"
        # --wandb_run_id="$WANDB_RUN_ID" 2> >(tee -a "$LOG_FILE" >&2) | tee -a "$LOG_FILE"
        
    echo "✅ Checkpoint $STEP complete!" | tee -a "$LOG_FILE"
done

echo "🎉 Pipeline finished successfully!"

# sbatch scripts/03_evaluation/infer_eval.sh

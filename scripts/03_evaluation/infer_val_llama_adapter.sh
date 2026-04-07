#!/bin/bash
#SBATCH --job-name=infer_val_llama_adapter_v2
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
MODEL="llama_adapter_v2"
DATASET="drivelm_v1_2_val0.1"
INFERENCE="llama_adapter_v2_infer"
LOAD_MODE="adapter"
# EVAL_DATASET_PATH="/mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/v1_2_val0.1.json"
EVAL_DATASET_PATH="/mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/v1_2_val0.1_converted_llama_with_tags.json"

# --- WANDB SETUP 
WANDB_PROJECT="RoadCap-Gen" 

# 250326
# WANDB_RUN_NAME="val_llama_adapter_v2_baseline"
# MODEL_BASE_DIR="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/external/DriveLM/challenge/llama_adapter_v2_multimodal7b/ckpts/1bcbffc43484332672092e0024a8699a6eb5f558161aebf98a7c6b1db67224d1_LORA-BIAS-7B.pth"
# WANDB_RUN_NAME="val_llama_adapter_v2_finetune_2026-03-20_03-50-25"
# MODEL_BASE_DIR="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/llama_adapter_v2_finetune_DriveLM-v1_1_train_conv2llama_2026-03-20_03-50-25"

WANDB_RUN_NAME="val_llama_adapter_v2_finetune_2026-03-26_23-12-11_v1_2_val0.1"
MODEL_BASE_DIR="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/llama_adapter_v2_finetune_2026-03-26_23-12-11_v1_2_val0.1"
JOB_TIMESTAMP=2026-03-26_23-31-39


# JOB_TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Define the master experiment name and folder path
MASTER_EXP_NAME="infer_${MODEL}_${DATASET}_${LOAD_MODE}_gpu${GPUS}_${WANDB_RUN_NAME}_${JOB_TIMESTAMP}"
MASTER_INFER_DIR="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/inference/${MASTER_EXP_NAME}"

# Create the master folder
mkdir -p "$MASTER_INFER_DIR"
echo "📂 Grouping results for this run under: $MASTER_INFER_DIR"
# ==========================================



# Check if MODEL_BASE_DIR is a directory of checkpoints or a single .pth file
if [ -d "$MODEL_BASE_DIR" ]; then
    CHECKPOINTS=$(ls -d ${MODEL_BASE_DIR}/checkpoint-* 2>/dev/null | sort -V)
    echo "Processing a folder $MODEL_BASE_DIR"
    echo "Processing checkpoints: $CHECKPOINTS"

else
    # It's a single file
    CHECKPOINTS="$MODEL_BASE_DIR"
    echo "Processing a single file $MODEL_BASE_DIR"
fi

if [ -z "$CHECKPOINTS" ]; then
    echo "❌ No checkpoints or valid model path found at $MODEL_BASE_DIR"
    exit 1
fi

echo "now Processing checkpoints: $CHECKPOINTS"
for MODEL_CKPT in $CHECKPOINTS; do
    # Extract step number (or name it 'base' if it's just the .pth file)
    # Extract step number (or default to 0 if it's just the .pth file)
    if [[ "$MODEL_CKPT" == *"checkpoint-"* ]]; then
        STEP=$(basename "$MODEL_CKPT" | sed 's/checkpoint-//g' | sed 's/\.pth//g')
        CKPT_SHORT="ckpt_${STEP}"
    else
        STEP=0
        CKPT_SHORT="ckpt_base"
    fi
    
    CKPT_INFER_DIR="${MASTER_INFER_DIR}/${CKPT_SHORT}"
    mkdir -p "$CKPT_INFER_DIR"
    
    LOG_FILE="${CKPT_INFER_DIR}/process_step_${STEP}.log"

    echo "========================================================" | tee -a "$LOG_FILE"
    echo "🚀 Processing Checkpoint: $CKPT_SHORT" | tee -a "$LOG_FILE"
    echo "📂 Output Directory: $CKPT_INFER_DIR" | tee -a "$LOG_FILE"
    echo "========================================================" | tee -a "$LOG_FILE"
    
    # --- 1. INFERENCE ---
    source /mnt/proj1/eu-25-10/envs/roadcap-gen/bin/activate
    cd /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/
    
    PRED_FILE="${CKPT_INFER_DIR}/drivelm_pred_for_HF_test_qas.json"

    if [ ! -f "$PRED_FILE" ]; then
        echo "⚙️  Running Generation..." | tee -a "$LOG_FILE"
        export MASTER_PORT=$(expr 10000 + $RANDOM % 1000)
        
        # 🌟 The critical PYTHONPATH fix and backslash formatting are here
        PYTHONPATH=. torchrun --nproc_per_node=$GPUS --master_port=$MASTER_PORT scripts/04_inference/ddp_inference.py \
            model=$MODEL \
            dataset=$DATASET \
            inference=$INFERENCE \
            experiment_name="${MASTER_EXP_NAME}/${CKPT_SHORT}" \
            inference.load_mode=$LOAD_MODE \
            inference.checkpoint_model=$MODEL_CKPT 2> >(tee -a "$LOG_FILE" >&2) | tee -a "$LOG_FILE"
            
        # DDP Inference script outputs with a timestamp, so we need to grab the actual generated filename
        PRED_FILE=$(ls -t ${CKPT_INFER_DIR}/*.json 2>/dev/null | head -n 1)
    else
        echo "✅ Inference JSON already exists, skipping generation..." | tee -a "$LOG_FILE"
    fi

    if [ -z "$PRED_FILE" ]; then
        echo "❌ Error: Inference failed to generate a prediction file for $CKPT_SHORT." | tee -a "$LOG_FILE"
        continue
    fi

    # --- 2. EVALUATION ---
    deactivate
    source /mnt/proj1/eu-25-10/envs/drivelm/bin/activate
    ml Java
    
    echo "📊 Evaluating $CKPT_SHORT and syncing to WandB (Run: $WANDB_RUN_NAME)..." | tee -a "$LOG_FILE"
    
    # 🌟 Evaluation script triggers a new W&B run automatically using these args
    PYTHONPATH="." python external/DriveLM/challenge/evaluation.py \
        --root_path1="$PRED_FILE" \
        --root_path2="$EVAL_DATASET_PATH" \
        --step=$STEP \
        --model_name="${MODEL}_Step_${STEP}" \
        --latex_out="$CKPT_INFER_DIR/latex_results.txt" \
        --wandb_project="$WANDB_PROJECT" \
        --wandb_run_name="$WANDB_RUN_NAME" 2> >(tee -a "$LOG_FILE" >&2) | tee -a "$LOG_FILE"
        
    echo "✅ Checkpoint $STEP complete!" | tee -a "$LOG_FILE"
done

echo "🎉 Pipeline finished successfully!"

# sbatch /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/scripts/03_evaluation/infer_val_llama_adapter.sh
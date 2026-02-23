#!/bin/bash

#SBATCH --job-name=QA_gen_array
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1           # Each sub-job gets its own GPU
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=OPEN-36-7
#SBATCH --time=33:00:00             # Time limit for running
#SBATCH --partition=qgpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dmytro.khursenko@valeo.com
#SBATCH --array=0-9                 # This creates 10 sub-jobs (0, 1, 2... 9)

# --- Settings ---
TOTAL_SCENES=874                    # Total number of scenes to process
NUM_JOBS=10                          # Must match the array range
STEP=$((TOTAL_SCENES / NUM_JOBS))    # scenes per job

# --- Calculate indices for THIS specific worker ---
START=$((SLURM_ARRAY_TASK_ID * STEP))
END=$((START + STEP))

echo "Worker $SLURM_ARRAY_TASK_ID processing from $START to $END"

cd /mnt/proj1/eu-25-10/dmytro/road_cap_gen_llm_integration/RoadCap-Gen || exit
source /mnt/proj1/eu-25-10/envs/roadcap-gen/bin/activate

# SET PARAMS
DATASET_NAME="nuscenes"
# DATASET_NAME="valeo"

YOLO_PATH="/scratch/project/eu-25-10/datasets/nuScenes_metadata/annotations_in_valeo_format/CAM_FRONT/n008-2018-07-27-12-07-38-0400/"
OUTPUT_FOLDER="data/qas_gen_output_$DATASET_NAME/n008-2018-07-27-12-07-38-0400/"

# --- Run the Python script ---
PYTHONPATH=. python scripts/01_pseudo_labels_gen/pseudo_qas_generation.py \
    --model="Qwen/Qwen3-14B" \
    --start_idx=$START \
    --end_idx=$END \
    --wandb_name="QWEN-14B-batch-$SLURM_ARRAY_TASK_ID" \
    --file_name="QA_qwen_part_$SLURM_ARRAY_TASK_ID" \
    --qas_ratios "configs/dataset/qas_drivelm_ratios.json" \
    --prompts_config "configs/inference/llm_prompt_config.yaml" \
    --output_folder "$OUTPUT_FOLDER" \
    --yolo_path "$YOLO_PATH" \
    --dataset_name "$DATASET_NAME" \

# sbatch scripts/slurms_scripts/stage01_pseudo_qas_generation.sh
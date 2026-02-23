#!/bin/bash

#SBATCH --job-name=QA_GEN_FULL_230226
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --account=OPEN-36-7
#SBATCH --time=48:00:00
#SBATCH --partition=qgpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dmytro.khursenko@valeo.com
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd /mnt/proj1/eu-25-10/dmytro/road_cap_gen_llm_integration/RoadCap-Gen || exit
source /mnt/proj1/eu-25-10/envs/roadcap-gen/bin/activate

# PATHS
# # nuscenes gt annotanions in valeo yolo format
BASE_YOLO="/scratch/project/eu-25-10/datasets/nuScenes_metadata/annotations_in_valeo_format/"
OUTPUT_FOLDER="/mnt/proj1/eu-25-10/datasets/nuScenes_metadata/qas_gen_from_gt_ann_no_tracks_no_llava_caption/"
DATASET_NAME="nuscenes"  

# yolov11 annotanions
# BASE_YOLO="/mnt/proj1/eu-25-10/datasets/nuScenes_metadata/Yolov11_1/"
# OUTPUT_FOLDER="/mnt/proj1/eu-25-10/datasets/nuScenes_metadata/qas_gen_from_yolov11_no_tracks_no_llava_caption/"
# DATASET_NAME="nuscenes"  
   

# # yolov11 + yoloworld annotanions TODO
# BASE_YOLO="/scratch/project/eu-25-10/datasets/nuScenes_metadata/annotations_in_valeo_format/"
# OUTPUT_FOLDER="/mnt/proj1/eu-25-10/datasets/nuScenes_metadata/qas_gen_from_gt_ann_no_tracks_no_llava_caption/" 

# Create log and output dirs
mkdir -p logs
mkdir -p "$OUTPUT_FOLDER"

# RUN
# start_idx=0 and end_idx=None means each worker will process every scene in any folder it locks.
# The array workers will distribute themselves across cameras and folders.
PYTHONPATH=. python scripts/01_pseudo_labels_gen/pseudo_qas_generation_whole_folder.py \
    --model="Qwen/Qwen3-14B" \
    --qas_ratios "configs/dataset/qas_drivelm_ratios.json" \
    --prompts_config "configs/inference/llm_prompt_config.yaml" \
    --output_folder "$OUTPUT_FOLDER" \
    --yolo_path "$BASE_YOLO" \
    --dataset_name "$DATASET_NAME" \
    --number_of_questions 15

# sbatch scripts/slurms_scripts/stage01_pseudo_qas_generation_whole_folder.sh

# # for i in {1..30}; do sbatch scripts/slurms_scripts/stage01_pseudo_qas_generation_whole_folder.sh; sleep 33; done
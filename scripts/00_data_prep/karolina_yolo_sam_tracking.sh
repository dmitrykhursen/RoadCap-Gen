#!/bin/bash

#SBATCH --job-name=yoloworld_sam_track_nuscenes
#SBATCH --account=open-36-7
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
# source /mnt/proj1/eu-25-10/envs/yolov11_sam_tracking/bin/activate
cd /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen

# Define variables
# FOLDER_with_IMGS="/mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/nuscenes/train_val_samples_grouped/"
# OUTPUT_DIR="output/nuscenes_train_val_metadata/"

# FOLDER_with_IMGS="/mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/nuscenes/test_data_grouped/"
# OUTPUT_DIR="output/nuscenes_test_metadata/"

#nuScenes v1.0 
FOLDER_with_IMGS="/scratch/project/eu-25-10/datasets/nuScenes/samples_grouped"
# OUTPUT_DIR="/mnt/proj1/eu-25-10/datasets/nuScenes_metadata/"
DATASET="drivelm"

# # valeo data
# FOLDER_with_IMGS="/scratch/project/eu-25-10/datasets/FRONT_CAM_zipped/20250527/22"
# OUTPUT_DIR="/scratch/project/eu-25-10/datasets/FRONT_CAM_zipped_metadata/YOLOv11_1/"
# DATASET="valeo"


# # yoloV11 args
# source /mnt/proj1/eu-25-10/envs/yolov11_sam_tracking/bin/activate
# DET_MODEL="yolov11"
# DET_MODEL_CKPT="external/models/YOLOv11/best.pt"
# DET_MODEL_CONFIG=""
# OPEN_CLASSES=""

# yoloWorld args
source /mnt/proj1/eu-25-10/envs/yoloworld-seg-track/bin/activate
export PYTHONPATH="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/external/models/YoloWorld:$PYTHONPATH"
DET_MODEL="yoloworld"
DET_MODEL_CKPT="external/models/YoloWorld/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"
DET_MODEL_CONFIG="external/models/YoloWorld/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"

OPEN_CLASSES="external/models/YoloWorld/open_classes_nuscnesgt_minus_yolov11_short_names.txt"
OUTPUT_DIR="/mnt/proj1/eu-25-10/datasets/nuScenes_metadata/YoloWorld_short_classnames"
# OPEN_CLASSES="external/models/YoloWorld/open_classes_nuscnesgt_minus_yolov11_long_names.txt"
# OUTPUT_DIR="/mnt/proj1/eu-25-10/datasets/nuScenes_metadata/YoloWorld_long_classnames"



# other args
SAM_CKPT="external/models/SAM/sam_vit_h_4b8939.pth"
TRACKER_CKPT="external/models/Tracker/mot17_sbs_S50.pth"
FAST_REID_CFG="external/fast_reid/configs/MOT17/sbs_S50.yml"
BATCH_SIZE=1

# Run the command
srun python scripts/00_data_prep/extract_metadata.py \
    --img $FOLDER_with_IMGS \
    --checkpoint $DET_MODEL_CKPT \
    --det-model $DET_MODEL \
    --det-model-config $DET_MODEL_CONFIG \
    --open-classes $OPEN_CLASSES \
    --segment-ckpt $SAM_CKPT \
    --bs $BATCH_SIZE \
    --dataset $DATASET \
    --out-dir $OUTPUT_DIR \
    --fast-reid-config $FAST_REID_CFG \
    --fast-reid-weights $TRACKER_CKPT \
    --use-tracking \
    --use-sam \
    # --show

# cd /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen
# sbatch /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/scripts/00_data_prep/karolina_yolo_sam_tracking.sh

# for i in {1..10}; do sbatch scripts/00_data_prep/karolina_yolo_sam_tracking.sh; sleep 7; done
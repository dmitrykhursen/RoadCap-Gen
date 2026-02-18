#!/bin/bash

#SBATCH --job-name=yolov11_sam_track_valeo_22
#SBATCH --account=EU-25-10
#SBATCH --time=47:00:00 # Time limit for running
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dmytro.khursenko@valeo.com
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Load necessary modules (if applicable)	
source /mnt/proj1/eu-25-10/envs/yolov11_sam_tracking/bin/activate
cd /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen

# Define variables
# FOLDER_with_IMGS="/mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/nuscenes/train_val_samples_grouped/"
# OUTPUT_DIR="output/nuscenes_train_val_metadata/"

# FOLDER_with_IMGS="/mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/nuscenes/test_data_grouped/"
# OUTPUT_DIR="output/nuscenes_test_metadata/"

# nuScenes v1.0 
# FOLDER_with_IMGS="/scratch/project/eu-25-10/datasets/nuScenes/samples_grouped"
# OUTPUT_DIR="/mnt/proj1/eu-25-10/datasets/nuScenes_metadata/"
# DATASET="drivelm"

# valeo data
FOLDER_with_IMGS="/scratch/project/eu-25-10/datasets/FRONT_CAM_zipped/20250527/22"
OUTPUT_DIR="/scratch/project/eu-25-10/datasets/FRONT_CAM_zipped_metadata/YOLOv11_1/"
DATASET="valeo"


YOLOv11_CKPT="external/models/YOLOv11/best.pt"
SAM_CKPT="external/models/SAM/sam_vit_h_4b8939.pth"
TRACKER_CKPT="external/models/Tracker/mot17_sbs_S50.pth"
FAST_REID_CFG="external/fast_reid/configs/MOT17/sbs_S50.yml"
BATCH_SIZE=1

# Run the command
srun python scripts/00_data_prep/extract_metadata.py \
    --img $FOLDER_with_IMGS \
    --checkpoint $YOLOv11_CKPT \
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

# for i in {1..30}; do sbatch scripts/00_data_prep/karolina_yolo_sam_tracking.sh; sleep 7; done
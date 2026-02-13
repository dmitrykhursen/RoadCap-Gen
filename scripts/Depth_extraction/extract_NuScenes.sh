#!/bin/bash
#SBATCH -A EU-25-10
#SBATCH -p qgpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH -t 8:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/%A_%a.out

CAM=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" cams.txt | tr -d '\r')


ml Python
source /mnt/proj1/eu-25-92/vbesnier/maskgit-vid-venv/bin/activate
cd /home/it4i-vbesnier/Project/RoadCap-gen/src/training/

python depth_emb_extract.py \
  --input_folder="/scratch/project/eu-25-10/datasets/nuScenes/samples_grouped/${CAM}/" \
  --output_folder="/mnt/proj1/eu-25-10/datasets/NuScenes/sampled_grouped/${CAM}/" \
  --hf_cache="/scratch/project/eu-25-50/huggingface_cache/"

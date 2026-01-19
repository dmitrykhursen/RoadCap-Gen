#!/bin/bash
#SBATCH --job-name=finetune_llava_valeo
#SBATCH --account=EU-25-10
#SBATCH --time=1:00:00 # Time limit for running
#SBATCH --partition=qgpu_exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dmytro.khursenko@valeo.com
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Load necessary modules (if applicable)	
source /mnt/proj1/eu-25-10/envs/roadcap-gen/bin/activate
cd /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/

# Pick a random port or use job ID to avoid 29500 collision
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(hostname)
echo "🚀 Master Addr: $MASTER_ADDR"
echo "🚀 Master Port: $MASTER_PORT"
# --- FIX END ---

# Define variables
MODEL="llava"
# DATASET="vqa_valeo_karolina"
DATASET="drivelm"
INFERENCE="drivelm_infer"
GPUS=1

# ----- CHANGE EXPERIMENT NAME FOR EACH RUN -----
experiment_name="karolina_torchrun_llava_pretrained_DriveLM_infer_data0.03_gpu1_bs8_nw4"

# experiment_name="karolina_torchrun_llava_FFmmp_DriveLM_infer_data1.0_gpu1_bs8_nw4"

# experiment_name="karolina_torchrun_llava_FFmmp_Valeo_infer_data1.0_gpu4_bs16_nw4"


# -----------------------------------------------
# Run the command
torchrun --nproc_per_node=$GPUS --master_port=$MASTER_PORT scripts/04_inference/ddp_inference.py \
    model=$MODEL     \
    dataset=$DATASET     \
    inference=$INFERENCE     \
    experiment_name=$experiment_name     \



# sbatch scripts/04_inference/karolina_infer.sh
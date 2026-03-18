#!/bin/bash
#SBATCH --job-name=infer_llava_valeo
#SBATCH --account=OPEN-36-7
#SBATCH --time=00:30:00 # Time limit for running
#SBATCH --partition=qgpu_exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
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


# Define variables
MODEL="llava"
# DATASET="vqa_valeo_karolina"
DATASET="drivelm"
INFERENCE="drivelm_infer"
GPUS=2

# MODEL_CKPT="llava-hf/llava-v1.6-mistral-7b-hf"
MODEL_CKPT="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/llava-v1.6-mistral-7b/karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-data_usage-1.0_gpu8_bs4/checkpoint-470"
experiment_name="for_testing_purposes_infer_llava_finetuned_on_drivelm_train0.7_gpu4_bs8_nw4_e10_ckpt470_local_tokenizer"
# experiment_name="infer_llava_finetuned_on_drivelm_train0.7_gpu4_bs8_nw4_e0_pretrained"
# experiment_name="infer_llava_finetuned_on_drivelm_train1.0_all_data_gpu4_bs8_nw4_e0_pretrained"
# MODEL_MODE="pretrained"

#  = = = = = = = = = = = = = = 
# trained on full drivelm 1.0
# MODEL_CKPT="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/llava-v1.6-mistral-7b/karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-train1.0-all-data_data_usage-1.0_gpu8_bs4_260226/checkpoint-116"
# experiment_name="infer_llava_finetuned_on_drivelm_train1.0-alldata_gpu4_bs8_nw4_e2_ckpt116"
# MODEL_MODE="fullfinetune"

# MODEL_CKPT="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/llava-v1.6-mistral-7b/karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-train1.0-all-data_data_usage-1.0_gpu8_bs4_260226/checkpoint-580"
# experiment_name="infer_llava_finetuned_on_drivelm_train1.0-alldata_gpu4_bs8_nw4_e10_ckpt580"
# MODEL_MODE="fullfinetune"

#  = = = = = = = = = = = = = = 
# trained on drivelm 0.7
# MODEL_CKPT="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/llava-v1.6-mistral-7b/karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-data_usage-1.0_gpu8_bs4/checkpoint-47"
# experiment_name="infer_llava_finetuned_on_drivelm_train0.7_gpu4_bs8_nw4_e1_ckpt47"
MODEL_MODE="fullfinetune"

# MODEL_CKPT="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/llava-v1.6-mistral-7b/karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-data_usage-1.0_gpu8_bs4/checkpoint-235"
# experiment_name="infer_llava_finetuned_on_drivelm_train0.7_gpu4_bs8_nw4_e5_ckpt235"

# MODEL_CKPT="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/llava-v1.6-mistral-7b/karolina_torchrun_trainval_llava_fullfinetune_MMP_DriveLM-data_usage-1.0_gpu8_bs4/checkpoint-470"
# experiment_name="infer_llava_finetuned_on_drivelm_train0.7_gpu4_bs8_nw4_e10_ckpt470"


# experiment_name="latest_karolina_torchrun_local_val_llava_pretrained_infer_data1_0_gpu6_bs8_nw4"

# experiment_name="karolina_torchrun_llava_FFmmp_DriveLM_infer_data1.0_gpu1_bs8_nw4"

# experiment_name="karolina_torchrun_llava_FFmmp_Valeo_infer_data1.0_gpu4_bs16_nw4"

# DATASET_PATH="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/external/DriveLM/challenge/data/v1_1_test_nus_q_only_conv2llama.json"
DATASET_PATH="/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/external/DriveLM/challenge/data/for_testing_purposes_vqas.json"

# export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG=INFO

# -----------------------------------------------
# Run the command
torchrun --nproc_per_node=$GPUS --master_port=$MASTER_PORT scripts/04_inference/ddp_inference.py \
    model=$MODEL     \
    dataset=$DATASET     \
    inference=$INFERENCE     \
    experiment_name=$experiment_name     \
    inference.load_mode=$MODEL_MODE  \
    inference.checkpoint_model=$MODEL_CKPT \
    dataset.data_path=$DATASET_PATH \
    dataset.image_folder="/mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/nuscenes/test_data" \
    inference.tokenizer_path=$MODEL_CKPT



# sbatch scripts/04_inference/karolina_infer.sh
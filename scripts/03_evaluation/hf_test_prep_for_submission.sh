#!/bin/bash

source /mnt/proj1/eu-25-10/envs/roadcap-gen/bin/activate

RESULTS_PATH=""

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --results_path) RESULTS_PATH="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if the argument was provided
if [ -z "$RESULTS_PATH" ]; then
    echo "Error: --results_path is required."
    echo "Usage: ./run_pipeline.sh --results_path /path/to/My-Awesome-Method"
    exit 1
fi

# Get the absolute, clean path
CLEAN_PATH=$(realpath "$RESULTS_PATH")

# Determine the directory and the input JSON file
if [ -f "$CLEAN_PATH" ]; then
    # User passed a file path (e.g., /path/to/method/output.json)
    TARGET_DIR=$(dirname "$CLEAN_PATH")
    INPUT_JSON="$CLEAN_PATH"
else
    # User passed a directory path (e.g., /path/to/method)
    TARGET_DIR="$CLEAN_PATH"
    INPUT_JSON="$TARGET_DIR/output.json"
fi

# Extract the last folder name to use as the method string
METHOD_NAME=$(basename "$TARGET_DIR")

echo "====================================="
echo "Target Directory : $TARGET_DIR"
echo "Method Name      : $METHOD_NAME"
echo "Input File       : $INPUT_JSON"
echo "====================================="

# Run the Python script (ensure the python script name matches yours)
python /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/external/DriveLM/challenge/prepare_submission.py \
    --input_json "$INPUT_JSON" \
    --output_folder "$TARGET_DIR" \
    --method "$METHOD_NAME"

# bash /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/scripts/03_evaluation/hf_test_prep_for_submission.sh --results_path /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/inference/infer_llava_finetuned_on_drivelm_train0.7_gpu4_bs8_nw4_e1_ckpt47_2026-02-27_16-36-48/drivelm_pred_for_HF_test_qas_2026-02-27_16-36-48.json

# bash /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/scripts/03_evaluation/hf_test_prep_for_submission.sh --results_path /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/output/inference/infer_llava_finetuned_on_drivelm_train1.0-alldata_gpu4_bs8_nw4_e10_ckpt580_2026-03-04_02-28-48/drivelm_pred_for_HF_test_qas_2026-03-04_02-28-48_corrected.json
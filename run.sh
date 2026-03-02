mkdir -p logs
mkdir -p /output

python scripts/01_pseudo_labels_gen/pseudo_qas_generation_whole_folder.py \
    --model="Qwen/Qwen3-14B" \
    --qas_ratios "configs/dataset/qas_drivelm_ratios.json" \
    --prompts_config "configs/inference/llm_prompt_config.yaml" \
    --output_folder /output \
    --yolo_path /yolo \
    --dataset_name "$DATASET_NAME"
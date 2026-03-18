import argparse
import json
import os

def format_size(size_in_bytes):
    """Converts bytes to a human-readable format (B, KB, MB, GB)."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Stack multiple JSON arrays into a single large JSON file.")
    parser.add_argument(
        '-i', '--inputs', 
        nargs='+', 
        required=True, 
        help="List of input JSON files to process."
    )
    parser.add_argument(
        '-o', '--output', 
        required=True, 
        help="Path for the output stacked JSON file."
    )
    parser.add_argument(
        '--indent',
        type=int,
        default=4,
        help="Indentation level for the output JSON (default: 4)."
    )
    
    args = parser.parse_args()

    all_items = []
    total_items = 0
    total_input_size_bytes = 0

    print("=" * 50)
    print("Processing JSON files...")
    print("=" * 50)

    # Process each file
    for file_path in args.inputs:
        if not os.path.exists(file_path):
            print(f"[WARNING] File not found: {file_path}")
            continue

        # Get and accumulate file size
        file_size = os.path.getsize(file_path)
        total_input_size_bytes += file_size

        # Read JSON and count items
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Ensure the JSON is a list (array of objects)
                if not isinstance(data, list):
                    print(f"[WARNING] Expected a list in {file_path}, skipping.")
                    continue

                num_items = len(data)
                total_items += num_items
                
                # .extend() appends elements from the iterable, stacking them flat
                all_items.extend(data) 

                print(f"File: {file_path}")
                print(f"  -> Items: {num_items}")
                print(f"  -> Size:  {format_size(file_size)}")

        except json.JSONDecodeError:
            print(f"[ERROR] Could not decode JSON in {file_path}")
        except Exception as e:
            print(f"[ERROR] Failed to process {file_path}: {e}")

    # Write combined data to the output file
    print("-" * 50)
    print(f"Saving combined data to {args.output}...")
    
    with open(args.output, 'w', encoding='utf-8') as f:
        # Using json.dump to write everything exactly as it was
        json.dump(all_items, f, indent=args.indent)
    
    output_size_bytes = os.path.getsize(args.output)

    # Print Final Summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total Files Processed: {len(args.inputs)}")
    print(f"Total Items Stacked:   {total_items}")
    print(f"Total Input Size:      {format_size(total_input_size_bytes)}")
    print(f"Output File Size:      {format_size(output_size_bytes)}")

if __name__ == "__main__":
    main()

# source /mnt/proj1/eu-25-10/envs/roadcap-gen/bin/activate

# python stack_jsons.py -i file1.json file2.json file3.json -o stacked_output.json



# python /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/src/utils/stack_qas_jsons.py -i /mnt/proj1/eu-25-10/martin/RoadCap-Gen/data/converted/*.json -o /mnt/proj1/eu-25-10/datasets/DRIVE_LM_metadata/v2_0_aug_qas_Qwen3-14B_nuscenes_think_no-tracks_dynamic_q_converted.json
# --------------------------------------------------
# Saving combined data to /mnt/proj1/eu-25-10/datasets/DRIVE_LM_metadata/v2_0_aug_qas_Qwen3-14B_nuscenes_think_no-tracks_dynamic_q_converted.json...
# ==================================================
# SUMMARY
# ==================================================
# Total Files Processed: 43
# Total Items Stacked:   33579
# Total Input Size:      41.84 MB
# Output File Size:      45.69 MB



# python /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/src/utils/stack_qas_jsons.py -i /mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/v1_2_train_converted_llama.json /mnt/proj1/eu-25-10/datasets/DRIVE_LM_metadata/v2_0_aug_qas_Qwen3-14B_nuscenes_think_no-tracks_dynamic_q_converted.json -o /mnt/proj1/eu-25-10/datasets/DRIVE_LM_metadata/v2_0_trainaug_qas_Qwen3-14B_nuscenes_think_no-tracks_dynamic_q_converted.json
# ==================================================
# Processing JSON files...
# ==================================================
# File: /mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/v1_2_train_converted_llama.json
#   -> Items: 345802
#   -> Size:  400.35 MB
# File: /mnt/proj1/eu-25-10/datasets/DRIVE_LM_metadata/v2_0_aug_qas_Qwen3-14B_nuscenes_think_no-tracks_dynamic_q_converted.json
#   -> Items: 33579
#   -> Size:  45.69 MB
# --------------------------------------------------
# Saving combined data to /mnt/proj1/eu-25-10/datasets/DRIVE_LM_metadata/v2_0_trainaug_qas_Qwen3-14B_nuscenes_think_no-tracks_dynamic_q_converted.json...
# ==================================================
# SUMMARY
# ==================================================
# Total Files Processed: 2
# Total Items Stacked:   379381
# Total Input Size:      446.04 MB
# Output File Size:      446.04 MB

# python /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/src/utils/stack_qas_jsons.py -i /mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/v1_2_val_converted_llama_with_tags.json /mnt/proj1/eu-25-10/datasets/DRIVE_LM_metadata/v
# 2_0_aug_qas_Qwen3-14B_nuscenes_think_no-tracks_dynamic_q_converted.json -o /mnt/proj1/eu-25-10/datasets/DRIVE_LM_metadata/v2_0_valaug
# _qas_Qwen3-14B_nuscenes_think_no-tracks_dynamic_q_converted.json
# ==================================================
# Processing JSON files...
# ==================================================
# File: /mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/v1_2_val_converted_llama_with_tags.json
#   -> Items: 29450
#   -> Size:  38.52 MB
# File: /mnt/proj1/eu-25-10/datasets/DRIVE_LM_metadata/v2_0_aug_qas_Qwen3-14B_nuscenes_think_no-tracks_dynamic_q_converted.json
#   -> Items: 33579
#   -> Size:  45.69 MB
# --------------------------------------------------
# Saving combined data to /mnt/proj1/eu-25-10/datasets/DRIVE_LM_metadata/v2_0_valaug_qas_Qwen3-14B_nuscenes_think_no-tracks_dynamic_q_converted.json...
# ==================================================
# SUMMARY
# ==================================================
# Total Files Processed: 2
# Total Items Stacked:   63029
# Total Input Size:      84.20 MB
# Output File Size:      84.20 MB
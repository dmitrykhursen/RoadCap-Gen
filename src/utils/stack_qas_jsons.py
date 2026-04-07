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
    parser = argparse.ArgumentParser(description="Stack multiple JSON/JSONL arrays into a single large JSON file.")
    parser.add_argument(
        '-i', '--inputs', 
        nargs='+', 
        required=True, 
        help="List of input JSON/JSONL files or directories to process recursively."
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

    # --- UPDATED LOGIC: Expand directories and look for BOTH .json and .jsonl ---
    expanded_files = []
    for input_path in args.inputs:
        if os.path.isfile(input_path):
            expanded_files.append(input_path)
        elif os.path.isdir(input_path):
            for root, dirs, files in os.walk(input_path, followlinks=True):
                for file_name in files:
                    # Use a tuple to check for multiple extensions
                    if file_name.lower().endswith(('.json', '.jsonl')):
                        full_path = os.path.join(root, file_name)
                        expanded_files.append(full_path)
        else:
            print(f"[WARNING] Path not found or invalid: {input_path}")
    # -----------------------------------------------------------------------------

    all_items = []
    total_items = 0
    total_input_size_bytes = 0

    print("=" * 50)
    print("Processing files...")
    print("=" * 50)

    # Process each file from our expanded list
    for file_path in expanded_files:
        file_size = os.path.getsize(file_path)
        total_input_size_bytes += file_size

        try:
            # --- UPDATED LOGIC: Handle standard JSON vs JSONL ---
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.lower().endswith('.jsonl'):
                    # Parse line-by-line for JSONL
                    data = [json.loads(line) for line in f if line.strip()]
                else:
                    # Parse as a single block for standard JSON
                    data = json.load(f)
                
                # Ensure the data is a list (JSONL parsing guarantees a list, but standard JSON might not)
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
    if not expanded_files:
        print("[ERROR] No valid JSON/JSONL files were found to process. Exiting.")
        return

    print("-" * 50)
    print(f"Saving combined data to {args.output}...")
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_items, f, indent=args.indent)
    
    output_size_bytes = os.path.getsize(args.output)

    # Print Final Summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total Files Processed: {len(expanded_files)}")
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



# python /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/src/utils/stack_qas_jsons.py -i /mnt/proj1/eu-25-10/datasets/DRIVE_LM_metadata/v2_0_fromtrain300_85k_qas_following_testdistr_converted_llama.json /mnt/proj1/eu-25-10/datasets/DRIVE_LM_metadata/v2_0_aug_qas_Qwen3-14B_nuscenes_think_no-tracks_dynamic_q_converted.json  -o /mnt/proj1/eu-25-10/datasets/DRIVE_LM_metadata/v2_0_train85kaug_qas_Qwen3-14B_nuscenes_think_no-tracks_dynamic_q_converted.json
# ==================================================
# Processing JSON files...
# ==================================================
# File: /mnt/proj1/eu-25-10/datasets/DRIVE_LM_metadata/v2_0_fromtrain300_85k_qas_following_testdistr_converted_llama.json
#   -> Items: 85344
#   -> Size:  103.94 MB
# File: /mnt/proj1/eu-25-10/datasets/DRIVE_LM_metadata/v2_0_aug_qas_Qwen3-14B_nuscenes_think_no-tracks_dynamic_q_converted.json
#   -> Items: 33579
#   -> Size:  45.69 MB
# --------------------------------------------------
# Saving combined data to /mnt/proj1/eu-25-10/datasets/DRIVE_LM_metadata/v2_0_train85kaug_qas_Qwen3-14B_nuscenes_think_no-tracks_dynamic_q_converted.json...
# ==================================================
# SUMMARY
# ==================================================
# Total Files Processed: 2
# Total Items Stacked:   118923
# Total Input Size:      149.63 MB
# Output File Size:      149.63 MB

# /mnt/proj1/eu-25-10/martin/RoadCap-Gen/data/nuscenes_tracks
# python /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/src/utils/stack_qas_jsons.py -i /mnt/proj1/eu-25-10/datasets/DRIVE_LM_metadata/v2_0_fromtrain300_85k_qas_following_testdistr_converted_llama.json /mnt/proj1/eu-25-10/martin/RoadCap-Gen/data/nuscenes_tracks/  -o /mnt/proj1/eu-25-10/datasets/DRIVE_LM_metadata/v2_0_train85kaug_qas_Qwen3-14B_nuscenes_think_relative-tracks_dynamic_q_converted.json
# Saving combined data to /mnt/proj1/eu-25-10/datasets/DRIVE_LM_metadata/v2_0_train85kaug_qas_Qwen3-14B_nuscenes_think_relative-tracks_dynamic_q_converted.json...
# ==================================================
# SUMMARY
# ==================================================
# Total Files Processed: 48
# Total Items Stacked:   99950
# Total Input Size:      233.19 MB
# Output File Size:      249.16 MB

#Saving combined data to /mnt/proj1/eu-25-10/datasets/DRIVE_LM_metadata/v2_0_aug_qas_Qwen3-14B_nuscenes_think_relative-tracks_dynamic_q_converted.json...
# ==================================================
# SUMMARY
# ==================================================
# Total Files Processed: 47
# Total Items Stacked:   14657
# Total Input Size:      129.72 MB
# Output File Size:      145.75 MB

# python /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/src/utils/stack_qas_jsons.py -i /mnt/proj1/eu-25-10/martin/RoadCap-Gen/data/nuscenes_tracks_global/  -o /mnt/proj1/eu-25-10/datasets/DRIVE_LM_metadata/v2_0_aug_qas_Qwen3-14B_nuscenes_think_tmp-global-tracks_dynamic_q_converted.json
# TODO
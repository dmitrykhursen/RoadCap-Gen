import json
import random
import argparse
import os


def main():
    # Set up argparse
    parser = argparse.ArgumentParser(description="Extract a test split from the nested DriveLM dataset.")
    parser.add_argument(
        "--input", 
        required=True, 
        help="Path to the full input JSON dataset."
    )
    parser.add_argument(
        "--output", 
        required=True, 
        help="Path to save the output testset JSON."
    )
    parser.add_argument(
        "--ratio", 
        type=float, 
        default=0.1, 
        help="Ratio of data to use for the test set (default: 0.1 for 10%)."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for deterministic shuffling. Matches the dataloader default (default: 42)."
    )

    args = parser.parse_args()

    # Verify input exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    print(f"📂 Loading data from: {args.input}")
    with open(args.input, "r") as f:
        raw_data = json.load(f)
    
    # 1. Get all top-level scenario IDs
    scenario_ids = list(raw_data.keys())
    
    # 2. Filter out invalid scenarios (ensuring 'key_frames' exists and isn't empty)
    valid_ids = [
        s_id for s_id in scenario_ids 
        if "key_frames" in raw_data[s_id] and raw_data[s_id]["key_frames"]
    ]

    print(f"🔍 Found {len(valid_ids)} valid scenarios out of {len(scenario_ids)} total.")
    
    # 3. Deterministic shuffle of the scenario IDs
    random.seed(args.seed)
    random.shuffle(valid_ids)
    
    # 4. Calculate split index
    total = len(valid_ids)
    test_size = int(total * args.ratio)
    
    # Take the LAST N scenarios to avoid overlapping with an 80% train split 
    test_ids = valid_ids[-test_size:]
    
    # 5. Reconstruct the dictionary mapping for the test set
    test_data = {s_id: raw_data[s_id] for s_id in test_ids}
    
    # 6. Save to JSON
    print(f"💾 Saving {len(test_data)} scenarios ({args.ratio * 100}%) to: {args.output}")
    with open(args.output, "w") as f:
        json.dump(test_data, f, indent=4)
        
    print("✅ Done!")

# def main():
#     # Set up argparse
#     parser = argparse.ArgumentParser(description="Extract a test split from the DriveLM dataset.")
#     parser.add_argument(
#         "--input", 
#         required=True, 
#         help="Path to the full input JSON dataset."
#     )
#     parser.add_argument(
#         "--output", 
#         required=True, 
#         help="Path to save the output testset JSON."
#     )
#     parser.add_argument(
#         "--ratio", 
#         type=float, 
#         default=0.1, 
#         help="Ratio of data to use for the test set (default: 0.1 for 10%)."
#     )
#     parser.add_argument(
#         "--seed", 
#         type=int, 
#         default=42, 
#         help="Random seed for deterministic shuffling. Matches the dataloader default (default: 42)."
#     )

#     args = parser.parse_args()

#     # Verify input exists
#     if not os.path.exists(args.input):
#         print(f"Error: Input file '{args.input}' not found.")
#         return

#     print(f"📂 Loading data from: {args.input}")
#     with open(args.input, "r") as f:
#         raw_data = json.load(f)
    
#     # 1. Filter out invalid samples (matching your dataloader logic)
#     valid_samples = [x for x in raw_data if 'image' in x]

#     print(f"🔍 Found {len(valid_samples)} valid samples out of {len(raw_data)} total.")
    
#     # 2. Deterministic shuffle
#     random.seed(args.seed)
#     random.shuffle(valid_samples)
    
#     # 3. Calculate split index
#     total = len(valid_samples)
#     test_size = int(total * args.ratio)
    
#     # Take the LAST N samples to avoid overlapping with an 80% train split 
#     # (which normally takes the first 80%)
#     test_data = valid_samples[-test_size:]
    
#     # 4. Save to JSON
#     print(f"💾 Saving {len(test_data)} samples ({args.ratio * 100}%) to: {args.output}")
#     with open(args.output, "w") as f:
#         json.dump(test_data, f, indent=4)
        
#     print("✅ Done!")

if __name__ == "__main__":
    main()

# source /mnt/proj1/eu-25-10/envs/roadcap-gen/bin/activate
# python /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/src/utils/split_dataset.py --input /mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/v1_2_train_converted_llama.json --output /mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/v1_2_train0.1_converted_llama.json --ratio 0.1

# python /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/src/utils/split_dataset.py --input /mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/v1_2_val.json --output /mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/v1_2_val0.1.json --ratio 0.1

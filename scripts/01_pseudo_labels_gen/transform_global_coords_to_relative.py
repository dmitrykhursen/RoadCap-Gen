import json
import os
import numpy as np
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

# 1. Initialize nuScenes
# Make sure dataroot points to the parent folder containing 'v1.0-trainval'
print("Loading nuScenes database...")
nusc = NuScenes(version="v1.0-trainval", dataroot="../nuscenes", verbose=False)

# 2. Build the fast lookup dictionary (basename -> ego_pose_token)
print("Building filename lookup dictionary...")
filename_to_pose_token = {}
for sd in nusc.sample_data:
    basename = os.path.basename(sd["filename"])
    filename_to_pose_token[basename] = sd["ego_pose_token"]

# 3. Define your directories and filenames
scenes_root_dir = Path("./data/tracks_by_scene/")
input_filename = "tracks.json"  # Change this if your raw files are named differently
output_filename = "tracks_ego_centric.json"
# 4. Loop over all scene folders recursively
print(f"Scanning '{scenes_root_dir}' for '{input_filename}' files...")

# .rglob() automatically looks inside all subfolders for the target file
for track_file_path in scenes_root_dir.rglob(input_filename):
    print(f"Processing: {track_file_path}")

    # Load the raw track JSON
    with open(track_file_path, "r") as f:
        data = json.load(f)

    # Transform the coordinates
    for track_group in data["tracks"]:
        for frame_data in track_group["track"]:
            frame_name = frame_data["frame_name"]

            # Skip if we can't find the frame in nuScenes
            if not frame_name or frame_name not in filename_to_pose_token:
                print(f"{15 * '-'}Did not find frame {frame_name} in nuScenes{15 * '-'}")
                continue

            # Get the exact ego pose token and pose data
            pose_token = filename_to_pose_token[frame_name]
            pose = nusc.get("ego_pose", pose_token)

            ego_translation = np.array(pose["translation"])
            ego_rotation = Quaternion(pose["rotation"])

            # Ensure the global coordinates actually exist in this frame
            if "x" in frame_data and "y" in frame_data and "z" in frame_data:
                global_pos = np.array([frame_data["x"], frame_data["y"], frame_data["z"]])

                # --- THE MATH ---
                # Step A: Shift origin to ego vehicle
                pos_shifted = global_pos - ego_translation
                # Step B: Align axes to the car (X-forward, Y-left, Z-up)
                pos_ego_frame = ego_rotation.inverse.rotate(pos_shifted)

                # Write the new coordinates to the dictionary
                frame_data["x_ego"] = round(pos_ego_frame[0], 2)
                frame_data["y_ego"] = round(pos_ego_frame[1], 2)
                frame_data["z_ego"] = round(pos_ego_frame[2], 2)

    # 5. Save the updated JSON in the EXACT SAME folder
    # .parent gets the folder where the input file lives
    output_path = track_file_path.parent / output_filename

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  -> Saved: {output_path}")

print("\nAll scenes successfully processed!")

import os
import json
import argparse
from datetime import datetime
from collections import defaultdict

def process_and_append_frames(json_files, tracks_dict, camera_name=None):
    """Parses a list of sorted JSON files and appends data to the shared tracks dictionary."""
    frame_idx = 0  # Reset frame counter to 0 for this specific sequence/camera
    img_path_base = ""

    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        if not img_path_base:
            img_path_base = data.get("img_path", "")
            
        for frame_data in data.get("results", []):
            image_name = frame_data.get("image_name", "unknown_image")
            
            feat_2d_map = {feat["object_id"]: feat for feat in frame_data.get("2d_feat", [])}
            feat_3d_map = {feat["object_id"]: feat for feat in frame_data.get("3d_feat", [])}
            
            all_ids_in_frame = set(feat_2d_map.keys()).union(set(feat_3d_map.keys()))
            
            for obj_id in all_ids_in_frame:
                if obj_id not in tracks_dict:
                    raw_category = feat_3d_map.get(obj_id, feat_2d_map.get(obj_id, {})).get("category_name", "unknown")
                    category = raw_category.split(".")[-1] if "." in raw_category else raw_category
                    
                    tracks_dict[obj_id] = {
                        "object_id": obj_id,
                        "category": category,
                        "track": []
                    }
                
                f2d = feat_2d_map.get(obj_id, {})
                f3d = feat_3d_map.get(obj_id, {})
                
                bbox_center_3d = f3d.get("bbox_center_3d", [None, None, None])
                bbox_center_2d = f2d.get("bbox_center", [None, None])
                depth = f3d.get("depth", None)
                
                # Build the track point
                track_point = {
                    "frame": frame_idx,
                    "frame_name": image_name,
                    "x": round(bbox_center_3d[0], 2) if bbox_center_3d[0] is not None else None,
                    "y": round(bbox_center_3d[1], 2) if bbox_center_3d[1] is not None else None,
                    "z": round(bbox_center_3d[2], 2) if bbox_center_3d[2] is not None else None,
                    "center_2d_px": [
                        int(bbox_center_2d[0]) if bbox_center_2d[0] is not None else None,
                        int(bbox_center_2d[1]) if bbox_center_2d[1] is not None else None
                    ],
                    "depth": round(depth, 2) if depth is not None else None
                }

                # If merging by scene, attach the camera name to disambiguate concurrent 2D projections
                if camera_name:
                    track_point["camera"] = camera_name
                
                tracks_dict[obj_id]["track"].append(track_point)
            
            frame_idx += 1

    return img_path_base, frame_idx

def save_tracks(tracks_dict, output_folder, img_path_base):
    """Saves the tracks dictionary to the specified output folder."""
    if not tracks_dict:
        return

    os.makedirs(output_folder, exist_ok=True)
    
    # Sort the track points by frame chronologically for each object
    for obj_id in tracks_dict:
        tracks_dict[obj_id]["track"] = sorted(tracks_dict[obj_id]["track"], key=lambda k: k['frame'])

    output_data = {
        "date_time_tracks_generated": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "img_path_base": img_path_base,
        "tracks": list(tracks_dict.values())
    }
    
    out_file_path = os.path.join(output_folder, "tracks.json")
    with open(out_file_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved merged tracks for {len(tracks_dict)} objects -> {out_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Process frame-by-frame JSONs into continuous object tracks.")
    parser.add_argument("--input_dir", type=str, required=True, help="Root dir of annotations (e.g., annotations_in_valeo_format)")
    parser.add_argument("--output_dir", type=str, required=True, help="Root dir to save tracking JSONs")
    parser.add_argument("--organize_by", type=str, choices=['scene', 'camera'], default='scene', 
                        help="Choose 'scene' to merge all cameras into one folder per scene, or 'camera' to keep them split.")
    
    args = parser.parse_args()

    # Data structure to group files based on the chosen organization method
    # If 'scene': groupings[scene_name][camera_name] = [file1, file2...]
    # If 'camera': groupings[camera_name][scene_name] = [file1, file2...]
    groupings = defaultdict(lambda: defaultdict(list))

    # 1. Map out the directory structure
    for cam_dir in os.listdir(args.input_dir):
        cam_path = os.path.join(args.input_dir, cam_dir)
        if not os.path.isdir(cam_path): continue
        
        for scene_dir in os.listdir(cam_path):
            scene_path = os.path.join(cam_path, scene_dir)
            if not os.path.isdir(scene_path): continue
            
            # Fetch all valid JSONs
            files = [f for f in os.listdir(scene_path) if f.endswith('.json') and f != 'merged_processed.json']
            if not files: continue
            
            files.sort() # Ensure chronological order
            full_file_paths = [os.path.join(scene_path, f) for f in files]
            
            if args.organize_by == 'scene':
                groupings[scene_dir][cam_dir] = full_file_paths
            else:
                groupings[cam_dir][scene_dir] = full_file_paths

    # 2. Process based on the groupings
    if args.organize_by == 'scene':
        print(f"Processing in SCENE mode. Outputting to {args.output_dir}/<scene_name>/")
        for scene_name, cameras in groupings.items():
            tracks_dict = {}
            scene_img_base = ""
            
            for cam_name, json_files in cameras.items():
                img_path, _ = process_and_append_frames(json_files, tracks_dict, camera_name=cam_name)
                if img_path: scene_img_base = img_path # Keep the last valid img_path as a reference
                
            output_folder = os.path.join(args.output_dir, scene_name)
            save_tracks(tracks_dict, output_folder, scene_img_base)

    else:
        print(f"Processing in CAMERA mode. Outputting to {args.output_dir}/<camera_name>/<scene_name>/")
        for cam_name, scenes in groupings.items():
            for scene_name, json_files in scenes.items():
                tracks_dict = {}
                img_path, _ = process_and_append_frames(json_files, tracks_dict, camera_name=None)
                
                output_folder = os.path.join(args.output_dir, cam_name, scene_name)
                save_tracks(tracks_dict, output_folder, img_path)

if __name__ == "__main__":
    main()
    
# before get GT annotanions from nuscenes in valeo format:
# python python-sdk/nuscenes/scripts/export_annotanions_in_valeo_format.py --dataroot /scratch/project/eu-25-10/datasets/nuScenes/ --version 'v1.0-trainval' --output_dir /scratch/project/eu-25-10/datasets/nuScenes_metadata/annotations_in_valeo_format/    
# - - - - - - - -
    
# python scripts/00_data_prep/generate_tracks.py --input_dir /scratch/project/eu-25-10/datasets/nuScenes_metadata/annotations_in_valeo_format --output_dir /scratch/project/eu-25-10/datasets/nuScenes_metadata/tracks_by_camera --organize_by camera

# python scripts/00_data_prep/generate_tracks.py --input_dir /scratch/project/eu-25-10/datasets/nuScenes_metadata/annotations_in_valeo_format --output_dir /scratch/project/eu-25-10/datasets/nuScenes_metadata/tracks_by_scene --organize_by scene

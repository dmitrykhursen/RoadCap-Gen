import argparse
import json
import math
import os
import random
import sys
import traceback
from bisect import bisect_left
from pathlib import Path
from typing import Any, Dict, List, Optional

import regex as re
import torch
import yaml
from tqdm import tqdm
from vllm import LLM, SamplingParams

# This forces Python to look at the root directory of your project so 'src' is discoverable.
project_root = str(Path(__file__).resolve().parents[2])  # Adjust parents index if your script is deeper
if project_root not in sys.path:
    sys.path.append(project_root)
try:
    from src.utils.qas_generation_helper import run_yolo_processing
except ImportError:
    run_yolo_processing = None


# =============================================
# Helper Functions & Logic
# =============================================


def get_prefixed_permutation(options: List[str], rng: random.Random) -> List[str]:
    permuted = options.copy()
    rng.shuffle(permuted)
    letters = ["A. ", "B. ", "C. ", "D. "]
    return [f"{letters[i]}{item}" for i, item in enumerate(permuted)]


def random_permutation(q_distribution: Dict[str, int], min_length: int, max_length: int, rng: random.Random) -> List[str]:
    available_keys = [k for k, v in q_distribution.items() if v > 0]
    if not available_keys:
        return []

    target_length = min(rng.randint(min_length, max_length), sum(q_distribution.values()))
    chosen = []

    rng.shuffle(available_keys)
    for q in available_keys:
        if len(chosen) >= target_length:
            break
        chosen.append(q)
        q_distribution[q] -= 1

    while len(chosen) < target_length:
        active_keys = [k for k, v in q_distribution.items() if v > 0]
        if not active_keys:
            break
        q = rng.choice(active_keys)
        chosen.append(q)
        q_distribution[q] -= 1

    return chosen


def get_test_distribution(ratio_data: List[Dict[str, Any]], no_dir_questions: Optional[List[str]] = None) -> Dict[str, float]:
    test_distribution = {}
    for entry in ratio_data:
        if entry["ratio_test"] > 0:
            if no_dir_questions is not None and entry["question"] not in no_dir_questions:
                continue
            question = entry["question"]
            test_distribution[question] = entry["ratio_test"]
    return test_distribution


def distribute_by_ratio(question_to_ratio: Dict[str, float], total_count: int) -> Dict[str, int]:
    ratio_sum = sum(question_to_ratio.values())
    if ratio_sum == 0:
        return {}
    normalized = {q: r / ratio_sum for q, r in question_to_ratio.items()}
    counts = {q: max(1, round(normalized[q] * total_count)) for q in question_to_ratio}
    return counts


def get_past(tracks: List[Dict[str, Any]], frames: List[int], current_frame: int, frames_back: int):
    start_f = current_frame - frames_back
    end_f = current_frame + 1
    i0 = bisect_left(frames, start_f)
    i1 = bisect_left(frames, end_f)
    window = tracks[i0:i1]
    expected = set(range(start_f, end_f))
    present = {d["frame"] for d in window}
    missing = expected - present
    return window, missing


def preprocess_scene_tracks(raw_tracks: Dict[str, Any], dataset_name: str) -> tuple[Dict[str, Any], Dict[str, int]]:
    """
    Cleans overlapping cameras, sorts tracks, and builds a timestamp lookup dictionary (nuScenes only).
    """
    timestamp_to_frame = {}
    preferred_cameras = {"CAM_FRONT", "CAM_BACK"}

    for track_obj in raw_tracks["tracks"]:
        filtered_tracks_dict = {}

        for d in track_obj["track"]:
            frame_num = d["frame"]
            camera = d.get("camera", "")

            # 1. Map the timestamp to the frame number globally (ONLY for nuScenes)
            if dataset_name == "nuscenes":
                timestamp_str = d["frame_name"].split("__")[-1].split(".")[0]
                timestamp_to_frame[timestamp_str] = frame_num

            # 2. Camera Deduplication logic
            if frame_num not in filtered_tracks_dict:
                filtered_tracks_dict[frame_num] = d
            else:
                existing_camera = filtered_tracks_dict[frame_num].get("camera", "")
                if camera in preferred_cameras and existing_camera not in preferred_cameras:
                    filtered_tracks_dict[frame_num] = d

        # 3. Overwrite raw tracks with cleaned, SORTED tracks
        cleaned_list = [filtered_tracks_dict[k] for k in sorted(filtered_tracks_dict.keys())]
        track_obj["track"] = cleaned_list

        # 4. Pre-compute the integer list of frames so get_past() is blazing fast
        track_obj["frame_ints"] = [d["frame"] for d in cleaned_list]

    return raw_tracks, timestamp_to_frame


def detections_to_text(
    data: Dict[str, Any],
    camera_name: str,
    tracks_by_object_id: Optional[Dict[str, Any]] = None,
    timestamp_to_frame: Optional[Dict[str, int]] = None,
    frame: Optional[str] = None,
    frames_back: int = 3,
    track_type: str = "m",
    include_tracks: bool = False,
    dataset_name: str = "valeo",
) -> str:
    if frame is None:
        raise ValueError("Frame must be provided to extract frame number.")

    # --- Dataset-Specific Frame Extraction ---
    if dataset_name == "valeo":
        frame_idx = int(frame.split("_")[-1].split(".")[0])

    elif dataset_name == "nuscenes":
        # nuScenes: Extract timestamp and look up the integer index
        timestamp_str = frame.split(".")[0].split("__")[-1]

        if timestamp_to_frame and timestamp_str in timestamp_to_frame:
            frame_idx = timestamp_to_frame[timestamp_str]
        else:
            # Fallback just in case a timestamp is somehow missing from the tracks
            frame_idx = int(timestamp_str)
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    lines = [f"\n--- Objects detected from {camera_name} ---"]

    for category, cat_objs in data["categories"].items():
        for idx, cat_obj in enumerate(cat_objs["objects"]):
            line = f"- {camera_name}_{category}_{idx}:\n\t - current bbox: {cat_obj['bbox']}\n\t - current middle point: {cat_obj['mid_point']}."

            if include_tracks and tracks_by_object_id:
                tracks = tracks_by_object_id.get(cat_obj["id"])

                if tracks:
                    track_list = tracks["track"]
                    frames = tracks["frame_ints"]

                    window, missing = get_past(track_list, frames, frame_idx, frames_back)

                    if len(window) > 0:
                        r = range(len(window), 0, -1) if frame_idx in missing else range(len(window) - 1, -1, -1)
                        if track_type == "m":
                            if dataset_name == "nuscenes":
                                pos = [f"[{d['x']}, {d['y']}, {d['z']}, {d['depth']}] at t-{t * 55} ms" for t, d in zip(r, window)]
                            else:
                                pos = [f"[{d['x']}, {d['y']}, {d['z']}] at t-{t * 55} ms" for t, d in zip(r, window)]
                            line += f"\n\t - relative positions: {', '.join(pos)}."
                        else:
                            mid = [f"{d['center_2d_px']} at t-{t * 55} ms" for t, d in zip(r, window)]
                            line += f"\n\t - midpoints: {', '.join(mid)}."
            lines.append(line)
    return "\n".join(lines)


def generate_prompt(
    q: str,
    description: Optional[str],
    objects: str,
    config_prompts: Dict[str, str],
    directions: List[str],
    rng: random.Random,
    tracks: bool = False,
    track_type: str = "m",
) -> str:
    parts = [
        config_prompts.get("context", ""),
        config_prompts.get("obj", "") if "<obj>" in q else config_prompts.get("no_obj", ""),
        config_prompts.get("qa_generation_QWEN", ""),
    ]
    if tracks:
        parts.append(f"\n{config_prompts.get('tracks_' + track_type, '')}")

    parts.append(objects)

    if "<position>" in q:
        parts.append(f"\n{config_prompts.get('position', '')}")
    if any(x in q for x in ["important objects", "priority"]):
        parts.append(f"\n{config_prompts.get('important_objects', '')}")
    if description:
        parts.append(f"\nThe scene description is:\n {description}")
    if "following options:" in q:
        q += " " + " ".join(get_prefixed_permutation(directions, rng))

    parts.append(f"\nThe question to use is:\n {q}\n")
    return "\n".join(parts)


# =============================================
# Experiment Management & IO
# =============================================


def generate_experiment_name(args: argparse.Namespace) -> str:
    model_short = args.model.split("/")[-1]
    think_tag = "think" if args.thinking else "no-think"
    track_tag = f"tracks-{args.track_type}" if args.use_tracks else "no-tracks"
    return f"{model_short}_{args.dataset_name}_{think_tag}_{track_tag}_q{args.number_of_questions}"


def save_metadata(args: argparse.Namespace, output_folder: str, exp_name: str):
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, f"metadata_{exp_name}.json")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=4, ensure_ascii=False)


def append_json_line(path: str, obj: Any):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# =============================================
# Path & Directory Helpers
# =============================================


def get_cameras_and_scenes(yolo_path: str):
    root_path = Path(yolo_path)
    cameras = sorted([d.name for d in root_path.iterdir() if d.is_dir()])
    if not cameras:
        raise ValueError(f"No camera folders found in {yolo_path}")

    ref_camera_path = root_path / cameras[0]
    scenes = sorted([d.name for d in ref_camera_path.iterdir() if d.is_dir()])

    return cameras, scenes


def list_drive_lm_scenes(root_dir: str) -> List[str]:
    """Scans for leaf directories (scenes) that contain images/jsons."""
    root_path = Path(root_dir)
    scene_folders = []
    print(f"Scanning {root_dir} for scenes...")
    for root, dirs, files in os.walk(root_path):
        if any(f.lower().endswith(".json") for f in files):
            scene_folders.append(str(Path(root)))
    return sorted(scene_folders)


def get_output_dir(zip_path: str, args_out_dir: str) -> Path:
    """Path resolver for Valeo dataset structure."""
    zip_path_obj = Path(zip_path).resolve()
    out_root = Path(args_out_dir).resolve()
    zip_parts = zip_path_obj.parts

    for key in ["PONE_zipped", "FRONT_CAM_zipped"]:
        if key in zip_parts:
            idx = zip_parts.index(key)
            relative_path = Path(*zip_parts[idx + 1 :])
            return out_root / relative_path.with_suffix("")

    return out_root / "out"


def get_output_dir_from_nuscenes_folder(input_folder_path: str, args_out_dir: str, input_root: str) -> Path:
    """Path resolver for nuScenes dataset structure."""
    input_path = Path(input_folder_path).resolve()
    root_path = Path(input_root).resolve()
    out_root = Path(args_out_dir).resolve()
    try:
        relative_path = input_path.relative_to(root_path)
    except ValueError:
        relative_path = Path(input_path.name)
    return out_root / relative_path


# =============================================
# Model Wrapper
# =============================================


class ModelRespondervllm:
    def __init__(self, model_name: str, thinking: bool = True):
        self.thinking = thinking
        self.model_name = model_name

        self.llm = LLM(
            model=model_name,
            # dtype="bfloat16", # local problem
            gpu_memory_utilization=0.9,
            enable_chunked_prefill=True,
            max_model_len=32768,
            enable_prefix_caching=True,
            attention_backend="TRITON_ATTN", # local problem
            max_num_seqs=32,
        )
        self.tokenizer = self.llm.get_tokenizer()

        # Pre-compile sampling parameters
        if thinking:
            self.sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768)
        else:
            self.sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=32768)

    def generate_batch(self, batch_messages: List[List[Dict[str, str]]]):
        """Accepts a list of conversations and processes them in parallel."""
        outputs = self.llm.chat(
            batch_messages,
            self.sampling_params,
            use_tqdm=True,
            chat_template_kwargs={"enable_thinking": self.thinking},
        )

        results = []
        for out in outputs:
            text = out.outputs[0].text
            if "</think>" in text:
                think, answer = text.split("</think>", 1)
                think = think.replace("<think>", "").strip()
            else:
                think, answer = "", text.strip()
            results.append((think, answer.strip()))

        return results


# =============================================
# Main Execution Logic
# =============================================


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", default=None)
    parser.add_argument("--use_tracks", action="store_true")
    parser.add_argument("--track_type", choices=["m", "px"], default="m")
    parser.add_argument("--yolo_path", type=str, required=True, help="Path to annotations/metadata")
    parser.add_argument("--output_folder", type=str, default="data/test")
    parser.add_argument("--qas_ratios", type=str, required=True)
    parser.add_argument("--prompts_config", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="nuscenes")
    parser.add_argument("--number_of_questions", type=int, default=15)
    parser.add_argument("--frames_back", type=int, default=3)
    parser.add_argument("--tracks_path", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")

    parser.add_argument("--chunk_id", type=int, default=0, help="Which chunk this worker processes (0-indexed)")
    parser.add_argument("--num_chunks", type=int, default=1, help="Total number of workers/chunks")
    return parser.parse_args()


def main():
    args = parse_args()
    exp_name = generate_experiment_name(args)

    # Configuration and Model Init
    with open(args.qas_ratios, "r", encoding="utf-8") as f:
        ratio_data = json.load(f)
    with open(args.prompts_config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    q_ratios = get_test_distribution(ratio_data)
    config_prompts = config["prompts"]
    directions = ["Turn right.", "Drive backward.", "Going ahead.", "Turn left."]

    cameras, all_scenes = get_cameras_and_scenes(args.yolo_path)
    total_scenes = len(all_scenes)
    print(f"Found {len(cameras)} cameras and {total_scenes} total scenes.")
    all_scenes = sorted(all_scenes)

    # --- CHUNKING LOGIC ---
    chunk_size = math.ceil(total_scenes / args.num_chunks)
    start_idx = args.chunk_id * chunk_size
    end_idx = min(start_idx + chunk_size, total_scenes)

    my_scene_paths = all_scenes[start_idx:end_idx]
    print(f"Worker {args.chunk_id + 1}/{args.num_chunks} taking scenes {start_idx} to {end_idx - 1} ({len(my_scene_paths)} total).")

    responder = ModelRespondervllm(args.model, thinking=args.thinking)

    # Iterate through ONLY your chunk of scenes
    for local_idx, scene_name in tqdm(enumerate(my_scene_paths), total=len(my_scene_paths), desc=f"Worker {args.chunk_id + 1}/{args.num_chunks}"):
        scene_idx = start_idx + local_idx  # Global scene index for accurate logging

        # Output directory parsing
        output_dir = Path(args.output_folder) / scene_name
        print(f"\n[{scene_idx + 1}/{total_scenes}] Processing: {output_dir}")

        expected_output_file = output_dir / f"qas_{exp_name}.jsonl"

        if expected_output_file.exists() and not args.overwrite:
            print(f"Skipping: {output_dir} (Already completed)")
            print(" - - - - - " * 3)
            continue

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # ---- Load Multi-Camera Files & Merge Data ----
            cam_json_files = {}
            cam_files_lens = []
            cam_merged_data = {}

            for cam in cameras:
                cam_dir = Path(args.yolo_path) / cam / scene_name
                jsons = sorted([f for f in cam_dir.glob("*.json") if "merged" not in f.name])
                cam_json_files[cam] = jsons
                cam_files_lens.append(len(jsons))

                processed_json = cam_dir / "merged_processed.json"
                if not processed_json.exists():
                    if run_yolo_processing is None:
                        raise ImportError("run_yolo_processing could not be imported. Ensure src.utils.qas_generation_helper exists.")

                    cam_merged_data[cam] = run_yolo_processing(input_path=str(cam_dir), output_path=str(cam_dir), dataset_name=args.dataset_name)
                else:
                    with open(processed_json, "r", encoding="utf-8") as f:
                        cam_merged_data[cam] = json.load(f)

            ref_cam = cameras[0]
            if len(set(cam_files_lens)) != 1:
                print(f"Warning: Different number of frames across cameras for {scene_name}. Numbers differ: {cam_files_lens}. Skipping.")
                continue

            num_frames = len(cam_json_files[ref_cam])
            if num_frames == 0:
                print(f"No frames found for {scene_name} in {ref_cam}. Skipping.")
                continue

            # ---- Load Tracks ----
            tracks_by_id = None
            timestamp_to_frame = {}
            if args.use_tracks and args.tracks_path:
                track_path = Path(args.tracks_path) / Path(scene_name).stem / "tracks.json"
                with open(track_path, "r", encoding="utf-8") as f:
                    raw_tracks = json.load(f)

                cleaned_tracks, timestamp_to_frame = preprocess_scene_tracks(raw_tracks, args.dataset_name)
                tracks_by_id = {d["object_id"]: d for d in cleaned_tracks["tracks"]}

            q_dist = distribute_by_ratio(q_ratios, num_frames * args.number_of_questions)

            # --- Batched Output Generation with Checkpointing ---
            output_file = output_dir / f"qas_{exp_name}.jsonl"
            checkpoint_file = output_dir / f"checkpoint_{exp_name}.json"
            pattern = r"\b\w+_\d+\b"
            # items = sorted(data.items())

            frame_states = []
            start_q_idx = 0

            if checkpoint_file.exists():
                print(f"Found checkpoint for {scene_name}. Resuming...")
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    chkpt = json.load(f)
                start_q_idx = chkpt["next_q_idx"]
                frame_states = chkpt["frame_states"]

                for state in frame_states:
                    state["used_objs"] = set(state["used_objs"])
            else:
                for frame_idx in range(num_frames):
                    # Use the reference camera to define the universal scene_id/frame name
                    frame_filename = cam_json_files[ref_cam][frame_idx].name
                    scene_id = frame_filename.replace(".json", "")

                    combined_scene_text = ["Here is the list of objects detected in the scene from all cameras. Use them to generate the QA pairs:"]
                    for cam in cameras:
                        frame_stem = cam_json_files[cam][frame_idx].stem
                        cam_data = cam_merged_data[cam].get(frame_stem + ".jpg", {})

                        if not cam_data:
                            continue

                        cam_text = detections_to_text(
                            data=cam_data,
                            camera_name=cam,
                            tracks_by_object_id=tracks_by_id,
                            timestamp_to_frame=timestamp_to_frame,
                            frame=scene_id,
                            frames_back=args.frames_back,
                            track_type=args.track_type,
                            include_tracks=args.use_tracks,
                            dataset_name=args.dataset_name,
                        )
                        if cam_text.strip():
                            combined_scene_text.append(cam_text)

                    final_scene_text = "\n".join(combined_scene_text)

                    # Create a deterministic seed based on the unique scene_id
                    rng = random.Random(f"{scene_name}_{scene_id}")

                    questions = ["What are the important objects in the current scene?"] + random_permutation(
                        q_dist, args.number_of_questions - 1, args.number_of_questions - 1, rng
                    )

                    frame_states.append(
                        {"scene_id": scene_id, "scene_text": final_scene_text, "questions": questions, "used_objs": set(), "results": {}}
                    )

            if start_q_idx < args.number_of_questions:
                print(f"Loaded {len(frame_states)} frames. Batching questions {start_q_idx + 1} to {args.number_of_questions}...")

            # 2. Iterate horizontally across all frames by Question Index
            with torch.inference_mode():
                for q_idx in range(start_q_idx, args.number_of_questions):
                    batch_messages = []

                    # Build the prompt batch
                    for state in frame_states:
                        q = state["questions"][q_idx]

                        # Use the same deterministic seed for prompt generation (e.g., shuffling multiple choice)
                        rng = random.Random(f"{scene_name}_{state['scene_id']}_Q{q_idx}")

                        prompt = generate_prompt(q, None, state["scene_text"], config_prompts, directions, rng, args.use_tracks, args.track_type)

                        if state["used_objs"] and "<obj>" in q:
                            prompt += f"\nAvoid reusing: {', '.join(sorted(state['used_objs']))}"
                        batch_messages.append([{"role": "user", "content": prompt}])

                    print(f"\n--- Processing Question {q_idx + 1}/{args.number_of_questions} for {len(frame_states)} frames ---")

                    # Send entire batch to VLLM
                    batch_outputs = responder.generate_batch(batch_messages)

                    # 3. Parse outputs and update state
                    for state, (think, content) in zip(frame_states, batch_outputs):
                        try:
                            clean_content = content.strip()
                            clean_content = "{" + f'"Question_{q_idx}": ' + clean_content + "}"

                            parsed = json.loads(clean_content)
                            state["results"].update(parsed)
                            state["used_objs"].update(re.findall(pattern, content))
                        except json.JSONDecodeError as e:
                            print(f"[Warning] JSON decode error for {state['scene_id']} Q{q_idx}: {e}")
                            continue

                    # 4. Save Checkpoint after the batch finishes
                    # We copy and convert sets to lists to be JSON serializable
                    chkpt_state = []
                    for state in frame_states:
                        s_copy = state.copy()
                        s_copy["used_objs"] = list(state["used_objs"])
                        chkpt_state.append(s_copy)

                    with open(checkpoint_file, "w", encoding="utf-8") as f:
                        json.dump({"next_q_idx": q_idx + 1, "frame_states": chkpt_state}, f)

            # 5. Write all completed frames to the JSONL file once the scene is entirely done
            for state in frame_states:
                append_json_line(str(output_file), {state["scene_id"]: state["results"]})

            # 6. Clean up the checkpoint file
            if checkpoint_file.exists():
                checkpoint_file.unlink()

        except Exception as e:
            print(f"❌ Error in [{scene_idx + 1}/{total_scenes}] {scene_name}: {e}")
            traceback.print_exc()

    print("Job complete.")


if __name__ == "__main__":
    main()

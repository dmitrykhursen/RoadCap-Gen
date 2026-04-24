import argparse
import json
import math
import os
import random
import sys
import traceback
from bisect import bisect_left
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pyquaternion import Quaternion
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

INIT_QUESTION = "What are the important objects in the current scene? Those objects will be considered for the future reasoning and driving decision."

YES_NO_QUESTIONS = set(
    [
        "Would <obj> be in the moving direction of the ego vehicle?",
        "Will <obj> be in the moving direction of <obj>?",
        "Will <obj> change its motion state based on <obj>?",
        "Would <obj> take <obj> into account?",
    ]
)

OPTIONS_QUESTIONS = set(
    [
        "Predict the behavior of the ego vehicle. Please select the correct answer from the following options:",
        "What is the moving status of object <obj>? Please select the correct answer from the following options:",
    ]
)

MULTIPLE_PARTS_QUESTIONS = (
    set(
        [
            "What actions could the ego vehicle take based on <obj>? Why take this action and what's the probability?",
            "What object should the ego vehicle notice first when the ego vehicle is getting to the next possible location? What is the state of the object that is first noticed by the ego vehicle and what action should the ego vehicle take? What object should the ego vehicle notice second when the ego vehicle is getting to the next possible location? What is the state of the object perceived by the ego vehicle as second and what action should the ego vehicle take? What object should the ego vehicle notice third? What is the state of the object perceived by the ego vehicle as third and what action should the ego vehicle take?",
        ]
    )
    | OPTIONS_QUESTIONS
)

EGO_ACTIONS = set(
    [
        "In this scenario, what are dangerous actions to take for the ego vehicle?",
        "In this scenario, what are safe actions to take for the ego vehicle?",
    ]
)

NO_ADDITIONAL_INFO = (
    set(
        [
            "What actions taken by the ego vehicle can lead to a collision with <obj>?",
            "Based on <obj> in this scene, what is the most possible action of the ego vehicle?",
            "Based on the observation of <obj>, what actions may <obj> take?",
            "What is the priority of the objects that the ego vehicle should consider?(in descending order)",
        ]
    )
    | YES_NO_QUESTIONS
    | OPTIONS_QUESTIONS
    | EGO_ACTIONS
)

QUESTION_RULES = [
    (YES_NO_QUESTIONS, ["yes_no_questions"]),
    (OPTIONS_QUESTIONS, ["multiple_choice_questions"]),
    # (MULTIPLE_PARTS_QUESTIONS, ["multiple_parts"]), # add manually at the end
    # (NO_ADDITIONAL_INFO, ["no_additional_info"]), # add manually at the end
    (EGO_ACTIONS, ["actions_for_ego"]),
    (
        {INIT_QUESTION},
        ["init_q"],
    ),
    (
        {"What actions could the ego vehicle take based on <obj>? Why take this action and what's the probability?"},
        ["actions_probability"],
    ),
    (
        {"What actions taken by the ego vehicle can lead to a collision with <obj>?"},
        ["collison"],
    ),
    (
        {
            "What object should the ego vehicle notice first when the ego vehicle is getting to the next possible location? What is the state of the object that is first noticed by the ego vehicle and what action should the ego vehicle take? What object should the ego vehicle notice second when the ego vehicle is getting to the next possible location? What is the state of the object perceived by the ego vehicle as second and what action should the ego vehicle take? What object should the ego vehicle notice third? What is the state of the object perceived by the ego vehicle as third and what action should the ego vehicle take?"
        },
        ["notice_three_obj"],
    ),
    (
        {
            "Based on <obj> in this scene, what is the most possible action of the ego vehicle?",
            "Based on the observation of <obj>, what actions may <obj> take?",
        },
        ["most_possible_action"],
    ),
    (
        {"What is the priority of the objects that the ego vehicle should consider?(in descending order)"},
        ["priority_of_objects"],
    ),
]

# =============================================
# Helper Functions & Logic
# =============================================


def get_prefixed_permutation(options: List[str]) -> List[str]:
    permuted = options.copy()
    random.shuffle(permuted)
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
            camera = d["camera"]

            # 1. Map the timestamp to the frame number globally (ONLY for nuScenes)
            if dataset_name == "nuscenes":
                timestamp_str = d["frame_name"].split("__")[-1].split(".")[0]
                timestamp_to_frame[timestamp_str] = frame_num

            # 2. Camera Deduplication logic
            if frame_num not in filtered_tracks_dict:
                filtered_tracks_dict[frame_num] = d
            else:
                existing_camera = filtered_tracks_dict[frame_num]["camera"]
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
    include_tracks: bool,
    global_coords: bool,
    tracks_by_object_id: Optional[Dict[str, Any]] = None,
    timestamp_to_frame: Optional[Dict[str, int]] = None,
    frame: Optional[str] = None,
    frames_back: int = 2,
    track_type: str = "m",
    dataset_name: str = "valeo",
) -> Tuple[List[Dict[str, str | float]], Optional[List[Tuple[List[float], List[float]]]]]:
    if frame is None:
        raise ValueError("Frame must be provided to extract frame number.")

    if data.get("categories", None) is None:
        print(f"No categories found in {frame} for {camera_name}. Skipping.")
        return [], None

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

    objects_list = []
    ego_pos = None
    for category, cat_objs in data["categories"].items():
        for idx, obj in enumerate(cat_objs["objects"]):
            obj_dict = {
                "id": f"{camera_name}_{category}_{idx}",
                "bbox": obj["bbox"],
                "center": obj["mid_point"],
            }

            if include_tracks and tracks_by_object_id:
                tracks = tracks_by_object_id[obj["id"]]

                if tracks:
                    window, missing = get_past(
                        tracks["track"],
                        tracks["frame_ints"],
                        frame_idx,
                        frames_back,
                    )

                    if window:
                        if frame_idx in missing:
                            time_range = range(len(window), 0, -1)
                        else:
                            time_range = range(len(window) - 1, -1, -1)

                        if track_type == "m":
                            if dataset_name == "nuscenes":
                                coords = ["x", "y", "z"] if global_coords else ["x_ego", "y_ego", "z_ego"]
                                fields = coords + ["depth"]
                            else:
                                fields = ["x", "y", "z"]

                            def get_value_m(d):
                                return [round(d[field], 3) for field in fields]

                            get_value = get_value_m
                        else:

                            def get_value_px(d):
                                return d["center_2d_px"]

                            get_value = get_value_px

                        obj_dict["pos_history"] = [get_value(d_item) for _, d_item in zip(time_range, window)]

                        if dataset_name == "nuscenes":
                            ego_pos = [(win["ego_translation"], win["ego_rotation"]) for _, win in zip(time_range, window)]
            objects_list.append(obj_dict)
    return objects_list, ego_pos


def add_formatting(parts: list, answer_formatting: Dict[str, str], q: str) -> list[str]:
    for question_set, formatting_keys in QUESTION_RULES:
        if q in question_set:
            for key in formatting_keys:
                parts.append("\n" + answer_formatting[key])

    if q in MULTIPLE_PARTS_QUESTIONS:
        parts.append(answer_formatting["multiple_parts"])
    if q in NO_ADDITIONAL_INFO:
        parts.append(answer_formatting["no_additional_info"])

    return parts


def generate_prompt(
    q: str,
    description: Optional[str],
    objects: str,
    config_prompts: Dict[str, str],
    config_prompts_fewshot: Dict[str, str],
    config_prompts_answer_formatting: Dict[str, str],
    enforce_formatting: bool,
    directions: List[str],
    global_coords: bool,
    tracks: bool,
    track_type: str = "m",
    dataset_name: str = "nuscenes",
) -> str:
    parts = [
        config_prompts["context"],
        config_prompts["answer_rules"],
    ]

    if "<obj>" in q or q in MULTIPLE_PARTS_QUESTIONS or q in INIT_QUESTION:
        parts.append(config_prompts["obj"])
        parts.append(config_prompts_fewshot["obj"])
    else:
        parts.append(config_prompts["no_obj"])

    parts.append(config_prompts["coordinate_system"])

    if tracks:
        parts.append(f"\n{config_prompts[dataset_name + '_tracks_coords_' + ('global_' if global_coords else '') + track_type]}")

    parts.append(f"\n{config_prompts['detected_objects' + ('_tracks' if tracks else '')]}")

    parts.append(objects)

    if dataset_name == "valeo":
        parts.append(config_prompts["valeo_dataset_car_in_view"])

    if "<position>" in q:
        parts.append("\n" + config_prompts["position"])

    if any(x in q for x in ["important objects", "priority"]):
        parts.append("\n" + config_prompts["important_objects"])

    if description:
        parts.append(f"\nThe scene description is:\n {description}")

    if "following options:" in q:
        parts.append(f"\nThe question to use is:\n {q + ' ' + ' '.join(get_prefixed_permutation(directions))}\n")
    else:
        parts.append(f"\nThe question to use is:\n {q}\n")

    if enforce_formatting:
        parts = add_formatting(parts, config_prompts_answer_formatting, q)

    return "".join(parts)


# =============================================
# Experiment Management & IO
# =============================================


def generate_experiment_name(args: argparse.Namespace) -> str:
    model_short = args.model.split("/")[-1]
    think_tag = "think" if args.thinking else "no-think"
    track_tag = f"tracks-{args.track_type}" if args.tracks_path is not None else "no-tracks"
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
            dtype="bfloat16",  # local problem
            gpu_memory_utilization=0.95,
            enable_chunked_prefill=True,
            max_model_len=32768,
            enable_prefix_caching=True,
            # attention_backend="TRITON_ATTN",  # local problem
        )
        self.tokenizer = self.llm.get_tokenizer()

        # Pre-compile sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.6 if thinking else 0.7,
            top_p=0.95 if thinking else 0.8,
            top_k=20,
            max_tokens=32768,
        )

    def generate(self, messages: List[Dict[str, str]]):
        """Accepts a list of conversations and processes them in parallel."""
        outputs = self.llm.chat(
            messages,  # type: ignore
            self.sampling_params,
            use_tqdm=True,
            chat_template_kwargs={"enable_thinking": self.thinking},
        )

        text = outputs[0].outputs[0].text

        if "</think>" in text:
            think, answer = text.split("</think>", 1)
            think = think.replace("<think>", "").strip()
        else:
            think, answer = "", text.strip()

        return think, answer.strip()

    def generate_batch(self, batch_messages: List[List[Dict[str, str]]], chunk_size: int = 2):
        """Processes massive batches by chunking them to avoid OOM spikes, preserving original order."""
        all_results = []

        for i in range(0, len(batch_messages), chunk_size):
            chunk = batch_messages[i : i + chunk_size]

            outputs = self.llm.chat(
                chunk,  # type: ignore
                self.sampling_params,
                use_tqdm=True,
                chat_template_kwargs={"enable_thinking": self.thinking},
            )

            for out in outputs:
                text = out.outputs[0].text
                if "</think>" in text:
                    think, answer = text.split("</think>", 1)
                    think = think.replace("<think>", "").strip()
                else:
                    think, answer = "", text.strip()

                all_results.append((think, answer.strip()))

        return all_results


def estimate_prompt_tokens(responder, prompt: str, thinking: bool) -> int:
    """
    Estimates the number of tokens a single prompt will produce after applying the chat template.
    """
    messages = [{"role": "user", "content": prompt}]
    return len(
        responder.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": thinking},
        )
    )


# =============================================
# Main Execution Logic
# =============================================


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--track_type", choices=["m", "px"], default="m")
    parser.add_argument("--yolo_path", type=str, required=True, help="Path to annotations/metadata")
    parser.add_argument("--output_folder", type=str, default="data/nuscenes_tracks")
    parser.add_argument("--qas_ratios", type=str, required=True)
    parser.add_argument("--prompts_config", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="nuscenes")
    parser.add_argument("--number_of_questions", type=int, default=15)
    parser.add_argument("--frames_back", type=int, default=2)  # 2 back and the current so 3
    parser.add_argument("--tracks_path", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    parser.add_argument("--no_init_q", action="store_true", help="Dont include the init question")
    parser.add_argument("--answer_formatting", action="store_true", help="Try to enforce DriveLM answer formatting")
    parser.add_argument("--scene_prefix", type=str, default=None, help="Filter scenes by prefix (e.g., 'n008' or 'n015')")
    parser.add_argument(
        "--global_coords",
        action="store_true",
        help="When using the nuScenes dataset with tracks, use global coordinates instead of ego-centric ones.",
    )

    # --- New Test Argument ---
    parser.add_argument(
        "--test", action="store_true", help="Test mode: generate 5 of each q with test_ratio > 0.5 uniformly in one folder, print prompts."
    )

    # --- Chunking Arguments for SLURM Arrays ---
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

    # --- Test Mode Initialization ---
    test_question_pool = []
    if args.test:
        target_questions = [entry["question"] for entry in ratio_data if entry.get("ratio_test", 0) > 0.5]
        target_questions.append(INIT_QUESTION)
        test_question_pool = target_questions * 4
        rng_test = random.Random(42)  # fixed seed to ensure uniformity
        rng_test.shuffle(test_question_pool)
        print(f"\n[TEST MODE ENABLED] Selected {len(target_questions)} question types. Pool size: {len(test_question_pool)}.")

    q_ratios = get_test_distribution(ratio_data)
    config_prompts = config["prompts"]
    config_prompts_fewshot = config["fewshot"]
    config_prompts_answer_formatting = config["answer_formatting"]
    directions = ["Turn right.", "Drive backward.", "Going ahead.", "Turn left."]

    cameras, all_scenes = get_cameras_and_scenes(args.yolo_path)
    if args.scene_prefix:
        all_scenes = [s for s in all_scenes if s.startswith(args.scene_prefix)]
        print(f"Filtering scenes with prefix: '{args.scene_prefix}'")

    total_scenes = len(all_scenes)
    chunk_size = math.ceil(total_scenes / args.num_chunks)
    start_idx = args.chunk_id * chunk_size
    end_idx = min(start_idx + chunk_size, total_scenes)

    print(f"Found {len(cameras)} cameras and {total_scenes} total scenes.")
    all_scenes = sorted(all_scenes)
    all_scenes = all_scenes[start_idx:end_idx]
    print(f"Filtering scenes to index range: [{start_idx}, {end_idx})")
    total_scenes = len(all_scenes)

    responder = ModelRespondervllm(args.model, thinking=args.thinking)

    for scene_idx, scene_name in tqdm(enumerate(all_scenes), total=total_scenes):
        # --- Output Directory Parsing ---
        if args.test:
            output_dir = Path("data/prompt_testing")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"test_results_{exp_name}.jsonl"
        else:
            output_dir = Path(args.output_folder) / scene_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"qas_{scene_name}_{exp_name}.jsonl"

        try:
            # ---- Load Multi-Camera Files & Merge Data ----
            cam_json_files = {}
            cam_files_lens = []
            for cam in cameras:
                cam_dir = Path(args.yolo_path) / cam / scene_name
                jsons = sorted([f for f in cam_dir.glob("*.json") if "merged" not in f.name])
                cam_json_files[cam] = jsons
                cam_files_lens.append(len(jsons))

            ref_cam = cameras[0]
            if len(set(cam_files_lens)) != 1:
                print(f"Warning: Different number of frames across cameras for {scene_name}. Numbers differ: {cam_files_lens}. Skipping.")
                continue

            num_frames = len(cam_json_files[ref_cam])
            if num_frames == 0:
                print(f"No frames found for {scene_name} in {ref_cam}. Skipping.")
                continue

            # Check for existing progress to resume mid-scene
            processed_frames = set()
            if output_file.exists() and not args.overwrite and not args.test:
                with open(output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            processed_frames.update(data.keys())
                        except Exception:
                            pass  # Skip corrupted lines

                if len(processed_frames) >= num_frames:
                    print(f"[{scene_idx + 1}/{total_scenes}] Skipping: {scene_name} (Fully processed {len(processed_frames)}/{num_frames} frames)")
                    continue
                else:
                    print(f"[{scene_idx + 1}/{total_scenes}] Resuming: {scene_name} ({len(processed_frames)}/{num_frames} frames done)")
            else:
                print(f"[{scene_idx + 1}/{total_scenes}] Processing: {scene_name}")

            # ---- Load Tracks ----
            tracks_by_id = None
            timestamp_to_frame = {}
            if args.tracks_path:
                track_path = (
                    Path(args.tracks_path)
                    / Path(scene_name).stem
                    / ("tracks" + ("_ego_centric" if args.dataset_name == "nuscenes" else "") + ".json")
                )
                with open(track_path, "r", encoding="utf-8") as f:
                    raw_tracks = json.load(f)

                cleaned_tracks, timestamp_to_frame = preprocess_scene_tracks(raw_tracks, args.dataset_name)
                tracks_by_id = {d["object_id"]: d for d in cleaned_tracks["tracks"]}

            q_dist = distribute_by_ratio(q_ratios, num_frames * args.number_of_questions - (not args.no_init_q))
            pattern = r"\b\S*\.\w+_\d+\b"

            for frame_idx in tqdm(range(num_frames), desc=f"Frames for {scene_name}"):
                # When using tracks, skip the first two frames where tracks are not yet available
                if args.tracks_path is not None and frame_idx < 2:
                    continue

                frame_master_id = cam_json_files[ref_cam][frame_idx].stem
                if frame_master_id in processed_frames and not args.test:
                    continue

                prefix = "Here is the list of objects detected in the scene from all cameras. Use them to generate the QA pairs:"
                combined_scene_text = []
                ego_poses_cams = []
                for cam in cameras:
                    cam_dir = Path(args.yolo_path) / cam / scene_name
                    processed_json = cam_dir / "merged_processed.json"
                    if not processed_json.exists():
                        if run_yolo_processing is None:
                            raise ImportError("run_yolo_processing could not be imported. Ensure src.utils.qas_generation_helper exists.")

                        data = run_yolo_processing(input_path=str(cam_dir), output_path=str(cam_dir), dataset_name=args.dataset_name)
                    else:
                        with open(processed_json, "r", encoding="utf-8") as f:
                            data = json.load(f)

                    frame_stem = cam_json_files[cam][frame_idx].stem
                    cam_data = data.get(frame_stem + ".jpg", {})
                    cam_text, ego_pos = detections_to_text(
                        data=cam_data,
                        camera_name=cam,
                        tracks_by_object_id=tracks_by_id,
                        timestamp_to_frame=timestamp_to_frame,
                        frame=frame_stem,
                        frames_back=args.frames_back,
                        track_type=args.track_type,
                        include_tracks=args.tracks_path is not None,
                        dataset_name=args.dataset_name,
                        global_coords=args.global_coords,
                    )
                    if cam_text:
                        combined_scene_text.extend(cam_text)

                    if ego_pos is not None:
                        ego_poses_cams.append(ego_pos)
                object_lines = [json.dumps(obj, separators=(",", ":")) for obj in combined_scene_text]
                json_data = "[\n" + ",\n".join(object_lines) + "\n]"

                if ego_poses_cams:
                    ego_poses = None
                    for pos in ego_poses_cams:
                        if len(pos) == args.frames_back:
                            ego_poses = pos
                            break
                        elif pos is not None:
                            ego_poses = pos

                    if ego_poses is None:
                        print(f"Warning: No ego positions found for {scene_name} frame {frame_idx}")
                        continue
                    if not args.global_coords:
                        poses = [np.array(translation) - np.array(ego_poses[-1][0]) for translation, _ in ego_poses]
                        poses = np.array([Quaternion(rotation[1]).inverse.rotate(pos) for pos, rotation in zip(poses, ego_poses)])
                        prefix = f"\nEgo vehicle pos_history (x, y, z): {[np.round(pos, 3).tolist() for pos in poses]} \n" + prefix

                    if args.global_coords:
                        prefix = (
                            f"\nEgo vehicle pos_history (x, y, z): {[np.round(translation, 3).tolist() for translation, _ in ego_poses]} \n" + prefix
                        )
                        prefix = (
                            f"Heading (Yaw): {[f'{round(np.degrees(Quaternion(rotation).yaw_pitch_roll[0]), 2)}°' for _, rotation in ego_poses]} (0° is East/Global +X, 90° is North/Global +Y)"
                            + prefix
                        )
                else:
                    print(f"Warning: No ego positions found for {scene_name} frame {frame_idx}")
                    continue
                final_scene_text = prefix + "\n" + json_data

                # Create a deterministic seed based on the scene
                rng = random.Random(f"{scene_name}_{frame_idx}")

                # --- Test Mode Question Assignment ---
                if args.test:
                    frames_remaining = num_frames - frame_idx
                    num_to_pop = math.ceil(len(test_question_pool) / frames_remaining) if frames_remaining > 0 else 0
                    questions = test_question_pool[:num_to_pop]
                    test_question_pool = test_question_pool[num_to_pop:]
                    if len(questions) == 0:
                        break
                else:
                    base = [] if args.no_init_q else [INIT_QUESTION]
                    remaining = args.number_of_questions - len(base)
                    questions = base + random_permutation(q_dist, remaining, remaining, rng)

                scene_results = {}
                used_objs = set()
                with torch.inference_mode():
                    for i, q in enumerate(questions):
                        prompt = generate_prompt(
                            q=q,
                            description=None,
                            objects=final_scene_text,
                            config_prompts=config_prompts,
                            config_prompts_fewshot=config_prompts_fewshot,
                            config_prompts_answer_formatting=config_prompts_answer_formatting,
                            enforce_formatting=args.answer_formatting,
                            directions=directions,
                            global_coords=args.global_coords,
                            tracks=args.tracks_path is not None,
                            track_type=args.track_type,
                            dataset_name=args.dataset_name,
                        )

                        if used_objs and "<obj>" in q:
                            prompt += f"\nAvoid reusing: {', '.join(sorted(used_objs))}"

                        # --- Print Prompt in Test Mode ---
                        if args.test:
                            print(f"\n\n{'=' * 70}\n[TEST MODE] Prompt for: {q}\n{'-' * 70}\n{prompt}\n{'=' * 70}\n")
                        messages = [{"role": "user", "content": prompt}]
                        _, content = responder.generate(messages)
                        try:
                            clean_content = content.strip()
                            clean_content = "{" + f'"Question_{i}": ' + clean_content + "}"
                            parsed = json.loads(clean_content)
                            scene_results.update(parsed)
                            question_text = parsed[f"Question_{i}"]["question"]
                            used_objs.update(re.findall(pattern, question_text))
                            if args.test:
                                print("Next <obj> prompt will have:")
                                print(f"\nAvoid reusing: {', '.join(sorted(used_objs))}")
                        except json.JSONDecodeError as e:
                            print(f"[Warning] JSON decode error for {scene_name} frame {frame_idx} question {i}: {e}\n{content}")

                append_json_line(str(output_file), {frame_master_id: scene_results})

            # --- Early Exit for Test Mode ---
            if args.test:
                print(f"\nTest mode finished processing folder '{scene_name}'. Exiting.")
                break

        except Exception as e:
            print(f"❌ Error in [{scene_idx + 1}/{total_scenes}] {scene_name}: {e}")
            print(traceback.print_exc())

    print("Job complete.")


if __name__ == "__main__":
    main()

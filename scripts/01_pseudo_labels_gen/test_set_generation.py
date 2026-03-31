import argparse
import json
import random
import traceback
from bisect import bisect_left
from pathlib import Path

import regex as re
import torch
import yaml
from tqdm import tqdm
from vllm import LLM, SamplingParams

# sbatch --job-name=n008_generation qa_test_gen.sh --yolo_path="/scratch/project/eu-25-10/datasets/nuScenes_metadata/annotations_in_valeo_format" --qas_counts_dir="distributed_counts" --prompts_config="configs/inference/llm_prompt_config.yaml" --scene_prefix n008 --thinking --scene_idx_start 0 --scene_idx_end 2
# sbatch --job-name=n015_generation qa_test_gen.sh --yolo_path="/scratch/project/eu-25-10/datasets/nuScenes_metadata/annotations_in_valeo_format" --qas_counts_dir="distributed_counts" --prompts_config="configs/inference/llm_prompt_config.yaml" --scene_prefix n015 --thinking

# =============================================
# Helper Functions & Logic
# =============================================


def get_prefixed_permutation(options):
    permuted = options.copy()
    random.shuffle(permuted)
    letters = ["A. ", "B. ", "C. ", "D. "]
    return [f"{letters[i]}{item}" for i, item in enumerate(permuted)]


def get_past(tracks, frames, current_frame, frames_back):
    start_f = current_frame - frames_back
    end_f = current_frame + 1
    i0 = bisect_left(frames, start_f)
    i1 = bisect_left(frames, end_f)
    window = tracks[i0:i1]
    expected = set(range(start_f, end_f))
    present = {d["frame"] for d in window}
    missing = expected - present
    return window, missing


def detections_to_text(data, cam_name, frame_idx, tracks_by_object_id=None, frames_back=3, track_type="m", include_tracks=False):
    lines = [f"\n--- Camera: {cam_name} ---"]
    if "categories" not in data:
        return ""

    for category, cat_objs in data["categories"].items():
        for idx, cat_obj in enumerate(cat_objs["objects"]):
            line = f"- {cam_name}_{category}_{idx}:\n\t - current bbox: {cat_obj['bbox']}\n\t - current middle point: {cat_obj['mid_point']}."
            if include_tracks and tracks_by_object_id:
                tracks = tracks_by_object_id.get(cat_obj.get("id"))
                if tracks:
                    track_list = tracks["track"]
                    frames = [d["frame"] for d in track_list]
                    window, missing = get_past(track_list, frames, frame_idx, frames_back)
                    if len(window) > 0:
                        r = range(len(window), 0, -1) if frame_idx in missing else range(len(window) - 1, -1, -1)
                        if track_type == "m":
                            pos = [f"[{d['x']}, {d['y']}, {d['z']}] at t-{t * 55} ms" for t, d in zip(r, window)]
                            line += f"\n\t - relative positions: {', '.join(pos)}."
                        else:
                            mid = [f"{d['center_2d_px']} at t-{t * 55} ms" for t, d in zip(r, window)]
                            line += f"\n\t - midpoints: {', '.join(mid)}."
            lines.append(line)
    return "\n".join(lines)


def generate_prompt(q, description, objects, config_prompts, directions, tracks=False, track_type="m"):
    parts = [
        config_prompts["context"],
        config_prompts["obj"] if "<obj>" in q else config_prompts["no_obj"],
        config_prompts["qa_generation_QWEN"],
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
        q += " " + " ".join(get_prefixed_permutation(directions))
    parts.append(f"\nThe question to use is:\n {q}\n")
    return "\n".join(parts)


def random_permutation(q_distribution, target_length):
    available_keys = [k for k, v in q_distribution.items() if v > 0]
    if not available_keys or target_length <= 0:
        return []

    L = min(target_length, sum(q_distribution.values()))

    chosen = []
    random.shuffle(available_keys)
    # 1. Grab unique options first
    for q in available_keys:
        if len(chosen) >= L:
            break
        chosen.append(q)
        q_distribution[q] -= 1

    # 2. Fill the rest if repeats are needed
    while len(chosen) < L:
        active_keys = [k for k, v in q_distribution.items() if v > 0]
        if not active_keys:
            break
        q = random.choice(active_keys)
        chosen.append(q)
        q_distribution[q] -= 1

    return chosen


# =============================================
# Multi-Camera Directory Helpers
# =============================================


def get_cameras_and_scenes(yolo_path: str):
    root_path = Path(yolo_path)
    cameras = sorted([d.name for d in root_path.iterdir() if d.is_dir()])
    if not cameras:
        raise ValueError(f"No camera folders found in {yolo_path}")

    ref_camera_path = root_path / cameras[0]
    scenes = sorted([d.name for d in ref_camera_path.iterdir() if d.is_dir()])

    return cameras, scenes


def append_json_line(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def generate_experiment_name(args):
    model_short = args.model.split("/")[-1]
    think_tag = "think" if args.thinking else "no-think"
    track_tag = f"tracks-{args.track_type}" if args.use_tracks else "no-tracks"
    return f"{model_short}_{args.dataset_name}_{think_tag}_{track_tag}_dynamic_q"


# =============================================
# Model Wrapper
# =============================================


class ModelRespondervllm:
    def __init__(self, model_name, thinking=True):
        self.thinking = thinking
        self.model_name = model_name

        self.llm = LLM(
            model=model_name,
            dtype="bfloat16",
            gpu_memory_utilization=0.95,
            enable_chunked_prefill=True,
            max_model_len=32768,
            enable_prefix_caching=True,
        )
        self.tokenizer = self.llm.get_tokenizer()

        if thinking:
            self.sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768)
        else:
            self.sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=32768)

    def generate(self, messages):
        outputs = self.llm.chat(
            messages,
            self.sampling_params,
            chat_template_kwargs={
                "enable_thinking": self.thinking,
            },
        )
        text = outputs[0].outputs[0].text

        if "</think>" in text:
            think, answer = text.split("</think>", 1)
            think = think.replace("<think>", "").strip()
        else:
            think, answer = "", text.strip()

        return think, answer.strip()


# =============================================
# Main Execution Logic
# =============================================


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_name", type=str, default="QA-Gen")
    parser.add_argument("--use_tracks", action="store_true")
    parser.add_argument("--track_type", choices=["m", "px"], default="m")

    # NEW: Argument to filter scenes
    parser.add_argument("--scene_prefix", type=str, default=None, help="Filter scenes by prefix (e.g., 'n008' or 'n015')")
    parser.add_argument("--scene_idx_start", type=int, default=None)
    parser.add_argument("--scene_idx_end", type=int, default=None)

    parser.add_argument("--yolo_path", type=str, required=True, help="Path to /annotations_in_valeo_format containing the 6 camera folders")
    parser.add_argument("--output_folder", type=str, default="data/output")
    parser.add_argument("--qas_counts_dir", type=str, required=True, help="Path to folder with pre-calculated JSON counts per scene")
    parser.add_argument("--prompts_config", type=str, required=True)

    parser.add_argument("--dataset_name", type=str, default="nuscenes")
    parser.add_argument("--frames_back", type=int, default=3)
    parser.add_argument("--tracks_path", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results entirely")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exp_name = generate_experiment_name(args)

    config = yaml.safe_load(open(args.prompts_config))
    config_prompts = config["prompts"]
    directions = ["Turn right.", "Drive backward.", "Going ahead.", "Turn left."]

    cameras, all_scenes = get_cameras_and_scenes(args.yolo_path)

    # Filter scenes if prefix is provided
    if args.scene_prefix:
        all_scenes = [s for s in all_scenes if s.startswith(args.scene_prefix)]
        print(f"🔍 Filtering scenes with prefix: '{args.scene_prefix}'")

    total_scenes = len(all_scenes)
    print(f"🎬 Found {total_scenes} total scenes to process.")

    all_scenes = sorted(all_scenes)
    if args.scene_idx_start is not None and args.scene_idx_end is not None:
        if args.scene_idx_end > total_scenes:
            print(f"⚠️ scene_idx_end {args.scene_idx_end} is greater than total scenes {total_scenes}. Adjusting to {total_scenes}.")
            args.scene_idx_end = total_scenes
        all_scenes = all_scenes[args.scene_idx_start : args.scene_idx_end]
        print(f"🔍 Further filtering scenes to index range: [{args.scene_idx_start}, {args.scene_idx_end})")
        print(f"🎬 Now processing {len(all_scenes)} scenes after index filtering.")

    responder = ModelRespondervllm(args.model, thinking=args.thinking)

    total_scenes = len(all_scenes)
    for iz, scene_name in tqdm(enumerate(all_scenes), total=total_scenes, desc="Scenes overall"):
        output_dir = Path(args.output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"qas_{scene_name}_{exp_name}.jsonl"
        try:
            counts_file = Path(args.qas_counts_dir) / f"{scene_name}.json"
            if not counts_file.exists():
                print(f"⚠️ Missing counts file for {scene_name}. Skipping.")
                continue
            # 1. File gathering and frame counting
            cam_json_files = {}
            cam_files_lens = []
            for cam in cameras:
                cam_dir = Path(args.yolo_path) / cam / scene_name
                jsons = sorted([f for f in cam_dir.glob("*.json") if "merged" not in f.name])
                cam_json_files[cam] = jsons
                cam_files_lens.append(len(jsons))

            ref_cam = cameras[0]
            # Check that all cameras have the same number of frames for this scene
            if len(set(cam_files_lens)) != 1:
                print("Numbers differ:", cam_files_lens)
                print(f"⚠️ Warning: Different number of frames across cameras for {scene_name}. Skipping this scene.")
                continue

            num_frames = len(cam_json_files[ref_cam])

            if num_frames == 0:
                print(f"⚠️ No frames found for {scene_name} in {ref_cam}. Skipping.")
                continue

            # 2. Check for existing progress to resume mid-scene
            processed_frames = set()
            if output_file.exists() and not args.overwrite:
                with open(output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            processed_frames.update(data.keys())
                        except Exception:
                            pass  # Skip corrupted lines

                if len(processed_frames) >= num_frames:
                    print(f"⏩ [{iz + 1}/{total_scenes}] Skipping: {scene_name} (Fully processed {len(processed_frames)}/{num_frames} frames)")
                    continue
                else:
                    print(f"🔄 [{iz + 1}/{total_scenes}] Resuming: {scene_name} ({len(processed_frames)}/{num_frames} frames done)")
            else:
                print(f"🚀 [{iz + 1}/{total_scenes}] Processing: {scene_name}")

            # 3. Load counts and tracks

            q_dist = json.load(open(counts_file))
            total_questions_needed = sum(q_dist.values())

            tracks_by_id = None
            if args.use_tracks and args.tracks_path:
                raw_tracks = json.load(open(args.tracks_path))
                tracks_by_id = {d["object_id"]: d for d in raw_tracks["tracks"]}

            # 4. Deterministic Question Allocation
            base_q_per_frame = total_questions_needed // num_frames
            remainder_q = total_questions_needed % num_frames

            # Seed the random generator specifically for this scene so the question
            # distribution remains perfectly identical even if the script crashes and restarts
            random.seed(scene_name)

            frame_questions_list = []
            for frame_idx in range(num_frames):
                num_q_this_frame = base_q_per_frame + (1 if frame_idx < remainder_q else 0)
                qs = random_permutation(q_dist, num_q_this_frame)
                frame_questions_list.append(qs)

            random.seed()  # Reset to truly random behavior for everything else

            pattern = r"\b\w+_\d+\b"

            # 5. Inference Loop
            for frame_idx in tqdm(range(num_frames), desc=f"Frames for {scene_name}"):
                frame_master_id = cam_json_files[ref_cam][frame_idx].stem
                # Skip if we already successfully wrote this frame to the JSONL
                if frame_master_id in processed_frames:
                    continue
                combined_scene_text = ["Here is the list of objects detected in the scene from all cameras. Use them to generate the QA pairs:"]
                for cam in cameras:
                    processed_json = Path(args.yolo_path) / cam / scene_name / "merged_processed.json"
                    if not processed_json.exists():
                        # TODO: I dont know how to actually use this, the import crashes cause cant find src
                        from src.utils.qas_generation_helper import run_yolo_processing

                        data = run_yolo_processing(input_path=scene_name, output_path=scene_name, dataset_name=args.dataset_name)
                    else:
                        data = json.load(open(processed_json))

                    cam_data = data.get(cam_json_files[cam][frame_idx].stem + ".jpg", {})
                    cam_text = detections_to_text(cam_data, cam, frame_idx, tracks_by_id, args.frames_back, args.track_type, args.use_tracks)
                    if cam_text.strip():
                        combined_scene_text.append(cam_text)

                final_scene_text = "\n".join(combined_scene_text)
                # Retrieve the pre-allocated questions for this frame
                questions = frame_questions_list[frame_idx]
                scene_results = {}
                used_objs = set()
                with torch.inference_mode():
                    for i, q in enumerate(questions):
                        prompt = generate_prompt(q, None, final_scene_text, config_prompts, directions, args.use_tracks, args.track_type)
                        if used_objs and "<obj>" in q:
                            prompt += f"\nAvoid reusing: {', '.join(sorted(used_objs))}"

                        messages = [{"role": "user", "content": prompt}]
                        _, content = responder.generate(messages)

                        try:
                            clean_content = content.strip()
                            clean_content = "{" + f'"Question_{i}": ' + clean_content + "}"
                            parsed = json.loads(clean_content)
                            scene_results.update(parsed)
                            used_objs.update(re.findall(pattern, content))
                        except Exception as e:
                            print(f"Error parsing content for question {i}: {e}")
                            continue
                append_json_line(str(output_file), {frame_master_id: scene_results})
        except Exception as e:
            print(f"❌ Error in [{iz + 1}/{total_scenes}] {scene_name}: {e}")
            traceback.print_exc()

    print("Job complete.")

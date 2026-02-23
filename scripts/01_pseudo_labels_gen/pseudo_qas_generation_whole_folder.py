import argparse
import atexit
import json
import os
import random
import tempfile
from bisect import bisect_left
from pathlib import Path
import demjson3
import regex as re
import torch
import yaml
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

# =============================================
# Helper Functions & Logic
# =============================================

def get_prefixed_permutation(options):
    permuted = options.copy()
    random.shuffle(permuted)
    letters = ["A. ", "B. ", "C. ", "D. "]
    return [f"{letters[i]}{item}" for i, item in enumerate(permuted)]

def get_test_distribution(ratio_data, no_dir_questions=None):
    test_distribution = {}
    for entry in ratio_data:
        if entry["ratio_test"] > 0:
            if no_dir_questions is not None and entry["question"] not in no_dir_questions:
                continue
            question = entry["question"]
            test_distribution[question] = entry["ratio_test"]
    return test_distribution

def distribute_by_ratio(question_to_ratio, total_count):
    ratio_sum = sum(question_to_ratio.values())
    normalized = {q: r / ratio_sum for q, r in question_to_ratio.items()}
    counts = {q: max(1, round(normalized[q] * total_count)) for q in question_to_ratio}
    return counts

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

def detections_to_text(data, tracks_by_object_id=None, frame=None, frames_back=3, track_type="m", include_tracks=False, dataset_name="valeo"):
    if dataset_name == "valeo":
        frame_idx = int(frame.split("_")[-1].split(".")[0])
    elif dataset_name == "nuscenes":
        frame_idx = int(frame.split(".")[0].split("_")[-1])
    
    lines = ["Here is the list of objects detected in the scene. Use them to generate the QA pairs:"]
    for category, cat_objs in data["categories"].items():
        for idx, cat_obj in enumerate(cat_objs["objects"]):
            line = f"- {category}_{idx}:\n\t - current bbox: {cat_obj['bbox']}\n\t - current middle point: {cat_obj['mid_point']}."
            if include_tracks and tracks_by_object_id:
                tracks = tracks_by_object_id.get(cat_obj["id"])
                if tracks:
                    track_list = tracks["track"]
                    frames = [d["frame"] for d in track_list]
                    window, missing = get_past(track_list, frames, frame_idx, frames_back)
                    if len(window) > 0:
                        r = range(len(window), 0, -1) if frame_idx in missing else range(len(window)-1, -1, -1)
                        if track_type == "m":
                            pos = [f"[{d['x']}, {d['y']}, {d['z']}] at t-{t*55} ms" for t, d in zip(r, window)]
                            line += f"\n\t - relative positions: {', '.join(pos)}."
                        else:
                            mid = [f"{d['center_2d_px']} at t-{t*55} ms" for t, d in zip(r, window)]
                            line += f"\n\t - midpoints: {', '.join(mid)}."
            lines.append(line)
    return "\n".join(lines)

def generate_prompt(q, description, objects, config_prompts, directions, tracks=False, track_type="m"):
    parts = [config_prompts["context"], config_prompts["obj"] if "<obj>" in q else config_prompts["no_obj"], config_prompts["qa_generation_QWEN"]]
    if tracks: parts.append(f"\n{config_prompts.get('tracks_' + track_type, '')}")
    parts.append(objects)
    if "<position>" in q: parts.append(f"\n{config_prompts.get('position', '')}")
    if any(x in q for x in ["important objects", "priority"]): parts.append(f"\n{config_prompts.get('important_objects', '')}")
    if description: parts.append(f"\nThe scene description is:\n {description}")
    if "following options:" in q: q += " " + " ".join(get_prefixed_permutation(directions))
    parts.append(f"\nThe question to use is:\n {q}\n")
    return "\n".join(parts)

# =============================================
# Experiment Management & IO
# =============================================

def generate_experiment_name(args):
    model_short = args.model.split('/')[-1]
    think_tag = "think" if args.thinking else "no-think"
    track_tag = f"tracks-{args.track_type}" if args.use_tracks else "no-tracks"
    return f"{model_short}_{args.dataset_name}_{think_tag}_{track_tag}_q{args.number_of_questions}"

def save_metadata(args, output_folder, exp_name):
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, f"metadata_{exp_name}.json")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=4, ensure_ascii=False)

def append_json_line(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def random_permutation(q_distribution, min_length=15, max_length=15):
    available_keys = [k for k, v in q_distribution.items() if v > 0]
    if not available_keys: return []
    L = min(random.randint(min_length, max_length), sum(q_distribution.values()))
    
    chosen = []
    # Mix of unique and then repeats
    random.shuffle(available_keys)
    for q in available_keys:
        if len(chosen) >= L: break
        chosen.append(q)
        q_distribution[q] -= 1
    
    while len(chosen) < L:
        active_keys = [k for k, v in q_distribution.items() if v > 0]
        if not active_keys: break
        q = random.choice(active_keys)
        chosen.append(q)
        q_distribution[q] -= 1
    return chosen

# =============================================
# Path & Directory Helpers
# =============================================

def list_drive_lm_scenes(root_dir: str) -> list[str]:
    """Scans for leaf directories (scenes) that contain images/jsons."""
    root_path = Path(root_dir)
    scene_folders = []
    print(f"🔍 Scanning {root_dir} for scenes...")
    for root, dirs, files in os.walk(root_path):
        if any(f.lower().endswith(('.json')) for f in files):
            scene_folders.append(str(Path(root)))
    return sorted(scene_folders)

def get_output_dir(zip_path, args_out_dir):
    """Path resolver for Valeo dataset structure."""
    zip_path = Path(zip_path).resolve()
    out_root = Path(args_out_dir).resolve()
    zip_parts = zip_path.parts
    if "PONE_zipped" in zip_parts:
        idx = zip_parts.index("PONE_zipped")
        relative_path = Path(*zip_parts[idx + 1:])
        output_dir = out_root / relative_path.with_suffix('')
    elif "FRONT_CAM_zipped" in zip_parts:
        idx = zip_parts.index("FRONT_CAM_zipped")
        relative_path = Path(*zip_parts[idx + 1:])
        output_dir = out_root / relative_path.with_suffix('')
    else:
        output_dir = out_root / "out"
        
    return output_dir

def get_output_dir_from_nuscenes_folder(input_folder_path, args_out_dir, input_root):
    """Path resolver for nuScenes dataset structure."""
    input_path = Path(input_folder_path).resolve()
    root_path = Path(input_root).resolve()
    out_root = Path(args_out_dir).resolve()
    try:
        relative_path = input_path.relative_to(root_path)
    except ValueError:
        relative_path = input_path.name
    return out_root / relative_path

# =============================================
# Model Wrapper
# =============================================

class ModelResponder:
    def __init__(self, model, tokenizer, model_name, thinking=True):
        self.model = model
        self.tokenizer = tokenizer
        self.thinking = thinking
        self.model_name = model_name.lower()

    def generate(self, messages, max_new_tokens=2048):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.thinking)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()
        
        THINK_END = 151668 
        try:
            idx = len(output_ids) - output_ids[::-1].index(THINK_END)
        except ValueError:
            idx = 0
        
        thinking_text = self.tokenizer.decode(output_ids[:idx], skip_special_tokens=True).strip()
        content = self.tokenizer.decode(output_ids[idx:], skip_special_tokens=True).strip()
        return thinking_text, content

# from vllm import LLM, SamplingParams
# class ModelRespondervllm:
#     def __init__(self, model_name, thinking=True):
#         self.thinking = thinking
#         self.model_name = model_name.lower()
        
#         # Initialize vLLM (gpu_memory_utilization can be lowered if you hit OOM)
#         print("Loading vLLM engine...")
#         self.llm = LLM(model=model_name, dtype="bfloat16", trust_remote_code=True, gpu_memory_utilization=0.9)
#         self.tokenizer = self.llm.get_tokenizer()
        
#         # Pre-compile sampling parameters
#         self.sampling_params = SamplingParams(
#             max_tokens=2048,
#             temperature=0.7, # Adjust if needed
#             top_p=0.9
#         )

#     def generate(self, messages):
#         # Format prompt
#         kwargs = {"tokenize": False, "add_generation_prompt": True}
#         if self.thinking:
#             # Note: Only keep this if your specific Qwen tokenizer version supports it
#             kwargs["enable_thinking"] = True 
            
#         text = self.tokenizer.apply_chat_template(messages, **kwargs)
        
#         # Run inference (use_tqdm=False prevents double progress bars)
#         outputs = self.llm.generate([text], self.sampling_params, use_tqdm=False)
        
#         # Extract token IDs directly from vLLM output to match your previous logic
#         output_ids = list(outputs[0].outputs[0].token_ids)
        
#         THINK_END = 151668 
#         try:
#             idx = len(output_ids) - output_ids[::-1].index(THINK_END)
#         except ValueError:
#             idx = 0
        
#         thinking_text = self.tokenizer.decode(output_ids[:idx], skip_special_tokens=True).strip()
#         content = self.tokenizer.decode(output_ids[idx:], skip_special_tokens=True).strip()
        
#         return thinking_text, content
    
# =============================================
# Main Execution Logic
# =============================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_name", type=str, default="QA-Gen")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", default=None)
    parser.add_argument("--use_tracks", action="store_true")
    parser.add_argument("--track_type", choices=["m", "px"], default="m")
    parser.add_argument("--yolo_path", type=str, required=True, help="Path to annotations/metadata")
    parser.add_argument("--output_folder", type=str, default="data/output")
    parser.add_argument("--qas_ratios", type=str, required=True)
    parser.add_argument("--prompts_config", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="nuscenes")
    parser.add_argument("--number_of_questions", type=int, default=15)
    parser.add_argument("--frames_back", type=int, default=3)
    parser.add_argument("--tracks_path", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    exp_name = generate_experiment_name(args)
    
    # Configuration and Model Init
    ratio_data = json.load(open(args.qas_ratios))
    config = yaml.safe_load(open(args.prompts_config))
    config_prompts = config["prompts"]
    directions = ["Turn right.", "Drive backward.", "Going ahead.", "Turn left."]
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", device_map="auto")
    responder = ModelResponder(model, tokenizer, args.model, thinking=args.thinking)
    
    # responder = ModelRespondervllm(args.model, thinking=args.thinking)

    # 1. Get all nested scene folders
    all_scene_paths = list_drive_lm_scenes(args.yolo_path)
    total_scenes = len(all_scene_paths)
    print(f"Found {total_scenes} total scenes.")

    # 2. Iterate with Progress Counter
    for iz, scene_folder in enumerate(all_scene_paths):
        
        # Output directory parsing
        if args.dataset_name == "valeo":
            output_dir = get_output_dir(scene_folder, args.output_folder)
        elif args.dataset_name == "nuscenes":
            output_dir = get_output_dir_from_nuscenes_folder(scene_folder, args.output_folder, args.yolo_path)
        else:
            output_dir = Path(args.output_folder) / Path(scene_folder).name
            
        print(f"{output_dir=}")

        # 3. Decision Logic (Check if the folder already processed)
        is_complete = False
        output_count = 0
        
        # We output `.jsonl` files in this script, so we search for those.
        # This mirrors your logic: "len(list(output_dir.glob('*.json'))) if output_dir.exists() else 'None'"
        print(f"Checking output directory: {output_dir}, exists: {output_dir.exists()}, glob(*.jsonl): {len(list(output_dir.glob('*.jsonl'))) if output_dir.exists() else 'None'}")
        
        if output_dir.exists():
            output_count = len(list(output_dir.glob("*.jsonl")))
            is_complete = True # if just folder exists, than it was processed
            
            # Here input_count logically maps to the generated file. Assuming 1 JSONL file completes the folder.
            if output_count >= 1 and not args.overwrite:
                is_complete = True

        if is_complete:
            print(f"⏩ [{iz+1}/{total_scenes}] Skipping: {output_dir} (Processed {output_count} jsonl files)")
            print(" - - - - - "*3)
            continue

        print(f"🚀 [{iz+1}/{total_scenes}] Processing: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 5. Data Loading
            processed_json = os.path.join(scene_folder, "merged_processed.json")
            if not os.path.exists(processed_json):
                from src.utils.qas_generation_helper import run_yolo_processing
                data = run_yolo_processing(input_path=scene_folder, output_path=scene_folder, dataset_name=args.dataset_name)
            else:
                data = json.load(open(processed_json))

            # Tracks processing
            tracks_by_id = None
            if args.use_tracks and args.tracks_path:
                raw_tracks = json.load(open(args.tracks_path))
                tracks_by_id = {d["object_id"]: d for d in raw_tracks["tracks"]}

            # Question distribution
            q_ratios = get_test_distribution(ratio_data)
            q_dist = distribute_by_ratio(q_ratios, len(data) * args.number_of_questions)

            # 6. Output Generation
            output_file = output_dir / f"qas_{exp_name}.jsonl"
            pattern = r"\b\w+_\d+\b"

            items = sorted(data.items())
            for scene_id, obj_info in tqdm(items, desc=f"Scene {iz+1}/{total_scenes}"):
                scene_text = detections_to_text(obj_info, tracks_by_id, scene_id, args.frames_back, args.track_type, args.use_tracks, args.dataset_name)
                
                questions = ["What are the important objects in the current scene?"] + \
                            random_permutation(q_dist, args.number_of_questions-1, args.number_of_questions-1)
                
                scene_results = {}
                used_objs = set()

                with torch.inference_mode():
                    for i, q in enumerate(questions):
                        prompt = generate_prompt(q, None, scene_text, config_prompts, directions, args.use_tracks, args.track_type)
                        if used_objs and "<obj>" in q:
                            prompt += f"\nAvoid reusing: {', '.join(sorted(used_objs))}"
                        
                        messages = [{"role": "user", "content": prompt}]
                        _, content = responder.generate(messages)
                        
                        try:
                            clean_content = content.strip()
                            if not clean_content.startswith("{"): 
                                clean_content = "{" + f'"Question_{i}": ' + clean_content + "}"
                            parsed = json.loads(clean_content)
                            scene_results.update(parsed)
                            used_objs.update(re.findall(pattern, content))
                        except:
                            continue
                
                append_json_line(str(output_file), {scene_id: scene_results})

        except Exception as e:
            print(f"❌ Error in [{iz+1}/{total_scenes}] {scene_folder}: {e}")

    print("Job complete.")
    
        
# salloc -A eu-25-10 -p qgpu_exp --gpus-per-node 1 -t 1:00:00 --nodes 1
# salloc -A OPEN-36-7 -p qgpu_exp --gpus-per-node 1 -t 1:00:00 --nodes 1

# source /mnt/proj1/eu-25-10/envs/roadcap-gen/bin/activate
# PYTHONPATH=. python scripts/01_pseudo_labels_gen/pseudo_qas_generation.py --model="Qwen/Qwen3-14B" --wandb_name="QWEN-14B-non-thinking" --file_name="debug_QA_qwen_non_thinking" --qas_ratios "configs/dataset/qas_drivelm_ratios.json" --prompts_config "configs/inference/llm_prompt_config.yaml" --output_folder "data/qas_gen_output/"

# PYTHONPATH=. python scripts/01_pseudo_labels_gen/pseudo_qas_generation.py --model="Qwen/Qwen3-14B" --wandb_name="QWEN-14B-non-thinking" --file_name="debug_QA_qwen_non_thinking" --qas_ratios "configs/dataset/qas_drivelm_ratios.json" --prompts_config "configs/inference/llm_prompt_config.yaml" --output_folder "data/qas_gen_output/" --yolo_path "/scratch/project/eu-25-10/datasets/nuScenes_metadata/annotations_in_valeo_format/CAM_FRONT/n008-2018-07-27-12-07-38-0400/" --dataset_name "nuscenes"

# python QA_gen/qa_gen.py --model="Qwen/Qwen3-14B" --wandb_name="QWEN-14B-non-thinking" --file_name="QA_qwen_non_thinking" --start_idx=162 --end_idx=312

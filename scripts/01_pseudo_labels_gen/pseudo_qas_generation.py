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
from src.utils.qas_generation_helper import (
    load_descriptions_from_tar,
    load_json,
    load_yaml,
    run_yolo_processing, generate_experiment_name, save_metadata
)
from src.utils.qas_generation_helper import ModelResponder
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


import wandb

"""
sbatch --job-name QA_gen_Qwen_thinking QA-gen.sh --model="Qwen/Qwen3-14B" --thinking --wandb_name="QWEN-14B-thinking" --file_name="QA_qwen_thinking" --start_idx=162 --end_idx=312 --use_tracks --track_type="m"

"""


def get_prefixed_permutation(options):
    permuted = options.copy()
    random.shuffle(permuted)
    letters = ["A. ", "B. ", "C. ", "D. "]
    return [f"{letters[i]}{item}" for i, item in enumerate(permuted)]


def get_test_distribution(ratio_data, no_dir_questions=None):
    """load the ratios"""
    test_distribution = {}
    for entry in ratio_data:
        if entry["ratio_test"] > 0:
            if no_dir_questions is not None and entry["question"] not in no_dir_questions:
                continue
            question = entry["question"]
            test_distribution[question] = entry["ratio_test"]
    return test_distribution


def distribute_by_ratio(question_to_ratio, total_count):
    """returns the number of questions to generate for each question based on the given ratios of the total count"""
    # Normalize ratios to sum to 1
    ratio_sum = sum(question_to_ratio.values())
    normalized = {q: r / ratio_sum for q, r in question_to_ratio.items()}
    counts = {q: max(1, round(normalized[q] * total_count)) for q in question_to_ratio}

    return counts


categories = set()


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


def detections_to_text(
    data,
    tracks_by_object_id: dict | None = None,
    frame: str | int | None = None,
    frames_back: int = 3,
    track_type: str = "m",
    include_tracks=False,
    dataset_name="valeo",
):
    """Converts bbox detectied data into list of sentences for the prompt"""
    
    if dataset_name == "valeo":
        frame = int(frame.split("_")[-1].split(".")[0])
    elif dataset_name == "nuscenes":
        frame = int(frame.split(".")[0].split("_")[-1])
    
    if include_tracks:
        if tracks_by_object_id is None or frame is None:
            raise ValueError("Tracks and frame must be provided when include_tracks is True.")
        # if frame > 1426:  # manual fix for bad data
        #     return

    lines = []
    lines.append("Here is the list of objects detected in the scene. Use them to generate the QA pairs:")

    for category, cat_objs in data["categories"].items():
        categories.add(category)
        # bbox is [x, y, w, h]
        for idx, cat_obj in enumerate(cat_objs["objects"]):
            line = (
                f"- {category}_{idx}:\n\t - current bbox: {cat_obj['bbox']}\n\t - current middle point: {cat_obj['mid_point']}."
            )

            if include_tracks:
                tracks = tracks_by_object_id.get(cat_obj["id"])  # type: ignore
                if tracks is None:
                    continue

                tracks = tracks["track"]
                frames = [d["frame"] for d in tracks]
                window, missing = get_past(tracks, frames, frame, frames_back)
                if len(window) == 1 and frame not in missing:
                    if track_type == "m":
                        line += f"\n\t - relative position at t-0 ms: [{window[0]['x']}, {window[0]['y']}, {window[0]['z']}]. No prior frames available."
                    elif track_type == "px":
                        line += f"\n\t - midpoint at t-0 ms: {window[0]['center_2d_px']}. No prior frames available."
                    lines.append(line)
                    continue
                elif len(window) == 0:
                    continue

                r = range(len(window), 0, -1) if frame in missing else range(len(window) - 1, -1, -1)
                if track_type == "m":
                    count = frames_back - len(missing) + 1

                    positions = [f"[{d['x']}, {d['y']}, {d['z']}] at t-{t * 55} ms" for t, d in zip(r, window)]

                    line += f"\n\t - relative positions from the last {count} frames: {', '.join(positions)}."
                elif track_type == "px":
                    count = frames_back - len(missing) + 1

                    midpoints = [f"{d['center_2d_px']} at t-{t * 55} ms" for t, d in zip(r, window)]

                    line += f"\n\t - midpoints from the last {count} frames: {', '.join(midpoints)}"
                else:
                    raise ValueError("Invalid track type: {track_type}")

            lines.append(line)

    return "\n".join(lines)


def random_permutation(q_distribution, min_length=15, max_length=15, test=False):
    """Pick a random permutation of questions"""
    # amount of questions to generate
    L = random.randint(min_length, max_length)

    if test:
        questions = []
        for i, k in enumerate(sorted(q_distribution, key=q_distribution.get, reverse=True)):
            if i >= L:
                print(*questions, sep="\n")
                return questions
            questions.append(k)

    chosen = []
    random.shuffle(available := [k for k, w in q_distribution.items() if w > 0])

    # add unique questions first
    for q in available:
        if len(chosen) >= L:
            break
        chosen.append(q)
        q_distribution[q] -= 1
        if q_distribution[q] <= 0:
            del q_distribution[q]

    # fill up to L with repeats if needed
    while len(chosen) < L and q_distribution:
        q = random.choice(list(q_distribution.keys()))
        chosen.append(q)
        q_distribution[q] -= 1
        if q_distribution[q] <= 0:
            del q_distribution[q]

    if len(chosen) < L:
        print("Less available questions than requested:", len(chosen), "<", L)

    return chosen


def generate_prompt(q, description, objects, tracks=False, track_type="m"):
    parts = [
        config_prompts["context"],
        config_prompts["obj"] if "<obj>" in q else config_prompts["no_obj"],
        config_prompts["qa_generation_QWEN"],
    ]

    if tracks:
        # Tracks are already included in the objects when the flag is set
        # This just adds context about them
        parts.append(f"\n{config_prompts['tracks_' + track_type]}")

    parts.append(objects)

    if "<position>" in q:
        parts.append(f"\n{config_prompts['position']}")

    if "What are the important objects in the current scene?" in q or "priority of the objects" in q:
        parts.append(f"\n{config_prompts['important_objects']}")

    if description is not None:
        parts.append(f"\nThe scene description is:\n {description}")
    
    if "following options:" in q:
        q += " " + " ".join(get_prefixed_permutation(directions))
    parts.append(f"\nThe question to use is:\n {q}\n")

    return "\n".join(parts)


def save_distribution(dist: dict):
    # Write to temp file first
    dir_name = os.path.dirname(DISTRIBUTION_FILE) or "."
    with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tmp:
        json.dump(dist, tmp, ensure_ascii=False, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())

    # Atomic replace
    os.replace(tmp.name, DISTRIBUTION_FILE)


def load_or_create_distribution(calculate_fn):
    if os.path.exists(DISTRIBUTION_FILE):
        with open(DISTRIBUTION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        dist = calculate_fn()
        save_distribution(dist)
        return dist


def append_json_line(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=4))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def get_last_scene_id(path):
    if not os.path.exists(path):
        return -1

    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        pos = f.tell()

        while pos > 0:
            pos -= 1
            f.seek(pos)
            if f.read(1) not in b"\n\r\t ":
                break

        if pos == 0:
            return -1

        while pos > 0:
            pos -= 1
            f.seek(pos)
            if f.read(1) == b"\n":
                pos += 1
                break

        f.seek(pos)
        line = f.readline().decode("utf-8").strip()

    if not line:
        return -1

    obj = json.loads(line)
    return int(next(iter(obj.keys())))


completed = {"done": False}


def cleanup_distribution_file():
    if completed["done"] and os.path.exists(DISTRIBUTION_FILE):
        os.remove(DISTRIBUTION_FILE)


def build_wandb_config(args):
    config = {
        "model": args.model,
        "file_name": args.file_name,
        "number_of_questions": args.number_of_questions,
    }

    if "qwen" in args.model.lower():
        config["thinking"] = args.thinking

    if args.use_tracks:
        config["use_tracks"] = True
        config["track_type"] = args.track_type
    else:
        config["use_tracks"] = False

    return config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-14B",
        help="HF model to use",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable Qwen thinking mode",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Whether to use wandb logging",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default="QWEN-14B-non_thinking",
        help="How to name the wandb run",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="From which scene index to start processing",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="At which scene index to stop processing",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="QA_qwen_non_thinking",
        help="How to name the the file to save results",
    )
    parser.add_argument(
        "--use_tracks",
        action="store_true",
        help="Add track information to the object descriptions in generation",
    )
    parser.add_argument(
        "--track_type",
        choices=["m", "px"],
        default="m",
        help="Type of tracks to use. Either 'm' for meters or 'px' for pixel coordinates.",
    )
    parser.add_argument(
        "--frames_back",
        type=int,
        default=3,
        help="How many frames back to include in the track information.",
    )
    parser.add_argument(
        "--use_llava_captions",
        action="store_true",
        help="Whether to use captions generated by LLAVA as descriptions in the prompts. If false, no descriptions will be used.",
    )
    parser.add_argument(
        "--tar_location",
        type=str,
        default="/scratch/project/eu-25-10/datasets/LLAVA_captions/clean_features/000471.tar",
        help="where is the tar file, default is for Karolina (local = 'Data/21/000471.tar')",
    )
    parser.add_argument(
        "--yolo_path",
        type=str,
        default="/scratch/project/eu-25-10/datasets/FRONT_CAM_zipped_metadata/YOLOv11_1/20250527/21/2024-11-27-10-28-35/camera1/",
        help="where is the yolo jsons path. Tries to load already precessed data. If not found, process the raw jsons from this path and save it there.",
    )
    parser.add_argument(
        "--number_of_questions",
        type=int,
        default=15,
        help="How many questions per scene to generate. This number isn't kept for all the scenes exactly due to the distribution constraints.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="camera1",
        help="What camera to use for yolo data",
    )
    parser.add_argument(
        "--tracks_path",
        type=str,
        default="Data/21/tracks_21_2024-11-27-10-28-35_camera1.json",
        help="Path to tracks file",
    )
    parser.add_argument(
        "--qas_ratios",
        type=str,
        default="Data/ratios.json",
        help="Path to qas ratio file to generate from the specified distributtion",
    )
    parser.add_argument(
        "--prompts_config",
        type=str,
        default="QA_gen/config.yaml",
        help="Path to prompts config yaml file",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="Data/",
        help="Path to output folder",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="valeo",
        choices=["valeo", "nuscenes", "drivelm"],
        help="Name of the dataset",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exp_name = generate_experiment_name(args)
    args.file_name = f"{exp_name}_{args.file_name}"
    
    print(args)
    if args.use_wandb:
        wandb.init(
            project="QA-gen",
            name=args.wandb_name,
            config=build_wandb_config(args),
        )
    # =============================================
    #               Data Loading
    # =============================================
    if not os.path.exists(os.path.join(args.yolo_path, "merged_processed.json")):
        # store it as one file to avoid the overhead of processing and loading many small files during generation
        data = run_yolo_processing(
            input_path=args.yolo_path,
            output_path=args.yolo_path,
            camera_id=args.camera,
            output_name="merged_processed.json",
            dataset_name=args.dataset_name,
        )
    else:
        print("Loading processed YOLO data...")
        data = load_json(os.path.join(args.yolo_path, "merged_processed.json"))

    ratio_data = load_json(args.qas_ratios)
    config = load_yaml(args.prompts_config)
    config_prompts = config["prompts"]
    # descriptions = load_descriptions_from_tar(args.tar_location)
    descriptions = None

    slice_tag = f"{args.start_idx}_{args.end_idx}"
    # slice_tag = f"{args.start_idx}_{args.end_idx or len(descriptions)}"
    full_run_id = f"{exp_name}_{slice_tag}" # for metadata save to reproduce experiment
    save_metadata(args, args.output_folder, full_run_id)
    
    DISTRIBUTION_FILE = f"{args.output_folder}/q_distribution_{args.file_name}_{slice_tag}.json"
    
    output_folder_path = Path(args.output_folder)
    output_folder_path.mkdir(parents=True, exist_ok=True)
    output_path = f"{args.output_folder}/{args.file_name}_{slice_tag}.jsonl"

    directions = ["Turn right.", "Drive backward.", "Going ahead.", "Turn left."]

    no_direction_questions = set(
        [
            "What actions could the ego vehicle take based on <obj>? Why take this action and what's the probability?",
            "What actions taken by the ego vehicle can lead to a collision with <obj>?",
            "In this scenario, what are safe actions to take for the ego vehicle?",
            "Predict the behavior of the ego vehicle. Please select the correct answer from the following options:",
            "What are the important objects in the current scene? Those objects will be considered for the future reasoning and driving decision.",
            "In this scenario, what are dangerous actions to take for the ego vehicle?",
            "What's your comment on this scene?",
            "Would <obj> take <obj> into account?",
            "Based on <obj> in this scene, what is the most possible action of the ego vehicle?",
            "Based on the observation of <obj>, what actions may <obj> take?",
            "Are there any other issues worth noting in this scene?",
            "What situation in this scene affects driving vehicles?",
            "What is the priority of the objects that the ego vehicle should consider?(in descending order)",
            "What impact does this situation have on driving vehicles?",
            "Are there any other notable issues in this scene?",
            "What situation in this scene affects the driving of vehicles?",
            "What situation in this scene affects driving?",
            "What is the impact of this situation on driving vehicles?",
            "What impact does the situation have on driving vehicles?",
            "What situation affects driving vehicles in this scene?",
            "What is the traffic signal that the ego vehicle should pay attention to?",
            "Under what circumstances does this scene affect driving vehicles?",
            "What impact does this situation have on driving a vehicle?",
            "What is the impact on driving vehicles in this situation?",
            "What is the impact of the situation on driving vehicles in this scene?",
            "What are objects to the <position>",
            "What is the target action of the ego vehicle?",
            "What will affect driving judgment in this scene?",
            "Are there any other issues in this scene that are worth noting?",
            "Are there any other notable issues in this road scene?",
            "Are there any other notable issues with this scene?",
            "Are there any other noteworthy issues in this scene?",
            "Under what conditions does this scene affect driving vehicles?",
            "Under what conditions does this scene affect the driving vehicle?",
            "What circumstances does this scene affect the driving vehicle?",
            "What conditions affect driving vehicles in this scene?",
            "What conditions does this scene affect driving vehicles?",
            "What conditions in this scenario affect driving?",
            "What conditions in this scenario affect the driving of vehicles?",
            "What conditions in this scene affect driving vehicles?",
            "What conditions in this scene affect driving?",
            "What conditions in this scene affect the driving vehicle?",
            "What effect does this situation have on driving vehicles?",
            "What is the impact of the situation on driving vehicles?",
            "What situation affects driving in this scene?",
            "What situation affects the driving of vehicles in this scene?",
            "What situation in this scene affects the driving vehicle?",
            "What situation in this scene affects the driving vehicles?",
            "What situations affect driving a vehicle?",
            "What situations affect driving vehicles?",
            "What situations does this scene affect driving vehicles?",
            "What situations does this scene affect the driving vehicles?",
        ]
    )

    q_ratios = get_test_distribution(ratio_data, no_direction_questions if not args.use_tracks else None)
    num_of_scenes = (args.end_idx - args.start_idx) if args.end_idx is not None else (len(data) - args.start_idx)
    q_distribution = load_or_create_distribution(
        lambda: distribute_by_ratio(q_ratios, num_of_scenes * args.number_of_questions - 1)
    )
    tracks_by_object_id = None
    if args.use_tracks:
        tracks_path = args.tracks_path
        tracks = load_json(tracks_path)
        tracks_by_object_id = {d["object_id"]: d for d in tracks["tracks"]}

    scene_objects = {
        img: detections_to_text(
            obj_info,
            tracks_by_object_id=tracks_by_object_id,
            frame=img,
            frames_back=args.frames_back,
            track_type=args.track_type,
            include_tracks=args.use_tracks,
        )
        for img, obj_info in data.items()
    }

    pattern = rf"\b(?:{'|'.join(categories)})_\d+\b"
    atexit.register(cleanup_distribution_file)
    # Initial question to ask each time
    init_question = "What are the important objects in the current scene? Those objects will be considered for the future reasoning and driving decision."

    # =============================================
    #               Model loading
    # =============================================
    print(f"Loading model {args.model}...")
    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
    )
    responder = ModelResponder(
        model=model,
        tokenizer=tokenizer,
        model_name=args.model,
        thinking=args.thinking,
    )
    # =============================================
    # Get questions and their prompts
    # =============================================
    # last = get_last_scene_id(output_path)
    # start_id = max(last + 1, args.start_idx)
    start_id = args.start_idx
    
    chronological_scenes = sorted(scene_objects.items())
    end = args.end_idx or len(chronological_scenes)
    chronological_scenes = chronological_scenes[start_id:end]
    print("Number of scenes: ", end - start_id)

    for id, objects in tqdm(chronological_scenes, desc="Processing scenes"):
        results = {id: {}}
        description = descriptions[id] if descriptions is not None else None

        questions_list = random_permutation(
            q_distribution, min_length=args.number_of_questions - 1, max_length=args.number_of_questions - 1
        )
        prompts_list = [
            generate_prompt(q, description, objects, tracks=args.use_tracks, track_type=args.track_type) for q in questions_list
        ]
        if not prompts_list:
            print("Ran out of questions.")
            break

        # Prepend the initial question
        prompts_list = [
            generate_prompt(init_question, description, objects, tracks=args.use_tracks, track_type=args.track_type)
        ] + prompts_list
        questions_list = [init_question] + questions_list

        # Uncomment to debug prompts
        # if id < 50:
        #     continue
        # print(*prompts_list, sep="\n====\n")
        # exit()
        # content = "asf"

        QAs = []
        used_obj = set()

        # =============================================
        #                   Inference
        # =============================================
        with torch.inference_mode():
            for i, prompt in enumerate(prompts_list):
                care_obj = "<obj>" in questions_list[i]

                messages = [
                    {
                        "role": "user",
                        "content": prompt
                        + (
                            f"\nYou have already used objects: {', '.join(sorted(used_obj))}. "
                            "Please avoid reusing them in your answer."
                            if care_obj and used_obj
                            else ""
                        ),
                    }
                ]

                _, content = responder.generate(messages)
                QAs.append(f'"Question_{i}": {content}')

                if care_obj:
                    used_obj.update(re.findall(pattern, content))

        inner_results = {}
        for q in QAs:
            try:
                obj = json.loads("{" + q + "}")
            except json.JSONDecodeError:
                try:
                    obj = demjson3.decode("{" + q + "}", strict=False)
                except Exception:
                    print("Unfixable JSON:", q, sep="\n")
                    continue

            if isinstance(obj, dict):
                inner_results.update(obj)

        results[id] = inner_results

        # ===== Write one scene =====
        append_json_line(output_path, results)

        # ===== Persist distribution every step =====
        save_distribution(q_distribution)
        
    completed["done"] = True
    # save args with which the experiment was run to be able to reproduce it later
    
    print("Ran out of scenes.")
    if args.use_wandb:
        wandb.finish()

# salloc -A eu-25-10 -p qgpu_exp --gpus-per-node 1 -t 1:00:00 --nodes 1
# salloc -A OPEN-36-7 -p qgpu_exp --gpus-per-node 1 -t 1:00:00 --nodes 1

# source /mnt/proj1/eu-25-10/envs/roadcap-gen/bin/activate
# PYTHONPATH=. python scripts/01_pseudo_labels_gen/pseudo_qas_generation.py --model="Qwen/Qwen3-14B" --wandb_name="QWEN-14B-non-thinking" --file_name="debug_QA_qwen_non_thinking" --qas_ratios "configs/dataset/qas_drivelm_ratios.json" --prompts_config "configs/inference/llm_prompt_config.yaml" --output_folder "data/qas_gen_output/"

# PYTHONPATH=. python scripts/01_pseudo_labels_gen/pseudo_qas_generation.py --model="Qwen/Qwen3-14B" --wandb_name="QWEN-14B-non-thinking" --file_name="debug_QA_qwen_non_thinking" --qas_ratios "configs/dataset/qas_drivelm_ratios.json" --prompts_config "configs/inference/llm_prompt_config.yaml" --output_folder "data/qas_gen_output/" --yolo_path "/scratch/project/eu-25-10/datasets/nuScenes_metadata/annotations_in_valeo_format/CAM_FRONT/n008-2018-07-27-12-07-38-0400/" --dataset_name "nuscenes"

# python QA_gen/qa_gen.py --model="Qwen/Qwen3-14B" --wandb_name="QWEN-14B-non-thinking" --file_name="QA_qwen_non_thinking" --start_idx=162 --end_idx=312

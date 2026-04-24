import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys
import re

project_root = str(Path(__file__).resolve().parents[2])  # Adjust parents index if your script is deeper
if project_root not in sys.path:
    sys.path.append(project_root)
from src.utils.qas_generation_helper import load_json


PATTERN_OBJ_GENERAL = r"<[^>]*\w+_\d+,\s*[^,]+,\s*-?\d+(?:\.\d+)?,\s*-?\d+(?:\.\d+)?>"
REPL_OBJ_GENERAL = r"<obj>"
PATTERN_SELECT_OPTIONS = r"(.*?Please select the correct answer from the following options:).*"
REPL_SELECT_OPTIONS = r"\1"
PATTERN_POSITION = r"(.*? to the )(?:front|back|left|right)(?: (?:left|right))? of the (?:ego )?(?:car|vehicle)"
REPL_POSITION = r"\1<position>"
PATTERNS = [
    (PATTERN_OBJ_GENERAL, REPL_OBJ_GENERAL),
    (PATTERN_SELECT_OPTIONS, REPL_SELECT_OPTIONS),
    (PATTERN_POSITION, REPL_POSITION),
]
COMPILED_PATTERNS = [(re.compile(p), r) for p, r in PATTERNS]

ANSWER_VARIATIONS = {
    "ahead": ["going ahead", "go ahead", "moving forward", "move forward", "moving straight", "move straight", "straight ahead", "drive forward"],
    "backward": ["drive backward", "driving backward", "move backward", "moving backward", "reverse", "going backward", "go backward"],
    "left": ["turn left", "turning left", "steer left"],
    "right": ["turn right", "turning right", "steer right"],
}
LETTER_OPTIONS = ["A.", "B.", "C.", "D."]


def generalize(q):
    for rgx, repl in COMPILED_PATTERNS:
        q = rgx.sub(repl, q)
    q = re.sub(" +", " ", q.strip())
    return q


def get_match_ids_img(data):
    out = defaultdict(lambda: {"scene_id": None, "frames": defaultdict(lambda: {"frame_id": None, "imgs_paths": None})})

    for scene_id, scene_data in data.items():
        key_frames = scene_data.get("key_frames", {})
        for frame_id, frame_data in key_frames.items():
            frame = defaultdict(list)
            imgs = frame_data.get("image_paths", [])

            path = [img.split("/")[4] for img in imgs.values() if "CAM_BACK" in img][0]
            path_split = path.split("__")
            image_paths = ["/".join(img.split("/")[3:]) for img in imgs.values()]

            folder = out[path_split[0]]
            folder["scene_id"] = scene_id
            frame = folder["frames"][path]  # type: ignore
            frame["frame_id"] = frame_id  # type: ignore
            frame["imgs_paths"] = image_paths  # type: ignore
    return out


def process_file(input_path: Path):
    output_path = input_path.with_name(f"{input_path.stem}_converted.json")
    results = []
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            data = json.loads(line)

            for sample_id, content in data.items():
                images = content.get("images", [])

                for key, q in content.items():
                    if key.startswith("Question_"):
                        question_idx = key.split("_")[-1]

                        question = q.get("question", "")
                        answer = q.get("short_answer", "")

                        new_item = {
                            "id": f"{sample_id}_{question_idx}",
                            "image": images,
                            "conversations": [
                                {"from": "human", "value": question},
                                {"from": "gpt", "value": answer},
                            ],
                        }

                        results.append(new_item)

        fout.write(json.dumps(results, indent=2, ensure_ascii=False))


def get_nuscenes_images(orig_jsons_path, scene, sample_id, root_img_paths) -> tuple[list[str], list[str]]:
    images = []
    not_found = []
    # print(f"Images not found for {frame}")
    cams = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]
    root = Path(orig_jsons_path)
    for cam in cams:
        if "n008" in scene:
            img_folder = root / cam / scene
        elif "n015" in scene:
            img_folder = root / cam / scene[:-5]
        else:
            raise ValueError(f"Unexpected scene format: {scene}")

        search_pattern = f"{scene}__{cam}__{sample_id.split('__')[2][:-5]}*"
        matches = list(img_folder.glob(search_pattern))
        if matches:
            matched_filename = matches[0].stem
            images.append((root_img_paths / cam / matched_filename).with_suffix(".jpg"))
        else:
            images.append(root_img_paths / cam / search_pattern[:-1])
            not_found.append(cam)
            print(f"Warning: No match found for {cam} in {scene} for {sample_id.split('__')[2][:-5]}")

    return images, not_found


def process_file_nuscenes(folder: Path, input_path: Path, ids_images: dict, tags: dict, acc: bool) -> int:
    orig_jsons_path = "/scratch/project/eu-25-10/datasets/nuScenes_metadata/annotations_in_valeo_format/"
    root_img_paths = Path("/scratch/project/eu-25-10/datasets/nuScenes/samples/")

    output_path = input_path.parent / (f"../{folder.name}_converted" + ("_normalized" if acc else "")) / f"{input_path.stem}_converted.json"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    broken = 0
    results = []
    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            data = json.loads(line)

            for sample_id, content in data.items():
                scene = sample_id.split("__")[0]
                frame = sample_id + ".jpg"
                images = ids_images[scene]["frames"][frame]["imgs_paths"]
                scene_id = ids_images[scene]["scene_id"]
                frame_id = ids_images[scene]["frames"][frame]["frame_id"]

                if not scene_id:
                    # print(f"Scene ID not found for {scene}")
                    scene_id = "generated" + hashlib.md5(scene.encode()).hexdigest()

                if not frame_id:
                    # print(f"Frame ID not found for {frame}")
                    frame_id = "generated" + hashlib.md5(frame.encode()).hexdigest()

                not_found = []
                if not images:
                    images, not_found = get_nuscenes_images(orig_jsons_path, scene, sample_id, root_img_paths)

                try:
                    images = [str(img) for img in images]  # type: ignore
                except Exception as e:
                    print(80 * "-")
                    print(f"Error processing images for {frame}: {e}")
                    print(80 * "-")

                    print(images)
                    print(sample_id)
                    exit()

                for key, q in content.items():
                    if key.startswith("Question_"):
                        question_idx = key.split("_")[-1]

                        question = q["question"]
                        answer = q["short_answer"]
                        question = question.replace(" side", "")
                        if "What are the important objects in the current scene?" in question:
                            if " Those objects will be considered for the future reasoning and driving decision." not in question:
                                question += " Those objects will be considered for the future reasoning and driving decision."
                        if "What actions could the ego vehicle take based on" in question:
                            if any(word in answer.lower() for word in ["probability", "chance", "likelihood", "low", "high", "medium"]):
                                if " Why take this action and what's the probability?" not in question:
                                    question += " Why take this action and what's the probability?"
                            else:
                                broken += 1
                                continue

                        if "What is the moving status of object" in question:
                            if "Please select the correct answer from the following options:" not in question:
                                question += " Please select the correct answer from the following options:"
                        if (ques := "In this scenario, what are dangerous actions to take for the ego vehicle?") in question:
                            question = question[: len(ques)]
                        if (ques := "In this scenario, what are safe actions to take for the ego vehicle?") in question:
                            question = question[: len(ques)]
                        if (ques := "What situation affects driving vehicles in this scene?") in question:
                            question = question[: len(ques)]
                        if (ques := "What situation in this scene affects driving vehicles?") in question:
                            question = question[: len(ques)]

                        gq = generalize(question)
                        if "<obj> and <obj>" in gq:
                            broken += 1
                            continue

                        if "What is the moving status of <obj>?" in gq:
                            gq = gq.replace("of <obj>?", "of object <obj>?")
                        if "Based on <obj>," in gq:
                            if "in this scene" not in gq:
                                gq = gq.replace("Based on <obj>,", "Based on object <obj> in this scene,")

                        result = tags.get(gq, {"tag": -1, "category": "unknown"})
                        tag, category = result["tag"], result["category"]

                        if tag == -1:
                            broken += 1
                            continue

                        if tag == 0:
                            if "following options" in question.lower():
                                if not all(letter in question for letter in LETTER_OPTIONS):
                                    broken += 1
                                    continue
                                q_options = re.findall(r"([A-D])\.\s*(.*?)(?=(?:[A-D]\.|$))", question, re.DOTALL)
                                concept_to_letter = {}
                                for letter, text in q_options:
                                    text_lower = text.lower()
                                    # Find which concept this option text belongs to
                                    for direction, variations in ANSWER_VARIATIONS.items():
                                        if any(var in text_lower for var in variations):
                                            concept_to_letter[direction] = letter
                                            break

                                extracted_letter = ''
                                letter_matches_bracket = re.findall(r"(?:^|\s|\()([A-D])(?:[.)]|\s|$)", answer)
                                letter_matches_option = re.findall(r"(?i)option\s+([a-d])", answer)

                                all_explicit_letters = set([m.upper() for m in letter_matches_bracket + letter_matches_option if m])

                                # look for explicit letter mentions first
                                if len(all_explicit_letters) == 1:
                                    extracted_letter = all_explicit_letters.pop()

                                elif len(all_explicit_letters) > 1:
                                    broken += 1
                                    continue

                                else:
                                    # look for written semantic variations
                                    answer_lower = answer.lower()
                                    found_directions = set()

                                    for direction, variations in ANSWER_VARIATIONS.items():
                                        if any(var in answer_lower for var in variations):
                                            found_directions.add(direction)

                                    if len(found_directions) == 1:
                                        found_concept = found_directions.pop()
                                        # Map it back to the corresponding letter from THIS specific question
                                        extracted_letter = concept_to_letter[found_concept]

                                        # If the mapping failed (e.g., semantic answer wasn't an option in the question)
                                        if not extracted_letter:
                                            broken += 1
                                            continue

                                    elif len(found_directions) > 1:
                                        # Multiple, discard
                                        broken += 1
                                        continue

                                    else:
                                        # Nothing matched at all, discard
                                        broken += 1
                                        continue

                            elif not any(word in answer.lower() for word in ["yes", "no"]):
                                print(80 * "-")
                                print(f"Question: {question}")
                                print(f"Answer: {answer}")
                                print(80 * "-")
                                broken += 1
                                continue

                            if acc:
                                if "yes" in answer.lower():
                                    answer = "Yes."
                                if "no" in answer.lower():
                                    answer = "No."
                                if "following options" in question.lower():
                                    answer = extracted_letter + '.' # type: ignore
                                    
                                    
                                    # match = re.search(r"\b([A-D])\.", answer)
                                    # answer = match.group(1) if match else answer

                        new_item = {
                            "id": f"{scene_id}_{frame_id}_{question_idx}_generated",
                            "image": images,
                            "not_found_images": not_found,
                            "conversations": [
                                {"from": "human", "value": question},
                                {"from": "gpt", "value": answer},
                            ],
                            "tag": tag,
                            "category": category,
                        }

                        results.append(new_item)

        # with output_path.open("w", encoding="utf-8") as fout:
        #     fout.write(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Retarded count for {input_path.name}: {broken}")
    return broken


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path, help="Path to folder containing jsonl files")
    parser.add_argument("--acc", action="store_true", help="Whether to post-process answers for accuracy questions")
    args = parser.parse_args()
    full_train = load_json("data/v1_1_train_nus.json")
    ids_images = get_match_ids_img(full_train)
    tags = load_json("data/local_splits/tags_for_generalized_qs.json")
    folder: Path = args.folder
    broken = 0
    for jsonl_file in folder.rglob("*.jsonl"):
        broken += process_file_nuscenes(folder, jsonl_file, ids_images, tags, args.acc)

    # for jsonl_file in folder.glob("*.jsonl"):
    #     process_file(jsonl_file)
    print(80 * "-")
    print(f"Total QAs discarded: {broken}")
    print(80 * "-")


if __name__ == "__main__":
    main()

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

project_root = str(Path(__file__).resolve().parents[2])  # Adjust parents index if your script is deeper
if project_root not in sys.path:
    sys.path.append(project_root)
from src.utils.qas_generation_helper import load_json


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


def process_file_nuscenes(folder: Path, input_path: Path, ids_images: dict):
    orig_jsons_path = "/scratch/project/eu-25-10/datasets/nuScenes_metadata/annotations_in_valeo_format/"
    root_img_paths = Path("/scratch/project/eu-25-10/datasets/nuScenes/samples/")

    output_path = input_path.parent / f"../{folder.name}_converted" / f"{input_path.stem}_converted.json"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
                    print(f"Scene ID not found for {scene}")
                    scene_id = "generated" + hashlib.md5(scene.encode()).hexdigest()


                if not frame_id:
                    print(f"Frame ID not found for {frame}")
                    frame_id = "generated" + hashlib.md5(frame.encode()).hexdigest()
                not_found = []
                if not images:
                    images = []
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

                        question = q.get("question", "")
                        answer = q.get("short_answer", "")

                        new_item = {
                            "id": f"{scene_id}_{frame_id}_{question_idx}_generated",
                            "image": images,
                            "not_found_images": not_found,
                            "conversations": [
                                {"from": "human", "value": question},
                                {"from": "gpt", "value": answer},
                            ],
                        }

                        results.append(new_item)

        with output_path.open("w", encoding="utf-8") as fout:
            fout.write(json.dumps(results, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path, help="Path to folder containing jsonl files")
    args = parser.parse_args()
    full_train = load_json("data/v1_1_train_nus.json")
    ids_images = get_match_ids_img(full_train)

    folder: Path = args.folder
    for jsonl_file in folder.rglob("*.jsonl"):
        process_file_nuscenes(folder, jsonl_file, ids_images)

    # for jsonl_file in folder.glob("*.jsonl"):
    #     process_file(jsonl_file)


if __name__ == "__main__":
    main()

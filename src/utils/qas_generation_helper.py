import regex as re
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import tarfile
import yaml

import json
import os
import zipfile

import numpy as np
import regex as re


def get_full_filenames_in_zip(folder, file_extension=".jpg"):
    """Returns a list of image filenames with the given extension inside zip files in the specified folder and its subfolders"""

    def get_filenames_from_zip(zip_file_path):
        """Get a list of filenames for images in the zip file."""
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            return [
                os.path.join(zip_file_path, info.filename).replace("\\", "/")
                for info in zip_ref.infolist()
                if info.filename.endswith(file_extension)
            ]

    files = []
    for root, _, filenames in os.walk(folder):
        for f in filenames:
            if f.endswith(".zip"):
                files.extend(get_filenames_from_zip(os.path.join(root, f)))
    return files


def load_json_from_zip(paths):
    """paths: list of strings like 'some_folder/file.zip/file_inside_zip.json' returns: list of JSON objects"""
    json_list = []
    for full_path in paths:
        zip_path, internal_path = full_path.split(".zip/", 1)
        zip_path += ".zip"

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            with zip_ref.open(internal_path) as f:
                json_list.append(json.load(f))
    return json_list


def get_paths_in_folder(folder, file_extension=".json"):
    """Returns a list of file paths with the given extension in the specified folder and its subfolders"""
    return [os.path.join(root, f) for root, _, files in os.walk(folder) for f in files if f.endswith(file_extension)]


def process_yolo_data(data_all, round_digits=13, dataset_name="valeo"):
    def bbox_center(bbox):
        x, y, w, h = bbox
        return x + w / 2, y + h / 2

    processed = {}
    for data in data_all:
        img_id = data["results"][0]["image_name"]

        objects = {}
        yolo_feats = []
        first_result = data["results"][0] 
        if dataset_name == "valeo":
            yolo_feats = first_result.get("yolo_feat", [])
        elif dataset_name == "nuscenes":
            # Looks for "2d_feat" first; if missing, falls back to "yolo_feat"; if both missing, defaults to []
            yolo_feats = first_result.get("2d_feat", first_result.get("yolo_feat", []))
        else:
            yolo_feats = []

        for bbox in yolo_feats:
            cat = bbox["category_name"]
            objects.setdefault(cat, {"count": 0, "objects": []})

            objects[cat]["objects"].append(
                {
                    "id": bbox["object_id"],
                    "bbox": np.round(bbox["bbox"], round_digits).tolist(),
                    "mid_point": np.round(bbox_center(bbox["bbox"]), round_digits).tolist(),
                }
            )
            objects[cat]["count"] += 1

        processed[img_id] = {
            "num_objects": len(yolo_feats),
            "categories": dict(sorted(objects.items(), key=lambda x: x[1]["count"], reverse=True)),
        }

    return processed


def run_yolo_processing(
    input_path,
    output_path,
    output_name="merged_processed.json",
    camera_id="camera1",
    round_digits=1,
    dataset_name="valeo",
):
    files = get_paths_in_folder(input_path)
    data_all = [load_json(f) for f in files]

    result = process_yolo_data(data_all, round_digits=round_digits, dataset_name=dataset_name)

    save_json(result, os.path.join(output_path, output_name))
    return result

def save_metadata(args, output_folder, exp_name):
    os.makedirs(output_folder, exist_ok=True)
    metadata_path = os.path.join(output_folder, f"metadata_{exp_name}.json")
    
    # vars(args) converts the Namespace to a standard dictionary
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)
    
    print(f"Metadata saved to: {metadata_path}")
    
def generate_experiment_name(args):
    # Extract model name (e.g., 'Qwen3-14B' from 'Qwen/Qwen3-14B')
    model_short = args.model.split('/')[-1]
    
    # Identify key settings
    think_tag = "think" if args.thinking else "no-think"
    track_tag = f"tracks-{args.track_type}" if args.use_tracks else "no-tracks"
    llava_tag = "llava-captions" if args.use_llava_captions else "no-llava-captions"
    
    # Combine into a readable string
    parts = [
        model_short,
        args.dataset_name,
        think_tag,
        track_tag,
        llava_tag,
        f"q{args.number_of_questions}"
    ]
    
    return "_".join(parts)

def load_json(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_yaml(yaml_file):
    with open(yaml_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def print_json(data, indent=2):
    print(json.dumps(data, indent=indent, ensure_ascii=False))


def load_txt(file_or_obj):
    """
    file_or_obj: either a filesystem path (str) or a file-like object from tar.extractfile()
    """
    if isinstance(file_or_obj, str):
        with open(file_or_obj, "r", encoding="utf-8") as f:
            return f.read()
    else:
        # tarfile returns a binary file-like object
        return file_or_obj.read().decode("utf-8")


def load_descriptions(foler_path):
    descriptions = {}
    for root, _, files in os.walk(foler_path):
        for file in files:
            if file.endswith(".txt"):
                txt_path = os.path.join(root, file)
                descriptions[file.removesuffix(".txt")] = load_txt(txt_path)

    return descriptions


def load_descriptions_from_tar(tar_path):
    descriptions = {}
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(".txt"):
                f = tar.extractfile(member)
                if f is None:
                    continue

                text = load_txt(f)

                inner_name = os.path.basename(member.name).replace(":", "_")
                key = re.sub(r"_\d+$", "", inner_name.removeprefix("camera1.zip_camera1_").removesuffix(".jpg.jpg.txt"))
                descriptions[int(key)] = text

    return descriptions


MODEL_DEFAULT_TOKENS = {
    "llama": 2048,
    "phi_reason": 8192,
    "phi_standard": 8192,
    "qwen": 32768,
    "gpt_oss": 8192,
    "generic": 1024,
}


class ModelResponder:
    def __init__(self, model, tokenizer, model_name: str, thinking: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.thinking = thinking

        self.model_type = self._detect_model_type(self.model_name)
        self.default_max_new_tokens = MODEL_DEFAULT_TOKENS[self.model_type]
        self._response_fn = self._select_response_fn()

    @staticmethod
    def _detect_model_type(model_name: str) -> str:
        mid = model_name.lower()

        if "qwen" in mid:
            return "qwen"
        if "phi-4-reasoning" in mid:
            return "phi_reason"
        if "phi-4" in mid:
            return "phi_standard"
        if "gpt-oss" in mid:
            return "gpt_oss"
        if "llama" in mid:
            return "llama"
        return "generic"

    def _select_response_fn(self):
        return {
            "qwen": self._get_qwen_response,
            "phi_reason": self._get_phi_response,
            "phi_standard": self._get_phi_response,
            "gpt_oss": self._get_gpt_oss_response,
            "llama": self._get_llama_response,
            "generic": self._get_generic_response,
        }.get(self.model_type, self._get_generic_response)

    # ========= Public API =========

    def generate(self, messages, max_new_tokens: int | None = None):
        if max_new_tokens is None:
            max_new_tokens = self.default_max_new_tokens
        return self._response_fn(messages, max_new_tokens)

    # ========= Implementations =========

    def _generate_with_template(self, messages, max_new_tokens):
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _get_llama_response(self, messages, max_new_tokens):
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        answer = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )
        return None, answer

    def _get_phi_response(self, messages, max_new_tokens):
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
        )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        think_blocks = re.findall(r"<think>([\s\S]*?)</think>", text)

        if not think_blocks:
            raise ValueError("No <think> block found.")

        thinking = think_blocks[-1].strip()
        answer = text.rsplit("</think>", 1)[1].strip()

        return thinking, answer

    def _get_qwen_response(self, messages, max_new_tokens):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=max_new_tokens)

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

        THINK_END = 151668
        try:
            idx = len(output_ids) - output_ids[::-1].index(THINK_END)
        except ValueError:
            idx = 0

        thinking = self.tokenizer.decode(output_ids[:idx], skip_special_tokens=True).strip()

        content = self.tokenizer.decode(output_ids[idx:], skip_special_tokens=True).strip()

        return thinking, content

    def _get_gpt_oss_response(self, messages, max_new_tokens):
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            reasoning_effort="high",
        ).to(self.model.device)

        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        prompt_len = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][prompt_len:]

        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        if "assistantfinal" in text:
            analysis, final = text.split("assistantfinal", 1)
            analysis = analysis.replace("analysis", "").strip()
            final = final.strip()
        else:
            analysis = ""
            final = text.strip()

        return analysis, final

    def _get_generic_response(self, messages, max_new_tokens):
        return None, self._generate_with_template(messages, max_new_tokens)


if __name__ == "__main__":
    model_name = "Qwen/Qwen-14B"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    responder = ModelResponder(model, tokenizer, model_name, thinking=True)
    messages = [
        {
            "role": "user",
            "content": "What are the important objects in the current scene? Those objects will be considered for the future reasoning and driving decision.",
        }
    ]
    thinking, content = responder.generate(messages)
    print(thinking)
    print(content)

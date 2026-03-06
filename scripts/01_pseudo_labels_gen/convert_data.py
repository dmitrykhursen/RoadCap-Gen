import json


import argparse
from pathlib import Path


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path, help="Path to folder containing jsonl files")
    args = parser.parse_args()

    folder: Path = args.folder

    for jsonl_file in folder.glob("*.jsonl"):
        process_file(jsonl_file)


if __name__ == "__main__":
    main()

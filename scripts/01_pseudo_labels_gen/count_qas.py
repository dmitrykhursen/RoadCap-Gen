import json
from pathlib import Path
import argparse

def process_folder(folder: Path):
    total_questions = 0
    bad_entries = []

    for jsonl_file in folder.rglob("*.jsonl"):
        # Skip files inside folders that contain "converted" in their name
        if "converted" in jsonl_file.parts: # should not happen as converted are in json instead of jsonl
            print(f"Skipping {jsonl_file} because it is inside a 'converted' folder.")
            continue

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Failed to parse JSON in {jsonl_file}:{line_num} -> {e}")
                    continue

                # Each line has one top-level key
                for top_id, content in data.items():
                    num_questions = len(content)
                    total_questions += num_questions

                    if num_questions != 15:
                        bad_entries.append({
                            "file": str(jsonl_file.stem),
                            "line": line_num,
                            "top_id": top_id.split("__")[-1],
                            "count": num_questions,
                        })

    return total_questions, bad_entries

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path, help="Path to folder containing jsonl files")
    args = parser.parse_args()

    total_questions, bad_entries = process_folder(args.folder)
    missing_qas = sum([15-entry["count"] for entry in bad_entries])
    print(f"\nTotal questions counted: {total_questions:,d}")
    print(f"Total questions missing: {missing_qas:,d}\n")

    if bad_entries:
        print(f"{len(bad_entries):,d} bad entries found with != 15 questions:\n")
        for entry in bad_entries:
            print(
                f"{entry['file']} (line {entry['line']}): "
                f"{entry['top_id']} -> {entry['count']} questions"
            )
    else:
        print("All entries have exactly 15 questions.")

if __name__ == "__main__":
    main()
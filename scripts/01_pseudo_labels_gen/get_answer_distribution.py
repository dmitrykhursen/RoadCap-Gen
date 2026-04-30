import json
import re
from collections import Counter, defaultdict
import tqdm

PATTERN_OBJ_GENERAL = r"<[^,]+,\s*[^,]+,\s*-?\d+(?:\.\d+)?,\s*-?\d+(?:\.\d+)?>"
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
INIT_QUESTION = "What are the important objects in the current scene? Those objects will be considered for the future reasoning and driving decision."


def load_json(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def generalize(q):
    for rgx, repl in COMPILED_PATTERNS:
        q = rgx.sub(repl, q)
    q = re.sub(" +", " ", q.strip())
    return q


full_train = load_json("data/v1_1_train_nus.json")
ratio_data = load_json("data/ratios.json")
test_questions = set()
should_find = set()
for entry in ratio_data:
    if entry["ratio_test"] > 0:
        test_questions.add(entry["question"])
        if entry["ratio_train"] > 0:
            should_find.add(entry["question"])


# found_q = set()
# answer_templates = defaultdict(set)
# for scene_id, scene_data in tqdm.tqdm(full_train.items()):
#     key_frames = scene_data["key_frames"]
#     for frame_id, frame_data in key_frames.items():
#         qa_lists = frame_data["QA"]

#         for cat, qa_list in qa_lists.items():
#             for qa in qa_list:
#                 gq = generalize(qa["Q"])
#                 if gq in test_questions or gq in INIT_QUESTION:
#                     found_q.add(gq)
#                     answer_templates[gq].add(generalize(qa["A"]))
# print(len(found_q))
# print(len(should_find))
# save_json({k: sorted(v) for k, v in answer_templates.items()}, "data/answer_templates.json")

obj_count_distribution = Counter()
total_matched_questions = 0
"""
33.52% have 3
27.31% have 4
21.49% have 5
17.68% have 6
"""
for scene_id, scene_data in tqdm.tqdm(full_train.items()):
    key_frames = scene_data["key_frames"]
    for frame_id, frame_data in key_frames.items():
        qa_lists = frame_data["QA"]

        for cat, qa_list in qa_lists.items():
            for qa in qa_list:
                gq = generalize(qa["Q"])
                if gq in INIT_QUESTION:
                    num_objs = generalize(qa["A"]).count("<obj>")
                    obj_count_distribution[num_objs] += 1
                    total_matched_questions += 1

print("\n--- Distribution of '<obj>' counts ---")
if total_matched_questions > 0:
    for num_objs, freq in sorted(obj_count_distribution.items()):
        percentage = (freq / total_matched_questions) * 100
        print(f"{percentage:.2f}% have {num_objs}")
else:
    print("No matching questions found in INIT_QUESTION.")

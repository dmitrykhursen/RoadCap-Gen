from collections import Counter
import json
import random


class TrackJSONQuestionGenerator:
    def __init__(self, templates_path: str):
        self.templates = self._load_templates(templates_path)

    def _load_templates(self, path: str) -> list[dict]:
        with open(path, "r") as f:
            data = json.load(f)
        return [cat for cat in data if cat["category"] not in ["Occlusion and Visibility", "Lane Topology and Ground Plane"]]

    def get_objects_for_frame(self, track_json_data: dict, target_frame_name: str ='', target_track_id: str = '') -> list[dict]:
        """
        Parses the JSON schema to find all objects present in a specific camera frame.
        """
        objects_in_frame = []

        category_counter = Counter()
        for track_obj in track_json_data["tracks"]:
            obj_id = track_obj["object_id"]
            category_name = track_obj["category"]

            for frame_data in track_obj["track"]:
                if frame_data["frame_name"] == target_frame_name and (target_track_id == '' or obj_id == target_track_id):
                    category_counter[category_name] += 1
        
                    objects_in_frame.append(
                        {
                            "id": obj_id,
                            "name": f"{category_name}_{category_counter[category_name]}",
                            'cam': frame_data["camera"],
                            "depth": frame_data["depth"],
                            "x": frame_data["x"],
                            "y": frame_data["y"],
                            "x_ego": frame_data["x_ego"],
                            "y_ego": frame_data["y_ego"],
                            "heading_x_ego": frame_data["heading_x_ego"],
                            "heading_y_ego": frame_data["heading_y_ego"],
                            "center_2d_px": frame_data["center_2d_px"],
                        }
                    )
                    break

        return objects_in_frame

    # --- GROUND TRUTH CALCULATION METHODS ---

    def gt_relative_position(self, obj1: dict, obj2: dict, threshold: float = 0.0) -> str:
        """
        Calculates 2D spatial arrangement of obj1 relative to obj2.
        nuScenes Ego Frame: +X is Right, +Y is Forward.
        """
        dx = obj1["x_ego"] - obj2["x_ego"]
        dy = obj1["y_ego"] - obj2["y_ego"]

        if threshold and (abs(dx) < threshold and abs(dy) < threshold):
            return "Too close to call"

        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        else:
            return "in front" if dy > 0 else "behind"

    def gt_orientation(self, obj: dict, thr_main: float = 0.85, thr_minor: float = 0.5) -> str:
        """
        Calculates where the object is facing relative to the ego vehicle.
        Combines position (x_ego, y_ego) with rotation (heading_x_ego, heading_y_ego).
        Returns 'unknown' if the orientation is ambiguous/diagonal.
        """
        hx = obj["heading_x_ego"]
        hy = obj["heading_y_ego"]
        py = obj["y_ego"]

        # Mostly Y-Aligned (Parallel to ego vehicle's path)
        if abs(hy) > thr_main and abs(hx) < thr_minor:
            if py > 0:
                # Object is IN FRONT of ego
                return "away from ego" if hy > 0 else "towards ego"
            else:
                # Object is BEHIND ego
                return "towards ego" if hy > 0 else "away from ego"

        # Mostly X-Aligned (Perpendicular to ego vehicle's path)
        elif abs(hx) > thr_main and abs(hy) < thr_minor:
            # For crossing traffic, absolute "left" or "right" of the image frame is usually best
            return "right" if hx > 0 else "left"

        # Diagonal or ambiguous
        return "unknown"

    def gt_same_direction(self, obj1: dict, obj2: dict) -> str:
        """
        Checks if two objects are facing the same general direction.
        """
        dir1 = self.gt_orientation(obj1)
        dir2 = self.gt_orientation(obj2)

        if dir1 == "unknown" or dir2 == "unknown":
            return "unknown"

        return "Yes." if dir1 == dir2 else "No."

    def gt_closer_object(self, obj1: dict, obj2: dict, threshold: float = 1.0) -> str:
        """
        Determines which object is closer based on depth.
        """
        if abs(obj1["depth"] - obj2["depth"]) < threshold:
            return "equally distant"
        return "obj1" if obj1["depth"] < obj2["depth"] else "obj2"

    # --- GENERATION METHOD ---

    def generate_scene_questions(self, track_json_data: dict, target_frame_name: str, num_questions=5) -> list[dict]:
        """Generates QA pairs for a specific image filename."""
        objects = self.get_objects_for_frame(track_json_data, target_frame_name)

        if len(objects) < 2:
            return [{"error": f"Found {len(objects)} objects in {target_frame_name}. Need at least 2 for relative questions."}]

        generated_qa = []

        while len(generated_qa) < num_questions:
            category = random.choice(self.templates)
            template = random.choice(category["templates"])
            q_str = template["question"]

            obj1, obj2 = random.sample(objects, 2)

            # Format question string
            obj1_str = f"<{obj1['name']}, {obj1['cam']}, {', '.join(map(str, obj1['center_2d_px']))}>"
            obj2_str = f"<{obj2['name']}, {obj2['cam']}, {', '.join(map(str, obj2['center_2d_px']))}>"
            q_str = q_str.replace("<obj>", obj1_str)
            q_str = q_str.replace("<obj1>", obj1_str)
            q_str = q_str.replace("<obj2>", obj2_str)

            answer = None

            if category["category"] == "Relative Position":
                rel_pos = self.gt_relative_position(obj1, obj2)
                if "Where is" in q_str:
                    answer = rel_pos
                elif "Is" in q_str and "<relation>" in q_str:
                    target_rel = random.choice(["left", "right", "in front", "behind"])
                    q_str = q_str.replace("<relation>", target_rel)
                    answer = "Yes." if target_rel == rel_pos else "No."

            elif category["category"] == "Depth and Distance":
                closer = self.gt_closer_object(obj1, obj2)
                if "Which object" in q_str:
                    answer = obj1_str if closer == "obj1" else obj2_str
                elif "closer" in q_str and "Is" in q_str:
                    answer = "Yes." if closer == "obj1" else "No."

            elif category["category"] == "Orientation":
                if "same direction" in q_str:
                    answer = self.gt_same_direction(obj1, obj2)
                    if answer == "unknown":
                        continue  # Skip if orientation data isn't present yet
                else:
                    facing = self.gt_orientation(obj1)
                    if facing == "unknown":
                        continue

                    if "Where is" in q_str:
                        answer = facing
                    elif "Is" in q_str and "<direction>" in q_str:
                        target_dir = random.choice(["towards ego", "away from ego", "left", "right"])
                        q_str = q_str.replace("<direction>", target_dir)
                        answer = "Yes." if target_dir == facing else "No."

            if answer:
                generated_qa.append(
                    {
                        "question": q_str,
                        "answer": answer,
                        "metadata": {
                            "frame_name": target_frame_name,
                            "category": category["category"],
                            "obj1_id": obj1["id"],
                            "obj2_id": obj2["id"],
                        },
                    }
                )

        return generated_qa

if __name__ == "__main__":
    # Example usage
    generator = TrackJSONQuestionGenerator("questions.json")
    with open("/scratch/project/eu-25-10/datasets/nuScenes_metadata/tracks_by_scene_v3/n008-2018-05-21-11-06-59-0400/tracks.json", "r") as f:
        track_data = json.load(f)

    questions = generator.generate_scene_questions(track_data, "n008-2018-05-21-11-06-59-0400__CAM_BACK__1526915243037570.jpg")
    for qa in questions:
        print(f"Q: {qa['question']}\nA: {qa['answer']}\nMetadata: {qa['metadata']}\n")
import argparse

from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time


def main(args):
    class ModelRespondervllm:
        def __init__(self, model_name, thinking=True):
            self.thinking = thinking
            self.model_name = model_name

            # Initialize vLLM (gpu_memory_utilization can be lowered if you hit OOM)
            self.llm = LLM(model=model_name, gpu_memory_utilization=0.9)
            self.tokenizer = self.llm.get_tokenizer()

            # Pre-compile sampling parameters, taken from https://qwen.readthedocs.io/en/latest/deployment/vllm.html#python-library
            if thinking:
                self.sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768)
            else:
                self.sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=32768)

        def generate(self, messages):
            outputs = self.llm.chat(
                messages,  # may need to nest in a list, unsure now
                self.sampling_params,
                chat_template_kwargs={"enable_thinking": self.thinking},  # Set to False to strictly disable thinking
            )

            return outputs[0].outputs[0].text.split("</think>")

    class ModelResponder:
        def __init__(self, model, tokenizer, model_name, thinking=True):
            self.model = model
            self.tokenizer = tokenizer
            self.thinking = thinking
            self.model_name = model_name.lower()

        @torch.inference_mode()
        def generate(self, messages, max_new_tokens=32768):
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.thinking
            )
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            kwargs = (
                {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
                if self.thinking
                else {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
            )
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)
            output_ids = generated_ids[0][len(inputs.input_ids[0]) :].tolist()

            THINK_END = 151668
            try:
                idx = len(output_ids) - output_ids[::-1].index(THINK_END)
            except ValueError:
                idx = 0

            thinking_text = self.tokenizer.decode(output_ids[:idx], skip_special_tokens=True).strip()
            content = self.tokenizer.decode(output_ids[idx:], skip_special_tokens=True).strip()
            return thinking_text, content

    test_prompt = """You are given a driving scene captured from the front camera of a car.
The scene is 2448 x 2048 (width x height) pixels in size.
You are also given a list of detected objects from a YOLO model, including:
- object type
- bounding box [x, y, width, height]
- middle point [cx, cy]

You must generate exactly one QA pair based on the provided question template and the specified object.

Use the question exactly as provided without any modifications.  

**ANSWER RULES**
Your answer MUST be supported by:
- the object list
- bounding box and coordinates
- visible spatial relationships
- the scene description (secondary, only for overall context. If the description contradicts the object list, trust the object list)

NEVER hallucinate:
- colors
- brands
- additional objects
- signs not listed

Do NOT use internal IDs in the answer.
Use descriptive phrases like:
    "a bus on the right side"
    "a pedestrian ahead on the crosswalk"

**COORDINATES AND SPATIAL INTERPRETATION**
(0,0) = top-left
x increases -> right  
y increases -> down  

smaller cy = farther away  
larger cy = closer to ego vehicle  

Horizontal zones (approx):
- Left: cx < 979
- Ahead: 979 < cx < 1468
- Right: cx > 1468

Ego vehicle:
    - bounding box [0, 1620, 1600, 428], midpoint [800, 1834]
    - always at bottom middle-left, partially visible, moving forward along the y axis

Bounding box format:
[x, y, width, height]


**OUTPUT FORMAT**

Output EXACTLY this JSON structure:

{
    "question": "...",
    "reasoning": "...",
    "short_answer": "..."
}


The question MUST contain correctly formatted object tags.
The reasoning MUST include spatial analysis.
The short_answer MUST be brief (yes/no or 1-2 sentences).


You are also provided consecutive 3D relative position measurements in meters, measured from the ego vehicle sensor from previous frames.
Each position is formatted as [x, y, z] with the following coordinate system:
- x < 0 left of ego vehicle, x > 0 right of ego vehicle
- y < 0 below sensor, y > 0 above sensor (sensor is mounted on the roof of the car about 1.5 meters high)
- z > 0 forward distance only (measured parallel to the ground from the sensor)

The positions are time-ordered from oldest to newest with 55 milliseconds between frames.
EXAMPLES:
- A position of [-1.5, 0.0, 10.0] means the object was 1.5 meters to the left of the ego vehicle and 10.0 meters ahead.
- [1.83, -2.48, 69.18] at t-165 ms, [1.81, -2.51, 69.18] at t-110 ms, [1.35, -2.37, 69.04] at t-55 ms, [1.35, -2.26, 69.04] at t-0 ms. means the object was 1.83 meters to the right, 2.48 meters below, and 69.18 meters ahead 165 milliseconds ago; 1.81 meters to the right, 2.51 meters below, and 69.18 meters ahead 110 milliseconds ago; 1.35 meters to the right, 2.37 meters below, and 69.04 meters ahead 55 milliseconds ago; and 1.35 meters to the right, 2.26 meters below, and 69.04 meters ahead now.

Special Case: New Objects
If an object is detected for the first time in the current frame (t=0 ms), it will have no historical measurements. Its entry will use this exact format:
- relative position at t-0 ms: [x, y, z]. No prior frames available.
EXAMPLES:
- relative position at t-0 ms: [-3.94, -0.48, 61.56]. No prior frames available.
This means the object is 3.94 meters to the left, 0.48 meters below, and 61.56 meters ahead of the ego vehicle in the current frame, with no previous data.

RULES FOR QA GENERATION:
- Never calculate velocity/direction for these objects (insufficient data).
- Prioritize objects with history for motion-based questions unless the query specifically references new objects.

Here is the list of objects detected in the scene. Use them to generate the QA pairs:
- pedestrian_0:
        -current bbox: [1108.7, 991.1, 23.6, 53.2]
        -current middle point: [1120.5, 1017.7].
        -relative positions from the last 4 frames: [-3.09, -0.55, 54.86] at t-165 ms, [-3.1, -0.63, 54.86] at t-110 ms, [-3.18, -0.45, 54.71] at t-55 ms, [-3.2, -0.44, 54.69] at t-0 ms.
- pedestrian_1:
        -current bbox: [1395.2, 981.8, 37.0, 90.8]
        -current middle point: [1413.7, 1027.3].
        -relative positions from the last 4 frames: [4.1, 0.05, 34.17] at t-165 ms, [4.11, 0.04, 34.17] at t-110 ms, [4.01, -0.17, 33.76] at t-55 ms, [4.04, -0.03, 33.78] at t-0 ms.
- pedestrian_2:
        -current bbox: [1097.1, 1001.9, 18.2, 47.3]
        -current middle point: [1106.2, 1025.5].
        -relative positions from the last 4 frames: [-3.3, -0.44, 54.89] at t-165 ms, [-3.28, -0.45, 54.92] at t-110 ms, [-3.31, -0.31, 54.75] at t-55 ms, [-3.29, -0.31, 54.74] at t-0 ms.
- pedestrian_3:
        -current bbox: [1002.9, 997.1, 20.1, 51.0]
        -current middle point: [1013.0, 1022.5].
        -relative positions from the last 4 frames: [-6.94, -0.36, 56.53] at t-165 ms, [-6.94, -0.35, 56.55] at t-110 ms, [-6.83, -0.44, 56.31] at t-55 ms, [-6.85, -0.4, 56.28] at t-0 ms.
- pedestrian_4:
        -current bbox: [952.5, 990.3, 26.3, 59.5]
        -current middle point: [965.6, 1020.0].
        -relative positions from the last 4 frames: [-7.41, -0.45, 49.13] at t-165 ms, [-7.39, -0.43, 49.1] at t-110 ms, [-7.35, -0.03, 48.88] at t-55 ms, [-7.35, -0.02, 48.91] at t-0 ms.
- pedestrian_5:
        -current bbox: [1070.6, 995.1, 21.3, 55.4]
        -current middle point: [1081.3, 1022.8].
        -relative positions from the last 4 frames: [-4.29, -0.37, 53.46] at t-165 ms, [-4.29, -0.36, 53.46] at t-110 ms, [-4.33, -0.2, 53.26] at t-55 ms, [-4.33, -0.12, 53.27] at t-0 ms.
- pedestrian_6:
        -current bbox: [931.1, 991.0, 24.2, 60.8]
        -current middle point: [943.3, 1021.4].
        -relative positions from the last 4 frames: [-7.69, -0.43, 46.12] at t-165 ms, [-7.71, -0.43, 46.12] at t-110 ms, [-7.65, -0.04, 45.84] at t-55 ms, [-7.67, 0.03, 45.86] at t-0 ms.
- pedestrian_7:
        -current bbox: [892.3, 988.1, 23.4, 58.7]
        -current middle point: [904.0, 1017.5].
        -relative positions from the last 4 frames: [-9.03, -0.54, 47.57] at t-165 ms, [-9.06, -0.57, 47.55] at t-110 ms, [-9.12, -0.33, 47.3] at t-55 ms, [-9.17, -0.11, 47.34] at t-0 ms.
- pedestrian_8:
        -current bbox: [981.5, 996.3, 22.5, 52.0]
        -current middle point: [992.8, 1022.3].
        -relative positions from the last 4 frames: [-6.84, -0.47, 50.92] at t-165 ms, [-6.86, -0.45, 50.92] at t-110 ms, [-6.9, -0.03, 50.68] at t-55 ms, [-6.9, 0.05, 50.68] at t-0 ms.
- pedestrian_9:
        -current bbox: [1085.6, 998.0, 12.6, 25.5]
        -current middle point: [1091.9, 1010.8].
        -relative positions from the last 4 frames: [-4.22, -0.75, 53.42] at t-165 ms, [-4.22, -0.75, 53.41] at t-110 ms, [-4.24, -0.66, 53.26] at t-55 ms, [-4.24, -0.56, 53.22] at t-0 ms.
- pedestrian_10:
        -current bbox: [455.2, 988.9, 42.5, 100.1]
        -current middle point: [476.5, 1038.9].
        -relative positions from the last 4 frames: [-12.0, 0.25, 25.86] at t-165 ms, [-12.07, 0.23, 25.83] at t-110 ms, [-12.06, 0.29, 25.52] at t-55 ms, [-12.09, 0.35, 25.51] at t-0 ms.
- pedestrian_11:
        -current bbox: [433.9, 983.7, 32.5, 95.5]
        -current middle point: [450.2, 1031.5].
        -relative positions from the last 4 frames: [-12.3, -0.08, 25.74] at t-165 ms, [-13.68, 0.12, 28.17] at t-110 ms, [-12.33, 0.02, 25.44] at t-55 ms, [-12.38, 0.03, 25.43] at t-0 ms.
- pedestrian_12:
        -current bbox: [338.2, 976.8, 46.2, 112.9]
        -current middle point: [361.3, 1033.2].
        -relative positions from the last 4 frames: [-13.67, 0.06, 25.06] at t-165 ms, [-13.73, 0.09, 25.02] at t-110 ms, [-13.76, -0.04, 24.93] at t-55 ms, [-13.75, -0.08, 25.0] at t-0 ms.
- pedestrian_13:
        -current bbox: [923.8, 991.5, 18.1, 57.0]
        -current middle point: [932.9, 1020.0].
        -relative position at t-0 ms: [-7.69, -0.05, 45.87]. No prior frames available.
- car_0:
        -current bbox: [1163.6, 1009.5, 183.2, 162.1]
        -current middle point: [1255.2, 1090.6].
        -relative positions from the last 4 frames: [0.14, 0.52, 15.34] at t-165 ms, [0.14, 0.52, 15.33] at t-110 ms, [0.13, 0.52, 15.37] at t-55 ms, [0.14, 0.54, 15.33] at t-0 ms.
- car_1:
        -current bbox: [1571.7, 981.7, 591.6, 421.4]
        -current middle point: [1867.5, 1192.4].
        -relative positions from the last 4 frames: [2.48, 0.54, 6.53] at t-165 ms, [2.52, 0.55, 6.47] at t-110 ms, [2.46, 0.56, 6.19] at t-55 ms, [2.49, 0.57, 6.14] at t-0 ms.
- car_2:
        -current bbox: [2021.3, 1102.3, 426.1, 639.3]
        -current middle point: [2234.4, 1421.9].
        -relative positions from the last 4 frames: [2.32, 0.75, 3.85] at t-165 ms, [2.28, 0.74, 3.73] at t-110 ms, [2.32, 0.72, 3.56] at t-55 ms, [2.27, 0.72, 3.43] at t-0 ms.
- car_3:
        -current bbox: [1312.8, 1013.3, 49.2, 45.3]
        -current middle point: [1337.5, 1036.0].
        -relative positions from the last 4 frames: [5.29, 0.01, 58.67] at t-165 ms, [5.29, 0.01, 58.67] at t-110 ms, [5.19, 0.05, 58.43] at t-55 ms, [5.19, 0.13, 58.44] at t-0 ms.
- car_4:
        -current bbox: [1130.9, 1012.8, 107.4, 100.4]
        -current middle point: [1184.7, 1063.0].
        -relative positions from the last 4 frames: [-0.7, 0.37, 24.11] at t-165 ms, [-0.7, 0.37, 24.07] at t-110 ms, [-0.7, 0.39, 24.04] at t-55 ms, [-0.69, 0.38, 24.06] at t-0 ms.
- car_5:
        -current bbox: [1348.6, 1013.2, 38.5, 38.5]
        -current middle point: [1367.9, 1032.5].
        -relative positions from the last 4 frames: [5.59, -0.01, 59.3] at t-165 ms, [5.54, 0.0, 59.13] at t-110 ms, [5.43, 0.04, 58.81] at t-55 ms, [5.44, 0.1, 58.78] at t-0 ms.
- car_6:
        -current bbox: [1497.6, 1003.7, 220.9, 224.8]
        -current middle point: [1608.0, 1116.1].
        -relative positions from the last 4 frames: [2.51, 0.47, 12.06] at t-165 ms, [2.51, 0.47, 11.86] at t-110 ms, [2.48, 0.59, 12.16] at t-55 ms, [2.48, 0.58, 11.98] at t-0 ms.
- car_7:
        -current bbox: [1113.9, 998.9, 93.2, 68.3]
        -current middle point: [1160.5, 1033.0].
        -relative positions from the last 4 frames: [-1.61, -0.26, 46.47] at t-165 ms, [-1.69, -0.28, 46.53] at t-110 ms, [-1.61, -0.41, 46.08] at t-55 ms, [-1.76, -0.28, 45.9] at t-0 ms.
- car_8:
        -current bbox: [1302.7, 989.5, 59.2, 27.1]
        -current middle point: [1332.3, 1003.0].
        -relative positions from the last 4 frames: [5.02, -1.34, 71.68] at t-165 ms, [4.96, -1.36, 71.68] at t-110 ms, [4.89, -1.32, 70.8] at t-55 ms, [4.84, -1.27, 70.71] at t-0 ms.
- car_9:
        -current bbox: [1381.5, 1015.7, 24.6, 33.7]
        -current middle point: [1393.8, 1032.6].
        -relative positions from the last 4 frames: [3.87, -0.23, 34.21] at t-165 ms, [3.89, -0.19, 34.19] at t-110 ms, [3.89, -0.13, 33.76] at t-55 ms, [3.89, -0.06, 33.78] at t-0 ms.
- car_10:
        -current bbox: [1248.5, 1006.7, 67.4, 19.2]
        -current middle point: [1282.1, 1016.4].
        -relative positions from the last 4 frames: [1.86, -0.57, 47.13] at t-165 ms, [1.89, -0.58, 47.18] at t-110 ms, [3.2, -1.23, 87.63] at t-55 ms, [1.89, -0.45, 46.93] at t-0 ms.
- traffic sign_0:
        -current bbox: [1549.7, 789.0, 101.3, 148.2]
        -current middle point: [1600.3, 863.1].
        -relative positions from the last 4 frames: [3.75, -1.38, 16.76] at t-165 ms, [3.77, -1.38, 16.76] at t-110 ms, [4.06, -1.97, 16.44] at t-55 ms, [4.09, -1.98, 16.44] at t-0 ms.
- traffic sign_1:
        -current bbox: [1318.6, 954.3, 23.8, 24.5]
        -current middle point: [1330.5, 966.5].
        -relative positions from the last 4 frames: [2.72, -1.65, 38.68] at t-165 ms, [2.72, -1.65, 38.68] at t-110 ms, [2.82, -1.55, 38.4] at t-55 ms, [2.82, -1.53, 38.4] at t-0 ms.
- traffic sign_2:
        -current bbox: [1447.2, 891.1, 39.5, 41.2]
        -current middle point: [1466.9, 911.7].
        -relative positions from the last 4 frames: [4.43, -2.17, 28.6] at t-165 ms, [4.46, -2.03, 28.62] at t-110 ms, [4.51, -2.09, 28.35] at t-55 ms, [4.51, -2.08, 28.35] at t-0 ms.
- traffic sign_3:
        -current bbox: [1360.6, 945.7, 20.3, 21.5]
        -current middle point: [1370.8, 956.5].
        -relative positions from the last 4 frames: [4.71, -2.29, 47.92] at t-165 ms, [4.73, -2.29, 47.91] at t-110 ms, [4.63, -2.16, 47.7] at t-55 ms, [4.62, -2.11, 47.7] at t-0 ms.
- traffic sign_4:
        -current bbox: [1453.5, 930.4, 28.3, 20.1]
        -current middle point: [1467.6, 940.4].
        -relative positions from the last 4 frames: [4.42, -1.68, 28.63] at t-165 ms, [4.42, -1.68, 28.63] at t-110 ms, [4.55, -1.75, 28.37] at t-55 ms, [4.54, -1.67, 28.36] at t-0 ms.
- traffic sign_5:
        -current bbox: [1232.3, 965.4, 31.2, 19.7]
        -current middle point: [1247.9, 975.2].
        -relative positions from the last 4 frames: [1.83, -2.48, 69.18] at t-165 ms, [1.81, -2.51, 69.18] at t-110 ms, [1.35, -2.37, 69.04] at t-55 ms, [1.35, -2.26, 69.04] at t-0 ms.

Do not include all objects in your answer. Focus only on the most important objects that influence the driving decision of the ego vehicle.


The scene description is:
The image shows a city street with a mix of vehicles, including cars and a truck, parked along the side. There are also some pedestrians visible on the sidewalk. The street is lined with buildings, and there are traffic lights and street signs. The sky is clear, suggesting it might be a sunny day. The street appears to be in a European city, as indicated by the architecture of the buildings and the style of the street signs.

The question to use is:
What are the important objects in the current scene? Those objects will be considered for the future reasoning and driving decision."""

    model_name = args.model_name

    if args.use_vllm:
        responder = ModelRespondervllm(model_name, args.thinking)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", device_map="auto")
        responder = ModelResponder(model, tokenizer, model_name, thinking=args.thinking)

    start = time.time()
    messages = [{"role": "user", "content": test_prompt}]
    response = responder.generate(messages)
    print("Time taken:", end := time.time() - start)
    
    if len(response) == 1:
        response = ("", response[0])

    print(f"Model: {model_name}, use_vllm: {args.use_vllm}, thinking: {args.thinking}")
    with open("results_tweaked_non_thinking.txt", "a") as f:
        f.write(f"Running test with model: {args.model_name}, use_vllm: {args.use_vllm}, thinking: {args.thinking}\n")
        f.write(f"Time taken: {end}\n")
        f.write(f"{80 * '='}\n")
        f.write(f"Thinking: {response[0]}\n")
        f.write(f"{80 * '-'}\n")
        f.write(f"Response: {response[1]}\n")
        f.write(f"{80 * '-'}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()
    main(args)

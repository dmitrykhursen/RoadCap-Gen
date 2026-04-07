import json
import os

def convert_json_format(input_filepath, output_filepath):
    """
    Reads a JSON file in the DriveLM LLaVA/Adapter format and converts it
    into a simplified ground truth format containing id, question, and answer.
    """
    
    # 1. Check if the input file exists
    if not os.path.exists(input_filepath):
        print(f"❌ Error: The file '{input_filepath}' was not found.")
        return

    # 2. Read the input JSON
    try:
        with open(input_filepath, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"❌ Error: '{input_filepath}' is not a valid JSON file.")
        return

    # 3. Process the data
    output_data = []
    
    for item in data:
        item_id = item.get("id")
        conversations = item.get("conversations", [])
        
        question = ""
        answer = ""
        
        # 4. Extract the Human question and GPT answer
        for msg in conversations:
            if msg.get("from") == "human":
                # We need to strip out the "<image>\n" tag from the question string
                raw_question = msg.get("value", "")
                question = raw_question.replace("<image>\n", "").strip()
                
            elif msg.get("from") == "gpt":
                answer = msg.get("value", "").strip()
        
        # 5. Build the new dictionary format
        # Only append if we successfully found both a question and an answer
        if item_id and question and answer:
            output_data.append({
                "id": item_id,
                "question": question,
                "answer": answer
            })

    # 6. Save the new list to the output JSON file
    try:
        with open(output_filepath, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"✅ Successfully converted {len(output_data)} items!")
        print(f"📂 Saved to: {output_filepath}")
    except Exception as e:
        print(f"❌ Error saving the output file: {e}")


# --- Execution ---
if __name__ == "__main__":
    INPUT_JSON = "/mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/v1_2_val0.1_converted_llama_with_tags.json" 
    OUTPUT_JSON = "/mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/v1_2_GT_val0.1_output.json"
    
    # Run the converter
    convert_json_format(INPUT_JSON, OUTPUT_JSON)

# source /mnt/proj1/eu-25-10/envs/roadcap-gen/bin/activate
# python /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/src/utils/convert_to_gt_json_format.py
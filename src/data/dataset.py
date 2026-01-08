# src/data/dataset.py
import json
import torch
from .frontcam_dataset import FRONTCAMDataset

class RoadCapDataset(FRONTCAMDataset):
    """
    Main dataset for Training.
    Inherits ZIP loading capabilities from PONEDatasetBase.
    Adds Question Flattening + Tokenization.
    """
    def __init__(self, data_path, image_folder, tokenizer, image_processor, model_style="llava", task="vqa"):
        # 1. Initialize Base (Scans ZIPs)
        super().__init__(image_folder)
        
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_style = model_style
        self.task = task

        # 2. Load & Flatten JSON
        raw_data = json.load(open(data_path))
        print(f"📄 JSON loaded. Found {len(raw_data)} entries.")

        self.flat_samples = []
        
        for img_id_str, questions_dict in raw_data.items():
            # Convert ID "2108" -> Index 2108
            try:
                img_idx = int(img_id_str)
            except ValueError:
                continue

            # Validate index exists in ZIPs
            if img_idx >= len(self.image_paths_list):
                continue

            # Flatten questions
            for q_key, q_val in questions_dict.items():
                if 'question' not in q_val or 'short_answer' not in q_val:
                    continue
                
                self.flat_samples.append({
                    "image_index": img_idx, # Just store the INT index
                    "question": q_val['question'],
                    "answer": q_val['short_answer']
                })
        
        print(f"✅ Flattened VQA Dataset. Ready to train on {len(self.flat_samples)} samples.")

    def __len__(self):
        return len(self.flat_samples)

    def __getitem__(self, idx):
        # 1. Get sample metadata
        sample = self.flat_samples[idx]
        
        # 2. Load Image (Using Base Class Method)
        # We pass the INTEGER index stored in our sample
        image = self.get_image_by_index(sample['image_index'])

        print(f"Processing Sample Index: {idx}")  # Debug print
        print(f"image type: {type(image)}")  # Debug print
        print(f"image size: {image.size}")  # Debug print
        
        # 3. Process Image (Transform to Tensor)
        # model_inputs = self.image_processor(image, return_tensors="pt", do_pad=False) #, do_resize=True, do_pad=False, size={"height": 336, "width": 336})
        model_inputs = self.image_processor(
            image, 
            return_tensors="pt", 
            size={"height": 336, "width": 336}, # Force exact size
            do_resize=True,
            # do_image_splitting=False,           # Explicitly disable AnyRes
            do_pad=False
        )

        pixel_values = model_inputs['pixel_values'][0]
        image_sizes = model_inputs['image_sizes'][0]

        print(f"Image Sizes Tensor: {image_sizes}")  # Debug print
        print(f"Pixel Values Shape: {pixel_values.shape}")  # Debug print
        # print(model_inputs)  # Debug print
        print()
     

        # 4. Process Text (Prompt + Tokenize)
        # prompt = f"[INST] <image>\n{sample['question']} [/INST] {sample['answer']}"
        prompt = f"USER: <image>\nExtract JSON.\nASSISTANT: {sample['question']}"
        tokenized = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

        return {
            "input_ids": tokenized.input_ids[0],
            "labels": tokenized.input_ids[0],
            "pixel_values": pixel_values,
            "image_sizes": image_sizes
        }
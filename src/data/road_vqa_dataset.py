# src/data/dataset.py
import json
import torch
from .frontcam_dataset import FRONTCAMDataset
import random


class RoadVQADataset(FRONTCAMDataset):
    def __init__(
        self, 
        data_path, 
        image_folder, 
        tokenizer, 
        image_processor, 
        model_style="llava", 
        task="vqa",
        split="train",             # "train", "val", or "test"
        split_ratio=(0.8, 0.1, 0.1), # (Train, Val, Test) sum should be 1.0
        data_usage=1.0,            # 1.0 = 100%, 0.1 = 10% of the dataset
        seed=42                    # fixed seed for consistent splitting
    ):
        super().__init__(image_folder)
        
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_style = model_style
        
        if hasattr(self.image_processor, "image_grid_pinpoints"):
             self.image_processor.image_grid_pinpoints = None
        self.image_processor.size = {"height": 336, "width": 336}
        self.image_processor.crop_size = {"height": 336, "width": 336}

        # 1. Load & Flatten JSON (Load ALL data first)
        raw_data = json.load(open(data_path))
        all_samples = []

        for img_id_str, questions_dict in raw_data.items():
            try:
                img_idx = int(img_id_str)
            except ValueError:
                continue
            if img_idx >= len(self.image_paths_list):
                continue

            for q_key, q_val in questions_dict.items():
                if 'question' not in q_val or 'short_answer' not in q_val:
                    continue
                
                all_samples.append({
                    "image_index": img_idx,
                    "question": q_val['question'],
                    "answer": q_val['short_answer']
                })

        
        # 2. SPLITTING LOGIC (Train / Val / Test)
        # deterministic Shuffle (Must happen before slicing!)
        random.seed(seed)
        random.shuffle(all_samples)
        
        total = len(all_samples)
        train_end = int(total * split_ratio[0])
        val_end = int(total * (split_ratio[0] + split_ratio[1]))
        
        # select the specific split
        if split == "train":
            self.flat_samples = all_samples[:train_end]
        elif split == "val":
            self.flat_samples = all_samples[train_end:val_end]
        elif split == "test":
            self.flat_samples = all_samples[val_end:]
        else:
            raise ValueError(f"Unknown split: {split}")

        # 3. SUBSET LOGIC (Train on x% of the data)
        # If data_usage < 1.0, we slice off a percentage of THIS split
        if data_usage < 1.0:
            subset_size = int(len(self.flat_samples) * data_usage)
            self.flat_samples = self.flat_samples[:subset_size]
            print(f"✂️  Subsampling: Using {data_usage*100}% of {split} set.")

        print(f"✅ Dataset Ready [{split.upper()}]: {len(self.flat_samples)} samples (Total pool: {total})")

    def __len__(self):
        return len(self.flat_samples)

    def __getitem__(self, idx):
        sample = self.flat_samples[idx]
        
        # 1. Load Image
        image = self.get_image_by_index(sample['image_index'])


        # 2. Resize to model expected size
        image = image.resize((224, 224))

        return {
            "image": image,
            "question": sample['question'],
            "answer": sample['answer'],          
        }
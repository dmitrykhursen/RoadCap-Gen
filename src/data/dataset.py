# src/data/dataset.py
import json
import os
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor
from .processor import PromptFormatter 

class RoadCapDataset(Dataset):
    def __init__(self, data_path, image_folder, tokenizer_path, task="qa", model_style="llava_vicuna"):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.image_folder = image_folder
        self.task = task
        
        # 1. Initialize Processor (HuggingFace) for tokenization/image processing
        self.processor = AutoProcessor.from_pretrained(tokenizer_path)
        
        # 2. Initialize Formatter (Our Code) for text structuring
        self.formatter = PromptFormatter(model_type=model_style)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load Image
        img_path = os.path.join(self.image_folder, item['image'])
        image = Image.open(img_path).convert('RGB')
        
        # Load Text
        question = item['question']
        answer = item['answer']
        
        # --- KEY CHANGE: Use the Formatter ---
        # The dataset doesn't need to know if it's [INST] or USER:
        full_text = self.formatter.apply_prompt(question, answer)

        return {
            "image": image,
            "text": full_text
        }
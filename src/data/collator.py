import torch
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class RoadCapCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # 1. Stack Images 
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        
        # 2. Stack Image Sizes
        # This creates a tensor of shape (Batch_Size, 2) -> [[h, w], [h, w], ...]
        image_sizes = torch.stack([item['image_sizes'] for item in batch])

        # 3. Extract Text
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        # 4. Dynamic Padding (Same as before)
        input_ids_padded = pad_sequence(
            input_ids, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )
        labels_padded = pad_sequence(
            labels, 
            batch_first=True, 
            padding_value=-100 
        )
        attention_mask = input_ids_padded.ne(self.tokenizer.pad_token_id)

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
            "labels": labels_padded
        }
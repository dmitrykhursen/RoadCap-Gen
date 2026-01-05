import torch

class RoadCapCollator:
    def __init__(self, processor):
        self.processor = processor
        # We need the tokenizer to know what the 'pad' token ID is
        self.tokenizer = processor.tokenizer 

    def __call__(self, batch):
        images = [x['image'] for x in batch]
        texts = [x['text'] for x in batch]

        # 1. Process Images and Text together
        # The processor handles resizing, normalizing images, and tokenizing text
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,          # Pad to longest sequence in batch
            truncation=True,
            max_length=2048
        )
        
        # 2. Create Labels
        inputs['labels'] = inputs['input_ids'].clone()
        
        # 3. CRITICAL FIX: Mask padding tokens
        # We find where the input is 'pad_token', and set the label to -100
        pad_token_id = self.tokenizer.pad_token_id
        
        # Create a mask: True where it is padding
        padding_mask = inputs['input_ids'] == pad_token_id
        
        # Apply -100 to labels
        inputs['labels'][padding_mask] = -100
        
        return inputs
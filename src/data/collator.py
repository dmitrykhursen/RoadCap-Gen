import torch
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class RoadCapCollator:
    def __init__(self, processor):
        self.processor = processor  # this includes tokenizer + image encoder
        self.tokenizer = processor.tokenizer
        # self.MAX_LENGTH = 2048
        self.MAX_LENGTH = 8192

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        images = []
        text_prompts = []

        # 1. Build prompts + collect images
        for ex in examples:
            image = ex["image"] if "image" in ex else None
            gt = ex["answer"] if "answer" in ex else ex["labels"].tolist()

            if isinstance(image, list):  # ensure 1 image
                image = image[0]
            images.append(image)

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": ex["question"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ex["answer"]},
                    ],
                }
            ]

            prompt = self.processor.apply_chat_template(conversation)
            text_prompts.append(prompt)

        # 2. Process batch through LLaVA processor
        batch = self.processor(
            text=text_prompts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.MAX_LENGTH,
            return_tensors="pt"
        )

        # 3. Create labels mask
        labels = batch["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch  # dict format required by Trainer
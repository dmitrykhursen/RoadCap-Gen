import torch
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class RoadCapCollator:
    def __init__(self, processor):
        self.processor = processor  # this includes tokenizer + image encoder
        self.tokenizer = processor.tokenizer
        self.MAX_LENGTH = 8192

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = []
        text_prompts = []
        
        # 1. Initialize metadata lists
        img_paths = []
        questions = []
        tags = []
        ids = []
        depth_latents_list = []

        # 2. Build prompts + collect images and metadata
        for ex in examples:
            # Handle image
            image = ex.get("image")
            if isinstance(image, list):  
                image = image[0]
            images.append(image)

            # Extract metadata (with safe fallbacks)
            img_paths.append(ex.get("img_path", "unknown"))
            questions.append(ex.get("question", ""))
            tags.append(ex.get("tag", [-1]))
            ids.append(ex.get("id", -1))
            depth_latents_list.append(ex.get("depth_latents", None))

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": ex.get("question", "")},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ex.get("answer", "")},
                    ],
                }
            ]

            prompt = self.processor.apply_chat_template(conversation)
            text_prompts.append(prompt)

        # 3. Process batch through LLaVA processor
        batch = self.processor(
            text=text_prompts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.MAX_LENGTH,
            return_tensors="pt"
        )

        # 4. Create labels mask
        labels = batch["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        # 5. Attach the metadata
        batch["img_paths"] = img_paths
        batch["questions"] = questions
        batch["tags"] = tags
        batch["ids"] = ids

        # Stack depth latents: use zeros for samples missing depth embeddings
        zero_latent = torch.zeros(32, 35, 63, dtype=torch.float32)
        stacked = torch.stack([
            d if d is not None else zero_latent
            for d in depth_latents_list
        ], dim=0)  # [B, 32, 35, 63]
        batch["depth_latents"] = stacked
        batch["depth_latents_mask"] = torch.tensor(
            [d is not None for d in depth_latents_list], dtype=torch.bool
        )

        # 6. Convert BatchEncoding to a standard dict
        return dict(batch)
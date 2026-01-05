# src/models/llava_module.py
import torch
import torch.nn as nn
from transformers import LlavaNextForConditionalGeneration

class RoadCapLLaVA(LlavaNextForConditionalGeneration):
    def __init__(self, config):
        # Load the base LLaVA 1.6 model
        super().__init__(config)
        
        # Initialize Projection Head (Required for your future Extended Mode)
        # LLaVA 1.6 Vicuna usually has hidden size 4096
        self.geo_projector = nn.Linear(4096, 256) 

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        # Override to load correctly from HF
        model = super().from_pretrained(path, **kwargs)
        # Manually initialize the projector since it's not in the pre-trained weights
        model.geo_projector = nn.Linear(4096, 256)
        return model

    def get_lora_target_modules(self):
        # LLaVA 1.6 uses these layers. Targeting 'vision_tower' is optional but often helps.
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def get_visual_features(self, outputs):
        # Extract features for Extended Mode
        # LLaVA-NeXT (1.6) outputs can vary, but usually:
        hidden_states = outputs.hidden_states[-1]
        return self.geo_projector(hidden_states)
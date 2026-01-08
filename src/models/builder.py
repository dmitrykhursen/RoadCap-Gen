# src/models/builder.py
from .llava_module import RoadCapLLaVA
from transformers import AutoConfig
import torch

def build_model(cfg):
    print(f"Loading Model: {cfg.model.path}")

    if getattr(cfg.training, "bf16", False):
        compute_dtype = torch.bfloat16
        print("Model loading with: bfloat16 (Best for A100)")
    else:
        compute_dtype = torch.float16
        print("Model loading with: float16")
    
    if cfg.model.type == "llava":
        # load config first
        hf_config = AutoConfig.from_pretrained(cfg.model.path)
        # print(f"HF Config Loaded: {hf_config}")
        
        # Load Model (handling 4-bit loading happens in peft_utils usually, 
        # but for simplicity we load full here, or let Trainer handle it)
        model = RoadCapLLaVA.from_pretrained(
            cfg.model.path,
            dtype=compute_dtype,
        )
        return model
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
# src/models/builder.py
from .llava_module import RoadCapLLaVA
from transformers import AutoConfig
import torch

def build_model(cfg):

    print(f"build model - model dtype: {cfg.model.dtype=}")
    if cfg.model.dtype == "bf16":
        compute_dtype = torch.bfloat16
        print("Model loading with: bfloat16 (Best for A100)")
    else:
        compute_dtype = torch.float16
        print("Model loading with: float16")
    
    if cfg.model.type == "llava":
        inference_cfg = cfg.get("inference") 
        if inference_cfg is None or inference_cfg.get("load_mode") == "pretrained":
            model = RoadCapLLaVA.from_pretrained(
                cfg.model.path,
                dtype=compute_dtype,
            )
            print(f"Loaded  Model: {cfg.model.path}")

        elif cfg.inference.load_mode == "finetune":
            model = RoadCapLLaVA.from_pretrained(
                cfg.inference.checkpoint_model,
                dtype=compute_dtype,
            )
            print(f"Loaded  Model: {cfg.inference.checkpoint_model}")

        else:
            # error
            raise ValueError(f"Unknown load_mode: {cfg.inference.load_mode}")

        # DriveLM uses a 2x3 grid of 336px tiles -> (672, 1008)
        if hasattr(model.config, "image_grid_pinpoints"):
            custom_resolution = (672, 1008) # define DriveLM 2x3 grid resolution: (Height=672, Width=1008)
            model.config.image_grid_pinpoints = list(model.config.image_grid_pinpoints) + [custom_resolution]

        
        return model
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
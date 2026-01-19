# src/models/builder.py
from .llava_module import RoadCapLLaVA
from transformers import AutoConfig
import torch

def build_model(cfg):

    if getattr(cfg.model.dtype, "bf16", False):
        compute_dtype = torch.bfloat16
        print("Model loading with: bfloat16 (Best for A100)")
    else:
        compute_dtype = torch.float16
        print("Model loading with: float16")
    
    if cfg.model.type == "llava":
        
        # Load Model (handling 4-bit loading happens in peft_utils usually, 
        # but for simplicity we load full here, or let Trainer handle it)

        # model = RoadCapLLaVA.from_pretrained(
        #     cfg.model.path,
        #     dtype=compute_dtype,
        # )
        print(f"Model Load Mode: {getattr(cfg.inference, 'load_mode', 'pretrained')}")

        if cfg.inference is None or cfg.inference.load_mode == "pretrained":
            model = RoadCapLLaVA.from_pretrained(
                cfg.model.path,
                dtype=compute_dtype,
            )
            print(f"Loaded  Model: {cfg.model.path}")

        elif cfg.inference.load_mode == "fullfinetune":
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
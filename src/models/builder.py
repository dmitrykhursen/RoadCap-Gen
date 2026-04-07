# src/models/builder.py
from .llava_module import RoadCapLLaVA
from transformers import AutoConfig
import torch
from external.DriveLM.challenge.llama_adapter_v2_multimodal7b import llama

def build_model(cfg):
    print(f"build model - model dtype: {cfg.model.dtype=}")
    if cfg.model.dtype == "bf16":
        compute_dtype = torch.bfloat16
        print("Model loading with: bfloat16 (Best for A100)")
    else:
        compute_dtype = torch.float16
        print("Model loading with: float16")
    
    # ---------------------------------
    # Branch 1: LLaVA
    # ---------------------------------
    if cfg.model.type == "llava":
        inference_cfg = cfg.get("inference") 
        if inference_cfg is None or inference_cfg.get("load_mode") == "pretrained":
            model = RoadCapLLaVA.from_pretrained(
                cfg.model.path,
                dtype=compute_dtype,
                attn_implementation="sdpa",
            )
            print(f"Loaded  Model: {cfg.model.path}")
        elif cfg.inference.load_mode in ["finetune", "adapter"]:
            model = RoadCapLLaVA.from_pretrained(
                cfg.inference.checkpoint_model,
                dtype=compute_dtype,
                attn_implementation="sdpa",
            )
            print(f"Loaded  Model: {cfg.inference.checkpoint_model}")
        else:
            raise ValueError(f"Unknown load_mode: {cfg.inference.load_mode}")

        # DriveLM uses a 2x3 grid of 336px tiles -> (672, 1008)
        if hasattr(model.config, "image_grid_pinpoints"):
            custom_resolution = (672, 1008)
            model.config.image_grid_pinpoints = list(model.config.image_grid_pinpoints) + [custom_resolution]

        return model

    # ---------------------------------
    # Branch 2: LLaMA-Adapter v2
    # ---------------------------------
    elif cfg.model.type == "llama_adapter_v2":
        # Checkpoint is the tuned adapter weights, llama_dir is the base Meta LLaMA weights
        checkpoint = cfg.inference.checkpoint_model
        llama_dir = cfg.model.llama_dir
        llama_type = "7B"
        
        print(f"Loading LLaMA-Adapter v2 (Type: {llama_type})...")
        # Load directly using the challenge's loader function
        # We load to CPU first; DDP script will move it to local_rank
        model, _ = llama.load(
            name=checkpoint, 
            llama_dir=llama_dir, 
            llama_type=llama_type, 
            device="cpu", 
            phase="finetune"
        )
        return model

    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
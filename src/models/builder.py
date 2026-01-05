# src/models/builder.py
from .llava_module import RoadCapLLaVA
from transformers import AutoConfig

def build_model(cfg):
    print(f"Loading Model: {cfg.model.path}")
    
    if cfg.model.type == "llava":
        # Load Config first
        hf_config = AutoConfig.from_pretrained(cfg.model.path)
        
        # Load Model (handling 4-bit loading happens in peft_utils usually, 
        # but for simplicity we load full here, or let Trainer handle it)
        model = RoadCapLLaVA.from_pretrained(
            cfg.model.path,
            torch_dtype="auto",
        )
        return model
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
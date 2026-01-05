import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def apply_peft(model, cfg):
    if not cfg.training.use_lora:
        return model

    print("Applying LoRA adapters...")
    
    # 1. Enable Gradient Checkpointing (Saves VRAM)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # 2. QLoRA Preparation (if using 4bit)
    if cfg.training.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    # 3. Define Config
    lora_config = LoraConfig(
        r=cfg.training.lora_r,
        lora_alpha=cfg.training.lora_alpha,
        target_modules=model.get_lora_target_modules(),
        lora_dropout=cfg.training.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 4. Wrap Model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Critical: Make sure our custom projector is trainable if we were in extended mode
    # (For simple mode, this doesn't matter, but good practice)
    if hasattr(model, "geo_projector"):
        for param in model.geo_projector.parameters():
            param.requires_grad = True
            
    return model
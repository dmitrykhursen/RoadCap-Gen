import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# src/training/peft_utils.py
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def apply_peft(model, cfg):
    """
    Applies the PEFT wrapper to efficient saving.
    """

    if not cfg.training.use_lora:
        # Full finetuning mode
        print("🚫 LoRA/PEFT disabled. Using standard full-model saving (Large Files).")
        # FREEZE EVERYTHING FIRST
        for param in model.parameters():
            param.requires_grad = False

        # Manual unfreeze for safety if user disables PEFT but wants to train projector
        if "multi_modal_projector" in getattr(cfg.training, "train_modules", []):
             for n, p in model.named_parameters():
                 if "multi_modal_projector" in n:
                     p.requires_grad = True
        
        print_trainable_parameters(model)
        return model

    print("🔧 Configuring PEFT (Efficient Saving Mode)...")
    
    # 2. QLoRA Prep
    if getattr(cfg.training, "load_in_4bit", False):
        model = prepare_model_for_kbit_training(model)

    # 3. Get Targets
    lora_targets, modules_to_save = _identify_training_targets(cfg, model)
    
    print(f"   🎯 LoRA Targets: {lora_targets}")
    print(f"   💾 Modules to Save: {modules_to_save}")

    # 4. Apply Wrapper
    # Even if lora_targets is empty, we use LoraConfig to manage 'modules_to_save'
    # Note: We need at least one target_module typically, so if empty, we might skip LoRA 
    # but strictly speaking, PEFT expects *something*. 
    
    if len(lora_targets) == 0 and len(modules_to_save) > 0:
        print("   ℹ️  Note: Pure Full-Finetuning via PEFT (0 LoRA adapters).")
        # We pass a dummy target (that doesn't exist) or just rely on modules_to_save
        # A clean hack: target 'none', PEFT will just handle the modules_to_save
        lora_targets = [] 

    lora_config = LoraConfig(
        r=cfg.training.lora_r,
        lora_alpha=cfg.training.lora_alpha,
        target_modules=lora_targets if lora_targets else None, # Pass None if empty
        modules_to_save=modules_to_save, 
        lora_dropout=cfg.training.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)

    # 5. Verify
    print_trainable_parameters(model)

    return model


def _identify_training_targets(cfg, model):
    """
    Decides WHICH modules to train. 
    Returns:
       lora_targets: List of module names for LoRA (Low Rank)
       full_save_modules: List of module names for Full Finetuning (Full Rank)
    """
    modules_to_train = getattr(cfg.training, "train_modules", ["language_model"])
    target_layers = getattr(cfg.training, "target_modules", ["q_proj", "v_proj"])
    projector_mode = getattr(cfg.training, "projector_mode", "full")
    
    lora_targets = []
    full_save_modules = []

    print(f"📋 Training Scope: {modules_to_train}")

    # 1. Projector Logic
    if "multi_modal_projector" in modules_to_train:
        if projector_mode == "lora":
            lora_targets.extend(["linear_1", "linear_2"])
        else:
            # FULL MODE: Add to "Save List" so PEFT handles the saving
            full_save_modules.append("multi_modal_projector")

    # 2. Vision Tower Logic
    if "vision_model" in modules_to_train:
        lora_targets.extend(["vision_tower"])

    # 3. LLM Logic
    if "language_model" in modules_to_train:
        lora_targets.extend(target_layers)

    # # 4. Custom Heads (Extended Mode)
    # if hasattr(model, "geo_head") or hasattr(model, "geo_projector"):
    #     full_save_modules.append("geo_projector")
    #     full_save_modules.append("geo_head")

    return list(set(lora_targets)), list(set(full_save_modules))


def print_trainable_parameters(model):
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    else:
        # Fallback: Manually count parameters for standard models
        trainable_params = 0
        all_param = 0
        for name, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"✅ {name} : {param.shape}")

        
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || "
            f"trainable %: {100 * trainable_params / all_param:.2f}"
        )
            
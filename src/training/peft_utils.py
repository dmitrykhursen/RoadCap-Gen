import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def apply_peft(model, cfg):
    if not cfg.training.use_lora:
        return model

    print("Applying LoRA adapters...")

    # 1. Enable Gradient Checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # 2. QLoRA Preparation
    if getattr(cfg.training, "load_in_4bit", False):
        model = prepare_model_for_kbit_training(model)

    # 3. Determine Targets based on Config
    # Default to just language model if not specified
    modules_to_train = getattr(cfg.training, "train_modules", ["language_model"])
    target_layers = getattr(cfg.training, "target_modules", ["q_proj", "v_proj"])
    projector_mode = getattr(cfg.training, "projector_mode", "full") # "lora" or "full"
    
    print(f"Training Blocks: {modules_to_train}")
    print(f"Target Layers: {target_layers}")
    print(f"Projector Mode: {projector_mode}")

    # 4. Filter Targets for PEFT
    # We need to specify the full path for modules if we want to target specific blocks.
    # LLaVA structure: model.vision_tower, model.multi_modal_projector, model.language_model
    
    final_lora_targets = []
    
    # --- LOGIC: BUILD LORA TARGET LIST ---
    if "language_model" in modules_to_train:
        final_lora_targets.extend(target_layers)

    # B. Projector LoRA (Only if mode is 'lora')
    if "multi_modal_projector" in modules_to_train:
        if projector_mode == "lora":
            # The internal layers of LLaVA projector are named 'linear_1' and 'linear_2'
            final_lora_targets.extend(["linear_1", "linear_2"])
        else:
            print("   -> Projector Mode: FULL (Will unfreeze original weights later)")

    # C. Vision Model (Optional)
    if "vision_model" in modules_to_train:
        final_lora_targets.extend(["vision_tower"])

    # Remove duplicates
    final_lora_targets = list(set(final_lora_targets))
    print(f"🎯 Final LoRA Targets: {final_lora_targets}")

    # 4. WRAP MODEL WITH PEFT
    if len(final_lora_targets) > 0:
        lora_config = LoraConfig(
            r=cfg.training.lora_r,
            lora_alpha=cfg.training.lora_alpha,
            target_modules=final_lora_targets,
            lora_dropout=cfg.training.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
    else:
        print("⚠️ No LoRA targets! Skipping PEFT wrapper.")
        for param in model.parameters():
            param.requires_grad = False

    # 5. UNFREEZE LOGIC (For 'full' mode or custom heads)
    for name, param in model.named_parameters():
        # # A. Always unfreeze your custom Geo Projector
        # if "geo_projector" in name:
        #     param.requires_grad = True
            
        # B. Unfreeze Projector (If mode is 'full')
        if "multi_modal_projector" in modules_to_train and projector_mode == "full":
             if "multi_modal_projector" in name:
                 param.requires_grad = True

    # if projector_mode == "lora":
    #     model.print_trainable_parameters()

    # # Print specific layer names that are trainable
    # print("\n=== TRAINABLE LAYERS ===")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"✅ {name} : {param.shape}")
    # print("========================\n")

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
            
    return model
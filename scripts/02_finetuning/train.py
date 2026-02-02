import hydra
import torch
import os
from transformers import TrainingArguments
from src.models.builder import build_model
from src.training import trainer
from src.training.peft_utils import apply_peft
from src.training.trainer import RoadCapTrainer
from src.data.dataset_builder import build_dataset
from src.data.collator import RoadCapCollator
from transformers import AutoTokenizer, AutoImageProcessor, AutoProcessor

import sys
import logging
import transformers
import wandb
from omegaconf import OmegaConf

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg):
    # --- DDP SETUP & INFO START ---
    # Get distributed variables provided by 'torchrun'
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Print info only on the main process (Rank 0) to keep logs clean
    if global_rank == 0:
        print("="*50)
        print(f"🚀 DDP TRAINING STATUS")
        print(f"   • Total GPUs (World Size): {world_size}")
        print(f"   • Backend: nccl (NVIDIA defaults)")
        print(f"   • Master Port: {os.environ.get('MASTER_PORT', 'Default')}")
        print("="*50)
        print("=== RoadCap-Gen Fine-tuning Script ===")
        print(f"Args / config: {cfg}")
    
    # Print a quick confirmation from every GPU so you know they are alive
    print(f"[GPU {global_rank}] Online. Local Rank: {local_rank}. Ready to load model.")
    # --- DDP SETUP & INFO END ---

    # convert Hydra config to dict
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Only init WandB on the main process
    if global_rank == 0:
        wandb_run = wandb.init(
            project="RoadCap-Gen",
            name=cfg.experiment_name,
            config=config_dict,
            group=cfg.model.name,
            job_type="finetune",
        )

    # 1. load Tokenizer
    if global_rank == 0: print(f"Loading generic tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.path, use_fast=True, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else tokenizer.eos_token
    
    image_processor = AutoImageProcessor.from_pretrained(cfg.model.path, trust_remote_code=True, use_fast=True)
    if hasattr(image_processor, "image_grid_pinpoints"):
        custom_resolution = (672, 1008)
        image_processor.image_grid_pinpoints = list(image_processor.image_grid_pinpoints) + [custom_resolution]

    # 2. Build Model
    # NOTE: The Trainer will handle moving this to the correct GPU later!
    if global_rank == 0: print(f"Building Model: {cfg.model.name}")
    model = build_model(cfg)  

    if hasattr(model.config, "image_grid_pinpoints"):
        custom_resolution = (672, 1008)
        model.config.image_grid_pinpoints = list(model.config.image_grid_pinpoints) + [custom_resolution]
        
    if global_rank == 0: print(f"Loading Datasets...")
    
    train_dataset = build_dataset(
        cfg=cfg,
        tokenizer=tokenizer,
        image_processor=image_processor,
        split="train",
        data_usage=cfg.training.data_usage
    )

    eval_dataset = build_dataset(
        cfg=cfg,
        tokenizer=tokenizer,
        image_processor=image_processor,
        split="val",
        data_usage=cfg.training.data_usage
    )

    # 5. Apply LoRA
    model = apply_peft(model, cfg)
    if cfg.training.grad_checkpointing:
        model.gradient_checkpointing_enable()

    # 6. Training Arguments
    use_bf16 = getattr(cfg.training, "bf16", False)

    training_args = TrainingArguments(
        output_dir=f"output/{cfg.model.name}/{cfg.experiment_name}",
        report_to="wandb" if global_rank == 0 else "none", # Only log from main process
        run_name=cfg.experiment_name,
        per_device_train_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.grad_accumulation,
        learning_rate=cfg.training.learning_rate,
        num_train_epochs=cfg.training.epochs,
        logging_steps=cfg.training.logging_steps,
        logging_strategy="steps",
        bf16=use_bf16,
        fp16=not use_bf16,
        dataloader_num_workers=cfg.training.num_workers,
        remove_unused_columns=False, 
        gradient_checkpointing=cfg.training.grad_checkpointing,
        warmup_ratio=cfg.training.warmup_ratio,
        per_device_eval_batch_size=cfg.training.batch_size * 2,
        save_strategy="epoch",
        
        eval_strategy="epoch",
        # metric_for_best_model="eval_loss", 
        # greater_is_better=False,
        
        # KEY DDP SETTINGS
        ddp_find_unused_parameters=False, # Essential for PEFT/LoRA
        local_rank=local_rank,            
    )

    processor = AutoProcessor.from_pretrained(cfg.model.path, use_fast=True)
    processor.tokenizer.padding_side = "right"
    if hasattr(image_processor, "image_grid_pinpoints"):
        processor.image_processor.image_grid_pinpoints = image_processor.image_grid_pinpoints

    collator = RoadCapCollator(processor=processor)
    
    # NOTE: The Trainer class below handles the 'DistributedSampler' automatically.
    # It will verify 'world_size' > 1 and slice the 'train_dataset' for you.
    trainer = RoadCapTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        mode="simple"
    )

    if global_rank == 0: print("Starting Training Loop...")
    
    trainer.train()

    if global_rank == 0: print("Finished.")

if __name__ == "__main__":
    main()

# salloc -A EU-25-10 -p qgpu_exp --gpus-per-node 1 -t 5:00:00 --nodes 1

# source /mnt/proj1/eu-25-10/envs/roadcap-gen/bin/activate
# python scripts/02_finetuning/train.py     model=llava     dataset=qa_dataset     training=lora              experiment_name=qa_debug
# python scripts/02_finetuning/train.py     model=llava     dataset=drivelm        training=full_finetune     experiment_name=debug_llava_drivelm_fullfinetune

# torchrun --nproc_per_node=8 scripts/02_finetuning/train.py     model=llava     dataset=drivelm        training=full_finetune     experiment_name=tmp_debug_llava_drivelm_fullfinetune
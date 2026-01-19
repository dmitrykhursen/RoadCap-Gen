import hydra
import torch
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

# # setup Python's native logging to print to stdout (Terminal)
# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     handlers=[logging.StreamHandler(sys.stdout)],
#     level=logging.INFO, # Force native logger to INFO
# )

# # force Hugging Face Transformers to print INFO logs
# transformers.utils.logging.set_verbosity_info()
# # only print on the main GPU (Rank 0) to avoid duplicates in distributed training
# transformers.utils.logging.enable_default_handler()
# transformers.utils.logging.enable_explicit_format()


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg):
    print("=== RoadCap-Gen Fine-tuning Script ===")
    print(f"Args / config: {cfg}")

    # convert Hydra config to  dict
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    # initi WANDB
    wandb_run = wandb.init(
        project="RoadCap-Gen",      # Your Project Name
        name=cfg.experiment_name,   # Name of this specific run (e.g. "qa_test_v1")
        config=config_dict,         # HYDRA CONFIG
        group=cfg.model.name,       # Optional: Group runs by model type
        job_type="finetune",
    )

    # ---------------------------------------------------------
    # 1. load Tokenizer & Processor
    print(f"Loading generic tokenizer and processor from {cfg.model.path}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.path, use_fast=False, trust_remote_code=True)
    
    # Fix Pad Token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else tokenizer.eos_token
    
    image_processor = AutoImageProcessor.from_pretrained(cfg.model.path, trust_remote_code=True)
    if hasattr(image_processor, "image_grid_pinpoints"):
        custom_resolution = (672, 1008) # define DriveLM 2x3 grid resolution: (Height=672, Width=1008)
        image_processor.image_grid_pinpoints = list(image_processor.image_grid_pinpoints) + [custom_resolution]

    # 2. Build Model
    print(f"Building Model: {cfg.model.name}")
    model = build_model(cfg)  

    if hasattr(model.config, "image_grid_pinpoints"):
        custom_resolution = (672, 1008) # define DriveLM 2x3 grid resolution: (Height=672, Width=1008)
        model.config.image_grid_pinpoints = list(model.config.image_grid_pinpoints) + [custom_resolution]
        
    
    print(f"Loading Datasets from {cfg.dataset.data_path}...")
    print(f"Loading Datasets: {cfg.dataset.name}")
    
    print("--- Loading Training Set ---")
    train_dataset = build_dataset(
        cfg=cfg,
        tokenizer=tokenizer,
        image_processor=image_processor,
        split="train",
        data_usage=cfg.training.data_usage
    )

    print("--- Loading Validation Set ---")
    eval_dataset = build_dataset(
        cfg=cfg,
        tokenizer=tokenizer,
        image_processor=image_processor,
        split="val",
        data_usage=cfg.training.data_usage
    )

    # 5. potentialy apply LoRA
    model = apply_peft(model, cfg)

    # 6. Training Arguments & Trainer
    use_bf16 = getattr(cfg.training, "bf16", False)


    training_args = TrainingArguments(
        output_dir=f"output/{cfg.model.name}/{cfg.experiment_name}",
        report_to="wandb",          # log to W&B
        run_name=cfg.experiment_name,
        per_device_train_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.grad_accumulation,
        learning_rate=cfg.training.learning_rate,
        num_train_epochs=cfg.training.epochs,
        logging_steps=cfg.training.logging_steps,
        # logging_steps=50,
        # logging_strategy="epoch", 
        logging_strategy="steps", # or "steps"
        # save_strategy="epoch",
        bf16=use_bf16,
        fp16=not use_bf16,
        dataloader_num_workers=cfg.training.num_workers,
        remove_unused_columns=False, 
        gradient_checkpointing=cfg.training.grad_checkpointing,
        warmup_ratio=cfg.training.warmup_ratio,

        # evaluation / validation
        # evaluation_strategy="epoch", # Run validation at end of every epoch
        per_device_eval_batch_size=cfg.training.batch_size * 2, # usually can be larger for eval because I don't need gradients
        eval_strategy="epoch",
        save_strategy="epoch",

        save_total_limit=2, # keep only last 2 checkpoints, saves space
        # ENABLE "BEST" SAVING and save_total_limit=2 means saving 1 Best + 1 Latest ckpts
        load_best_model_at_end=True,  # Automatically load the best model when training finishes
        # DEFINE "BEST"
        metric_for_best_model="eval_loss", # or "accuracy" if you calculate it
        greater_is_better=False,           # False for loss (lower is better), True for accuracy

        # # Reporting
        # report_to="wandb",            
        # run_name="llava-finetune-v1"

    )
    # collator = RoadCapCollator(tokenizer=tokenizer)

    

    processor = AutoProcessor.from_pretrained(cfg.model.path)
    processor.tokenizer.padding_side = "right" # during training, one always uses padding on the right
    if hasattr(image_processor, "image_grid_pinpoints"):
        processor.image_processor.image_grid_pinpoints = image_processor.image_grid_pinpoints

    collator = RoadCapCollator(processor=processor)
    
    trainer = RoadCapTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        mode="simple"
    )

    print("📊 Running Epoch 0 (Baseline) Evaluation / Validation...")
    # initial_metrics = trainer.evaluate()
    # initial_metrics["epoch"] = 0. # force 'epoch' to be 0.0
    # trainer.log(initial_metrics) # log to W&B (and console)
    # print(f"Epoch 0 Metrics: {initial_metrics}")


    print("Starting Training...")
    trainer.train()

    print("Finished.")


if __name__ == "__main__":
    main()

# salloc -A EU-25-10 -p qgpu_exp --gpus-per-node 1 -t 5:00:00 --nodes 1

# source /mnt/proj1/eu-25-10/envs/roadcap-gen/bin/activate
# python scripts/02_finetuning/train.py     model=llava     dataset=qa_dataset     training=lora              experiment_name=qa_debug
# python scripts/02_finetuning/train.py     model=llava     dataset=drivelm        training=full_finetune     experiment_name=debug_llava_drivelm_fullfinetune

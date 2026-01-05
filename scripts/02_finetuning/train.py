import hydra
import torch
from transformers import TrainingArguments
from src.models.builder import build_model
from src.training.peft_utils import apply_peft
from src.training.trainer import RoadCapTrainer
from src.data.dataset import RoadCapDataset
from src.data.collator import RoadCapCollator

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg):
    
    # 1. Prepare Data
    print(f"Loading Dataset from {cfg.dataset.data_path}")
    dataset = RoadCapDataset(
        data_path=cfg.dataset.data_path,
        image_folder=cfg.dataset.image_folder,
        tokenizer_path=cfg.model.tokenizer_path,
        task=cfg.dataset.task
    )
    
    # 2. Build Model
    model = build_model(cfg)
    
    # 3. Apply LoRA
    model = apply_peft(model, cfg)
    
    # 4. Setup Training Arguments
    training_args = TrainingArguments(
        output_dir=f"output/{cfg.model.name}/{cfg.experiment_name}",
        per_device_train_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.grad_accumulation,
        learning_rate=cfg.training.learning_rate,
        num_train_epochs=cfg.training.epochs,
        warmup_ratio=cfg.training.warmup_ratio,
        logging_steps=cfg.training.logging_steps,
        save_strategy="epoch",
        bf16=True,                  # Use BF16 for Ampere GPUs (A100/3090), else fp16
        dataloader_num_workers=4,
        remove_unused_columns=False # CRITICAL: Trainer tries to delete 'image' column otherwise
    )
    
    # 5. Initialize Collator
    # Note: dataset.processor was init inside dataset
    collator = RoadCapCollator(dataset.processor)
    
    # 6. Initialize Trainer
    trainer = RoadCapTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        mode="simple"
    )
    
    # 7. Train
    print("Starting Training...")
    trainer.train()

if __name__ == "__main__":
    main()
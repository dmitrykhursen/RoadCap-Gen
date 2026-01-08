import hydra
import torch
from transformers import TrainingArguments
from src.models.builder import build_model
from src.training.peft_utils import apply_peft
from src.training.trainer import RoadCapTrainer
from src.data.dataset import RoadCapDataset
from src.data.collator import RoadCapCollator
from transformers import AutoTokenizer, AutoImageProcessor

# def test_single_batch(dataset):
#     from torch.utils.data import DataLoader

#     print(f"\n🔍 DEBUGGING DATASET...")
#     print(f"Dataset type: {type(dataset.flat_samples)}")
    
#     # 1. Test direct access (The exact line that crashed before)
#     try:
#         print("Attempting dataset[0]...")
#         item = dataset[0]
#         print("✅ dataset[0] success!")
#         # Optional: Print keys to verify structure
#         # print(f"Sample keys: {item.keys()}") 
#     except KeyError as e:
#         print(f"❌ CRASH: dataset[0] failed with KeyError: {e}")
#         print(f"Reason: self.data is a Dictionary, but we tried to access index 0.")
#         return

#     except Exception as e:
#         print(f"❌ CRASH: {e}")
#         return
    
#     # Initialize it with the tokenizer from the dataset
#     collator = RoadCapCollator(tokenizer=dataset.tokenizer)
    

#     # 2. Test DataLoader (what the Trainer does)
#     print("\nAttempting DataLoader next(iter())...")
#     loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collator)

    
#     try:
#         batch = next(iter(loader))
#         print("✅ DataLoader success!")
#         print(f"Batch keys: {batch.keys()}")
#         if 'input_ids' in batch:
#             print(f"Input shape: {batch['input_ids'].shape}")
#     except Exception as e:
#         print(f"❌ DataLoader failed: {e}")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg):
    # ---------------------------------------------------------
    # 1. LOAD TOKENIZER FIRST
    # ---------------------------------------------------------
    print(f"Loading generic tokenizer and processor from {cfg.model.path}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.path, use_fast=False, trust_remote_code=True)
    
    # Fix Pad Token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else tokenizer.eos_token

    # ---------------------------------------------------------
    # 2. BUILD MODEL (DO THIS ONLY ONCE)
    # ---------------------------------------------------------
    print(f"Building Model: {cfg.model.name}")
    model = build_model(cfg)  
    # #####
    # # model.config.image_grid_pinpoints = None 
    # model.config.image_grid_pinpoints = [[336, 336]]
    # model.config.vision_feature_select_strategy = "default"
    # model.config.image_aspect_ratio = "square"
    # if hasattr(model.config.vision_config, "patch_size"):
    #     model.config.vision_config.patch_size = 14
    # #####

    # ---------------------------------------------------------
    # 3. FIX TOKENIZER & MODEL CONFIG (SYNC THEM)
    # ---------------------------------------------------------
    image_token = "<image>"
    token_ids = tokenizer.encode(image_token, add_special_tokens=False)
    
    # A. Ensure <image> is a single token
    if len(token_ids) > 1 or image_token not in tokenizer.get_vocab():
        print(f"⚠️ Tokenizer is splitting '{image_token}'! Fixing it now...")
        tokenizer.add_tokens([image_token], special_tokens=True)
        
        # B. RESIZE MODEL EMBEDDINGS (Critical step applied to the loaded model)
        model.resize_token_embeddings(len(tokenizer))
        print(f"✅ Added <image> token and resized model embeddings.")
    else:
        print(f"✅ Tokenizer correctly recognizes <image>")

    # C. SYNC CONFIG (Critical step applied to the loaded model)
    model.config.image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    model.config.pad_token_id = tokenizer.pad_token_id
    print(f"⚙️ Synced Config: image_token_index={model.config.image_token_index}")

    # ---------------------------------------------------------
    # 4. LOAD DATASET
    # ---------------------------------------------------------
    # image_processor = AutoImageProcessor.from_pretrained(
    #             cfg.model.path,
    #             do_resize=True,
    #             size={"shortest_edge": 336},
    #             # size={"shortest_edge": 224},
    #             do_center_crop=False,
    #             do_rescale=True,
    #             # crop_size={"height": 336, "width": 336},
    #             rescale_factor=1 / 255,
    #             do_normalize=True,
    #             image_mean=[0.48145466, 0.4578275, 0.40821073], # this is based on imagenet !!! so compute based on our training data
    #             image_std=[0.26862954, 0.26130258, 0.27577711], # this is based on imagenet !!! so compute based on our training data
    #             # image_grid_pinpoints=[[336, 1008], [1008, 336]],
    #             do_pad=True,
    #             do_convert_rgb=True
    #         )
        
    image_processor = AutoImageProcessor.from_pretrained(cfg.model.path, trust_remote_code=True)
    print("image_processor:", image_processor)
    # exit(0)
    
    print(f"Loading Dataset from {cfg.dataset.data_path}")
    dataset = RoadCapDataset(
        data_path=cfg.dataset.data_path,
        image_folder=cfg.dataset.image_folder,
        tokenizer=tokenizer,
        image_processor=image_processor,
        model_style="llava"
    )

    # 5. Apply LoRA
    model = apply_peft(model, cfg)

    # 6. Training Arguments & Trainer
    use_bf16 = getattr(cfg.training, "bf16", False)
    
    training_args = TrainingArguments(
        output_dir=f"output/{cfg.model.name}/{cfg.experiment_name}",
        per_device_train_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.grad_accumulation,
        learning_rate=cfg.training.learning_rate,
        num_train_epochs=cfg.training.epochs,
        logging_steps=cfg.training.logging_steps,
        save_strategy="epoch",
        bf16=use_bf16,
        fp16=not use_bf16,
        dataloader_num_workers=1,
        remove_unused_columns=False
    )
    
    collator = RoadCapCollator(tokenizer=tokenizer)
    
    trainer = RoadCapTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        mode="simple"
    )
    
    print("Starting Training...")
    trainer.train()


    print("Finished.")


if __name__ == "__main__":
    main()

# salloc -A EU-25-10 -p qgpu --gpus-per-node 1 -t 5:00:00 --nodes 1

# source /mnt/proj1/eu-25-10/envs/roadcap-gen/bin/activate
# python scripts/02_finetuning/train.py     model=llava     dataset=qa_dataset     training=lora     experiment_name=first_qa_test
import os
import json
import torch
import hydra
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoProcessor
from peft import PeftModel
import time
from omegaconf import OmegaConf

# Import your project modules
from src.models.builder import build_model
from src.data.dataset_builder import build_dataset

def setup_ddp():
    """Initialize Distributed Data Parallel"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def inference_collator(batch, processor):
    """
    Custom collator to return exactly what you asked for:
    images (tensor), prompts (tensor), ids (list), questions (list), gt_answers (list)
    """
    # 1. Collect Metadata Lists
    # DriveLMDataset returns: {'id': ..., 'question': ..., 'answer': ..., 'image': ...}
    ids = [item['id'] for item in batch]
    questions = [item['question'] for item in batch] # Already cleaned (no <image> tag)
    gt_answers = [item['answer'] for item in batch]
    
    # 2. Collect Images & Text for Processor
    raw_images = [item['image'] for item in batch]
    
    # Construct prompts with the template (similar to your training collator logic)
    text_prompts = []
    for q in questions:
        # LLaVA formatting
        conversation = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]},
        ]
        text_prompts.append(processor.apply_chat_template(conversation))

    # 3. Process Tensors
    inputs = processor(
        text=text_prompts,
        images=raw_images,
        padding=True,
        return_tensors="pt"
    )

    # Return tuple as requested: (images_tensor, input_ids, ids, questions, gt_answers)
    # Note: inputs['pixel_values'] is the image tensor
    #       inputs['input_ids'] is the tokenized prompt
    return inputs, ids, questions, gt_answers


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg):
    # 1. DDP Setup
    local_rank = setup_ddp()
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    
    if global_rank == 0:
        print(f"🚀 Starting DDP Inference on {world_size} GPUs")
        start_time = time.time()
        print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    # 2. Load Processor
    # Use tokenizer_path if using adapter, else model.path
    load_path = cfg.model.tokenizer_path if cfg.inference.load_mode == "adapter" else cfg.model.path
    processor = AutoProcessor.from_pretrained(load_path, trust_remote_code=True)
    
    # Handle DriveLM 2x3 Grid (LLaVA 1.6 specific)
    if hasattr(processor.image_processor, "image_grid_pinpoints"):
        res = (672, 1008) # 2x3 grid
        if res not in list(processor.image_processor.image_grid_pinpoints):
            processor.image_processor.image_grid_pinpoints = list(processor.image_processor.image_grid_pinpoints) + [res]

    # 3. Load Model
    # (Simplified loading logic based on your request)
    model = build_model(cfg)
    
    if cfg.inference.load_mode == "adapter":
        model = PeftModel.from_pretrained(model, cfg.inference.checkpoint_path)
        model = model.merge_and_unload()
    elif cfg.inference.load_mode == "full" and cfg.inference.checkpoint_path:
        # Assuming build_model loads weights, or we load state_dict here
        pass 

    model.to(local_rank)
    # DDP Wrap (Not strictly necessary for inference, but good practice if using buffers)
    # For pure inference, we can just use the model on device, but DDP helps sync buffers if any.
    # model = DDP(model, device_ids=[local_rank]) 

    # 4. Dataset & Sampler
    # dataset = build_dataset(cfg, processor.tokenizer, processor.image_processor, split="all_data", data_usage=1.0)
    dataset = build_dataset(cfg, processor.tokenizer, processor.image_processor, split="val", data_usage=0.03)

    
    # DistributedSampler automatically splits data among GPUs
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=False, drop_last=False)
    
    # Use lambda to pass processor to our custom collator
    collate_fn = lambda b: inference_collator(b, processor)

    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.inference.batch_size, 
        sampler=sampler, 
        num_workers=cfg.inference.num_workers, 
        collate_fn=collate_fn,
        pin_memory=True
    )

    # 5. Inference Loop
    results = []
    
    # Tqdm only on main process
    iterator = tqdm(dataloader, disable=(global_rank != 0), desc="Inferencing")
    
    for batch in iterator:
        # UNPACKING exactly as you wanted
        inputs, ids, questions, gt_answers = batch
        
        # Move tensors to GPU
        input_ids = inputs["input_ids"].to(local_rank)
        pixel_values = inputs["pixel_values"].to(local_rank)
        attention_mask = inputs["attention_mask"].to(local_rank)
        image_sizes = inputs.get("image_sizes", None) # Important for LLaVA 1.6

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                image_sizes=image_sizes,
                pad_token_id=processor.tokenizer.pad_token_id,
                **cfg.inference.generation
            )

        # 1. Get the Clean Answer (Sliced)
        new_tokens = output_ids[:, input_ids.shape[1]:]
        generated_answers = processor.batch_decode(new_tokens, skip_special_tokens=True)

        # 2. Get the Full Debug Output (Original)
        full_texts = processor.batch_decode(output_ids, skip_special_tokens=True)

        for i in range(len(generated_answers)):
            # Clean answer
            pred = generated_answers[i].strip()
            if "[/INST]" in pred:
                pred = pred.split("[/INST]")[-1].strip()
            
            # Full text (raw model output including prompt)
            raw_full = full_texts[i]

            results.append({
                "id": ids[i],
                "question": questions[i],
                "answer": pred,         # clean answer for metrics
                # "raw_response": raw_full   # Full output for debugging
            })

    # 6. Save Partial Results (Each GPU saves its own file)
    os.makedirs(cfg.inference.output_dir, exist_ok=True)
    partial_path = os.path.join(cfg.inference.output_dir, f"results_rank_{global_rank}.json")
    
    with open(partial_path, "w") as f:
        json.dump(results, f, indent=4)
    
    dist.barrier() # Wait for all

    # 7. Merge (Only Rank 0 does this)
    if global_rank == 0:
        print("🔄 Merging all GPU results...")

        # Structure: output/inference/experiment_name_YYYY-MM-DD_HH-MM-SS/
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(start_time))
        run_folder_name = f"{cfg.experiment_name}_{timestamp}"
        run_dir = os.path.join(cfg.inference.output_dir, run_folder_name)
        
        os.makedirs(run_dir, exist_ok=True)
        print(f"📁 Created Run Folder: {run_dir}")

        final_data = []
        for r in range(world_size):
            p_file = os.path.join(cfg.inference.output_dir, f"results_rank_{r}.json")
            if os.path.exists(p_file):
                with open(p_file, "r") as f:
                    final_data.extend(json.load(f))
                os.remove(p_file) # Clean up
        
        # Format: drivelm_test_2026-01-17_18-30-00.json
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(start_time))
        
        # Split "drivelm_test.json" -> "drivelm_test" + ".json"
        base_name, ext = os.path.splitext(cfg.inference.output_file)
        new_filename = f"{base_name}_{timestamp}{ext}"
        
        # final_path = os.path.join(cfg.inference.output_dir, new_filename)
        final_path = os.path.join(run_dir, new_filename)

        # ------------------------------------------
        with open(final_path, "w") as f:
            json.dump(final_data, f, indent=4)

        # --- D. Save Hydra Config (Logs) ---
        config_path = os.path.join(run_dir, "config.yaml")
        OmegaConf.save(config=cfg, f=config_path)
        
        print(f"✅ Results saved to: {final_path}")
        print(f"✅ Config saved to: {config_path}")
        
        # # Close WandB
        # if wandb.run is not None:
        #     wandb.finish()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

# salloc -A EU-25-10 -p qgpu_exp --gpus-per-node 1 -t 1:00:00 --nodes 1

# source /mnt/proj1/eu-25-10/envs/roadcap-gen/bin/activate
# torchrun --nproc_per_node=1 scripts/04_inference/ddp_inference.py     model=llava     dataset=drivelm      inference=drivelm_infer     experiment_name=debug_infer_llava_drivelm_FF_MMP_ddp

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
from datetime import timedelta

def setup_ddp():
    """Initialize Distributed Data Parallel"""
    dist.init_process_group(
        backend="nccl", 
        timeout=timedelta(hours=1)
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def inference_collator(batch, processor):
    """
    Custom collator to return exactly what you asked for:
    images (tensor), prompts (tensor), ids (list), questions (list), gt_answers (list)
    """
    # 1. Collect Metadata Lists
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

    return inputs, ids, questions, gt_answers


def save_partial_results(partial_path, new_results):
    """Helper function to load, extend, and save partial results"""
    if not new_results:
        return

    if os.path.exists(partial_path):
        with open(partial_path, "r") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
        existing_data.extend(new_results)
    else:
        existing_data = new_results
        
    with open(partial_path, "w") as f:
        json.dump(existing_data, f, indent=4)

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg):
    # 1. DDP Setup
    local_rank = setup_ddp()
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    
    # Check if we should append the timestamp (Defaults to True for backwards compatibility)
    add_time = cfg.inference.get("add_timestamp", True)
    
    # --- FOLDER CREATION & SYNC ACROSS GPUs ---
    timestamp_list = [None]
    if global_rank == 0:
        print(f"🚀 Starting DDP Inference on {world_size} GPUs")
        start_time = time.time()
        print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        
        timestamp_list[0] = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(start_time))
        
        # 🌟 MODIFIED: Only append timestamp if requested
        if add_time:
            run_folder_name = f"{cfg.experiment_name}_{timestamp_list[0]}"
        else:
            run_folder_name = f"{cfg.experiment_name}"
            
        run_dir = os.path.join(cfg.inference.output_dir, run_folder_name)
        partial_dir = os.path.join(run_dir, "partial_results")
        
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(partial_dir, exist_ok=True)
        print(f"📁 Created Run Folder: {run_dir}")
        print(f"📁 Created Partial Results Folder: {partial_dir}")
        
        config_path = os.path.join(run_dir, "config.yaml")
        OmegaConf.save(config=cfg, f=config_path)

    dist.broadcast_object_list(timestamp_list, src=0)
    timestamp = timestamp_list[0]
    
    if add_time:
        run_folder_name = f"{cfg.experiment_name}_{timestamp}"
    else:
        run_folder_name = f"{cfg.experiment_name}"
        
    run_dir = os.path.join(cfg.inference.output_dir, run_folder_name)
    partial_dir = os.path.join(run_dir, "partial_results")
    partial_path = os.path.join(partial_dir, f"results_rank_{global_rank}.json")
    
    print(f"GPU {global_rank} will save partial results to: {partial_path}")

    dist.barrier()
    # ------------------------------------------

    # 1. Resolve Paths based on load_mode
    load_mode = cfg.inference.load_mode
    ckpt_model = cfg.inference.get("checkpoint_model")
    ckpt_adapter = cfg.inference.get("checkpoint_adapter")

    if load_mode == "fullfinetune" and ckpt_model:
        load_path = ckpt_model
        is_local = os.path.isdir(load_path)
    elif load_mode == "adapter" and ckpt_model:
        load_path = ckpt_model
        is_local = os.path.isdir(load_path)
    else:
        load_path = cfg.model.path
        is_local = False

    if global_rank == 0:
        print(f"🛠️  Load Mode: {load_mode.upper()}")
        print(f"📂 Source Path: {load_path}")

    # 2. Load Processor
    tokenizer_source = cfg.inference.get("tokenizer_path") or load_path
    processor = AutoProcessor.from_pretrained(
        tokenizer_source, 
        trust_remote_code=True,
        local_files_only=os.path.isdir(tokenizer_source) if tokenizer_source else False,
        use_fast=True  # forces the fast processor (rust implementation instead of python based)
    )
    # Force left-padding strictly for generation
    processor.tokenizer.padding_side = "left"
    
    if hasattr(processor.image_processor, "image_grid_pinpoints"):
        res = (672, 1008) # 2x3 grid
        if res not in list(processor.image_processor.image_grid_pinpoints):
            processor.image_processor.image_grid_pinpoints = list(processor.image_processor.image_grid_pinpoints) + [res]

    # 3. Load/Build Model
    cfg.model.path = load_path
    model = build_model(cfg)
    
    # 4. Apply Adapters if necessary
    if load_mode == "adapter":
        adapter_path = ckpt_adapter or load_path 
        if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            if global_rank == 0:
                print(f"🔗 Loading LoRA Adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()
        else:
            raise ValueError(f"Adapter mode requested but no adapter_config.json found in {adapter_path}")

    model.to(local_rank)

    # 4. Dataset & Sampler
    dataset = build_dataset(cfg, processor.tokenizer, processor.image_processor, split="all_data", data_usage=1.0) 
    # dataset = build_dataset(cfg, processor.tokenizer, processor.image_processor, split="all_data", data_usage=0.05) 
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=False, drop_last=False)
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
    results_buffer = []
    iterator = tqdm(dataloader, disable=(global_rank != 0), desc="Inferencing")
    
    for step, batch in enumerate(iterator):
        inputs, ids, questions, gt_answers = batch
        
        input_ids = inputs["input_ids"].to(local_rank)
        pixel_values = inputs["pixel_values"].to(local_rank)
        attention_mask = inputs["attention_mask"].to(local_rank)
        image_sizes = inputs.get("image_sizes", None) 

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                image_sizes=image_sizes,
                pad_token_id=processor.tokenizer.pad_token_id,
                **cfg.inference.generation
            )

        new_tokens = output_ids[:, input_ids.shape[1]:]
        generated_answers = processor.batch_decode(new_tokens, skip_special_tokens=True)

        for i in range(len(generated_answers)):
            pred = generated_answers[i].strip()
            if "[/INST]" in pred:
                pred = pred.split("[/INST]")[-1].strip()
            
            results_buffer.append({
                "id": ids[i],
                "question": questions[i],
                "answer": pred,
            })
            
        if (step + 1) % 10 == 0:
            save_partial_results(partial_path, results_buffer)
            results_buffer = [] 

    if len(results_buffer) > 0:
        save_partial_results(partial_path, results_buffer)
        results_buffer = []
    
    try:
        dist.barrier() 
    except Exception as e:
        print(f"Rank {global_rank}: Barrier timed out, but files should be saved. Proceeding...")

    # 7. Merge (Only Rank 0 does this)
    if global_rank == 0:
        print("🔄 Merging all GPU results...")

        final_data = []
        for r in range(world_size):
            p_file = os.path.join(partial_dir, f"results_rank_{r}.json")
            if os.path.exists(p_file):
                with open(p_file, "r") as f:
                    final_data.extend(json.load(f))
        
        base_name, ext = os.path.splitext(cfg.inference.output_file)
        
        # 🌟 MODIFIED: Only append timestamp to final JSON file if requested
        if add_time:
            new_filename = f"{base_name}_{timestamp}{ext}"
        else:
            new_filename = f"{base_name}{ext}"
            
        final_path = os.path.join(run_dir, new_filename)

        with open(final_path, "w") as f:
            json.dump(final_data, f, indent=4)

        print(f"✅ Results saved to: {final_path}")
        print(f"✅ Config saved to: {config_path}")
        print(f"ℹ️  Partial results kept intact in: {partial_dir}")

    dist.destroy_process_group()
    
    
if __name__ == "__main__":
    main()

# salloc -A EU-25-10 -p qgpu_exp --gpus-per-node 1 -t 1:00:00 --nodes 1

# source /mnt/proj1/eu-25-10/envs/roadcap-gen/bin/activate
# torchrun --nproc_per_node=1 scripts/04_inference/ddp_inference.py     model=llava     dataset=drivelm      inference=drivelm_infer     experiment_name=tmp_debug_infer_llava_drivelm_local_val

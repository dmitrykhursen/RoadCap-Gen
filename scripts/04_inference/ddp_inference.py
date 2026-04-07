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

from PIL import Image
import torchvision.transforms as transforms
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from external.DriveLM.challenge.llama_adapter_v2_multimodal7b import llama



from src.models.builder import build_model
from src.data.dataset_builder import build_dataset
from datetime import timedelta

def setup_ddp():
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=1))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

# --- LLaVA Collator ---
def inference_collator(batch, processor):
    ids = [item['id'] for item in batch]
    questions = [item['question'] for item in batch]
    gt_answers = [item['answer'] for item in batch]
    raw_images = [item['image'] for item in batch]
    
    text_prompts = []
    for q in questions:
        conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        text_prompts.append(processor.apply_chat_template(conversation))

    inputs = processor(text=text_prompts, images=raw_images, padding=True, return_tensors="pt")
    return inputs, ids, questions, gt_answers

# --- LLaMA-Adapter Collator ---
def llama_adapter_collator(batch, transform):
    ids = [item['id'] for item in batch]
    questions = [item['question'] for item in batch]
    gt_answers = [item['answer'] for item in batch]

    prompts = [llama.format_prompt(q) for q in questions]

    images_batch = []
    for item in batch:
        img_data = item['image']
        
        # Helper function to handle both strings and PIL images safely
        def load_and_convert(img_input):
            if isinstance(img_input, str):
                return Image.open(img_input).convert('RGB')
            else:
                # If it's not a string, assume it's already a PIL Image
                return img_input.convert('RGB')

        if isinstance(img_data, list):
            image_all = []
            for img_item in img_data:
                img = load_and_convert(img_item)
                image_all.append(transform(img))
            images_batch.append(torch.stack(image_all, dim=0))
        else:
            img = load_and_convert(img_data)
            images_batch.append(transform(img).unsqueeze(0)) 

    images_tensor = torch.stack(images_batch, dim=0)
    return images_tensor, prompts, ids, questions, gt_answers

def save_partial_results(partial_path, new_results):
    if not new_results: return
    if os.path.exists(partial_path):
        with open(partial_path, "r") as f:
            try: existing_data = json.load(f)
            except json.JSONDecodeError: existing_data = []
        existing_data.extend(new_results)
    else:
        existing_data = new_results
    with open(partial_path, "w") as f:
        json.dump(existing_data, f, indent=4)

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg):
    local_rank = setup_ddp()
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    
    add_time = cfg.inference.get("add_timestamp", True)
    
    # --- FOLDER CREATION ---
    timestamp_list = [None]
    if global_rank == 0:
        print(f"🚀 Starting DDP Inference on {world_size} GPUs")
        start_time = time.time()
        timestamp_list[0] = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(start_time))
        run_folder_name = f"{cfg.experiment_name}_{timestamp_list[0]}" if add_time else cfg.experiment_name
        run_dir = os.path.join(cfg.inference.output_dir, run_folder_name)
        partial_dir = os.path.join(run_dir, "partial_results")
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(partial_dir, exist_ok=True)
        OmegaConf.save(config=cfg, f=os.path.join(run_dir, "config.yaml"))

    dist.broadcast_object_list(timestamp_list, src=0)
    run_folder_name = f"{cfg.experiment_name}_{timestamp_list[0]}" if add_time else cfg.experiment_name
    run_dir = os.path.join(cfg.inference.output_dir, run_folder_name)
    partial_dir = os.path.join(run_dir, "partial_results")
    partial_path = os.path.join(partial_dir, f"results_rank_{global_rank}.json")
    dist.barrier()

    # --- PROCESSOR & COLLATOR SETUP ---
    if cfg.model.type == "llama_adapter_v2":
        processor = transforms.Compose([
            transforms.Resize((224, 224), interpolation=BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        collate_fn = lambda b: llama_adapter_collator(b, processor)
        
    elif cfg.model.type == "llava":
        tokenizer_source = cfg.inference.get("tokenizer_path") or cfg.model.path
        processor = AutoProcessor.from_pretrained(
            tokenizer_source, trust_remote_code=True, use_fast=True
        )
        processor.tokenizer.padding_side = "left"
        if hasattr(processor.image_processor, "image_grid_pinpoints"):
            res = (672, 1008)
            if res not in list(processor.image_processor.image_grid_pinpoints):
                processor.image_processor.image_grid_pinpoints = list(processor.image_processor.image_grid_pinpoints) + [res]
        collate_fn = lambda b: inference_collator(b, processor)
        
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    # --- MODEL BUILD ---
    model = build_model(cfg)
    model.to(local_rank)
    
    # Model-specific setup after moving to GPU
    if cfg.model.type == "llava":
        if cfg.inference.load_mode == "adapter":
            adapter_path = cfg.inference.get("checkpoint_adapter") or cfg.model.path 
            if global_rank == 0: print(f"🔗 Loading LoRA Adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
        model.eval()
    elif cfg.model.type == "llama_adapter_v2":
        model.eval() 

    # --- DATASET & DATALOADER ---
    if cfg.model.type == "llava":
        dataset = build_dataset(cfg, processor.tokenizer, processor.image_processor, split="all_data", data_usage=1.0)
    elif cfg.model.type == "llama_adapter_v2":
        dataset = build_dataset(cfg, None, None, split="all_data", data_usage=1.0)
        
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(
        dataset, batch_size=cfg.inference.batch_size, sampler=sampler, 
        num_workers=cfg.inference.num_workers, collate_fn=collate_fn, pin_memory=True
    )

    # --- INFERENCE LOOP ---
    results_buffer = []
    iterator = tqdm(dataloader, disable=(global_rank != 0), desc="Inferencing")
    
    for step, batch in enumerate(iterator):
        # LLaMA-Adapter Inference
        if cfg.model.type == "llama_adapter_v2":
            images, prompts, ids, questions, gt_answers = batch
            images = images.to(local_rank)
            
            with torch.no_grad():
                generated_answers = model.generate(
                    imgs=images,
                    prompts=prompts,
                    max_gen_len=cfg.inference.generation.get("max_new_tokens", 256),
                    temperature=cfg.inference.generation.get("temperature", 0.2),
                    top_p=cfg.inference.generation.get("top_p", 0.1)
                )
                
        # LLaVA1.6 Inference
        elif cfg.model.type == "llava":
            inputs, ids, questions, gt_answers = batch
            input_ids = inputs["input_ids"].to(local_rank)
            pixel_values = inputs["pixel_values"].to(local_rank)
            attention_mask = inputs["attention_mask"].to(local_rank)
            image_sizes = inputs.get("image_sizes", None) 

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids, pixel_values=pixel_values, attention_mask=attention_mask,
                    image_sizes=image_sizes, pad_token_id=processor.tokenizer.pad_token_id,
                    **cfg.inference.generation
                )
            new_tokens = output_ids[:, input_ids.shape[1]:]
            generated_answers = processor.batch_decode(new_tokens, skip_special_tokens=True)

        # Output saving
        for i in range(len(generated_answers)):
            pred = generated_answers[i].strip()
            if "[/INST]" in pred:
                pred = pred.split("[/INST]")[-1].strip()
            
            results_buffer.append({"id": ids[i], "question": questions[i], "answer": pred})
            
        if (step + 1) % 10 == 0:
            save_partial_results(partial_path, results_buffer)
            results_buffer = [] 

    if len(results_buffer) > 0:
        save_partial_results(partial_path, results_buffer)
        
    try: dist.barrier() 
    except Exception: pass

    # --- MERGE RESULTS (Rank 0) ---
    if global_rank == 0:
        print("🔄 Merging all GPU results...")
        final_data = []
        for r in range(world_size):
            p_file = os.path.join(partial_dir, f"results_rank_{r}.json")
            if os.path.exists(p_file):
                with open(p_file, "r") as f: final_data.extend(json.load(f))
        
        base_name, ext = os.path.splitext(cfg.inference.output_file)
        new_filename = f"{base_name}_{timestamp_list[0]}{ext}" if add_time else f"{base_name}{ext}"
        final_path = os.path.join(run_dir, new_filename)

        with open(final_path, "w") as f:
            json.dump(final_data, f, indent=4)

        print(f"✅ Results saved to: {final_path}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

# salloc -A EU-25-10 -p qgpu_exp --gpus-per-node 1 -t 1:00:00 --nodes 1
# salloc -A OPEN-36-7 -p qgpu --gpus-per-node 2 -t 6:00:00 --nodes 1

# source /mnt/proj1/eu-25-10/envs/roadcap-gen/bin/activate
# torchrun --nproc_per_node=1 scripts/04_inference/ddp_inference.py     model=llava     dataset=drivelm      inference=drivelm_infer     experiment_name=tmp_debug_infer_llava_drivelm_local_val

# source /mnt/proj1/eu-25-10/envs/roadcap-gen/bin/activate
# cd /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/
# export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
# export MASTER_ADDR=$(hostname)

# PYTHONPATH=. torchrun --nproc_per_node=2 --master_port=$MASTER_PORT scripts/04_inference/ddp_inference.py model=llama_adapter_v2 dataset=drivelm inference=llama_adapter_v2_infer experiment_name=infer_llama_adapter_v2_tmp_gpu4_bs8_nw4_e5_pretrained inference.load_mode=adapter dataset.data_path=/mnt/proj1/eu-25-10/dmytro/RoadCap-Gen/external/DriveLM/challenge/data/for_testing_purposes_vqas.json
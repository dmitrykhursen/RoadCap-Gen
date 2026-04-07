import json
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from transformers import Trainer
from external.DriveLM.challenge.evaluation import evaluation_suit
import torch.distributed as dist

def gather_python_objects(local_list):
    if not dist.is_initialized():
        return local_list
    # Create a list of Nones to hold the incoming data from all GPUs
    gathered_lists = [None for _ in range(dist.get_world_size())]
    # Gather objects from all ranks into gathered_lists
    dist.all_gather_object(gathered_lists, local_list)
    # Flatten the list of lists into a single 1D list
    return [item for sublist in gathered_lists for item in sublist]


class RoadCapTrainer(Trainer):
    def __init__(self, mode="simple", geo_weight=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.geo_weight = geo_weight
        self.custom_loss_tracker = {"lm_loss": 0.0, "geo_loss": 0.0, "steps": 0}
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 1. Pop ALL custom auxiliary keys to prevent C++ deadlocks
        depth_latents = inputs.pop("depth_latents", None)
        depth_latents_mask = inputs.pop("depth_latents_mask", None)
        
        _ = inputs.pop("img_paths", None)
        _ = inputs.pop("questions", None)
        _ = inputs.pop("tags", None)
        _ = inputs.pop("ids", None)

        # Unwrap DDP → PEFT
        unwrapped = model
        if hasattr(unwrapped, "module"):        
            unwrapped = unwrapped.module
        if hasattr(unwrapped, "base_model"):    
            unwrapped = unwrapped.base_model
        if hasattr(unwrapped, "model"):         
            unwrapped = unwrapped.model

        # 2. Extended Mode Logic
        if self.mode == "extended" and depth_latents is not None:
            inputs["output_hidden_states"] = True
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states[-1]
            outputs.hidden_states = None 

            lm_loss = outputs.loss # 🌟 Grab the base LLM loss

            depth_pred = unwrapped.project_to_depth_latents(hidden_states, inputs["attention_mask"]) 
            depth_target = depth_latents.to(device=depth_pred.device, dtype=depth_pred.dtype)

            if depth_latents_mask is not None and not depth_latents_mask.all():
                valid = depth_latents_mask.to(depth_pred.device)
                if valid.sum() > 0:
                    geo_loss = F.smooth_l1_loss(depth_pred[valid], depth_target[valid])
                else:
                    geo_loss = 0.0 * depth_pred.sum() 
            else:
                geo_loss = F.smooth_l1_loss(depth_pred, depth_target)

            total_loss = lm_loss + self.geo_weight * geo_loss

            # 🌟 2. Add to tracker (Only on Rank 0 to prevent DDP lag)
            if self.args.process_index == 0:
                self.custom_loss_tracker["lm_loss"] += lm_loss.detach().item()
                
                # Handle cases where geo_loss is a pure float 0.0 or a Tensor
                if isinstance(geo_loss, torch.Tensor):
                    self.custom_loss_tracker["geo_loss"] += geo_loss.detach().item()
                else:
                    self.custom_loss_tracker["geo_loss"] += geo_loss
                    
                self.custom_loss_tracker["steps"] += 1

        else:
            outputs = model(**inputs)
            total_loss = outputs.loss

        return (total_loss, outputs) if return_outputs else total_loss
    
    def log(self, logs: dict, *args, **kwargs):
        """
        Intercept the standard HF log function. 
        """
        if getattr(self.args, "process_index", 0) == 0 and self.custom_loss_tracker["steps"] > 0:
            steps = self.custom_loss_tracker["steps"]
            
            # Calculate the average since the last time we logged
            avg_lm_loss = self.custom_loss_tracker["lm_loss"] / steps
            avg_geo_loss = self.custom_loss_tracker["geo_loss"] / steps
            
            # Inject our custom metrics into the HF logs dictionary
            logs["train_lm_loss"] = avg_lm_loss
            logs["train_geo_loss"] = avg_geo_loss
            logs["train_geo_loss_weighted"] = avg_geo_loss * self.geo_weight

            # Reset the tracker for the next logging window
            self.custom_loss_tracker = {"lm_loss": 0.0, "geo_loss": 0.0, "steps": 0}

        super().log(logs, *args, **kwargs)
    
    # def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    #     # Pop ALL custom auxiliary keys so they don't break PyTorch!
    #     #print(f"[DIAG] compute_loss called with mode={self.mode}")
    #     depth_latents = inputs.pop("depth_latents", None)
    #     depth_latents_mask = inputs.pop("depth_latents_mask", None)

    #     _ = inputs.pop("img_paths", None)
    #     _ = inputs.pop("questions", None)
    #     _ = inputs.pop("tags", None)
    #     _ = inputs.pop("ids", None)

    #     #print(f"[DIAG] inputs keys after pop: {list(inputs.keys())}")

    #     # Unwrap DDP → PEFT → LoraModel → RoadCapLLaVA.
    #     # Use sequential `if` (NOT `while`): in some PEFT versions LoraModel.base_model
    #     # is a property returning `self`, so `while` loops forever.
    #     unwrapped = model
    #     if hasattr(unwrapped, "module"):        # DDP wrapper
    #         unwrapped = unwrapped.module
    #     if hasattr(unwrapped, "base_model"):    # PeftModel → LoraModel
    #         unwrapped = unwrapped.base_model
    #     if hasattr(unwrapped, "model"):         # LoraModel → RoadCapLLaVA
    #         unwrapped = unwrapped.model

    #     #print(f"[DIAG] unwrapped type: {type(unwrapped).__name__}")

    #     rank = self.args.process_index

    #     # # 1. Extended Mode: use last hidden state captured by the permanent norm hook
    #     # # registered in RoadCapLLaVA.from_pretrained (no per-forward hook needed).
    #     # if self.mode == "extended" and depth_latents is not None:
    #     #     #print(f"[DIAG rank={rank}] >>> extended forward", flush=True)
    #     #     outputs = model(**inputs)
    #     #     #print(f"[DIAG rank={rank}] <<< forward done, computing geo loss", flush=True)

    #     #     total_loss = outputs.loss

    #     #     hidden_states = unwrapped._last_hidden_state  # [B, seq_len, 4096]
    #     #     depth_pred = unwrapped.project_to_depth_latents(
    #     #         hidden_states, inputs["attention_mask"]
    #     #     )  # [B, 32, 35, 63]

    #     #     depth_target = depth_latents.to(device=depth_pred.device, dtype=depth_pred.dtype)
    #     #     if depth_latents_mask is not None and not depth_latents_mask.all():
    #     #         valid = depth_latents_mask.to(depth_pred.device)
    #     #         geo_loss = F.smooth_l1_loss(depth_pred[valid], depth_target[valid])
    #     #     else:
    #     #         geo_loss = F.smooth_l1_loss(depth_pred, depth_target)

    #     #     total_loss = total_loss + self.geo_weight * geo_loss
    #     #     #print(f"[DIAG rank={rank}] loss={total_loss.item():.4f} lm={outputs.loss.item():.4f} geo={geo_loss.item():.4f}")
    #     # print(f"[DIAG] depth_latents: {depth_latents}   depth_latents_mask: {depth_latents_mask.shape if depth_latents_mask is not None else None}")
    #     if self.mode == "extended": # and depth_latents is not None:
            
    #         # 🌟 Tell the model to return hidden states natively
    #         inputs["output_hidden_states"] = True
            
    #         outputs = model(**inputs)
    #         total_loss = outputs.loss

    #         # 🌟 Extract ONLY the final layer's hidden states [B, seq_len, 4096]
    #         hidden_states = outputs.hidden_states[-1]
            
    #         # 🌟 CRITICAL MEMORY TRICK: Instantly delete the massive tuple 
    #         # to free ~8GB of VRAM before the backward pass starts!
    #         outputs.hidden_states = None 

    #         # Project to depth
    #         depth_pred = unwrapped.project_to_depth_latents(
    #             hidden_states, inputs["attention_mask"]
    #         )  

    #         depth_target = depth_latents.to(device=depth_pred.device, dtype=depth_pred.dtype)

    #         if depth_latents_mask is not None and not depth_latents_mask.all():
    #             valid = depth_latents_mask.to(depth_pred.device)
    #             geo_loss = F.smooth_l1_loss(depth_pred[valid], depth_target[valid])
    #         else:
    #             geo_loss = F.smooth_l1_loss(depth_pred, depth_target)

    #         total_loss = total_loss + self.geo_weight * geo_loss

    #     # 2. Simple Mode / no depth latents: standard forward pass only
    #     else:
    #         #print(f"[DIAG rank={rank}] >>> simple forward", flush=True)
    #         outputs = model(**inputs)
    #         total_loss = outputs.loss
    #         #print(f"[DIAG rank={rank}] <<< simple forward done, loss={total_loss.item():.4f}", flush=True)

    #     return (total_loss, outputs) if return_outputs else total_loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Overriding evaluate to run standard loss-based eval + custom DriveLM metrics.
        """
        # 1. Run standard HF evaluation (calculates eval_loss)
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        if self.args.process_index == 0:
            print(f"\n--- Epoch {self.state.epoch}: Starting Custom DriveLM Generation (Distributed) ---")
            
        # 2. RUN ON ALL RANKS: Distributed Generation
        # Every GPU evaluates its own shard of the dataset
        custom_results = self.run_drive_lm_metrics(metric_key_prefix)
        
        # 3. LOG ON RANK 0 ONLY
        if self.args.process_index == 0:
            metrics.update(custom_results)
            self.log(custom_results)
            
        # 4. BARRIER: Wait for Rank 0 to finish calculating scores and saving JSON
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
                
        return metrics

    def run_drive_lm_metrics(self, prefix):
        self.model.eval()
        eval_dataloader = self.get_eval_dataloader() # Automatically sharded by DDP!
        
        # --- LOCAL Trackers (Data processed by THIS specific GPU) ---
        local_preds = []
        local_gts = []
        local_tags = []
        local_ids = []
        local_img_paths = []
        local_questions = []

        # All GPUs run this loop in parallel
        for batch in tqdm(eval_dataloader, desc=f"VQA Eval Rank {self.args.process_index}"):
            with torch.no_grad():
                
                # Build generation arguments dynamically
                gen_kwargs = {
                    "input_ids": batch["input_ids"].to(self.args.device),
                    "max_new_tokens": 300,
                    "pad_token_id": self.processing_class.tokenizer.eos_token_id
                }
                
                if "attention_mask" in batch:
                    gen_kwargs["attention_mask"] = batch["attention_mask"].to(self.args.device)
                
                if "pixel_values" in batch and batch["pixel_values"] is not None:
                    gen_kwargs["pixel_values"] = batch["pixel_values"].to(self.args.device)
                    
                if "image_sizes" in batch and batch["image_sizes"] is not None:
                    gen_kwargs["image_sizes"] = batch["image_sizes"].to(self.args.device)

                # Generate
                generated_ids = self.model.generate(**gen_kwargs)
                
                input_length = batch["input_ids"].shape[1]
                new_tokens = generated_ids[:, input_length:]
                
                preds_a = self.processing_class.batch_decode(new_tokens, skip_special_tokens=True)
                
                labels = batch["labels"].clone()
                labels[labels == -100] = self.processing_class.tokenizer.pad_token_id
                gt_a = self.processing_class.batch_decode(labels, skip_special_tokens=True)
                
                gt_a = [gt.split("[/INST]")[-1].strip() for gt in gt_a]
                preds_a = [pred.strip() for pred in preds_a]

                # Store into local lists
                local_preds.extend(preds_a)
                local_gts.extend(gt_a)
                local_tags.extend(batch["tags"])
                local_ids.extend(batch["ids"])
                local_img_paths.extend(batch["img_paths"])
                local_questions.extend(batch["questions"])

        self.model.train() # Switch back to train mode

        # --- GATHER DATA FROM ALL 8 GPUs TO RANK 0 ---
        gathered_preds = gather_python_objects(local_preds)
        gathered_gts = gather_python_objects(local_gts)
        gathered_tags = gather_python_objects(local_tags)
        gathered_ids = gather_python_objects(local_ids)
        gathered_img_paths = gather_python_objects(local_img_paths)
        gathered_questions = gather_python_objects(local_questions)

        # --- RANK 0: Calculate Metrics & Save ---
        if self.args.process_index == 0:
            suit = evaluation_suit()
            detailed_results = []
            pred_token_lengths = []
            length_differences = []
            
            # Deduplicate items (DDP sometimes pads the dataset so it divides evenly by 8)
            unique_results = {}
            for i in range(len(gathered_ids)):
                uid = gathered_ids[i]
                if uid not in unique_results:
                    unique_results[uid] = {
                        "pred": gathered_preds[i],
                        "gt": gathered_gts[i],
                        "tag": gathered_tags[i],
                        "img_path": gathered_img_paths[i],
                        "question": gathered_questions[i]
                    }
                    
            #print(f"Total Unique Validated Samples: {len(unique_results)}")

            # Process all unique gathered results
            for uid, res in unique_results.items():
                suit.forward(res["tag"], res["pred"], res["gt"])
                
                pred_len = len(self.processing_class.tokenizer.encode(res["pred"], add_special_tokens=False))
                gt_len = len(self.processing_class.tokenizer.encode(res["gt"], add_special_tokens=False))
                len_diff = pred_len - gt_len
                
                pred_token_lengths.append(pred_len)
                length_differences.append(len_diff)
                
                detailed_results.append({
                    "id": uid,
                    "img_path": res["img_path"],
                    "question": res["question"],
                    "pred_a": res["pred"],
                    "gt_a": res["gt"],
                    "tag": res["tag"],
                    "pred_token_len": pred_len,
                    "gt_token_len": gt_len,
                    "len_diff": len_diff
                })
            
            # Calculate final DriveLM Scores
            output = suit.evaluation()
            
            lang_score = 0
            lang_results = output["language"]
            for idx, key in enumerate(lang_results.keys()):
                if idx < 4: lang_score += lang_results[key] / 4. / 3.
                elif idx == 4: lang_score += lang_results[key] / 3. 
                else: lang_score += lang_results[key] / 10. / 3.

            scores_vector = [output["chatgpt"]/100., lang_score, output["match"]/100., output["accuracy"]]
            weights = [0.4, 0.2, 0.2, 0.2]
            final_score = sum([x * y for x, y in zip(scores_vector, weights)])
            pure_coordinate_match_score = output["pure_coordinates_match"] / 100.0
            
            avg_pred_len = sum(pred_token_lengths) / max(len(pred_token_lengths), 1)
            avg_len_diff = sum(length_differences) / max(len(length_differences), 1)
            
            current_epoch = int(self.state.epoch) if self.state.epoch is not None else 0
            save_path = os.path.join(self.args.output_dir, f"val_vqa_results_epoch_{current_epoch}.jsonl")
            
            #print(f"saving val res to {save_path}")
            #print(f"final score: {final_score}")
            #print(f"pure coordinate match score: {pure_coordinate_match_score}")

            eval_metrcis_res = {
                f"{prefix}_drivelm_final_score": final_score,
                f"{prefix}_accuracy": output["accuracy"],
                f"{prefix}_chatgpt": output["chatgpt"],
                f"{prefix}_match": output["match"],
                f"{prefix}_pure_coordinate_match": pure_coordinate_match_score,
                f"{prefix}_lang_score": lang_score,
                f"{prefix}_avg_pred_token_len": avg_pred_len,
                f"{prefix}_avg_token_len_diff": avg_len_diff,
                **{f"{prefix}_lang_{k}": v for k, v in lang_results.items()}
            }
            
            detailed_results.append(eval_metrcis_res)
            
            with open(save_path, "w") as f:
                for entry in detailed_results:
                    f.write(json.dumps(entry, indent=4) + "\n")

            return eval_metrcis_res
        
        # Ranks 1-7 return an empty dictionary. They will wait at the barrier in `evaluate()`
        else:
            return {}
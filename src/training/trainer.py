import json
import os
import torch
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

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 1. Standard Forward
        outputs = model(**inputs)
        total_loss = outputs.loss 

        # 2. Extended Logic (Placeholder for now)
        if self.mode == "extended":
            pass # We will add this later!

        return (total_loss, outputs) if return_outputs else total_loss
    
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
                    
            print(f"Total Unique Validated Samples: {len(unique_results)}")

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
            
            print(f"saving val res to {save_path}")
            print(f"final score: {final_score}")
            print(f"pure coordinate match score: {pure_coordinate_match_score}")

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
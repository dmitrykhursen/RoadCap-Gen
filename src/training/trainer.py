import json
import os
import torch
from tqdm import tqdm
import wandb
from transformers import Trainer
from external.DriveLM.challenge.evaluation import evaluation_suit


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
        
        # 2. Run custom generation & DriveLM metrics (Rank 0 only)
        if self.args.process_index == 0:
            print(f"\n--- Epoch {self.state.epoch}: Starting Custom DriveLM Evaluation ---")
            custom_results = self.run_drive_lm_metrics(metric_key_prefix)
            metrics.update(custom_results)
            
            # Log the merged metrics to WandB manually to ensure they appear
            if self.args.report_to and "wandb" in self.args.report_to:
                wandb.log(metrics, step=self.state.global_step)
                
        return metrics

    def run_drive_lm_metrics(self, prefix):
        self.model.eval()
        suit = evaluation_suit()
        eval_dataloader = self.get_eval_dataloader()
        detailed_results = []
        
        # --- NEW: Trackers for length metrics ---
        pred_token_lengths = []
        length_differences = []

        for batch in tqdm(eval_dataloader, desc="VQA Evaluation"):
            with torch.no_grad():
                
                # 1. Build generation arguments dynamically
                gen_kwargs = {
                    "input_ids": batch["input_ids"].to(self.args.device),
                    "max_new_tokens": 300,
                    "pad_token_id": self.processing_class.tokenizer.eos_token_id
                }
                
                # 2. Add attention mask (fixes the warning)
                if "attention_mask" in batch:
                    gen_kwargs["attention_mask"] = batch["attention_mask"].to(self.args.device)
                
                # 3. Add vision tensors
                if "pixel_values" in batch and batch["pixel_values"] is not None:
                    gen_kwargs["pixel_values"] = batch["pixel_values"].to(self.args.device)
                    
                # 4. CRITICAL FOR LLAVA 1.6: Add image sizes!
                if "image_sizes" in batch and batch["image_sizes"] is not None:
                    gen_kwargs["image_sizes"] = batch["image_sizes"].to(self.args.device)

                # 5. Generate
                generated_ids = self.model.generate(**gen_kwargs)
                
                # Find the length of the input prompt
                input_length = batch["input_ids"].shape[1]
                # Slice the generated IDs to drop the prompt
                new_tokens = generated_ids[:, input_length:]
                # Decode ONLY the new answer
                preds_a = self.processing_class.batch_decode(new_tokens, skip_special_tokens=True)
                
                labels = batch["labels"].clone()
                labels[labels == -100] = self.processing_class.tokenizer.pad_token_id
                gt_a = self.processing_class.batch_decode(labels, skip_special_tokens=True)
                
                # Clean up the ground truth string if needed (optional)
                gt_a = [gt.split("[/INST]")[-1].strip() for gt in gt_a]
                # Clean up prediction just in case any leading spaces exist
                preds_a = [pred.strip() for pred in preds_a]

                # Retrieve our metadata lists
                img_paths = batch["img_paths"]
                questions = batch["questions"]
                tags = batch["tags"]

                for i in range(len(preds_a)):
                    suit.forward(tags[i], preds_a[i], gt_a[i])
                    
                    # --- NEW: Calculate Token Lengths ---
                    # We encode the strings back to get the exact token count (ignoring special tokens)
                    pred_len = len(self.processing_class.tokenizer.encode(preds_a[i], add_special_tokens=False))
                    gt_len = len(self.processing_class.tokenizer.encode(gt_a[i], add_special_tokens=False))
                    len_diff = pred_len - gt_len
                    
                    pred_token_lengths.append(pred_len)
                    length_differences.append(len_diff)
                    
                    # Store the full info
                    detailed_results.append({
                        "img_path": img_paths[i],
                        "question": questions[i],
                        "pred_a": preds_a[i],
                        "gt_a": gt_a[i],
                        "tag": tags[i],
                        "pred_token_len": pred_len,    # Saved to JSONL
                        "gt_token_len": gt_len,        # Saved to JSONL
                        "len_diff": len_diff           # Saved to JSONL
                    })
        
        print("eval pred:")
        print(detailed_results)
        
        # Calculate DriveLM Scores
        output = suit.evaluation()
        
        # Weighted Scoring Logic
        lang_score = 0
        lang_results = output["language"]
        for idx, key in enumerate(lang_results.keys()):
            if idx < 4: lang_score += lang_results[key] / 4. / 3.
            elif idx == 4: lang_score += lang_results[key] / 3. 
            else: lang_score += lang_results[key] / 10. / 3.

        scores_vector = [output["chatgpt"]/100., lang_score, output["match"]/100., output["accuracy"]]
        weights = [0.4, 0.2, 0.2, 0.2]
        final_score = sum([x * y for x, y in zip(scores_vector, weights)])

        # --- NEW: Calculate Average Length Metrics ---
        avg_pred_len = sum(pred_token_lengths) / max(len(pred_token_lengths), 1)
        avg_len_diff = sum(length_differences) / max(len(length_differences), 1)
        
        current_epoch = int(self.state.epoch) if self.state.epoch is not None else 0
        # Save to JSONL for easy inspection
        save_path = os.path.join(self.args.output_dir, f"vqa_results_epoch_{current_epoch}.jsonl")
        print(f"saving evbal res to {save_path}")
        print(f"final score: {final_score}")
        print(f"scores vector: {scores_vector}")

        # Format for Trainer/WandB
        eval_metrcis_res = {
            f"{prefix}/drive_lm_final": final_score,
            f"{prefix}/accuracy": output["accuracy"],
            f"{prefix}/chatgpt": output["chatgpt"],
            f"{prefix}/match": output["match"],
            f"{prefix}/lang_score": lang_score,

            f"{prefix}/avg_pred_token_len": avg_pred_len,
            f"{prefix}/avg_token_len_diff": avg_len_diff,
            **{f"{prefix}/lang_{k}": v for k, v in lang_results.items()}
        }
        
        # Append summary to JSONL
        detailed_results.append(eval_metrcis_res)
        
        with open(save_path, "w") as f:
            for entry in detailed_results:
                f.write(json.dumps(entry, indent=4) + "\n")

        self.model.train()
        
        print(f"eval metrics results: ")
        print(eval_metrcis_res)

        return eval_metrcis_res
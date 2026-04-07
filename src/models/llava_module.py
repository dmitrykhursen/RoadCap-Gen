# src/models/llava_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlavaNextForConditionalGeneration

# Flux VAE depth latent dimensions (fixed for NuScenes image resolution)
DEPTH_C, DEPTH_H, DEPTH_W = 32, 35, 63


class RoadCapLLaVA(LlavaNextForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        model = super().from_pretrained(path, **kwargs)
        # Initialize depth distillation head (randomly; not in pretrained weights)
        model.depth_projector = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.GELU(),
            nn.Linear(1024, DEPTH_C * DEPTH_H * DEPTH_W),
        ).to(dtype=model.dtype)
        # Register a permanent hook on the final norm to capture last hidden state.
        # Permanent (not per-forward) avoids any interaction with gradient checkpointing.
        model._last_hidden_state = None
        def _capture_norm_output(module, input, output):
            model._last_hidden_state = output if isinstance(output, torch.Tensor) else output[0]
        # transformers 4.57: LlavaNextForConditionalGeneration.model = LlavaNextModel,
        # LlavaNextModel.language_model = MistralModel (no CausalLM wrapper).
        # lm_head lives directly on the outer model, not inside .language_model.
        model.model.language_model.norm.register_forward_hook(_capture_norm_output)
        return model

    def project_to_depth_latents(self, hidden_states, attention_mask):
        """
        Pool last hidden states and project to Flux VAE depth latent space.

        Args:
            hidden_states: [B, seq_len, 4096]
            attention_mask: [B, seq_len]
        Returns:
            [B, DEPTH_C, DEPTH_H, DEPTH_W]
        """
        # Mean-pool over non-padding token positions
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # [B, seq_len, 1]
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B, 4096]
        
        # Grab the dtype dynamically from the module's parameter generator
        target_dtype = next(self.depth_projector.parameters()).dtype
        pooled = pooled.to(target_dtype)
        # Project to depth latent space
        pred = self.depth_projector(pooled)  # [B, 32*35*63]
        return pred.view(-1, DEPTH_C, DEPTH_H, DEPTH_W)

    def get_visual_features(self, outputs):
        # Extract last hidden states (used in Extended Mode)
        return outputs.hidden_states[-1]
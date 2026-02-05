import os
import sys
from pathlib import Path

import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torchvision.transforms as T

from diffusers import AutoencoderKL
# VB: I do not recommand this for import but I don't want to destroy the structure of the repo
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.join(ROOT, "external", "depth-anything-3"))
from depth_anything_3.api import DepthAnything3

# -----------------------------
# Parse input arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Extract depth from images in a folder and encode/decode using Flux VAE")
parser.add_argument("--input_folder", type=str, required=True, help="Path to folder containing input images")
parser.add_argument("--output_folder", type=str, required=True, help="Path to folder containing embedding images")
parser.add_argument("--hf_cache", type=str, default="", help="data destination")
parser.add_argument("--bsize", type=int, default=8, help="batch size")
parser.add_argument("--visu", action="store_true", help="Visualise the reconstruction")

args = parser.parse_args()
input_folder = args.input_folder
output_folder = args.output_folder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Depth Anything 3 model
model = DepthAnything3.from_pretrained("depth-anything/da3-base", cache_dir=args.hf_cache)
model = model.to(device=device)

# Load Flux VAE
vae = AutoencoderKL.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B", subfolder="vae", torch_dtype=torch.float16, cache_dir=args.hf_cache
).cuda().eval()


# Transform for VAE
transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def compute_psnr(img1, img2, max_val=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()

# stream images
# replace with the DriveLM dataloader
# dset = [f for f in os.listdir(input_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

# Prepare dataset
dset = []
for root, _, files in os.walk(input_folder):
    for f in files:
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            dset.append(os.path.join(root, f))

# Make sure output folder exists
Path(output_folder).mkdir(parents=True, exist_ok=True)

# Process in batches
for i in tqdm(range(0, len(dset), args.bsize)):
    batch_paths = dset[i:i+args.bsize]
    
    images = []
    for img_path in batch_paths:
        # Depth inference
        prediction = model.inference([img_path], export_dir=None, export_format="npz")
        depth = prediction.depth[0]

        # Normalize & convert to RGB
        depth_vis = (depth - depth.min()) / (depth.max() - depth.min())
        depth_vis_rgb = (plt.cm.plasma((depth_vis * 255).astype(np.uint8))[:, :, :3] * 255).astype(np.uint8)
        depth_rgb_img = Image.fromarray(depth_vis_rgb)
        images.append(transform(depth_rgb_img))

    # Stack batch
    x = torch.stack(images, dim=0).cuda().half()  # [B, C, H, W]

    # Encode batch with VAE
    with torch.no_grad():
        latents = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor

    # Save latents individually
    for j, img_path in enumerate(batch_paths):
        fname = Path(output_folder) / (Path(img_path).stem + ".pt")
        torch.save(latents[j].cpu(), fname)

    # Optional visualization for first image in batch
    if args.visu:
        with torch.no_grad():
            recon = vae.decode(latents[0:1] / vae.config.scaling_factor).sample
        recon = (recon.clamp(-1, 1) + 1) / 2
        recon = recon[0].cpu().float()
        orig = (x[0].cpu() * 0.5 + 0.5).clamp(0, 1).float()

        psnr_value = compute_psnr(orig, recon)
        print(f"PSNR: {psnr_value:.2f} dB")

        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.title("Original Depth RGB")
        plt.imshow(orig.permute(1,2,0))
        plt.axis("off")
        plt.subplot(1,2,2)
        plt.title("Reconstructed Depth RGB")
        plt.imshow(recon.permute(1,2,0))
        plt.axis("off")
        plt.tight_layout()
        plt.show()


# python depth_emb_extract.py --input_folder="/mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/nuscenes/train_val_samples/" --output_folder="/mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/nuscenes/train_val_sample_depth_emb/" --hf_cache="/scratch/project/eu-25-50/huggingface_cache/"
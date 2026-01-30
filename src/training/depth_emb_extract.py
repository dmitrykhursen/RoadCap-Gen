import os
import sys

import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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
parser.add_argument(
    "--input_folder",
    type=str,
    required=True,
    help="Path to folder containing input images"
)

parser.add_argument(
    "--output_folder",
    type=str,
    required=True,
    help="Path to folder containing embedding images"
)

parser.add_argument(
    "--visu",
    type=str,
    action="store_true",
    help="Visualise the reconstruction"
)

args = parser.parse_args()
input_folder = args.input_folder
output_folder = args.output_folder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Depth Anything 3 model
model = DepthAnything3.from_pretrained("depth-anything/da3-base")
model = model.to(device=device)

# Load Flux VAE
vae = AutoencoderKL.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    subfolder="vae",
    torch_dtype=torch.float16
).cuda().eval()


# Transform for VAE
transform = T.Compose([
    T.Resize((1024, 1024)), # image size? 
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
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

for img_name in image_files:
    img_path = os.path.join(input_folder, img_name)
    print(f"\nProcessing: {img_name}")

    # Depth inference
    prediction = model.inference([img_path], export_dir=None, export_format="npz")  # no saving
    depth = prediction.depth[0]  # [H, W] float32

    # Normalize RGB depth
    depth_vis = (depth - depth.min()) / (depth.max() - depth.min())
    depth_vis_rgb = (plt.cm.plasma((depth_vis * 255).astype(np.uint8))[:, :, :3] * 255).astype(np.uint8)
    depth_rgb_img = Image.fromarray(depth_vis_rgb)

    # Encode & Decode with VAE
    x = transform(depth_rgb_img).unsqueeze(0).cuda().half()

    with torch.no_grad():
        latents = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
    
    # TODO: add saving function
    
    if args.visu:
        with torch.no_grad():
            recon = vae.decode(latents / vae.config.scaling_factor).sample
        recon = (recon.clamp(-1, 1) + 1) / 2
        recon = recon[0].cpu().float()
        orig = (x[0].cpu() * 0.5 + 0.5).clamp(0, 1).float()

        # -----------------------------
        # Compute PSNR
        # -----------------------------
        psnr_value = compute_psnr(orig, recon)
        print(f"PSNR: {psnr_value:.2f} dB")

        # -----------------------------
        # Display original vs reconstruction
        # -----------------------------
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

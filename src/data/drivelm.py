import json
import os
import torch
import random
from torch.utils.data import Dataset
from PIL import Image

# 🌟 CRITICAL FIX FOR HPC DEADLOCKS 🌟
# This prevents OpenCV (if imported anywhere in your pipeline) from clashing with PyTorch
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
# -----------------------------------

class DriveLMDataset(Dataset):
    def __init__(
        self, 
        data_path, 
        image_folder, 
        tokenizer=None, 
        image_processor=None, 
        split="train", 
        split_ratio=(0.8, 0.1, 0.1), # (Train, Val, Test)
        data_usage=1.0,
        seed=42,
        depth_emb_folder=None
    ):
        self.image_folder = image_folder
        self.depth_emb_folder = depth_emb_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
        print(f"📂 [DriveLM] Loading data from: {data_path}")
        with open(data_path, "r") as f:
            raw_data = json.load(f)
        
        # 1. Filter: Ensure 'image' field exists
        valid_samples = [x for x in raw_data if 'image' in x]
        
        # 2. SPLITTING LOGIC (Train / Val / Test)
        # Deterministic Shuffle
        if split in ["train", "val", "test"]:
            random.seed(seed)
            random.shuffle(valid_samples)
        
        total = len(valid_samples)
        train_end = int(total * split_ratio[0])
        val_end = int(total * (split_ratio[0] + split_ratio[1]))
        
        # Select the specific split
        if split == "train":
            self.samples = valid_samples[:train_end]
        elif split == "val":
            self.samples = valid_samples[train_end:val_end]
        elif split == "test":
            self.samples = valid_samples[val_end:]
        elif split == "all_data":
            self.samples = valid_samples
        else:
            raise ValueError(f"Unknown split: {split}")

        # 3. SUBSET LOGIC (Data Usage)
        if data_usage < 1.0:
            subset_size = int(len(self.samples) * data_usage)
            self.samples = self.samples[:subset_size]
            print(f"✂️  Subsampling: Using {data_usage*100}% of {split} set.")

        print(f"✅ Dataset Ready [{split.upper()}]: {len(self.samples)} samples (Total pool: {total})")

    def __len__(self):
        return len(self.samples)

    def _concat_images_horizontal(self, images):
        """Helper to stitch a list of images side-by-side."""
        w, h = images[0].size
        dst = Image.new('RGB', (w * len(images), h))
        for i, img in enumerate(images):
            dst.paste(img, (i * w, 0))
        return dst

    def _concat_images_vertical(self, images):
        """Helper to stitch a list of images top-to-bottom."""
        w, h = images[0].size
        dst = Image.new('RGB', (w, h * len(images)))
        for i, img in enumerate(images):
            dst.paste(img, (0, i * h))
        return dst

    def _create_surround_collage(self, image_map):
        """
        Smartly stitches images.
        1. If only 1 image exists -> Returns it directly (RoadCap style).
        2. If multiple -> Creates a 2x3 grid (DriveLM style).
        """
        tile_size = 336
        
        # --- CASE 1: SINGLE IMAGE (RoadCap / Front-Only) ---
        if len(image_map) == 1:
            img = list(image_map.values())[0]
            # Resize strictly to tile_size for consistency
            return img.resize((tile_size, tile_size), Image.BICUBIC)

        # --- CASE 2: SURROUND VIEW (DriveLM) ---
        grid_layout = [
            ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'], # Row 1
            ['CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT']   # Row 2
        ]

        row_images = []
        
        for row_keys in grid_layout:
            col_images = []
            for cam_key in row_keys:
                if cam_key in image_map:
                    # Found the camera: Resize and use it
                    img = image_map[cam_key].resize((tile_size, tile_size), Image.BICUBIC)
                else:
                    # Missing camera: Create a black placeholder tile
                    img = Image.new('RGB', (tile_size, tile_size), color=(0, 0, 0))
                
                col_images.append(img)
            
            # Stitch this row horizontally
            row_strip = self._concat_images_horizontal(col_images)
            row_images.append(row_strip)

        # Stitch all rows vertically
        full_collage = self._concat_images_vertical(row_images)
        
        return full_collage

    def _identify_camera(self, path):
        """Simple substring matching to identify camera ID."""
        if "CAM_FRONT_LEFT" in path: return "CAM_FRONT_LEFT"
        if "CAM_FRONT_RIGHT" in path: return "CAM_FRONT_RIGHT"
        if "CAM_FRONT" in path: return "CAM_FRONT"
        if "CAM_BACK_LEFT" in path: return "CAM_BACK_LEFT"
        if "CAM_BACK_RIGHT" in path: return "CAM_BACK_RIGHT"
        if "CAM_BACK" in path: return "CAM_BACK"
        return "UNKNOWN"

    def __getitem__(self, index):
        data_item = self.samples[index]
        
        # --- 1. LOAD AND CLEAN PATHS ---
        image_paths = data_item.get('image', [])
        if isinstance(image_paths, str):
            image_paths = [image_paths]
            
        image_map = {}
        for p in image_paths:
            parts = p.split('/')
            if len(parts) >= 2:
                short_path = os.path.join(parts[-2], parts[-1]) 
            else:
                short_path = p
            
            full_path = os.path.join(self.image_folder, short_path)
            cam_name = self._identify_camera(p)
            
            try:
                img = Image.open(full_path).convert("RGB")
                image_map[cam_name] = img
            except Exception:
                pass

        # Safety net: if all images failed to load, inject a blank collage
        if len(image_map) == 0:
            final_image = Image.new('RGB', (336 * 3, 336 * 2), color=(0, 0, 0))
        else:
            # --- 2. CREATE COLLAGE ---
            final_image = self._create_surround_collage(image_map)


        # --- 1b. LOAD DEPTH LATENTS ---
        depth_latents = None
        if self.depth_emb_folder is not None:
            loaded = []
            for cam_name, _ in image_map.items():
                for p in image_paths:
                    if self._identify_camera(p) == cam_name:
                        stem = os.path.splitext(os.path.basename(p))[0]
                        emb_path = os.path.join(self.depth_emb_folder, cam_name, stem + ".pt")
                        if os.path.isfile(emb_path):
                            try:
                                loaded.append(torch.load(emb_path, map_location="cpu", weights_only=True))
                            except Exception:
                                pass
                        break
            if len(loaded) > 0:
                depth_latents = torch.stack(loaded, dim=0).float().mean(dim=0)

        # --- 3. TEXT ---
        convs = data_item['conversations']
        question = convs[0]['value'].replace("<image>", "").strip()
        answer = convs[1]['value'].strip()

        return {
            "image": final_image,
            "question": question,
            "answer": answer,
            "img_path": data_item.get('image', ["unknown"])[0],
            "tag": data_item.get('tag', [-1]),
            "id": data_item.get('id', str(index)),
            "depth_latents": depth_latents,
        }
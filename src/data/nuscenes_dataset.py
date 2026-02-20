import os
import re
from PIL import Image, ImageFile
import numpy as np
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

class NuscenesDataset(Dataset):
    def __init__(
        self, 
        image_folder,           # Path to 'samples_grouped'
        transform=None,         
        target_cameras=None     # e.g. ["CAM_FRONT"]
    ):
        """
        Dataset for the 'grouped' Nuscenes structure:
        root/CAM_FRONT/SceneID/Image.jpg
        """
        self.image_folder = image_folder
        self.transform = transform
        self.target_cameras = target_cameras
        
        # 1. Scan and Sort
        self.image_paths = self._scan_images_optimized(image_folder)
        
        print(f"✅ [NuscenesDataset] Ready: {len(self.image_paths)} images found.")
        if len(self.image_paths) > 0:
            print(f"   First image: {os.path.basename(self.image_paths[0])}")
            print(f"   Last image:  {os.path.basename(self.image_paths[-1])}")

    def _scan_images_optimized(self, root_dir):
        """
        Optimized scanner: If target_cameras is set, only walk those specific folders.
        Structure: root_dir / CAM_NAME / SCENE_ID / img.jpg
        """
        print(f"🔍 Scanning {root_dir}...")
        valid_extensions = ('.jpg', '.jpeg', '.png')
        found_images = []

        # Decide which folders to check
        if self.target_cameras:
            # Only look inside 'CAM_FRONT', 'CAM_BACK', etc.
            dirs_to_check = [os.path.join(root_dir, cam) for cam in self.target_cameras]
        else:
            # Check everything
            dirs_to_check = [root_dir]

        for specific_dir in dirs_to_check:
            if not os.path.exists(specific_dir):
                print(f"⚠️ Warning: Directory not found: {specific_dir}")
                continue

            # Recursive walk from the camera folder downwards
            for root, _, files in os.walk(specific_dir):
                for file in files:
                    if file.lower().endswith(valid_extensions):
                        full_path = os.path.join(root, file)
                        found_images.append(full_path)

        # Sort by Timestamp
        print("⏳ Sorting images by timestamp...")
        try:
            found_images.sort(key=self._extract_timestamp)
        except Exception as e:
            print(f"⚠️ Warning: Timestamp sort failed ({e}). Using Name sort.")
            found_images.sort()

        return found_images

    def _extract_timestamp(self, path):
        """
        Parses: n015...__CAM_FRONT__1542801576412460.jpg
        Returns: 1542801576412460 (int)
        """
        filename = os.path.basename(path)
        # Regex: Find a sequence of 13+ digits (NuScenes timestamps are 16 digits)
        # This avoids matching the date '2018...' which is separated by dashes
        match = re.search(r'(\d{13,})', filename)
        if match:
            return int(match.group(1))
        
        # Fallback: if filename doesn't have the standard format, use 0
        return 0

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            
            if self.transform:
                img = self.transform(img)
            else:
                img = np.array(img)
                    
            return {
                "img": img,
                "filepath": path,
                "filename": os.path.basename(path)
            }
        except Exception as e:
            print(f"❌ Error loading {path}: {e}")
            return None
        
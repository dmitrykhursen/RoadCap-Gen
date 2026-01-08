import os
import zipfile
import io
import datetime
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

class FRONTCAMDataset(Dataset):
    """
    Base class for handling PONE dataset structure (Zipped images).
    Does NOT handle text/questions.
    """
    def __init__(self, image_folder, cameras=["camera2"]):
        self.image_folder = image_folder
        self.cameras = cameras
        
        # Scan on init
        print(f"📂 PONE: Scanning ZIPs in {image_folder}...")
        self.image_paths_list = self._scan_zip_images(image_folder)
        print(f"✅ PONE: Found {len(self.image_paths_list)} images.")

    def __len__(self):
        return len(self.image_paths_list)

    def get_image_by_index(self, idx):
        """Returns PIL Image object for a given index."""
        if idx >= len(self.image_paths_list):
            raise IndexError(f"Index {idx} out of range for {len(self.image_paths_list)} images.")
            
        img_pseudo_path = self.image_paths_list[idx]
        try:
            image_bytes = self._read_zip_bytes(img_pseudo_path)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return image
        except Exception as e:
            print(f"❌ Error reading image {img_pseudo_path}: {e}")
            # Return black image on failure to prevent crash
            return Image.new('RGB', (336, 336), color='black')

    # ================= INTERNAL HELPERS =================

    def _scan_zip_images(self, path_to_scan):
        image_paths = []
        
        # CASE 1: The input is a direct ZIP file
        if os.path.isfile(path_to_scan) and path_to_scan.endswith(".zip"):
            print(f"📂 Scanning single ZIP file: {path_to_scan}")
            image_paths.extend(self._list_zip_contents(path_to_scan))

        # CASE 2: The input is a Directory
        elif os.path.isdir(path_to_scan):
            print(f"📂 Scanning directory: {path_to_scan}")
            # 1. Look for subfolders (standard PONE structure)
            for recording_path in sorted(os.listdir(path_to_scan)):
                rec_full_path = os.path.join(path_to_scan, recording_path)
                
                # If it's a folder, check inside for camera zips
                if os.path.isdir(rec_full_path):
                    for cam in self.cameras:
                        zip_name = f"{recording_path}_{cam}.npz.zip"
                        zip_path = os.path.join(rec_full_path, zip_name)
                        if os.path.exists(zip_path):
                            image_paths.extend(self._list_zip_contents(zip_path))
                
                # If it's a zip file directly in the root folder
                elif rec_full_path.endswith(".zip"):
                     image_paths.extend(self._list_zip_contents(rec_full_path))
        
        else:
            print(f"❌ Error: Path is not a valid file or directory: {path_to_scan}")
            return []

        # 3. Sort by frame number (Crucial for alignment)
        # Assumes format: ..._FRAMENUM_TIMESTAMP.jpg
        try:
            image_paths.sort(key=lambda x: int(x.rsplit('_', 2)[-2]))
        except Exception:
            print("⚠️ Could not sort by frame number. Using generic sort.")
            image_paths.sort()
            
        return image_paths

    def _list_zip_contents(self, zip_path):
        paths = []
        valid = (".jpg", ".jpeg", ".png")
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                for f in z.infolist():
                    if f.filename.lower().endswith(valid):
                        paths.append(os.path.join(zip_path, f.filename))
        except zipfile.BadZipFile:
            print(f"❌ Bad Zip: {zip_path}")
        return paths

    def _read_zip_bytes(self, pseudo_path):
        # Split /path/to.zip/inner/file.jpg
        if ".zip" not in pseudo_path:
             raise ValueError("Path must contain .zip")
        
        zip_path, inner_path = pseudo_path.split('.zip', 1)
        zip_path += '.zip'
        inner_path = inner_path.replace("\\", "/").lstrip("/")
        
        with zipfile.ZipFile(zip_path, "r") as z:
            with z.open(inner_path) as f:
                return f.read()
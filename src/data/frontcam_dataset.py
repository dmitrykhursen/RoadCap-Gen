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
    Handles PONE/Valeo dataset structure (Zipped images).
    Reads directly from ZIP without extraction.
    """
    def __init__(self, image_folder, cameras=["camera2"], transform=None):
        self.image_folder = image_folder
        self.cameras = cameras
        self.transform = transform
        
        # Scan on init
        print(f"📂 PONE: Scanning ZIPs in {image_folder}...")
        self.image_paths_list = self._scan_zip_images(image_folder)
        print(f"✅ PONE: Found {len(self.image_paths_list)} images.")

    def __len__(self):
        return len(self.image_paths_list)

    def __getitem__(self, idx):
        """Standard PyTorch access method."""
        img_pseudo_path = self.image_paths_list[idx]
        try:
            image_bytes = self._read_zip_bytes(img_pseudo_path)
            # Use Image.open and ensure it's RGB
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Apply transforms if provided (e.g., ToTensor, Resize)
            if self.transform:
                image = self.transform(image)
            else:
                # If no transform, return as numpy array for YOLO consistency
                image = np.array(image)

            return {
                'img': image,
                'filepath': img_pseudo_path
            }
        except Exception as e:
            print(f"❌ Error reading image {img_pseudo_path}: {e}")
            return None # safe_collate in your script will handle this

    def get_image_by_index(self, idx):
        """Legacy helper for backward compatibility."""
        data = self.__getitem__(idx)
        return data['img'] if data else None

    # ================= INTERNAL HELPERS (Same as before) =================

    def _scan_zip_images(self, path_to_scan):
        image_paths = []
        if os.path.isfile(path_to_scan) and path_to_scan.endswith(".zip"):
            image_paths.extend(self._list_zip_contents(path_to_scan))
        elif os.path.isdir(path_to_scan):
            for recording_path in sorted(os.listdir(path_to_scan)):
                rec_full_path = os.path.join(path_to_scan, recording_path)
                if os.path.isdir(rec_full_path):
                    for cam in self.cameras:
                        zip_name = f"{recording_path}_{cam}.npz.zip"
                        zip_path = os.path.join(rec_full_path, zip_name)
                        if os.path.exists(zip_path):
                            image_paths.extend(self._list_zip_contents(zip_path))
                elif rec_full_path.endswith(".zip"):
                     image_paths.extend(self._list_zip_contents(rec_full_path))
        
        try:
            # Sorting by frame number (assumes ..._FRAMENUM_TIMESTAMP.jpg)
            image_paths.sort(key=lambda x: int(x.rsplit('_', 2)[-2]))
        except Exception:
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
        zip_path, inner_path = pseudo_path.split('.zip', 1)
        zip_path += '.zip'
        inner_path = inner_path.replace("\\", "/").lstrip("/")
        with zipfile.ZipFile(zip_path, "r") as z:
            with z.open(inner_path) as f:
                return f.read()
                
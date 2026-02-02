import io
import os
import time
import random
import zipfile
import numpy as np
from PIL import Image, ImageFile
import datetime

# some images can be truncated, but they should be ok to process
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

def profiler_time(func):
    """ Decorator for time profiling """

    def inner(*args, **kwargs):
        begin = time.time()
        out = func(*args, **kwargs)
        total_time = time.time() - begin
        return out, total_time

    return inner


class PONELoaderOriginal(Dataset):
    """
    A custom PyTorch Dataset to load images directly from a zip file.
    """

    def __init__(self, folder, cameras=["camera2"], file_extension="jpg", transform=None, sample_per_zipfile=1, texts=None):

        self.folder = folder
        self.file_extension = file_extension
        self.transform = transform
        self.texts = texts

        self.list_img = []
        for recording_path in os.listdir(folder):
            print("here 0 ")

            for cam in cameras:
                zip_file_path = os.path.join(folder, recording_path, recording_path + "_" + cam + ".npz.zip")

                print("here")

                # Get a list of image filenames from the zip file
                recording_img = self.get_image_filenames(zip_file_path, shuffle=False, sample=sample_per_zipfile)
                # add to the existing path
                self.list_img += recording_img

        print(f"self.list_img : {self.list_img}")

    def get_image_filenames(self, zip_file_path, shuffle=False, sample=1):

        """
        Get a list of filenames for images in the zip file.

        Args:
            zip_file_path (str): the path of the zip file
            shuffle (bool, optional): Whether to shuffle the list of filenames (default is False).
            sample (int, optional): Defines if a result should be sampled. sample=2 means from every 2 files, only 1 is returned.

        Returns:
            list: List of filenames.
        """

        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            filenames = [
                os.path.join(zip_file_path, file_info.filename)
                for file_info in zip_ref.infolist()
                if file_info.filename.endswith(f".{self.file_extension}")
            ]

            #########
            # numpy.get_sample
            if shuffle:
                random.shuffle(filenames)

            if sample > 1:
                filenames = filenames[::sample]
            ######

            # print(f"fileanames: {filenames}")
            return filenames

    # @staticmethod
    def read_image(self, file_path):
        """
        Read image data from the zip file.

        Args:
            file_path (str): The full path of the file of the image in the zip file.

        Returns:
            bytes: Image data.
        """

        # print(f"os.path.dirname(file_path): {os.path.dirname(file_path)}")
        # print(f"+++ self.list_img : {len(self.list_img)}")
        # print(f"file_path: {file_path}")
        # with open(file_path, "rb") as file:
        #     return file.read()
        #
        # # with open(file_path, "r") as file:
        # #     return file.read()

        # with zipfile.ZipFile(os.path.dirname(file_path), "r") as zip_ref:
        #     with zip_ref.open(os.path.basename(file_path)) as file:
        #         return file.read()
        zip_file_path, base_file_path = file_path.split('.zip', 1)
        zip_file_path += '.zip'
        base_file_path = base_file_path.lstrip('/')
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            with zip_ref.open(base_file_path) as file:
                return file.read()

    def __len__(self):
        return len(self.list_img)

    def __getitem__(self, idx):
        """
        Get item by index.

        Args:
            idx (int): Index.

        Returns:
            dict ->
                torch.Tensor: Processed image tensor.
                str: the full path of the image.
        """
        img_path = self.list_img[idx]
        try:
            image_data = self.read_image(img_path)
            image = np.array(Image.open(io.BytesIO(image_data)).convert("RGB"))

            outputs = {'img': image, 'texts': self.texts}
            if self.transform:
                outputs['img'] = self.transform(Image.fromarray(outputs['img']))
            outputs['filepath'] = img_path
            return outputs

        except Exception as e:
            # catch bad images (truncated images or empty)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("error_log.txt", "a") as f:
                f.write(f"[{timestamp}] ERROR at {img_path} -> {repr(e)}\n")

            print(f"[SKIP] Skipping {img_path} because of error: {repr(e)}")
            return None


class PONELoader(PONELoaderOriginal):
    def __init__(self, zip_file_path, file_extension="jpg", transform=None, sample_per_zipfile=1, texts=""):
        self.zip_file_path = zip_file_path
        self.file_extension = file_extension
        self.transform = transform
        self.texts = texts

        # Get a list of image filenames from the zip file
        self.list_img = self.get_image_filenames(zip_file_path, shuffle=False, sample=sample_per_zipfile)
        self.list_img.sort(key=lambda x: int(x.rsplit('_', 2)[-2])) # smth_N_TIMESTAMP.png, sort by N of the frame

        # # sort images in backward order (for test purposes)
        # self.list_img =  self.list_img[::-1]

        print(f"=len= self.list_img : {len(self.list_img)}")
        # print(f"== self.list_img : {self.list_img}")

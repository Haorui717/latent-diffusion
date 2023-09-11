# TODO: Add PolypSegDataset class
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import albumentations

class PolypSegDataset(Dataset):
    def __init__(self, image_path, mask_path, size=256, transpose=False):
        self.image_list = []
        self.mask_list = []
        with open(image_path, 'r') as f:
            for line in f:
                self.image_list.append(line.strip())
        with open(mask_path, 'r') as f:
            for line in f:
                self.mask_list.append(line.strip())

        self.image_list.sort()
        self.mask_list.sort()

        self.min_crop_f = 0.5
        self.max_crop_f = 1.0
        self.image_rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)
        self.transpose = transpose

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        # load image and mask with suffix jpg or png
        image = Image.open(self.image_list[item]).convert('RGB')
        mask = Image.open(self.mask_list[item]).convert('L')
        image = np.array(image).astype(np.float32) / 255.0
        mask = np.array(mask).astype(np.float32) / 255.0

        # randomly crop image and mask at the same relative location
        min_side_len = min(image.shape[:2])
        crop_side_len = min_side_len * np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
        crop_side_len = int(crop_side_len)
        h, w = image.shape[:2]
        top = np.random.randint(0, h - crop_side_len)
        left = np.random.randint(0, w - crop_side_len)
        image = image[top:top + crop_side_len, left:left + crop_side_len]
        mask = mask[top:top + crop_side_len, left:left + crop_side_len]

        image = self.image_rescaler(image=image)['image']
        mask = self.image_rescaler(image=mask)['image'][..., None]
        if image.shape[1] != 256:
            raise Exception
        if mask.shape[1] != 256:
            raise Exception

        if self.transpose:
            image = image.transpose(2, 0, 1)
            mask = mask.transpose(2, 0, 1)
        return torch.from_numpy(image), torch.from_numpy(mask)
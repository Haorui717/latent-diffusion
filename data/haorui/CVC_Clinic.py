import os

import albumentations
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class CVC_Clinic_Reconstruction(Dataset):
    # dataset for loading CVC_Clinic image and mask
    def __init__(self, root_path, image_path, mask_path, size=256):
        # read image path and file path into list. Each line of image_path and mask_path is a path to an image or mask
        self.image_list = []
        self.mask_list = []
        with open(image_path, 'r') as f:
            for line in f:
                self.image_list.append(os.path.join(root_path, line.strip()))
        with open(mask_path, 'r') as f:
            for line in f:
                self.mask_list.append(os.path.join(root_path, line.strip()))


        self.image_list.sort()
        self.mask_list.sort()

        self.min_crop_f = 0.5  # min crop factor
        self.max_crop_f = 1.0  # max crop factor
        self.image_rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)

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

        # resize image and mask to (size x size)
        image = self.image_rescaler(image=image)['image']
        mask = self.image_rescaler(image=mask)['image'][..., None]

        # set mask to 0 or 1
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        # get masked image and rescale three images to [-1, 1]
        masked_image = (1 - mask) * image
        masked_image = masked_image * 2.0 - 1.0
        image = image * 2.0 - 1.0
        mask = mask * 2.0 - 1.0

        batch = dict()
        batch['image'] = image
        batch['mask'] = mask
        batch['masked_image'] = masked_image
        batch['image_path'] = self.image_list[item]
        batch['mask_path'] = self.mask_list[item]

        return batch
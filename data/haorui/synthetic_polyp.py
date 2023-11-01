import os, sys
from typing import Any
sys.path.append("/home/yixiao/haorui/ccvl15/haorui/latent-diffusion-Local")

import albumentations
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from util.gen_polyp_utils import check_polyp_size, reshape_mask

class Random_Mask_Dataset(Dataset):
    '''
    Synthetic Polyp Dataset
    Randomly load healthy images and masks. Then generate polyps using diffusion model.
    '''
    def __init__(self, image_path, mask_path, len, transpose=False, size=256) -> None:
        super().__init__()
        self.len = len
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
        self.image_rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)
        self.transpose = transpose

    def __len__(self):
        return self.len
    
    def __getitem__(self, index: Any) -> Any:
        image_path = self.image_list[np.random.randint(0, len(self.image_list))]
        image = Image.open(image_path).convert('RGB')
        image = np.array(image).astype(np.float32) / 255.0
        # crop the image at 2/3 of the longer side
        min_side_len = min(image.shape[:2])
        crop_side_len = min_side_len
        if image.shape[0] > image.shape[1]:
            start_x = min(image.shape[0] // 3 * 2 - crop_side_len // 2, image.shape[0] - crop_side_len)
            start_y = 0
        else:
            start_x = 0
            start_y = min(image.shape[1] // 3 * 2 - crop_side_len // 2, image.shape[1] - crop_side_len)
        image = image[start_x:start_x+crop_side_len, start_y:start_y+crop_side_len]
        # get mask
        while True:
            mask_path = self.mask_list[np.random.randint(0, len(self.mask_list))]
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask).astype(np.float32)
            mask = reshape_mask(mask, image.shape[:2], random_resize=False, random_crop=True)
            if check_polyp_size(mask):
                # finalize mask
                mask = mask / 255.0
                mask[mask > 0.5] = 1.0
                mask[mask <= 0.5] = 0.0
                break
        image = self.image_rescaler(image=image)['image']
        mask = self.image_rescaler(image=mask)['image'][..., None]
        masked_image = (1 - mask) * image

        masked_image = masked_image * 2 - 1
        image = image * 2 - 1
        mask = mask * 2 - 1

        batch = dict()
        if self.transpose:
            batch['image'] = image.transpose(2, 0, 1)
            batch['mask'] = mask.transpose(2, 0, 1)
            batch['masked_image'] = masked_image.transpose(2, 0, 1)
        else:
            batch['image'] = image
            batch['mask'] = mask
            batch['masked_image'] = masked_image
        batch['image_path'] = image_path
        batch['condition'] = dict(
            mask=batch['mask'],
            masked_image=batch['masked_image']
        )
        return batch
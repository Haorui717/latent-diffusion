import os

image_path = "/mnt/ccvl15/haorui/datasets/CVC-ClinicDB/PNG/Original"
mask_path = "/mnt/ccvl15/haorui/datasets/CVC-ClinicDB/PNG/Ground Truth"
# randomly split a dataset into train and test
image_list = []
mask_list = []
for filename in os.listdir(image_path):
    image_list.append(os.path.join("Original", filename))

for filename in os.listdir(mask_path):
    mask_list.append(os.path.join("Ground Truth", filename))

image_list.sort()
mask_list.sort()

# randomly select 10% of the dataset as validation set
val_num = int(len(image_list) * 0.1)

import random

random.seed(0)  # set random seed
val_ind = random.sample(range(len(image_list)), val_num)

# split dataset into train and test
train_image_list = []
train_mask_list = []
val_image_list = []
val_mask_list = []
for i in range(len(image_list)):
    if i in val_ind:
        val_image_list.append(image_list[i])
        val_mask_list.append(mask_list[i])
    else:
        train_image_list.append(image_list[i])
        train_mask_list.append(mask_list[i])

train_image_list.sort()
train_mask_list.sort()
val_image_list.sort()
val_mask_list.sort()

# save list to txt file
with open('train_image_list.txt', 'w') as f:
    for item in train_image_list:
        f.write("%s\n" % item)

with open('train_mask_list.txt', 'w') as f:
    for item in train_mask_list:
        f.write("%s\n" % item)

with open('val_image_list.txt', 'w') as f:
    for item in val_image_list:
        f.write("%s\n" % item)

with open('val_mask_list.txt', 'w') as f:
    for item in val_mask_list:
        f.write("%s\n" % item)

print('done')

# Path: util\split_CVC.py
# Compare this snippet from data\haorui\CVC_Clinic.py:
# import os
#
# import albumentations
# import cv2
# import numpy as np
# import torch
# from torch.utils.data import Dataset
#
# class CVC_Clinic_Reconstruction(Dataset):
#     # dataset for loading CVC_Clinic image and mask
#     def __init__(self, image_path, mask_path, size=256):
#         # load filenames of images into a list
#         self.image_list = []
#         self.mask_list = []
#         for filename in os.listdir(image_path):
#             self.image_list.append(os.path.join(image_path, filename))
#         for filename in os.listdir(mask_path):
#             self.mask_list.append(os.path.join(mask_path, filename))
#
#         self.image_list.sort()
#         self.mask_list.sort()
#
#         self.min_crop_f = 0.5  # min crop factor
#         self.max_crop_f = 1.0  # max
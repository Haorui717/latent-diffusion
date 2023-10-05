import os
import sys
import numpy as np
import PIL.Image as Image
from tqdm import tqdm
sys.path.append('/home/yixiao/haorui/ccvl15/haorui/latent-diffusion-Local')
from util.preprocess import load_filepaths, save_filepaths


# save_filepaths(['/home/yixiao/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/masksAll_positive'],
#                '/home/yixiao/haorui/ccvl15/haorui/datasets/combined_dataset/PolypGEN/PolypGEN_p_all_masks.txt')


mask_paths = load_filepaths('/home/yixiao/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/masksAll_positive')
masks = {}
for mask_path in tqdm(mask_paths, total=len(mask_paths)):
    mask = np.array(Image.open(mask_path))
    masks[mask_path] = mask
polyps_masks = {path: mask for path, mask in tqdm(masks.items(), total=len(masks)) if np.sum(mask) > 20000}
empty_masks = {path: mask for path, mask in tqdm(masks.items(), total=len(masks)) if np.sum(mask) <= 20000}
# print(len(polyps_masks), len(empty_masks))
empty_mask_filepath = "/home/yixiao/haorui/ccvl15/haorui/datasets/combined_dataset/PolypGEN/PolypGEN_p_empty_masks.txt"
polyp_mask_filepath = "/home/yixiao/haorui/ccvl15/haorui/datasets/combined_dataset/PolypGEN/PolypGEN_p_nonempty_masks.txt"
with open(empty_mask_filepath, 'w') as f:
    for path in empty_masks.keys():
        f.write(path + '\n')
with open(polyp_mask_filepath, 'w') as f:
    for path in polyps_masks.keys():
        f.write(path + '\n')


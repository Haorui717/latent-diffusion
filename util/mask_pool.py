from contour_box import crop_bounding_box
import numpy as np
from PIL import Image
import os

from tqdm import tqdm

if __name__ == '__main__':
    mask_files = ['/home/zongwei/haorui/ccvl15/haorui/datasets/combined_dataset/Kvasir_masks.txt']
    mask_paths = []
    save_dir = '/home/zongwei/haorui/ccvl15/haorui/datasets/mask_pool'
    for mask_file in mask_files:
        with open(mask_file, 'r') as f:
            for line in f:
                mask_paths.append(line.strip())
    cnt = 2172
    for mask_path in tqdm(mask_paths, total=len(mask_paths)):
        mask = np.array(Image.open(mask_path).convert('L'))
        boxes = crop_bounding_box(mask, min_area=100)
        for j in range(len(boxes)):
            x, y, w, h = boxes[j]
            boxed_mask = mask[x:x+w, y:y+h]
            Image.fromarray(boxed_mask.astype(np.uint8)).save(os.path.join(save_dir, f'{cnt}.png'))
            cnt += 1

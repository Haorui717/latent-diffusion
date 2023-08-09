'''
util functions to crop the bounding boxes of polyps
'''

import numpy as np
import cv2
from PIL import Image
import os

from tqdm import tqdm


def crop_bounding_box(mask, min_area=10):
    '''
    crop the bounding box of the polyp
    :param image: image with polyp
    :param mask: mask of the polyp
    :return: cropped image and mask
    '''
    # check if mask is uint8. If not, convert it to uint8
    if mask.dtype != np.uint8:
        mask *= 255
        mask = mask.astype(np.uint8)

    _, thresh = cv2.threshold(mask, 128, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []

    for contour in contours:
        y, x, h, w = cv2.boundingRect(contour)
        if h * w > min_area:
            bounding_boxes.append([x, y, w, h])

    return bounding_boxes

if __name__ == '__main__':
    image_path = '/home/zongwei/haorui/ccvl15/haorui/datasets/combined_dataset/Kvasir_images.txt'
    mask_path = '/home/zongwei/haorui/ccvl15/haorui/datasets/combined_dataset/Kvasir_masks.txt'
    image_list = []
    mask_list = []
    save_dir = '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/bounding_box_visual'
    os.makedirs(save_dir, exist_ok=True)

    with open(image_path, 'r') as f:
        for line in f:
            image_list.append(line.strip())
    with open(mask_path, 'r') as f:
        for line in f:
            mask_list.append(line.strip())

    for i in tqdm(range(len(image_list))):
        image = np.array(Image.open(image_list[i]).convert('RGB'))
        mask = np.array(Image.open(mask_list[i]).convert('L'))
        boxed_images = crop_bounding_box(mask)
        for j in range(len(boxed_images)):
            x, y, w, h = boxed_images[j]
            boxed_image = image[x:x+w, y:y+h, :]
            boxed_mask = mask[x:x+w, y:y+h]
            combined = np.concatenate((boxed_image, np.repeat(boxed_mask[:, :, np.newaxis], 3, axis=2)), axis=1)
            Image.fromarray(combined.astype(np.uint8)).save(os.path.join(save_dir, f'{os.path.basename(image_list[i])[:-4]}_{x}_{y}.png'))
            print(h, w)

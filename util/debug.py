from PIL import Image
import numpy as np
import cv2

#%%
# load image and mask paths from txt file
with open('/home/zongwei/haorui/ccvl15/haorui/datasets/combined_dataset/reduce_data_dataset/CVC_all_images.txt', 'r') as f:
    all_images = f.readlines()

with open('/home/zongwei/haorui/ccvl15/haorui/datasets/combined_dataset/reduce_data_dataset/CVC_all_masks.txt', 'r') as f:
    all_masks = f.readlines()

#%%
# randomly select 100 images and corresponding masks as validation set
import random

# randomly sample 100 indices from 0 to len(image_paths)
val_indices = random.sample(range(len(all_images)), 100)

val_indices.sort()

val_images = [all_images[i] for i in val_indices]
val_masks = [all_masks[i] for i in val_indices]

train_images = [all_images[i] for i in range(len(all_images)) if i not in val_indices]
train_masks = [all_masks[i] for i in range(len(all_masks)) if i not in val_indices]

#%%

def random_select(images, masks, num):
    indices = random.sample(range(len(images)), num)
    indices.sort()
    return [images[i] for i in indices], [masks[i] for i in indices]

train_images_100, train_masks_100 = random_select(train_images, train_masks, 100)
train_images_200, train_masks_200 = random_select(train_images, train_masks, 200)
train_images_300, train_masks_300 = random_select(train_images, train_masks, 300)
train_images_400, train_masks_400 = random_select(train_images, train_masks, 400)
train_images_500, train_masks_500 = random_select(train_images, train_masks, 500)
train_images_600, train_masks_600 = random_select(train_images, train_masks, 600)
train_images_700, train_masks_700 = random_select(train_images, train_masks, 700)
train_images_800, train_masks_800 = random_select(train_images, train_masks, 800)
train_images_900, train_masks_900 = random_select(train_images, train_masks, 900)

#%%
# save image and mask paths to txt files
with open('/home/zongwei/haorui/ccvl15/haorui/datasets/combined_dataset/reduce_data_dataset/CVC_val_images.txt', 'w') as f:
    for image in val_images:
        f.write(image)

with open('/home/zongwei/haorui/ccvl15/haorui/datasets/combined_dataset/reduce_data_dataset/CVC_val_masks.txt', 'w') as f:
    for mask in val_masks:
        f.write(mask)

with open('/home/zongwei/haorui/ccvl15/haorui/datasets/combined_dataset/reduce_data_dataset/CVC_train_images.txt', 'w') as f:
    for image in train_images:
        f.write(image)

with open('/home/zongwei/haorui/ccvl15/haorui/datasets/combined_dataset/reduce_data_dataset/CVC_train_masks.txt', 'w') as f:
    for mask in train_masks:
        f.write(mask)

for i in range(100, 1000, 100):
    with open('/home/zongwei/haorui/ccvl15/haorui/datasets/combined_dataset/reduce_data_dataset/CVC_train_images_' + str(i) + '.txt', 'w') as f:
        for image in eval('train_images_' + str(i)):
            f.write(image)

    with open('/home/zongwei/haorui/ccvl15/haorui/datasets/combined_dataset/reduce_data_dataset/CVC_train_masks_' + str(i) + '.txt', 'w') as f:
        for mask in eval('train_masks_' + str(i)):
            f.write(mask)
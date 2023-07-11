import os
import shutil



def load_filepaths(dir):
    filepaths = []
    for filename in os.listdir(dir):
        filepaths.append(os.path.join(dir, filename))
    return filepaths

dirs = [
    '/home/zongwei/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/data_C1/images_C1',
    '/home/zongwei/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/data_C2/images_C2',
    '/home/zongwei/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/data_C3/images_C3',
    '/home/zongwei/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/data_C4/images_C4',
    '/home/zongwei/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/data_C5/images_C5',
    '/home/zongwei/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/data_C6/images_C6',
    '/home/zongwei/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/imagesAll_positive',
]

all_images = []

for dir in dirs:
    all_images += load_filepaths(dir)

print(len(all_images))

# randomly select 10% images as validation set

import random

random.shuffle(all_images)

val_images = all_images[:int(len(all_images) * 0.1)]
train_images = all_images[int(len(all_images) * 0.1):]

# sort three lists and save them to txt files
val_images.sort()
train_images.sort()
all_images.sort()

with open('PolyGEN_p+n_val_images.txt', 'w') as f:
    for image in val_images:
        f.write(image + '\n')

with open('PolyGEN_p+n_train_images.txt', 'w') as f:
    for image in train_images:
        f.write(image + '\n')

with open('PolyGEN_p+n_all_images.txt', 'w') as f:
    for image in all_images:
        f.write(image + '\n')




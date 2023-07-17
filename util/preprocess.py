import os
import shutil



def load_filepaths(dir):
    filepaths = []
    for root, dirs, files in os.walk(dir):
        files = [f for f in files if not f[0] == '.']
        dirs[:] = [d for d in dirs if not d[0] == '.']
        for file in files:
            filepaths.append(os.path.join(root, file))
    return filepaths

dirs = [
    '/home/zongwei/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/sequenceData/negativeOnly'
]

all_images = []

for dir in dirs:
    all_images += load_filepaths(dir)

print(len(all_images))

# randomly select 10% images as validation set

import random

random.shuffle(all_images)

val_images = all_images[:int(len(all_images) * 0.02)]
train_images = all_images[int(len(all_images) * 0.02):]

# sort three lists and save them to txt files
val_images.sort()
train_images.sort()
all_images.sort()

with open('PolyGEN_n_sample_images.txt', 'w') as f:
    for image in val_images:
        f.write(image + '\n')

# with open('PolyGEN_p+n_train_images.txt', 'w') as f:
#     for image in train_images:
#         f.write(image + '\n')

with open('PolyGEN_n_all_images.txt', 'w') as f:
    for image in all_images:
        f.write(image + '\n')




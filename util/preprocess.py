import os
import shutil
import random
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
import numpy as np
from tqdm import tqdm


#%%
def load_filepaths(dir):
    filepaths = []
    for root, dirs, files in os.walk(dir):
        files = [f for f in files if not f[0] == '.']
        dirs[:] = [d for d in dirs if not d[0] == '.']
        for file in files:
            filepaths.append(os.path.join(root, file))
    return filepaths

def save_filepaths(dirs, save_path):
    all_images = []
    for dir in dirs:
        all_images += load_filepaths(dir)
    all_images.sort()
    with open(save_path, 'w') as f:
        for image in all_images:
            f.write(image + '\n')

def separate_train_val(image_path, val_num, mask_path=None):
    all_images = []
    with open(image_path, 'r') as f:
        for line in f:
            all_images.append(line.strip())
    selected_idx = random.sample(range(len(all_images)), val_num)
    val_images = [all_images[i] for i in selected_idx]
    train_images = [all_images[i] for i in range(len(all_images)) if i not in selected_idx]
    if mask_path is not None:
        all_masks = []
        with open(mask_path, 'r') as f:
            for line in f:
                all_masks.append(line.strip())
        val_masks = [all_masks[i] for i in selected_idx]
        train_masks = [all_masks[i] for i in range(len(all_masks)) if i not in selected_idx]
        train_images.sort()
        val_images.sort()
        train_masks.sort()
        val_masks.sort()
        return train_images, val_images, train_masks, val_masks
    else:
        train_images.sort()
        val_images.sort()
        return train_images, val_images

def check_image_mask_pair(image_list, mask_list):
    # check if the image and mask are paired
    image_list.sort()
    mask_list.sort()
    image_list = [image.split('/')[-1] for image in image_list]
    mask_list = [mask.split('/')[-1] for mask in mask_list]
    flag = True
    for image in image_list:
        if image not in mask_list:
            print(image)
            flag = False
    for mask in mask_list:
        if mask not in image_list:
            print(mask)
            flag = False
    return flag

def rename_images(dir):
    # rename the images in the dir (remove "mask"). Rename images from "..._mask.jpg" to "....jpg"
    for root, dirs, files in os.walk(dir):
        files = [f for f in files if not f[0] == '.']
        dirs[:] = [d for d in dirs if not d[0] == '.']
        for file in files:
            if file.endswith('_mask.jpg'):
                new_name = file[:-9] + '.jpg'
                os.rename(os.path.join(root, file), os.path.join(root, new_name))
    

#%%
# # write the file names into different txt files
# # dirs_list contains a list of dirs, each element is a list of dirs for one txt file
# if __name__ == '__main__':
#     dirs_list = [
#         # ['/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/0/boxed_images'],
#         # ['/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/1/boxed_images'],
#         # ['/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/2/boxed_images'],
#         # ['/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/3/boxed_images'],
#         # ['/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/4/boxed_images'],
#         # ['/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/5/boxed_images'],
#         # ['/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/6/boxed_images'],
#         # ['/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/7/boxed_images'],
#         # ['/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/8/boxed_images'],
#         # ['/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/mask']
#         # ['/home/zongwei/haorui/ccvl15/haorui/datasets/polyp_bounding_box/PolyGen/boxed_images']
#         # ['/home/zongwei/haorui/ccvl15/haorui/datasets/__2023-08-10T15-00-25_synthetic_polyps_10images/0']
#         # ['/home/zongwei/haorui/ccvl15/haorui/datasets/__2023-08-11synthesized_polyps_merged/images'],
#         # ['/home/zongwei/haorui/ccvl15/haorui/datasets/__2023-08-11synthesized_polyps_merged/masks']
#         # ['/home/yixiao/haorui/ccvl15/haorui/latent-diffusion-Laptop/latent-diffusion/outputs/gen_samples/2023-08-16T14-04-18/0'],
#         # ['/home/yixiao/haorui/ccvl15/haorui/latent-diffusion-Laptop/latent-diffusion/outputs/gen_samples/2023-08-16T14-04-18/mask']
#         [
#         # '/home/yixiao/haorui/ccvl15/haorui/datasets/Kvasir-SEG/masks',
#         # '/home/yixiao/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/data_C1/images_C1',
#         # '/home/yixiao/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/data_C2/images_C2',
#         # '/home/yixiao/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/data_C3/images_C3',
#         # '/home/yixiao/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/data_C4/images_C4',
#         # '/home/yixiao/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/data_C5/images_C5',
#         # '/home/yixiao/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/data_C6/images_C6',
#         '/home/yixiao/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/sequenceData/positive/seq1/images_seq1',
#         '/home/yixiao/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/sequenceData/positive/seq2/images_seq2',
#         '/home/yixiao/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/sequenceData/positive/seq3/images_seq3',
#         '/home/yixiao/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/sequenceData/positive/seq4/images_seq4',
#         '/home/yixiao/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/sequenceData/positive/seq5/images_seq5',
#         '/home/yixiao/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/sequenceData/positive/seq6/images_seq6',
#         '/home/yixiao/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/sequenceData/positive/seq7/images_seq7',
#         '/home/yixiao/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/sequenceData/positive/seq8/images_seq8',
#         '/home/yixiao/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/sequenceData/positive/seq9/images_seq9',
#         '/home/yixiao/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/sequenceData/positive/seq10/images_seq10',
#         '/home/yixiao/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/sequenceData/positive/seq11/images_seq11',
#         '/home/yixiao/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/sequenceData/positive/seq12/images_seq12',
#         '/home/yixiao/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/sequenceData/positive/seq13/images_seq13',
#         '/home/yixiao/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/sequenceData/positive/seq14/images_seq14',
#         '/home/yixiao/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/sequenceData/positive/seq15/images_seq15',
#         '/home/yixiao/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/sequenceData/positive/seq16/images_seq16',
#         '/home/yixiao/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/sequenceData/positive/seq17/images_seq17',
#         '/home/yixiao/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/sequenceData/positive/seq18/images_seq18',
#         '/home/yixiao/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/sequenceData/positive/seq19/images_seq19',
#         '/home/yixiao/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/sequenceData/positive/seq20/images_seq20',
#         '/home/yixiao/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/sequenceData/positive/seq21/images_seq21',
#         '/home/yixiao/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/sequenceData/positive/seq22/images_seq22',
#         '/home/yixiao/haorui/ccvl15/haorui/datasets/PolypGen2021_MultiCenterData_v3/sequenceData/positive/seq23/images_seq23',
#         ],
#         # ['/home/yixiao/haorui/ccvl15/haorui/datasets/CVC-ColonDB/images'],
#         # ['/home/yixiao/haorui/ccvl15/haorui/datasets/CVC-ColonDB/masks']
#     ]

#     for dirs in dirs_list:
#         save_path = dirs[0].split('/')[-1] + '.txt'
#         save_filepaths(dirs, save_path)


#%%
# split train and val
if __name__ == "__main__":
    image_path = '/home/yixiao/haorui/ccvl15/haorui/datasets/combined_dataset/reduce_data_dataset/PolypGEN/PolypGEN_images_except_test_all.txt'
    mask_path = '/home/yixiao/haorui/ccvl15/haorui/datasets/combined_dataset/reduce_data_dataset/PolypGEN/PolypGEN_images_except_test_all.txt'
    train_images, val_images, train_masks, val_masks = separate_train_val(image_path, 700, mask_path)
    with open('/home/yixiao/haorui/ccvl15/haorui/datasets/combined_dataset/reduce_data_dataset/PolypGEN/PolypGEN_images_except_test_train.txt', 'w') as f:
        for image in train_images:
            f.write(image + '\n')
    with open('/home/yixiao/haorui/ccvl15/haorui/datasets/combined_dataset/reduce_data_dataset/PolypGEN/PolypGEN_images_except_test_val.txt', 'w') as f:
        for image in val_images:
            f.write(image + '\n')
    # with open('/home/yixiao/haorui/ccvl15/haorui/datasets/combined_dataset/reduce_data_dataset/PolypGEN/PolypGEN_p_nonempty_masks_train.txt', 'w') as f:
    #     for mask in train_masks:
    #         f.write(mask + '\n')
    # with open('/home/yixiao/haorui/ccvl15/haorui/datasets/combined_dataset/reduce_data_dataset/PolypGEN/PolypGEN_p_nonempty_masks_train_500.txt', 'w') as f:
    #     for mask in val_masks:
    #         f.write(mask + '\n')

#%%
# # check if the image and mask are paired
# if __name__ == '__main__':
#     image_file = '/home/yixiao/haorui/ccvl15/haorui/images.txt'
#     mask_file = '/home/yixiao/haorui/ccvl15/haorui/masks.txt'
#     image_list = []
#     mask_list = []
#     with open(image_file, 'r') as f:
#         for line in f:
#             image_list.append(line.strip())
#     with open(mask_file, 'r') as f:
#         for line in f:
#             mask_list.append(line.strip())
#     print(len(image_list))
#     print(check_image_mask_pair(image_list, mask_list))

#%%
# # randomly select different sets of 10 images and masks
# if __name__ == '__main__':
#     image_file = '/home/yixiao/haorui/ccvl15/haorui/datasets/combined_dataset/reduce_data_dataset/CVC_all_images.txt'
#     mask_file = '/home/yixiao/haorui/ccvl15/haorui/datasets/combined_dataset/reduce_data_dataset/CVC_all_masks.txt'
#     image_list = []
#     mask_list = []
#     with open(image_file, 'r') as f:
#         for line in f:
#             image_list.append(line.strip())
#     with open(mask_file, 'r') as f:
#         for line in f:
#             mask_list.append(line.strip())
#     num_sets = 20
#     num_images = 10
#     for i in range(num_sets):
#         selected_idx = random.sample(range(len(image_list)), num_images)
#         selected_images = [image_list[i] for i in selected_idx]
#         selected_masks = [mask_list[i] for i in selected_idx]
#         selected_images.sort()
#         selected_masks.sort()
#         selected_images = selected_images * 100  # repeat 100 times to make the list longer
#         selected_masks = selected_masks * 100
#         with open(f'/home/yixiao/haorui/ccvl15/haorui/datasets/combined_dataset/reduce_data_dataset/shuffle_10_images/CVC_10images_{i+1:02d}.txt', 'w') as f:
#             for image in selected_images:
#                 f.write(image + '\n')
#         with open(f'/home/yixiao/haorui/ccvl15/haorui/datasets/combined_dataset/reduce_data_dataset/shuffle_10_images/CVC_10masks_{i+1:02d}.txt', 'w') as f:
#             for mask in selected_masks:
#                 f.write(mask + '\n')

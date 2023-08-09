import os
import shutil
import random


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

dirs_list = [
    ['/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/0/boxed_images'],
    ['/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/1/boxed_images'],
    ['/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/2/boxed_images'],
    ['/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/3/boxed_images'],
    ['/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/4/boxed_images'],
    ['/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/5/boxed_images'],
    ['/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/6/boxed_images'],
    ['/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/7/boxed_images'],
    ['/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/8/boxed_images'],
    # ['/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/mask']
    # ['/home/zongwei/haorui/ccvl15/haorui/datasets/polyp_bounding_box/PolyGen/boxed_images']
]

for dirs in dirs_list:
    save_path = dirs[0].split('/')[-2] + '.txt'
    save_filepaths(dirs, save_path)

#%%


# with open('PolyGEN_n_sample_images.txt', 'w') as f:
#     for image in val_images:
#         f.write(image + '\n')

# with open('PolyGEN_p+n_train_images.txt', 'w') as f:
#     for image in train_images:
#         f.write(image + '\n')

# with open('0.txt', 'w') as f:
#     for image in all_images:
#         f.write(image + '\n')




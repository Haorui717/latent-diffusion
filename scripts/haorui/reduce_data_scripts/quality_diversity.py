import glob
import sys, os

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import torch
from prdc import compute_prdc
from PIL import Image
import numpy as np

from ldm.haorui.inception import InceptionV3
from pytorch_lightning import seed_everything

seed_everything(23)
def compute_feature(model, image_list):
    feature_list = []
    for i, image_path in tqdm(enumerate(image_list)):
        image = Image.open(image_path)
        # if min side of the image is less than 299, resize it such that the min side is 299 and keep the aspect ratio
        if min(image.size) < 299:
            ratio = 299 / min(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size)

        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float().unsqueeze(0).cuda()
        feature = model(image)
        feature = feature[0].detach().squeeze().cpu().numpy()
        feature_list.append(feature)
    return np.array(feature_list)

def save_feature():  # save feature of the image through inception_v3
    image_paths = [
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/0/boxed_images',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/1/boxed_images',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/2/boxed_images',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/3/boxed_images',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/4/boxed_images',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/5/boxed_images',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/6/boxed_images',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/7/boxed_images',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/8/boxed_images',
        # '/home/zongwei/haorui/ccvl15/haorui/datasets/polyp_bounding_box/PolyGen/PolyGenBoxed.txt'
    ]
    save_feature_paths = [
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/features',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/features',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/features',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/features',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/features',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/features',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/features',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/features',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/features',
        # '/home/zongwei/haorui/ccvl15/haorui/datasets/polyp_bounding_box/PolyGen'
    ]
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx], resize_input=False, use_fid_inception=True).cuda()
    for i, image_path in enumerate(image_paths):
        # if image path is a file, read the file
        if os.path.isfile(image_path):
            image_list = []
            with open(image_path, 'r') as f:
                for line in f:
                    image_list.append(line.strip())
        else:
            image_list = glob.glob(os.path.join(image_path, '*'))
        feature_list = compute_feature(model, image_list)
        os.makedirs(save_feature_paths[i], exist_ok=True)
        np.save(os.path.join(save_feature_paths[i], f'pretrain_weight_{i}.npy'), feature_list)

def compute_dc(fake_feature_list, real_feature, nearest_k=5):
    result_list = []
    for fake_feature in fake_feature_list:
        res = compute_prdc(real_features=real_feature, fake_features=fake_feature, nearest_k=nearest_k)
        result_list.append(res)
    return result_list

def compute_variance(fake_feature_list):
    result_list = []
    for fake_feature in fake_feature_list:
        # compute variance
        centroid = np.mean(fake_feature, axis=0)
        squared_distances = np.linalg.norm(fake_feature - centroid, axis=1)
        mean_squared_distance = np.mean(squared_distances)
        result_list.append(mean_squared_distance)
    print(result_list)
    return result_list

# if __name__ == '__main__':
#     fake_feature_paths = [
#         '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/features/pretrain_weight_0.npy',
#         '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/features/pretrain_weight_1.npy',
#         '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/features/pretrain_weight_2.npy',
#         '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/features/pretrain_weight_3.npy',
#         '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/features/pretrain_weight_4.npy',
#         '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/features/pretrain_weight_5.npy',
#         '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/features/pretrain_weight_6.npy',
#         '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/features/pretrain_weight_7.npy',
#         '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/features/pretrain_weight_8.npy',
#     ]
#     real_feature_path = '/home/zongwei/haorui/ccvl15/haorui/datasets/polyp_bounding_box/PolyGen/pretrain_weight.npy'
#     real_feature = np.load(real_feature_path)
#     fake_feature_list = [np.load(fake_feature_path) for fake_feature_path in fake_feature_paths]
#     result_list = compute_dc(fake_feature_list, real_feature)
#     for entry in result_list:
#         formatted_dict = {key: f"{value:.4f}" for key, value in entry.items()}
#         print(formatted_dict)
#
#     print(result_list)

if __name__ == '__main__':
    fake_feature_paths = [
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/features/pretrain_weight_0.npy',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/features/pretrain_weight_1.npy',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/features/pretrain_weight_2.npy',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/features/pretrain_weight_3.npy',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/features/pretrain_weight_4.npy',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/features/pretrain_weight_5.npy',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/features/pretrain_weight_6.npy',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/features/pretrain_weight_7.npy',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/single_condition_diversity/2023-08-10T20-48-43/boxed_image/features/pretrain_weight_8.npy',
    ]
    fake_feature_list = [np.load(fake_feature_path) for fake_feature_path in fake_feature_paths]
    result_list = compute_variance(fake_feature_list)



# if __name__ == '__main__':
#     save_feature()
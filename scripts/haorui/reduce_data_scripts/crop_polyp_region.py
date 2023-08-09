import sys, os

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from util.contour_box import crop_bounding_box
import numpy as np
import cv2
from PIL import Image
from models_comparison import save_image_with_suffix

if __name__ == "__main__":
    image_paths = [
        # '/home/zongwei/haorui/ccvl15/haorui/datasets/combined_dataset/PolyGen/PolyGen_images.txt',
        # '/home/zongwei/haorui/ccvl15/haorui/datasets/combined_dataset/Kvasir_images.txt',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/0.txt',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/1.txt',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/2.txt',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/3.txt',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/4.txt',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/5.txt',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/6.txt',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/7.txt',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/8.txt',
    ]
    mask_paths = [
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/mask.txt',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/mask.txt',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/mask.txt',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/mask.txt',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/mask.txt',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/mask.txt',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/mask.txt',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/mask.txt',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/mask.txt',
    ]
    save_dirs = [
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/0',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/1',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/2',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/3',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/4',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/5',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/6',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/7',
        '/home/zongwei/haorui/ccvl15/haorui/latent-diffusion-PC-ccvl23/outputs/reduce_data/gen_samples/2023-08-07T14-07-15/boxed_image/8',
    ]
    for image_path, mask_path, save_dir in zip(image_paths, mask_paths, save_dirs):
        image_list = []
        mask_list = []
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
                save_image_with_suffix(combined, os.path.join(save_dir, "combine"), os.path.basename(image_list[i]))
                save_image_with_suffix(boxed_image, os.path.join(save_dir, "boxed_images"), os.path.basename(image_list[i]))
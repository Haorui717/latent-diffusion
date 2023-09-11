'''
2023.07.17
Customized version of gen_polyp.py. given an image, you need to generate a mask for it,
and then generate a polyp on the mask. You can also specify the noises added to it to check
whether the model can generate diverse and complex polyps.
'''
#%%
import argparse, os, sys, glob
sys.path.append("/home/yixiao/haorui/ccvl15/haorui/latent-diffusion-Local")
import datetime
import shutil

import albumentations
import cv2
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from util.gen_polyp_utils import load_batch
# import matplotlib.pyplot as plt

image_rescaler = albumentations.SmallestMaxSize(max_size=256, interpolation=cv2.INTER_AREA)


#%%
# randomly generate masks and show the masks.
# mask = gen_mask((256, 256))
# # use plt to show the mask
# plt.imshow(mask)
# plt.show()

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, default='configs/polyp.yaml')
    parser.add_argument('--image_path', type=str, help='file containing lines of filenames of images')
    parser.add_argument('--mask_path', type=str, help='file containing lines of filenames of masks', required=False)
    parser.add_argument('--mask_shuffle', action='store_true', help='shuffle the mask list', default=False)
    parser.add_argument('--outdir', type=str, help='output directory', required=True)
    parser.add_argument('--steps', type=int, help='denoise steps', default=200)

    parser.add_argument('--model', type=str, help='model name', required=True)
    parser.add_argument('--random_resize_mask', action='store_true', help='randomly resize the mask', default=False)
    parser.add_argument('--random_crop_mask', action='store_true', help='random crop the mask, otherwise crop from middle', default=False)
    parser.add_argument('--random_crop_image', action='store_true', help='randomly crop image, otherwise crop from middle', default=False)
    parser.add_argument('--debug', action='store_true', help='debug mode, do not log', default=False)

    opt = parser.parse_args()
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if not opt.debug:
        opt.outdir = os.path.join(opt.outdir, now)
        os.makedirs(opt.outdir, exist_ok=True)
        # copy config file to output directory
        shutil.copy(opt.base, os.path.join(opt.outdir, 'config.yaml'))

    image_list = []
    with open(opt.image_path, 'r') as f:
        for line in f:
            image_list.append(line.strip())
    mask_list = []
    if opt.mask_path is not None:
        with open(opt.mask_path, 'r') as f:
            for line in f:
                mask_list.append(line.strip())

    image_list.sort()
    mask_list.sort()

    config = OmegaConf.load(opt.base)
    model = instantiate_from_config(getattr(config, opt.model))
    model = model.cuda()
    model.eval()

    sampler = DDIMSampler(model, opt.steps)

    with torch.no_grad():
        for item in tqdm(range(len(image_list))):
            batch = load_batch(image_list, mask_list, item, opt.mask_shuffle, opt.random_resize_mask, opt.random_crop_mask, opt.random_crop_image)
            mask = batch['mask']
            masked_image = batch['masked_image']

            # sample
            condition = model.cond_stage_model.encode(masked_image)
            m_condition = torch.nn.functional.interpolate(mask, size=condition.shape[-2:])
            c = torch.cat([condition, m_condition], dim=1)

            shape = (c.shape[1] - 1,) + c.shape[2:]
            samples_ddim, _ = sampler.sample(S=opt.steps,
                                             conditioning=c,
                                             batch_size=c.shape[0],
                                             shape=shape,
                                             verbose=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)

            image = torch.clamp((batch["image"] + 1.0) / 2.0,
                                min=0.0, max=1.0)
            masked_image = torch.clamp((masked_image + 1.0) / 2.0,
                                       min=0.0, max=1.0)
            predicted_image = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                                          min=0.0, max=1.0)

            # save image, masked image, and predicted image
            image = image[0].cpu().numpy().transpose(1, 2, 0) * 255.0
            masked_image = masked_image[0].cpu().numpy().transpose(1, 2, 0) * 255.0
            predicted_image = predicted_image[0].cpu().numpy().transpose(1, 2, 0) * 255.0
            # combine image, masked_image, predicted_image horizontally
            combined = np.concatenate([image, masked_image, predicted_image], axis=1)
            # save combined image
            # if file exists, rename it with a suffix _1, _2, ...
            if os.path.exists(os.path.join(opt.outdir, os.path.basename(batch['image_path']))):
                suffix = 1
                while os.path.exists(os.path.join(opt.outdir, os.path.splitext(os.path.basename(batch['image_path']))[0] + '_' + str(suffix) + '.jpg')):
                    suffix += 1
                Image.fromarray(combined.astype(np.uint8)).save(
                    os.path.join(opt.outdir, os.path.splitext(os.path.basename(batch['image_path']))[0] + '_' + str(suffix) + '.jpg'))
            else:
                Image.fromarray(combined.astype(np.uint8)).save(
                    os.path.join(opt.outdir, os.path.basename(batch['image_path'])))

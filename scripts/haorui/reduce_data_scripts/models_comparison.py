import argparse, os, sys, glob
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
import matplotlib.pyplot as plt

from scripts.haorui.gen_polyp_custom import gen_mask, reshape_mask, load_batch

image_rescaler = albumentations.SmallestMaxSize(max_size=256, interpolation=cv2.INTER_AREA)

@torch.no_grad()
def image_quality_compare(models, samplers, image_list, mask_list, opt):
    # compare the image quality of different models
    for item in tqdm(range(len(image_list))):
        batch = load_batch(image_list, mask_list, item, opt.mask_shuffle, opt.random_resize_mask, opt.random_crop_mask, opt.random_crop_image)
        mask = batch['mask']
        masked_image = batch['masked_image']

        condition = models[0].cond_stage_model.encode(masked_image)
        m_condition = torch.nn.functional.interpolate(mask, size=condition.shape[-2:])
        c = torch.cat([condition, m_condition], dim=1)
        shape = (c.shape[1] - 1,) + c.shape[2:]

        # save images
        input_image = batch['image']
        input_image = torch.clamp((input_image + 1.) / 2., 0., 1.)[0].cpu().numpy().transpose(1, 2, 0) * 255.0
        masked_image = torch.clamp((masked_image + 1.) / 2., 0., 1.)[0].cpu().numpy().transpose(1, 2, 0) * 255.0

        l = [input_image, masked_image]
        for i in range(len(models)):
            model = models[i]
            sampler = samplers[i]
            samples_ddim, _ = sampler.sample(S=opt.steps,
                                             conditioning=c,
                                             batch_size=c.shape[0],
                                             shape=shape,
                                             verbose=False)
            x_sample_ddim = model.decode_first_stage(samples_ddim)
            predicted_image = torch.clamp((x_sample_ddim + 1.) / 2., 0., 1.)[0].cpu().numpy().transpose(1, 2, 0) * 255.0
            l.append(predicted_image)
        combined = np.concatenate(l, axis=1)

        # save combined image
        # if file exists, rename it with a suffix _1, _2, ...
        if os.path.exists(os.path.join(opt.outdir, os.path.basename(batch['image_path']))):
            suffix = 1
            while os.path.exists(os.path.join(opt.outdir,
                                              os.path.splitext(os.path.basename(batch['image_path']))[0] + '_' + str(
                                                      suffix) + '.jpg')):
                suffix += 1
            Image.fromarray(combined.astype(np.uint8)).save(
                os.path.join(opt.outdir,
                             os.path.splitext(os.path.basename(batch['image_path']))[0] + '_' + str(suffix) + '.jpg'))
        else:
            Image.fromarray(combined.astype(np.uint8)).save(
                os.path.join(opt.outdir, os.path.basename(batch['image_path'])))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, default='configs/polyp.yaml')
    parser.add_argument('--image_path', type=str, help='file containing lines of filenames of images')
    parser.add_argument('--mask_path', type=str, help='file containing lines of filenames of masks', required=False)
    parser.add_argument('--mask_shuffle', action='store_true', help='shuffle the mask list', default=False)
    parser.add_argument('--outdir', type=str, help='output directory', required=True)
    parser.add_argument('--steps', type=int, help='denoise steps', default=200)

    parser.add_argument('--random_resize_mask', action='store_true', help='randomly resize the mask', default=False)
    parser.add_argument('--random_crop_mask', action='store_true',
                        help='random crop the mask, otherwise crop from middle', default=False)
    parser.add_argument('--random_crop_image', action='store_true',
                        help='randomly crop image, otherwise crop from middle', default=False)
    parser.add_argument('--debug', action='store_true', help='debug mode, do not log', default=False)
    opt = parser.parse_args()

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    opt.outdir = os.path.join(opt.outdir, now)
    os.makedirs(opt.outdir, exist_ok=True)
    # copy config file to output directory
    merged_config = OmegaConf.merge(OmegaConf.load(opt.base), OmegaConf.create({"opt":vars(opt)}))
    OmegaConf.save(merged_config, os.path.join(opt.outdir, 'config.yaml'))

    for model in merged_config.models:
        model.model.params.ckpt_path = model.ckpt_path

    image_list = []
    with open(opt.image_path, 'r') as f:
        for line in f:
            image_list.append(line.strip())
    mask_list = []
    if opt.mask_path is not None:
        with open(opt.mask_path, 'r') as f:
            for line in f:
                mask_list.append(line.strip())

    models =[instantiate_from_config(i.model).cuda().eval() for i in merged_config.models]
    samplers = [DDIMSampler(model, opt.steps) for model in models]
    image_quality_compare(models, samplers, image_list, mask_list, opt)
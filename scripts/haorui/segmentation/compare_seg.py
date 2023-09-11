# Visualize some segmentation results
# Model trained on real images performs better than model trained on synthetic images
# Need to visualize the results and find the reason.
import sys
sys.path.append('/home/yixiao/haorui/ccvl15/haorui/latent-diffusion-Local')
import torch
import numpy as np

import argparse, os, sys, glob
import concurrent
import datetime
import shutil
from concurrent.futures import ThreadPoolExecutor
import albumentations
import cv2
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from torchmetrics import Dice
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

from scripts.haorui.gen_polyp_custom import load_batch
from scripts.haorui.reduce_data_scripts.models_comparison import save_image_with_suffix

@torch.no_grad()
def visualize_typical_samples(models, image_list, mask_list, opt, 
                              nums=20, threshold=0.1):
    '''
    Visualize typical samples where the two models have different dice results.
    @param models: a list of two models
    @param nums: number of samples to visualize
    '''
    os.makedirs(os.path.join(opt.outdir, 'input'), exist_ok=True)
    os.makedirs(os.path.join(opt.outdir, 'mask_ground_truth'), exist_ok=True)
    os.makedirs(os.path.join(opt.outdir, 'mask_model_0'), exist_ok=True)
    os.makedirs(os.path.join(opt.outdir, 'mask_model_1'), exist_ok=True)
    os.makedirs(os.path.join(opt.outdir, 'combined'), exist_ok=True)

    dice = Dice(num_classes=2, reduction='none').cuda()
    for model in models:
        model.eval()

    while nums > 0:
        item = np.random.randint(len(image_list))
        batch = load_batch(image_list, mask_list, item, opt.mask_shuffle,
                        opt.random_resize_mask, opt.random_crop_mask, opt.random_crop_image)
        input = batch['image']
        mask = batch['mask']
        mask = (mask + 1) / 2  # [-1, 1] -> [0, 1]
        mask = mask.to(torch.int32)
        dice_list = []
        pred_list = []
        for model in models:
            pred = model(input)
            pred[pred > 0] = 1
            pred[pred <= 0] = 0
            pred = pred.to(torch.int32)
            dice_list.append(dice(pred, mask))
            pred_list.append(pred)
        if dice_list[0] - dice_list[1] > threshold:
            nums -= 1
            input = torch.clamp((input[0] + 1) / 2., 0., 1.).cpu().numpy().transpose(1, 2, 0) * 255
            input = input.astype(np.uint8)
            mask = mask[0].cpu().numpy().transpose(1, 2, 0) * 255
            mask = mask.astype(np.uint8)
            mask = np.repeat(mask, 3, axis=2)
            pred_list[0] = pred_list[0][0].cpu().numpy().transpose(1, 2, 0) * 255
            pred_list[1] = pred_list[1][0].cpu().numpy().transpose(1, 2, 0) * 255
            pred_list[0] = np.repeat(pred_list[0], 3, axis=2)
            pred_list[1] = np.repeat(pred_list[1], 3, axis=2)
            pred_list[0] = pred_list[0].astype(np.uint8)
            pred_list[1] = pred_list[1].astype(np.uint8)
            combined = np.concatenate([input, mask, pred_list[0], pred_list[1]], axis=1).astype(np.uint8)
            save_image_with_suffix(combined, os.path.join(opt.outdir, 'combined'), os.path.basename(batch['image_path']))
            save_image_with_suffix(input, os.path.join(opt.outdir, 'input'), os.path.basename(batch['image_path']))
            save_image_with_suffix(mask, os.path.join(opt.outdir, 'mask_ground_truth'), os.path.basename(batch['image_path']))
            save_image_with_suffix(pred_list[0], os.path.join(opt.outdir, 'mask_model_0'), os.path.basename(batch['image_path']))
            save_image_with_suffix(pred_list[1], os.path.join(opt.outdir, 'mask_model_1'), os.path.basename(batch['image_path']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, default='configs/polyp.yaml')
    parser.add_argument('--image_path', type=str, help='file containing lines of filenames of images')
    parser.add_argument('--mask_path', type=str, help='file containing lines of filenames of masks', required=False)
    parser.add_argument('--mask_shuffle', action='store_true', help='shuffle the mask list', default=False)
    parser.add_argument('--batch_size', type=int, help='batch size', default=4)
    parser.add_argument('--outdir', type=str, help='output directory', required=True)
    parser.add_argument('--steps', type=int, help='denoise steps', default=200)
    parser.add_argument('--nums', type=int, help='number of images to generate', default=None)
    parser.add_argument('--random_image', action='store_true', help='randomly select image', default=False)

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
    visualize_typical_samples(models, image_list, mask_list, opt, nums=20, threshold=0.1)
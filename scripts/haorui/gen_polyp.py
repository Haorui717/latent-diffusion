import argparse, os, sys, glob

import albumentations
import cv2
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from main import instantiate_from_config
from torch.utils.data import DataLoader
from ldm.models.diffusion.ddim import DDIMSampler

image_rescaler = albumentations.SmallestMaxSize(max_size=256, interpolation=cv2.INTER_AREA)
def load_batch(root_path, image_list, mask_list, item):
    # load image and mask
    image_path = os.path.join(root_path, 'images',  image_list[item])
    mask_path = os.path.join(root_path, 'masks', mask_list[item])
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    image = np.array(image).astype(np.float32) / 255.0
    mask = np.array(mask).astype(np.float32) / 255.0

    # crop image and mask into square
    min_side_len = min(image.shape[:2])
    crop_side_len = min_side_len
    h, w = image.shape[:2]
    top = np.random.randint(0, h - crop_side_len + 1)
    left = np.random.randint(0, w - crop_side_len + 1)
    image = image[top:top + crop_side_len, left:left + crop_side_len]
    mask = mask[top:top + crop_side_len, left:left + crop_side_len]

    # resize image and mask to (256 x size)
    image = image_rescaler(image=image)['image']
    mask = image_rescaler(image=mask)['image'][..., None]

    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    masked_image = (1 - mask) * image
    masked_image = masked_image * 2.0 - 1.0
    image = image * 2.0 - 1.0
    mask = mask * 2.0 - 1.0

    batch = dict()
    batch['image'] = image.transpose(2,0,1)
    batch['mask'] = mask.transpose(2,0,1)
    batch['masked_image'] = masked_image.transpose(2,0,1)
    for k in batch:
        batch[k] = torch.from_numpy(batch[k]).float().cuda()
        batch[k] = batch[k].unsqueeze(0)
    batch['image_path'] = image_list[item]
    batch['mask_path'] = mask_list[item]

    return batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, nargs='?', help='config file path')
    parser.add_argument('--image_path', type=str, nargs='?', help='file containing lines of filenames of images')
    parser.add_argument('--mask_path', type=str, nargs='?', help='file containing lines of filenames of masks')
    parser.add_argument('--root_path', type=str, nargs='?', help='path to root data dir')
    parser.add_argument('--outdir', type=str, nargs='?', help='dir to write results to')
    parser.add_argument('--steps', type=int, default=200, help='number of ddim sampling steps')
    opt = parser.parse_args()
    os.makedirs(opt.outdir, exist_ok=True)

    image_list = []
    mask_list = []
    with open(opt.image_path, 'r') as f:
        for line in f:
            image_list.append(line.strip())
    with open(opt.mask_path, 'r') as f:
        for line in f:
            mask_list.append(line.strip())

    images = sorted(image_list)
    masks = sorted(mask_list)

    config = OmegaConf.load(opt.base)
    model = instantiate_from_config(config.model)
    model = model.cuda()
    model.eval()  # whether to use eval()?

    sampler = DDIMSampler(model, opt.steps)

    with torch.no_grad():
        with model.ema_scope():
            for item in tqdm(range(len(image_list))):
                batch = load_batch(opt.root_path, image_list, mask_list, item)
                mask = batch['mask']
                masked_image = batch['masked_image']

                # sample
                condition = model.cond_stage_model.encode(masked_image)
                m_condition = torch.nn.functional.interpolate(mask, size=condition.shape[-2:])
                c = torch.cat([condition, m_condition], dim=1)

                shape = (c.shape[1]-1,)+c.shape[2:]
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

                # save image, masked_image, predicted_image
                image = image[0].cpu().numpy().transpose(1,2,0) * 255.0
                masked_image = masked_image[0].cpu().numpy().transpose(1,2,0) * 255.0
                predicted_image = predicted_image[0].cpu().numpy().transpose(1,2,0) * 255.0
                # combine image, masked_image, predicted_image horizontally
                combined = np.concatenate([image, masked_image, predicted_image], axis=1)
                # save combined image
                Image.fromarray(combined.astype(np.uint8)).save(os.path.join(opt.outdir, os.path.basename(batch['image_path'])))

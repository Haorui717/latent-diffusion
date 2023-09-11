import argparse, os

import albumentations
import cv2
import numpy as np
import torch
from PIL import Image
from main import instantiate_from_config
from omegaconf import OmegaConf

image_rescaler = albumentations.SmallestMaxSize(max_size=256, interpolation=cv2.INTER_AREA)

def load_batch(image_list, item):
    # load image
    image_path = image_list[item]
    image = Image.open(image_path).convert('RGB')
    image = np.array(image).astype(np.float32) / 255.0

    # crop image into square
    min_side_len = min(image.shape[:2])
    crop_side_len = min_side_len
    image = image[:crop_side_len, :crop_side_len]

    # resize image to (256 x size)
    image = image_rescaler(image=image)['image']

    # convert to [-1, 1]
    image = image * 2.0 - 1.0

    batch = dict()
    batch['image'] = image.transpose(2,0,1)
    batch['image'] = torch.from_numpy(batch['image']).float().cuda()
    batch['image'] = batch['image'].unsqueeze(0)
    batch['image_path'] = image_list[item]

    return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, nargs='?', help='config file path')
    parser.add_argument('--image_path', type=str, nargs='?', help='file containing lines of filenames of images')
    parser.add_argument('--outdir', type=str, nargs='?', help='dir to write results to')

    opt = parser.parse_args()
    os.makedirs(opt.outdir, exist_ok=True)

    image_list = []
    with open(opt.image_path, 'r') as f:
        for line in f:
            image_list.append(line.strip())
    images = sorted(image_list)

    config = OmegaConf.load(opt.base)
    model_pretrained = instantiate_from_config(config.model_pretrained)
    model_pretrained = model_pretrained.cuda()
    model_pretrained.eval()

    model_fromscratch = instantiate_from_config(config.model_fromscratch)
    model_fromscratch = model_fromscratch.cuda()
    model_fromscratch.eval()

    with torch.no_grad():
        for i in range(len(images)):
            batch = load_batch(images, i)
            image = batch['image']
            image_path = batch['image_path']
            image_name = os.path.basename(image_path)

            # run model
            z = model_pretrained.encode(image)
            recon_pretrained = model_pretrained.decode(z)

            z = model_fromscratch.encode(image)
            recon_fromscratch = model_fromscratch.decode(z)

            # save results
            image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
            recon_pretrained = torch.clamp((recon_pretrained + 1.0) / 2.0, min=0.0, max=1.0)
            recon_fromscratch = torch.clamp((recon_fromscratch + 1.0) / 2.0, min=0.0, max=1.0)
            image = image[0].cpu().numpy().transpose(1,2,0) * 255.0
            recon_pretrained = recon_pretrained[0].cpu().numpy().transpose(1,2,0) * 255.0
            recon_fromscratch = recon_fromscratch[0].cpu().numpy().transpose(1,2,0) * 255.0

            combined = np.concatenate([image, recon_pretrained, recon_fromscratch], axis=1)
            combined = Image.fromarray(combined.astype(np.uint8))
            combined.save(os.path.join(opt.outdir, image_name))

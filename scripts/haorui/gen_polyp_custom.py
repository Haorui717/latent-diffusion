'''
2023.07.17
Customized version of gen_polyp.py. given an image, you need to generate a mask for it,
and then generate a polyp on the mask. You can also specify the noises added to it to check
whether the model can generate diverse and complex polyps.
'''
#%%
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
#%%
image_rescaler = albumentations.SmallestMaxSize(max_size=256, interpolation=cv2.INTER_AREA)

def gen_mask(shape, start_x=None, start_y=None):
    mask = np.zeros(shape)
    prob = 1.0
    prob_decay = np.random.randint(1, 5) * 1e-5
    if start_x is None:
        start_x = np.random.randint(0, shape[0])
    if start_y is None:
        start_y = np.random.randint(0, shape[1])
    # use BFS to generate a random mask
    queue = []
    queue.append((start_x, start_y))
    while len(queue) > 0:
        x, y = queue.pop(0)
        if mask[x, y] == 0 and np.random.rand() < prob:
            mask[x, y] = 1
            if x > 0:
                queue.append((x - 1, y))
            if x < shape[0] - 1:
                queue.append((x + 1, y))
            if y > 0:
                queue.append((x, y - 1))
            if y < shape[1] - 1:
                queue.append((x, y + 1))
        prob *= (1 - prob_decay)

    # there may be some small holes in the mask, fill them
    kernel = np.ones((11, 11), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def reshape_mask(mask, shape, random_resize=False, random_crop=False):
    # mask's shape should be the same as shape
    # randomly pad the mask if it is smaller than shape
    # if mask.shape[0] < shape[0]:
    #     pad = shape[0] - mask.shape[0]
    #     start_x = np.random.randint(0, pad)
    #     mask = np.pad(mask, ((start_x, pad - start_x), (0, 0)), 'constant', constant_values=0)
    #
    # if mask.shape[1] < shape[1]:
    #     pad = shape[1] - mask.shape[1]
    #     start_y = np.random.randint(0, pad)
    #     mask = np.pad(mask, ((0, 0), (start_y, pad - start_y)), 'constant', constant_values=0)

    # if mask's shape is smaller than shape, then rescale it to shape with nearest interpolation
    if mask.shape[0] < shape[0]:
        mask = cv2.resize(mask, (int(shape[0] * mask.shape[1] / mask.shape[0]), shape[0]), interpolation=cv2.INTER_NEAREST)

    if mask.shape[1] < shape[1]:
        mask = cv2.resize(mask, (shape[1], int(shape[1] * mask.shape[0] / mask.shape[1])), interpolation=cv2.INTER_NEAREST)

    if random_resize:  # randomly resize the mask to bigger shape [1 ~ 2] times
        scale = np.random.rand() + 1
        mask = cv2.resize(mask, (int(mask.shape[1] * scale), int(mask.shape[0] * scale)), interpolation=cv2.INTER_NEAREST)

    # crop the mask if it is larger than shape
    if mask.shape[0] > shape[0]:
        start_x = np.random.randint(0, mask.shape[0] - shape[0]) if \
            random_crop else (mask.shape[0] - shape[0]) // 2
        mask = mask[start_x:start_x+shape[0], :]
    if mask.shape[1] > shape[1]:
        start_y = np.random.randint(0, mask.shape[1] - shape[1]) if \
            random_crop else (mask.shape[1] - shape[1]) // 2
        mask = mask[:, start_y:start_y+shape[1]]

    return mask




#%%
# randomly generate masks and show the masks.
# mask = gen_mask((256, 256))
# # use plt to show the mask
# plt.imshow(mask)
# plt.show()

#%%

def load_batch(image_list, mask_list, item, mask_shuffle=True, random_resize=False, random_crop=False, random_crop_image=False):
    # load image and mask
    image_path = image_list[item]
    image = Image.open(image_path).convert('RGB')
    try:
        if mask_shuffle:
            mask_path = mask_list[np.random.randint(0, len(mask_list))]
        else:
            mask_path = mask_list[item]
        mask = Image.open(mask_path).convert('L')
    except:
        # mask_list is empty, generate a random mask of the same size as image
        mask = Image.fromarray(gen_mask(image.size))
    image = np.array(image).astype(np.float32) / 255.0
    mask = np.array(mask).astype(np.float32) / 255.0

    # crop image and mask to make them square
    if random_crop_image:
        min_side_len = min(image.shape[:2])
        crop_side_len = min_side_len * np.random.uniform(0.5, 1.0, size=None)
        crop_side_len = int(crop_side_len)
        start_x = np.random.randint(0, image.shape[0] - crop_side_len)
        start_y = np.random.randint(0, image.shape[1] - crop_side_len)
        image = image[start_x:start_x+crop_side_len, start_y:start_y+crop_side_len]
    else:# crop from the middle
        min_side_len = min(image.shape[:2])
        crop_side_len = min_side_len
        image = image[(image.shape[0] - crop_side_len) // 2:(image.shape[0] + crop_side_len) // 2, (image.shape[1] - crop_side_len) // 2:(image.shape[1] + crop_side_len) // 2]

    mask = reshape_mask(mask, image.shape[:2], random_resize=random_resize, random_crop=random_crop)

    # resize image and mask
    image = image_rescaler(image=image)['image']
    mask = image_rescaler(image=mask)['image'][..., None]

    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    masked_image = (1 - mask) * image
    masked_image = masked_image * 2.0 - 1.0
    image = image * 2.0 - 1.0
    mask = mask * 2.0 - 1.0

    batch = dict()
    batch['image'] = image.transpose(2, 0, 1)
    batch['mask'] = mask.transpose(2, 0, 1)
    batch['masked_image'] = masked_image.transpose(2, 0, 1)
    for k in batch:
        batch[k] = torch.from_numpy(batch[k]).float().cuda()
        batch[k] = batch[k].unsqueeze(0)
    batch['image_path'] = image_list[item]
    batch['mask_path'] = mask_list[item]

    return batch

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

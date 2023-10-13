import argparse, os, sys, glob
import concurrent

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
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
from torch.utils.data import DataLoader
# from torchmetrics import Dice, Recall, Precision, JaccardIndex
from monai.metrics import DiceMetric, ConfusionMatrixMetric, MeanIoU
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from data.haorui.CVC_Clinic import CVC_Clinic_Reconstruction
from data.haorui.synthetic_polyp import Random_Mask_Dataset
# import matplotlib.pyplot as plt

from util.gen_polyp_utils import reshape_mask, load_batch

image_rescaler = albumentations.SmallestMaxSize(max_size=256, interpolation=cv2.INTER_AREA)

def save_image_with_suffix(image, dir, filename):
    # # if file exists, rename it with a suffix _1, _2, ...
    os.makedirs(dir, exist_ok=True)
    if os.path.exists(os.path.join(dir, filename)):
        suffix = 1
        while os.path.exists(os.path.join(dir,
                                          os.path.splitext(filename)[0] + '_' + str(
                                              suffix) + os.path.splitext(filename)[1])):
            suffix += 1
        Image.fromarray(image).save(
            os.path.join(dir,
                         os.path.splitext(filename)[0] + '_' + str(suffix) + os.path.splitext(filename)[1]))
    else:
        Image.fromarray(image).save(
            os.path.join(dir, filename))

# @torch.no_grad()
# def image_quality_compare(models, samplers, image_list, mask_list, opt):
#     # compare the image quality of different models
#     for item in tqdm(range(0, len(image_list), opt.batch_size)):
#         # load batch with batch size of opt.batch_size
#         batches = []
#         for i in range(opt.batch_size):
#             tmp = load_batch(image_list, mask_list, min(item+i, len(image_list)-1), opt.mask_shuffle, opt.random_resize_mask, opt.random_crop_mask, opt.random_crop_image)
#             batches.append(tmp)
#         batch = dict()
#         for key in batches[0].keys():
#             batch[key] = torch.cat([b[key] for b in batches], dim=0) if isinstance(batches[0][key], torch.Tensor) \
#                                 else [b[key] for b in batches]
#         mask = batch['mask']
#         masked_image = batch['masked_image']

#         condition = models[0].cond_stage_model.encode(masked_image)
#         m_condition = torch.nn.functional.interpolate(mask, size=condition.shape[-2:])
#         c = torch.cat([condition, m_condition], dim=1)
#         shape = (c.shape[1] - 1,) + c.shape[2:]

#         input_image = batch['image']  # (B, 3, H, W)
#         input_image = torch.clamp((input_image + 1.) / 2., 0., 1.).cpu().numpy().transpose(0, 2, 3, 1) * 255.0
#         masked_image = torch.clamp((masked_image + 1.) / 2., 0., 1.).cpu().numpy().transpose(0, 2, 3, 1) * 255.0

#         l = [input_image, masked_image]
#         for i in range(len(models)):
#             model = models[i]
#             sampler = samplers[i]
#             samples_ddim, _ = sampler.sample(S=opt.steps,
#                                              conditioning=c,
#                                              batch_size=c.shape[0],
#                                              shape=shape,
#                                              verbose=False)
#             x_sample_ddim = model.decode_first_stage(samples_ddim)
#             predicted_image = torch.clamp((x_sample_ddim + 1.) / 2., 0., 1.).cpu().numpy().transpose(0, 2, 3, 1) * 255.0
#             l.append(predicted_image)

#         for i in range(len(input_image)):  # for each result in the batch
#             combined = np.concatenate([img[i] for img in l], axis=1).astype(np.uint8)

#             # save combined image
#             # if file exists, rename it with a suffix _1, _2, ...
#             save_image_with_suffix(combined, os.path.join(opt.outdir, 'combined'), os.path.basename(batch['image_path'][i]))



@torch.no_grad()
def gen_samples(models, samplers, image_list, mask_list, opt, nums=None):
    nums = len(image_list) if nums is None else min(nums, len(image_list))

    # def load_single_batch(index, image_list, mask_list, random_image, opt):
    #     if random_image:
    #         return load_batch(image_list, mask_list, np.random.randint(len(image_list)), opt.mask_shuffle,
    #                           opt.random_resize_mask, opt.random_crop_mask, opt.random_crop_image)
    #     else:
    #         return load_batch(image_list, mask_list, min(index, len(image_list) - 1), opt.mask_shuffle,
    #                           opt.random_resize_mask, opt.random_crop_mask, opt.random_crop_image)

    # make directories for each model
    os.makedirs(os.path.join(opt.outdir, 'input'), exist_ok=True)
    os.makedirs(os.path.join(opt.outdir, 'masked_image'), exist_ok=True)
    os.makedirs(os.path.join(opt.outdir, 'combined'), exist_ok=True)
    os.makedirs(os.path.join(opt.outdir, 'mask'), exist_ok=True)
    for i in range(len(models)):
        os.makedirs(os.path.join(opt.outdir, str(i)), exist_ok=True)

    # dataset = Random_Mask_Dataset(opt.image_path, opt.mask_path, len=nums, transpose=True)
    dataset = CVC_Clinic_Reconstruction(opt.image_path, opt.mask_path, transpose=True)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4, drop_last=False)
    for batch in tqdm(dataloader):
        # future_to_index = {executor.submit(load_single_batch, item + i, image_list, mask_list, random_image, opt): i
        #                     for i in range(opt.batch_size)}
        # batches = []
        # for future in concurrent.futures.as_completed(future_to_index):
        #     batches.append(future.result())
        # batch = dict()
        # for key in batches[0].keys():
        #     batch[key] = torch.cat([b[key] for b in batches], dim=0) if isinstance(batches[0][key], torch.Tensor) \
        #         else [b[key] for b in batches]

        input_image = batch['image'].cuda()  # (B, 3, H, W)
        mask = batch['mask'].cuda()  # (B, 1, H, W)
        masked_image = batch['masked_image'].cuda()  # (B, 3, H, W)

        condition = models[0].cond_stage_model.encode(masked_image)
        m_condition = torch.nn.functional.interpolate(mask, size=condition.shape[-2:])
        c = torch.cat([condition, m_condition], dim=1)
        shape = (c.shape[1] - 1,) + c.shape[2:]

        input_image  = torch.clamp((input_image + 1.) / 2., 0., 1.).cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        input_image  = input_image.astype(np.uint8)
        masked_image = torch.clamp((masked_image + 1.) / 2., 0., 1.).cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        masked_image = masked_image.astype(np.uint8)
        mask         = torch.clamp((mask + 1.) / 2., 0., 1.).cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        mask         = np.repeat(mask, 3, axis=3)
        mask         = mask.astype(np.uint8)

        l = [input_image, masked_image]
        for i in range(len(models)):
            model = models[i]
            sampler = samplers[i]
            samples_ddim, _ = sampler.sample(S=opt.steps[i],
                                                conditioning=c,
                                                batch_size=c.shape[0],
                                                shape=shape,
                                                verbose=False)
            x_sample_ddim = model.decode_first_stage(samples_ddim)
            predicted_image = torch.clamp((x_sample_ddim + 1.) / 2., 0., 1.).cpu().numpy().transpose(0, 2, 3, 1) * 255.0
            predicted_image = predicted_image.astype(np.uint8)
            l.append(predicted_image)

        # save images
        for i in range(len(input_image)):  # for each result in the batch
            combined = np.concatenate([img[i] for img in l], axis=1).astype(np.uint8)

            # save images
            save_image_with_suffix(input_image[i], os.path.join(opt.outdir, 'input'), os.path.basename(batch['image_path'][i]))
            save_image_with_suffix(masked_image[i], os.path.join(opt.outdir, 'masked_image'), os.path.basename(batch['image_path'][i]))
            save_image_with_suffix(combined, os.path.join(opt.outdir, 'combined'), os.path.basename(batch['image_path'][i]))
            save_image_with_suffix(mask[i], os.path.join(opt.outdir, 'mask'), os.path.basename(batch['image_path'][i]))
            for j in range(len(models)):
                save_image_with_suffix(l[j+2][i], os.path.join(opt.outdir, str(j)), os.path.basename(batch['image_path'][i]))

@torch.no_grad()
def dice_compare(models, image_list, mask_list, opt):
    # def load_single_batch(index, image_list, mask_list, random_image, opt):
    #     if random_image:
    #         return load_batch(image_list, mask_list, np.random.randint(len(image_list)), opt.mask_shuffle,
    #                           opt.random_resize_mask, opt.random_crop_mask, opt.random_crop_image)
    #     else:
    #         return load_batch(image_list, mask_list, min(index, len(image_list) - 1), opt.mask_shuffle,
    #                           opt.random_resize_mask, opt.random_crop_mask, opt.random_crop_image)

    dice_res, precision_res, recall_res, IoU_res = [], [], [], []
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    miou_metric = MeanIoU(include_background=True, reduction="mean")
    precision_metric = ConfusionMatrixMetric(include_background=True, metric_name="precision", reduction="mean")
    recall_metric = ConfusionMatrixMetric(include_background=True, metric_name="recall", reduction="mean")
    # confusion_matrix_metric = ConfusionMatrixMetric(include_background=True, metric_name="all", reduction="mean")
    

    for idx, model in enumerate(models):
        model.eval()
        dice_metric.reset()
        miou_metric.reset()
        dataset = CVC_Clinic_Reconstruction(opt.image_path, opt.mask_path, transpose=True, random_crop=False)
        dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4, drop_last=False)
        for batch in tqdm(dataloader):

            input_image = batch['image'].cuda()
            mask = batch['mask'].cuda()
            mask = (mask + 1) / 2  # convert to 0-1
            mask = mask.to(torch.int32)
            pred = model(input_image)
            pred[pred > 0] = 1
            pred[pred <= 0] = 0
            pred = pred.to(torch.int32)

            dice_val = dice_metric(y_pred=pred, y=mask)
            miou_val = miou_metric(y_pred=pred, y=mask)
            precision_val = precision_metric(y_pred=pred, y=mask)  # [tp, fp, tn, fn]
            recall_val = recall_metric(y_pred=pred, y=mask)  # [tp, fp, tn, fn]
            
            # save and compare predicted results and masks
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
            pred = pred.astype(np.uint8)
            mask = mask.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
            mask = mask.astype(np.uint8)
            if opt.save_image:
                for i in range(len(input_image)):  # for each result in the batch
                    combined = np.concatenate([(input_image[i].cpu().numpy().transpose(1, 2, 0) + 1) / 2 * 255.0,
                                                mask[i].repeat(3, axis=2),
                                                pred[i].repeat(3, axis=2)], axis=1).astype(np.uint8)
                    save_image_with_suffix(combined, os.path.join(opt.outdir, 'combined', str(idx)), 
                                        f"{os.path.splitext(os.path.basename(batch['image_path'][i]))[0]}_dice_{dice_val[i].item():.4f}_miou_{miou_val[i].item():.4f}_precision_{(precision_val[i][0][0] / (precision_val[i][0][0] + precision_val[i][0][1])).item():.4f}_recall_{(recall_val[i][0][0] / (recall_val[i][0][0] + recall_val[i][0][3])).item():.4f}.png")

        dice_res.append(dice_metric.aggregate()[0].item())
        precision_res.append(precision_metric.aggregate()[0].item())
        recall_res.append(recall_metric.aggregate()[0].item())
        IoU_res.append(miou_metric.aggregate().item())
    with open(os.path.join(opt.outdir, 'results.txt'), 'w') as f:
        f.write('dice:\n')
        for i in range(len(models)):
            f.write(f'{i}: {dice_res[i]}\n')
        f.write('precision:\n')
        for i in range(len(models)):
            f.write(f'{i}: {precision_res[i]}\n')
        f.write('recall:\n')
        for i in range(len(models)):
            f.write(f'{i}: {recall_res[i]}\n')
        f.write('IoU:\n')
        for i in range(len(models)):
            f.write(f'{i}: {IoU_res[i]}\n')

    print("dice", dice_res)
    print("precision", precision_res)
    print("recall", recall_res)
    print("IoU", IoU_res)
    return dice_res, precision_res, recall_res, IoU_res



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, default='configs/polyp.yaml')
    parser.add_argument('--image_path', type=str, help='file containing lines of filenames of images')
    parser.add_argument('--mask_path', type=str, help='file containing lines of filenames of masks', required=False)
    # parser.add_argument('--mask_shuffle', action='store_true', help='shuffle the mask list', default=False)
    parser.add_argument('--batch_size', type=int, help='batch size', default=4)
    parser.add_argument('--outdir', type=str, help='output directory', required=True)
    parser.add_argument('--steps',nargs='*', type=int, help='denoise steps', default=[200])
    parser.add_argument('--nums', type=int, help='number of images to generate', default=None)
    parser.add_argument('--save_image', action='store_true', help='dice compare whether to save images', default=False)
    # parser.add_argument('--random_image', action='store_true', help='randomly select image', default=False)

    # parser.add_argument('--random_resize_mask', action='store_true', help='randomly resize the mask', default=False)
    # parser.add_argument('--random_crop_mask', action='store_true',
    #                     help='random crop the mask, otherwise crop from middle', default=False)
    # parser.add_argument('--random_crop_image', action='store_true',
    #                     help='randomly crop image, otherwise crop from middle', default=False)
    parser.add_argument('--debug', action='store_true', help='debug mode, do not log', default=False)
    opt = parser.parse_args()

    # create output directory
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    opt.outdir = os.path.join(opt.outdir, now)
    os.makedirs(opt.outdir, exist_ok=True)

    # copy config file to output directory
    merged_config = OmegaConf.merge(OmegaConf.load(opt.base), OmegaConf.create({"opt":vars(opt)}))
    OmegaConf.save(merged_config, os.path.join(opt.outdir, 'config.yaml'))

    # broadcast opt.steps to each model
    if len(opt.steps) == 1:
        opt.steps = opt.steps * len(merged_config.models)
    assert len(opt.steps) == len(merged_config.models), "number of steps must match number of models"
    
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
    # samplers = [DDIMSampler(model, step) for model, step in zip(models, opt.steps)]
    # image_quality_compare(models, samplers, image_list, mask_list, opt)
    # gen_samples(models, samplers, image_list, mask_list, opt, nums=opt.nums)
    dice_compare(models, image_list, mask_list, opt)
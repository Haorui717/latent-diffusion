import albumentations
import cv2
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch

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

def load_batch(image_list, mask_list, item, mask_shuffle=True, random_resize=False, random_crop=False, random_crop_image=False):
    # load image and mask
    image_path = image_list[item]
    image = Image.open(image_path).convert('RGB')
    try:
        if mask_shuffle:
            while True:
                mask_path = mask_list[np.random.randint(0, len(mask_list))]
                mask = Image.open(mask_path).convert('L')
                if check_polyp_size(mask):
                    break
        else:
            mask_path = mask_list[item]
            mask = Image.open(mask_path).convert('L')
    except:
        # mask_list is empty, generate a random mask of the same size as image
        print('mask_list is empty, generate a random mask of the same size as image')
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
    else:# crop the image at 2/3 of the longer side
        min_side_len = min(image.shape[:2])
        crop_side_len = min_side_len
        if image.shape[0] > image.shape[1]:
            start_x = min(image.shape[0] // 3 * 2 - crop_side_len // 2, image.shape[0] - crop_side_len)
            start_y = 0
        else:
            start_x = 0
            start_y = min(image.shape[1] // 3 * 2 - crop_side_len // 2, image.shape[1] - crop_side_len)
        image = image[start_x:start_x+crop_side_len, start_y:start_y+crop_side_len]
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
    # batch['mask_path'] = mask_list[item]

    return batch

def check_polyp_size(mask, min_size=0.02, max_size=0.3):
    # check the size of the polyp in the mask, if it is too small or too large, then regenerate the mask
    # the polyp should occupy at least 1% of the mask and at most 50% of the mask
    mask = np.array(mask).astype(np.float32) / 255.0
    h, w = mask.shape
    polyp_area = np.sum(mask > 0.5)
    if polyp_area < h * w * min_size or polyp_area > h * w * max_size:
        return False
    else:
        return True



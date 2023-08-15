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
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import matplotlib.pyplot as plt

from scripts.haorui.gen_polyp_custom import reshape_mask, load_batch
image_rescaler = albumentations.SmallestMaxSize(max_size=256, interpolation=cv2.INTER_AREA)

def compute_dice(image_list, mask_list, model):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    image_list = []
    mask_list = []
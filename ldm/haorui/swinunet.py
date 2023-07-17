import torch
import torch.nn as nn
import numpy as np
from monai.networks.nets import SwinUNETR
from monai.losses import DiceLoss
from monai.transforms import Activations, AsDiscrete
from monai.inferers import sliding_window_inference
import pytorch_lightning as pl


class SwinUNETR_polyp(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.image_size = (256, 256)
        self.in_channel = 3
        self.out_channel = 1
        self.feature_size = 48
        self.drop_rate = 0.0
        self.attn_drop_rate = 0.0
        self.dropout_path_rate = 0.0
        self.use_checkpoint = True
        self.unet = SwinUNETR(
            img_size=self.image_size,
            in_channels=self.in_channel,
            out_channels=self.out_channel,
            feature_size=self.feature_size,
            drop_rate=self.drop_rate,
            attn_drop_rate=self.attn_drop_rate,
            dropout_path_rate=self.dropout_path_rate,
            use_checkpoint=self.use_checkpoint,
            spatial_dims=2,
        )
        self.dice_loss = DiceLoss(to_onehot_y=False, softmax=True)

    def training_step(self, batch, batch_idx):
        pass  # TODO: add training step

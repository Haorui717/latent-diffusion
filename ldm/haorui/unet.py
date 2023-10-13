import sys
sys.path.append('/home/yixiao/haorui/ccvl15/haorui/latent-diffusion-Local')
from typing import Any
import torch
import torch.nn as nn
import numpy as np
import monai.networks.nets as monai_nets
from monai.losses import DiceLoss
from monai.transforms import Activations, AsDiscrete
from monai.inferers import sliding_window_inference
import pytorch_lightning as pl

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import count_params
from ldm.util import instantiate_from_config, NO_OUTPUT

class SwinUNETR(pl.LightningModule):
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
        self.unet = monai_nets.SwinUNETR(
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
        self.dice_loss = DiceLoss(softmax=True)
        count_params(self)
    def training_step(self, batch, batch_idx):
        pass  # TODO: add training step


class Unet(pl.LightningModule):
    def __init__(self, 
                #  in_channel=3,
                #  out_channel=1,
                #  image_size=(256, 256),
                #  channels=(16, 32, 64, 128, 256),
                #  strides=(2, 2, 2, 2),
                 unet_config,
                 max_epochs=1000,
                 dice=True,
                 bce=True,
                 ckpt_path=None,
                 image_scale_01=False):
        super().__init__()
        # self.image_size = image_size
        # self.in_channel = in_channel
        # self.out_channel = out_channel
        # self.channels = channels
        # self.strides = strides
        # self.unet = monai_nets.UNet(
        #     spatial_dims=2,
        #     in_channels=self.in_channel,
        #     out_channels=self.out_channel,
        #     channels=self.channels,
        #     strides=self.strides,
        # )
        self.unet = instantiate_from_config(unet_config)
        self.dice = dice
        self.bce = bce
        self.image_scale_01 = image_scale_01
        self.max_epochs = max_epochs
        if self.dice:
            self.dice_loss = DiceLoss(sigmoid=True)  # sigmoid=True for binary label
        else:
            self.dice_loss = None
        if self.bce:
            self.bce_loss = torch.nn.BCEWithLogitsLoss()
        else:
            self.bce_loss = None
        count_params(self, verbose=True)
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            if "state_dict" in list(ckpt.keys()):
                ckpt = ckpt["state_dict"]
            self.load_state_dict(ckpt)
            print(f'Loaded checkpoint from {ckpt_path}')
    def configure_optimizers(self):
        lr = self.learning_rate
        optimizer = torch.optim.Adam(self.unet.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1.0 - x / self.max_epochs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        }
    def forward(self, x):
        if self.image_scale_01:
            x = (x + 1) / 2
        return self.unet(x)
    def training_step(self, batch, batch_idx):
        images = batch['image']
        masks = batch['mask']
        masks = (masks + 1) / 2  # convert to 0-1
        if self.image_scale_01:
            images = (images + 1) / 2
        outputs = self.unet(images)
        loss = 0
        if self.dice:
            dice_loss = self.dice_loss(outputs, masks)
            self.log('dice_loss', dice_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        if self.bce:
            bce_loss = self.bce_loss(outputs, masks)
            self.log('bce_loss', bce_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        loss = dice_loss + bce_loss
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        masks = batch['mask']
        masks = (masks + 1) / 2  # convert to 0-1
        if self.image_scale_01:
            images = (images + 1) / 2
        outputs = self.unet(images)
        dice_loss = self.dice_loss(outputs, masks)
        bce_loss = self.bce_loss(outputs, masks)
        loss = dice_loss + bce_loss
        self.log('dice_loss', dice_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('bce_loss', bce_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss

    @torch.no_grad()
    def log_images(self, batch, N=8, **kwargs):
        log = dict()
        N = min(N, len(batch['image']))
        if self.image_scale_01:
            batch['image'] = (batch['image'] + 1) / 2
        pred = self.unet(batch['image'])[:N]
        if self.image_scale_01:
            batch['image'] = batch['image'] * 2 - 1
        log['image'] = batch['image'][:N]
        log['mask'] = batch['mask'][:N]
        pred[pred > 0] = 1
        pred[pred <= 0] = -1
        log['pred'] = pred
        # compare pred and mask and image
        log['compare'] = torch.cat([log['image'], log['mask'].repeat(1, 3, 1, 1) , log['pred'].repeat(1, 3, 1, 1)], dim=3)

        return log
    

class Unet_ldm(pl.LightningModule):
    '''
    Unet with latent diffusion model which synthesize polyps on the fly.
    '''
    def __init__(self, unet_config,
                 ldm_config=None,
                 scheduler_config=None,
                 dice=True,
                 bce=True,
                 ckpt_path=None,
                 image_scale_01=False,
                 sampler_steps=50,
                 max_epochs=1000,
                 monitor=None,
                 ) -> None:
        super().__init__()
        if monitor is not None:
            self.monitor = monitor
        self.image_scale_01 = image_scale_01
        self.max_epochs = max_epochs
        self.unet = instantiate_from_config(unet_config)
        self.dice = dice
        self.bce = bce
        if self.dice:
            self.dice_loss = DiceLoss(sigmoid=True)  # sigmoid=True for binary label
        else:
            self.dice_loss = None
        if self.bce:
            self.bce_loss = torch.nn.BCEWithLogitsLoss()
        else:
            self.bce_loss = None
        
        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.ldm_config = ldm_config
        self.ldm = instantiate_from_config(self.ldm_config)
        for param in self.ldm.parameters():
            param.requires_grad = False
        self.sampler = DDIMSampler(self.ldm)
        self.sampler_steps = sampler_steps

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            if "state_dict" in list(ckpt.keys()):
                ckpt = ckpt["state_dict"]
            self.load_state_dict(ckpt)
            print(f'Loaded checkpoint from {ckpt_path}')

    def configure_optimizers(self):
        lr = self.learning_rate
        optimizer = torch.optim.AdamW(self.unet.parameters(), lr=lr)

        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler.schedule)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        return optimizer
    
    def forward(self, x):  # only pass the unet
        if self.image_scale_01:
            x = (x + 1) / 2
        return self.unet(x)
    
    @torch.no_grad()
    def gen_polyps(self, batch):  # generate polyps on the fly
        images = batch['image']
        masks = batch['mask']
        masked_images = batch['masked_image']
        condition = self.ldm.cond_stage_model.encode(masked_images)
        m_condition = torch.nn.functional.interpolate(masks, size=condition.shape[-2:])
        c = torch.cat([condition, m_condition], dim=1)
        shape = (c.shape[1] - 1,) + c.shape[2:]
        with NO_OUTPUT():
            samples_ddim, _ = self.sampler.sample(S=self.sampler_steps,
                                                conditioning=c,
                                                batch_size=c.shape[0],
                                                shape=shape,
                                                verbose=False)
        synthetic_images = self.ldm.decode_first_stage(samples_ddim)
        synthetic_images = torch.clamp(synthetic_images, -1, 1)
        return images, masks, masked_images, synthetic_images

    def training_step(self, batch, batch_idx):
        images, masks, masked_images, synthetic_images = self.gen_polyps(batch)
        masks = (masks + 1) / 2  # convert to 0-1
        if self.image_scale_01:
            synthetic_images = (synthetic_images + 1) / 2
        outputs = self.unet(synthetic_images)
        loss = 0
        if self.dice:
            dice_loss = self.dice_loss(outputs, masks)
            self.log('train_dice_loss', dice_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        if self.bce:
            bce_loss = self.bce_loss(outputs, masks)
            self.log('train_bce_loss', bce_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        loss = dice_loss + bce_loss
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks, masked_images, synthetic_images = self.gen_polyps(batch)
        masks = (masks + 1) / 2  # convert to 0-1
        if self.image_scale_01:
            synthetic_images = (synthetic_images + 1) / 2
        outputs = self.unet(synthetic_images)
        loss = 0
        if self.dice:
            dice_loss = self.dice_loss(outputs, masks)
            self.log('val_dice_loss', dice_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        if self.bce:
            bce_loss = self.bce_loss(outputs, masks)
            self.log('val_bce_loss', bce_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        loss = dice_loss + bce_loss
        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss

    @torch.no_grad()
    def log_images(self, batch, N=8, **kwargs):
        images, masks, masked_images, synthetic_images = self.gen_polyps(batch)
        if self.image_scale_01:
            synthetic_images = (synthetic_images + 1) / 2
        pred = self.unet(synthetic_images)
        if self.image_scale_01:
            synthetic_images = synthetic_images * 2 - 1
        pred[pred > 0] = 1
        pred[pred <= 0] = -1
        log = dict()
        N = min(N, len(images))
        log['image'] = images[:N]
        log['mask'] = masks[:N]
        log['masked_image'] = masked_images[:N]
        log['synthetic_image'] = synthetic_images[:N]
        log['pred'] = pred[:N]
        # compare pred and mask and image
        log['compare'] = torch.cat([log['image'], log['synthetic_image'], log['mask'].repeat(1, 3, 1, 1) , log['pred'].repeat(1, 3, 1, 1)], dim=3)

        return log

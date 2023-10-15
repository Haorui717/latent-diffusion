from typing import Any

import torch
from ldm.models.autoencoder import VQModelInterface
from ldm.util import instantiate_from_config
from torch import nn

class MaskedImageCondStage(nn.Module):
    def __init__(self, ae_config:dict, ) -> None:
        super().__init__()
        self.model: VQModelInterface = instantiate_from_config(ae_config)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
    @torch.no_grad()
    def __call__(self, condition: dict) -> Any:
        mask = condition['mask']
        masked_image = condition['masked_image']
        c = self.model.encode(masked_image)
        mask = torch.functional.F.interpolate(mask, size=c.shape[2:])
        c = torch.cat([c, mask], dim=1)
        return c

class OnlyMaskCondStage(nn.Module):
    def __init__(self, image_size) -> None:
        super().__init__()
        self.image_size = image_size
        
    @torch.no_grad()
    def __call__(self, condition: dict) -> Any:
        mask = condition['mask']
        c = torch.functional.F.interpolate(mask, size=self.image_size)
        return c
    
    def decode(self, c: torch.Tensor) -> torch.Tensor:
        return torch.functional.F.interpolate(c, size=256)
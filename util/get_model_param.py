import argparse

from main import instantiate_from_config
from omegaconf import OmegaConf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, default='configs/polyp.yaml')

    opt = parser.parse_args()

    config = OmegaConf.load(opt.base)
    model = instantiate_from_config(config.model)

    # print number of parameters of the model
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {num_params}')
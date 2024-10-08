import torch
from autoencoder import Encoder
from utils import instantiate_from_config
from omegaconf import OmegaConf

images = torch.randn(2, 3, 256, 256)


conf = OmegaConf.load("./config.yaml")

model = instantiate_from_config(conf.model)

output = model(images)

print("Done")

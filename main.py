from ldm.models.first_stage.autoencoder import AutoencoderKL
import torch


images = torch.randn(2, 3, 256, 256)


autoencoder = AutoencoderKL(config_path="ldm/config/kl-f4/config.yaml")

output = autoencoder.encoder(images)

print("Success")

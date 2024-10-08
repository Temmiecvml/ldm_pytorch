import torch
import pytorch_lightning as pl
from .unet import Encoder
from ldm.utils import load_config


class AutoencoderKL(pl.LightningModule):
    def __init__(
        self,
        #  ddconfig,
        #  lossconfig,
        #  embed_dim,
        #  ckpt_path=None,
        #  ignore_keys=[],
        #  image_key="image",
        #  colorize_nlabels=None,
        #  monitor=None,
        config_path: str = "",
    ):
        super().__init__()
        # self.image_key = image_key
        self.encoder = Encoder(**load_config(config_path, "model.encoder"))
        # self.decoder = Decoder(**ddconfig)
        # self.loss = instantiate_from_config(lossconfig)
        # assert ddconfig["double_z"]
        # self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        # self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        # self.embed_dim = embed_dim
        # if colorize_nlabels is not None:
        #     assert type(colorize_nlabels)==int
        #     self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        # if monitor is not None:
        #     self.monitor = monitor
        # if ckpt_path is not None:
        #     self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        pass

    def encode(self, x):
        pass

    def decode(self, x):
        pass

    def forward(self, input, sample_posterior=True):
        pass

    def get_input(self, batch, k):
        pass

    def training_step(self, batch, batch_idx, optimizer_idx):
        pass

    def validation_step(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        pass

    def get_last_layer(self):
        pass

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        pass

    def to_rgb(self, x):
        pass

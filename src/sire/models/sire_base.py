import pytorch_lightning as pl
import torch

from monai.utils import set_determinism

from src.sire.models.networks.gem_gcn import GEMGCN


class SIREBase(pl.LightningModule):
    def __init__(self, lr: float = 0.001, seed: int = 0):
        super().__init__()

        set_determinism(seed)
        self.save_hyperparameters()

        self.lr = lr
        self.model = GEMGCN()

    def backbone(self, data):
        bs = len(data["sample"]["index"])
        nverts, num_scales = data["global"]["nverts"][0], len(data["global"]["scales"][0])

        scales_ft = self.model(data["sample"]["spheres"], nverts, num_scales).view(
            bs, -1, num_scales, self.model.out_channels
        )

        return scales_ft

    def shared_step(self, data, batch_idx, stage):
        raise NotImplementedError()

    def training_step(self, data, batch_idx):
        return self.shared_step(data, batch_idx, stage="train")

    def validation_step(self, data, batch_idx):
        return self.shared_step(data, batch_idx, stage="val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

import torch

from torch.nn import CosineSimilarity, MSELoss

from src.sire.models.sire_base import SIREBase
from src.sire.utils.tracker import TrackerSphere


class SIRETracker(SIREBase):
    def __init__(
        self,
        subdivisions: int = 3,
        lr: float = 0.001,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(lr, seed)
        self.sphere = TrackerSphere(subdivisions=subdivisions)

        self.mse_loss = MSELoss()
        self.cosine_similarity = CosineSimilarity()

    def get_direction(self, direction_heatmap: torch.tensor):
        ind = torch.argmax(direction_heatmap, dim=1).detach().cpu().numpy()
        direction = torch.tensor(self.sphere.cartverts[ind, :]).reshape(-1, 3)

        return direction

    def forward(self, data):
        scales_ft = self.backbone(data)
        scales_vertex_max, _ = torch.max(scales_ft, dim=2)

        return scales_vertex_max

    def shared_step(self, data, batch_idx, stage):
        direction_pred = self(data)
        direction_true = data["sample"]["direction"]

        # Losses
        mse_loss = self.mse_loss(direction_pred, direction_true)
        cosine_similarity = torch.abs(
            self.cosine_similarity(
                self.get_direction(direction_pred).view(-1, 3), self.get_direction(direction_true).view(-1, 3)
            )
        ).mean()

        # Total loss
        loss = mse_loss

        # Logger
        self.log(f"{stage}/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{stage}/mse_loss", mse_loss, on_epoch=True, logger=True)
        self.log(f"{stage}/cosine_similarity", cosine_similarity, on_epoch=True, logger=True)

        return loss
